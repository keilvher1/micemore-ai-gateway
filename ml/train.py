"""XGBoost 학습 파이프라인 — Phase 3 Sprint 1~2.

흐름:
  Firestore → BigQuery export (별도 스케줄)
  BigQuery → DataFrame
  features.py 로 변환
  XGBoost 학습 + cross-validation
  S3 로 model.json + meta.json 저장
  CloudWatch 로 metrics 발행

CLI:
  python -m ml.train --event ev_2026_06 --bucket micemore-models --version v1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Iterable

from ml.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    TrainingRecord,
    behaviors_to_records,
)

log = logging.getLogger("ml.train")

MIN_SAMPLES_FOR_ML = 200
MAX_TRAIN_SECONDS = 1800  # 30 min hard cap


# ---------------------------------------------------------------------------
# Loader (BigQuery 또는 로컬 JSONL — mock 환경에서는 후자)
# ---------------------------------------------------------------------------
def load_training_rows(source: str) -> list[dict]:
    """source 가 'bq://...' 면 BigQuery, 'file://...' 면 로컬 JSONL."""
    if source.startswith("file://"):
        path = source[len("file://"):]
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if source.startswith("bq://"):
        # bq://<project>.<dataset>.<table>
        from google.cloud import bigquery  # type: ignore

        ref = source[len("bq://"):]
        client = bigquery.Client()
        return [dict(r) for r in client.query(f"SELECT * FROM `{ref}`").result()]
    raise ValueError(f"unsupported source: {source}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(
    rows: list[dict],
    *,
    event_id: str,
    salt: str = "micemore",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """학습 + (model, meta) 반환. xgboost 미설치 시 ImportError."""
    import numpy as np  # type: ignore
    import xgboost as xgb  # type: ignore
    from sklearn.metrics import roc_auc_score, precision_recall_curve  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore

    records: list[TrainingRecord] = behaviors_to_records(rows, event_id, salt)
    if len(records) < MIN_SAMPLES_FOR_ML:
        raise RuntimeError(
            f"insufficient samples: {len(records)} < {MIN_SAMPLES_FOR_ML}. "
            "룰베이스 v0 유지."
        )
    pos = sum(1 for r in records if r.label == 1)
    if pos < 20 or pos > len(records) - 20:
        raise RuntimeError(
            f"라벨 균형 부족: positives={pos}/{len(records)}. "
            "라벨 정의 점검 또는 데이터 추가 수집."
        )

    X = np.array([r.features for r in records], dtype=float)
    y = np.array([r.label for r in records], dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y,
    )
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=list(FEATURE_NAMES))
    dte = xgb.DMatrix(Xte, label=yte, feature_names=list(FEATURE_NAMES))

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "max_depth": 4,
        "eta": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 4,
        "seed": seed,
    }
    started = time.time()
    booster = xgb.train(
        params,
        dtr,
        num_boost_round=400,
        evals=[(dtr, "train"), (dte, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    elapsed = time.time() - started
    if elapsed > MAX_TRAIN_SECONDS:
        log.warning("학습 시간 cap 초과: %.1fs", elapsed)

    val_proba = booster.predict(dte)
    auc = roc_auc_score(yte, val_proba)
    p, r, _ = precision_recall_curve(yte, val_proba)
    pr_auc = float(np.trapz(p[::-1], r[::-1]))

    meta = {
        "training_samples": len(records),
        "positives": int(pos),
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "auc": float(auc),
        "pr_auc": pr_auc,
        "trained_at": int(time.time()),
        "elapsed_sec": round(elapsed, 2),
        "best_iteration": booster.best_iteration if hasattr(booster, "best_iteration") else None,
        "event_id": event_id,
    }
    log.info("train done | n=%d auc=%.3f pr_auc=%.3f elapsed=%.1fs",
             len(records), auc, pr_auc, elapsed)
    return booster, meta


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
def save_to_s3(booster: Any, meta: dict[str, Any], *,
               bucket: str, version: str) -> None:
    import boto3  # type: ignore

    s3 = boto3.client("s3")
    booster.save_model(f"/tmp/model_{version}.json")
    with open(f"/tmp/model_{version}.json", "rb") as fh:
        s3.put_object(
            Bucket=bucket,
            Key=f"models/{version}/model.json",
            Body=fh.read(),
        )
    s3.put_object(
        Bucket=bucket,
        Key=f"models/{version}/meta.json",
        Body=json.dumps({**meta, "model_version": version}, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    log.info("saved s3://%s/models/%s/", bucket, version)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=os.getenv("TRAIN_SOURCE", "file:///tmp/training.jsonl"))
    p.add_argument("--event", required=True, help="event_id (meta 보존용)")
    p.add_argument("--bucket", default=os.getenv("MODEL_S3_BUCKET", ""))
    p.add_argument("--version", required=True, help="ex) v1, v1-rc2 …")
    p.add_argument("--salt", default=os.getenv("FEATURE_SALT", "micemore"))
    return p.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse(argv)
    rows = load_training_rows(args.source)
    booster, meta = train(rows, event_id=args.event, salt=args.salt)
    if args.bucket:
        save_to_s3(booster, meta, bucket=args.bucket, version=args.version)
    print(json.dumps({"ok": True, **meta}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
