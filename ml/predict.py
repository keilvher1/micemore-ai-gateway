"""Lambda inference + A/B 라우팅 + 룰베이스 폴백.

라우팅 정책:
  1. 200 건 이상 학습 데이터 + 모델 파일 존재 시 → ML
  2. 그 외 → P2 룰베이스 (leads/scorer.py)
  3. A/B 모드: visitor_id 해시 기반 50/50 splitting (안정적)

USE_MOCK=true 일 때:
  - xgboost · S3 import 안 함
  - 결정론적 더미 점수 반환 (테스트 + dev)
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any

from leads.scorer import VisitorBehavior, score as rule_score
from ml.features import FEATURE_NAMES, FEATURE_VERSION, behavior_to_features

log = logging.getLogger("ml.predict")


# ---------------------------------------------------------------------------
# 환경
# ---------------------------------------------------------------------------
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
USE_ML_SCORING = os.getenv("USE_ML_SCORING", "false").lower() == "true"
MIN_TRAINING_SAMPLES = 200
MODEL_S3_BUCKET = os.getenv("MODEL_S3_BUCKET", "")
MODEL_VERSION = os.getenv("MODEL_VERSION", "")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    score: int                 # 0~100
    level: str                 # hot/warm/cold
    source: str                # "ml" | "rule" | "mock"
    confidence: float          # 0.0~1.0
    top_factors: list[dict[str, Any]]
    feature_version: str
    model_version: str | None


# ---------------------------------------------------------------------------
# 모델 로더 — 싱글톤. lazy import.
# ---------------------------------------------------------------------------
_model = None
_model_meta: dict[str, Any] | None = None


def _load_model() -> tuple[Any, dict[str, Any]] | tuple[None, None]:
    """S3 의 model.json 을 lazy 로드. 실패 시 None 반환 → 룰베이스 폴백."""
    global _model, _model_meta
    if _model is not None:
        return _model, _model_meta or {}
    if USE_MOCK or not USE_ML_SCORING:
        return None, None
    if not MODEL_S3_BUCKET or not MODEL_VERSION:
        log.warning("MODEL_S3_BUCKET/VERSION 미설정 → 룰베이스 폴백")
        return None, None
    try:
        import json
        import boto3  # type: ignore
        import xgboost as xgb  # type: ignore

        s3 = boto3.client("s3")
        body = s3.get_object(
            Bucket=MODEL_S3_BUCKET,
            Key=f"models/{MODEL_VERSION}/model.json",
        )["Body"].read()
        booster = xgb.Booster()
        booster.load_model(bytearray(body))
        meta_body = s3.get_object(
            Bucket=MODEL_S3_BUCKET,
            Key=f"models/{MODEL_VERSION}/meta.json",
        )["Body"].read()
        meta = json.loads(meta_body)
        _model, _model_meta = booster, meta
        log.info("loaded model %s, samples=%d", MODEL_VERSION,
                 meta.get("training_samples", 0))
        return _model, _model_meta
    except Exception as exc:  # noqa: BLE001
        log.exception("model load failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# A/B 라우팅
# ---------------------------------------------------------------------------
def _ab_arm(visitor_id: str, ratio: float = 0.5) -> str:
    """visitor_id 의 hash 첫 바이트로 결정론적 50/50."""
    h = hashlib.sha1(visitor_id.encode("utf-8")).digest()[0]
    return "ml" if (h / 255.0) < ratio else "rule"


def _level_from_score(s: int) -> str:
    return "hot" if s >= 70 else "warm" if s >= 40 else "cold"


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------
def predict(behavior: dict, *, force_ml: bool = False) -> Prediction:
    """단일 방문자 점수 추론. 룰베이스 폴백 안전망 내장."""
    visitor_id = str(behavior.get("visitor_id", ""))

    # mock 모드 — 결정론적 더미 (룰베이스 활용)
    if USE_MOCK:
        return _rule_based(behavior, source="mock")

    # 1) ML 사용 가능 여부
    model, meta = _load_model()
    if model is None:
        return _rule_based(behavior, source="rule")
    if (meta or {}).get("training_samples", 0) < MIN_TRAINING_SAMPLES:
        log.info("training samples < %d → 룰베이스", MIN_TRAINING_SAMPLES)
        return _rule_based(behavior, source="rule")

    # 2) A/B 분기 (force_ml 이면 항상 ML)
    arm = "ml" if force_ml else _ab_arm(visitor_id)
    if arm == "rule":
        return _rule_based(behavior, source="rule")

    # 3) ML 추론
    return _ml_predict(model, meta, behavior)


def _rule_based(behavior: dict, *, source: str) -> Prediction:
    b = VisitorBehavior(
        visitor_id=str(behavior.get("visitor_id", "")),
        booth_id=str(behavior.get("booth_id", "")),
        booth_dwell_time_sec=int(behavior.get("booth_dwell_time_sec") or 0),
        copilot_questions_count=int(behavior.get("copilot_questions_count") or 0),
        copilot_question_topics=list(behavior.get("copilot_question_topics") or []),
        pamphlet_downloaded=bool(behavior.get("pamphlet_downloaded")),
        business_card_saved=bool(behavior.get("business_card_saved")),
        translation_session_minutes=int(behavior.get("translation_session_minutes") or 0),
        other_booths_visited=int(behavior.get("other_booths_visited") or 0),
        competitor_booths_visited=int(behavior.get("competitor_booths_visited") or 0),
        revisit_count=int(behavior.get("revisit_count") or 0),
    )
    bd, lv = rule_score(b)
    return Prediction(
        score=bd.total,
        level=lv,
        source=source,
        confidence=1.0,  # 룰베이스는 결정론적
        top_factors=[{"name": k, "contribution": v} for k, v in bd.top_factors(4)],
        feature_version=FEATURE_VERSION,
        model_version=None,
    )


def _ml_predict(model: Any, meta: dict[str, Any], behavior: dict) -> Prediction:
    """xgboost + SHAP. 실 환경 only."""
    import numpy as np  # type: ignore
    import xgboost as xgb  # type: ignore

    feat = behavior_to_features(behavior)
    dmat = xgb.DMatrix(np.array([feat], dtype=float),
                       feature_names=list(FEATURE_NAMES))
    proba = float(model.predict(dmat)[0])
    score = max(0, min(100, int(round(proba * 100))))

    # SHAP top-N (lazy 계산)
    try:
        contribs = model.predict(dmat, pred_contribs=True)[0]
        idx_score = sorted(
            range(len(FEATURE_NAMES)),
            key=lambda i: abs(contribs[i]),
            reverse=True,
        )[:4]
        top = [
            {"name": FEATURE_NAMES[i], "contribution": round(float(contribs[i]), 3)}
            for i in idx_score
        ]
    except Exception:  # noqa: BLE001
        top = []

    return Prediction(
        score=score,
        level=_level_from_score(score),
        source="ml",
        confidence=round(proba if proba >= 0.5 else 1 - proba, 3),
        top_factors=top,
        feature_version=meta.get("feature_version", FEATURE_VERSION),
        model_version=meta.get("model_version", MODEL_VERSION) or None,
    )
