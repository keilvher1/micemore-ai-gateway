"""모델 모니터링 — drift 감지.

체크 항목:
  - 학습 분포 vs 운영 분포 (PSI · Population Stability Index)
  - 라벨 분포 균형 (positives ratio)
  - 점수 분포 변화 (히스토그램 KL divergence)

실 운영: weekly cron (EventBridge) 으로 호출 → CloudWatch metric +
임계 초과 시 SNS 알림. 본 모듈은 결정론적 metric 계산만, 외부 호출 없음.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    score_psi: float
    positives_ratio_baseline: float
    positives_ratio_current: float
    alert: bool
    reasons: list[str]


def population_stability_index(
    expected: list[float], actual: list[float], buckets: int = 10
) -> float:
    """PSI — 0~0.1 안정 / 0.1~0.25 주의 / 0.25+ 위험."""
    if not expected or not actual:
        return 0.0
    lo = min(min(expected), min(actual))
    hi = max(max(expected), max(actual))
    if hi == lo:
        return 0.0
    width = (hi - lo) / buckets
    edges = [lo + i * width for i in range(buckets + 1)]
    edges[-1] = hi + 1e-9  # right inclusive

    def _hist(xs: list[float]) -> list[float]:
        h = [0] * buckets
        for x in xs:
            for i in range(buckets):
                if edges[i] <= x < edges[i + 1]:
                    h[i] += 1
                    break
        n = len(xs)
        return [(c / n) if n else 0 for c in h]

    e = _hist(expected)
    a = _hist(actual)
    psi = 0.0
    for ei, ai in zip(e, a):
        ei_ = max(ei, 1e-6)
        ai_ = max(ai, 1e-6)
        psi += (ai_ - ei_) * math.log(ai_ / ei_)
    return round(psi, 4)


def evaluate(
    *,
    baseline_features: dict[str, list[float]],
    current_features: dict[str, list[float]],
    baseline_scores: list[float],
    current_scores: list[float],
    baseline_positives: float,
    current_positives: float,
    psi_alert_threshold: float = 0.25,
    positives_ratio_max_drift: float = 0.15,
) -> DriftReport:
    """학습 시점 대비 현재 운영 데이터의 drift 보고."""
    feat_psi = {
        name: population_stability_index(
            baseline_features[name], current_features.get(name, [])
        )
        for name in baseline_features
    }
    score_psi = population_stability_index(baseline_scores, current_scores)

    reasons: list[str] = []
    drifted = [n for n, v in feat_psi.items() if v >= psi_alert_threshold]
    if drifted:
        reasons.append(f"feature drift: {', '.join(drifted)}")
    if score_psi >= psi_alert_threshold:
        reasons.append(f"score distribution drift PSI={score_psi}")
    if abs(current_positives - baseline_positives) >= positives_ratio_max_drift:
        reasons.append(
            f"positives ratio drift {baseline_positives:.2f} → {current_positives:.2f}"
        )

    return DriftReport(
        feature_psi=feat_psi,
        score_psi=score_psi,
        positives_ratio_baseline=round(baseline_positives, 3),
        positives_ratio_current=round(current_positives, 3),
        alert=bool(reasons),
        reasons=reasons,
    )
