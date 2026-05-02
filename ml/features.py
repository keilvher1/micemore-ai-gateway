"""피처 엔지니어링 — VisitorBehavior → numeric feature vector.

원칙:
  - **순수 함수**: 외부 호출/DB 의존 없음. 단위 테스트 결정론적.
  - **익명화**: visitor_id 는 hashed_id 로만 보존, 메일·이름은 ML 입력에 절대 안 들어감.
  - **결측 안전**: 모든 필드 default 0/False. 학습 시점과 추론 시점 동일 함수 사용.
  - **버전 고정**: FEATURE_VERSION 변경 시 모델 재학습 필수.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

# 학습/추론에서 동일한 순서·이름. 변경 시 v2 로 올리고 모델 재학습.
FEATURE_VERSION = "v1"

FEATURE_NAMES: tuple[str, ...] = (
    "dwell_min",
    "questions_count",
    "pricing_topic_ratio",
    "tech_topic_ratio",
    "pamphlet_downloaded",
    "business_card_saved",
    "translation_min",
    "other_booths_visited",
    "competitor_booths_visited",
    "revisit_count",
    "first_visit_hour",         # 0~23
    "booth_visit_order",        # 1=처음 / N번째
    "avg_question_length",      # tokens
    "question_complexity",      # 0~1, LLM 분류 결과 (없으면 0.5)
)


@dataclass
class TrainingRecord:
    """학습 시점의 (피처, 라벨, 메타) 단일 행."""

    visitor_hash: str
    booth_id: str
    event_id: str
    features: list[float]
    label: int  # converted: 0 / 1 (follow-up 응답 또는 미팅 성사)
    feature_version: str = FEATURE_VERSION


# ---------------------------------------------------------------------------
# 익명화
# ---------------------------------------------------------------------------
def hash_visitor_id(visitor_id: str, salt: str = "micemore") -> str:
    """SHA256(salt|visitor_id)[:16] — 학습/추론 양쪽 동일 함수.

    salt 는 운영 시 SecretsManager 에서 주입. 같은 salt 면 join 가능,
    salt 변경 시 과거 데이터와 분리 (의도적).
    """
    h = hashlib.sha256(f"{salt}|{visitor_id}".encode("utf-8")).hexdigest()
    return h[:16]


# ---------------------------------------------------------------------------
# 단일 행 피처 추출
# ---------------------------------------------------------------------------
def _topic_ratio(topics: list[str], target: set[str]) -> float:
    if not topics:
        return 0.0
    hits = sum(1 for t in topics if t.lower() in target)
    return round(hits / len(topics), 3)


_PRICING = {"pricing", "price", "cost", "가격", "비용", "할인"}
_TECH = {"tech", "technology", "integration", "api", "spec", "기술", "스펙"}


def behavior_to_features(b: dict) -> list[float]:
    """VisitorBehavior dict → FEATURE_NAMES 순서의 float 리스트.

    Args:
      b: VisitorBehavior 가 dict 로 변환된 것 (Firestore raw + 추가 메타).
         예상 키: booth_dwell_time_sec, copilot_questions_count,
                  copilot_question_topics(list[str]), pamphlet_downloaded(bool),
                  business_card_saved(bool), translation_session_minutes,
                  other_booths_visited, competitor_booths_visited, revisit_count,
                  first_visit_hour, booth_visit_order,
                  avg_question_length, question_complexity.
    """
    topics = list(b.get("copilot_question_topics") or [])
    dwell_min = round((b.get("booth_dwell_time_sec") or 0) / 60.0, 2)
    return [
        dwell_min,
        float(b.get("copilot_questions_count") or 0),
        _topic_ratio(topics, _PRICING),
        _topic_ratio(topics, _TECH),
        1.0 if b.get("pamphlet_downloaded") else 0.0,
        1.0 if b.get("business_card_saved") else 0.0,
        float(b.get("translation_session_minutes") or 0),
        float(b.get("other_booths_visited") or 0),
        float(b.get("competitor_booths_visited") or 0),
        float(b.get("revisit_count") or 0),
        float(b.get("first_visit_hour") if b.get("first_visit_hour") is not None else 12),
        float(b.get("booth_visit_order") or 1),
        float(b.get("avg_question_length") or 0),
        float(b.get("question_complexity") if b.get("question_complexity") is not None else 0.5),
    ]


def behaviors_to_records(
    rows: Iterable[dict], event_id: str, salt: str = "micemore"
) -> list[TrainingRecord]:
    """배치 변환 — `train.py` 가 호출."""
    out: list[TrainingRecord] = []
    for r in rows:
        out.append(
            TrainingRecord(
                visitor_hash=hash_visitor_id(str(r.get("visitor_id", "")), salt),
                booth_id=str(r.get("booth_id", "")),
                event_id=event_id,
                features=behavior_to_features(r),
                label=int(r.get("converted") or 0),
            )
        )
    return out
