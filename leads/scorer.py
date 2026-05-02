"""룰베이스 리드 스코어링 v0.

설계 원칙:
  - 결정론적: 같은 입력 → 같은 점수. 디버깅·테스트 쉬움.
  - 분리 가능: 가중치는 모두 모듈 상수 → A/B 비교 시 fork 만 떠 비교.
  - 설명 가능: ScoreBreakdown 으로 항목별 기여도 노출 → narrator 로 한 문장.

가중치 합계 (이론 max):
  체류 25 + 질문 20 + 가격 5 + 팜플릿 8 + 명함 7 + 통역 15 + 재방문 15 + 경쟁 5 = 100
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# 가중치 (튜닝 시 이 블록만 수정)
# ---------------------------------------------------------------------------
DWELL_FULL_SECONDS = 300        # 5분에 25점 만점
DWELL_MAX = 25
QUESTIONS_PER_POINT = 4         # 1 질문 = 4점
QUESTIONS_MAX = 20              # 5 질문 만점
PRICING_TOPICS = {"pricing", "price", "cost", "가격", "비용", "할인"}
PRICING_BONUS = 5
PAMPHLET_POINTS = 8
BUSINESS_CARD_POINTS = 7
TRANSLATION_PER_MINUTE = 3      # 1 분 = 3점
TRANSLATION_MAX = 15            # 5분 만점
REVISIT_PER_VISIT = 7           # 1 회 = 7점
REVISIT_MAX = 15                # 2회 이상이면 만점
COMPETITOR_POINTS = 5           # 1곳 이상 방문 시
SCORE_HOT_CUTOFF = 70
SCORE_WARM_CUTOFF = 40


LeadLevel = Literal["hot", "warm", "cold"]


# ---------------------------------------------------------------------------
# 입력 / 출력 모델
# ---------------------------------------------------------------------------
@dataclass
class VisitorBehavior:
    """행사 종료 시점에 Firestore 에서 집계된 한 방문자×부스 행동."""

    visitor_id: str
    booth_id: str
    booth_dwell_time_sec: int = 0
    copilot_questions_count: int = 0
    copilot_question_topics: list[str] = field(default_factory=list)
    pamphlet_downloaded: bool = False
    business_card_saved: bool = False
    translation_session_minutes: int = 0
    other_booths_visited: int = 0
    competitor_booths_visited: int = 0
    revisit_count: int = 0


@dataclass
class ScoreBreakdown:
    """항목별 기여 — narrator 가 자연어로 풀어줌."""

    dwell: int = 0
    questions: int = 0
    pricing_bonus: int = 0
    pamphlet: int = 0
    business_card: int = 0
    translation: int = 0
    revisit: int = 0
    competitor: int = 0

    @property
    def total(self) -> int:
        raw = sum(asdict(self).values())
        return min(100, max(0, raw))

    def top_factors(self, n: int = 3) -> list[tuple[str, int]]:
        """기여도 상위 n 개 (label, score)."""
        items = [(k, v) for k, v in asdict(self).items() if v > 0]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]


# ---------------------------------------------------------------------------
# 스코어 함수
# ---------------------------------------------------------------------------
def _topic_match(topics: list[str], target: set[str]) -> bool:
    return any(t.lower() in target for t in topics)


def score(b: VisitorBehavior) -> tuple[ScoreBreakdown, LeadLevel]:
    """입력 행동 → 점수 + 레벨.

    Returns:
        (breakdown, level) — breakdown.total 로 합계 접근.
    """
    bd = ScoreBreakdown()

    # 1. 체류 시간 — 선형
    bd.dwell = min(
        DWELL_MAX,
        round(b.booth_dwell_time_sec / DWELL_FULL_SECONDS * DWELL_MAX),
    )

    # 2. AI 질문 개수
    bd.questions = min(QUESTIONS_MAX, b.copilot_questions_count * QUESTIONS_PER_POINT)
    if _topic_match(b.copilot_question_topics, PRICING_TOPICS):
        bd.pricing_bonus = PRICING_BONUS

    # 3. 자료 저장
    if b.pamphlet_downloaded:
        bd.pamphlet = PAMPHLET_POINTS
    if b.business_card_saved:
        bd.business_card = BUSINESS_CARD_POINTS

    # 4. 통역 사용
    bd.translation = min(
        TRANSLATION_MAX,
        b.translation_session_minutes * TRANSLATION_PER_MINUTE,
    )

    # 5. 재방문 — 충성도 신호
    bd.revisit = min(REVISIT_MAX, b.revisit_count * REVISIT_PER_VISIT)

    # 6. 경쟁사 비교 — binary
    if b.competitor_booths_visited >= 1:
        bd.competitor = COMPETITOR_POINTS

    total = bd.total
    if total >= SCORE_HOT_CUTOFF:
        level: LeadLevel = "hot"
    elif total >= SCORE_WARM_CUTOFF:
        level = "warm"
    else:
        level = "cold"
    return bd, level
