"""리드 스코어링 라우터.

Phase 2 v0:
  - GET /leads/booth/{booth_id} : Top N hot leads (mock 데이터로 시연)
  - GET /leads/visitor/{visitor_id} : 단일 방문자 점수
  - POST /leads/score : 행동 페이로드를 받아 즉석 채점 (테스트/개발용)

Phase 3:
  - Firestore 직접 조회 → 실데이터로 교체
  - POST /leads/recompute/{event_id} 추가 (이벤트 종료 후 일괄)
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from leads.scorer import VisitorBehavior, score
from leads.narrator import narrate

router = APIRouter(prefix="/leads", tags=["leads"])
log = logging.getLogger("leads")

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Mock 시드 데이터 — 5 페르소나
# ---------------------------------------------------------------------------
_MOCK_BEHAVIORS: list[VisitorBehavior] = [
    VisitorBehavior(
        visitor_id="v_sarah", booth_id="lumen",
        booth_dwell_time_sec=420, copilot_questions_count=5,
        copilot_question_topics=["pricing", "pricing", "timeline", "tech", "pricing"],
        pamphlet_downloaded=False, business_card_saved=True,
        translation_session_minutes=4, other_booths_visited=8,
        competitor_booths_visited=2, revisit_count=2,
    ),
    VisitorBehavior(
        visitor_id="v_park", booth_id="lumen",
        booth_dwell_time_sec=360, copilot_questions_count=4,
        copilot_question_topics=["tech", "tech", "timeline", "integration"],
        pamphlet_downloaded=True, business_card_saved=True,
        translation_session_minutes=3, other_booths_visited=6,
        competitor_booths_visited=1, revisit_count=1,
    ),
    VisitorBehavior(
        visitor_id="v_mike", booth_id="lumen",
        booth_dwell_time_sec=240, copilot_questions_count=2,
        copilot_question_topics=["overview", "timeline"],
        pamphlet_downloaded=True, business_card_saved=False,
        translation_session_minutes=2, other_booths_visited=4,
        competitor_booths_visited=0, revisit_count=1,
    ),
    VisitorBehavior(
        visitor_id="v_yuki", booth_id="lumen",
        booth_dwell_time_sec=180, copilot_questions_count=1,
        copilot_question_topics=["overview"],
        pamphlet_downloaded=True, business_card_saved=False,
        translation_session_minutes=0, other_booths_visited=10,
        competitor_booths_visited=1, revisit_count=0,
    ),
    VisitorBehavior(
        visitor_id="v_drift", booth_id="lumen",
        booth_dwell_time_sec=60, copilot_questions_count=0,
        copilot_question_topics=[],
        pamphlet_downloaded=False, business_card_saved=False,
        translation_session_minutes=0, other_booths_visited=20,
        competitor_booths_visited=0, revisit_count=0,
    ),
]


# ---------------------------------------------------------------------------
# 응답 모델
# ---------------------------------------------------------------------------
class LeadCard(BaseModel):
    visitor_id: str
    booth_id: str
    score: int = Field(ge=0, le=100)
    level: str
    narrative: str
    factors: list[dict[str, int]]


class ScoreRequest(BaseModel):
    behavior: dict[str, Any]


def _to_card(b: VisitorBehavior) -> LeadCard:
    bd, lv = score(b)
    return LeadCard(
        visitor_id=b.visitor_id,
        booth_id=b.booth_id,
        score=bd.total,
        level=lv,
        narrative=narrate(b, bd, lv, mock=True),
        factors=[{k: v} for k, v in bd.top_factors(n=4)],
    )


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------
@router.get("/booth/{booth_id}", response_model=list[LeadCard])
async def top_leads(
    booth_id: str,
    event_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
):
    """부스별 Top N 리드. v0 은 mock 시드로 응답."""
    if not USE_MOCK:
        # Phase 3: Firestore aggregation pipeline → VisitorBehavior 리스트 빌드
        raise HTTPException(503, "live data integration in Phase 3")

    cards = [_to_card(b) for b in _MOCK_BEHAVIORS if b.booth_id == booth_id]
    cards.sort(key=lambda c: c.score, reverse=True)
    return cards[:limit]


@router.get("/visitor/{visitor_id}", response_model=LeadCard)
async def visitor_lead(visitor_id: str, booth_id: str = Query(...)):
    if not USE_MOCK:
        raise HTTPException(503, "live data integration in Phase 3")
    for b in _MOCK_BEHAVIORS:
        if b.visitor_id == visitor_id and b.booth_id == booth_id:
            return _to_card(b)
    raise HTTPException(404, "visitor not found")


@router.post("/score", response_model=LeadCard)
async def score_adhoc(req: ScoreRequest):
    """테스트/개발용 즉석 채점. 실 데이터 미저장."""
    try:
        b = VisitorBehavior(**req.behavior)
    except TypeError as exc:
        raise HTTPException(422, f"invalid behavior: {exc}") from exc
    return _to_card(b)


@router.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "module": "leads", "mock": USE_MOCK}
