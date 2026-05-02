"""동선 라우터 — Phase 4-C.

POST /routing/crowd-event   : 부스 입장/퇴장 이벤트 → CrowdTracker 갱신
POST /routing/recommend     : 다음 부스 추천 (MyPass 통합)
GET  /routing/state         : 현재 모든 부스 혼잡도 (운영자용)
GET  /routing/healthz
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mypass.challenge import MyPassChallenge
from mypass.progress import MyPassProgress
from routing.crowd_tracker import CrowdTracker
from routing.fastpass import boost_with_mypass
from routing.graph_builder import BoothNode, build_graph
from routing.recommender import Recommendation, recommend_next_booth, to_dict

router = APIRouter(prefix="/routing", tags=["routing"])

# 베타 단계: 단일 인스턴스. 정상 운영에선 Redis 백엔드.
tracker = CrowdTracker()


# ---------------------------------------------------------------------------
# Crowd events
# ---------------------------------------------------------------------------
class CrowdEventIn(BaseModel):
    booth_id: str
    kind: Literal["enter", "exit"]
    dwell_sec: int = 0


@router.post("/crowd-event", response_model=dict)
async def crowd_event_route(req: CrowdEventIn):
    if req.kind == "enter":
        tracker.on_enter(req.booth_id)
    else:
        tracker.on_exit(req.booth_id, dwell_sec=req.dwell_sec)
    return asdict(tracker.get(req.booth_id))


@router.get("/state", response_model=list[dict])
async def state_route():
    return [asdict(s) for s in tracker.all()]


# ---------------------------------------------------------------------------
# Recommend
# ---------------------------------------------------------------------------
class BoothNodeIn(BaseModel):
    booth_id: str
    name: str
    lat: float
    lon: float
    category: str = ""
    keywords: list[str] = Field(default_factory=list)


class ChallengeIn(BaseModel):
    challenge_id: str
    event_id: str
    target_booth: str
    partner_booths: list[str]
    required_visits: int = 3
    reward_type: str = "fast_track"
    valid_until: int = 0


class ProgressIn(BaseModel):
    visitor_id: str
    challenge_id: str
    visited_partners: list[str] = Field(default_factory=list)
    last_tag_at: int = 0
    completed_at: int = 0
    redeemed_at: int = 0
    flagged_review: bool = False
    flag_reason: str | None = None


class RecommendIn(BaseModel):
    visitor_interests: list[str] = Field(default_factory=list)
    visitor_persona_id: int | None = None
    current_booth_id: str
    visited_booth_ids: list[str] = Field(default_factory=list)
    time_left_min: int = Field(60, ge=1, le=720)
    booths: list[BoothNodeIn] = Field(..., min_length=2, max_length=200)
    booth_persona_distribution: dict[str, dict[int, float]] | None = None
    mypass_challenge: ChallengeIn | None = None
    mypass_progress: ProgressIn | None = None
    top_n: int = 3


@router.post("/recommend", response_model=list[dict])
async def recommend_route(req: RecommendIn):
    nodes = [
        BoothNode(
            booth_id=b.booth_id, name=b.name, lat=b.lat, lon=b.lon,
            category=b.category,
        )
        for b in req.booths
    ]
    graph = build_graph(nodes)
    keywords_map = {b.booth_id: b.keywords for b in req.booths}
    crowd_map = {s.booth_id: s for s in tracker.all()}

    recs: list[Recommendation] = recommend_next_booth(
        visitor_interests=req.visitor_interests,
        visitor_persona_id=req.visitor_persona_id,
        current_booth_id=req.current_booth_id,
        visited_booth_ids=set(req.visited_booth_ids),
        time_left_min=req.time_left_min,
        graph=graph,
        booth_keywords=keywords_map,
        crowd=crowd_map,
        booth_persona_distribution=req.booth_persona_distribution,
        top_n=req.top_n,
    )

    if req.mypass_challenge:
        challenge = MyPassChallenge(**req.mypass_challenge.model_dump())  # type: ignore[arg-type]
        progress = (
            MyPassProgress(**req.mypass_progress.model_dump())
            if req.mypass_progress else None
        )
        recs = boost_with_mypass(
            recommendations=recs, challenge=challenge, progress=progress,
        )

    return [to_dict(r) for r in recs]


@router.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "module": "routing",
        "tracked_booths": len(tracker.all()),
        "mock": os.getenv("USE_MOCK", "false").lower() == "true",
    }
