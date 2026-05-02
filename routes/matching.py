"""실시간 매칭 라우터 — Phase 4-A.

POST /matching/icp        : 전시자 ICP 자유 텍스트 → 구조화+임베딩
POST /matching/profile    : 참가자 프로필 → 임베딩 (opt-in 게이트)
POST /matching/scan       : booth 1개 vs candidates → top_k MatchEvent
POST /matching/push       : MatchEvent → FCM (한도 enforce)
DELETE /matching/me       : right-to-erasure 24h SLA 등록
GET  /matching/healthz
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from matching.governance import (
    Consent,
    ErasureRequest,
    PushBudget,
)
from matching.icp_embedder import embed_icp, to_dict as icp_to_dict
from matching.matcher import MatchEvent, match
from matching.push_dispatcher import dispatch
from matching.visitor_profiler import (
    ConsentDeniedError,
    profile_visitor,
    to_dict as profile_to_dict,
)

router = APIRouter(prefix="/matching", tags=["matching"])


# ---------------------------------------------------------------------------
# ICP
# ---------------------------------------------------------------------------
class IcpIn(BaseModel):
    booth_id: str
    raw_text: str = Field(..., min_length=4, max_length=2000)


@router.post("/icp", response_model=dict)
async def icp_route(req: IcpIn):
    icp = embed_icp(booth_id=req.booth_id, raw_text=req.raw_text)
    d = icp_to_dict(icp)
    # 응답 크기 절감 — embedding 은 hash 만 (검증용)
    import hashlib
    d["embedding_hash"] = hashlib.sha256(
        bytes(str(d["embedding"]), "utf-8")
    ).hexdigest()[:16]
    d.pop("embedding", None)
    return d


# ---------------------------------------------------------------------------
# Visitor profile
# ---------------------------------------------------------------------------
class ConsentIn(BaseModel):
    matching: bool = False
    location: bool = False
    analytics: bool = False


class ProfileIn(BaseModel):
    visitor_hash: str
    role: str | None = None
    industry: str | None = None
    interests: list[str] = Field(default_factory=list, max_length=20)
    consent: ConsentIn


@router.post("/profile", response_model=dict)
async def profile_route(req: ProfileIn):
    consent = Consent(**req.consent.model_dump())
    try:
        prof = profile_visitor(
            visitor_hash=req.visitor_hash,
            role=req.role,
            industry=req.industry,
            interests=list(req.interests),
            consent=consent,
        )
    except ConsentDeniedError as exc:
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    d = profile_to_dict(prof)
    d.pop("embedding", None)  # 임베딩은 서버에서만 보관
    return d


# ---------------------------------------------------------------------------
# Scan — 매칭 코어
# ---------------------------------------------------------------------------
class CandidateIn(BaseModel):
    visitor_hash: str
    embedding: list[float] = Field(..., min_length=2)
    distance_m: float = Field(..., ge=0)


class ScanIn(BaseModel):
    booth_id: str
    icp_embedding: list[float] = Field(..., min_length=2)
    icp_keywords: list[str] = Field(default_factory=list)
    candidates: list[CandidateIn] = Field(..., min_length=1, max_length=500)
    score_threshold: float = 0.55
    top_k: int = 3


@router.post("/scan", response_model=list[dict])
async def scan_route(req: ScanIn):
    events = match(
        icp_embedding=req.icp_embedding,
        booth_id=req.booth_id,
        candidates=[
            (c.visitor_hash, c.embedding, c.distance_m) for c in req.candidates
        ],
        now_epoch=int(time.time()),
        score_threshold=req.score_threshold,
        top_k=req.top_k,
        icp_keywords=req.icp_keywords,
    )
    return [asdict(e) for e in events]


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------
class MatchEventIn(BaseModel):
    booth_id: str
    visitor_hash: str
    score: float
    cosine: float
    distance_m: float
    time_bonus: float
    reason: str
    triggered_at: int


class PushIn(BaseModel):
    event: MatchEventIn
    visitor_token: str | None = None
    booth_operator_token: str | None = None
    booth_today: dict[str, int] = Field(default_factory=dict)
    visitor_today: dict[str, int] = Field(default_factory=dict)


@router.post("/push", response_model=dict)
async def push_route(req: PushIn):
    ev = MatchEvent(**req.event.model_dump())
    budget = PushBudget(
        booth_today=dict(req.booth_today),
        visitor_today=dict(req.visitor_today),
    )
    visitor_res, booth_res = dispatch(
        event=ev,
        budget=budget,
        visitor_token=req.visitor_token,
        booth_operator_token=req.booth_operator_token,
    )
    return {
        "visitor": asdict(visitor_res),
        "booth": asdict(booth_res),
        "budget_after": {
            "booth_today": budget.booth_today,
            "visitor_today": budget.visitor_today,
        },
    }


# ---------------------------------------------------------------------------
# Right to erasure
# ---------------------------------------------------------------------------
class ErasureIn(BaseModel):
    visitor_hash: str


@router.delete("/me", response_model=dict)
async def erasure_route(req: ErasureIn):
    er = ErasureRequest.make(req.visitor_hash, int(time.time()))
    # 실 운영: Cloud Function 큐 enqueue → Firestore tombstone + Pinecone delete
    return asdict(er)


@router.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "module": "matching",
        "mock": os.getenv("USE_MOCK", "false").lower() == "true",
    }
