"""Operator visitor analytics — booth visitor list + engagement metrics.

GET /operator/booth/{booth_id}/visitors → aggregated visitor stats + language distribution

Phase 1: Mock data (6 visitors with realistic intent scores + language distribution).
Phase 2+: Firestore aggregation pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/operator", tags=["visitors"])
log = logging.getLogger(__name__)

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Request & Response models
# ---------------------------------------------------------------------------
class VisitorSummary(BaseModel):
    topics: list[str]
    intent_score: float = Field(ge=0.0, le=1.0)


class VisitorRecord(BaseModel):
    tap_id: str
    anonymized_user_id: str
    first_tapped_at: str
    duration_sec: int = Field(ge=0)
    chat_messages_count: int = Field(ge=0)
    user_language: str
    summary: VisitorSummary


class LanguageDistribution(BaseModel):
    ko: float = Field(ge=0.0, le=1.0)
    en: float = Field(ge=0.0, le=1.0)
    ja: float = Field(ge=0.0, le=1.0)
    zh: float = Field(ge=0.0, le=1.0)


class VisitorListResponse(BaseModel):
    booth_id: str
    total_visitors: int
    unique_visitors: int
    total_chat_messages: int
    language_distribution: LanguageDistribution
    visitors: list[VisitorRecord]


# ---------------------------------------------------------------------------
# Mock data — 6 visitors for Lumen Labs booth
# ---------------------------------------------------------------------------
_MOCK_VISITORS = [
    VisitorRecord(
        tap_id="TAP-001",
        anonymized_user_id="usr_a1b2c3",
        first_tapped_at="2026-05-01T09:15:00Z",
        duration_sec=420,
        chat_messages_count=8,
        user_language="ko",
        summary=VisitorSummary(
            topics=["pricing", "integration", "timeline"],
            intent_score=0.92,
        ),
    ),
    VisitorRecord(
        tap_id="TAP-002",
        anonymized_user_id="usr_d4e5f6",
        first_tapped_at="2026-05-01T09:45:00Z",
        duration_sec=360,
        chat_messages_count=6,
        user_language="ko",
        summary=VisitorSummary(
            topics=["tech_stack", "support", "pricing"],
            intent_score=0.81,
        ),
    ),
    VisitorRecord(
        tap_id="TAP-003",
        anonymized_user_id="usr_g7h8i9",
        first_tapped_at="2026-05-01T10:20:00Z",
        duration_sec=240,
        chat_messages_count=4,
        user_language="en",
        summary=VisitorSummary(
            topics=["overview", "demo"],
            intent_score=0.65,
        ),
    ),
    VisitorRecord(
        tap_id="TAP-004",
        anonymized_user_id="usr_j0k1l2",
        first_tapped_at="2026-05-01T11:00:00Z",
        duration_sec=180,
        chat_messages_count=3,
        user_language="ko",
        summary=VisitorSummary(
            topics=["features"],
            intent_score=0.41,
        ),
    ),
    VisitorRecord(
        tap_id="TAP-005",
        anonymized_user_id="usr_m3n4o5",
        first_tapped_at="2026-05-01T11:35:00Z",
        duration_sec=120,
        chat_messages_count=2,
        user_language="ja",
        summary=VisitorSummary(
            topics=["general"],
            intent_score=0.28,
        ),
    ),
    VisitorRecord(
        tap_id="TAP-006",
        anonymized_user_id="usr_p6q7r8",
        first_tapped_at="2026-05-01T12:10:00Z",
        duration_sec=60,
        chat_messages_count=0,
        user_language="zh",
        summary=VisitorSummary(
            topics=[],
            intent_score=0.12,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("/booth/{booth_id}/visitors", response_model=VisitorListResponse)
async def get_booth_visitors(
    booth_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> VisitorListResponse:
    """
    GET /operator/booth/{booth_id}/visitors

    Returns:
        - total_visitors: 87
        - unique_visitors: 73
        - total_chat_messages: 23 (aggregated)
        - language_distribution: {ko: 52%, en: 31%, ja: 12%, zh: 5%}
        - visitors: paginated list with intent_score + summary
    """
    if not USE_MOCK:
        # TODO: Firestore aggregation pipeline
        # TODO: compute intent_score from visitor behavior
        # TODO: aggregate language distribution
        raise HTTPException(503, "live data integration in Phase 2")

    # Mock: return Lumen Labs booth (B-2026-001-042) visitor list
    if booth_id != "B-2026-001-042":
        # Return empty for other booths
        return VisitorListResponse(
            booth_id=booth_id,
            total_visitors=0,
            unique_visitors=0,
            total_chat_messages=0,
            language_distribution=LanguageDistribution(ko=0, en=0, ja=0, zh=0),
            visitors=[],
        )

    # Aggregate stats
    total_msgs = sum(v.chat_messages_count for v in _MOCK_VISITORS)

    return VisitorListResponse(
        booth_id=booth_id,
        total_visitors=87,
        unique_visitors=73,
        total_chat_messages=total_msgs,
        language_distribution=LanguageDistribution(
            ko=0.52,
            en=0.31,
            ja=0.12,
            zh=0.05,
        ),
        visitors=_MOCK_VISITORS[offset : offset + limit],
    )


@router.get("/healthz", tags=["health"])
async def healthz() -> dict:
    return {"ok": True, "module": "visitors", "mock": USE_MOCK}
