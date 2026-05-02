"""Next event match notifications — smart booth recommendations.

GET /me/matches → list upcoming booth matches (paginated)
POST /me/matches/{match_id}/respond → {"action": "interested" | "dismissed"}
GET /me/matches/settings → {"match_notifications_enabled": bool}
POST /me/matches/settings → toggle notifications

Phase 1: Mock data (3 matches with different countdown days).
Phase 2+: ML ranking + Firestore notifications.
"""
from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/me/matches", tags=["match"])
log = logging.getLogger(__name__)

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Request & Response models
# ---------------------------------------------------------------------------
class RespondRequest(BaseModel):
    action: str = Field(..., pattern=r"^(interested|dismissed)$")


class MatchNotificationSettings(BaseModel):
    match_notifications_enabled: bool


class MatchItem(BaseModel):
    match_id: str
    booth_id: str
    booth_name: str
    company: str
    event_id: str
    days_until_event: int
    relevance_score: float = Field(ge=0.0, le=1.0)
    reason: str  # "Similar company profile", "Trending in your industry", etc.


class MatchListResponse(BaseModel):
    total_matches: int
    matches: list[MatchItem]


# ---------------------------------------------------------------------------
# In-memory match state (mock only)
# ---------------------------------------------------------------------------
_USER_MATCH_SETTINGS = {"match_notifications_enabled": True}
_USER_RESPONSES: dict[str, str] = {}  # match_id -> "interested" | "dismissed"


# ---------------------------------------------------------------------------
# Mock data — 3 upcoming matches
# ---------------------------------------------------------------------------
_MOCK_MATCHES = [
    MatchItem(
        match_id="MATCH-001",
        booth_id="B-2026-002-010",
        booth_name="Lumen Labs",
        company="Lumen Labs Inc.",
        event_id="E-2026-002",
        days_until_event=7,
        relevance_score=0.95,
        reason="Similar AI/automation focus based on your booth visits",
    ),
    MatchItem(
        match_id="MATCH-002",
        booth_id="B-2026-003-042",
        booth_name="Paperless Co.",
        company="Paperless Innovations",
        event_id="E-2026-003",
        days_until_event=14,
        relevance_score=0.78,
        reason="Trending in document automation (your search category)",
    ),
    MatchItem(
        match_id="MATCH-003",
        booth_id="B-2026-004-015",
        booth_name="PulseGrid",
        company="PulseGrid Technologies",
        event_id="E-2026-004",
        days_until_event=21,
        relevance_score=0.62,
        reason="Strong network overlap with your connections",
    ),
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("", response_model=MatchListResponse)
async def list_matches(
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
) -> MatchListResponse:
    """
    GET /me/matches — List upcoming booth matches.

    Returns paginated list of smart recommendations for future events.
    """
    if not USE_MOCK:
        # TODO: ML ranking pipeline (similarity + trends + network)
        # TODO: Firestore query user matches
        raise HTTPException(503, "live data integration in Phase 2")

    # Mock: return 3 hardcoded matches
    matches = [m for m in _MOCK_MATCHES if m.match_id not in _USER_RESPONSES or _USER_RESPONSES[m.match_id] != "dismissed"]

    return MatchListResponse(
        total_matches=len(matches),
        matches=matches[offset : offset + limit],
    )


@router.post("/{match_id}/respond")
async def respond_to_match(match_id: str, req: RespondRequest) -> dict:
    """
    POST /me/matches/{match_id}/respond

    Record user response: "interested" or "dismissed".
    """
    if not USE_MOCK:
        # TODO: Firestore update user match response
        # TODO: trigger follow-up notification if "interested"
        raise HTTPException(503, "live data integration in Phase 2")

    # Find match
    match = next((m for m in _MOCK_MATCHES if m.match_id == match_id), None)
    if not match:
        raise HTTPException(404, "match not found")

    _USER_RESPONSES[match_id] = req.action
    log.info("User responded to match %s: %s", match_id, req.action)

    return {
        "match_id": match_id,
        "action": req.action,
        "status": "recorded",
    }


@router.get("/settings", response_model=MatchNotificationSettings)
async def get_match_settings() -> MatchNotificationSettings:
    """
    GET /me/matches/settings — Get notification preferences.
    """
    if not USE_MOCK:
        # TODO: Firestore user settings lookup
        raise HTTPException(503, "live data integration in Phase 2")

    return MatchNotificationSettings(
        match_notifications_enabled=_USER_MATCH_SETTINGS["match_notifications_enabled"]
    )


@router.post("/settings", response_model=MatchNotificationSettings)
async def update_match_settings(
    req: MatchNotificationSettings,
) -> MatchNotificationSettings:
    """
    POST /me/matches/settings — Toggle match notifications.
    """
    if not USE_MOCK:
        # TODO: Firestore user settings update
        raise HTTPException(503, "live data integration in Phase 2")

    _USER_MATCH_SETTINGS["match_notifications_enabled"] = req.match_notifications_enabled
    log.info(
        "Match notifications toggled: %s",
        req.match_notifications_enabled,
    )

    return MatchNotificationSettings(
        match_notifications_enabled=_USER_MATCH_SETTINGS["match_notifications_enabled"]
    )


@router.get("/healthz", tags=["health"])
async def healthz() -> dict:
    return {"ok": True, "module": "match", "mock": USE_MOCK}
