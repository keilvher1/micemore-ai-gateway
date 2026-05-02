"""NFC tap entry point — POST /nfc/tap.

Handles NFC tap events at booth entrance. Returns booth info + opens session.
Phase 1: Mock data only (Lumen Labs booth).
Phase 2+: Firestore lookup + session creation.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/nfc", tags=["nfc"])
log = logging.getLogger(__name__)

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Request & Response models
# ---------------------------------------------------------------------------
class DeviceInfo(BaseModel):
    platform: str = Field(..., pattern=r"^(ios|android|web)$")
    app_version: str = Field(..., min_length=1, max_length=32)
    locale: str = Field(default="en-US")


class NFCTapRequest(BaseModel):
    booth_id: str = Field(..., min_length=1, max_length=128)
    event_id: str = Field(..., min_length=1, max_length=128)
    timestamp: int = Field(..., ge=0)
    device_info: DeviceInfo


class BoothMaterial(BaseModel):
    material_id: str
    title: str
    type: str  # "pdf", "image", "video", "link"
    url: str
    size_bytes: int | None = None


class BoothContact(BaseModel):
    contact_id: str
    name: str
    title: str
    email: str
    phone: str | None = None


class BoothInfo(BaseModel):
    id: str
    name: str
    company: str
    description: str
    materials: list[BoothMaterial]
    contacts: list[BoothContact]


class CharacterInfo(BaseModel):
    id: str
    initial_message: str
    voice_id: str


class Analytics(BaseModel):
    is_first_visit: bool
    user_visit_count: int


class NFCTapResponse(BaseModel):
    session_id: str
    booth: BoothInfo
    character: CharacterInfo
    analytics: Analytics


# ---------------------------------------------------------------------------
# Mock data — Lumen Labs booth (B-2026-001-042)
# ---------------------------------------------------------------------------
_MOCK_BOOTH = BoothInfo(
    id="B-2026-001-042",
    name="Lumen Labs",
    company="Lumen Labs Inc.",
    description="AI-powered event experience platform. Live translation, booth copilot, lead insights.",
    materials=[
        BoothMaterial(
            material_id="M-001",
            title="Product Overview Brochure",
            type="pdf",
            url="https://example.com/lumen-overview.pdf",
            size_bytes=2400000,
        ),
        BoothMaterial(
            material_id="M-002",
            title="Case Study: APAC Rollout",
            type="pdf",
            url="https://example.com/lumen-casestudy.pdf",
            size_bytes=1800000,
        ),
        BoothMaterial(
            material_id="M-003",
            title="Live Demo Video (2min)",
            type="video",
            url="https://example.com/lumen-demo.mp4",
            size_bytes=45000000,
        ),
    ],
    contacts=[
        BoothContact(
            contact_id="C-001",
            name="Sarah Kim",
            title="Chief Product Officer",
            email="sarah.kim@lumen-labs.io",
            phone="+82-2-1234-5678",
        ),
        BoothContact(
            contact_id="C-002",
            name="James Park",
            title="Head of Sales, APAC",
            email="james.park@lumen-labs.io",
            phone="+65-6123-4567",
        ),
    ],
)

_MOCK_CHARACTER = CharacterInfo(
    id="boomi",
    initial_message="안녕하세요! 저는 부스 도우미 '부미'입니다. Lumen Labs에 오신 것을 환영합니다.",
    voice_id="ko-KR-Neural2-C",
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/tap", response_model=NFCTapResponse)
async def nfc_tap(request: NFCTapRequest) -> NFCTapResponse:
    """
    NFC tap entry point.

    Returns:
        - session_id: newly created session
        - booth: full booth info (materials, contacts)
        - character: copilot character details
        - analytics: is_first_visit, visit_count
    """
    if not USE_MOCK:
        # TODO: integrate with Firestore booth lookup
        # TODO: create session in Firestore
        # TODO: fetch analytics (user_visit_count, is_first_visit)
        raise HTTPException(503, "live data integration in Phase 2")

    # Mock mode: return Lumen Labs booth
    session_id = f"S-{request.event_id}-{uuid.uuid4().hex[:8]}"

    return NFCTapResponse(
        session_id=session_id,
        booth=_MOCK_BOOTH,
        character=_MOCK_CHARACTER,
        analytics=Analytics(is_first_visit=True, user_visit_count=1),
    )


@router.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "module": "nfc", "mock": USE_MOCK}
