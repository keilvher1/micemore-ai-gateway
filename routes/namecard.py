"""Digital namecard CRUD — saved booths management.

GET /me/namecard — list saved booths (paginated)
GET /me/namecard/{saved_booth_id} — detail with materials, contacts, chat_history
POST /me/namecard/{saved_booth_id}/notes — update notes
POST /me/namecard/{saved_booth_id}/tags — update tags
POST /me/namecard/{saved_booth_id}/pin — toggle pin state

Phase 1: Mock data only (3 saved booths).
Phase 2+: Firestore lookups + user context.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/me/namecard", tags=["namecard"])
log = logging.getLogger(__name__)

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Request & Response models
# ---------------------------------------------------------------------------
class NotesUpdate(BaseModel):
    notes: str = Field(..., min_length=0, max_length=5000)


class TagsUpdate(BaseModel):
    tags: list[str] = Field(..., min_items=0, max_items=20)


class PinUpdate(BaseModel):
    is_pinned: bool


class BoothMaterial(BaseModel):
    material_id: str
    title: str
    type: str  # "pdf", "image", "video", "link"
    url: str


class BoothContact(BaseModel):
    contact_id: str
    name: str
    title: str
    email: str


class ChatMessage(BaseModel):
    timestamp: str
    role: str  # "user", "copilot"
    text: str


class SavedBoothListItem(BaseModel):
    saved_booth_id: str
    booth_id: str
    booth_name: str
    company: str
    saved_at: str
    is_pinned: bool
    tags: list[str]
    notes: str


class SavedBoothDetail(BaseModel):
    saved_booth_id: str
    booth_id: str
    booth_name: str
    company: str
    saved_at: str
    is_pinned: bool
    tags: list[str]
    notes: str
    materials: list[BoothMaterial]
    contacts: list[BoothContact]
    chat_history: list[ChatMessage]


# ---------------------------------------------------------------------------
# Mock data — 3 saved booths
# ---------------------------------------------------------------------------
_MOCK_SAVED_BOOTHS = {
    "SB-001": SavedBoothDetail(
        saved_booth_id="SB-001",
        booth_id="B-2026-001-001",
        booth_name="Acme Inc.",
        company="Acme Inc.",
        saved_at="2026-05-01T14:32:00Z",
        is_pinned=True,
        tags=["software", "ai", "follow-up"],
        notes="Very interested in their ML ops platform. Sarah mentioned Q3 pilot.",
        materials=[
            BoothMaterial(
                material_id="M-A1",
                title="Acme ML Ops Whitepaper",
                type="pdf",
                url="https://example.com/acme-mlops.pdf",
            ),
            BoothMaterial(
                material_id="M-A2",
                title="Demo Video",
                type="video",
                url="https://example.com/acme-demo.mp4",
            ),
        ],
        contacts=[
            BoothContact(
                contact_id="C-A1",
                name="Alice Johnson",
                title="VP Sales",
                email="alice@acme.io",
            ),
        ],
        chat_history=[
            ChatMessage(
                timestamp="2026-05-01T14:35:00Z",
                role="user",
                text="What's your pricing model?",
            ),
            ChatMessage(
                timestamp="2026-05-01T14:35:30Z",
                role="copilot",
                text="Acme offers flexible per-seat and usage-based pricing...",
            ),
        ],
    ),
    "SB-002": SavedBoothDetail(
        saved_booth_id="SB-002",
        booth_id="B-2026-001-015",
        booth_name="BlueCorp",
        company="BlueCorp Solutions",
        saved_at="2026-05-01T15:10:00Z",
        is_pinned=False,
        tags=["analytics", "data"],
        notes="Interesting but needs to evaluate against current stack.",
        materials=[
            BoothMaterial(
                material_id="M-B1",
                title="BlueCorp Analytics Guide",
                type="pdf",
                url="https://example.com/bluecorp-guide.pdf",
            ),
        ],
        contacts=[
            BoothContact(
                contact_id="C-B1",
                name="Bob Chen",
                title="Sales Engineer",
                email="bob@bluecorp.io",
            ),
        ],
        chat_history=[
            ChatMessage(
                timestamp="2026-05-01T15:12:00Z",
                role="user",
                text="Do you integrate with Snowflake?",
            ),
            ChatMessage(
                timestamp="2026-05-01T15:12:45Z",
                role="copilot",
                text="Yes, full Snowflake integration. Native connector available.",
            ),
        ],
    ),
    "SB-003": SavedBoothDetail(
        saved_booth_id="SB-003",
        booth_id="B-2026-001-028",
        booth_name="GreenTech",
        company="GreenTech Innovations",
        saved_at="2026-04-30T11:50:00Z",
        is_pinned=False,
        tags=["sustainability", "climate"],
        notes="Cool ESG reporting tools, but early stage.",
        materials=[
            BoothMaterial(
                material_id="M-G1",
                title="ESG Reporting Framework",
                type="pdf",
                url="https://example.com/greentech-esg.pdf",
            ),
            BoothMaterial(
                material_id="M-G2",
                title="Carbon Accounting Demo",
                type="video",
                url="https://example.com/greentech-carbon.mp4",
            ),
        ],
        contacts=[
            BoothContact(
                contact_id="C-G1",
                name="Grace Lee",
                title="Founder & CEO",
                email="grace@greentech.io",
            ),
            BoothContact(
                contact_id="C-G2",
                name="Henry Liu",
                title="Product Lead",
                email="henry@greentech.io",
            ),
        ],
        chat_history=[],
    ),
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("", response_model=list[SavedBoothListItem])
async def list_namecards(
    event_id: str | None = Query(default=None),
    sort: str = Query(default="recent", pattern=r"^(recent|pinned|name)$"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[SavedBoothListItem]:
    """List saved booths (paginated)."""
    if not USE_MOCK:
        # TODO: Firestore query by user_id + event_id
        # TODO: implement pagination
        raise HTTPException(503, "live data integration in Phase 2")

    items = [
        SavedBoothListItem(
            saved_booth_id=detail.saved_booth_id,
            booth_id=detail.booth_id,
            booth_name=detail.booth_name,
            company=detail.company,
            saved_at=detail.saved_at,
            is_pinned=detail.is_pinned,
            tags=detail.tags,
            notes=detail.notes,
        )
        for detail in _MOCK_SAVED_BOOTHS.values()
    ]

    # Sort
    if sort == "pinned":
        items.sort(key=lambda x: (not x.is_pinned, x.saved_at), reverse=True)
    elif sort == "name":
        items.sort(key=lambda x: x.booth_name)
    else:  # "recent"
        items.sort(key=lambda x: x.saved_at, reverse=True)

    return items[offset : offset + limit]


@router.get("/{saved_booth_id}", response_model=SavedBoothDetail)
async def get_namecard_detail(saved_booth_id: str) -> SavedBoothDetail:
    """Get detailed namecard (materials, contacts, chat history)."""
    if not USE_MOCK:
        # TODO: Firestore lookup
        raise HTTPException(503, "live data integration in Phase 2")

    if saved_booth_id not in _MOCK_SAVED_BOOTHS:
        raise HTTPException(404, "namecard not found")

    return _MOCK_SAVED_BOOTHS[saved_booth_id]


@router.post("/{saved_booth_id}/notes")
async def update_notes(saved_booth_id: str, req: NotesUpdate) -> dict:
    """Update notes on saved booth."""
    if not USE_MOCK:
        # TODO: Firestore update
        raise HTTPException(503, "live data integration in Phase 2")

    if saved_booth_id not in _MOCK_SAVED_BOOTHS:
        raise HTTPException(404, "namecard not found")

    # Mock: just confirm
    return {"saved_booth_id": saved_booth_id, "notes_updated": True}


@router.post("/{saved_booth_id}/tags")
async def update_tags(saved_booth_id: str, req: TagsUpdate) -> dict:
    """Update tags on saved booth."""
    if not USE_MOCK:
        # TODO: Firestore update
        raise HTTPException(503, "live data integration in Phase 2")

    if saved_booth_id not in _MOCK_SAVED_BOOTHS:
        raise HTTPException(404, "namecard not found")

    return {"saved_booth_id": saved_booth_id, "tags": req.tags}


@router.post("/{saved_booth_id}/pin")
async def toggle_pin(saved_booth_id: str, req: PinUpdate) -> dict:
    """Toggle pin state on saved booth."""
    if not USE_MOCK:
        # TODO: Firestore update
        raise HTTPException(503, "live data integration in Phase 2")

    if saved_booth_id not in _MOCK_SAVED_BOOTHS:
        raise HTTPException(404, "namecard not found")

    return {"saved_booth_id": saved_booth_id, "is_pinned": req.is_pinned}


@router.get("/healthz", tags=["health"])
async def healthz() -> dict:
    return {"ok": True, "module": "namecard", "mock": USE_MOCK}
