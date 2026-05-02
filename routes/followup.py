"""Follow-up 메일 라우터.

POST /followup/draft     : 단일 톤·언어 초안 1개
POST /followup/draft-ab  : 두 톤 A/B 초안 두 개
POST /followup/send      : 초안 발송 (SendGrid)
GET  /followup/healthz
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from followup.generator import (
    ExhibitorContext,
    FollowupDraft,
    VisitorContext,
    generate_ab_pair,
    generate_draft,
)
from followup.sender import SendResult, send
from followup.templates import Lang, Tone

router = APIRouter(prefix="/followup", tags=["followup"])
log = logging.getLogger("followup")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
class VisitorIn(BaseModel):
    visitor_name: str
    visited_at: str
    booth_name: str
    dwell_minutes: int = Field(ge=0, le=600)
    copilot_questions: list[str] = Field(default_factory=list)
    translation_summary: str | None = None
    competitor_booths: list[str] | None = None
    pamphlet_pages_viewed: list[int] | None = None


class ExhibitorIn(BaseModel):
    exhibitor_name: str
    company_name: str
    cta_calendly_url: str | None = None
    cta_pdf_url: str | None = None
    signature: str | None = None


class DraftIn(BaseModel):
    visitor: VisitorIn
    exhibitor: ExhibitorIn
    lang: Lang = "ko"
    tone: Tone = "balanced"


class DraftABIn(BaseModel):
    visitor: VisitorIn
    exhibitor: ExhibitorIn
    lang: Lang = "ko"
    tone_a: Tone = "formal"
    tone_b: Tone = "balanced"


class DraftOut(BaseModel):
    subject: str
    body: str
    cta_type: str
    cta_label: str
    tone: Tone
    lang: Lang
    model: str


def _to_visitor(v: VisitorIn) -> VisitorContext:
    return VisitorContext(**v.model_dump())


def _to_exhibitor(e: ExhibitorIn) -> ExhibitorContext:
    return ExhibitorContext(**e.model_dump())


def _to_out(d: FollowupDraft) -> DraftOut:
    return DraftOut(
        subject=d.subject, body=d.body, cta_type=d.cta_type,
        cta_label=d.cta_label, tone=d.tone, lang=d.lang, model=d.model,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/draft", response_model=DraftOut)
async def draft(req: DraftIn):
    d = generate_draft(
        visitor=_to_visitor(req.visitor),
        exhibitor=_to_exhibitor(req.exhibitor),
        lang=req.lang, tone=req.tone,
    )
    return _to_out(d)


@router.post("/draft-ab", response_model=list[DraftOut])
async def draft_ab(req: DraftABIn):
    a, b = generate_ab_pair(
        visitor=_to_visitor(req.visitor),
        exhibitor=_to_exhibitor(req.exhibitor),
        lang=req.lang, tone_a=req.tone_a, tone_b=req.tone_b,
    )
    return [_to_out(a), _to_out(b)]


class SendIn(BaseModel):
    draft: DraftOut
    to_email: EmailStr
    ab_arm: str = "A"
    event_id: str | None = None
    visitor_hash: str | None = None


class SendOut(BaseModel):
    accepted: bool
    provider_id: str | None
    status_code: int
    error: str | None = None


@router.post("/send", response_model=SendOut)
async def send_route(req: SendIn):
    d = FollowupDraft(
        subject=req.draft.subject,
        body=req.draft.body,
        cta_type=req.draft.cta_type,  # type: ignore[arg-type]
        cta_label=req.draft.cta_label,
        tone=req.draft.tone,
        lang=req.draft.lang,
        model=req.draft.model,
        raw_prompt="",  # 발송에는 사용 안 함
    )
    res: SendResult = send(
        draft=d,
        to_email=str(req.to_email),
        ab_arm=req.ab_arm,
        event_id=req.event_id,
        visitor_hash=req.visitor_hash,
    )
    if not res.accepted and res.error and "SENDGRID_API_KEY" in res.error:
        # 키 없는 환경 → 클라에 명확한 신호
        raise HTTPException(status_code=503, detail=res.error)
    return SendOut(**res.__dict__)


@router.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "module": "followup",
        "mock": os.getenv("USE_MOCK", "false").lower() == "true",
    }
