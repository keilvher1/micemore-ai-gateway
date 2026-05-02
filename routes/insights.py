"""인사이트 리포트 라우터.

POST /insights/aggregate     : raw rows → EventStats
POST /insights/generate      : EventStats → markdown report (3 audience)
POST /insights/render-html   : report → standalone HTML
GET  /insights/healthz
"""
from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from insights.aggregator import EventStats, aggregate
from insights.pdf_renderer import render_html
from insights.report_generator import Report, generate, has_banned_words

router = APIRouter(prefix="/insights", tags=["insights"])


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
class AggregateIn(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    event_id: str
    period_start: int
    period_end: int
    booth_names: dict[str, str] | None = None


@router.post("/aggregate", response_model=dict)
async def aggregate_route(req: AggregateIn):
    stats = aggregate(
        req.rows,
        event_id=req.event_id,
        period_start=req.period_start,
        period_end=req.period_end,
        booth_names=req.booth_names,
    )
    # dataclass → dict
    from dataclasses import asdict
    return asdict(stats)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
class GenerateIn(BaseModel):
    stats: dict[str, Any]
    audience: Literal["organizer", "exhibitor", "municipality"]
    booth_id: str | None = None
    region: str | None = None
    category_avg: dict[str, Any] | None = None


class GenerateOut(BaseModel):
    audience: str
    markdown: str
    model: str
    banned_words_found: list[str]


def _stats_from_dict(d: dict) -> EventStats:
    # dataclass 의 reconstruction — 필드 누락 시 default 적용
    from insights.aggregator import BoothStat, HourBucket
    booths = [BoothStat(**b) for b in d.get("booths", [])]
    hours = [HourBucket(**h) for h in d.get("hours", [])]
    return EventStats(
        event_id=d.get("event_id", ""),
        period_start=int(d.get("period_start", 0)),
        period_end=int(d.get("period_end", 0)),
        total_visitors=int(d.get("total_visitors", 0)),
        foreigner_ratio=float(d.get("foreigner_ratio", 0)),
        translation_sessions=int(d.get("translation_sessions", 0)),
        avg_dwell_min=float(d.get("avg_dwell_min", 0)),
        booths=booths,
        hours=hours,
        age_distribution=dict(d.get("age_distribution", {})),
        gender_distribution=dict(d.get("gender_distribution", {})),
        top_topics=[tuple(t) for t in d.get("top_topics", [])],
        nps_score=d.get("nps_score"),
        nps_comments_summary=d.get("nps_comments_summary"),
    )


@router.post("/generate", response_model=GenerateOut)
async def generate_route(req: GenerateIn):
    stats = _stats_from_dict(req.stats)
    rep: Report = generate(
        stats=stats,
        audience=req.audience,
        booth_id=req.booth_id,
        region=req.region,
        category_avg=req.category_avg,
    )
    return GenerateOut(
        audience=rep.audience,
        markdown=rep.markdown,
        model=rep.model,
        banned_words_found=has_banned_words(rep.markdown),
    )


# ---------------------------------------------------------------------------
# Render HTML
# ---------------------------------------------------------------------------
class RenderIn(BaseModel):
    audience: Literal["organizer", "exhibitor", "municipality"]
    markdown: str
    model: str = "claude"
    charts_caption: list[str] = Field(default_factory=list)


@router.post("/render-html")
async def render_html_route(req: RenderIn):
    rep = Report(
        audience=req.audience,
        markdown=req.markdown,
        model=req.model,
        raw_prompt="",
    )
    html = render_html(rep, charts_caption=req.charts_caption)
    return {"html": html, "size_bytes": len(html.encode("utf-8"))}


@router.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "module": "insights",
        "mock": os.getenv("USE_MOCK", "false").lower() == "true",
    }


# ---------------------------------------------------------------------------
# D-4 단계 5 — POST /insights/generate-and-send
# 행사 종료 24h 후 (Cloud Scheduler) 호출. KPI aggregate → PDF → SendGrid 발송.
# WeasyPrint 미설치 또는 SENDGRID_API_KEY=PLACEHOLDER 면 mock dry-run 결과 반환.
# ---------------------------------------------------------------------------
PLACEHOLDER_TOKENS = ("PLACEHOLDER", "SG.PLACE")


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    upper = value.upper()
    return any(token.upper() in upper for token in PLACEHOLDER_TOKENS)


class GenerateAndSendIn(BaseModel):
    event_id: str
    audience: Literal["organizer", "exhibitor", "municipality"] = "organizer"
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    period_start: int
    period_end: int
    booth_names: dict[str, str] | None = None
    to_email: str
    booth_id: str | None = None
    region: str | None = None


class GenerateAndSendOut(BaseModel):
    event_id: str
    audience: str
    pdf_bytes: int
    sent: bool
    provider_id: str | None = None
    s3_url: str | None = None
    fallback_reason: str | None = None


@router.post("/generate-and-send", response_model=GenerateAndSendOut)
async def generate_and_send(req: GenerateAndSendIn):
    """KPI aggregate → markdown report → HTML → PDF (WeasyPrint) → S3 → SendGrid."""
    stats = aggregate(
        req.rows,
        event_id=req.event_id,
        period_start=req.period_start,
        period_end=req.period_end,
        booth_names=req.booth_names,
    )
    rep: Report = generate(
        stats=stats, audience=req.audience, booth_id=req.booth_id, region=req.region,
    )
    html = render_html(rep)

    pdf_bytes_data: bytes = b""
    fallback_reason: str | None = None
    try:
        from weasyprint import HTML  # type: ignore

        pdf_bytes_data = HTML(string=html).write_pdf() or b""
    except Exception as exc:  # noqa: BLE001
        fallback_reason = f"weasyprint unavailable: {exc.__class__.__name__}"
        pdf_bytes_data = html.encode("utf-8")  # PDF 미생성 시 HTML payload 그대로

    s3_url: str | None = None
    bucket = os.getenv("REPORT_S3_BUCKET")
    if bucket and not _is_placeholder(bucket):
        try:
            import boto3  # type: ignore

            key = f"reports/{req.event_id}/{req.audience}.pdf"
            boto3.client("s3").put_object(
                Bucket=bucket, Key=key, Body=pdf_bytes_data, ContentType="application/pdf",
            )
            s3_url = f"s3://{bucket}/{key}"
        except Exception as exc:  # noqa: BLE001
            fallback_reason = (fallback_reason or "") + f"; s3 upload failed: {exc}"

    sent = False
    provider_id: str | None = None
    sg_key = os.getenv("SENDGRID_API_KEY")
    use_mock = os.getenv("USE_MOCK", "false").lower() == "true"
    if use_mock or _is_placeholder(sg_key):
        sent = True
        provider_id = "mock-insights-id"
        fallback_reason = (fallback_reason or "") + "; sendgrid placeholder → mock send"
    else:
        try:
            from sendgrid import SendGridAPIClient  # type: ignore
            from sendgrid.helpers.mail import (  # type: ignore
                Attachment, FileContent, FileName, FileType, Mail,
            )
            import base64

            mail = Mail(
                from_email=os.getenv("FROM_EMAIL", "noreply@micemore.com"),
                to_emails=req.to_email,
                subject=f"[MICEMORE 인사이트] {req.event_id} — {req.audience} 리포트",
                plain_text_content=rep.markdown[:1500],
            )
            mail.attachment = Attachment(
                FileContent(base64.b64encode(pdf_bytes_data).decode()),
                FileName(f"{req.event_id}_{req.audience}.pdf"),
                FileType("application/pdf"),
            )
            res = SendGridAPIClient(sg_key).send(mail)
            sent = 200 <= res.status_code < 300
            provider_id = (res.headers or {}).get("X-Message-Id")
        except Exception as exc:  # noqa: BLE001
            fallback_reason = (fallback_reason or "") + f"; sendgrid failed: {exc}"

    return GenerateAndSendOut(
        event_id=req.event_id,
        audience=req.audience,
        pdf_bytes=len(pdf_bytes_data),
        sent=sent,
        provider_id=provider_id,
        s3_url=s3_url,
        fallback_reason=fallback_reason or None,
    )
