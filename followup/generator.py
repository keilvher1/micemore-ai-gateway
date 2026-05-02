"""Follow-up 메일 초안 생성기 — Claude.

전시자가 톤·언어 선택 → A 안 + B 안 두 개 동시 생성 → 둘 중 하나 발송.
A/B 메트릭은 sender.py + SendGrid event tracking 으로 수집.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

from followup.templates import (
    COMMON_RULES,
    Lang,
    Tone,
    get_system_prompt,
)

log = logging.getLogger("followup.generator")

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass
class VisitorContext:
    visitor_name: str
    visited_at: str               # ISO date — "2026-06-19"
    booth_name: str
    dwell_minutes: int
    copilot_questions: list[str]  # 핵심 질문 텍스트 (개인정보 제외)
    translation_summary: str | None = None  # 통역 대화 요약 1~2줄
    competitor_booths: list[str] | None = None
    pamphlet_pages_viewed: list[int] | None = None


@dataclass
class ExhibitorContext:
    exhibitor_name: str
    company_name: str
    cta_calendly_url: str | None = None
    cta_pdf_url: str | None = None
    signature: str | None = None


@dataclass
class FollowupDraft:
    subject: str
    body: str
    cta_type: Literal["calendly", "pdf", "demo"]
    cta_label: str
    tone: Tone
    lang: Lang
    model: str        # "claude" | "gpt-4o" | "mock"
    raw_prompt: str   # 디버깅 / 재현용


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------
def _format_context(v: VisitorContext) -> str:
    qs = "\n".join(f"  • {q}" for q in v.copilot_questions[:5]) or "  (no questions)"
    pages = ", ".join(str(p) for p in (v.pamphlet_pages_viewed or [])) or "(none)"
    comps = ", ".join(v.competitor_booths or []) or "(none)"
    trans = v.translation_summary or "(no translation session)"
    return (
        f"Visitor: {v.visitor_name}\n"
        f"Visited: {v.visited_at} at {v.booth_name}\n"
        f"Dwell time: {v.dwell_minutes} min\n"
        f"Copilot questions:\n{qs}\n"
        f"Translation summary: {trans}\n"
        f"Pamphlet pages viewed: {pages}\n"
        f"Competitor booths visited: {comps}"
    )


def _format_exhibitor(e: ExhibitorContext) -> str:
    bits = [f"Exhibitor: {e.exhibitor_name} ({e.company_name})"]
    if e.cta_calendly_url:
        bits.append(f"Calendly: {e.cta_calendly_url}")
    if e.cta_pdf_url:
        bits.append(f"Materials PDF: {e.cta_pdf_url}")
    if e.signature:
        bits.append(f"Signature note: {e.signature}")
    return "\n".join(bits)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(
    *,
    visitor: VisitorContext,
    exhibitor: ExhibitorContext,
    lang: Lang,
    tone: Tone,
) -> str:
    template = get_system_prompt(lang, tone)
    return template.format(
        context=_format_context(visitor),
        exhibitor_context=_format_exhibitor(exhibitor),
        visitor_name=visitor.visitor_name,
        exhibitor_name=exhibitor.exhibitor_name,
        common_rules=COMMON_RULES,
    )


# ---------------------------------------------------------------------------
# JSON 파싱 (LLM 출력의 안전한 추출)
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_llm_json(text: str) -> dict:
    """LLM 이 markdown fence 로 감싸 보내도 추출."""
    # 1) 통째로 시도
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) 첫 JSON 블록 추출
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON in LLM output: {text[:200]}")
    return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def generate_draft(
    *,
    visitor: VisitorContext,
    exhibitor: ExhibitorContext,
    lang: Lang = "ko",
    tone: Tone = "balanced",
) -> FollowupDraft:
    prompt = build_prompt(
        visitor=visitor, exhibitor=exhibitor, lang=lang, tone=tone
    )

    if USE_MOCK:
        return _mock_draft(visitor, exhibitor, lang, tone, prompt)

    # 실 호출 — Claude primary, GPT-4o fallback
    try:
        text, model_used = _call_claude(prompt)
    except Exception as exc:  # noqa: BLE001
        log.warning("Claude failed (%s) → GPT fallback", exc)
        text, model_used = _call_gpt(prompt)

    parsed = _parse_llm_json(text)
    return FollowupDraft(
        subject=str(parsed["subject"]),
        body=str(parsed["body"]),
        cta_type=parsed.get("cta_type", "calendly"),
        cta_label=str(parsed.get("cta_label", "Schedule a call")),
        tone=tone,
        lang=lang,
        model=model_used,
        raw_prompt=prompt,
    )


def generate_ab_pair(
    *,
    visitor: VisitorContext,
    exhibitor: ExhibitorContext,
    lang: Lang = "ko",
    tone_a: Tone = "formal",
    tone_b: Tone = "balanced",
) -> tuple[FollowupDraft, FollowupDraft]:
    """A/B 두 안 동시 생성 — 전시자가 비교 후 선택."""
    a = generate_draft(visitor=visitor, exhibitor=exhibitor, lang=lang, tone=tone_a)
    b = generate_draft(visitor=visitor, exhibitor=exhibitor, lang=lang, tone=tone_b)
    return a, b


# ---------------------------------------------------------------------------
# LLM wrappers
# ---------------------------------------------------------------------------
def _call_claude(prompt: str) -> tuple[str, str]:
    from anthropic import Anthropic  # type: ignore

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=600,
        system=prompt,
        messages=[
            {"role": "user", "content": "Generate the email now. JSON only."}
        ],
    )
    text = msg.content[0].text  # type: ignore[index]
    return text, "claude"


def _call_gpt(prompt: str) -> tuple[str, str]:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.getenv("GPT_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the email now. JSON only."},
        ],
        max_tokens=600,
    )
    return resp.choices[0].message.content or "", "gpt-4o"


# ---------------------------------------------------------------------------
# Mock — 결정론적, 테스트/dev 용
# ---------------------------------------------------------------------------
def _mock_draft(
    v: VisitorContext, e: ExhibitorContext, lang: Lang, tone: Tone, prompt: str
) -> FollowupDraft:
    first_q = v.copilot_questions[0] if v.copilot_questions else "(질문 없음)"
    if lang == "ko":
        subject = f"[{e.company_name}] 어제 {v.booth_name} 부스 방문 감사드립니다"
        body = (
            f"{v.visitor_name} 님, 어제 {v.dwell_minutes}분간 자리해 주셔서 감사합니다.\n\n"
            f"문의해 주신 \"{first_q}\" 관련해서 자료 정리해 보내드립니다. "
            f"30분 정도 시간 되실 때 일정 잡아 주시면 데모 진행해 드리겠습니다.\n\n"
            f"{e.exhibitor_name} 드림"
        )
        cta_label = "일정 잡기"
    else:
        subject = f"[{e.company_name}] Following up on yesterday's chat at {v.booth_name}"
        body = (
            f"Hi {v.visitor_name},\n\n"
            f"Thank you for spending {v.dwell_minutes} minutes at our booth yesterday. "
            f"On your question — \"{first_q}\" — I've put together the materials. "
            f"Could we schedule a 30-minute demo when you're free?\n\n"
            f"Best,\n{e.exhibitor_name}"
        )
        cta_label = "Schedule a demo"
    return FollowupDraft(
        subject=subject,
        body=body,
        cta_type="calendly" if e.cta_calendly_url else "pdf",
        cta_label=cta_label,
        tone=tone,
        lang=lang,
        model="mock",
        raw_prompt=prompt,
    )
