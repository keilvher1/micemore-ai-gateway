"""SendGrid 발송 wrapper — A/B 추적 가능한 메타 포함.

USE_MOCK=true 인 경우 실제 발송 없이 SentResult 만 반환 (CI/dev 안전망).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from followup.generator import FollowupDraft

log = logging.getLogger("followup.sender")

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
PLACEHOLDER_TOKENS = ("PLACEHOLDER", "SG.PLACE")


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    upper = value.upper()
    return any(token.upper() in upper for token in PLACEHOLDER_TOKENS)


@dataclass
class SendResult:
    accepted: bool
    provider_id: str | None
    status_code: int
    error: str | None = None


def send(
    *,
    draft: FollowupDraft,
    to_email: str,
    from_email: str | None = None,
    ab_arm: str = "A",        # "A" | "B" — 메트릭 분리용
    event_id: str | None = None,
    visitor_hash: str | None = None,
) -> SendResult:
    """단일 메일 발송.

    SendGrid event tracking 을 위해 custom_args 에 ab_arm/event_id/tone/lang
    삽입 — 나중에 webhook 으로 open/click/reply 수신 시 식별.
    """
    sender = from_email or os.getenv("FROM_EMAIL", "noreply@micemore.com")

    api_key = os.getenv("SENDGRID_API_KEY")
    if USE_MOCK or _is_placeholder(api_key):
        log.info("[mock] send to %s | tone=%s lang=%s arm=%s (placeholder=%s)",
                 to_email, draft.tone, draft.lang, ab_arm, _is_placeholder(api_key))
        return SendResult(accepted=True, provider_id="mock-id", status_code=202)

    try:
        from sendgrid import SendGridAPIClient  # type: ignore
        from sendgrid.helpers.mail import (  # type: ignore
            Mail, From, To, Subject, PlainTextContent, CustomArg,
        )
    except ImportError as exc:
        return SendResult(accepted=False, provider_id=None, status_code=0,
                          error=f"sendgrid not installed: {exc}")

    mail = Mail(
        from_email=From(sender),
        to_emails=To(to_email),
        subject=Subject(draft.subject),
        plain_text_content=PlainTextContent(draft.body),
    )
    custom_args = {
        "ab_arm": ab_arm,
        "tone": draft.tone,
        "lang": draft.lang,
        "model": draft.model,
    }
    if event_id:
        custom_args["event_id"] = event_id
    if visitor_hash:
        custom_args["visitor_hash"] = visitor_hash
    for k, v in custom_args.items():
        mail.add_custom_arg(CustomArg(k, v))

    # api_key 가 위에서 placeholder/USE_MOCK 면 이미 mock 반환됨.
    client = SendGridAPIClient(api_key)
    try:
        resp = client.send(mail)
        msg_id = resp.headers.get("X-Message-Id") if resp.headers else None
        return SendResult(
            accepted=200 <= resp.status_code < 300,
            provider_id=msg_id,
            status_code=resp.status_code,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("sendgrid send failed")
        return SendResult(accepted=False, provider_id=None, status_code=0,
                          error=str(exc))
