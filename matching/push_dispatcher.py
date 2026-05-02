"""FCM/APNs push 디스패처 — 한도 enforce + audit log.

USE_MOCK=true: 발송 안 함, 결과 객체만 반환.
실 모드: firebase_admin.messaging.send (lazy import).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from matching.governance import (
    PushBudget,
    can_push,
    record_push,
)
from matching.matcher import MatchEvent

log = logging.getLogger("matching.push")

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


@dataclass
class PushResult:
    accepted: bool
    provider_id: str | None
    reason: str | None = None
    audience: Literal["booth", "visitor"] = "visitor"


def _build_payload_visitor(ev: MatchEvent) -> dict:
    return {
        "title": "관심사 일치 부스 발견",
        "body": f"{ev.reason} · 매칭 {int(ev.score * 100)}%",
        "data": {
            "booth_id": ev.booth_id,
            "score": str(ev.score),
            "kind": "match.visitor",
        },
    }


def _build_payload_booth(ev: MatchEvent) -> dict:
    return {
        "title": "타겟 ICP 매칭",
        "body": f"매칭 {int(ev.score * 100)}% · 거리 {int(ev.distance_m)}m",
        "data": {
            "visitor_hash": ev.visitor_hash,
            "score": str(ev.score),
            "kind": "match.booth",
        },
    }


def dispatch(
    *,
    event: MatchEvent,
    budget: PushBudget,
    visitor_token: str | None = None,
    booth_operator_token: str | None = None,
) -> tuple[PushResult, PushResult]:
    """양쪽 push 시도. 한도 초과 또는 토큰 없음 → accepted=False."""
    ok, reason = can_push(
        booth_id=event.booth_id,
        visitor_hash=event.visitor_hash,
        budget=budget,
    )
    if not ok:
        return (
            PushResult(False, None, reason, audience="visitor"),
            PushResult(False, None, reason, audience="booth"),
        )

    visitor_res = _send(
        token=visitor_token,
        payload=_build_payload_visitor(event),
        audience="visitor",
    )
    booth_res = _send(
        token=booth_operator_token,
        payload=_build_payload_booth(event),
        audience="booth",
    )
    if visitor_res.accepted or booth_res.accepted:
        record_push(
            booth_id=event.booth_id,
            visitor_hash=event.visitor_hash,
            budget=budget,
        )
    return visitor_res, booth_res


def _send(
    *,
    token: str | None,
    payload: dict,
    audience: Literal["booth", "visitor"],
) -> PushResult:
    if not token:
        return PushResult(False, None, "no_token", audience=audience)
    if USE_MOCK:
        log.info("[mock] push %s | %s", audience, payload["title"])
        return PushResult(True, "mock-id", audience=audience)
    try:
        from firebase_admin import messaging  # type: ignore
    except ImportError as exc:
        return PushResult(False, None, f"firebase-admin missing: {exc}",
                          audience=audience)
    msg = messaging.Message(
        token=token,
        notification=messaging.Notification(
            title=payload["title"], body=payload["body"]
        ),
        data=payload.get("data", {}),
    )
    try:
        msg_id = messaging.send(msg)
        return PushResult(True, msg_id, audience=audience)
    except Exception as exc:  # noqa: BLE001
        log.exception("fcm send failed")
        return PushResult(False, None, str(exc), audience=audience)
