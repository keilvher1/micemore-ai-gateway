"""Boomi 세션 메모리 — Firestore `chat_sessions/{user_id}_{booth_id}` 문서 단위.

Boomi 가 사용자의 이름, 관심사, 최근 발화를 기억해서 "처음 만난 챗봇"이 아닌
"두 번째 만나면 알아보는 캐릭터" 처럼 동작하게 합니다.

스키마:
    chat_sessions/{session_key}  (session_key = sha1(user_id + booth_id) 첫 16자)
      user_id: str
      booth_id: str
      user_profile: {
        name?: str,
        interests?: [str],
        visited_booths?: [str],
        language?: str,
      }
      messages: [
        {role: 'user'|'assistant', content: str, ts: timestamp}
      ]   (마지막 12 turns 만 유지 — 잘리면 가장 오래된 것 drop)
      created_at: ts
      updated_at: ts

firebase-admin 없거나 USE_MOCK=true 면 in-memory dict 폴백 (시연 안전).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
HISTORY_KEEP = 12  # 최근 N turns 유지

# in-memory fallback (USE_MOCK 또는 firebase-admin 부재 시)
_MEMORY_STORE: dict[str, dict] = {}


def _session_key(user_id: str, booth_id: str) -> str:
    raw = f"{user_id}|{booth_id}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _firestore():
    """firebase-admin client. 미설치/미초기화 시 None."""
    try:
        from firebase_admin import firestore, initialize_app  # type: ignore

        try:
            initialize_app()
        except ValueError:
            pass  # already initialized
        return firestore.client()
    except Exception:  # noqa: BLE001
        return None


async def load_session(user_id: str, booth_id: str) -> dict:
    """이 사용자×부스 세션의 user_profile + 최근 messages 반환.
    없으면 빈 dict 반환 (Boomi 가 첫 만남으로 인사).
    """
    if not user_id:
        return {}

    key = _session_key(user_id, booth_id)
    if USE_MOCK or _firestore() is None:
        return _MEMORY_STORE.get(key, {})

    try:
        doc = _firestore().collection("chat_sessions").document(key).get()
        if not doc.exists:
            return {}
        return doc.to_dict() or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("chat_memory.load_session failed for %s: %s", key, exc)
        return _MEMORY_STORE.get(key, {})


async def append_turn(
    user_id: str,
    booth_id: str,
    role: str,
    content: str,
    user_profile: dict | None = None,
) -> None:
    """대화 1 turn 을 chat_sessions 문서에 누적. 최근 HISTORY_KEEP 만 유지."""
    if not user_id or not content:
        return

    key = _session_key(user_id, booth_id)
    now = time.time()
    new_msg = {"role": role, "content": content[:2000], "ts": now}

    # in-memory 갱신 (양쪽 동기 — Firestore 실패 시도 fallback 일관)
    existing = _MEMORY_STORE.get(key, {})
    msgs = existing.get("messages", [])
    msgs.append(new_msg)
    if len(msgs) > HISTORY_KEEP:
        msgs = msgs[-HISTORY_KEEP:]
    existing.update({
        "user_id": user_id,
        "booth_id": booth_id,
        "messages": msgs,
        "updated_at": now,
    })
    if user_profile:
        existing["user_profile"] = {
            **(existing.get("user_profile") or {}),
            **user_profile,
        }
    if "created_at" not in existing:
        existing["created_at"] = now
    _MEMORY_STORE[key] = existing

    if USE_MOCK or _firestore() is None:
        return

    try:
        ref = _firestore().collection("chat_sessions").document(key)
        snap = ref.get()
        if snap.exists:
            data = snap.to_dict() or {}
            old = data.get("messages", [])
            old.append(new_msg)
            if len(old) > HISTORY_KEEP:
                old = old[-HISTORY_KEEP:]
            update = {"messages": old, "updated_at": now}
            if user_profile:
                update["user_profile"] = {
                    **(data.get("user_profile") or {}),
                    **user_profile,
                }
            ref.update(update)
        else:
            doc = {
                "user_id": user_id,
                "booth_id": booth_id,
                "messages": [new_msg],
                "user_profile": user_profile or {},
                "created_at": now,
                "updated_at": now,
            }
            ref.set(doc)
    except Exception as exc:  # noqa: BLE001
        log.warning("chat_memory.append_turn failed for %s: %s", key, exc)


def history_for_prompt(session: dict) -> list[dict]:
    """build_system_prompt 의 history 인자에 맞는 형태로 변환."""
    msgs = (session or {}).get("messages") or []
    return [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in msgs
        if m.get("content")
    ]


def user_context_for_prompt(session: dict, fallback_lang: str = "ko") -> dict:
    """build_system_prompt 의 user_context 인자에 맞는 형태로 변환."""
    profile = (session or {}).get("user_profile") or {}
    out: dict[str, Any] = {}
    for k in ("name", "interests", "visited_booths"):
        if v := profile.get(k):
            out[k] = v
    out["language"] = profile.get("language") or fallback_lang
    return out
