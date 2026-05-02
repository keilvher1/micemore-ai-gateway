"""실시간 통역 WebSocket 라우터.

Sprint 1 (현재):
  - WS /translation/session 엔드포인트
  - SessionManager 로 양쪽 폰 broadcast
  - audio.end 시 mock segment + tts 응답
  - JWT 검증은 형식만 (Sprint 2 에서 firebase_admin.auth.verify_id_token)

Sprint 2:
  - audio.chunk → Whisper streaming
  - VAD 800ms 침묵 감지로 audio.end 자동 트리거 (클라 측)
  - segment.partial broadcast

Sprint 3:
  - Claude 번역 + ElevenLabs TTS bytes broadcast
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from translation.mock import build_mock_segment
from translation.schemas import ClientMessage, SessionStart
from translation.session_manager import SessionManager, SessionRole

router = APIRouter(prefix="/translation", tags=["translation"])
log = logging.getLogger("translation")

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

# 단일 인스턴스 — Sprint 2 에서 Redis 백엔드로 교체 시 인터페이스 동일 유지.
manager = SessionManager()


# ---------------------------------------------------------------------------
# 인증 (Sprint 1: 형식만)
# ---------------------------------------------------------------------------
async def _verify_jwt(token: str | None) -> str:
    """Firebase ID Token 검증. Sprint 1 은 형식 체크만."""
    if not token or not token.lower().startswith("bearer "):
        return "anon"
    raw = token.split(" ", 1)[1].strip()
    if not raw:
        return "anon"
    # TODO(sprint-2): firebase_admin.auth.verify_id_token(raw)
    return f"u_{raw[:8]}"


def _peer_role(role: SessionRole) -> SessionRole:
    return SessionRole.VISITOR if role == SessionRole.OPERATOR else SessionRole.OPERATOR


# ---------------------------------------------------------------------------
# WebSocket 핸들러
# ---------------------------------------------------------------------------
@router.websocket("/session")
async def session_ws(ws: WebSocket):
    user_id = await _verify_jwt(ws.headers.get("authorization"))
    await ws.accept()

    session_id: str | None = None
    role: SessionRole | None = None

    try:
        while True:
            try:
                raw = await ws.receive_json()
            except WebSocketDisconnect:
                raise
            try:
                msg = ClientMessage.model_validate(raw)
            except ValidationError as exc:
                await ws.send_json(
                    {"type": "error", "code": "schema", "message": str(exc)}
                )
                continue

            # ------------------------------------------------ session.start
            if msg.type == "session.start":
                try:
                    start = SessionStart.model_validate(msg.payload)
                except ValidationError as exc:
                    await ws.send_json(
                        {"type": "error", "code": "schema",
                         "message": f"session.start: {exc}"}
                    )
                    continue
                session_id = start.session_id
                role = SessionRole(start.role)
                sess = await manager.attach(
                    session_id=session_id,
                    role=role,
                    ws=ws,
                    user_id=user_id,
                    booth_id=start.booth_id,
                    lang_pair=start.lang_pair,
                )
                await manager.broadcast(session_id, {
                    "type": "session.ready",
                    "session_id": session_id,
                    "lang_pair": list(sess.lang_pair),
                    "paired": sess.is_paired(),
                })

            # ------------------------------------------------- audio.chunk
            elif msg.type == "audio.chunk":
                if not (USE_MOCK and session_id and role):
                    continue
                # Sprint 1: chunk 본문 무시. partial 텍스트만 흘려 UI 검증.
                await manager.broadcast(session_id, {
                    "type": "segment.partial",
                    "speaker": role.value,
                    "source_text": "(말씀 중…)",
                    "lang": manager.get(session_id).src_lang_for(role),  # type: ignore[union-attr]
                })

            # --------------------------------------------------- audio.end
            elif msg.type == "audio.end":
                if not (session_id and role):
                    continue
                sess = manager.get(session_id)
                if sess is None:
                    continue
                if USE_MOCK:
                    seg = build_mock_segment(role=role, session=sess)
                    await manager.broadcast(session_id, seg.as_segment_final())
                    # TTS 는 상대방한테만 보냄 (자기 음성 자기가 안 듣게)
                    await manager.send_to(
                        session_id, _peer_role(role), seg.as_tts_audio()
                    )
                else:
                    # Sprint 2: SttPipeline.transcribe(...) → broadcast partial+final
                    # Sprint 3: TranslatePipeline + TtsPipeline
                    await ws.send_json({
                        "type": "error",
                        "code": "not_implemented",
                        "message": "live STT/translate/TTS arrives in Sprint 2~3",
                    })

            # ---------------------------------------------------- ping
            elif msg.type == "ping":
                await ws.send_json({"type": "pong"})

            # -------------------------------------------------- session.end
            elif msg.type == "session.end":
                if session_id:
                    await manager.finalize(session_id)
                break

    except WebSocketDisconnect:
        log.info("WS disconnect | user=%s session=%s", user_id, session_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("WS error")
        try:
            await ws.send_json(
                {"type": "error", "code": "server", "message": str(exc)}
            )
        finally:
            try:
                await ws.close(code=status.WS_1011_INTERNAL_ERROR)
            except Exception:  # noqa: BLE001
                pass
    finally:
        await manager.detach(session_id, role, ws)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@router.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "module": "translation",
        "mock": USE_MOCK,
        "active_sessions": manager.active_count(),
    }
