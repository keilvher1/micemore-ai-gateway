"""WebSocket 메시지 스키마 — Pydantic.

Sprint 1 은 mock 모드라 audio.chunk 의 data 는 사용하지 않지만, Sprint 2 에서
바로 STT 로 보낼 수 있도록 스키마는 미리 확정.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 클라이언트 → 서버
# ---------------------------------------------------------------------------
class SessionStart(BaseModel):
    session_id: str
    booth_id: str
    lang_pair: list[str] = Field(min_length=2, max_length=2)
    role: Literal["operator", "visitor"]


class AudioChunk(BaseModel):
    data: str  # base64-encoded opus/wav
    format: Literal["opus", "wav"] = "opus"
    seq: int = 0


class ClientMessage(BaseModel):
    type: Literal[
        "session.start", "audio.chunk", "audio.end", "session.end", "ping"
    ]
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 서버 → 클라이언트 (참고용 — 실제 송신은 dict 직접 만들어 전송)
# ---------------------------------------------------------------------------
class SegmentFinal(BaseModel):
    type: Literal["segment.final"] = "segment.final"
    segment_id: str
    speaker: Literal["operator", "visitor"]
    source_lang: str
    source_text: str
    target_lang: str
    target_text: str | None = None
    confidence: float = 1.0
    ts: int


class TtsAudio(BaseModel):
    type: Literal["tts.audio"] = "tts.audio"
    segment_id: str
    audio_url: str | None = None
    audio_b64: str | None = None
    duration_ms: int = 0
    voice_id: str = ""


class ServerError(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
