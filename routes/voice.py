"""POST /voice/stt + POST /voice/tts — 음성 입출력 라우트.

이 모듈은 .env 의 OPENAI_API_KEY (Whisper) 와 ELEVENLABS_API_KEY 가 실값이면
자동 활성화됩니다. PLACEHOLDER 면 mock fixture 를 반환 (텍스트 + 빈 mp3 1프레임).

USE_MOCK=true 면 키 검사를 건너뛰고 즉시 mock 반환.

Voice ID 매핑 (ElevenLabs):
    KO → ELEVENLABS_VOICE_ID_KO
    EN → ELEVENLABS_VOICE_ID_EN
    JA → ELEVENLABS_VOICE_ID_JA
    ZH → ELEVENLABS_VOICE_ID_ZH
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter()

USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
PLACEHOLDER_TOKENS = ("PLACEHOLDER", "sk-PLACE", "el-PLACE")


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    upper = value.upper()
    return any(token.upper() in upper for token in PLACEHOLDER_TOKENS)


def _voice_id(lang: str) -> str | None:
    key = f"ELEVENLABS_VOICE_ID_{lang.upper()}"
    val = os.getenv(key)
    if _is_placeholder(val):
        return None
    return val


# ── STT (Whisper) ─────────────────────────────────────────────
@router.post("/stt")
async def stt(
    audio: UploadFile = File(...),
    lang: str = Form("ko"),
):
    """음성 → 텍스트. mock 모드 또는 PLACEHOLDER 면 결정론 mock 반환."""
    audio_bytes = await audio.read()

    if USE_MOCK or _is_placeholder(os.getenv("OPENAI_API_KEY")):
        return {
            "text": _mock_transcript(lang),
            "lang": lang,
            "duration_ms": len(audio_bytes) // 32,  # rough estimate
            "source": "mock",
        }

    try:
        import openai  # type: ignore

        client = openai.AsyncOpenAI()
        result = await client.audio.transcriptions.create(
            model=os.getenv("WHISPER_MODEL", "whisper-1"),
            file=(audio.filename or "speech.wav", audio_bytes, audio.content_type or "audio/wav"),
            language=lang,
        )
        return {"text": result.text, "lang": lang, "source": "openai"}
    except Exception as exc:  # noqa: BLE001
        log.warning("STT failed; mock fallback: %s", exc)
        return {"text": _mock_transcript(lang), "lang": lang, "source": "mock-fallback"}


# ── TTS (ElevenLabs streaming) ────────────────────────────────
class TtsBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    lang: str = Field(default="ko", pattern=r"^(ko|en|ja|zh)$")


@router.post("/tts")
async def tts(body: TtsBody):
    """텍스트 → mp3 streaming. PLACEHOLDER 또는 USE_MOCK 면 빈 mp3 1프레임 반환."""
    voice_id = _voice_id(body.lang)
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if USE_MOCK or voice_id is None or _is_placeholder(api_key):
        async def empty() -> AsyncIterator[bytes]:
            # ID3v2 헤더만 — 클라이언트가 mp3 디코더 init 만 해도 무사 종료.
            yield b"ID3\x04\x00\x00\x00\x00\x00\x00"
        return StreamingResponse(empty(), media_type="audio/mpeg")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": body.text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    async def proxy() -> AsyncIterator[bytes]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as r:
                    if r.status_code >= 400:
                        log.warning("ElevenLabs %s %s", r.status_code, await r.aread())
                        yield b"ID3\x04\x00\x00\x00\x00\x00\x00"
                        return
                    async for chunk in r.aiter_bytes():
                        yield chunk
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS stream failed; mock fallback: %s", exc)
            yield b"ID3\x04\x00\x00\x00\x00\x00\x00"

    return StreamingResponse(proxy(), media_type="audio/mpeg")


# ── helpers ────────────────────────────────────────────────────
def _mock_transcript(lang: str) -> str:
    return {
        "en": "Hello, can you tell me more about your booth?",
        "ja": "こんにちは、ブースについて教えてください.",
        "zh": "你好，可以介绍一下你们的展位吗？",
    }.get(lang, "안녕하세요, 이 부스의 주요 제품을 알려주세요.")
