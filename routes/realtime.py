"""OpenAI Realtime API WebSocket 프록시 — Boomi 캐릭터 음성-음성 대화.

클라이언트(Flutter 앱) ⟷ 이 게이트웨이 ⟷ OpenAI Realtime API

- API 키 노출 없이 클라이언트가 Realtime API 사용 가능
- 양방향 PCM 16kHz 오디오 + JSON 이벤트 패스스루
- 기본 voice = "shimmer" (한국어 발음 자연스러움). 클라이언트가 query param 으로 override 가능
- Boomi persona 시스템 프롬프트 자동 주입

사용:
  ws = WebSocketChannel.connect(
    "wss://api.micemore.com/realtime?lang=ko&voice=shimmer&booth=B-2026-001",
  )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)
router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

BOOMI_PERSONA_KO = """당신은 'Boomi'입니다. MICE 행사 현장의 친근한 AI 가이드 캐릭터예요.

성격:
- 따뜻하고 호기심 많고 약간 장난기 있음
- 말은 짧고 명확하게 (한 번에 1-3문장)
- 사용자 질문에 즉답하고 후속 질문으로 대화 이어가기
- 부스 이름·제품·일정 같은 사실 정보는 정확하게

스타일:
- 자연스러운 한국어 구어체 ("~예요", "~죠")
- 가끔 작은 감탄사 ("오!", "아하", "그렇구나")
- 사용자가 외국어로 말하면 그 언어로 답변

금지:
- 길고 장황한 설명
- 형식적인 인삿말 반복
- 출처 없는 추측"""


def _is_real_key(value: str | None) -> bool:
    if not value:
        return False
    upper = value.upper()
    return not any(t in upper for t in ("PLACEHOLDER", "SK-PLACE"))


@router.websocket("/realtime")
async def realtime_proxy(
    ws: WebSocket,
    lang: str = Query(default="ko"),
    voice: str = Query(default="shimmer"),
    booth: Optional[str] = Query(default=None),
):
    """양방향 WS 프록시 — Flutter ⟷ OpenAI Realtime API."""
    await ws.accept()

    if not _is_real_key(OPENAI_API_KEY):
        await ws.send_json({
            "type": "error",
            "error": {
                "code": "no_api_key",
                "message": "OPENAI_API_KEY 미설정 또는 placeholder. .env 확인 필요.",
            },
        })
        await ws.close(code=1011)
        return

    try:
        import websockets  # type: ignore
    except ImportError:
        await ws.send_json({
            "type": "error",
            "error": {"code": "missing_dep", "message": "pip install websockets"},
        })
        await ws.close(code=1011)
        return

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with websockets.connect(REALTIME_URL, additional_headers=headers) as upstream:
            # 1) 세션 초기화 — persona 주입 + voice 설정
            init = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": BOOMI_PERSONA_KO,
                    "voice": voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 700,
                    },
                    "temperature": 0.8,
                },
            }
            if booth:
                init["session"]["instructions"] += f"\n\n현재 부스 ID: {booth}"
            await upstream.send(json.dumps(init))

            # 2) 양방향 패스스루
            async def client_to_openai():
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            return
                        # 텍스트(JSON 이벤트) 또는 바이너리(PCM 청크)
                        if "text" in msg and msg["text"] is not None:
                            await upstream.send(msg["text"])
                        elif "bytes" in msg and msg["bytes"] is not None:
                            # 클라이언트가 raw PCM 보내면 input_audio_buffer.append 로 래핑
                            import base64
                            audio_b64 = base64.b64encode(msg["bytes"]).decode("ascii")
                            await upstream.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64,
                            }))
                except WebSocketDisconnect:
                    return
                except Exception as e:
                    log.warning("client_to_openai err: %s", e)

            async def openai_to_client():
                try:
                    async for raw in upstream:
                        if isinstance(raw, str):
                            await ws.send_text(raw)
                        else:
                            await ws.send_bytes(raw)
                except Exception as e:
                    log.warning("openai_to_client err: %s", e)

            await asyncio.gather(
                client_to_openai(),
                openai_to_client(),
                return_exceptions=True,
            )

    except Exception as e:
        log.exception("realtime proxy error")
        try:
            await ws.send_json({"type": "error", "error": {"code": "proxy_error", "message": str(e)}})
        except Exception:
            pass
        await ws.close(code=1011)
