"""MICEMORE AI Gateway — FastAPI entrypoint.

부스 코파일럿(RAG), 라이브 통역, 리드 스코어링 등 LLM 관련 모든 호출을
이 게이트웨이로 모아 인증·rate limit·로깅·비용 추적을 일원화한다.

Phase 1 에서는 `/copilot/query` (SSE 스트리밍) 만 노출.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# .env 자동 로드 — uvicorn / lambda 진입 시 환경변수 주입.
# OS 환경변수가 이미 있으면 덮어쓰지 않음 (override=False).
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes.copilot import router as copilot_router
from routes.leads import router as leads_router
from routes.translation import router as translation_router
from routes.ml_leads import router as ml_leads_router
from routes.followup import router as followup_router
from routes.insights import router as insights_router
from routes.mypass import router as mypass_router
from routes.personas import router as personas_router
from routes.matching import router as matching_router
from routes.trends import router as trends_router
from routes.routing import router as routing_router
from routes.nfc import router as nfc_router
from routes.namecard import router as namecard_router
from routes.materials import router as materials_router
from routes.materials import upload_router as materials_upload_router
from routes.visitors import router as visitors_router
from routes.match import router as match_router
from routes.tourapi import router as tourapi_router
from routes.voice import router as voice_router
from routes.realtime import router as realtime_router

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ai-gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AI Gateway starting (mock=%s, log=%s)", USE_MOCK, LOG_LEVEL)
    # Sentry 초기화는 실 환경에서만
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn and not USE_MOCK:
        try:
            import sentry_sdk  # type: ignore

            sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=0.1)
            log.info("Sentry enabled")
        except ImportError:
            log.warning("sentry_sdk not installed; skipping")
    yield
    log.info("AI Gateway shutting down")


app = FastAPI(
    title="MICEMORE AI Gateway",
    version="0.1.0",
    description="LLM gateway for booth copilot, live translation, lead scoring.",
    lifespan=lifespan,
)

# CORS — Flutter 웹 + 모바일 모두 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "https://micemore-participant.web.app,https://micemore-organizer.web.app,http://localhost:*",
    ).split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """간단한 access log — 비용 추적/디버깅용."""
    log.info("%s %s", request.method, request.url.path)
    response = await call_next(request)
    log.info("→ %s %s [%s]", request.method, request.url.path, response.status_code)
    return response


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc) if USE_MOCK else None},
    )


@app.get("/health")
def health():
    return {"status": "ok", "mock": USE_MOCK}


app.include_router(copilot_router, prefix="/copilot", tags=["copilot"])
app.include_router(leads_router)         # prefix /leads in router
app.include_router(translation_router)   # prefix /translation in router (WS)
# Phase 3
app.include_router(ml_leads_router)      # /ml-leads
app.include_router(followup_router)      # /followup
app.include_router(insights_router)      # /insights
app.include_router(mypass_router)        # /mypass
# Phase 4
app.include_router(personas_router)      # /personas
app.include_router(matching_router)      # /matching
app.include_router(trends_router)        # /trends (X-API-Tier middleware)
app.include_router(routing_router)       # /routing
# Phase 5 — NFC + Digital namecard + Materials RAG + Operator analytics + Smart matching
app.include_router(nfc_router)           # /nfc
app.include_router(namecard_router)      # /me/namecard
app.include_router(materials_router)     # /booth/materials
app.include_router(materials_upload_router)  # /materials/upload + /materials/uploads/{id}
app.include_router(visitors_router)      # /operator
app.include_router(match_router)         # /me/matches
app.include_router(tourapi_router)       # /tourapi (한국관광공사 OpenAPI — 공모전 필수)
# D-4 단계 3 — Whisper STT + ElevenLabs TTS (4 voice 매핑). PLACEHOLDER 시 mock 폴백.
app.include_router(voice_router, prefix="/voice", tags=["voice"])
# Realtime API WebSocket 프록시 — Boomi 음성-음성 대화
app.include_router(realtime_router, tags=["realtime"])


# Lambda 배포 시 사용하는 ASGI → Lambda 어댑터.
# 로컬 개발에서는 `uvicorn main:app --reload` 로 실행한다.
try:
    from mangum import Mangum  # type: ignore

    handler = Mangum(app)
except ImportError:  # pragma: no cover
    handler = None
