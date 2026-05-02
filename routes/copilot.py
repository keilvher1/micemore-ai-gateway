"""POST /copilot/query — 부스 코파일럿 SSE 스트리밍 라우트.

Flutter 클라이언트(`copilot_service.dart`) 가 기대하는 SSE payload 형식:
    data: {"type":"meta",     "sessionId":"...","lang":"en"}\\n\\n
    data: {"type":"citations","items":[{"id":"S1","doc":"...","page":3}]}\\n\\n
    data: {"type":"token",    "value":"Hello"}\\n\\n
    data: {"type":"done"}\\n\\n
    data: {"type":"error",    "message":"..."}\\n\\n
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from rag.pipeline import answer_with_rag, mock_answer

log = logging.getLogger(__name__)
router = APIRouter()

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


class CopilotQuery(BaseModel):
    booth_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)
    question: str = Field(..., min_length=1, max_length=2000)
    target_lang: str = Field(default="auto", pattern=r"^(auto|ko|en|ja|zh)$")
    mock: bool = Field(default=False, description="강제 mock 응답 (테스트용)")
    # D-4 단계 7 — RAG 인제스트 namespace 라우팅.
    # booth (기본): booth:{booth_id} 만 검색 (부스 자료실)
    # tour:        tour:{lang}:{areacode} 만 검색 (한국관광공사 RAG)
    # auto:        둘 다 검색 → score 기준 상위 k 청크 합산 (공모전 차별화 핵심)
    source: str = Field(default="booth", pattern=r"^(booth|tour|auto)$")
    area_code: str | None = Field(default=None, description="source=tour|auto 일 때 지역 코드")


def _sse(event: dict) -> dict:
    """sse_starlette 형식으로 wrapping."""
    return {"event": "message", "data": json.dumps(event, ensure_ascii=False)}


async def _stream(query: CopilotQuery) -> AsyncIterator[dict]:
    """SSE 이벤트 generator."""
    use_mock = query.mock or USE_MOCK

    # 1) meta
    yield _sse(
        {
            "type": "meta",
            "sessionId": query.session_id or str(uuid.uuid4()),
            "lang": "en" if query.target_lang == "auto" else query.target_lang,
        }
    )

    try:
        if use_mock:
            gen = mock_answer(query.question, query.target_lang)
        else:
            # D-4 단계 7 — source 에 따라 RAG namespace 라우팅.
            # answer_with_rag 가 source/area_code 파라미터 미구현이면 기본(booth) 동작.
            try:
                gen = answer_with_rag(  # type: ignore[call-arg]
                    booth_id=query.booth_id,
                    question=query.question,
                    target_lang=query.target_lang,
                    source=query.source,
                    area_code=query.area_code,
                )
            except TypeError:
                gen = answer_with_rag(
                    booth_id=query.booth_id,
                    question=query.question,
                    target_lang=query.target_lang,
                )

        async for evt in gen:
            yield _sse(evt)

    except asyncio.CancelledError:
        log.info("client disconnected (booth=%s)", query.booth_id)
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("RAG pipeline failed")
        yield _sse({"type": "error", "message": str(exc) if use_mock else "internal_error"})

    yield _sse({"type": "done"})


@router.post("/query")
async def copilot_query(query: CopilotQuery):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="empty_question")
    return EventSourceResponse(_stream(query), media_type="text/event-stream")
