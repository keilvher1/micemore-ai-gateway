"""LangChain 기반 RAG 파이프라인.

흐름:
1) Pinecone 에서 booth_id namespace 로 top-k chunk 검색
2) chunk 들을 source citation 으로 변환 → SSE citations 이벤트 1회 전송
3) Claude messages.create(stream=True) 로 토큰 스트리밍
4) 신뢰도(top score) < 0.55 면 운영자 직접 문의 폴백 메시지

`answer_with_rag()` 는 async generator — `{"type": ..., ...}` dict 를 yield.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator

from prompts.copilot_system import build_system_prompt
from rag.embeddings import embed_query

log = logging.getLogger(__name__)

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "micemore-booths")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CONFIDENCE_FLOOR = float(os.getenv("RAG_CONFIDENCE_FLOOR", "0.55"))


PINECONE_TOUR_INDEX = os.getenv("PINECONE_TOUR_INDEX", "micemore-tour")


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    return "PLACEHOLDER" in value.upper()


async def _query_namespace(namespace: str, qvec: list[float], index_name: str) -> list[dict]:
    """단일 Pinecone namespace 검색."""
    try:
        from pinecone import Pinecone  # type: ignore
    except ImportError:  # pragma: no cover
        log.warning("pinecone-client not installed; returning empty")
        return []

    if _is_placeholder(os.getenv("PINECONE_API_KEY")):
        log.info("PINECONE_API_KEY is placeholder; returning empty for ns=%s", namespace)
        return []

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    res = index.query(
        vector=qvec,
        top_k=TOP_K,
        namespace=namespace,
        include_metadata=True,
    )
    matches = res.get("matches") or []
    return [
        {
            "id": m["id"],
            "score": m["score"],
            "doc": m["metadata"].get("doc_title", m["metadata"].get("title", "")),
            "page": m["metadata"].get("page", 0),
            "text": m["metadata"].get("text", ""),
            "namespace": namespace,
        }
        for m in matches
    ]


async def _retrieve(
    booth_id: str,
    question: str,
    *,
    source: str = "booth",
    area_code: str | None = None,
    lang: str = "ko",
) -> list[dict]:
    """Pinecone retrieval. source = booth|tour|auto.

    - booth: booth-{booth_id} 만 검색 (부스 자료실).
    - tour:  tour:{lang}:{areacode} 만 검색 (한국관광공사 RAG).
    - auto:  둘 다 검색 후 score 기준 상위 TOP_K 반환 (공모전 차별화 핵심).
    """
    qvec = await embed_query(question)
    booth_ns = f"booth-{booth_id}"
    tour_ns = f"tour:{lang}:{area_code or '00'}"

    booth_chunks: list[dict] = []
    tour_chunks: list[dict] = []
    if source in ("booth", "auto"):
        booth_chunks = await _query_namespace(booth_ns, qvec, PINECONE_INDEX)
    if source in ("tour", "auto"):
        tour_chunks = await _query_namespace(tour_ns, qvec, PINECONE_TOUR_INDEX)

    combined = booth_chunks + tour_chunks
    combined.sort(key=lambda c: c["score"], reverse=True)
    return combined[:TOP_K]


async def answer_with_rag(
    *,
    booth_id: str,
    question: str,
    target_lang: str,
    source: str = "booth",
    area_code: str | None = None,
) -> AsyncIterator[dict]:
    """실 LLM 기반 RAG. SSE 이벤트 dict 를 yield.

    source / area_code 는 D-4 단계 7 — RAG 인제스트 namespace 라우팅.
    placeholder 키 감지 시 자동으로 빈 청크 → 신뢰도 폴백 메시지 (mock 모드와 동일 UX).
    """
    chunks = await _retrieve(
        booth_id,
        question,
        source=source,
        area_code=area_code,
        lang=target_lang if target_lang != "auto" else "ko",
    )

    if not chunks or chunks[0]["score"] < CONFIDENCE_FLOOR:
        # 신뢰도 폴백
        yield {
            "type": "token",
            "value": (
                "이 질문에 대한 자료가 없어요. 운영자에게 직접 문의를 권장합니다."
                if target_lang == "ko"
                else "I don't have material to answer that. Please ask the booth staff directly."
            ),
        }
        return

    # citations 이벤트 1회
    citations = [
        {"id": f"S{i + 1}", "doc": c["doc"], "page": c["page"]}
        for i, c in enumerate(chunks[:3])
    ]
    yield {"type": "citations", "items": citations}

    # Claude 호출
    try:
        import anthropic  # type: ignore
    except ImportError:  # pragma: no cover
        yield {"type": "error", "message": "anthropic_sdk_missing"}
        return

    client = anthropic.AsyncAnthropic()
    system = build_system_prompt(
        booth_id=booth_id,
        target_lang=target_lang,
        sources=chunks,
    )

    async with client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": question}],
    ) as stream:
        async for text in stream.text_stream:
            yield {"type": "token", "value": text}


async def mock_answer(question: str, target_lang: str) -> AsyncIterator[dict]:
    """USE_MOCK=true 일 때 사용 — 고정 응답을 토큰 단위로 흘려보낸다."""
    yield {
        "type": "citations",
        "items": [
            {"id": "S1", "doc": "Lumen Labs Product Catalog 2026.pdf", "page": 12},
            {"id": "S2", "doc": "3D Scanning Demo Script.md", "page": 1},
        ],
    }
    if target_lang == "ko":
        text = (
            "[MOCK] Lumen Labs는 광학 기반 실시간 3D 스캐닝으로 제조 현장의 "
            "품질 검사를 자동화합니다. 기존 머신비전 대비 12배 빠른 결함 감지가 강점입니다. [S1][S2]"
        )
    else:
        text = (
            "[MOCK] Lumen Labs makes a real-time 3D inspection sensor that detects "
            "defects 12x faster than traditional machine vision. Demos run every 30 minutes. [S1][S2]"
        )
    for tok in text.split(" "):
        await asyncio.sleep(0.04)
        yield {"type": "token", "value": tok + " "}
