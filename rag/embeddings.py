"""text-embedding-3-small wrapper.

OpenAI 임베딩 1회당 ~1536 차원, 8K 토큰까지. Pinecone index 도 동일한 1536d.

`embed_query()`: 사용자 질문 1건 → 단일 벡터
`embed_chunks()`: 부스 자료 chunk 다수 → batch embedding (최대 100개씩)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Sequence

log = logging.getLogger(__name__)

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536
BATCH = 96


async def _client():
    import openai  # type: ignore

    return openai.AsyncOpenAI()


async def embed_query(text: str) -> list[float]:
    if not text.strip():
        return [0.0] * EMBED_DIM

    if os.getenv("USE_MOCK", "false").lower() == "true":
        # 결정적 mock 임베딩 — 길이만 맞춤
        h = sum(ord(c) for c in text)
        return [(h + i) % 100 / 100.0 for i in range(EMBED_DIM)]

    client = await _client()
    resp = await client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return resp.data[0].embedding


async def embed_chunks(texts: Sequence[str]) -> list[list[float]]:
    """대량 임베딩 — Pinecone upsert 직전 단계."""
    if os.getenv("USE_MOCK", "false").lower() == "true":
        return [await embed_query(t) for t in texts]

    client = await _client()
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = list(texts[i : i + BATCH])
        resp = await client.embeddings.create(model=OPENAI_EMBED_MODEL, input=chunk)
        out.extend(d.embedding for d in resp.data)
        # gentle backoff
        await asyncio.sleep(0.05)
    return out
