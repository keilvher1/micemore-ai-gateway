"""부스 자료 PDF → chunk → embed → Pinecone upsert.

운영자가 부스 자료를 업로드하면 (Firebase Storage), Cloud Function 또는
별도 워커가 이 스크립트를 실행해 Pinecone 의 booth-{booth_id} namespace 에
chunk 들을 인덱싱한다.

CLI:
    python -m rag.ingest --booth lumen --pdf /tmp/lumen_catalog.pdf
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Iterable

from rag.embeddings import embed_chunks

log = logging.getLogger(__name__)

CHUNK_CHARS = int(os.getenv("INGEST_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "120"))
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "micemore-booths")


def _split_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """간단한 sliding window — 문단 경계를 우선 존중."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        # 문장 경계로 백오프
        if end < len(text):
            last_period = text.rfind(". ", start + size // 2, end)
            if last_period != -1:
                end = last_period + 1
        out.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def _read_pdf(path: Path) -> Iterable[tuple[int, str]]:
    """페이지별 (page_num, text)."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pypdf required for ingest") from e

    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        yield i, page.extract_text() or ""


async def ingest_pdf(booth_id: str, pdf_path: Path, doc_title: str | None = None) -> int:
    doc_title = doc_title or pdf_path.name

    # 1) 페이지 → chunk
    records: list[dict] = []
    for page_num, page_text in _read_pdf(pdf_path):
        for chunk in _split_text(page_text):
            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "doc_title": doc_title,
                    "page": page_num,
                }
            )

    if not records:
        log.warning("No text extracted from %s", pdf_path)
        return 0

    # 2) embed
    vectors = await embed_chunks([r["text"] for r in records])

    # 3) upsert
    if os.getenv("USE_MOCK", "false").lower() == "true":
        log.info("[MOCK] would upsert %d chunks to namespace booth-%s", len(records), booth_id)
        return len(records)

    from pinecone import Pinecone  # type: ignore

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(PINECONE_INDEX)
    index.upsert(
        namespace=f"booth-{booth_id}",
        vectors=[
            {
                "id": r["id"],
                "values": v,
                "metadata": {
                    "text": r["text"],
                    "doc_title": r["doc_title"],
                    "page": r["page"],
                },
            }
            for r, v in zip(records, vectors)
        ],
    )
    return len(records)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest a PDF into Pinecone")
    parser.add_argument("--booth", required=True, help="booth_id (Pinecone namespace key)")
    parser.add_argument("--pdf", required=True, help="path to PDF file")
    parser.add_argument("--title", help="doc_title metadata override")
    args = parser.parse_args()

    n = asyncio.run(ingest_pdf(args.booth, Path(args.pdf), args.title))
    print(f"Ingested {n} chunks for booth-{args.booth}")


if __name__ == "__main__":
    main()
