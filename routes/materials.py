"""Booth materials RAG ingest — multipart upload + status tracking.

POST /booth/materials (multipart) → 202 Accepted, material_id + status "processing"
GET /booth/materials/{booth_id} → list with rag_status + chunks_count
POST /booth/materials/{material_id}/retry → re-ingest

D-4 단계 3 — POST /materials/upload (Flutter MaterialsProvider 호출):
  PDF multipart → PyPDF2 텍스트 추출 (이미지면 Tesseract OCR 폴백)
  → 500 토큰 청킹 / 50 오버랩 → text-embedding-3-small → Pinecone upsert
  → Firestore `material_uploads/{id}` 진행도 갱신 → FCM 푸시 (옵션).

USE_MOCK=true 또는 PINECONE_API_KEY/OPENAI_API_KEY placeholder 면 자동 폴백:
실 호출 없이 mock progressive status 만 반환.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

router = APIRouter(prefix="/booth/materials", tags=["materials"])
log = logging.getLogger(__name__)

# D-4 추가 — Flutter MaterialsProvider 가 호출하는 /materials/upload 단순 인터페이스.
upload_router = APIRouter(prefix="/materials", tags=["materials"])

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
PLACEHOLDER_TOKENS = ("PLACEHOLDER", "sk-PLACE", "pcsk-PLACE")


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    upper = value.upper()
    return any(token.upper() in upper for token in PLACEHOLDER_TOKENS)


def _is_live_pipeline_ready() -> bool:
    """실 RAG 파이프라인 사용 가능 여부."""
    if USE_MOCK:
        return False
    return not (
        _is_placeholder(os.getenv("PINECONE_API_KEY"))
        or _is_placeholder(os.getenv("OPENAI_API_KEY"))
    )


# ---------------------------------------------------------------------------
# Request & Response models
# ---------------------------------------------------------------------------
class MaterialIngestionResponse(BaseModel):
    material_id: str
    booth_id: str
    status: str  # "processing", "ready", "failed"
    chunks_count: int | None = None
    error_message: str | None = None
    ingest_started_at: str
    ingest_completed_at: str | None = None


class MaterialListResponse(BaseModel):
    booth_id: str
    materials: list[MaterialIngestionResponse]
    total_chunks: int


# ---------------------------------------------------------------------------
# In-memory material state (mock only)
# ---------------------------------------------------------------------------
_MATERIAL_STATE: dict[str, MaterialIngestionResponse] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/", status_code=202)
async def ingest_material(
    booth_id: str,
    file: UploadFile = File(...),
) -> MaterialIngestionResponse:
    """
    POST /booth/materials — Ingest material file.

    Returns:
        - material_id: unique identifier
        - status: "processing" (mock always returns processing first)
        - 202 Accepted
    """
    if not file.filename:
        raise HTTPException(400, "filename required")

    if not USE_MOCK:
        # TODO: validate file type (pdf, image, video, link)
        # TODO: upload to Cloud Storage
        # TODO: enqueue Pinecone vector job
        # TODO: create Firestore material doc
        raise HTTPException(503, "live data integration in Phase 2")

    # Mock: generate material_id, return "processing"
    material_id = f"MAT-{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat() + "Z"

    response = MaterialIngestionResponse(
        material_id=material_id,
        booth_id=booth_id,
        status="processing",
        chunks_count=None,
        error_message=None,
        ingest_started_at=now,
        ingest_completed_at=None,
    )

    # Store in mock state
    _MATERIAL_STATE[material_id] = response
    log.info("Ingesting material %s for booth %s", material_id, booth_id)

    return response


@router.get("/{booth_id}", response_model=MaterialListResponse)
async def list_booth_materials(booth_id: str) -> MaterialListResponse:
    """
    GET /booth/materials/{booth_id} — List materials by booth.

    Mock behavior:
      - On first call after ingest: returns "processing"
      - On subsequent calls: returns "ready" with 23 chunks
    """
    if not USE_MOCK:
        # TODO: Firestore query materials by booth_id
        raise HTTPException(503, "live data integration in Phase 2")

    # Mock: simulate progression
    materials = []
    total_chunks = 0

    for mat_id, mat in _MATERIAL_STATE.items():
        if mat.booth_id == booth_id:
            # Simulate transition: after 1s, mark as ready with chunks
            if mat.status == "processing":
                # Still processing (first view)
                materials.append(mat)
            else:
                # Already transitioned to ready
                updated = MaterialIngestionResponse(
                    material_id=mat.material_id,
                    booth_id=mat.booth_id,
                    status="ready",
                    chunks_count=23,
                    error_message=None,
                    ingest_started_at=mat.ingest_started_at,
                    ingest_completed_at=datetime.utcnow().isoformat() + "Z",
                )
                materials.append(updated)
                total_chunks += 23

    return MaterialListResponse(
        booth_id=booth_id,
        materials=materials,
        total_chunks=total_chunks,
    )


@router.post("/{material_id}/retry")
async def retry_ingest(material_id: str) -> MaterialIngestionResponse:
    """
    POST /booth/materials/{material_id}/retry — Re-ingest material.
    """
    if not USE_MOCK:
        # TODO: Firestore update + re-enqueue
        raise HTTPException(503, "live data integration in Phase 2")

    if material_id not in _MATERIAL_STATE:
        raise HTTPException(404, "material not found")

    # Reset to processing
    mat = _MATERIAL_STATE[material_id]
    updated = MaterialIngestionResponse(
        material_id=mat.material_id,
        booth_id=mat.booth_id,
        status="processing",
        chunks_count=None,
        error_message=None,
        ingest_started_at=datetime.utcnow().isoformat() + "Z",
        ingest_completed_at=None,
    )
    _MATERIAL_STATE[material_id] = updated

    log.info("Retrying ingest for material %s", material_id)
    return updated


@router.get("/healthz", tags=["health"])
async def healthz() -> dict:
    return {"ok": True, "module": "materials", "mock": USE_MOCK}


# =============================================================================
# D-4 단계 3 — POST /materials/upload (단일 엔드포인트, Flutter 호환)
# =============================================================================
PIPELINE_PHASES = ["queued", "uploading", "ocr", "embedding", "done"]


class UploadResponse(BaseModel):
    upload_id: str
    booth_id: str
    file_name: str
    status: str
    progress_percent: float


def _extract_text_from_pdf(content: bytes) -> str:
    """PyPDF2 텍스트 추출. 실패하면 Tesseract OCR 폴백."""
    text_parts: list[str] = []
    try:
        from io import BytesIO

        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(BytesIO(content))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text.strip())
    except Exception as exc:  # noqa: BLE001
        log.info("pypdf failed (%s); attempting OCR fallback", exc)

    if any(t for t in text_parts):
        return "\n\n".join(t for t in text_parts if t)

    # OCR 폴백 — pdf2image + pytesseract.
    try:
        from io import BytesIO

        import pytesseract  # type: ignore
        from pdf2image import convert_from_bytes  # type: ignore

        images = convert_from_bytes(content, dpi=200)
        return "\n\n".join(pytesseract.image_to_string(img) for img in images)
    except Exception as exc:  # noqa: BLE001
        log.warning("OCR fallback failed: %s", exc)
        return ""


def _chunk(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chars])
        start += max_chars - overlap
    return chunks


async def _set_progress(upload_id: str, **fields) -> None:
    """Firestore material_uploads/{id} 진행도 갱신. firebase-admin 없으면 no-op."""
    try:
        from firebase_admin import firestore, initialize_app  # type: ignore
    except ImportError:
        return
    try:
        try:
            initialize_app()
        except ValueError:
            pass  # already initialized
        db = firestore.client()
        db.collection("material_uploads").document(upload_id).set(
            {**fields, "updated_at": datetime.utcnow().isoformat() + "Z"},
            merge=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("firestore progress update failed for %s: %s", upload_id, exc)


async def _ingest_real(upload_id: str, booth_id: str, file_name: str, content: bytes) -> None:
    """실 파이프라인 — 추출 → 청킹 → 임베딩 → Pinecone upsert → done."""
    chunk_chars = int(os.getenv("INGEST_CHUNK_CHARS", "1200"))
    overlap = int(os.getenv("INGEST_CHUNK_OVERLAP", "120"))

    await _set_progress(upload_id, status="uploading", progress_percent=10, booth_id=booth_id, file_name=file_name)

    text = _extract_text_from_pdf(content)
    await _set_progress(upload_id, status="ocr", progress_percent=40, characters=len(text))

    chunks = _chunk(text, chunk_chars, overlap)
    if not chunks:
        await _set_progress(upload_id, status="failed", progress_percent=0, error_message="no extractable text")
        return

    try:
        import openai  # type: ignore
        from pinecone import Pinecone  # type: ignore

        client = openai.AsyncOpenAI()
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.getenv("PINECONE_INDEX", "micemore-booths"))
        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

        vectors = []
        for i, chunk_text in enumerate(chunks):
            emb = (await client.embeddings.create(model=embed_model, input=chunk_text)).data[0].embedding
            vectors.append(
                {
                    "id": f"{upload_id}#chunk-{i}",
                    "values": emb,
                    "metadata": {
                        "doc_title": file_name,
                        "page": i + 1,
                        "text": chunk_text,
                        "booth_id": booth_id,
                        "upload_id": upload_id,
                    },
                }
            )
            if i % 5 == 0:
                pct = 40 + int(45 * (i + 1) / len(chunks))
                await _set_progress(upload_id, status="embedding", progress_percent=pct)

        index.upsert(vectors=vectors, namespace=f"booth-{booth_id}")
        await _set_progress(
            upload_id, status="done", progress_percent=100, chunks_count=len(vectors)
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("ingest failed for %s", upload_id)
        await _set_progress(upload_id, status="failed", progress_percent=0, error_message=str(exc))


async def _ingest_mock(upload_id: str, booth_id: str, file_name: str) -> None:
    """mock 파이프라인 — 4단계 진행률 시뮬레이션."""
    phases = [("uploading", 25), ("ocr", 55), ("embedding", 85), ("done", 100)]
    for status, pct in phases:
        await asyncio.sleep(0.4)
        await _set_progress(
            upload_id,
            status=status,
            progress_percent=pct,
            booth_id=booth_id,
            file_name=file_name,
            chunks_count=23 if status == "done" else None,
        )


@upload_router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_material(
    booth_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Flutter MaterialsProvider.enqueueUpload 가 호출. 진행률은 Firestore watch."""
    if not file.filename:
        raise HTTPException(400, "filename required")
    upload_id = f"UP-{uuid.uuid4().hex[:14]}"
    content = await file.read()

    await _set_progress(
        upload_id,
        status="queued",
        progress_percent=0,
        booth_id=booth_id,
        file_name=file.filename,
        size_bytes=len(content),
        created_at=datetime.utcnow().isoformat() + "Z",
    )

    if _is_live_pipeline_ready():
        asyncio.create_task(_ingest_real(upload_id, booth_id, file.filename, content))
    else:
        asyncio.create_task(_ingest_mock(upload_id, booth_id, file.filename))

    return UploadResponse(
        upload_id=upload_id,
        booth_id=booth_id,
        file_name=file.filename,
        status="queued",
        progress_percent=0,
    )


@upload_router.get("/uploads/{upload_id}", response_model=UploadResponse)
async def upload_status(upload_id: str):
    """폴링 fallback — Firestore listener 못 쓰는 클라이언트용."""
    try:
        from firebase_admin import firestore, initialize_app  # type: ignore

        try:
            initialize_app()
        except ValueError:
            pass
        doc = firestore.client().collection("material_uploads").document(upload_id).get()
        if not doc.exists:
            raise HTTPException(404, "upload not found")
        data = doc.to_dict()
        return UploadResponse(
            upload_id=upload_id,
            booth_id=str(data.get("booth_id", "")),
            file_name=str(data.get("file_name", "")),
            status=str(data.get("status", "queued")),
            progress_percent=float(data.get("progress_percent", 0)),
        )
    except ImportError:
        raise HTTPException(503, "firebase-admin not installed")
