"""seed_pinecone.py — micemore-booths Pinecone 인덱스 부트스트랩 + 데모 부스 시드.

USAGE:
    # PLACEHOLDER 인 상태에서는 자동으로 dry-run (인덱스 생성/upsert 시도 X)
    python -m tools.seed_pinecone

    # 실 키 + 강제 dry-run
    python -m tools.seed_pinecone --dry-run

    # 실 키로 풀 시드 (인덱스 없으면 생성 + 데모 부스 3개 mock PDF 5개 임베드+upsert)
    python -m tools.seed_pinecone --apply

ENV:
    PINECONE_API_KEY      pcsk-... (PLACEHOLDER 면 dry-run 강제)
    PINECONE_INDEX        micemore-booths (default)
    PINECONE_REGION       us-east-1 (default, serverless)
    PINECONE_CLOUD        aws (default)
    PINECONE_DIMENSION    1536 (text-embedding-3-small)
    OPENAI_API_KEY        sk-... (실 임베딩에 필요)
    INGEST_CHUNK_TOKENS   500
    INGEST_CHUNK_TOKEN_OVERLAP  50

이 스크립트는 production-ready 입니다. 키가 placeholder 이면 안전하게 schema validate
+ 통계 출력만 하고 종료합니다 — 외부 호출 0.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

# .env auto-load — uvicorn 과 동일 동작.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


PLACEHOLDER_TOKENS = ("PLACEHOLDER", "sk-ant-PLACE", "sk-PLACE", "pcsk-PLACE")


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    upper = value.upper()
    return any(token.upper() in upper for token in PLACEHOLDER_TOKENS)


# ── 데모 부스 mock 자료 (5 PDF 가상 청킹) ────────────────────────────
DEMO_BOOTHS = [
    {
        "booth_id": "lumen",
        "title": "Lumen Semiconductors",
        "files": [
            ("lumen_company.pdf", "Lumen Semiconductors는 차세대 EUV 노광 광원 기업입니다. 2018년 설립, 본사 서울 판교."),
            ("lumen_product_xeon.pdf", "Lumen XEON 광원 모듈은 13.5nm EUV 광원 출력 250W 안정 운영, 가용 시간 90% 이상."),
            ("lumen_product_atlas.pdf", "ATLAS 시리즈는 Foundry 7nm 노드용 검사 광원, 0.1nm 정밀도 보장."),
            ("lumen_pricing.pdf", "라이선스 모델 — Enterprise $1.2M/year, Pilot $300K/3-month, OEM 별도 협의."),
            ("lumen_case_study_samsung.pdf", "삼성 평택 P3 fab 도입 사례 — 노광 수율 7.4% 향상, ROI 18개월 회수."),
        ],
    },
    {
        "booth_id": "everblue",
        "title": "Everblue Marine Tech",
        "files": [
            ("everblue_intro.pdf", "Everblue Marine Tech 는 자율 무인선박 (USV) 솔루션 회사. 부산 본사."),
            ("everblue_specs.pdf", "Atlas-USV 12m 클래스, 최대 항해속도 28노트, 자율 항해 72시간 연속."),
            ("everblue_use_cases.pdf", "한국 해양경찰청, 한국가스공사 LNG 터미널 보안 순찰 적용 중."),
            ("everblue_environment.pdf", "전기 추진 + 배터리 800kWh, CO2 배출 0g/h."),
            ("everblue_partners.pdf", "MOU 체결 파트너 — KAIST, 한화시스템, KT SAT."),
        ],
    },
    {
        "booth_id": "noctura",
        "title": "Noctura Bio",
        "files": [
            ("noctura_company.pdf", "Noctura Bio 는 ML 기반 항암 단일항체 발굴 플랫폼 기업. 송도 IBS 단지."),
            ("noctura_pipeline.pdf", "주요 파이프라인 NB-101 (HER2+ 위암, IND 단계), NB-204 (PD-L1 폐암, 전임상)."),
            ("noctura_platform.pdf", "딥러닝 기반 ANTIGEN-AI 플랫폼 — 신규 항원 후보 20만개 스크리닝/주."),
            ("noctura_clinical.pdf", "NB-101 phase 1, n=18, ORR 33%, mPFS 6.2개월 (Q3 2025 데이터)."),
            ("noctura_partners.pdf", "공동 연구 — 서울대병원, MD Anderson, Genentech."),
        ],
    },
]


def _chunk_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """매우 단순한 character-based 청킹. 실 인제스트는 tiktoken 기반."""
    if len(text) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_tokens])
        start += max_tokens - overlap
    return chunks


def plan_seed(verbose: bool = True) -> dict:
    """모든 부스 × 파일 × 청크 수를 계산. 실 호출 X."""
    chunk_tokens = int(os.getenv("INGEST_CHUNK_TOKENS", "500"))
    overlap = int(os.getenv("INGEST_CHUNK_TOKEN_OVERLAP", "50"))
    plan = {"booths": [], "total_chunks": 0, "total_files": 0}
    for booth in DEMO_BOOTHS:
        booth_chunks = 0
        for filename, body in booth["files"]:
            chunks = _chunk_text(body, chunk_tokens, overlap)
            booth_chunks += len(chunks)
        plan["booths"].append(
            {"booth_id": booth["booth_id"], "files": len(booth["files"]), "chunks": booth_chunks}
        )
        plan["total_chunks"] += booth_chunks
        plan["total_files"] += len(booth["files"])
    if verbose:
        print(f"[plan] booths: {len(DEMO_BOOTHS)}")
        for b in plan["booths"]:
            print(f"  - {b['booth_id']:14}  files={b['files']:2}  chunks={b['chunks']:3}")
        print(f"[plan] total_files={plan['total_files']}  total_chunks={plan['total_chunks']}")
        print(
            f"[plan] embed model={os.getenv('OPENAI_EMBED_MODEL', 'text-embedding-3-small')} "
            f"dim={os.getenv('PINECONE_DIMENSION', '1536')}"
        )
    return plan


def validate_schema() -> bool:
    """ENV/구조 사전 검증. PLACEHOLDER 도 통과."""
    required = ["PINECONE_INDEX", "PINECONE_DIMENSION", "PINECONE_REGION", "PINECONE_CLOUD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[error] missing env: {missing}")
        return False
    print(
        f"[schema] index={os.getenv('PINECONE_INDEX')} "
        f"dim={os.getenv('PINECONE_DIMENSION')} "
        f"cloud={os.getenv('PINECONE_CLOUD')} region={os.getenv('PINECONE_REGION')}"
    )
    return True


def ensure_index() -> None:
    """실 모드 — 인덱스 없으면 serverless 로 생성."""
    from pinecone import Pinecone, ServerlessSpec  # type: ignore

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    name = os.environ["PINECONE_INDEX"]
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name in existing:
        print(f"[index] {name} 이미 존재")
        return
    print(f"[index] {name} 생성 중 (serverless {os.environ['PINECONE_CLOUD']}/{os.environ['PINECONE_REGION']})")
    pc.create_index(
        name=name,
        dimension=int(os.environ["PINECONE_DIMENSION"]),
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.environ["PINECONE_CLOUD"],
            region=os.environ["PINECONE_REGION"],
        ),
    )
    print(f"[index] {name} 생성 완료")


def upsert_demo_booths() -> None:
    """실 모드 — 데모 부스 청크를 임베딩 후 namespace=booth-{booth_id} 로 upsert."""
    import openai  # type: ignore
    from pinecone import Pinecone  # type: ignore

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX"])
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chunk_tokens = int(os.getenv("INGEST_CHUNK_TOKENS", "500"))
    overlap = int(os.getenv("INGEST_CHUNK_TOKEN_OVERLAP", "50"))

    client = openai.OpenAI()
    total = 0
    for booth in DEMO_BOOTHS:
        ns = f"booth-{booth['booth_id']}"
        vectors: list[dict] = []
        for filename, body in booth["files"]:
            for i, chunk in enumerate(_chunk_text(body, chunk_tokens, overlap)):
                emb = client.embeddings.create(model=embed_model, input=chunk).data[0].embedding
                vectors.append(
                    {
                        "id": f"{filename}#chunk-{i}",
                        "values": emb,
                        "metadata": {
                            "doc_title": filename,
                            "page": i + 1,
                            "text": chunk,
                            "booth_id": booth["booth_id"],
                        },
                    }
                )
        if vectors:
            index.upsert(vectors=vectors, namespace=ns)
            print(f"[upsert] ns={ns} vectors={len(vectors)}")
            total += len(vectors)
    print(f"[done] total upserted = {total}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="실 인덱스 생성 + upsert")
    parser.add_argument("--dry-run", action="store_true", help="강제 dry-run (기본은 placeholder 자동 감지)")
    args = parser.parse_args(list(argv) if argv else None)

    if not validate_schema():
        return 2

    placeholder = _is_placeholder(os.getenv("PINECONE_API_KEY")) or _is_placeholder(os.getenv("OPENAI_API_KEY"))
    dry = args.dry_run or placeholder or not args.apply

    plan = plan_seed(verbose=True)

    if dry:
        reason = "placeholder keys" if placeholder else ("--dry-run" if args.dry_run else "no --apply")
        print(f"\n[dry-run] {reason} — Pinecone 호출 없이 종료. 활성화: --apply + 실 키 주입.")
        return 0

    print("\n[apply] 실 모드 — 인덱스 확인 + upsert 시작")
    ensure_index()
    upsert_demo_booths()
    print("[apply] 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
