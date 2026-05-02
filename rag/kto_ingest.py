"""한국관광공사 TourAPI → Pinecone RAG 인제스트.

8개 한국관광공사 OpenAPI 오퍼레이션을 통합해 관광 콘텐츠를
4개 언어 (Ko/En/Ja/Zh) 로 임베딩하여 Pinecone 의 tour:* namespace 에 적재.

파이프라인:
    areaBasedSyncList2 (변경분) 또는 areaBasedList2 (전체 시드)
        ↓
    각 contentid → detailCommon2 + detailIntro2 + detailInfo2 (lang × 4)
        ↓
    텍스트 통합 → 청킹 (1200자, 120 overlap) → text-embedding-3-small
        ↓
    Pinecone upsert
        index   = micemore-tour
        namespace = tour:{lang}:{areacode}
        metadata  = {contentid, title, addr, content_type, mapx, mapy, source: "kto"}

CLI:
    # 전체 시드 (베타 초기 1회)
    python -m rag.kto_ingest --mode full --areas 1,35,6,39 --langs ko,en,ja,zh

    # 일일 incremental
    python -m rag.kto_ingest --mode sync --since 20260501

    # dry-run (Pinecone 호출 X, 시드만 카운트)
    python -m rag.kto_ingest --mode full --dry-run

키 미발급 시 자동으로 dry-run + mock 데이터로 동작.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

import httpx

from rag.embeddings import embed_chunks

log = logging.getLogger(__name__)

USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
KTO_TOURAPI_KEY = os.getenv("KTO_TOURAPI_KEY", "PLACEHOLDER")
PINECONE_TOUR_INDEX = os.getenv("PINECONE_TOUR_INDEX", "micemore-tour")
SERVICE_NAME = "MICEMore"

# 청킹 파라미터 — 관광 콘텐츠는 부스 자료보다 짧으므로 작게
CHUNK_CHARS = int(os.getenv("KTO_CHUNK_CHARS", "1000"))
CHUNK_OVERLAP = int(os.getenv("KTO_CHUNK_OVERLAP", "100"))

LANG_SERVICE = {
    "ko": "KorService2",
    "en": "EngService2",
    "ja": "JpnService2",
    "zh": "ChsService2",
}

CONTENT_TYPE_LABEL = {
    "12": "관광지",
    "14": "문화시설",
    "15": "행사",
    "25": "여행코스",
    "28": "레저",
    "32": "숙박",
    "38": "쇼핑",
    "39": "음식점",
}


def _is_real_key() -> bool:
    return bool(KTO_TOURAPI_KEY) and "PLACEHOLDER" not in KTO_TOURAPI_KEY and not USE_MOCK


# ============================================================================
# 헬퍼 — TourAPI 호출
# ============================================================================
async def _kto_get(service: str, endpoint: str, params: dict) -> list[dict]:
    """공사 OpenAPI 호출. 실패 시 빈 리스트."""
    if not _is_real_key():
        return []

    base = {
        "serviceKey": KTO_TOURAPI_KEY,
        "MobileOS": "ETC",
        "MobileApp": SERVICE_NAME,
        "_type": "json",
    }
    url = f"https://apis.data.go.kr/B551011/{service}/{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params={**base, **params})
            r.raise_for_status()
            data = r.json()
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
        if isinstance(items, dict):
            items = [items]
        return items
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        log.warning("KTO %s/%s failed: %s", service, endpoint, e)
        return []


# ============================================================================
# Mock seed fixtures — dry-run 환경에서도 파이프라인 흐름 검증 가능
# ============================================================================
_MOCK_SEEDS = {
    "35": [  # 경북
        {"contentid": "3110308", "contenttypeid": "15", "title": "POEX 2026", "areacode": "35"},
        {"contentid": "126508", "contenttypeid": "12", "title": "송도해수욕장", "areacode": "35"},
        {"contentid": "129380", "contenttypeid": "12", "title": "영일대해수욕장", "areacode": "35"},
        {"contentid": "812044", "contenttypeid": "39", "title": "포항 죽도시장 — 회 거리", "areacode": "35"},
        {"contentid": "172843", "contenttypeid": "32", "title": "포항 베스트웨스턴 호텔", "areacode": "35"},
        {"contentid": "2840125", "contenttypeid": "15", "title": "경주 실크로드 컨퍼런스 2026", "areacode": "35"},
    ],
    "1": [  # 서울
        {"contentid": "2950011", "contenttypeid": "15", "title": "Smart Korea 2026 — Seoul", "areacode": "1"},
        {"contentid": "126186", "contenttypeid": "12", "title": "경복궁", "areacode": "1"},
        {"contentid": "264326", "contenttypeid": "14", "title": "국립중앙박물관", "areacode": "1"},
    ],
    "6": [  # 부산
        {"contentid": "264319", "contenttypeid": "12", "title": "해운대해수욕장", "areacode": "6"},
    ],
    "39": [  # 제주
        {"contentid": "126295", "contenttypeid": "12", "title": "성산일출봉", "areacode": "39"},
    ],
}


async def _seed_areacode(area_code: str, content_type_id: Optional[str], num_rows: int = 100) -> list[dict]:
    """areaBasedList2 (전체 시드) 호출."""
    if not _is_real_key():
        items = _MOCK_SEEDS.get(area_code, [])
        if content_type_id:
            items = [it for it in items if it.get("contenttypeid") == content_type_id]
        return items[:num_rows]

    params = {"areaCode": area_code, "numOfRows": num_rows, "pageNo": 1, "arrange": "C"}
    if content_type_id:
        params["contentTypeId"] = content_type_id
    return await _kto_get("KorService2", "areaBasedList2", params)


async def _seed_sync(area_code: str, since_date: str, num_rows: int = 500) -> list[dict]:
    """areaBasedSyncList2 (변경분) 호출."""
    if not _is_real_key():
        # mock: 어제 변경된 것으로 시뮬레이션 — 시드의 절반만 반환
        items = _MOCK_SEEDS.get(area_code, [])
        return items[: max(1, len(items) // 2)]

    params = {
        "areaCode": area_code,
        "syncModifiedSince": since_date,
        "numOfRows": num_rows,
        "pageNo": 1,
    }
    return await _kto_get("KorService2", "areaBasedSyncList2", params)


async def _detail_lang(content_id: str, content_type_id: str, lang: str) -> dict:
    """detailCommon2 + detailIntro2 + detailInfo2 한 언어 병렬."""
    service = LANG_SERVICE.get(lang, "KorService2")
    params = {"contentId": content_id}
    intro_params = {**params, "contentTypeId": content_type_id} if content_type_id else params

    common, intro, info = await asyncio.gather(
        _kto_get(service, "detailCommon2", params),
        _kto_get(service, "detailIntro2", intro_params),
        _kto_get(service, "detailInfo2", intro_params),
    )

    return {
        "common": common[0] if common else {},
        "intro": intro[0] if intro else {},
        "info": info if isinstance(info, list) else [],
    }


# ============================================================================
# 텍스트 통합 + 청킹
# ============================================================================
def _strip_html(s: str) -> str:
    """homepage 같은 필드의 HTML 태그 제거."""
    if not s:
        return ""
    return re.sub(r"<[^>]+>", " ", s)


def _flatten_detail(detail: dict, lang: str) -> str:
    """3개 detail 응답 → 단일 텍스트 (RAG 학습용)."""
    common = detail.get("common", {}) or {}
    intro = detail.get("intro", {}) or {}
    info = detail.get("info", []) or []

    parts: list[str] = []

    # 제목 + 주소 (앵커)
    if common.get("title"):
        parts.append(f"[{common['title']}]")
    if common.get("addr1"):
        parts.append(common["addr1"])

    # 본문 overview
    overview = _strip_html(common.get("overview", ""))
    if overview:
        parts.append(overview)

    # 운영 메타
    intro_lines = []
    if intro.get("usefee"):
        intro_lines.append(f"입장료/Fee: {intro['usefee']}")
    if intro.get("parking"):
        intro_lines.append(f"주차/Parking: {intro['parking']}")
    if intro.get("opentime"):
        intro_lines.append(f"운영시간/Hours: {intro['opentime']}")
    if intro.get("restdate"):
        intro_lines.append(f"휴무일/Closed: {intro['restdate']}")
    if intro.get("eventstartdate") or intro.get("eventenddate"):
        intro_lines.append(f"행사일/Event: {intro.get('eventstartdate', '')} ~ {intro.get('eventenddate', '')}")
    if intro.get("eventplace"):
        intro_lines.append(f"행사장소/Venue: {intro['eventplace']}")
    if intro.get("sponsor1"):
        intro_lines.append(f"주최/Host: {intro['sponsor1']}")
    if intro_lines:
        parts.append(" / ".join(intro_lines))

    # 반복정보
    for it in info:
        name = _strip_html(it.get("infoname", ""))
        text = _strip_html(it.get("infotext", ""))
        if name and text:
            parts.append(f"{name}: {text}")
        elif text:
            parts.append(text)

    return "\n\n".join(p for p in parts if p)


def _split_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """간단한 sliding window — 문단 경계를 우선 존중."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            last_para = text.rfind("\n\n", start + size // 2, end)
            if last_para != -1:
                end = last_para + 2
            else:
                last_period = text.rfind(". ", start + size // 2, end)
                if last_period != -1:
                    end = last_period + 1
        out.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return out


# ============================================================================
# Pinecone upsert
# ============================================================================
async def _pinecone_upsert(vectors: list[dict], namespace: str) -> int:
    """Pinecone 의 micemore-tour 인덱스에 batch upsert.

    vectors: [{"id": str, "values": list[float], "metadata": dict}, ...]
    """
    if not vectors:
        return 0

    if USE_MOCK or "PLACEHOLDER" in os.getenv("PINECONE_API_KEY", "PLACEHOLDER"):
        log.info("Pinecone upsert dry-run — %d vectors → namespace=%s", len(vectors), namespace)
        return len(vectors)

    try:
        from pinecone import Pinecone  # type: ignore
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_TOUR_INDEX)
        # batch 100 씩
        total = 0
        for i in range(0, len(vectors), 100):
            batch = vectors[i : i + 100]
            index.upsert(vectors=batch, namespace=namespace)
            total += len(batch)
        log.info("Pinecone upsert OK — %d vectors → namespace=%s", total, namespace)
        return total
    except Exception as e:
        log.error("Pinecone upsert failed: %s", e)
        return 0


# ============================================================================
# 인제스트 메인
# ============================================================================
async def ingest_one_content(
    content_id: str, content_type_id: str, area_code: str, langs: list[str],
) -> dict:
    """contentid 1건을 4개 언어로 인제스트."""
    if not _is_real_key():
        # mock — 단순히 카운트만
        return {"content_id": content_id, "vectors": len(langs) * 3, "source": "mock"}

    total_vectors = 0
    for lang in langs:
        try:
            detail = await _detail_lang(content_id, content_type_id, lang)
            text = _flatten_detail(detail, lang)
            if not text:
                continue
            chunks = _split_text(text)
            if not chunks:
                continue
            embeddings = await embed_chunks(chunks)
            common = detail.get("common", {}) or {}
            metadata_base = {
                "contentid": content_id,
                "title": common.get("title", ""),
                "addr": common.get("addr1", ""),
                "content_type": CONTENT_TYPE_LABEL.get(content_type_id, "기타"),
                "content_type_id": content_type_id,
                "mapx": common.get("mapx", ""),
                "mapy": common.get("mapy", ""),
                "areacode": area_code,
                "lang": lang,
                "source": "한국관광공사 TourAPI",
            }
            vectors = [
                {
                    "id": f"kto-{content_id}-{lang}-{i}",
                    "values": emb,
                    "metadata": {**metadata_base, "chunk_idx": i, "text": chunk[:1500]},
                }
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]
            namespace = f"tour:{lang}:{area_code}"
            n = await _pinecone_upsert(vectors, namespace)
            total_vectors += n
        except Exception as e:
            log.error("ingest failed contentid=%s lang=%s: %s", content_id, lang, e)
            continue

    return {"content_id": content_id, "vectors": total_vectors, "source": "kto"}


async def ingest_pipeline(
    *,
    mode: str = "sync",  # 'full' | 'sync'
    area_codes: list[str] = None,
    content_type_ids: list[str] = None,
    langs: list[str] = None,
    since_date: Optional[str] = None,
    max_per_area: int = 100,
    dry_run: bool = False,
) -> dict:
    """전체 파이프라인 실행. CLI 와 HTTP 트리거 모두에서 호출."""
    area_codes = area_codes or ["1", "35", "6", "39"]
    content_type_ids = content_type_ids or ["12", "14", "15", "32", "39"]
    langs = langs or ["ko", "en", "ja", "zh"]
    since_date = since_date or (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # 1. 시드 수집
    seeds: list[dict] = []
    for area in area_codes:
        if mode == "sync":
            items = await _seed_sync(area, since_date, num_rows=max_per_area)
        else:
            for ctid in content_type_ids:
                items = await _seed_areacode(area, ctid, num_rows=max_per_area)
                seeds.extend([
                    {"contentid": it.get("contentid"), "contenttypeid": it.get("contenttypeid", ctid), "areacode": area, "title": it.get("title", "")}
                    for it in items
                    if it.get("contentid")
                ])
            continue
        seeds.extend([
            {"contentid": it.get("contentid"), "contenttypeid": it.get("contenttypeid", "12"), "areacode": area, "title": it.get("title", "")}
            for it in items
            if it.get("contentid")
        ])

    log.info("Seeded %d contentids across %d areas (mode=%s)", len(seeds), len(area_codes), mode)

    if dry_run or not _is_real_key():
        return {
            "mode": mode,
            "dry_run": True,
            "seed_count": len(seeds),
            "estimated_vectors": len(seeds) * len(langs) * 5,  # avg 5 chunks
            "areas": {area: sum(1 for s in seeds if s["areacode"] == area) for area in area_codes},
            "next_step": "set KTO_TOURAPI_KEY + PINECONE_API_KEY then re-run without --dry-run",
        }

    # 2. 각 contentid 인제스트 (병렬 8개씩)
    semaphore = asyncio.Semaphore(8)

    async def _bounded(seed: dict) -> dict:
        async with semaphore:
            return await ingest_one_content(
                content_id=seed["contentid"],
                content_type_id=seed["contenttypeid"],
                area_code=seed["areacode"],
                langs=langs,
            )

    results = await asyncio.gather(*(_bounded(s) for s in seeds))
    total_vectors = sum(r.get("vectors", 0) for r in results)

    return {
        "mode": mode,
        "dry_run": False,
        "ingested_contentids": len(results),
        "total_vectors": total_vectors,
        "by_area": {a: sum(r["vectors"] for r, s in zip(results, seeds) if s["areacode"] == a) for a in area_codes},
    }


# ============================================================================
# CLI
# ============================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="한국관광공사 TourAPI → Pinecone RAG 인제스트")
    p.add_argument("--mode", choices=["full", "sync"], default="sync", help="full = areaBasedList2 / sync = areaBasedSyncList2")
    p.add_argument("--areas", default="1,35,6,39", help="지역코드 (쉼표 구분, 1=서울 35=경북 6=부산 39=제주)")
    p.add_argument("--types", default="12,14,15,32,39", help="콘텐츠 타입 (12=관광지 14=문화시설 15=행사 32=숙박 39=음식점)")
    p.add_argument("--langs", default="ko,en,ja,zh", help="다국어")
    p.add_argument("--since", default=None, help="sync 모드 기준일 YYYYMMDD")
    p.add_argument("--max-per-area", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


async def _main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    result = await ingest_pipeline(
        mode=args.mode,
        area_codes=args.areas.split(","),
        content_type_ids=args.types.split(","),
        langs=args.langs.split(","),
        since_date=args.since,
        max_per_area=args.max_per_area,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
