"""한국관광공사 TourAPI 4.0 연동 — 공모전 자격 조건 + RAG 학습 데이터 인제스트.

활용 API 8종 (단순 호출 + AI 학습용):
  [UI 호출용]
  1. searchFestival2     — 행사/축제 검색 (다음 행사 매칭)
  2. locationBasedList2  — 위치 기반 관광지/숙박/음식점 (부스 주변)
  [RAG 학습 인제스트용]
  3. areaBasedList2      — 지역별 contentid 시드
  4. areaBasedSyncList2  — 일일 변경분 동기화
  5. detailCommon2       — 콘텐츠 본문 텍스트 (overview 100~500자)
  6. detailIntro2        — 운영 메타 (입장료·주차·휴일·운영시간)
  7. detailInfo2         — 반복 정보 (객실·메뉴·코스 단계)
  [다국어]
  8. EngService2 / JpnService2 / ChsService2 — 4개 언어 동시 인제스트

키 활성화: .env 의 KTO_TOURAPI_KEY 가 'PLACEHOLDER' 가 아니면 실 호출,
         아니면 mock JSON 반환. USE_MOCK=true 강제 시 항상 mock.

엔드포인트:
  [UI]
  GET  /tourapi/festivals         — 행사 검색 (날짜/지역)
  GET  /tourapi/nearby            — 부스 주변 (GPS+반경)
  GET  /tourapi/match-bonus       — 사용자 지역 다음 행사
  POST /tourapi/cache/refresh     — 일일 캐시 갱신
  [RAG]
  GET  /tourapi/area-list         — areaBasedList2
  GET  /tourapi/sync              — areaBasedSyncList2 (변경분)
  GET  /tourapi/detail            — detailCommon2 + detailIntro2 + detailInfo2 통합
  POST /tourapi/ingest            — RAG 인제스트 파이프라인 트리거
  POST /tourapi/search-rag        — 학습된 데이터에서 RAG 검색 (Boomi Chat 호출)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/tourapi", tags=["tourapi"])
log = logging.getLogger(__name__)

USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
KTO_TOURAPI_KEY = os.getenv("KTO_TOURAPI_KEY", "PLACEHOLDER")
KTO_BASE_URL = "https://apis.data.go.kr/B551011/KorService2"
SERVICE_NAME = "MICEMore"  # MobileApp 파라미터

# 다국어 서비스 prefix 매핑
LANG_SERVICE = {
    "ko": "KorService2",
    "en": "EngService2",
    "ja": "JpnService2",
    "zh": "ChsService2",
}


def _is_real_key() -> bool:
    """실 API 호출 가능 여부."""
    return bool(KTO_TOURAPI_KEY) and "PLACEHOLDER" not in KTO_TOURAPI_KEY and not USE_MOCK


# ============================================================================
# Mock fixtures — 시연 + 키 미발급 환경 폴백
# ============================================================================

_MOCK_FESTIVALS = [
    {
        "contentid": "3110308",
        "title": "포항국제이차전지전시회 (POEX) 2026",
        "addr1": "경상북도 포항시 남구 송도동",
        "areacode": "35",  # 경북
        "sigungucode": "11",  # 포항
        "eventstartdate": "20260910",
        "eventenddate": "20260912",
        "mapx": "129.3650",
        "mapy": "36.0190",
        "firstimage": "",
        "tel": "054-289-1230",
    },
    {
        "contentid": "2840125",
        "title": "경주 실크로드 컨퍼런스 2026",
        "addr1": "경상북도 경주시 신평동",
        "areacode": "35",
        "sigungucode": "2",
        "eventstartdate": "20260820",
        "eventenddate": "20260822",
        "mapx": "129.3450",
        "mapy": "35.8530",
        "firstimage": "",
        "tel": "054-779-6394",
    },
    {
        "contentid": "2950011",
        "title": "Smart Korea 2026 — Seoul",
        "addr1": "서울특별시 강남구 영동대로 513 COEX",
        "areacode": "1",
        "sigungucode": "1",
        "eventstartdate": "20260508",
        "eventenddate": "20260510",
        "mapx": "127.0599",
        "mapy": "37.5126",
        "firstimage": "",
        "tel": "02-6000-0114",
    },
]

_MOCK_NEARBY = {
    # contentTypeId → 항목
    "12": [  # 관광지
        {"contentid": "126508", "title": "송도해수욕장", "addr1": "경북 포항시 남구 송도동", "dist": "420", "mapx": "129.3691", "mapy": "36.0124"},
        {"contentid": "129380", "title": "영일대해수욕장", "addr1": "경북 포항시 북구 두호동", "dist": "3120", "mapx": "129.3853", "mapy": "36.0541"},
    ],
    "32": [  # 숙박
        {"contentid": "172843", "title": "포항 베스트웨스턴 호텔", "addr1": "경북 포항시 남구", "dist": "850", "mapx": "129.3712", "mapy": "36.0203"},
    ],
    "39": [  # 음식점
        {"contentid": "812044", "title": "포항 죽도시장 — 회 거리", "addr1": "경북 포항시 북구 죽도동", "dist": "2100", "mapx": "129.3708", "mapy": "36.0341"},
        {"contentid": "812891", "title": "송도 활어회센터", "addr1": "경북 포항시 남구 송도동", "dist": "510", "mapx": "129.3679", "mapy": "36.0156"},
    ],
}


# ============================================================================
# Pydantic models
# ============================================================================
class FestivalItem(BaseModel):
    contentid: str
    title: str
    addr1: str = ""
    areacode: str = ""
    sigungucode: str = ""
    eventstartdate: str = ""
    eventenddate: str = ""
    mapx: str = ""
    mapy: str = ""
    firstimage: str = ""
    tel: str = ""


class NearbyItem(BaseModel):
    contentid: str
    title: str
    addr1: str = ""
    dist: str = "0"  # 거리 (m)
    mapx: str = ""
    mapy: str = ""
    contenttypeid: str = ""


# ============================================================================
# 1) searchFestival2 — 행사/축제 검색
# ============================================================================
async def _do_search_festivals(
    event_start_date: str,
    event_end_date: Optional[str] = None,
    area_code: Optional[str] = None,
    sigungu_code: Optional[str] = None,
    lang: str = "ko",
    num_of_rows: int = 20,
    page_no: int = 1,
) -> dict:
    """비즈니스 로직 분리 — FastAPI Query 객체 의존성 제거하여 다른 라우트에서 재사용 가능."""
    if not _is_real_key():
        log.info("searchFestival2: USE_MOCK or PLACEHOLDER key — returning fixture")
        items = [f for f in _MOCK_FESTIVALS if f["eventstartdate"] >= event_start_date]
        if area_code:
            items = [f for f in items if f["areacode"] == area_code]
        return {"items": items, "total": len(items), "source": "mock"}

    service = LANG_SERVICE.get(lang, "KorService2")
    url = f"https://apis.data.go.kr/B551011/{service}/searchFestival2"
    params = {
        "serviceKey": KTO_TOURAPI_KEY,
        "MobileOS": "ETC",
        "MobileApp": SERVICE_NAME,
        "_type": "json",
        "eventStartDate": event_start_date,
        "numOfRows": num_of_rows,
        "pageNo": page_no,
        "arrange": "C",
    }
    if event_end_date:
        params["eventEndDate"] = event_end_date
    if area_code:
        params["areaCode"] = area_code
    if sigungu_code:
        params["sigunguCode"] = sigungu_code

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
        if isinstance(items, dict):
            items = [items]
        total = data.get("response", {}).get("body", {}).get("totalCount", len(items))
        return {"items": items, "total": total, "source": "kto"}
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        log.warning("searchFestival2 failed (%s) — fallback to mock", e)
        return {"items": _MOCK_FESTIVALS, "total": len(_MOCK_FESTIVALS), "source": "mock_fallback"}


@router.get("/festivals")
async def search_festivals(
    event_start_date: str = Query(..., description="YYYYMMDD"),
    event_end_date: Optional[str] = Query(None, description="YYYYMMDD"),
    area_code: Optional[str] = Query(None, description="지역코드 (35=경북, 1=서울 등)"),
    sigungu_code: Optional[str] = Query(None, description="시군구코드"),
    lang: str = Query("ko", pattern=r"^(ko|en|ja|zh)$"),
    num_of_rows: int = Query(20, ge=1, le=100),
    page_no: int = Query(1, ge=1),
):
    """한국관광공사 searchFestival2 wrapper. Boomi Chat / Match Notification 에서 사용."""
    return await _do_search_festivals(
        event_start_date=event_start_date,
        event_end_date=event_end_date,
        area_code=area_code,
        sigungu_code=sigungu_code,
        lang=lang,
        num_of_rows=num_of_rows,
        page_no=page_no,
    )


# ============================================================================
# 2) locationBasedList2 — 부스 GPS 주변 관광/숙박/음식점
# ============================================================================
@router.get("/nearby")
async def nearby_by_location(
    map_x: float = Query(..., description="경도 (longitude)"),
    map_y: float = Query(..., description="위도 (latitude)"),
    radius: int = Query(5000, ge=100, le=20000, description="반경 (m), max 20km"),
    content_type_id: str = Query(
        "12",
        pattern=r"^(12|14|15|25|28|32|38|39)$",
        description="12=관광지 14=문화시설 15=행사 25=여행코스 28=레저 32=숙박 38=쇼핑 39=음식점",
    ),
    lang: str = Query("ko", pattern=r"^(ko|en|ja|zh)$"),
    num_of_rows: int = Query(10, ge=1, le=50),
):
    """한국관광공사 locationBasedList2 wrapper.

    NFC 태깅 직후 부스 좌표 + 반경 5km → 주변 관광지/숙박/음식점.
    Boomi Chat 의 "이 행사 끝나고 어디 가볼만해?" 답변에도 사용.
    """
    if not _is_real_key():
        log.info("locationBasedList2: USE_MOCK or PLACEHOLDER — returning fixture for type %s", content_type_id)
        items = _MOCK_NEARBY.get(content_type_id, [])
        return {"items": items, "total": len(items), "source": "mock", "lang": lang}

    service = LANG_SERVICE.get(lang, "KorService2")
    url = f"https://apis.data.go.kr/B551011/{service}/locationBasedList2"
    params = {
        "serviceKey": KTO_TOURAPI_KEY,
        "MobileOS": "ETC",
        "MobileApp": SERVICE_NAME,
        "_type": "json",
        "mapX": map_x,
        "mapY": map_y,
        "radius": radius,
        "contentTypeId": content_type_id,
        "numOfRows": num_of_rows,
        "arrange": "S",  # 거리순
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
        if isinstance(items, dict):
            items = [items]
        total = data.get("response", {}).get("body", {}).get("totalCount", len(items))
        return {"items": items, "total": total, "source": "kto", "lang": lang}
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        log.warning("locationBasedList2 failed (%s) — fallback to mock", e)
        items = _MOCK_NEARBY.get(content_type_id, [])
        return {"items": items, "total": len(items), "source": "mock_fallback", "lang": lang}


# ============================================================================
# 3) Match bonus — saved_booths 지역 기반 다음 행사 자동 추천
# ============================================================================
@router.get("/match-bonus")
async def match_bonus(
    user_area_code: str = Query("35", description="사용자 관심 지역 (기본 경북=35)"),
    days_ahead: int = Query(30, ge=1, le=180, description="향후 N일 내"),
    lang: str = Query("ko", pattern=r"^(ko|en|ja|zh)$"),
):
    """다음 행사 매칭 알림 보강용.

    사용자의 saved_booths 지역 + 향후 N일 → 같은 지역 행사 자동 매칭.
    GET /me/matches 응답에 한국관광공사 공식 행사 정보를 합성하는 데 사용.
    """
    today = datetime.now().strftime("%Y%m%d")
    end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y%m%d")

    return await _do_search_festivals(
        event_start_date=today,
        event_end_date=end_date,
        area_code=user_area_code,
        lang=lang,
        num_of_rows=10,
        page_no=1,
    )


# ============================================================================
# 4) 캐시 갱신 트리거 (Cloud Scheduler 일일 새벽 1시)
# ============================================================================
class CacheRefreshRequest(BaseModel):
    area_codes: list[str] = Field(default_factory=lambda: ["1", "35", "6", "39"])  # 서울/경북/부산/제주
    days_ahead: int = 60


@router.post("/cache/refresh")
async def refresh_cache(req: CacheRefreshRequest):
    """일일 배치로 향후 N일 행사를 미리 가져와 Firestore tour_events 컬렉션에 캐시.

    실제 Firestore 쓰기는 Cloud Function 에서 처리하고,
    이 엔드포인트는 fetched 데이터만 반환 — 클라이언트는 Firestore 만 읽음.
    """
    today = datetime.now().strftime("%Y%m%d")
    end_date = (datetime.now() + timedelta(days=req.days_ahead)).strftime("%Y%m%d")

    results = {}
    for area in req.area_codes:
        try:
            res = await _do_search_festivals(
                event_start_date=today,
                event_end_date=end_date,
                area_code=area,
                lang="ko",
                num_of_rows=100,
                page_no=1,
            )
            results[area] = {"count": len(res.get("items", [])), "source": res.get("source")}
        except Exception as e:
            log.error("cache refresh failed for area %s: %s", area, e)
            results[area] = {"error": str(e)}
    return {"refreshed_at": datetime.now().isoformat(), "results": results}


# ============================================================================
# 5) areaBasedList2 — 지역별 contentid 시드 (RAG 인제스트 파이프라인 시작점)
# ============================================================================
async def _do_area_based_list(
    area_code: str,
    sigungu_code: Optional[str] = None,
    content_type_id: Optional[str] = None,
    lang: str = "ko",
    num_of_rows: int = 100,
    page_no: int = 1,
) -> dict:
    """비즈니스 헬퍼."""
    if not _is_real_key():
        # Mock: areacode 35 (경북) 면 _MOCK_FESTIVALS 의 행사 contentid 반환
        items = []
        if area_code == "35":
            items = [
                {"contentid": "3110308", "contenttypeid": "15", "title": "포항국제이차전지전시회 (POEX) 2026", "addr1": "경상북도 포항시 남구 송도동", "areacode": "35", "sigungucode": "11", "mapx": "129.3650", "mapy": "36.0190"},
                {"contentid": "126508", "contenttypeid": "12", "title": "송도해수욕장", "addr1": "경북 포항시 남구 송도동", "areacode": "35", "sigungucode": "11", "mapx": "129.3691", "mapy": "36.0124"},
                {"contentid": "129380", "contenttypeid": "12", "title": "영일대해수욕장", "addr1": "경북 포항시 북구 두호동", "areacode": "35", "sigungucode": "11", "mapx": "129.3853", "mapy": "36.0541"},
                {"contentid": "812044", "contenttypeid": "39", "title": "포항 죽도시장 — 회 거리", "addr1": "경북 포항시 북구 죽도동", "areacode": "35", "sigungucode": "11", "mapx": "129.3708", "mapy": "36.0341"},
                {"contentid": "172843", "contenttypeid": "32", "title": "포항 베스트웨스턴 호텔", "addr1": "경북 포항시 남구", "areacode": "35", "sigungucode": "11", "mapx": "129.3712", "mapy": "36.0203"},
            ]
        if content_type_id:
            items = [it for it in items if it["contenttypeid"] == content_type_id]
        return {"items": items, "total": len(items), "source": "mock"}

    service = LANG_SERVICE.get(lang, "KorService2")
    url = f"https://apis.data.go.kr/B551011/{service}/areaBasedList2"
    params = {
        "serviceKey": KTO_TOURAPI_KEY,
        "MobileOS": "ETC",
        "MobileApp": SERVICE_NAME,
        "_type": "json",
        "areaCode": area_code,
        "numOfRows": num_of_rows,
        "pageNo": page_no,
        "arrange": "C",  # 수정일순
    }
    if sigungu_code:
        params["sigunguCode"] = sigungu_code
    if content_type_id:
        params["contentTypeId"] = content_type_id

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
        if isinstance(items, dict):
            items = [items]
        total = data.get("response", {}).get("body", {}).get("totalCount", len(items))
        return {"items": items, "total": total, "source": "kto"}
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        log.warning("areaBasedList2 failed (%s) — fallback empty", e)
        return {"items": [], "total": 0, "source": "mock_fallback"}


@router.get("/area-list")
async def area_based_list(
    area_code: str = Query(..., description="지역코드 (35=경북, 1=서울 등)"),
    sigungu_code: Optional[str] = Query(None),
    content_type_id: Optional[str] = Query(None, pattern=r"^(12|14|15|25|28|32|38|39)?$"),
    lang: str = Query("ko", pattern=r"^(ko|en|ja|zh)$"),
    num_of_rows: int = Query(100, ge=1, le=1000),
    page_no: int = Query(1, ge=1),
):
    """RAG 인제스트 시드 — 지역별 콘텐츠 contentid 리스트."""
    return await _do_area_based_list(
        area_code=area_code,
        sigungu_code=sigungu_code,
        content_type_id=content_type_id,
        lang=lang,
        num_of_rows=num_of_rows,
        page_no=page_no,
    )


# ============================================================================
# 6) areaBasedSyncList2 — 변경분만 (일일 incremental)
# ============================================================================
@router.get("/sync")
async def sync_list(
    sync_modified_since: str = Query(..., description="YYYYMMDD — 이 날짜 이후 변경된 항목만"),
    area_code: Optional[str] = Query(None),
    lang: str = Query("ko", pattern=r"^(ko|en|ja|zh)$"),
    num_of_rows: int = Query(500, ge=1, le=1000),
    page_no: int = Query(1, ge=1),
):
    """API 호출량 90% 절감 — 변경된 contentid 만 일일 페치."""
    if not _is_real_key():
        # mock: 일부 항목만 변경된 것으로 시뮬레이션
        return {
            "items": [
                {"contentid": "3110308", "contenttypeid": "15", "title": "POEX 2026 (수정됨)", "modifiedtime": sync_modified_since + "120000", "areacode": area_code or "35"},
            ],
            "total": 1,
            "source": "mock",
        }

    service = LANG_SERVICE.get(lang, "KorService2")
    url = f"https://apis.data.go.kr/B551011/{service}/areaBasedSyncList2"
    params = {
        "serviceKey": KTO_TOURAPI_KEY,
        "MobileOS": "ETC",
        "MobileApp": SERVICE_NAME,
        "_type": "json",
        "syncModifiedSince": sync_modified_since,
        "numOfRows": num_of_rows,
        "pageNo": page_no,
    }
    if area_code:
        params["areaCode"] = area_code

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
        if isinstance(items, dict):
            items = [items]
        total = data.get("response", {}).get("body", {}).get("totalCount", len(items))
        return {"items": items, "total": total, "source": "kto"}
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        log.warning("areaBasedSyncList2 failed (%s) — fallback empty", e)
        return {"items": [], "total": 0, "source": "mock_fallback"}


# ============================================================================
# 7) detailCommon2 + detailIntro2 + detailInfo2 통합 — RAG 인제스트의 핵심
# ============================================================================
async def _do_detail_combined(
    content_id: str,
    content_type_id: Optional[str] = None,
    lang: str = "ko",
) -> dict:
    """3개 detail API 를 병렬 호출해 하나의 풍부한 콘텐츠 객체로 통합."""
    if not _is_real_key():
        # mock: contentid 별로 가공된 텍스트 반환
        mock_db = {
            "126508": {
                "common": {
                    "contentid": "126508",
                    "title": "송도해수욕장",
                    "addr1": "경북 포항시 남구 송도동",
                    "homepage": "<a href='https://pohang.go.kr'>포항시 공식</a>",
                    "tel": "054-289-1230",
                    "overview": "송도해수욕장은 포항 남구 송도동에 위치한 해수욕장으로, 길이 약 1.7km 의 백사장과 푸른 동해바다가 어우러진 명소입니다. 여름철에는 가족 단위 피서객이 몰리며, 인근에 송도 송림공원과 죽도시장이 있어 함께 둘러보기 좋습니다. 주차장 무료, 입장료 없음.",
                },
                "intro": {
                    "parking": "무료 주차장 200대",
                    "usefee": "무료",
                    "opentime": "24시간 개방",
                    "restdate": "연중무휴",
                    "infocenter": "054-289-1230",
                },
                "info": [],
            },
            "3110308": {
                "common": {
                    "contentid": "3110308",
                    "title": "포항국제이차전지전시회 (POEX) 2026",
                    "addr1": "경상북도 포항시 남구 송도동",
                    "homepage": "<a href='https://poex.kr'>POEX 공식</a>",
                    "tel": "054-289-1230",
                    "overview": "POEX 2026 은 한국이차전지산업협회가 주관하는 국내 최대 규모의 이차전지 전문 전시회입니다. 2026년 9월 10일부터 12일까지 포항 송도 컨벤션센터에서 열리며, 200여 개 부스와 1만 명 이상의 산업 관계자가 참여합니다. 글로벌 이차전지 트렌드와 K-배터리의 미래를 한자리에서 확인할 수 있습니다.",
                },
                "intro": {
                    "eventstartdate": "20260910",
                    "eventenddate": "20260912",
                    "eventplace": "포항 송도 컨벤션센터",
                    "sponsor1": "한국이차전지산업협회",
                    "agelimit": "전체 관람가",
                    "usefee": "사전등록 무료, 현장등록 30,000원",
                    "discountinfoFestival": "학생 50% 할인",
                },
                "info": [
                    {"infoname": "Day 1 (9/10)", "infotext": "개막식 + 키노트 + 부스 오픈"},
                    {"infoname": "Day 2 (9/11)", "infotext": "기술 컨퍼런스 + 1:1 비즈니스 미팅"},
                    {"infoname": "Day 3 (9/12)", "infotext": "투어 프로그램 + 폐막식"},
                ],
            },
        }
        rec = mock_db.get(content_id, {
            "common": {"contentid": content_id, "title": "Sample", "overview": "Mock overview text for RAG ingest demonstration."},
            "intro": {},
            "info": [],
        })
        return {"detail": rec, "source": "mock", "lang": lang}

    service = LANG_SERVICE.get(lang, "KorService2")

    async def _fetch(endpoint: str, extra_params: dict = None) -> dict:
        url = f"https://apis.data.go.kr/B551011/{service}/{endpoint}"
        params = {
            "serviceKey": KTO_TOURAPI_KEY,
            "MobileOS": "ETC",
            "MobileApp": SERVICE_NAME,
            "_type": "json",
            "contentId": content_id,
        }
        if extra_params:
            params.update(extra_params)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
            return data.get("response", {}).get("body", {}).get("items", {}).get("item", []) or []
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            log.warning("%s failed for %s: %s", endpoint, content_id, e)
            return []

    intro_params = {"contentTypeId": content_type_id} if content_type_id else {}
    common_task = _fetch("detailCommon2")
    intro_task = _fetch("detailIntro2", intro_params)
    info_task = _fetch("detailInfo2", intro_params)

    common_res, intro_res, info_res = await asyncio.gather(common_task, intro_task, info_task)

    if isinstance(common_res, list) and common_res:
        common = common_res[0]
    elif isinstance(common_res, dict):
        common = common_res
    else:
        common = {}

    if isinstance(intro_res, list) and intro_res:
        intro = intro_res[0]
    elif isinstance(intro_res, dict):
        intro = intro_res
    else:
        intro = {}

    info = info_res if isinstance(info_res, list) else ([info_res] if info_res else [])

    return {
        "detail": {"common": common, "intro": intro, "info": info},
        "source": "kto",
        "lang": lang,
    }


@router.get("/detail")
async def detail_combined(
    content_id: str = Query(...),
    content_type_id: Optional[str] = Query(None),
    lang: str = Query("ko", pattern=r"^(ko|en|ja|zh)$"),
):
    """detailCommon2 + detailIntro2 + detailInfo2 통합 호출 — RAG 인제스트의 핵심 텍스트 소스."""
    return await _do_detail_combined(
        content_id=content_id, content_type_id=content_type_id, lang=lang,
    )


# ============================================================================
# 8) RAG 인제스트 트리거 — 8개 API 통합 파이프라인
# ============================================================================
class IngestRequest(BaseModel):
    area_codes: list[str] = Field(default_factory=lambda: ["1", "35", "6", "39"])
    content_type_ids: list[str] = Field(default_factory=lambda: ["12", "14", "15", "32", "39"])
    langs: list[str] = Field(default_factory=lambda: ["ko", "en", "ja", "zh"])
    incremental: bool = True  # True 면 areaBasedSyncList2, False 면 areaBasedList2 (전체)
    max_per_area: int = 100   # 지역당 최대 contentid (Pinecone 부담 제어)
    dry_run: bool = True       # True 면 인덱싱 시도 X, 시드 contentid 만 반환


@router.post("/ingest")
async def ingest_pipeline(req: IngestRequest):
    """RAG 인제스트 파이프라인 트리거.

    1. areaBasedList2 또는 areaBasedSyncList2 로 contentid 시드
    2. 각 contentid 에 detailCommon2 + detailIntro2 + detailInfo2 (lang × 4)
    3. 텍스트 통합 → 청킹 → 임베딩 → Pinecone upsert
       (실 임베딩은 별도 모듈 tools/ingest_kto.py 가 처리, 이 엔드포인트는 트리거만)

    dry_run=true 또는 KTO key PLACEHOLDER 면 시드만 수집하고 Pinecone 호출 X.
    """
    sync_since = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    seed_contentids: dict[str, list[dict]] = {}  # area_code → list

    for area in req.area_codes:
        ids: list[dict] = []
        for ctid in req.content_type_ids:
            try:
                if req.incremental:
                    res = await sync_list(
                        sync_modified_since=sync_since,
                        area_code=area,
                        lang="ko",
                        num_of_rows=req.max_per_area,
                        page_no=1,
                    )
                else:
                    res = await _do_area_based_list(
                        area_code=area,
                        content_type_id=ctid,
                        lang="ko",
                        num_of_rows=req.max_per_area,
                        page_no=1,
                    )
                items = res.get("items", []) if isinstance(res, dict) else []
                ids.extend([{"contentid": it.get("contentid"), "contenttypeid": it.get("contenttypeid", ctid), "title": it.get("title", ""), "areacode": area} for it in items if it.get("contentid")])
            except Exception as e:
                log.error("ingest seed failed area=%s type=%s: %s", area, ctid, e)
        seed_contentids[area] = ids

    total_seeds = sum(len(v) for v in seed_contentids.values())
    log.info("ingest pipeline collected %d seed contentids across %d areas", total_seeds, len(req.area_codes))

    if req.dry_run or not _is_real_key():
        return {
            "status": "dry_run",
            "total_seeds": total_seeds,
            "areas": {k: len(v) for k, v in seed_contentids.items()},
            "langs_planned": req.langs,
            "estimated_documents": total_seeds * len(req.langs),
            "estimated_vectors": total_seeds * len(req.langs) * 5,  # avg 5 chunks/doc
            "next_step": "tools/ingest_kto.py 실행 (실제 임베딩 + Pinecone upsert)",
        }

    # 실 인제스트는 tools/ingest_kto.py 가 별도로 수행 (이 엔드포인트는 트리거 + 시드 캐시 역할)
    return {
        "status": "seeds_cached",
        "total_seeds": total_seeds,
        "areas": {k: len(v) for k, v in seed_contentids.items()},
        "next_step": "tools/ingest_kto.py 가 Pinecone upsert 수행",
    }


# ============================================================================
# 9) RAG 검색 — Boomi Chat 이 학습된 데이터 검색
# ============================================================================
class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    lang: str = Field("ko", pattern=r"^(ko|en|ja|zh)$")
    area_code: Optional[str] = Field(None, description="필터: 특정 지역만")
    top_k: int = Field(5, ge=1, le=20)


@router.post("/search-rag")
async def search_rag(req: RagSearchRequest):
    """학습된 한국관광공사 데이터에서 RAG 검색 — Boomi Chat 이 호출.

    Pinecone namespace: tour:{lang}:{area_code or 'all'}
    USE_MOCK 또는 키 미발급 환경에서는 mock_db 의 overview 텍스트를 그대로 반환.
    """
    if USE_MOCK or "PLACEHOLDER" in KTO_TOURAPI_KEY:
        # 키워드 매칭 mock
        kw = req.query.lower()
        mock_results = []
        if any(k in kw for k in ["송도", "포항", "해수욕장", "songdo", "pohang"]):
            mock_results.append({
                "contentid": "126508",
                "title": "송도해수욕장",
                "score": 0.91,
                "snippet": "송도해수욕장은 포항 남구 송도동에 위치한 해수욕장으로, 길이 약 1.7km 의 백사장과 푸른 동해바다가 어우러진 명소입니다.",
                "addr": "경북 포항시 남구 송도동",
                "source": "한국관광공사 TourAPI",
            })
        if any(k in kw for k in ["행사", "전시", "poex", "이차전지"]):
            mock_results.append({
                "contentid": "3110308",
                "title": "POEX 2026",
                "score": 0.88,
                "snippet": "POEX 2026 은 한국이차전지산업협회가 주관하는 국내 최대 규모의 이차전지 전문 전시회입니다.",
                "addr": "경상북도 포항시 남구 송도동",
                "source": "한국관광공사 TourAPI",
            })
        if not mock_results:
            mock_results.append({
                "contentid": "126508",
                "title": "송도해수욕장",
                "score": 0.42,
                "snippet": "포항 인근 해수욕장 정보입니다.",
                "addr": "경북 포항시 남구",
                "source": "한국관광공사 TourAPI",
            })
        return {
            "results": mock_results[: req.top_k],
            "namespace": f"tour:{req.lang}:{req.area_code or 'all'}",
            "source": "mock",
        }

    # TODO: Pinecone 검색 — tools/ingest_kto.py 와 동일 인덱스 + namespace
    # 실 구현: openai embed query → pinecone.query(top_k=req.top_k, filter={"areacode": req.area_code})
    return {"results": [], "namespace": f"tour:{req.lang}:{req.area_code or 'all'}", "source": "stub"}
