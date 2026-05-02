"""트렌드 라우터 — Phase 4-B.

POST /trends/forecast    : 시계열 forecast (이동평균 또는 Prophet)
POST /trends/keywords    : 분기별 키워드 변화 비교
POST /trends/report      : 분기 markdown 리포트
GET  /trends/healthz

가격 티어 미들웨어:
  - X-API-Tier: standard | enterprise | trial
  - standard: rate-limit 10K/월
  - enterprise: 무제한 + custom segments
  - trial: 100/일

본 라우트는 단일 진실 소스 — 가격 정책 변경은 routes/trends.py 의 _enforce_tier 만 수정.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import asdict

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from trends.keyword_tracker import (
    KeywordChange,
    compare_quarters,
    declining_keywords,
    emerging_keywords,
)
from trends.report_generator import generate as generate_report
from trends.timeseries import ForecastResult, forecast as ts_forecast

router = APIRouter(prefix="/trends", tags=["trends"])


# ---------------------------------------------------------------------------
# 매우 단순한 in-memory rate-limit (베타) — 실 운영에선 Redis 로 교체
# ---------------------------------------------------------------------------
_TIER_QUOTA = {
    "trial": (100, 86400),         # 100 req / 1d
    "standard": (10_000, 30 * 86400),  # 10K req / 30d
    "enterprise": (10_000_000, 30 * 86400),
}
_USAGE: dict[tuple[str, str], list[int]] = defaultdict(list)


def _enforce_tier(tier: str | None, key: str | None) -> str:
    t = (tier or "trial").lower()
    if t not in _TIER_QUOTA:
        raise HTTPException(400, f"unknown tier: {t}")
    if t != "trial" and not key:
        raise HTTPException(401, "X-API-Key required for non-trial tiers")
    quota, window = _TIER_QUOTA[t]
    now = int(time.time())
    bucket = _USAGE[(t, key or "anon")]
    bucket[:] = [ts for ts in bucket if now - ts < window]
    if len(bucket) >= quota:
        raise HTTPException(
            429,
            detail={
                "error": "quota_exceeded",
                "tier": t,
                "quota": quota,
                "window_sec": window,
            },
        )
    bucket.append(now)
    return t


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
class ForecastIn(BaseModel):
    series: dict[str, float] = Field(..., description="ISO date → count")
    horizon_days: int = Field(30, ge=1, le=365)
    ma_window: int = 14


@router.post("/forecast", response_model=dict)
async def forecast_route(
    req: ForecastIn,
    x_api_tier: str | None = Header(default=None, alias="X-API-Tier"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _enforce_tier(x_api_tier, x_api_key)
    res: ForecastResult = ts_forecast(
        req.series,
        horizon_days=req.horizon_days,
        ma_window=req.ma_window,
    )
    return asdict(res)


# ---------------------------------------------------------------------------
# Keyword change
# ---------------------------------------------------------------------------
class KeywordsIn(BaseModel):
    prev_quarter_docs: list[str] = Field(..., min_length=1, max_length=10000)
    curr_quarter_docs: list[str] = Field(..., min_length=1, max_length=10000)
    top_n: int = 50


@router.post("/keywords", response_model=dict)
async def keywords_route(
    req: KeywordsIn,
    x_api_tier: str | None = Header(default=None, alias="X-API-Tier"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _enforce_tier(x_api_tier, x_api_key)
    changes = compare_quarters(
        prev_docs=req.prev_quarter_docs,
        curr_docs=req.curr_quarter_docs,
        top_n=req.top_n,
    )
    return {
        "changes": [asdict(c) for c in changes],
        "emerging": [asdict(c) for c in emerging_keywords(changes)],
        "declining": [asdict(c) for c in declining_keywords(changes)],
    }


# ---------------------------------------------------------------------------
# Quarterly report
# ---------------------------------------------------------------------------
class ReportIn(BaseModel):
    quarter: str
    region: str
    series: dict[str, float] = Field(default_factory=dict)
    horizon_days: int = 30
    prev_quarter_docs: list[str] = Field(default_factory=list)
    curr_quarter_docs: list[str] = Field(default_factory=list)
    foreigner_ratio: float | None = None


@router.post("/report", response_model=dict)
async def report_route(
    req: ReportIn,
    x_api_tier: str | None = Header(default=None, alias="X-API-Tier"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    tier = _enforce_tier(x_api_tier, x_api_key)
    if tier == "trial":
        raise HTTPException(
            402,
            detail={
                "error": "tier_required",
                "message": "report 는 standard 이상 필요. /trends pricing 안내 참조.",
            },
        )
    forecast_res = ts_forecast(req.series, horizon_days=req.horizon_days)
    keyword_changes = (
        compare_quarters(
            prev_docs=req.prev_quarter_docs,
            curr_docs=req.curr_quarter_docs,
        )
        if req.prev_quarter_docs and req.curr_quarter_docs else []
    )
    report = generate_report(
        quarter=req.quarter,
        region=req.region,
        forecast=forecast_res,
        keyword_changes=keyword_changes,
        foreigner_ratio=req.foreigner_ratio,
    )
    return {
        "quarter": report.quarter,
        "region": report.region,
        "markdown": report.markdown,
        "model": report.model,
    }


@router.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "module": "trends",
        "mock": os.getenv("USE_MOCK", "false").lower() == "true",
        "tiers": list(_TIER_QUOTA.keys()),
    }
