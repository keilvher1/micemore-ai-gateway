"""ML 리드 스코어링 라우터.

POST /ml-leads/predict   — 단일 방문자 (A/B 라우팅 자동)
POST /ml-leads/predict-batch — N 명 일괄
GET  /ml-leads/healthz

Phase 2 의 /leads 와 별도로 노출 — 점진적 마이그레이션 동안 양쪽 모두 응답.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ml.predict import predict, USE_ML_SCORING

router = APIRouter(prefix="/ml-leads", tags=["ml-leads"])
log = logging.getLogger("ml-leads")


class PredictIn(BaseModel):
    behavior: dict[str, Any] = Field(..., description="VisitorBehavior dict")
    force_ml: bool = False


class PredictOut(BaseModel):
    score: int
    level: str
    source: str
    confidence: float
    top_factors: list[dict[str, Any]]
    feature_version: str
    model_version: str | None


@router.post("/predict", response_model=PredictOut)
async def predict_one(req: PredictIn):
    p = predict(req.behavior, force_ml=req.force_ml)
    return PredictOut(**p.__dict__)


class BatchIn(BaseModel):
    items: list[dict[str, Any]] = Field(..., min_length=1, max_length=500)


@router.post("/predict-batch", response_model=list[PredictOut])
async def predict_batch(req: BatchIn):
    return [PredictOut(**predict(b).__dict__) for b in req.items]


@router.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "module": "ml-leads",
        "use_ml_scoring": USE_ML_SCORING,
    }
