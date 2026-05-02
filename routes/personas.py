"""페르소나 클러스터링 라우터.

POST /personas/cluster   : visitor 행동 dict 리스트 → 클러스터 결과
POST /personas/name      : 클러스터 결과 → 페르소나 이름 + 설명
POST /personas/health    : 클러스터 결과 → 품질 리포트
GET  /personas/healthz
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml.features import FEATURE_NAMES, behavior_to_features
from personas.clusterer import (
    ClusteringResult,
    cluster_visitors,
    to_dict as result_to_dict,
)
from personas.monitor import evaluate
from personas.namer import name_clusters

router = APIRouter(prefix="/personas", tags=["personas"])


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------
class ClusterIn(BaseModel):
    behaviors: list[dict[str, Any]] = Field(..., min_length=1, max_length=20000)
    target_clusters: int = 10
    seed: int = 42


@router.post("/cluster", response_model=dict)
async def cluster_route(req: ClusterIn):
    visitor_hashes = [
        str(b.get("visitor_id") or f"v_{i}")
        for i, b in enumerate(req.behaviors)
    ]
    feature_matrix = [behavior_to_features(b) for b in req.behaviors]
    result = cluster_visitors(
        visitor_hashes=visitor_hashes,
        feature_matrix=feature_matrix,
        feature_names=list(FEATURE_NAMES),
        target_clusters=req.target_clusters,
        seed=req.seed,
    )
    return result_to_dict(result)


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------
class NameIn(BaseModel):
    cluster_result: dict[str, Any]


def _result_from_dict(d: dict) -> ClusteringResult:
    from personas.clusterer import (
        ClusterCentroid,
        ClusterMember,
        ClusteringResult,
    )
    return ClusteringResult(
        n_clusters=int(d.get("n_clusters", 0)),
        n_noise=int(d.get("n_noise", 0)),
        method=str(d.get("method", "")),
        members=[ClusterMember(**m) for m in d.get("members", [])],
        centroids=[ClusterCentroid(**c) for c in d.get("centroids", [])],
        silhouette=d.get("silhouette"),
        feature_names=list(d.get("feature_names", [])),
    )


@router.post("/name", response_model=list[dict])
async def name_route(req: NameIn):
    try:
        result = _result_from_dict(req.cluster_result)
    except (KeyError, TypeError) as exc:
        raise HTTPException(422, f"invalid cluster_result: {exc}") from exc
    names = name_clusters(result)
    return [
        {
            "cluster_id": n.cluster_id,
            "name": n.name,
            "tagline": n.tagline,
            "description": n.description,
            "feature_importance": n.feature_importance,
            "model": n.model,
        }
        for n in names
    ]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthIn(BaseModel):
    cluster_result: dict[str, Any]


@router.post("/health", response_model=dict)
async def health_route(req: HealthIn):
    result = _result_from_dict(req.cluster_result)
    rep = evaluate(result)
    return asdict(rep)


@router.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "module": "personas",
        "mock": os.getenv("USE_MOCK", "false").lower() == "true",
    }
