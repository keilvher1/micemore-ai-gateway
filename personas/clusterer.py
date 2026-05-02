"""페르소나 클러스터링 — UMAP + HDBSCAN.

설계:
  - **실 모드**: UMAP 으로 14차원 → 2D, HDBSCAN 으로 밀도 기반 클러스터.
    소음 (noise label = -1) 을 자연스럽게 분리하므로 K-means 보다 안전.
  - **mock 모드**: 외부 의존성 없이 결정론적 KMeans-lite (greedy farthest-point).
    CI · USE_MOCK · 데이터 부족 시 대체. 결과는 의미상 클러스터링 X — 형식만.

피처 입력은 Phase 3 의 `ml.features.behavior_to_features()` 출력과 호환.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass, field

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

# 데이터 부족 시 클러스터링 시도 안 함 (모트 보호 + 통계적 의미)
MIN_SAMPLES_FOR_CLUSTERING = int(os.getenv("PERSONAS_MIN_SAMPLES", "5000"))

# 운영 권장: 8~12. 베타 시작값 10.
DEFAULT_N_CLUSTERS = 10


# ---------------------------------------------------------------------------
# 결과 모델
# ---------------------------------------------------------------------------
@dataclass
class ClusterMember:
    visitor_hash: str
    cluster_id: int           # -1 = noise (HDBSCAN), >=0 = 실 클러스터
    confidence: float         # 0~1, 거리 기반 (실모드에서는 HDBSCAN probability)


@dataclass
class ClusterCentroid:
    cluster_id: int
    size: int
    avg_features: list[float]    # FEATURE_NAMES 순서
    feature_top: list[tuple[str, float]]  # 상위 기여 피처 (이름, 값)


@dataclass
class ClusteringResult:
    n_clusters: int
    n_noise: int
    method: str                  # "hdbscan" | "kmeans-lite-mock"
    members: list[ClusterMember]
    centroids: list[ClusterCentroid]
    silhouette: float | None = None
    feature_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------
def cluster_visitors(
    *,
    visitor_hashes: list[str],
    feature_matrix: list[list[float]],
    feature_names: list[str],
    target_clusters: int = DEFAULT_N_CLUSTERS,
    seed: int = 42,
) -> ClusteringResult:
    """visitor 행동 피처 → 페르소나 클러스터 할당."""
    n = len(visitor_hashes)
    if n != len(feature_matrix):
        raise ValueError(
            f"visitor_hashes({n}) != feature_matrix({len(feature_matrix)})"
        )
    if n == 0:
        raise ValueError("empty input")
    if any(len(row) != len(feature_names) for row in feature_matrix):
        raise ValueError("feature row width mismatch with feature_names")

    if USE_MOCK or n < MIN_SAMPLES_FOR_CLUSTERING:
        return _mock_kmeans_lite(
            visitor_hashes=visitor_hashes,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            target_clusters=min(target_clusters, n),
            seed=seed,
        )
    return _hdbscan_real(
        visitor_hashes=visitor_hashes,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 실모드 — UMAP + HDBSCAN
# ---------------------------------------------------------------------------
def _hdbscan_real(
    *,
    visitor_hashes: list[str],
    feature_matrix: list[list[float]],
    feature_names: list[str],
    seed: int,
) -> ClusteringResult:
    import numpy as np  # type: ignore
    import umap  # type: ignore
    import hdbscan  # type: ignore
    from sklearn.metrics import silhouette_score  # type: ignore

    X = np.asarray(feature_matrix, dtype=float)
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed
    )
    X2 = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(20, len(feature_matrix) // 200),
        min_samples=10,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X2)

    members = [
        ClusterMember(
            visitor_hash=h,
            cluster_id=int(labels[i]),
            confidence=float(clusterer.probabilities_[i])
            if hasattr(clusterer, "probabilities_") else 1.0,
        )
        for i, h in enumerate(visitor_hashes)
    ]
    centroids = _build_centroids(members, feature_matrix, feature_names)
    n_noise = sum(1 for m in members if m.cluster_id == -1)

    sil: float | None = None
    valid_mask = labels != -1
    if valid_mask.sum() > 1 and len(set(labels[valid_mask])) > 1:
        sil = float(silhouette_score(X2[valid_mask], labels[valid_mask]))

    return ClusteringResult(
        n_clusters=len({c.cluster_id for c in centroids}),
        n_noise=n_noise,
        method="hdbscan",
        members=members,
        centroids=centroids,
        silhouette=sil,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Mock — 결정론 KMeans-lite (sklearn 의존성 없음)
# ---------------------------------------------------------------------------
def _mock_kmeans_lite(
    *,
    visitor_hashes: list[str],
    feature_matrix: list[list[float]],
    feature_names: list[str],
    target_clusters: int,
    seed: int,
) -> ClusteringResult:
    """순수 Python KMeans-lite — 외부 dep 없이 결정론.

    1) seed 로 첫 centroid 1개 무작위 선택
    2) 나머지 centroid 는 farthest-point 휴리스틱
    3) 각 점 가장 가까운 centroid 에 할당
    4) Lloyd 반복 5회
    """
    rng = random.Random(seed)
    n_features = len(feature_names)
    n_points = len(feature_matrix)
    k = max(1, min(target_clusters, n_points))

    # 1) seed centroid
    first_idx = rng.randrange(n_points)
    centroid_indices = [first_idx]
    # 2) farthest-point picking
    while len(centroid_indices) < k:
        best_idx, best_dist = -1, -1.0
        for i in range(n_points):
            if i in centroid_indices:
                continue
            d_min = min(
                _l2(feature_matrix[i], feature_matrix[c])
                for c in centroid_indices
            )
            if d_min > best_dist:
                best_idx, best_dist = i, d_min
        centroid_indices.append(best_idx)
    centroids = [list(feature_matrix[i]) for i in centroid_indices]

    # 3-4) Lloyd 반복
    assignments = [0] * n_points
    for _ in range(5):
        changed = False
        for i, x in enumerate(feature_matrix):
            best_c, best_d = 0, float("inf")
            for ci, c in enumerate(centroids):
                d = _l2(x, c)
                if d < best_d:
                    best_c, best_d = ci, d
            if assignments[i] != best_c:
                assignments[i] = best_c
                changed = True
        if not changed:
            break
        # update centroids
        for ci in range(k):
            members = [
                feature_matrix[i] for i in range(n_points) if assignments[i] == ci
            ]
            if not members:
                continue
            for f in range(n_features):
                centroids[ci][f] = sum(m[f] for m in members) / len(members)

    members_out: list[ClusterMember] = []
    for i, h in enumerate(visitor_hashes):
        ci = assignments[i]
        d = _l2(feature_matrix[i], centroids[ci])
        confidence = 1.0 / (1.0 + d)
        members_out.append(
            ClusterMember(
                visitor_hash=h,
                cluster_id=ci,
                confidence=round(confidence, 4),
            )
        )

    centroids_out = _build_centroids(members_out, feature_matrix, feature_names)
    return ClusteringResult(
        n_clusters=k,
        n_noise=0,
        method="kmeans-lite-mock",
        members=members_out,
        centroids=centroids_out,
        silhouette=None,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def _build_centroids(
    members: list[ClusterMember],
    feature_matrix: list[list[float]],
    feature_names: list[str],
) -> list[ClusterCentroid]:
    by_cluster: dict[int, list[int]] = {}
    for i, m in enumerate(members):
        by_cluster.setdefault(m.cluster_id, []).append(i)

    out: list[ClusterCentroid] = []
    for cid, idx_list in sorted(by_cluster.items()):
        if cid == -1:  # HDBSCAN noise — centroid 없음
            continue
        n_feat = len(feature_names)
        avg = [
            sum(feature_matrix[i][f] for i in idx_list) / len(idx_list)
            for f in range(n_feat)
        ]
        # 상위 기여 — 평균 + 정규화 (전체 평균 대비) 비교는 namer 가 처리
        ranked = sorted(
            range(n_feat), key=lambda f: avg[f], reverse=True
        )[:5]
        feature_top = [(feature_names[f], round(avg[f], 3)) for f in ranked]
        out.append(
            ClusterCentroid(
                cluster_id=cid,
                size=len(idx_list),
                avg_features=[round(v, 4) for v in avg],
                feature_top=feature_top,
            )
        )
    return out


def to_dict(result: ClusteringResult) -> dict:
    """JSON 직렬화 친화적 dict."""
    return {
        "n_clusters": result.n_clusters,
        "n_noise": result.n_noise,
        "method": result.method,
        "silhouette": result.silhouette,
        "feature_names": result.feature_names,
        "members": [asdict(m) for m in result.members],
        "centroids": [asdict(c) for c in result.centroids],
    }
