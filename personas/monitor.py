"""페르소나 클러스터링 품질 모니터.

Silhouette 는 clusterer 가 이미 계산하지만, 추가로:
  - 클러스터 크기 분포 (gini-like — 한 페르소나 독점 방지)
  - drift: 이전 학습 결과와 클러스터 매핑 비교 (centroid 거리)

순수 함수, 외부 의존성 없음.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from personas.clusterer import ClusterCentroid, ClusteringResult


@dataclass
class HealthReport:
    n_clusters: int
    n_total: int
    n_noise: int
    noise_ratio: float
    largest_cluster_ratio: float
    smallest_cluster_ratio: float
    size_gini: float                 # 0=균등, 1=한 클러스터 독점
    silhouette: float | None
    alert: bool
    reasons: list[str]


def evaluate(
    result: ClusteringResult,
    *,
    max_noise_ratio: float = 0.15,
    max_largest_ratio: float = 0.40,
    min_silhouette: float = 0.20,
) -> HealthReport:
    n_total = len(result.members)
    if n_total == 0:
        return HealthReport(
            n_clusters=0, n_total=0, n_noise=0, noise_ratio=0.0,
            largest_cluster_ratio=0.0, smallest_cluster_ratio=0.0,
            size_gini=0.0, silhouette=None, alert=True,
            reasons=["empty result"],
        )

    sizes = [c.size for c in result.centroids]
    largest = max(sizes) if sizes else 0
    smallest = min(sizes) if sizes else 0
    noise_ratio = result.n_noise / n_total
    largest_ratio = (largest / n_total) if n_total else 0.0
    smallest_ratio = (smallest / n_total) if n_total else 0.0
    gini = _gini(sizes) if sizes else 0.0

    reasons: list[str] = []
    if noise_ratio > max_noise_ratio:
        reasons.append(
            f"noise {noise_ratio:.0%} > {max_noise_ratio:.0%}"
        )
    if largest_ratio > max_largest_ratio:
        reasons.append(
            f"largest cluster {largest_ratio:.0%} > {max_largest_ratio:.0%}"
        )
    if result.silhouette is not None and result.silhouette < min_silhouette:
        reasons.append(
            f"silhouette {result.silhouette:.2f} < {min_silhouette}"
        )

    return HealthReport(
        n_clusters=result.n_clusters,
        n_total=n_total,
        n_noise=result.n_noise,
        noise_ratio=round(noise_ratio, 3),
        largest_cluster_ratio=round(largest_ratio, 3),
        smallest_cluster_ratio=round(smallest_ratio, 3),
        size_gini=round(gini, 3),
        silhouette=result.silhouette,
        alert=bool(reasons),
        reasons=reasons,
    )


def _gini(sizes: list[int]) -> float:
    """클러스터 크기의 Gini 계수. 0=균등, 1=한 클러스터 독점."""
    if not sizes:
        return 0.0
    sorted_s = sorted(sizes)
    n = len(sorted_s)
    cum = sum((i + 1) * s for i, s in enumerate(sorted_s))
    total = sum(sorted_s) or 1
    return (2 * cum) / (n * total) - (n + 1) / n


def cluster_overlap(
    prev: list[ClusterCentroid], curr: list[ClusterCentroid]
) -> list[tuple[int, int, float]]:
    """이전 vs 현재 클러스터 best-match — 거리 기반 Hungarian-lite.

    Returns: [(prev_id, curr_id, distance), ...]  prev 기준 1:1 매핑
             거리가 클수록 페르소나가 유의미하게 변경됐다는 뜻 → 재명명 신호.
    """
    matches: list[tuple[int, int, float]] = []
    used: set[int] = set()
    for p in prev:
        best, best_d = -1, math.inf
        for c in curr:
            if c.cluster_id in used:
                continue
            d = sum(
                (a - b) ** 2
                for a, b in zip(p.avg_features, c.avg_features)
            ) ** 0.5
            if d < best_d:
                best, best_d = c.cluster_id, d
        if best >= 0:
            used.add(best)
            matches.append((p.cluster_id, best, round(best_d, 3)))
    return matches
