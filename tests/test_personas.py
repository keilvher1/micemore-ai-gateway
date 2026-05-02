"""personas/{clusterer,namer,monitor}.py 검증.

mock 모드 + USE_MOCK=true 로 외부 의존성 없이 결정론 검증.
"""
from __future__ import annotations

import os
import random

os.environ["USE_MOCK"] = "true"

import pytest  # noqa: E402

from ml.features import FEATURE_NAMES, behavior_to_features  # noqa: E402
from personas.clusterer import (  # noqa: E402
    cluster_visitors,
    to_dict,
)
from personas.monitor import cluster_overlap, evaluate  # noqa: E402
from personas.namer import name_clusters  # noqa: E402


def _seed_behaviors(n: int = 60, seed: int = 7) -> list[dict]:
    """5 페르소나 패턴이 섞인 합성 visitor 데이터."""
    rng = random.Random(seed)
    rows: list[dict] = []
    archetypes = [
        # (dwell_sec, q_count, topics, pamphlet, card, trans, others, comp, revisit)
        (420, 6, ["pricing", "pricing", "tech"], False, True, 4, 8, 2, 2),   # 적극 정보 수집
        (240, 2, ["overview"], True, True, 3, 4, 1, 0),                       # 외국인 비즈니스 (translation 우세)
        (180, 5, ["tech", "integration"], True, True, 0, 6, 3, 0),            # 네트워킹 우선
        (300, 3, ["overview", "timeline"], True, False, 0, 1, 0, 4),          # 단일 솔루션 탐색 (재방문)
        (90, 1, ["overview"], False, False, 0, 20, 0, 0),                     # 호기심 관광
    ]
    for i in range(n):
        a = archetypes[i % len(archetypes)]
        rows.append({
            "visitor_id": f"v_{i:04d}",
            "booth_id": "booth_a",
            "booth_dwell_time_sec": a[0] + rng.randint(-30, 30),
            "copilot_questions_count": max(0, a[1] + rng.randint(-1, 1)),
            "copilot_question_topics": list(a[2]),
            "pamphlet_downloaded": a[3],
            "business_card_saved": a[4],
            "translation_session_minutes": a[5],
            "other_booths_visited": a[6] + rng.randint(0, 3),
            "competitor_booths_visited": a[7],
            "revisit_count": a[8],
            "first_visit_hour": 11,
            "booth_visit_order": (i % 5) + 1,
            "avg_question_length": 18,
            "question_complexity": 0.5,
        })
    return rows


# ---------------------------------------------------------------------------
# Clusterer (mock = KMeans-lite)
# ---------------------------------------------------------------------------
def test_cluster_returns_requested_n():
    rows = _seed_behaviors(60)
    visitor_hashes = [r["visitor_id"] for r in rows]
    feats = [behavior_to_features(r) for r in rows]
    result = cluster_visitors(
        visitor_hashes=visitor_hashes,
        feature_matrix=feats,
        feature_names=list(FEATURE_NAMES),
        target_clusters=5,
        seed=42,
    )
    assert result.n_clusters == 5
    assert result.method == "kmeans-lite-mock"
    assert len(result.members) == 60


def test_cluster_deterministic_same_seed():
    rows = _seed_behaviors(40)
    feats = [behavior_to_features(r) for r in rows]
    hashes = [r["visitor_id"] for r in rows]
    a = cluster_visitors(
        visitor_hashes=hashes, feature_matrix=feats,
        feature_names=list(FEATURE_NAMES),
        target_clusters=4, seed=123,
    )
    b = cluster_visitors(
        visitor_hashes=hashes, feature_matrix=feats,
        feature_names=list(FEATURE_NAMES),
        target_clusters=4, seed=123,
    )
    assert [m.cluster_id for m in a.members] == [m.cluster_id for m in b.members]


def test_cluster_different_seed_changes_assignment():
    rows = _seed_behaviors(40)
    feats = [behavior_to_features(r) for r in rows]
    hashes = [r["visitor_id"] for r in rows]
    a = cluster_visitors(
        visitor_hashes=hashes, feature_matrix=feats,
        feature_names=list(FEATURE_NAMES), target_clusters=5, seed=1,
    )
    b = cluster_visitors(
        visitor_hashes=hashes, feature_matrix=feats,
        feature_names=list(FEATURE_NAMES), target_clusters=5, seed=999,
    )
    # 다른 seed → 적어도 일부 점은 다른 클러스터에 할당
    diffs = sum(
        1 for x, y in zip(a.members, b.members)
        if x.cluster_id != y.cluster_id
    )
    assert diffs > 0


def test_cluster_input_mismatch_raises():
    with pytest.raises(ValueError):
        cluster_visitors(
            visitor_hashes=["a", "b"],
            feature_matrix=[[1.0, 2.0]],  # mismatch
            feature_names=["x", "y"],
        )


def test_cluster_empty_input_raises():
    with pytest.raises(ValueError):
        cluster_visitors(
            visitor_hashes=[],
            feature_matrix=[],
            feature_names=list(FEATURE_NAMES),
        )


def test_cluster_centroids_have_top_features():
    rows = _seed_behaviors(50)
    feats = [behavior_to_features(r) for r in rows]
    hashes = [r["visitor_id"] for r in rows]
    result = cluster_visitors(
        visitor_hashes=hashes, feature_matrix=feats,
        feature_names=list(FEATURE_NAMES), target_clusters=5,
    )
    for c in result.centroids:
        assert len(c.feature_top) <= 5
        assert all(name in FEATURE_NAMES for name, _ in c.feature_top)


def test_to_dict_round_trip_serializable():
    import json
    rows = _seed_behaviors(30)
    result = cluster_visitors(
        visitor_hashes=[r["visitor_id"] for r in rows],
        feature_matrix=[behavior_to_features(r) for r in rows],
        feature_names=list(FEATURE_NAMES),
        target_clusters=4,
    )
    d = to_dict(result)
    s = json.dumps(d)
    assert "n_clusters" in s
    assert "centroids" in s


# ---------------------------------------------------------------------------
# Namer (mock)
# ---------------------------------------------------------------------------
def test_name_clusters_returns_one_per_centroid():
    rows = _seed_behaviors(40)
    result = cluster_visitors(
        visitor_hashes=[r["visitor_id"] for r in rows],
        feature_matrix=[behavior_to_features(r) for r in rows],
        feature_names=list(FEATURE_NAMES),
        target_clusters=4,
    )
    names = name_clusters(result)
    assert len(names) == len(result.centroids)


def test_name_mock_uses_pattern_rules():
    rows = _seed_behaviors(50)
    result = cluster_visitors(
        visitor_hashes=[r["visitor_id"] for r in rows],
        feature_matrix=[behavior_to_features(r) for r in rows],
        feature_names=list(FEATURE_NAMES),
        target_clusters=5,
    )
    names = name_clusters(result)
    expected = {
        "적극 정보 수집형", "외국인 비즈니스", "네트워킹 우선형",
        "단일 솔루션 탐색형", "호기심 관광형", "구매 검토형",
        "기술 검토형", "일반 방문형",
    }
    for p in names:
        assert p.model == "mock"
        assert p.name in expected
        assert len(p.tagline) <= 24
        assert len(p.description) > 0
        assert len(p.feature_importance) <= 4


def test_name_empty_centroids_returns_empty_list():
    from personas.clusterer import ClusteringResult
    empty = ClusteringResult(
        n_clusters=0, n_noise=0, method="mock",
        members=[], centroids=[], feature_names=list(FEATURE_NAMES),
    )
    assert name_clusters(empty) == []


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
def test_evaluate_health_basic():
    rows = _seed_behaviors(50)
    result = cluster_visitors(
        visitor_hashes=[r["visitor_id"] for r in rows],
        feature_matrix=[behavior_to_features(r) for r in rows],
        feature_names=list(FEATURE_NAMES),
        target_clusters=5,
    )
    rep = evaluate(result)
    assert rep.n_total == 50
    assert rep.n_clusters == 5
    assert 0.0 <= rep.size_gini <= 1.0


def test_evaluate_alerts_on_oversized_cluster():
    """target_clusters=2 + 합성 데이터 50 → 한 클러스터 50% 이상 차지 가능."""
    rows = _seed_behaviors(50)
    result = cluster_visitors(
        visitor_hashes=[r["visitor_id"] for r in rows],
        feature_matrix=[behavior_to_features(r) for r in rows],
        feature_names=list(FEATURE_NAMES),
        target_clusters=2,
    )
    rep = evaluate(result, max_largest_ratio=0.40)
    if rep.largest_cluster_ratio > 0.40:
        assert rep.alert is True
        assert any("largest cluster" in r for r in rep.reasons)


def test_evaluate_handles_empty():
    from personas.clusterer import ClusteringResult
    empty = ClusteringResult(
        n_clusters=0, n_noise=0, method="mock",
        members=[], centroids=[], feature_names=[],
    )
    rep = evaluate(empty)
    assert rep.alert is True
    assert "empty result" in rep.reasons


def test_cluster_overlap_simple():
    from personas.clusterer import ClusterCentroid
    prev = [
        ClusterCentroid(0, 30, [1, 1, 1], [("a", 1)]),
        ClusterCentroid(1, 30, [9, 9, 9], [("a", 9)]),
    ]
    curr = [
        ClusterCentroid(0, 35, [1.1, 1.0, 0.9], [("a", 1)]),
        ClusterCentroid(1, 32, [9.2, 8.9, 9.1], [("a", 9)]),
    ]
    matches = cluster_overlap(prev, curr)
    # 0→0, 1→1 매핑이 자연
    by_prev = {p: (c, d) for p, c, d in matches}
    assert by_prev[0][0] == 0
    assert by_prev[1][0] == 1
    assert by_prev[0][1] < 0.5
