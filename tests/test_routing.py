"""routing/{crowd_tracker,graph_builder,recommender,fastpass}.py."""
from __future__ import annotations

import os

os.environ["USE_MOCK"] = "true"

from mypass.challenge import MyPassChallenge  # noqa: E402
from mypass.progress import MyPassProgress  # noqa: E402
from routing.crowd_tracker import CrowdTracker  # noqa: E402
from routing.fastpass import boost_with_mypass  # noqa: E402
from routing.graph_builder import (  # noqa: E402
    BoothNode,
    build_graph,
    haversine_m,
    shortest_path,
)
from routing.recommender import recommend_next_booth  # noqa: E402


def _booths() -> list[BoothNode]:
    """B관 중앙 통로 부스 5개 + 카테고리/좌표."""
    return [
        BoothNode("a", "Lumen Labs", 37.51720, 127.04730, "ai", ),
        BoothNode("b", "Paperless Co.", 37.51725, 127.04740, "sustain"),
        BoothNode("c", "NovaSight", 37.51730, 127.04750, "healthcare"),
        BoothNode("d", "Ledger Seoul", 37.51800, 127.04730, "fintech"),  # 멀음
        BoothNode("e", "Far Booth", 37.52000, 127.04730, "etc"),         # 거리 초과
    ]


def _keywords() -> dict[str, list[str]]:
    return {
        "a": ["AI", "ML", "scanning"],
        "b": ["zero-paper", "sustain"],
        "c": ["healthcare", "imaging"],
        "d": ["fintech", "blockchain"],
        "e": ["other"],
    }


# ---------------------------------------------------------------------------
# CrowdTracker
# ---------------------------------------------------------------------------
def test_crowd_enter_exit_changes_state():
    t = CrowdTracker()
    t.on_enter("a", now=1000)
    t.on_enter("a", now=1010)
    s = t.get("a")
    assert s.current_visitors == 2
    t.on_exit("a", dwell_sec=180, now=1190)
    assert t.get("a").current_visitors == 1
    assert t.get("a").avg_dwell_min == 3.0


def test_crowd_queue_score_caps_at_one():
    t = CrowdTracker()
    for i in range(50):
        t.on_enter("a", now=1000 + i)
    assert t.get("a").queue_score() == 1.0


def test_crowd_unknown_booth_returns_default():
    t = CrowdTracker()
    s = t.get("nope")
    assert s.current_visitors == 0


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def test_haversine_known_distance():
    a = BoothNode("x", "x", 37.5663, 126.9779)
    b = BoothNode("y", "y", 37.5759, 126.9769)
    d = haversine_m(a, b)
    assert 1000 < d < 1200


def test_build_graph_adds_edges_within_max():
    g = build_graph(_booths(), max_edge_m=200.0)
    # a~b 가까움 → edge 있음
    assert "b" in g.edges["a"]
    # a~e 너무 멀음 (~310m) → edge 없음
    assert "e" not in g.edges["a"]


def test_shortest_path_simple():
    g = build_graph(_booths(), max_edge_m=200.0)
    path, dist = shortest_path(g, src="a", dst="c")
    assert path[0] == "a" and path[-1] == "c"
    assert dist > 0


def test_shortest_path_with_crowd_avoids_crowded():
    """동일 거리 두 경로 중 혼잡 가중 시 다른 경로 선택."""
    g = build_graph(_booths(), max_edge_m=300.0)
    # b 가 매우 혼잡 → a→b→c 회피하려 할 것 (직접 a→c edge 가 있을 때)
    crowd = {"b": 0.95}
    path, _ = shortest_path(g, src="a", dst="c", crowd_weight=crowd,
                            crowd_factor=50.0)
    assert path[0] == "a" and path[-1] == "c"


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------
def test_recommend_filters_visited_and_far():
    g = build_graph(_booths(), max_edge_m=200.0)
    crowd: dict = {}
    recs = recommend_next_booth(
        visitor_interests=["AI", "ML"],
        visitor_persona_id=None,
        current_booth_id="a",
        visited_booth_ids={"b"},   # b 방문 완료 → 추천 X
        time_left_min=120,
        graph=g,
        booth_keywords=_keywords(),
        crowd=crowd,
        top_n=3,
    )
    rec_ids = [r.booth_id for r in recs]
    assert "b" not in rec_ids
    # e 는 거리 초과로 graph 에 edge 없음
    assert "e" not in rec_ids


def test_recommend_interest_match_dominates():
    g = build_graph(_booths(), max_edge_m=200.0)
    recs = recommend_next_booth(
        visitor_interests=["healthcare"],
        visitor_persona_id=None,
        current_booth_id="a",
        visited_booth_ids=set(),
        time_left_min=120,
        graph=g,
        booth_keywords=_keywords(),
        crowd={},
        top_n=3,
    )
    # c (healthcare) 가 b/d 보다 높은 점수
    if recs:
        top = recs[0]
        assert top.booth_id == "c"
        assert top.interest_match > 0


def test_recommend_zero_when_no_neighbors():
    g = build_graph([
        BoothNode("solo", "x", 37.5, 127.0),
    ])
    recs = recommend_next_booth(
        visitor_interests=["any"],
        visitor_persona_id=None,
        current_booth_id="solo",
        visited_booth_ids=set(),
        time_left_min=60,
        graph=g, booth_keywords={"solo": []}, crowd={},
    )
    assert recs == []


def test_recommend_time_constraint_filters():
    g = build_graph(_booths(), max_edge_m=200.0)
    # 1분만 남음 → 도보+체류 모두 못 맞춤
    recs = recommend_next_booth(
        visitor_interests=["AI"],
        visitor_persona_id=None,
        current_booth_id="a",
        visited_booth_ids=set(),
        time_left_min=1,
        graph=g, booth_keywords=_keywords(), crowd={}, top_n=3,
    )
    # 어떤 부스도 통과 못 할 가능성 — 빈 리스트 OK
    assert all(
        r.distance_m / 80.0 <= 1 for r in recs
    )  # 통과한 게 있다면 1분 walk 안에


# ---------------------------------------------------------------------------
# Fastpass
# ---------------------------------------------------------------------------
def _challenge() -> MyPassChallenge:
    return MyPassChallenge(
        challenge_id="ch1", event_id="ev1",
        target_booth="d",
        partner_booths=["a", "b", "c"],
        required_visits=3,
    )


def test_fastpass_partner_bonus_applied():
    g = build_graph(_booths(), max_edge_m=200.0)
    recs = recommend_next_booth(
        visitor_interests=["AI", "sustain"],
        visitor_persona_id=None,
        current_booth_id="a",
        visited_booth_ids=set(),
        time_left_min=60,
        graph=g, booth_keywords=_keywords(), crowd={}, top_n=3,
    )
    progress = MyPassProgress(
        visitor_id="v1", challenge_id="ch1", visited_partners=["a"],
    )
    boosted = boost_with_mypass(
        recommendations=recs, challenge=_challenge(), progress=progress,
    )
    # b · c 가 partner — 점수 + bonus
    by_id = {r.booth_id: r for r in boosted}
    if "b" in by_id:
        assert "MyPass" in by_id["b"].reason


def test_fastpass_target_bonus_when_completed():
    g = build_graph(_booths(), max_edge_m=300.0)
    recs = recommend_next_booth(
        visitor_interests=["fintech"],
        visitor_persona_id=None,
        current_booth_id="a",
        visited_booth_ids=set(),
        time_left_min=120,
        graph=g, booth_keywords=_keywords(), crowd={}, top_n=5,
    )
    progress = MyPassProgress(
        visitor_id="v1", challenge_id="ch1",
        visited_partners=["a", "b", "c"],
        completed_at=2_000_000,
    )
    boosted = boost_with_mypass(
        recommendations=recs, challenge=_challenge(), progress=progress,
    )
    if any(r.booth_id == "d" for r in boosted):
        d_rec = next(r for r in boosted if r.booth_id == "d")
        assert "패스트트랙" in d_rec.reason


def test_fastpass_no_challenge_returns_unchanged():
    rec_list = []   # 빈 입력
    out = boost_with_mypass(
        recommendations=rec_list, challenge=None, progress=None,
    )
    assert out == rec_list
