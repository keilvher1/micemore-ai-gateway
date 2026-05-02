"""5 페르소나 회귀 테스트.

각 페르소나는 Hot/Warm/Cold 3-tier 의 anchor 역할.
가중치 튜닝 시 이 테스트가 가드레일 역할을 한다.
"""
from __future__ import annotations

import pytest

from leads.scorer import VisitorBehavior, score
from leads.narrator import narrate

# (label, kwargs, expected_min, expected_max, level)
PERSONAS = [
    (
        "sarah_hot_buy_now",
        dict(
            booth_dwell_time_sec=420, copilot_questions_count=5,
            copilot_question_topics=["pricing", "pricing", "timeline", "tech", "pricing"],
            pamphlet_downloaded=False, business_card_saved=True,
            translation_session_minutes=4, other_booths_visited=8,
            competitor_booths_visited=2, revisit_count=2,
        ),
        85, 92, "hot",
    ),
    (
        "park_hot_tech_lead",
        dict(
            booth_dwell_time_sec=360, copilot_questions_count=4,
            copilot_question_topics=["tech", "tech", "timeline", "integration"],
            pamphlet_downloaded=True, business_card_saved=True,
            translation_session_minutes=3, other_booths_visited=6,
            competitor_booths_visited=1, revisit_count=1,
        ),
        73, 80, "hot",
    ),
    (
        "mike_warm_curious",
        dict(
            booth_dwell_time_sec=240, copilot_questions_count=2,
            copilot_question_topics=["overview", "timeline"],
            pamphlet_downloaded=True, business_card_saved=False,
            translation_session_minutes=2, other_booths_visited=4,
            competitor_booths_visited=0, revisit_count=1,
        ),
        45, 53, "warm",
    ),
    (
        "yuki_cold_browser",
        dict(
            booth_dwell_time_sec=180, copilot_questions_count=1,
            copilot_question_topics=["overview"],
            pamphlet_downloaded=True, business_card_saved=False,
            translation_session_minutes=0, other_booths_visited=10,
            competitor_booths_visited=1, revisit_count=0,
        ),
        28, 36, "cold",
    ),
    (
        "drift_cold_minimal",
        dict(
            booth_dwell_time_sec=60, copilot_questions_count=0,
            copilot_question_topics=[],
            pamphlet_downloaded=False, business_card_saved=False,
            translation_session_minutes=0, other_booths_visited=20,
            competitor_booths_visited=0, revisit_count=0,
        ),
        0, 10, "cold",
    ),
]


@pytest.mark.parametrize("name,kw,lo,hi,expected_level", PERSONAS)
def test_persona_score_within_band(name, kw, lo, hi, expected_level):
    b = VisitorBehavior(visitor_id=f"v_{name}", booth_id="lumen", **kw)
    bd, level = score(b)
    assert lo <= bd.total <= hi, (
        f"{name}: total={bd.total} not in [{lo},{hi}]; bd={bd}"
    )
    assert level == expected_level, f"{name}: level={level}, expected {expected_level}"


def test_score_caps_at_100():
    """모든 신호 max 일 때도 100 을 넘지 않음."""
    b = VisitorBehavior(
        visitor_id="v_max", booth_id="lumen",
        booth_dwell_time_sec=10_000,
        copilot_questions_count=100,
        copilot_question_topics=["pricing"] * 5,
        pamphlet_downloaded=True, business_card_saved=True,
        translation_session_minutes=600,
        other_booths_visited=0,
        competitor_booths_visited=10,
        revisit_count=20,
    )
    bd, level = score(b)
    assert bd.total == 100
    assert level == "hot"


def test_empty_behavior_is_zero_cold():
    b = VisitorBehavior(visitor_id="v_empty", booth_id="lumen")
    bd, level = score(b)
    assert bd.total == 0
    assert level == "cold"


def test_pricing_bonus_only_when_topic_present():
    base = dict(
        visitor_id="v_a", booth_id="lumen",
        booth_dwell_time_sec=120, copilot_questions_count=1,
        copilot_question_topics=["overview"],
    )
    bd1, _ = score(VisitorBehavior(**base))
    base["copilot_question_topics"] = ["pricing"]
    bd2, _ = score(VisitorBehavior(**base))
    assert bd2.pricing_bonus == 5
    assert bd1.pricing_bonus == 0
    assert bd2.total - bd1.total == 5


def test_narrator_mock_format():
    b = VisitorBehavior(
        visitor_id="v_demo", booth_id="lumen",
        booth_dwell_time_sec=420, copilot_questions_count=5,
        copilot_question_topics=["pricing"],
        business_card_saved=True, translation_session_minutes=4,
        competitor_booths_visited=1, revisit_count=2,
    )
    bd, lv = score(b)
    text = narrate(b, bd, lv, mock=True)
    assert "Hot Lead" in text
    assert f"{bd.total}/100" in text
    assert "체류" in text
    assert "명함 저장" in text
