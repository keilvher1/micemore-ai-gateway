"""ml/features.py + ml/predict.py 의 mock 경로 검증."""
from __future__ import annotations

import os

os.environ["USE_MOCK"] = "true"

from ml.features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    behavior_to_features,
    behaviors_to_records,
    hash_visitor_id,
)
from ml.predict import predict  # noqa: E402


def test_feature_names_count_matches_vector():
    b = {
        "booth_dwell_time_sec": 420,
        "copilot_questions_count": 5,
        "copilot_question_topics": ["pricing", "tech"],
        "pamphlet_downloaded": True,
        "business_card_saved": True,
        "translation_session_minutes": 4,
        "other_booths_visited": 8,
        "competitor_booths_visited": 2,
        "revisit_count": 2,
        "first_visit_hour": 10,
        "booth_visit_order": 3,
        "avg_question_length": 24,
        "question_complexity": 0.7,
    }
    feats = behavior_to_features(b)
    assert len(feats) == len(FEATURE_NAMES)
    assert feats[0] == 7.0  # 420s = 7분


def test_topic_ratio_pricing_tech():
    b = {"copilot_question_topics": ["pricing", "pricing", "tech", "overview"]}
    feats = behavior_to_features(b)
    pricing = feats[FEATURE_NAMES.index("pricing_topic_ratio")]
    tech = feats[FEATURE_NAMES.index("tech_topic_ratio")]
    assert pricing == 0.5
    assert tech == 0.25


def test_missing_fields_default_safely():
    feats = behavior_to_features({})
    assert len(feats) == len(FEATURE_NAMES)
    # first_visit_hour default 12
    assert feats[FEATURE_NAMES.index("first_visit_hour")] == 12


def test_hash_visitor_id_deterministic():
    a = hash_visitor_id("v_alice")
    b = hash_visitor_id("v_alice")
    c = hash_visitor_id("v_bob")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_behaviors_to_records_label_propagates():
    rows = [
        {"visitor_id": "v1", "booth_id": "lumen",
         "booth_dwell_time_sec": 300, "converted": 1},
        {"visitor_id": "v2", "booth_id": "lumen",
         "booth_dwell_time_sec": 60, "converted": 0},
    ]
    recs = behaviors_to_records(rows, event_id="ev_test")
    assert len(recs) == 2
    assert recs[0].label == 1
    assert recs[1].label == 0
    assert recs[0].feature_version == FEATURE_VERSION
    assert recs[0].visitor_hash != "v1"  # 익명화


def test_predict_mock_falls_back_to_rule():
    p = predict({
        "visitor_id": "v_test",
        "booth_id": "lumen",
        "booth_dwell_time_sec": 420,
        "copilot_questions_count": 5,
        "copilot_question_topics": ["pricing"],
        "business_card_saved": True,
        "translation_session_minutes": 4,
        "competitor_booths_visited": 2,
        "revisit_count": 2,
    })
    assert p.source == "mock"
    assert 0 <= p.score <= 100
    assert p.level in {"hot", "warm", "cold"}
    assert p.feature_version == FEATURE_VERSION


def test_predict_empty_behavior_returns_cold():
    p = predict({"visitor_id": "v_empty", "booth_id": "lumen"})
    assert p.score == 0
    assert p.level == "cold"
