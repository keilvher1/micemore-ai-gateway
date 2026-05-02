"""matching/{governance,icp_embedder,visitor_profiler,matcher,push_dispatcher}."""
from __future__ import annotations

import os

os.environ["USE_MOCK"] = "true"

import pytest  # noqa: E402

from matching.governance import (  # noqa: E402
    Consent,
    GridCell,
    PushBudget,
    can_push,
    filter_by_k,
    is_eligible_for_matching,
    passes_k_anonymity,
    quantize,
    record_push,
)
from matching.icp_embedder import embed_icp  # noqa: E402
from matching.matcher import cosine, match, time_bonus  # noqa: E402
from matching.push_dispatcher import dispatch  # noqa: E402
from matching.visitor_profiler import (  # noqa: E402
    ConsentDeniedError,
    profile_visitor,
)


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------
def test_quantize_50m_is_deterministic():
    a = quantize(37.5172, 127.0473)
    b = quantize(37.5172, 127.0473)
    assert a == b
    assert isinstance(a, GridCell)


def test_quantize_nearby_points_share_cell():
    a = quantize(37.51720, 127.04730)
    b = quantize(37.51725, 127.04735)  # ~5m
    assert a == b


def test_quantize_far_points_different_cell():
    a = quantize(37.51720, 127.04730)
    b = quantize(37.51810, 127.04730)  # ~100m 북쪽
    assert a != b


def test_eligible_requires_both_consents():
    assert is_eligible_for_matching(Consent(matching=True, location=True))
    assert not is_eligible_for_matching(Consent(matching=True, location=False))
    assert not is_eligible_for_matching(Consent(matching=False, location=True))
    assert not is_eligible_for_matching(Consent())


def test_k_anonymity_threshold():
    assert passes_k_anonymity(30)
    assert passes_k_anonymity(31)
    assert not passes_k_anonymity(29)


def test_filter_by_k_drops_small_groups():
    groups = {"a": 50, "b": 10, "c": 30}
    out = filter_by_k(groups, min_k=30)
    assert "a" in out and "c" in out and "b" not in out


def test_push_budget_blocks_after_limit():
    budget = PushBudget(booth_today={"b1": 10}, visitor_today={})
    ok, reason = can_push(booth_id="b1", visitor_hash="vh", budget=budget)
    assert ok is False
    assert "booth_limit" in reason


def test_push_budget_visitor_limit():
    budget = PushBudget(booth_today={}, visitor_today={"vh": 5})
    ok, reason = can_push(booth_id="b1", visitor_hash="vh", budget=budget)
    assert ok is False
    assert "visitor_limit" in reason


def test_push_budget_record_increments():
    budget = PushBudget(booth_today={}, visitor_today={})
    record_push(booth_id="b1", visitor_hash="vh", budget=budget)
    record_push(booth_id="b1", visitor_hash="vh", budget=budget)
    assert budget.booth_today["b1"] == 2
    assert budget.visitor_today["vh"] == 2


# ---------------------------------------------------------------------------
# ICP embedder (mock)
# ---------------------------------------------------------------------------
def test_icp_extract_industries_and_roles():
    icp = embed_icp(
        booth_id="lumen",
        raw_text="바이오테크 R&D 책임자, 100~1000명 회사, mass spec 또는 NMR 검토 중",
    )
    assert icp.model == "mock"
    assert "biotech" in icp.target_industries
    assert "rd" in icp.target_roles
    assert icp.target_company_size == "50-500"
    assert len(icp.embedding) == 1536


def test_icp_deterministic_same_text():
    a = embed_icp(booth_id="b", raw_text="AI ML CTO 검토")
    b = embed_icp(booth_id="b", raw_text="AI ML CTO 검토")
    assert a.embedding == b.embedding


def test_icp_keywords_extracted():
    icp = embed_icp(
        booth_id="b", raw_text="Samsung NVIDIA partnership for ML inference"
    )
    assert "Samsung" in icp.target_keywords
    assert "NVIDIA" in icp.target_keywords


# ---------------------------------------------------------------------------
# Visitor profiler (consent gate)
# ---------------------------------------------------------------------------
def test_profile_blocked_without_consent():
    with pytest.raises(ConsentDeniedError):
        profile_visitor(
            visitor_hash="vh", role="CTO", industry="biotech",
            interests=["mass spec"],
            consent=Consent(matching=True, location=False),
        )


def test_profile_pii_stripped_from_interests():
    p = profile_visitor(
        visitor_hash="vh", role="CTO", industry="ai",
        interests=["LLM eval contact me at dev@example.com"],
        consent=Consent(matching=True, location=True),
    )
    assert "dev@example.com" not in p.interests[0]
    assert "[email]" in p.interests[0]


def test_profile_records_consent_snapshot():
    p = profile_visitor(
        visitor_hash="vh", role="VP", industry="biotech",
        interests=["NMR"],
        consent=Consent(matching=True, location=True, analytics=True),
    )
    assert p.consent_snapshot["matching"] is True
    assert p.consent_snapshot["analytics"] is True


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
def test_cosine_identical_is_one():
    a = [1.0, 0.0, 0.5]
    assert abs(cosine(a, a) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine(a, b)) < 1e-9


def test_match_filters_below_threshold_and_far():
    icp = [1.0, 0.0]
    candidates = [
        ("v1", [1.0, 0.0], 10.0),    # cosine=1, 가까움 → 통과
        ("v2", [0.5, 0.5], 50.0),    # cosine~0.7
        ("v3", [-1.0, 0.0], 5.0),    # cosine=-1 → 컷
        ("v4", [1.0, 0.0], 200.0),   # 거리 초과 → 컷
    ]
    events = match(
        icp_embedding=icp, booth_id="b1", candidates=candidates,
        now_epoch=int(__import__("time").mktime((2026, 6, 19, 11, 0, 0, 0, 0, 0))),
        score_threshold=0.4, top_k=3,
    )
    ids = [e.visitor_hash for e in events]
    assert "v1" in ids
    assert "v3" not in ids
    assert "v4" not in ids


def test_time_bonus_morning_full():
    import time as t
    epoch = int(t.mktime((2026, 6, 19, 11, 0, 0, 0, 0, 0)))
    assert time_bonus(epoch) == 1.0


def test_time_bonus_late_night_half():
    import time as t
    epoch = int(t.mktime((2026, 6, 19, 22, 0, 0, 0, 0, 0)))
    assert time_bonus(epoch) == 0.5


# ---------------------------------------------------------------------------
# Push dispatcher (mock)
# ---------------------------------------------------------------------------
def test_dispatch_mock_sends_when_token_present():
    from matching.matcher import MatchEvent
    ev = MatchEvent(
        booth_id="b1", visitor_hash="vh", score=0.8,
        cosine=0.9, distance_m=20, time_bonus=1.0,
        reason="관심사 일치, 20m", triggered_at=0,
    )
    budget = PushBudget(booth_today={}, visitor_today={})
    visitor_res, booth_res = dispatch(
        event=ev, budget=budget,
        visitor_token="tok_v", booth_operator_token="tok_b",
    )
    assert visitor_res.accepted is True
    assert booth_res.accepted is True
    assert budget.booth_today["b1"] == 1


def test_dispatch_blocks_when_over_budget():
    from matching.matcher import MatchEvent
    ev = MatchEvent(
        booth_id="b1", visitor_hash="vh", score=0.8,
        cosine=0.9, distance_m=20, time_bonus=1.0,
        reason="r", triggered_at=0,
    )
    # 부스 한도 이미 도달
    budget = PushBudget(booth_today={"b1": 10}, visitor_today={})
    v, b = dispatch(
        event=ev, budget=budget,
        visitor_token="tok_v", booth_operator_token="tok_b",
    )
    assert v.accepted is False
    assert "booth_limit" in (v.reason or "")


def test_dispatch_no_token_returns_not_accepted():
    from matching.matcher import MatchEvent
    ev = MatchEvent(
        booth_id="b1", visitor_hash="vh", score=0.8,
        cosine=0.9, distance_m=20, time_bonus=1.0,
        reason="r", triggered_at=0,
    )
    budget = PushBudget(booth_today={}, visitor_today={})
    v, b = dispatch(event=ev, budget=budget,
                    visitor_token=None, booth_operator_token=None)
    assert v.accepted is False
    assert b.accepted is False
