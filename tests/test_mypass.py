"""MyPass 챌린지 + 5 anti-abuse 시나리오 회귀 테스트.

A1 같은 부스 중복     : test_a1_duplicate_tag_ignored
A2 rapid-fire         : test_a2_rapid_fire_denied
A3 위치 위조           : test_a3_geofence_violation_denied
A4 redeem 다회         : test_a4_redeem_only_once
A5 다중 계정 farming   : test_a5_device_farming_flag
"""
from __future__ import annotations

import os
import time

os.environ["USE_MOCK"] = "true"

import pytest  # noqa: E402

from mypass.challenge import MyPassChallenge  # noqa: E402
from mypass.progress import (  # noqa: E402
    GEOFENCE_RADIUS_M,
    MIN_INTERVAL_MIN,
    MyPassProgress,
    detect_device_farming,
    haversine_m,
    process_tag,
)
from mypass.redeem import redeem  # noqa: E402


def _ch() -> MyPassChallenge:
    return MyPassChallenge(
        challenge_id="ch_demo",
        event_id="ev_2026_06",
        target_booth="popular",
        partner_booths=["partnerA", "partnerB", "partnerC"],
        required_visits=3,
    )


def _p(visitor: str = "v_alice", **kw) -> MyPassProgress:
    return MyPassProgress(visitor_id=visitor, challenge_id="ch_demo", **kw)


# ---------------------------------------------------------------------------
# Challenge 검증
# ---------------------------------------------------------------------------
def test_challenge_validate_happy_path():
    assert _ch().validate() == []


def test_challenge_target_in_partners_rejected():
    bad = MyPassChallenge(
        challenge_id="bad", event_id="ev", target_booth="A",
        partner_booths=["A", "B", "C"],
    )
    errors = bad.validate()
    assert any("target_booth must NOT" in e for e in errors)


def test_challenge_too_few_partners():
    bad = MyPassChallenge(
        challenge_id="bad", event_id="ev", target_booth="X",
        partner_booths=["A", "B"], required_visits=3,
    )
    errors = bad.validate()
    assert any("partner_booths(2)" in e for e in errors)


# ---------------------------------------------------------------------------
# A1 — 같은 부스 중복 카운트 무시
# ---------------------------------------------------------------------------
def test_a1_duplicate_tag_ignored():
    p = _p()
    ch = _ch()
    now = 1_000_000

    r1 = process_tag(progress=p, challenge=ch, booth_id="partnerA", now=now)
    assert r1.outcome == "ok_partner_added"
    assert p.visited_partners == ["partnerA"]

    # 같은 부스 5분+1초 후 재태깅 — A2 통과지만 A1 으로 무시
    r2 = process_tag(progress=p, challenge=ch, booth_id="partnerA",
                     now=now + MIN_INTERVAL_MIN * 60 + 1)
    assert r2.outcome == "ok_already_counted"
    assert p.visited_partners == ["partnerA"]  # 그대로


# ---------------------------------------------------------------------------
# A2 — rapid-fire (5분 미만 간격) 거부
# ---------------------------------------------------------------------------
def test_a2_rapid_fire_denied():
    p = _p()
    ch = _ch()
    now = 1_000_000
    process_tag(progress=p, challenge=ch, booth_id="partnerA", now=now)
    # 60초 후 — 5분 미만
    r = process_tag(progress=p, challenge=ch, booth_id="partnerB", now=now + 60)
    assert r.outcome == "denied_too_fast"
    assert p.visited_partners == ["partnerA"]  # B 추가 X


def test_a2_rapid_fire_allowed_after_interval():
    p = _p()
    ch = _ch()
    now = 1_000_000
    process_tag(progress=p, challenge=ch, booth_id="partnerA", now=now)
    r = process_tag(progress=p, challenge=ch, booth_id="partnerB",
                    now=now + MIN_INTERVAL_MIN * 60 + 5)
    assert r.outcome == "ok_partner_added"


# ---------------------------------------------------------------------------
# A3 — Geofence 위반 (20m 초과)
# ---------------------------------------------------------------------------
def test_a3_geofence_violation_denied():
    p = _p()
    ch = _ch()
    booth_geofence = (37.5172, 127.0473)  # 코엑스 근처
    visitor_far = (37.5180, 127.0473)     # 약 89m 북쪽
    r = process_tag(
        progress=p, challenge=ch, booth_id="partnerA",
        visitor_gps=visitor_far, booth_geofence=booth_geofence,
    )
    assert r.outcome == "denied_out_of_range"
    assert p.visited_partners == []


def test_a3_geofence_within_range_allowed():
    p = _p()
    ch = _ch()
    booth_geofence = (37.5172, 127.0473)
    visitor_close = (37.51721, 127.04731)  # 1m 정도 차이
    r = process_tag(
        progress=p, challenge=ch, booth_id="partnerA",
        visitor_gps=visitor_close, booth_geofence=booth_geofence,
    )
    assert r.outcome == "ok_partner_added"


def test_haversine_known_distance():
    # 서울 시청 (37.5663, 126.9779) → 광화문 (37.5759, 126.9769) ≈ 1070m
    d = haversine_m(37.5663, 126.9779, 37.5759, 126.9769)
    assert 1000 < d < 1200


# ---------------------------------------------------------------------------
# A4 — Redeem 1회 제한
# ---------------------------------------------------------------------------
def test_a4_redeem_only_once():
    # 챌린지 완료 상태로 시작
    p = _p(visited_partners=["partnerA", "partnerB", "partnerC"],
           completed_at=2_000_000)
    r1 = redeem(progress=p, target_booth="popular",
                booth_at_redeem="popular", now=2_000_500)
    assert r1.outcome == "ok"
    assert p.redeemed_at == 2_000_500

    r2 = redeem(progress=p, target_booth="popular",
                booth_at_redeem="popular", now=2_000_900)
    assert r2.outcome == "denied_already_redeemed"


def test_a4_redeem_blocks_when_not_completed():
    p = _p(visited_partners=["partnerA"])  # 아직 1/3
    r = redeem(progress=p, target_booth="popular",
               booth_at_redeem="popular")
    assert r.outcome == "denied_not_completed"


def test_a4_redeem_blocks_when_flagged():
    p = _p(visited_partners=["partnerA", "partnerB", "partnerC"],
           completed_at=2_000_000, flagged_review=True,
           flag_reason="device farming")
    r = redeem(progress=p, target_booth="popular",
               booth_at_redeem="popular")
    assert r.outcome == "denied_flagged"


# ---------------------------------------------------------------------------
# A5 — 다중 계정 farming
# ---------------------------------------------------------------------------
def test_a5_device_farming_flag():
    now = 3_000_000
    completes = [
        ("v_alice", now - 3600),     # 1h 전
        ("v_bob",   now - 7200),     # 2h 전 (다른 visitor!)
        ("v_carol", now - 10000),    # 3h 전
    ]
    flagged = detect_device_farming(
        device_recent_completes=completes,
        challenge_id="ch_demo",
        now=now,
    )
    assert flagged is True


def test_a5_single_visitor_per_device_ok():
    now = 3_000_000
    completes = [("v_alice", now - 3600)]
    flagged = detect_device_farming(
        device_recent_completes=completes,
        challenge_id="ch_demo",
        now=now,
    )
    assert flagged is False


def test_a5_old_completes_outside_window_ignored():
    now = 3_000_000
    completes = [
        ("v_alice", now - 3600),
        ("v_bob", now - 30 * 24 * 3600),  # 30일 전
    ]
    flagged = detect_device_farming(
        device_recent_completes=completes,
        challenge_id="ch_demo",
        now=now,
    )
    # 24h 윈도 밖이라 alice 만 카운트 → flag X
    assert flagged is False


# ---------------------------------------------------------------------------
# 정상 완주 시나리오
# ---------------------------------------------------------------------------
def test_complete_challenge_with_3_partner_visits():
    p = _p()
    ch = _ch()
    interval = (MIN_INTERVAL_MIN * 60) + 1
    now = 1_000_000
    process_tag(progress=p, challenge=ch, booth_id="partnerA", now=now)
    process_tag(progress=p, challenge=ch, booth_id="partnerB",
                now=now + interval)
    r = process_tag(progress=p, challenge=ch, booth_id="partnerC",
                    now=now + interval * 2)
    assert r.outcome == "completed_now"
    assert p.completed_at > 0
    assert sorted(p.visited_partners) == ["partnerA", "partnerB", "partnerC"]


def test_invalid_partner_booth_denied():
    p = _p()
    ch = _ch()
    r = process_tag(progress=p, challenge=ch, booth_id="random_booth")
    assert r.outcome == "denied_invalid_partner"


def test_expired_challenge_denied():
    p = _p()
    ch = MyPassChallenge(
        challenge_id="ch_old", event_id="ev",
        target_booth="popular",
        partner_booths=["A", "B", "C"],
        valid_until=1000,  # 한참 전
    )
    r = process_tag(progress=p, challenge=ch, booth_id="A", now=2_000_000)
    assert r.outcome == "denied_expired"
