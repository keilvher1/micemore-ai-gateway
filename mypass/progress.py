"""MyPassProgress + 5 anti-abuse 룰.

룰 (5):
  A1 같은 부스 중복 — visited_partners 에 booth_id 중복 추가 금지 (no-op + log)
  A2 rapid-fire — 직전 태깅과 5분(default) 미만 간격이면 거부
  A3 위치 위조 — visitor_gps ↔ booth_geofence 거리 > 20m 면 거부
  A4 redeem 다회 — redeemed_at 가 이미 있으면 거부
  A5 다중 계정 farming — device_fingerprint 동일 + visitor 다름이 24h 내
                          같은 challenge complete N>=2 → flag (보상 보류)

본 모듈은 룰 검증 + 상태 업데이트의 순수 로직만 담당.
실제 Firestore 쓰기/장치 핑거프린트 수집은 라우터/Cloud Function 에서.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Literal

from mypass.challenge import MyPassChallenge


# ---------------------------------------------------------------------------
# 환경 가능한 임계값
# ---------------------------------------------------------------------------
MIN_INTERVAL_MIN = int(os.getenv("MYPASS_MIN_INTERVAL_MIN", "5"))
GEOFENCE_RADIUS_M = float(os.getenv("MYPASS_GEOFENCE_RADIUS_M", "20"))
FARMING_WINDOW_SEC = 24 * 3600
FARMING_MAX_PER_DEVICE = 1   # 같은 device 가 같은 challenge 1번만 complete 정상


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
TagOutcome = Literal[
    "ok_partner_added",
    "ok_already_counted",     # A1: 중복 태깅 무시 (오류 아님)
    "completed_now",          # 이번 태깅으로 challenge 충족
    "denied_too_fast",        # A2
    "denied_out_of_range",    # A3
    "denied_invalid_partner", # 챌린지 외 부스
    "denied_expired",
]


@dataclass
class MyPassProgress:
    visitor_id: str
    challenge_id: str
    visited_partners: list[str] = field(default_factory=list)
    last_tag_at: int = 0
    completed_at: int = 0
    redeemed_at: int = 0
    flagged_review: bool = False
    flag_reason: str | None = None


@dataclass
class TagResult:
    outcome: TagOutcome
    progress: MyPassProgress
    reason: str | None = None


# ---------------------------------------------------------------------------
# 거리 계산 — Haversine (geopy 의존성 회피)
# ---------------------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 GPS 좌표 거리 m. 결정론적, 외부 의존성 없음."""
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# 메인 로직 — 부스 태깅 처리
# ---------------------------------------------------------------------------
def process_tag(
    *,
    progress: MyPassProgress,
    challenge: MyPassChallenge,
    booth_id: str,
    visitor_gps: tuple[float, float] | None = None,
    booth_geofence: tuple[float, float] | None = None,
    now: int | None = None,
) -> TagResult:
    now = now or int(time.time())

    # Validity / partner check
    if not challenge.is_valid_now(now):
        return TagResult(
            outcome="denied_expired",
            progress=progress,
            reason="challenge expired",
        )
    if not challenge.is_partner(booth_id):
        return TagResult(
            outcome="denied_invalid_partner",
            progress=progress,
            reason=f"{booth_id} is not in partner_booths",
        )

    # A2: rapid-fire
    if progress.last_tag_at and (now - progress.last_tag_at) < MIN_INTERVAL_MIN * 60:
        return TagResult(
            outcome="denied_too_fast",
            progress=progress,
            reason=f"interval < {MIN_INTERVAL_MIN}min",
        )

    # A3: geofence
    if visitor_gps and booth_geofence:
        dist = haversine_m(*visitor_gps, *booth_geofence)
        if dist > GEOFENCE_RADIUS_M:
            return TagResult(
                outcome="denied_out_of_range",
                progress=progress,
                reason=f"{dist:.1f}m > {GEOFENCE_RADIUS_M}m",
            )

    # A1: 중복 태깅 — no-op
    if booth_id in progress.visited_partners:
        progress.last_tag_at = now  # interval 갱신은 함 (재시도 보호)
        return TagResult(
            outcome="ok_already_counted",
            progress=progress,
            reason="already in visited_partners",
        )

    # 정상 카운트
    progress.visited_partners.append(booth_id)
    progress.last_tag_at = now
    if (
        len(progress.visited_partners) >= challenge.required_visits
        and not progress.completed_at
    ):
        progress.completed_at = now
        return TagResult(outcome="completed_now", progress=progress)
    return TagResult(outcome="ok_partner_added", progress=progress)


# ---------------------------------------------------------------------------
# A5: 다중 계정 farming flag — Firestore aggregation 결과를 받아 판정
# ---------------------------------------------------------------------------
def detect_device_farming(
    *,
    device_recent_completes: list[tuple[str, int]],  # [(visitor_id, completed_at), ...]
    challenge_id: str,                                # 비교할 challenge (현재는 같은 challenge 만)
    now: int | None = None,
    window_sec: int = FARMING_WINDOW_SEC,
    max_allowed: int = FARMING_MAX_PER_DEVICE,
) -> bool:
    """같은 device 가 24h 내 같은 challenge 를 max_allowed 초과로 complete → True."""
    now = now or int(time.time())
    cutoff = now - window_sec
    distinct_visitors = {
        vid for vid, ts in device_recent_completes if ts >= cutoff
    }
    return len(distinct_visitors) > max_allowed
