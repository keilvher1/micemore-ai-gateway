"""MyPass 챌린지 라우터.

POST /mypass/challenges               : 챌린지 등록 (운영자)
POST /mypass/tag                      : 부스 태깅 처리 (참가자 NFC/QR)
POST /mypass/redeem                   : 보상 사용 (target 부스 입구)
GET  /mypass/healthz

Phase 3 v0 — Firestore CRUD 는 라우터 단계에서 직접 처리 X (Cloud Function /
Flutter Firestore SDK 가 담당). 본 라우터는 룰 검증의 단일 진실 소스.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mypass.challenge import MyPassChallenge
from mypass.progress import (
    MyPassProgress,
    TagResult,
    detect_device_farming,
    process_tag,
)
from mypass.redeem import RedeemResult, redeem

router = APIRouter(prefix="/mypass", tags=["mypass"])


# ---------------------------------------------------------------------------
# Challenge validation
# ---------------------------------------------------------------------------
class ChallengeIn(BaseModel):
    challenge_id: str
    event_id: str
    target_booth: str
    partner_booths: list[str] = Field(min_length=1, max_length=20)
    required_visits: int = 3
    reward_type: str = "fast_track"
    valid_until: int = 0


@router.post("/challenges/validate", response_model=dict)
async def validate_challenge(req: ChallengeIn):
    ch = MyPassChallenge(**req.model_dump())  # type: ignore[arg-type]
    errors = ch.validate()
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    return {"ok": True, "challenge_id": ch.challenge_id}


# ---------------------------------------------------------------------------
# Tag processing
# ---------------------------------------------------------------------------
class ProgressIn(BaseModel):
    visitor_id: str
    challenge_id: str
    visited_partners: list[str] = Field(default_factory=list)
    last_tag_at: int = 0
    completed_at: int = 0
    redeemed_at: int = 0
    flagged_review: bool = False
    flag_reason: str | None = None


class TagIn(BaseModel):
    progress: ProgressIn
    challenge: ChallengeIn
    booth_id: str
    visitor_gps: tuple[float, float] | None = None
    booth_geofence: tuple[float, float] | None = None


class TagOut(BaseModel):
    outcome: str
    progress: ProgressIn
    reason: str | None = None


@router.post("/tag", response_model=TagOut)
async def tag_route(req: TagIn):
    progress = MyPassProgress(**req.progress.model_dump())
    challenge = MyPassChallenge(**req.challenge.model_dump())  # type: ignore[arg-type]
    res: TagResult = process_tag(
        progress=progress,
        challenge=challenge,
        booth_id=req.booth_id,
        visitor_gps=req.visitor_gps,
        booth_geofence=req.booth_geofence,
    )
    return TagOut(
        outcome=res.outcome,
        progress=ProgressIn(**res.progress.__dict__),
        reason=res.reason,
    )


# ---------------------------------------------------------------------------
# Redeem
# ---------------------------------------------------------------------------
class RedeemIn(BaseModel):
    progress: ProgressIn
    target_booth: str
    booth_at_redeem: str


@router.post("/redeem", response_model=TagOut)
async def redeem_route(req: RedeemIn):
    progress = MyPassProgress(**req.progress.model_dump())
    res: RedeemResult = redeem(
        progress=progress,
        target_booth=req.target_booth,
        booth_at_redeem=req.booth_at_redeem,
    )
    return TagOut(
        outcome=res.outcome,
        progress=ProgressIn(**res.progress.__dict__),
        reason=res.reason,
    )


# ---------------------------------------------------------------------------
# Anti-abuse — device farming check
# ---------------------------------------------------------------------------
class FarmingCheckIn(BaseModel):
    challenge_id: str
    device_recent_completes: list[tuple[str, int]]


@router.post("/anti-abuse/farming")
async def farming_route(req: FarmingCheckIn):
    flagged = detect_device_farming(
        device_recent_completes=req.device_recent_completes,
        challenge_id=req.challenge_id,
    )
    return {"flagged": flagged}


@router.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "module": "mypass",
            "mock": os.getenv("USE_MOCK", "false").lower() == "true"}


# ---------------------------------------------------------------------------
# D-4 단계 5 — POST /mypass/redemptions
# 참가자 토큰 사용 시 호출. Geofence (기본 20m) + 5min 간격 검증.
# 통과 시 mypass_redemptions 컬렉션에 영속 기록 (사용자별 정산용).
# ---------------------------------------------------------------------------
import math
import time
from datetime import datetime, timezone

GEOFENCE_RADIUS_M = float(os.getenv("MYPASS_GEOFENCE_RADIUS_M", "20"))
MIN_INTERVAL_MIN = int(os.getenv("MYPASS_MIN_INTERVAL_MIN", "5"))


def _haversine_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Approx 거리 (m). lat1,lon1 / lat2,lon2."""
    r = 6371000
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class RedemptionIn(BaseModel):
    user_id: str
    challenge_id: str
    booth_id: str
    visitor_gps: tuple[float, float] = Field(..., description="(lat, lon)")
    booth_gps: tuple[float, float] = Field(..., description="(lat, lon)")
    last_redeemed_at: int = Field(default=0, description="동일 챌린지 마지막 redeem unix ms")
    radius_m_override: float | None = None


class RedemptionOut(BaseModel):
    accepted: bool
    redemption_id: str | None = None
    reason: str | None = None
    distance_m: float
    cooldown_seconds_left: int = 0


@router.post("/redemptions", response_model=RedemptionOut)
async def create_redemption(req: RedemptionIn):
    radius = req.radius_m_override or GEOFENCE_RADIUS_M
    distance = _haversine_m(req.visitor_gps, req.booth_gps)
    now_ms = int(time.time() * 1000)

    # 1) Geofence 검증
    if distance > radius:
        return RedemptionOut(
            accepted=False,
            reason=f"out_of_geofence (distance={distance:.1f}m, allowed={radius:.1f}m)",
            distance_m=distance,
        )

    # 2) Cooldown 검증 (5 분)
    elapsed = (now_ms - req.last_redeemed_at) / 1000 if req.last_redeemed_at else 1e9
    cooldown = max(0, MIN_INTERVAL_MIN * 60 - int(elapsed))
    if cooldown > 0:
        return RedemptionOut(
            accepted=False,
            reason=f"cooldown_active ({cooldown}s left)",
            distance_m=distance,
            cooldown_seconds_left=cooldown,
        )

    # 3) Firestore mypass_redemptions 영속 (firebase-admin 가능 시)
    redemption_id = f"RDM-{int(now_ms)}-{req.user_id[-6:]}"
    try:
        from firebase_admin import firestore, initialize_app  # type: ignore

        try:
            initialize_app()
        except ValueError:
            pass
        firestore.client().collection("mypass_redemptions").document(redemption_id).set(
            {
                "user_id": req.user_id,
                "challenge_id": req.challenge_id,
                "booth_id": req.booth_id,
                "redeemed_at": datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).isoformat(),
                "distance_m": distance,
                "geofence_radius_m": radius,
            }
        )
    except ImportError:
        pass  # mock — firebase-admin 미설치
    except Exception as exc:  # noqa: BLE001
        return RedemptionOut(
            accepted=False,
            reason=f"firestore_write_failed: {exc.__class__.__name__}",
            distance_m=distance,
        )

    return RedemptionOut(accepted=True, redemption_id=redemption_id, distance_m=distance)
