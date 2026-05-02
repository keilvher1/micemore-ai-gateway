"""매칭 데이터 거버넌스 — Phase 4 governance 정책의 코드 진실 소스.

원칙:
  - opt-in 안 한 visitor 는 매칭 파이프라인 어디에도 못 들어옴 (DB write 단계 차단).
  - 위치는 백엔드에 도달하기 전 클라가 50m 양자화. 서버는 그리드 셀만 저장.
  - k-anonymity (≥30) 충족 안 되면 외부 노출 차단.
  - push 한도: 부스 10/일, 참가자 5/일.

순수 함수 + 단위 테스트 결정론. 실제 Firestore/Redis 조회는 라우터 단계에서.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

# 환경 가능한 임계값
QUANTIZE_RADIUS_M = float(os.getenv("MATCH_QUANTIZE_RADIUS_M", "50"))
K_ANONYMITY_MIN = int(os.getenv("MATCH_K_ANONYMITY_MIN", "30"))
PUSH_LIMIT_BOOTH_PER_DAY = int(os.getenv("MATCH_PUSH_LIMIT_BOOTH", "10"))
PUSH_LIMIT_VISITOR_PER_DAY = int(os.getenv("MATCH_PUSH_LIMIT_VISITOR", "5"))


@dataclass(frozen=True)
class GridCell:
    """50m 그리드 셀. (lat_idx, lon_idx) 의 정수 인덱스."""
    lat_idx: int
    lon_idx: int

    def to_str(self) -> str:
        return f"{self.lat_idx},{self.lon_idx}"

    @classmethod
    def from_str(cls, s: str) -> "GridCell":
        a, b = s.split(",")
        return cls(int(a), int(b))


def quantize(lat: float, lon: float, radius_m: float = QUANTIZE_RADIUS_M) -> GridCell:
    """GPS → 정수 그리드 셀.

    위도 1도 ≈ 111,000m, 경도 1도는 위도에 따라 다름 (cos(lat) × 111km).
    radius_m 단위로 floor 해서 셀 인덱스 만든다.
    """
    lat_step = radius_m / 111_000.0
    cos_lat = max(0.01, math.cos(math.radians(lat)))
    lon_step = radius_m / (111_000.0 * cos_lat)
    return GridCell(
        lat_idx=int(math.floor(lat / lat_step)),
        lon_idx=int(math.floor(lon / lon_step)),
    )


# ---------------------------------------------------------------------------
# Opt-in
# ---------------------------------------------------------------------------
@dataclass
class Consent:
    """참가자 동의 상태 — 모두 default False."""
    matching: bool = False     # 매칭 푸시 수신
    location: bool = False     # 50m 양자화 위치 사용
    analytics: bool = False    # 익명화 분석 사용


def is_eligible_for_matching(consent: Consent) -> bool:
    """매칭 파이프라인 진입 게이트. matching+location 둘 다 필요."""
    return consent.matching and consent.location


# ---------------------------------------------------------------------------
# k-anonymity — 외부 노출 (트렌드 리포트, B2B API) 직전 검증
# ---------------------------------------------------------------------------
def passes_k_anonymity(group_size: int, min_k: int = K_ANONYMITY_MIN) -> bool:
    return group_size >= min_k


def filter_by_k(groups: dict[str, int], min_k: int = K_ANONYMITY_MIN) -> dict[str, int]:
    """그룹별 카운트 dict → k 미달 키 제거."""
    return {k: v for k, v in groups.items() if v >= min_k}


# ---------------------------------------------------------------------------
# Push 한도
# ---------------------------------------------------------------------------
@dataclass
class PushBudget:
    booth_today: dict[str, int]      # booth_id → 오늘 발송한 횟수
    visitor_today: dict[str, int]    # visitor_hash → 오늘 수신한 횟수


def can_push(
    *, booth_id: str, visitor_hash: str, budget: PushBudget,
) -> tuple[bool, str | None]:
    if budget.booth_today.get(booth_id, 0) >= PUSH_LIMIT_BOOTH_PER_DAY:
        return False, f"booth_limit:{PUSH_LIMIT_BOOTH_PER_DAY}/day"
    if budget.visitor_today.get(visitor_hash, 0) >= PUSH_LIMIT_VISITOR_PER_DAY:
        return False, f"visitor_limit:{PUSH_LIMIT_VISITOR_PER_DAY}/day"
    return True, None


def record_push(*, booth_id: str, visitor_hash: str, budget: PushBudget) -> None:
    budget.booth_today[booth_id] = budget.booth_today.get(booth_id, 0) + 1
    budget.visitor_today[visitor_hash] = (
        budget.visitor_today.get(visitor_hash, 0) + 1
    )


# ---------------------------------------------------------------------------
# Right-to-erasure — 24h SLA 자리. 실제 삭제는 라우터 + Cloud Function.
# ---------------------------------------------------------------------------
@dataclass
class ErasureRequest:
    visitor_hash: str
    requested_at: int
    must_complete_by: int

    @classmethod
    def make(cls, visitor_hash: str, now: int) -> "ErasureRequest":
        return cls(
            visitor_hash=visitor_hash,
            requested_at=now,
            must_complete_by=now + 24 * 3600,
        )

    def is_overdue(self, now: int) -> bool:
        return now > self.must_complete_by
