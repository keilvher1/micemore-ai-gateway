"""MyPass 챌린지 정의.

운영자가 행사 시작 전에 등록:
  - target_booth: 보상 부스 (인기 부스, 가입비 지불자)
  - partner_booths: 거쳐야 하는 제휴 부스 N개 (트래픽 받음)
  - reward_type: fast_track | discount | gift
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


RewardType = Literal["fast_track", "discount", "gift"]


@dataclass
class MyPassChallenge:
    challenge_id: str
    event_id: str
    target_booth: str
    partner_booths: list[str]
    required_visits: int = 3
    reward_type: RewardType = "fast_track"
    valid_until: int = 0           # epoch sec, 0 = 무제한
    created_at: int = field(default_factory=lambda: int(time.time()))

    def is_valid_now(self, now: int | None = None) -> bool:
        if self.valid_until == 0:
            return True
        return (now or int(time.time())) <= self.valid_until

    def is_partner(self, booth_id: str) -> bool:
        return booth_id in self.partner_booths

    def validate(self) -> list[str]:
        """필수 룰 — 챌린지 등록 시 1회 검증."""
        errors: list[str] = []
        if not self.challenge_id:
            errors.append("challenge_id required")
        if not self.event_id:
            errors.append("event_id required")
        if not self.target_booth:
            errors.append("target_booth required")
        if len(self.partner_booths) < self.required_visits:
            errors.append(
                f"partner_booths({len(self.partner_booths)}) < required({self.required_visits})"
            )
        if self.target_booth in self.partner_booths:
            errors.append("target_booth must NOT be in partner_booths")
        if len(set(self.partner_booths)) != len(self.partner_booths):
            errors.append("partner_booths contains duplicates")
        if self.required_visits < 1 or self.required_visits > 10:
            errors.append("required_visits must be 1..10")
        return errors
