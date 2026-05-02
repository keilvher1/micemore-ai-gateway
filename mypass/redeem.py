"""보상 사용 (redeem) — 1회 제한.

Phase 3 Sprint 6: target booth 입구에서 redeem token 검증 후 패스트트랙.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from mypass.progress import MyPassProgress

RedeemOutcome = Literal[
    "ok",
    "denied_not_completed",
    "denied_already_redeemed",
    "denied_flagged",
]


@dataclass
class RedeemResult:
    outcome: RedeemOutcome
    progress: MyPassProgress
    reason: str | None = None


def redeem(
    *,
    progress: MyPassProgress,
    target_booth: str,
    booth_at_redeem: str,
    now: int | None = None,
) -> RedeemResult:
    now = now or int(time.time())

    if booth_at_redeem != target_booth:
        return RedeemResult(
            outcome="denied_not_completed",
            progress=progress,
            reason=f"redeem booth {booth_at_redeem} != target {target_booth}",
        )
    if not progress.completed_at:
        return RedeemResult(
            outcome="denied_not_completed",
            progress=progress,
            reason="challenge not completed",
        )
    if progress.flagged_review:
        return RedeemResult(
            outcome="denied_flagged",
            progress=progress,
            reason=f"under review: {progress.flag_reason}",
        )
    if progress.redeemed_at:
        return RedeemResult(
            outcome="denied_already_redeemed",
            progress=progress,
            reason="already redeemed",
        )
    progress.redeemed_at = now
    return RedeemResult(outcome="ok", progress=progress)
