"""MyPass 챌린지 통합 — 동선 추천에 패스트트랙 인센티브 부여.

추천 결과의 booth 가 active MyPass challenge 의 partner/target 인 경우:
  - partner: 점수에 +0.1 보너스 (challenge 진행도 가속 인센티브)
  - target (challenge 완주 후): 점수에 +0.2 (보상 사용 가능 신호)

mypass 모듈과 의존성 일방 — fastpass 가 mypass 의 challenge/progress 를 읽음.
"""
from __future__ import annotations

from dataclasses import replace

from mypass.challenge import MyPassChallenge
from mypass.progress import MyPassProgress
from routing.recommender import Recommendation


def boost_with_mypass(
    *,
    recommendations: list[Recommendation],
    challenge: MyPassChallenge | None,
    progress: MyPassProgress | None,
    partner_bonus: float = 0.10,
    target_bonus: float = 0.20,
) -> list[Recommendation]:
    """추천 리스트에 MyPass 인센티브 가산 + reason 갱신."""
    if challenge is None:
        return recommendations
    visited = set((progress.visited_partners if progress else []) or [])
    completed = bool(progress and progress.completed_at)

    boosted: list[Recommendation] = []
    for r in recommendations:
        bonus = 0.0
        extra_reason: str | None = None
        if r.booth_id == challenge.target_booth and completed:
            bonus = target_bonus
            extra_reason = "MyPass 패스트트랙 가능"
        elif challenge.is_partner(r.booth_id) and r.booth_id not in visited:
            bonus = partner_bonus
            n_left = challenge.required_visits - len(visited)
            extra_reason = f"MyPass 진행 +1 (남은 {n_left}/{challenge.required_visits})"

        if bonus > 0:
            new_score = round(min(1.0, r.score + bonus), 4)
            new_reason = (r.reason + " · " + extra_reason) if extra_reason else r.reason
            boosted.append(replace(r, score=new_score, reason=new_reason))
        else:
            boosted.append(r)

    boosted.sort(key=lambda x: x.score, reverse=True)
    return boosted
