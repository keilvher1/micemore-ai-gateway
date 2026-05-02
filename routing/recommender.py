"""다음 부스 추천 — 0.4·interest + 0.2/dist + 0.3/(1+crowd) + 0.1·persona.

순수 함수, 외부 의존성 없음.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from routing.crowd_tracker import BoothCrowdState
from routing.graph_builder import BoothGraph

# 가중치 — 한 곳에 모음
W_INTEREST = 0.40
W_DIST = 0.20
W_CROWD = 0.30
W_PERSONA = 0.10
ASSERT_TOTAL = round(W_INTEREST + W_DIST + W_CROWD + W_PERSONA, 4)
assert ASSERT_TOTAL == 1.0


@dataclass
class Recommendation:
    booth_id: str
    score: float
    distance_m: float
    wait_min: float
    interest_match: float
    persona_affinity: float
    reason: str


def _interest_match(
    visitor_interests: set[str], booth_keywords: set[str]
) -> float:
    """Jaccard — 둘 다 비어 있으면 0."""
    if not visitor_interests or not booth_keywords:
        return 0.0
    intersect = len(visitor_interests & booth_keywords)
    union = len(visitor_interests | booth_keywords)
    return intersect / union if union else 0.0


def _persona_affinity(
    visitor_persona_id: int | None,
    booth_persona_distribution: dict[int, float] | None,
) -> float:
    """페르소나 매칭 — booth 의 페르소나 비율 중 visitor 의 페르소나 비중."""
    if visitor_persona_id is None or not booth_persona_distribution:
        return 0.0
    return float(booth_persona_distribution.get(visitor_persona_id, 0.0))


def recommend_next_booth(
    *,
    visitor_interests: list[str],
    visitor_persona_id: int | None,
    current_booth_id: str,
    visited_booth_ids: set[str],
    time_left_min: int,
    graph: BoothGraph,
    booth_keywords: dict[str, list[str]],
    crowd: dict[str, BoothCrowdState],
    booth_persona_distribution: dict[str, dict[int, float]] | None = None,
    top_n: int = 3,
) -> list[Recommendation]:
    """unvisited ∩ time_left 내 도달 가능 → 점수 정렬 top_n."""
    if current_booth_id not in graph.nodes:
        return []

    v_interests = {s.lower() for s in visitor_interests}
    bp_dist = booth_persona_distribution or {}
    out: list[Recommendation] = []

    for candidate_id, dist_m in graph.neighbors(current_booth_id).items():
        if candidate_id in visited_booth_ids:
            continue
        # 도보 80m/분 가정 + 부스 평균 체류 (대기 추정 포함)
        cs = crowd.get(candidate_id, BoothCrowdState(booth_id=candidate_id))
        walk_min = dist_m / 80.0
        wait_min = cs.queue_estimate_min
        total_min = walk_min + wait_min + cs.avg_dwell_min
        if total_min > time_left_min:
            continue

        keywords = {k.lower() for k in booth_keywords.get(candidate_id, [])}
        im = _interest_match(v_interests, keywords)
        # 거리 기여 — 200m=0, 0m=1
        ds = max(0.0, 1.0 - dist_m / 200.0)
        # 혼잡 기여 — 1/(1+crowd_score)
        crowd_score = cs.queue_score()
        cs_score = 1.0 / (1.0 + crowd_score)
        pa = _persona_affinity(
            visitor_persona_id, bp_dist.get(candidate_id)
        )

        score = round(
            im * W_INTEREST + ds * W_DIST + cs_score * W_CROWD + pa * W_PERSONA,
            4,
        )
        if score <= 0:
            continue

        bits: list[str] = []
        if im > 0.3:
            bits.append("관심사 일치")
        if dist_m < 50:
            bits.append(f"가까움 {int(dist_m)}m")
        if crowd_score < 0.3:
            bits.append("여유")
        elif crowd_score > 0.7:
            bits.append(f"혼잡 (대기 {int(wait_min)}분)")
        reason = ", ".join(bits) or "일반 추천"

        out.append(Recommendation(
            booth_id=candidate_id,
            score=score,
            distance_m=round(dist_m, 1),
            wait_min=round(wait_min, 1),
            interest_match=round(im, 3),
            persona_affinity=round(pa, 3),
            reason=reason,
        ))

    out.sort(key=lambda r: r.score, reverse=True)
    return out[:top_n]


def to_dict(r: Recommendation) -> dict:
    return asdict(r)
