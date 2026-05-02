"""ICP ↔ Visitor 매칭 — cosine NN + 거리·시간 가중.

입력: ICP 1개 + 100m 이내 visitor 후보 N명.
출력: top_k MatchEvent (score 정렬).

점수:
  cosine(ICP.emb, visitor.emb) × 0.6
  + (1 - dist_m/100)             × 0.25
  + time_window_bonus             × 0.15

time_window_bonus: 행사 진행 시간 중 9~12시 / 13~17시 = 1.0,
                   기타 = 0.5 (낮은 트래픽 시간대)

순수 함수, USE_MOCK 무관.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass
class MatchEvent:
    booth_id: str
    visitor_hash: str
    score: float            # 0.0~1.0
    cosine: float
    distance_m: float
    time_bonus: float
    reason: str             # 1-line explainability — push 본문에 그대로 사용
    triggered_at: int       # epoch sec


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------
def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def time_bonus(now_epoch: int) -> float:
    """행사 황금시간 1.0, 그 외 0.5."""
    h = datetime.fromtimestamp(now_epoch).hour
    if 9 <= h <= 12 or 13 <= h <= 17:
        return 1.0
    return 0.5


# ---------------------------------------------------------------------------
# 매칭 코어
# ---------------------------------------------------------------------------
def match(
    *,
    icp_embedding: list[float],
    booth_id: str,
    candidates: Iterable[tuple[str, list[float], float]],
    # candidates: [(visitor_hash, embedding, distance_m), ...]
    now_epoch: int,
    score_threshold: float = 0.55,
    top_k: int = 3,
    icp_keywords: list[str] | None = None,
) -> list[MatchEvent]:
    """후보 중 score≥threshold 인 top_k 반환."""
    tb = time_bonus(now_epoch)
    out: list[MatchEvent] = []
    for vh, vemb, dist_m in candidates:
        if dist_m > 100.0:
            continue  # 100m 초과는 매칭 풀에서 제외
        c = cosine(icp_embedding, vemb)
        # cosine 은 [-1,1] → 0~1 로 매핑 (음수는 비매칭)
        c01 = max(0.0, c)
        dist_score = max(0.0, 1.0 - dist_m / 100.0)
        score = round(c01 * 0.6 + dist_score * 0.25 + tb * 0.15, 4)
        if score < score_threshold:
            continue
        kw = (icp_keywords or [None])[0] or "관심사 일치"
        reason = f"{kw} 일치, {int(dist_m)}m"
        out.append(
            MatchEvent(
                booth_id=booth_id,
                visitor_hash=vh,
                score=score,
                cosine=round(c, 4),
                distance_m=round(dist_m, 1),
                time_bonus=tb,
                reason=reason,
                triggered_at=now_epoch,
            )
        )
    out.sort(key=lambda e: e.score, reverse=True)
    return out[:top_k]
