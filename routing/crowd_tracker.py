"""부스 혼잡도 추적 — Phase 4-C.

베타: 인-메모리 dict + 단일 프로세스 (테스트 결정론).
정상 운영: Redis Streams XADD/XLEN — 본 클래스 인터페이스 그대로 백엔드만 교체.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BoothCrowdState:
    booth_id: str
    current_visitors: int = 0
    avg_dwell_min: float = 0.0
    queue_estimate_min: float = 0.0
    updated_at: int = 0

    def queue_score(self) -> float:
        """0(여유)~1(혼잡) 스케일. 추천 점수 가중치에서 1/(1+crowd) 형태로 사용."""
        # 30명 이상이면 만점 혼잡 가정 — 베타 단계 단순 비례
        return min(1.0, self.current_visitors / 30.0)


class CrowdTracker:
    def __init__(self) -> None:
        self._states: dict[str, BoothCrowdState] = {}
        self._dwell_window: dict[str, list[int]] = {}  # 최근 dwell sec

    def on_enter(self, booth_id: str, *, now: int | None = None) -> None:
        now = now or int(time.time())
        s = self._states.setdefault(booth_id, BoothCrowdState(booth_id=booth_id))
        s.current_visitors += 1
        s.updated_at = now

    def on_exit(
        self, booth_id: str, *, dwell_sec: int, now: int | None = None
    ) -> None:
        now = now or int(time.time())
        s = self._states.setdefault(booth_id, BoothCrowdState(booth_id=booth_id))
        s.current_visitors = max(0, s.current_visitors - 1)
        s.updated_at = now
        # 이동 평균 (최근 50개)
        win = self._dwell_window.setdefault(booth_id, [])
        win.append(dwell_sec)
        if len(win) > 50:
            del win[0]
        s.avg_dwell_min = round(sum(win) / len(win) / 60.0, 2)
        # 대기 추정 — 혼잡도 × 평균 체류 ÷ 5 (5명 처리 가정)
        s.queue_estimate_min = round(
            s.current_visitors * s.avg_dwell_min / 5.0, 2
        )

    def get(self, booth_id: str) -> BoothCrowdState:
        return self._states.get(
            booth_id, BoothCrowdState(booth_id=booth_id)
        )

    def all(self) -> list[BoothCrowdState]:
        return list(self._states.values())

    def reset(self) -> None:
        self._states.clear()
        self._dwell_window.clear()
