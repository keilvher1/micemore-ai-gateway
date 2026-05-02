"""행사 종료 후 데이터 집계 — Firestore raw → EventStats.

원칙:
  - **순수 함수**: dict 리스트 in → dataclass out. 단위 테스트 결정론.
  - **익명화**: 이메일·전화 등 PII 는 입력 단계에서 제거된 dict 만 받음.
  - **수치 정확성**: 추측 표현 없는 사실값만 — narrator 단계가 추측 모드.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# 입력 dict 키 가정 (Firestore export 형식)
# ---------------------------------------------------------------------------
#   visitor_id, age_band ('20s'|'30s'|...), gender, country_code, lang,
#   booth_id, dwell_sec, copilot_questions_count, copilot_question_topics,
#   translation_minutes, pamphlet_downloaded, business_card_saved,
#   visit_started_at(int sec), visit_ended_at(int sec)


@dataclass
class BoothStat:
    booth_id: str
    booth_name: str
    visits: int = 0
    unique_visitors: int = 0
    total_dwell_min: float = 0.0
    avg_dwell_min: float = 0.0
    foreigner_visits: int = 0
    questions_total: int = 0
    cards_saved: int = 0


@dataclass
class HourBucket:
    hour: int        # 0~23
    visits: int = 0
    queue_proxy: float = 0.0  # avg dwell × concurrent visitors approx


@dataclass
class EventStats:
    event_id: str
    period_start: int
    period_end: int

    total_visitors: int = 0
    foreigner_ratio: float = 0.0
    translation_sessions: int = 0
    avg_dwell_min: float = 0.0
    booths: list[BoothStat] = field(default_factory=list)
    hours: list[HourBucket] = field(default_factory=list)
    age_distribution: dict[str, int] = field(default_factory=dict)
    gender_distribution: dict[str, int] = field(default_factory=dict)
    top_topics: list[tuple[str, int]] = field(default_factory=list)
    nps_score: float | None = None
    nps_comments_summary: str | None = None


# ---------------------------------------------------------------------------
# 집계 메인
# ---------------------------------------------------------------------------
def aggregate(
    rows: Iterable[dict],
    *,
    event_id: str,
    period_start: int,
    period_end: int,
    booth_names: dict[str, str] | None = None,
    foreigner_iso: set[str] | None = None,
) -> EventStats:
    booth_names = booth_names or {}
    foreigner_iso = foreigner_iso or {"US", "JP", "CN", "DE", "FR", "GB", "VN"}
    rows_list = list(rows)

    visitors: set[str] = set()
    foreigner_visitors: set[str] = set()
    booth_acc: dict[str, BoothStat] = {}
    booth_unique: dict[str, set[str]] = defaultdict(set)
    hour_acc: dict[int, HourBucket] = {}
    age_counter: Counter[str] = Counter()
    gender_counter: Counter[str] = Counter()
    topic_counter: Counter[str] = Counter()
    translation_minutes_total = 0
    translation_sessions = 0
    total_dwell_sec = 0

    for r in rows_list:
        vid = str(r.get("visitor_id", ""))
        if not vid:
            continue
        visitors.add(vid)
        if r.get("country_code") in foreigner_iso:
            foreigner_visitors.add(vid)

        age_counter[str(r.get("age_band") or "unknown")] += 1
        gender_counter[str(r.get("gender") or "unknown")] += 1

        for t in (r.get("copilot_question_topics") or []):
            topic_counter[str(t).lower()] += 1

        bid = str(r.get("booth_id", ""))
        if bid:
            stat = booth_acc.setdefault(
                bid,
                BoothStat(booth_id=bid, booth_name=booth_names.get(bid, bid)),
            )
            stat.visits += 1
            booth_unique[bid].add(vid)
            stat.total_dwell_min += (int(r.get("dwell_sec") or 0)) / 60.0
            stat.questions_total += int(r.get("copilot_questions_count") or 0)
            if r.get("business_card_saved"):
                stat.cards_saved += 1
            if vid in foreigner_visitors:
                stat.foreigner_visits += 1

        # hour bucket — visit_started_at 기준
        ts = int(r.get("visit_started_at") or 0)
        if ts:
            hr = (ts // 3600) % 24
            hb = hour_acc.setdefault(hr, HourBucket(hour=hr))
            hb.visits += 1
            hb.queue_proxy += (int(r.get("dwell_sec") or 0)) / 60.0

        if r.get("translation_minutes"):
            translation_minutes_total += int(r["translation_minutes"])
            translation_sessions += 1
        total_dwell_sec += int(r.get("dwell_sec") or 0)

    # finalize
    booths: list[BoothStat] = []
    for bid, stat in booth_acc.items():
        stat.unique_visitors = len(booth_unique[bid])
        if stat.visits:
            stat.avg_dwell_min = round(stat.total_dwell_min / stat.visits, 2)
        stat.total_dwell_min = round(stat.total_dwell_min, 1)
        booths.append(stat)
    booths.sort(key=lambda s: s.visits, reverse=True)

    hours = sorted(hour_acc.values(), key=lambda h: h.hour)
    for hb in hours:
        if hb.visits:
            hb.queue_proxy = round(hb.queue_proxy / hb.visits, 2)

    n = len(rows_list) or 1
    return EventStats(
        event_id=event_id,
        period_start=period_start,
        period_end=period_end,
        total_visitors=len(visitors),
        foreigner_ratio=round(
            (len(foreigner_visitors) / len(visitors)) if visitors else 0, 3
        ),
        translation_sessions=translation_sessions,
        avg_dwell_min=round((total_dwell_sec / 60.0) / n, 2),
        booths=booths,
        hours=hours,
        age_distribution=dict(age_counter),
        gender_distribution=dict(gender_counter),
        top_topics=topic_counter.most_common(10),
    )
