"""insights/{aggregator,report_generator,pdf_renderer}.py 검증."""
from __future__ import annotations

import os

os.environ["USE_MOCK"] = "true"

from insights.aggregator import aggregate  # noqa: E402
from insights.pdf_renderer import render_html  # noqa: E402
from insights.report_generator import (  # noqa: E402
    Report,
    generate,
    has_banned_words,
)


def _seed_rows() -> list[dict]:
    return [
        # 한국 참가자 1
        {"visitor_id": "v1", "country_code": "KR", "age_band": "30s",
         "gender": "F", "booth_id": "lumen", "dwell_sec": 420,
         "copilot_questions_count": 5,
         "copilot_question_topics": ["pricing", "tech"],
         "translation_minutes": 0, "business_card_saved": True,
         "visit_started_at": 1718784000},
        # 외국인 참가자 1 (US)
        {"visitor_id": "v2", "country_code": "US", "age_band": "40s",
         "gender": "M", "booth_id": "lumen", "dwell_sec": 360,
         "copilot_questions_count": 3,
         "copilot_question_topics": ["tech"],
         "translation_minutes": 4, "business_card_saved": True,
         "visit_started_at": 1718787600},
        # 외국인 참가자 2 (JP)
        {"visitor_id": "v3", "country_code": "JP", "age_band": "30s",
         "gender": "F", "booth_id": "novasight", "dwell_sec": 240,
         "copilot_questions_count": 2,
         "copilot_question_topics": ["pricing"],
         "translation_minutes": 3,
         "visit_started_at": 1718791200},
        # 한국 참가자 2 — 다른 부스
        {"visitor_id": "v4", "country_code": "KR", "age_band": "20s",
         "gender": "M", "booth_id": "novasight", "dwell_sec": 180,
         "copilot_questions_count": 1,
         "copilot_question_topics": ["overview"],
         "translation_minutes": 0,
         "visit_started_at": 1718791200},
    ]


def test_aggregate_basic_counts():
    stats = aggregate(
        _seed_rows(),
        event_id="ev_test",
        period_start=1718784000,
        period_end=1718870400,
        booth_names={"lumen": "Lumen Labs", "novasight": "NovaSight"},
    )
    assert stats.total_visitors == 4
    assert stats.foreigner_ratio == 0.5  # 2/4
    assert stats.translation_sessions == 2
    # booth top by visits
    names = [b.booth_name for b in stats.booths]
    assert "Lumen Labs" in names and "NovaSight" in names
    lumen = next(b for b in stats.booths if b.booth_id == "lumen")
    assert lumen.visits == 2
    assert lumen.cards_saved == 2
    assert lumen.foreigner_visits == 1


def test_aggregate_handles_empty_safely():
    stats = aggregate(
        [], event_id="empty", period_start=0, period_end=0,
    )
    assert stats.total_visitors == 0
    assert stats.foreigner_ratio == 0


def test_top_topics_counted():
    stats = aggregate(_seed_rows(), event_id="ev_test",
                      period_start=0, period_end=0)
    topics = dict(stats.top_topics)
    assert topics["pricing"] == 2
    assert topics["tech"] == 2


def test_generate_organizer_mock_contains_kpis():
    stats = aggregate(_seed_rows(), event_id="ev_test",
                      period_start=0, period_end=0,
                      booth_names={"lumen": "Lumen Labs",
                                   "novasight": "NovaSight"})
    rep: Report = generate(stats=stats, audience="organizer")
    assert rep.audience == "organizer"
    assert rep.model == "mock"
    assert "ev_test" in rep.markdown
    assert "외국인 비율" in rep.markdown
    # 광고성 표현 금지
    assert has_banned_words(rep.markdown) == []


def test_generate_exhibitor_uses_booth():
    stats = aggregate(_seed_rows(), event_id="ev_test",
                      period_start=0, period_end=0,
                      booth_names={"lumen": "Lumen Labs"})
    rep = generate(stats=stats, audience="exhibitor", booth_id="lumen")
    assert "Lumen Labs" in rep.markdown


def test_generate_municipality_with_region():
    stats = aggregate(_seed_rows(), event_id="ev_test",
                      period_start=0, period_end=0)
    rep = generate(stats=stats, audience="municipality", region="포항")
    assert "포항" in rep.markdown


def test_render_html_contains_doctype_and_styles():
    stats = aggregate(_seed_rows(), event_id="ev_test",
                      period_start=0, period_end=0)
    rep = generate(stats=stats, audience="organizer")
    html = render_html(rep, charts_caption=["방문자 시간대별 분포"])
    assert html.startswith("<!DOCTYPE html>")
    assert "Noto Sans KR" in html
    assert "방문자 시간대별 분포" in html
    assert "<h1>" in html
