"""trends/{timeseries,keyword_tracker,report_generator}.py."""
from __future__ import annotations

import os

os.environ["USE_MOCK"] = "true"

import pytest  # noqa: E402

from trends.keyword_tracker import (  # noqa: E402
    compare_quarters,
    declining_keywords,
    emerging_keywords,
    term_frequency,
    tokenize,
)
from trends.report_generator import generate as generate_trends_report  # noqa: E402
from trends.timeseries import (  # noqa: E402
    detect_weekly_seasonality,
    forecast,
    moving_average,
)


# ---------------------------------------------------------------------------
# Timeseries
# ---------------------------------------------------------------------------
def test_moving_average_basic():
    ma = moving_average([1, 2, 3, 4, 5], window=3)
    assert ma[0] == 1.0
    assert ma[1] == 1.5
    assert ma[2] == 2.0
    assert ma[3] == 3.0
    assert ma[4] == 4.0


def test_moving_average_invalid_window():
    with pytest.raises(ValueError):
        moving_average([1, 2, 3], window=0)


def test_weekly_seasonality_buckets():
    series = {
        "2026-04-20": 100,  # Monday
        "2026-04-21": 110,  # Tuesday
        "2026-04-26": 90,   # Sunday
    }
    s = detect_weekly_seasonality(series)
    # all 7 days present
    assert set(s.keys()) == set(range(7))


def test_forecast_empty_returns_empty():
    res = forecast({}, horizon_days=7)
    assert res.history_size == 0
    assert res.forecast == []


def test_forecast_mock_method_when_few_points():
    series = {
        f"2026-04-{i:02d}": float(10 + i)
        for i in range(1, 16)  # 15 days
    }
    res = forecast(series, horizon_days=7)
    assert res.method == "moving-avg-mock"
    assert len(res.forecast) == 7
    for p in res.forecast:
        assert p.lower <= p.value <= p.upper


def test_forecast_mock_history_size_recorded():
    series = {f"2026-04-{i:02d}": 10.0 for i in range(1, 11)}
    res = forecast(series)
    assert res.history_size == 10


# ---------------------------------------------------------------------------
# Keyword tracker
# ---------------------------------------------------------------------------
def test_tokenize_filters_stopwords():
    out = tokenize("AI and ML for Korea biotech")
    assert "ai" in out
    assert "ml" in out
    assert "Korea".lower() in out or "korea" in out
    assert "and" not in out


def test_tokenize_handles_korean():
    out = tokenize("바이오 회사 mass spec 검토")
    assert "바이오" in out
    assert "mass" in out
    # 한국어 2자 이상만
    assert "검토" in out


def test_term_frequency_counts():
    docs = ["ai ml ai", "AI AI ML"]
    tf = term_frequency(docs)
    # 2 + 2 = 4 (case-folded)
    assert tf["ai"] == 4
    # 1 + 1 = 2
    assert tf["ml"] == 2


def test_compare_quarters_emerging_and_rising():
    prev = ["older topic"] * 10
    curr = ["NewKeyword AI quantum"] * 15
    changes = compare_quarters(prev_docs=prev, curr_docs=curr, top_n=20)
    keys = {c.keyword: c for c in changes}
    # ai 는 emerging (prev=0)
    assert keys["ai"].label in {"emerging", "rising"}
    assert keys["ai"].prev_count == 0
    assert keys["ai"].curr_count == 15
    # newkeyword 도 emerging (15 ≥ EMERGING_MIN_COUNT)
    assert keys["newkeyword"].label == "emerging"


def test_declining_keywords_filter():
    prev = ["legacy"] * 100
    curr = ["legacy"] * 30   # -70%
    changes = compare_quarters(prev_docs=prev, curr_docs=curr)
    declining = declining_keywords(changes)
    assert any(c.keyword == "legacy" for c in declining)


def test_emerging_below_min_count_not_classified():
    prev = ["other"] * 5
    curr = ["raretag"] * 3    # 3 < EMERGING_MIN_COUNT(=10)
    changes = compare_quarters(prev_docs=prev, curr_docs=curr)
    emerging = emerging_keywords(changes)
    assert all(c.keyword != "raretag" for c in emerging)


# ---------------------------------------------------------------------------
# Report generator (mock)
# ---------------------------------------------------------------------------
def test_report_mock_under_threshold_data():
    from trends.timeseries import ForecastResult, ForecastPoint

    forecast_res = ForecastResult(
        method="moving-avg-mock",
        history_size=10,
        forecast=[
            ForecastPoint(date="2026-07-01", value=12.0,
                          lower=10.0, upper=14.0),
        ],
    )
    rep = generate_trends_report(
        quarter="2026Q3", region="포항·경북",
        forecast=forecast_res, keyword_changes=[],
        foreigner_ratio=0.18,
    )
    assert rep.model == "mock"
    assert "2026Q3" in rep.markdown
    assert "포항·경북" in rep.markdown
    assert "외국인 비율 18.0%" in rep.markdown


def test_report_no_foreigner_data():
    from trends.timeseries import ForecastResult
    forecast_res = ForecastResult(
        method="moving-avg-mock", history_size=5, forecast=[],
    )
    rep = generate_trends_report(
        quarter="2026Q3", region="서울",
        forecast=forecast_res, keyword_changes=[],
        foreigner_ratio=None,
    )
    assert "데이터 없음" in rep.markdown
