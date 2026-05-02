"""시계열 분석 — 이동평균 + 단순 계절성.

Phase 4-B 정상모드: Prophet/NeuralProphet (lazy import, 데이터 20행사+).
mock + 데이터 부족 시: 순수 Python 이동평균 + 주간 계절성.

입력은 dict[YYYY-MM-DD] -> float (일별/주별 카운트).
"""
from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
MIN_POINTS_FOR_PROPHET = int(os.getenv("TRENDS_MIN_POINTS", "365"))


@dataclass
class ForecastPoint:
    date: str          # ISO yyyy-mm-dd
    value: float       # 예측값
    lower: float       # 80% 신뢰구간 하한
    upper: float       # 80% 신뢰구간 상한


@dataclass
class ForecastResult:
    method: str        # "prophet" | "moving-avg-mock"
    history_size: int
    forecast: list[ForecastPoint]


def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def detect_weekly_seasonality(
    series: dict[str, float]
) -> dict[int, float]:
    """요일별 평균 → 0(월)~6(일) dict. 비어있으면 모든 요일 0."""
    by_dow: dict[int, list[float]] = {i: [] for i in range(7)}
    for d, v in series.items():
        by_dow[_parse(d).weekday()].append(v)
    return {
        dow: round(sum(vs) / len(vs), 3) if vs else 0.0
        for dow, vs in by_dow.items()
    }


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
def forecast(
    series: dict[str, float],
    *,
    horizon_days: int = 30,
    ma_window: int = 14,
) -> ForecastResult:
    """과거 series → horizon_days 예측."""
    if not series:
        return ForecastResult(
            method="moving-avg-mock",
            history_size=0,
            forecast=[],
        )

    n = len(series)
    if not USE_MOCK and n >= MIN_POINTS_FOR_PROPHET:
        return _prophet_forecast(series, horizon_days)

    return _ma_forecast(series, horizon_days, ma_window)


def _ma_forecast(
    series: dict[str, float], horizon: int, window: int
) -> ForecastResult:
    sorted_dates = sorted(series.keys())
    values = [series[d] for d in sorted_dates]
    ma = moving_average(values, window)
    base = ma[-1] if ma else 0.0
    seasonality = detect_weekly_seasonality(series)
    seasonal_baseline = (
        statistics.mean(seasonality.values()) if seasonality else 0.0
    ) or 1.0
    # 표본 표준편차 — 신뢰구간용
    sd = (
        statistics.pstdev(values[-min(len(values), 30):])
        if len(values) > 1 else 0.0
    )

    last_d = _parse(sorted_dates[-1])
    points: list[ForecastPoint] = []
    for i in range(1, horizon + 1):
        future = last_d + timedelta(days=i)
        dow = future.weekday()
        seasonal_factor = (
            seasonality.get(dow, seasonal_baseline) / seasonal_baseline
            if seasonal_baseline else 1.0
        )
        forecast_val = base * seasonal_factor
        points.append(
            ForecastPoint(
                date=future.isoformat(),
                value=round(forecast_val, 2),
                lower=round(forecast_val - 1.28 * sd, 2),
                upper=round(forecast_val + 1.28 * sd, 2),
            )
        )
    return ForecastResult(
        method="moving-avg-mock",
        history_size=len(series),
        forecast=points,
    )


def _prophet_forecast(
    series: dict[str, float], horizon: int
) -> ForecastResult:
    """실 모드 — Prophet 호출. 의존성 lazy import."""
    from prophet import Prophet  # type: ignore
    import pandas as pd  # type: ignore

    df = pd.DataFrame(
        [{"ds": d, "y": v} for d, v in sorted(series.items())]
    )
    df["ds"] = pd.to_datetime(df["ds"])
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                interval_width=0.8)
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon)
    pred = m.predict(future).tail(horizon)
    points = [
        ForecastPoint(
            date=row["ds"].strftime("%Y-%m-%d"),
            value=round(float(row["yhat"]), 2),
            lower=round(float(row["yhat_lower"]), 2),
            upper=round(float(row["yhat_upper"]), 2),
        )
        for _, row in pred.iterrows()
    ]
    return ForecastResult(
        method="prophet",
        history_size=len(series),
        forecast=points,
    )
