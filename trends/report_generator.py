"""분기 트렌드 리포트 outline 생성.

insights/report_generator.py 와는 다른 audience:
  - PCO/지자체/컨벤션센터 — B2B 가격 ₩500K/Q (PDF) · ₩1M/월 (API).

본 모듈은 outline + LLM prompt 만 — PDF 렌더는 insights/pdf_renderer 재사용.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from trends.keyword_tracker import KeywordChange
from trends.timeseries import ForecastResult

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


@dataclass
class TrendsReport:
    quarter: str                # "2026Q4"
    region: str                 # "한국 전국" | "수도권" | "포항·경북" | ...
    markdown: str
    model: str


_PROMPT = """\
당신은 MICE 산업 분기 트렌드 리포트를 한국어 markdown 으로 작성합니다.

INPUT (사실값):
- 분기: {quarter}
- 지역: {region}
- 시계열 forecast (다음 분기 30일 예측, mock 또는 prophet):
{forecast_json}
- 직전 분기 대비 키워드 변화 (top 20):
{kw_json}
- 외국인 비율 (옵션):
{foreigner_block}

리포트 구조:
# {quarter} {region} MICE 트렌드 리포트
## 한 페이지 요약
- 3 bullet, 각 한 줄. 핵심 시그널.

## 시계열 신호
- 다음 분기 트래픽 예측. 신뢰구간 명시. method 가 "moving-avg-mock" 인 경우
  "보수적 추정" 표지 필수.

## 신흥 키워드 (Emerging)
- top 5. 빈도 + 직전 분기 대비.

## 쇠퇴 키워드 (Declining)
- top 3. 의미 해석은 "추정" 표지로.

## 정책·기획 시사점
- 3 항목, 우선순위 순.

## 데이터 한계
- 표본 크기 · 측정 누락 · 지역 편향.

규칙:
1. 광고성 표현 금지 ("최고", "혁신적인", "world-class").
2. 700자 ±100.
3. 추측/확인 명시 표지.
"""


def _format_foreigner(foreigner_ratio: float | None) -> str:
    if foreigner_ratio is None:
        return "(데이터 없음)"
    return f"외국인 비율 {foreigner_ratio*100:.1f}% (확인)"


def generate(
    *,
    quarter: str,
    region: str,
    forecast: ForecastResult,
    keyword_changes: list[KeywordChange],
    foreigner_ratio: float | None = None,
) -> TrendsReport:
    import json
    forecast_json = json.dumps(asdict(forecast), ensure_ascii=False)
    kw_json = json.dumps(
        [asdict(k) for k in keyword_changes[:20]],
        ensure_ascii=False,
    )
    prompt = _PROMPT.format(
        quarter=quarter,
        region=region,
        forecast_json=forecast_json,
        kw_json=kw_json,
        foreigner_block=_format_foreigner(foreigner_ratio),
    )

    if USE_MOCK or forecast.history_size < 30:
        return TrendsReport(
            quarter=quarter,
            region=region,
            markdown=_mock_markdown(
                quarter, region, forecast, keyword_changes, foreigner_ratio
            ),
            model="mock",
        )

    # 실 호출 — Claude
    text, model = _call_claude(prompt)
    return TrendsReport(quarter=quarter, region=region, markdown=text, model=model)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------
def _mock_markdown(
    quarter: str,
    region: str,
    forecast: ForecastResult,
    keyword_changes: list[KeywordChange],
    foreigner_ratio: float | None,
) -> str:
    emerging = [k for k in keyword_changes if k.label == "emerging"][:5]
    declining = [k for k in keyword_changes if k.label == "declining"][:3]
    next_30_avg = (
        sum(p.value for p in forecast.forecast) / len(forecast.forecast)
        if forecast.forecast else 0.0
    )

    em_lines = "\n".join(
        f"- **{k.keyword}**: {k.prev_count} → {k.curr_count} (+{k.delta})"
        for k in emerging
    ) or "- 데이터 없음"
    dec_lines = "\n".join(
        f"- **{k.keyword}**: {k.prev_count} → {k.curr_count} ({k.delta})"
        for k in declining
    ) or "- 데이터 없음"

    return (
        f"# {quarter} {region} MICE 트렌드 리포트 [MOCK]\n"
        f"## 한 페이지 요약\n"
        f"- 다음 30일 일평균 트래픽 {next_30_avg:.1f} (보수적 추정, "
        f"history {forecast.history_size}일)\n"
        f"- 신흥 키워드 {len(emerging)}건 발견\n"
        f"- 쇠퇴 신호 {len(declining)}건 — 추정\n\n"
        f"## 시계열 신호\n"
        f"method={forecast.method}. 보수적 추정. 표본 부족 시 신뢰구간 넓음.\n\n"
        f"## 신흥 키워드\n{em_lines}\n\n"
        f"## 쇠퇴 키워드\n{dec_lines}\n\n"
        f"## 정책·기획 시사점\n"
        f"1. 신흥 키워드 카테고리 부스 우선 유치 (확인된 행동 데이터 근거)\n"
        f"2. 쇠퇴 신호는 1분기 추가 관측 후 판단 (추정)\n"
        f"3. 외국인 인프라 ({_format_foreigner(foreigner_ratio)}) 점검\n\n"
        f"## 데이터 한계\n"
        f"- 표본 {forecast.history_size}일, mock 모드 (베타 단계).\n"
    )


def _call_claude(prompt: str) -> tuple[str, str]:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=2000,
        system=prompt,
        messages=[{"role": "user", "content": "Write the report now."}],
    )
    return resp.content[0].text, "claude"  # type: ignore[index]
