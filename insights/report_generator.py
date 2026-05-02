"""LLM 자연어 리포트 — 3 audience.

audience:
  - organizer  : 행사 전체 KPI + 트렌드 + 다음 행사 액션 3개
  - exhibitor  : 우리 부스만 + 동 카테고리 평균 비교 + Top 10 hot lead
  - municipality (지자체): 외국인 비율, 통역 사용량, 지역 산업 매칭, 정책 제언

원칙:
  - 데이터 빈약 (n<30) 시 "데이터 부족 — 보수적 해석" 명시
  - 추측은 "추정" 표지, 확인된 사실은 "확인"
  - 광고성 표현 금지 (excellent, world-class, unprecedented…)
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Literal

from insights.aggregator import EventStats

log = logging.getLogger("insights.report")

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

Audience = Literal["organizer", "exhibitor", "municipality"]


# ---------------------------------------------------------------------------
# System prompts (3 audience)
# ---------------------------------------------------------------------------
_PROMPT_ORGANIZER = """\
당신은 MICE 주최자에게 보내는 행사 사후 리포트를 한국어 markdown 으로 작성합니다.

INPUT (사실값 — 이 안에 없는 수치는 절대 만들지 말 것):
{stats_json}

리포트 구조 (markdown):

# {event_id} 행사 인사이트 리포트
## 요약
- 3 bullet, 각 한 줄. 핵심 KPI (참가자·외국인 비율·통역 사용).

## 흥행 포인트
- 어떤 부스/시간대/콘텐츠가 강했나. 부스명·수치·비교 명시.

## 개선이 필요한 부분
- 구체 신호 (대기 정체 시간, 낮은 체류 부스 등) 인용. 추측은 "추정" 표지.

## 다음 행사 추천 액션
1. ... (우선순위 1, 데이터 근거 포함)
2. ...
3. ...

## 데이터 한계
- 표본 크기, 측정 누락, 편향 가능성 명시.

규칙:
1. 추측은 "추정", 확인된 값은 "확인" 으로 명시.
2. 분량 700자 ±100.
3. 광고성 형용사("최고", "혁신적", "world-class") 금지.
4. 데이터 없는 항목은 "데이터 없음" 으로 표기, 추측 금지.
"""

_PROMPT_EXHIBITOR = """\
당신은 MICE 전시자에게 보내는 부스 사후 리포트를 한국어 markdown 으로 작성합니다.

INPUT — 본인 부스 통계 + 행사 평균 비교:
{stats_json}

본 부스 ID: {booth_id}
본 부스 카테고리 평균 (옵션): {category_avg}

리포트 구조:

# {booth_id} 부스 ROI 리포트
## 핵심 수치
- 방문자 N명 (행사 평균 대비 ±%) / 평균 체류 X분 / 명함 Y건 / Hot Lead Z명

## 강점 신호
- 평균보다 잘한 항목 2~3개. 비교 수치 명시.

## 개선 영역
- 평균보다 떨어진 항목. 원인 가설 1개 ("추정" 표지).

## 다음 행사 추천 액션
- 구체 액션 3개 (자료 보강, 시연 시간 조정, 가격 안내 명확화 등).

## Top 10 Hot Lead 요약
- (별도 첨부된 lead 목록 인용. 본 본문에는 카운트만)

규칙: 광고성 표현 금지. 분량 600자 ±100. 비교 데이터 없는 항목은 비교 생략.
"""

_PROMPT_MUNICIPALITY = """\
You write a post-event policy brief in Korean markdown for a regional MICE bureau (지자체).

INPUT:
{stats_json}

Region: {region}

Sections:
# {region} 지역 MICE 행사 데이터 정책 브리프
## 외국인 참가 현황
- 비율, 출신국, 체류 시간 — 행사 평균 vs 지자체 목표 비교 (있을 때).

## 통역 인프라 활용
- 통역 세션 수, 평균 길이, 사용 부스 수.

## 지역 산업 매칭
- 어떤 산업/카테고리 부스에 외국인 참가자가 몰렸나.

## 정책 제언
- 3개. 다음 분기 예산/사업 우선순위에 도움되는 형태로.

규칙:
- 한국어 격식체.
- "정책 제언" 외 섹션은 사실 위주, 의견 금지.
- 분량 500자 ±100.
"""


_PROMPTS: dict[Audience, str] = {
    "organizer": _PROMPT_ORGANIZER,
    "exhibitor": _PROMPT_EXHIBITOR,
    "municipality": _PROMPT_MUNICIPALITY,
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class Report:
    audience: Audience
    markdown: str
    model: str
    raw_prompt: str


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_prompt(
    *,
    stats: EventStats,
    audience: Audience,
    booth_id: str | None = None,
    region: str | None = None,
    category_avg: dict | None = None,
) -> str:
    template = _PROMPTS[audience]
    stats_json = json.dumps(asdict(stats), ensure_ascii=False, indent=2)
    return template.format(
        stats_json=stats_json,
        event_id=stats.event_id,
        booth_id=booth_id or "(N/A)",
        region=region or "(미지정)",
        category_avg=json.dumps(category_avg or {}, ensure_ascii=False),
    )


def generate(
    *,
    stats: EventStats,
    audience: Audience,
    booth_id: str | None = None,
    region: str | None = None,
    category_avg: dict | None = None,
) -> Report:
    prompt = build_prompt(
        stats=stats,
        audience=audience,
        booth_id=booth_id,
        region=region,
        category_avg=category_avg,
    )
    if USE_MOCK or stats.total_visitors < 30:
        return Report(
            audience=audience,
            markdown=_mock_markdown(stats, audience, booth_id, region),
            model="mock",
            raw_prompt=prompt,
        )

    # 실 호출
    try:
        text, model = _call_claude(prompt)
    except Exception as exc:  # noqa: BLE001
        log.warning("claude failed (%s) → gpt", exc)
        text, model = _call_gpt(prompt)
    return Report(
        audience=audience, markdown=text, model=model, raw_prompt=prompt,
    )


# ---------------------------------------------------------------------------
# LLM wrappers
# ---------------------------------------------------------------------------
def _call_claude(prompt: str) -> tuple[str, str]:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=2000,
        system=prompt,
        messages=[{"role": "user", "content": "Write the report now."}],
    )
    return msg.content[0].text, "claude"  # type: ignore[index]


def _call_gpt(prompt: str) -> tuple[str, str]:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.getenv("GPT_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Write the report now."},
        ],
        max_tokens=2000,
    )
    return resp.choices[0].message.content or "", "gpt-4o"


# ---------------------------------------------------------------------------
# Mock — 결정론적 markdown
# ---------------------------------------------------------------------------
def _mock_markdown(
    stats: EventStats,
    audience: Audience,
    booth_id: str | None,
    region: str | None,
) -> str:
    if audience == "organizer":
        top = stats.booths[:3]
        top_lines = "\n".join(
            f"- {b.booth_name}: {b.visits}회 방문, 평균 {b.avg_dwell_min}분"
            for b in top
        ) or "- 데이터 없음"
        return (
            f"# {stats.event_id} 행사 인사이트 리포트 [MOCK]\n"
            f"## 요약\n"
            f"- 총 {stats.total_visitors}명 참가 (확인)\n"
            f"- 외국인 비율 {stats.foreigner_ratio*100:.1f}% (확인)\n"
            f"- 통역 세션 {stats.translation_sessions}회 (확인)\n\n"
            f"## 흥행 포인트\n{top_lines}\n\n"
            f"## 개선이 필요한 부분\n- 데이터 부족 — 보수적 해석 필요 (추정).\n\n"
            f"## 다음 행사 추천 액션\n"
            f"1. Top 부스 카테고리 확장\n"
            f"2. 통역 부스 운영자 사전 교육\n"
            f"3. 시간대별 참가자 안내 동선 최적화\n\n"
            f"## 데이터 한계\n- 본 mock 리포트는 시연용입니다.\n"
        )
    if audience == "exhibitor":
        my = next((b for b in stats.booths if b.booth_id == booth_id), None)
        if my is None:
            return f"# {booth_id} 부스 리포트 [MOCK]\n부스 데이터 없음.\n"
        return (
            f"# {my.booth_name} 부스 ROI 리포트 [MOCK]\n"
            f"## 핵심 수치\n"
            f"- 방문 {my.visits}회 / 평균 체류 {my.avg_dwell_min}분 / "
            f"명함 {my.cards_saved}건 (확인)\n\n"
            f"## 강점 신호\n- 외국인 방문 {my.foreigner_visits}건 (확인)\n\n"
            f"## 개선 영역\n- mock 데이터 — 비교 생략\n\n"
            f"## 다음 행사 추천 액션\n"
            f"1. 시연 시간 30분 단위로 명확히 게시\n"
            f"2. 영문 자료 보강 (외국인 방문 비율 고려)\n"
            f"3. Hot Lead 24h 내 follow-up\n"
        )
    return (
        f"# {region or '미지정'} 지역 MICE 정책 브리프 [MOCK]\n"
        f"## 외국인 참가 현황\n"
        f"- 외국인 비율 {stats.foreigner_ratio*100:.1f}% (확인)\n\n"
        f"## 통역 인프라 활용\n- 통역 세션 {stats.translation_sessions}회 (확인)\n\n"
        f"## 지역 산업 매칭\n- 데이터 분석 필요 (추정).\n\n"
        f"## 정책 제언\n"
        f"1. 차기 행사 통역 인프라 확대\n"
        f"2. 외국인 친화 부스 인센티브\n"
        f"3. 지역 산업 매칭 사전 큐레이션\n"
    )


# ---------------------------------------------------------------------------
# Light stripping — markdown 안전성 검사 (간단 sanitization 자리)
# ---------------------------------------------------------------------------
_BANNED_WORDS = ("최고", "혁신적", "world-class", "unprecedented", "혁신적인")


def has_banned_words(md: str) -> list[str]:
    found: list[str] = []
    for w in _BANNED_WORDS:
        if re.search(rf"\b{re.escape(w)}\b" if w.isascii() else re.escape(w), md):
            found.append(w)
    return found
