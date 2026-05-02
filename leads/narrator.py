"""점수 → 한국어 한 문장 narrative.

Mock 모드: 결정론적 템플릿. 외부 호출 없이 회귀 테스트 가능.
실 모드: Claude (Anthropic) primary, GPT-4o fallback. 80자 ±20.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from leads.scorer import ScoreBreakdown, VisitorBehavior, LeadLevel

log = logging.getLogger("leads.narrator")

_LEVEL_PREFIX: dict[LeadLevel, str] = {
    "hot": "🔥 Hot Lead",
    "warm": "🌤 Warm Lead",
    "cold": "❄ Cold Lead",
}


def _bits(b: VisitorBehavior, bd: ScoreBreakdown) -> Iterable[str]:
    """기여한 신호를 사람이 읽을 수 있는 토막으로 변환."""
    if bd.dwell >= 15:
        yield f"체류 {b.booth_dwell_time_sec // 60}분"
    if bd.questions:
        topics = ", ".join(b.copilot_question_topics[:2]) or "일반 문의"
        yield f"질문 {b.copilot_questions_count}회({topics})"
    if bd.pricing_bonus:
        yield "가격 토픽"
    if bd.business_card:
        yield "명함 저장"
    if bd.pamphlet:
        yield "팜플릿 다운로드"
    if bd.translation:
        yield f"통역 {b.translation_session_minutes}분"
    if bd.revisit:
        yield f"재방문 {b.revisit_count}회"
    if bd.competitor:
        yield f"경쟁사 {b.competitor_booths_visited}곳 비교"


def _mock_narrate(
    b: VisitorBehavior, bd: ScoreBreakdown, level: LeadLevel
) -> str:
    head = _LEVEL_PREFIX[level]
    bits_list = list(_bits(b, bd))
    body = ", ".join(bits_list) if bits_list else "단순 방문"
    return f"{head} · {body} → {bd.total}/100"


_SYSTEM_PROMPT = """\
당신은 MICE 행사 전시자에게 한 문장 리드 요약을 작성합니다.

INPUT (사실값 — 절대 만들지 마세요):
- score: {score}/100, level: {level}
- 신호 (스코어에 기여한 항목):
{bits}

규칙:
1. 정확히 한 문장 한국어 격식체.
2. 80자 ±20 (이모지 1개 허용).
3. 광고성 표현 금지 ("최고", "혁신적", "world-class").
4. 신호에 없는 행동을 만들지 말 것.
5. 끝에 "{score}/100" 표기 포함.

OUTPUT: 문장 한 줄. 따옴표 없이 본문만.
"""


def narrate(
    b: VisitorBehavior,
    bd: ScoreBreakdown,
    level: LeadLevel,
    *,
    mock: bool | None = None,
) -> str:
    """점수 근거를 한 문장 한국어로 풀어줌.

    USE_MOCK=true: 결정론 템플릿. mock 인자가 명시되면 우선.
    실 모드: Claude primary, GPT-4o fallback, 둘 다 실패 시 mock 으로 폴백.
    """
    use_mock = mock if mock is not None else (
        os.getenv("USE_MOCK", "false").lower() == "true"
    )
    if use_mock:
        return _mock_narrate(b, bd, level)

    bits_list = list(_bits(b, bd))
    bits_block = "\n".join(f"  - {x}" for x in bits_list) or "  - (없음)"
    prompt = _SYSTEM_PROMPT.format(
        score=bd.total, level=level, bits=bits_block,
    )

    try:
        return _call_claude(prompt)
    except Exception as exc:  # noqa: BLE001
        log.warning("claude narrator failed → gpt: %s", exc)
    try:
        return _call_gpt(prompt)
    except Exception as exc:  # noqa: BLE001
        log.exception("gpt narrator also failed → mock 폴백")
        return _mock_narrate(b, bd, level)


# ---------------------------------------------------------------------------
# LLM wrappers — lazy import
# ---------------------------------------------------------------------------
def _call_claude(prompt: str) -> str:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=200,
        system=prompt,
        messages=[
            {"role": "user", "content": "한 문장 요약을 본문만 출력하세요."}
        ],
    )
    text = msg.content[0].text.strip()  # type: ignore[index]
    return text.split("\n")[0].strip()  # 한 줄 강제


def _call_gpt(prompt: str) -> str:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.getenv("GPT_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "한 문장 요약을 본문만 출력하세요."},
        ],
        max_tokens=200,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text.split("\n")[0].strip()
