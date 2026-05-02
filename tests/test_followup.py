"""followup/{generator,templates,sender}.py mock 경로 검증."""
from __future__ import annotations

import os

os.environ["USE_MOCK"] = "true"

import pytest  # noqa: E402

from followup.generator import (  # noqa: E402
    ExhibitorContext,
    VisitorContext,
    build_prompt,
    generate_ab_pair,
    generate_draft,
)
from followup.sender import send  # noqa: E402
from followup.templates import get_system_prompt  # noqa: E402


_VISITOR = VisitorContext(
    visitor_name="박재은",
    visited_at="2026-06-19",
    booth_name="Lumen Labs",
    dwell_minutes=7,
    copilot_questions=["가격이 어떻게 되나요?", "0.05mm 정확도 보장 방법?"],
    translation_summary="3D 스캐너 정확도 vs 속도 트레이드오프 논의",
    competitor_booths=["NovaSight"],
    pamphlet_pages_viewed=[12, 13],
)
_EXHIBITOR = ExhibitorContext(
    exhibitor_name="김현호",
    company_name="Lumen Labs",
    cta_calendly_url="https://cal.com/lumen/demo",
    cta_pdf_url=None,
    signature="MICE 행사장 B2 부스",
)


@pytest.mark.parametrize("lang,tone", [
    ("ko", "formal"), ("ko", "balanced"), ("ko", "casual"),
    ("en", "formal"), ("en", "balanced"), ("en", "casual"),
])
def test_template_lookup_for_all_pairs(lang, tone):
    prompt = get_system_prompt(lang, tone)
    assert "{context}" in prompt
    assert "{exhibitor_context}" in prompt
    assert "{common_rules}" in prompt


def test_unknown_pair_raises():
    with pytest.raises(ValueError):
        get_system_prompt("ko", "absurd")  # type: ignore[arg-type]


def test_build_prompt_includes_visitor_context():
    p = build_prompt(visitor=_VISITOR, exhibitor=_EXHIBITOR,
                     lang="ko", tone="balanced")
    assert "박재은" in p
    assert "Lumen Labs" in p
    assert "가격이 어떻게 되나요?" in p
    assert "https://cal.com/lumen/demo" in p
    # COMMON_RULES 가 자동 주입됐는지
    assert "ONE call-to-action" in p


def test_generate_draft_mock_korean():
    d = generate_draft(visitor=_VISITOR, exhibitor=_EXHIBITOR,
                       lang="ko", tone="balanced")
    assert d.model == "mock"
    assert d.tone == "balanced"
    assert "박재은" in d.body
    assert "Lumen Labs" in d.subject or "Lumen Labs" in d.body
    assert d.cta_label
    # 광고성 표현 금지
    for banned in ("최고", "혁신적", "world-class"):
        assert banned not in d.body


def test_generate_ab_pair_returns_two_distinct_tones():
    a, b = generate_ab_pair(
        visitor=_VISITOR, exhibitor=_EXHIBITOR, lang="ko",
        tone_a="formal", tone_b="casual",
    )
    assert a.tone == "formal"
    assert b.tone == "casual"


def test_send_mock_returns_accepted():
    d = generate_draft(visitor=_VISITOR, exhibitor=_EXHIBITOR,
                       lang="en", tone="formal")
    res = send(draft=d, to_email="visitor@example.com",
               from_email="exhibitor@lumen.io",
               ab_arm="A", event_id="ev_2026_06", visitor_hash="abcd1234")
    assert res.accepted is True
    assert res.status_code == 202


def test_korean_mock_no_emoji_in_formal():
    d = generate_draft(visitor=_VISITOR, exhibitor=_EXHIBITOR,
                       lang="ko", tone="formal")
    # mock 한국어는 emoji 없음 (공통 규칙)
    for emoji in ("👋", "🙌", "🎉"):
        assert emoji not in d.body
