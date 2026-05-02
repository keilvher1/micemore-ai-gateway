"""copilot_system.build_system_prompt() — 출처 주입 + 다국어 + 글자수 제한 검증."""

from __future__ import annotations

from prompts.copilot_system import LANG_NAMES, build_system_prompt


SOURCES = [
    {
        "id": "S1",
        "doc": "Lumen Labs Catalog.pdf",
        "page": 12,
        "text": "Lumen Labs makes a real-time 3D inspection sensor.",
    },
    {
        "id": "S2",
        "doc": "Demo Script.md",
        "page": 1,
        "text": "Demos run every 30 minutes at booth B12.",
    },
]


def test_includes_booth_id_and_lang():
    prompt = build_system_prompt(booth_id="lumen", target_lang="en", sources=SOURCES)
    assert "lumen" in prompt
    assert LANG_NAMES["en"] in prompt


def test_supports_all_known_languages():
    for lang in ("auto", "ko", "en", "ja", "zh"):
        prompt = build_system_prompt(booth_id="x", target_lang=lang, sources=SOURCES)
        assert LANG_NAMES[lang] in prompt


def test_unknown_language_falls_back_to_auto():
    prompt = build_system_prompt(booth_id="x", target_lang="fr", sources=SOURCES)
    assert LANG_NAMES["auto"] in prompt


def test_injects_source_blocks():
    prompt = build_system_prompt(booth_id="lumen", target_lang="en", sources=SOURCES)
    assert "[S1]" in prompt
    assert "[S2]" in prompt
    assert "Lumen Labs Catalog.pdf" in prompt
    assert "p=12" in prompt


def test_truncates_long_chunks():
    long_text = "x" * 1500
    prompt = build_system_prompt(
        booth_id="x",
        target_lang="en",
        sources=[{"id": "S1", "doc": "d", "page": 1, "text": long_text}],
    )
    # 600자 이후 ellipsis 처리 확인
    assert "…" in prompt


def test_handles_empty_sources():
    prompt = build_system_prompt(booth_id="x", target_lang="en", sources=[])
    assert "no sources retrieved" in prompt
