"""MICE 사전 로더 + system block 검증."""
from __future__ import annotations

import pytest

from translation.glossary import build_system_block, load_glossary, reset_cache


@pytest.fixture(autouse=True)
def _reset():
    reset_cache()
    yield
    reset_cache()


def test_loads_yaml_files():
    g = load_glossary()
    assert g.schema_version == 1
    assert "MICE" in g.preserved_acronyms
    assert "PCO" in g.preserved_acronyms
    assert any(c["name"] == "Lumen Labs" for c in g.companies)
    assert len(g.companies) >= 30


def test_acronyms_appear_in_system_block():
    block = build_system_block("ko", "en")
    assert "MICE" in block
    assert "PCO" in block
    assert "ROI" in block


def test_companies_appear_in_system_block():
    block = build_system_block("ko", "en")
    assert "Samsung" in block
    assert "KOTRA" in block
    assert "Lumen Labs" in block


def test_tone_guide_per_pair():
    ko_en = build_system_block("ko", "en")
    en_ko = build_system_block("en", "ko")
    assert "합쇼체" in en_ko
    assert "business neutral" in ko_en
    assert ko_en != en_ko


def test_term_pairs_direction_aware():
    ko_en = build_system_block("ko", "en")
    assert "전시면적 → exhibition floor area" in ko_en
    en_ko = build_system_block("en", "ko")
    assert "exhibition floor area → 전시면적" in en_ko


def test_unknown_pair_falls_back_gracefully():
    block = build_system_block("ja", "en")  # tone_guide 미정 쌍
    assert "professional business register" in block
