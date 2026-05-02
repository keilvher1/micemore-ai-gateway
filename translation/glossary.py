"""MICE 도메인 사전 로더.

`data/glossary.yaml` + `data/companies_seed.yaml` 을 읽어 Claude system prompt 에
주입할 텍스트 블록을 만든다. 운영자가 YAML 만 갱신해도 코드 변경 없이 반영됨.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Glossary:
    schema_version: int
    preserved_acronyms: list[str]
    preserve_patterns: list[str]
    tone_guides: dict[str, str]
    term_pairs: list[dict[str, str]] = field(default_factory=list)
    companies: list[dict[str, Any]] = field(default_factory=list)


@lru_cache(maxsize=1)
def load_glossary() -> Glossary:
    """기본 위치(`data/`)에서 사전 로드. 결과는 캐시."""
    g_path = _DATA_DIR / "glossary.yaml"
    c_path = _DATA_DIR / "companies_seed.yaml"
    g_raw: dict[str, Any] = yaml.safe_load(g_path.read_text(encoding="utf-8"))
    c_raw: dict[str, Any] = yaml.safe_load(c_path.read_text(encoding="utf-8"))
    return Glossary(
        schema_version=int(g_raw.get("schema_version", 1)),
        preserved_acronyms=list(g_raw.get("preserved_acronyms", [])),
        preserve_patterns=list(g_raw.get("preserve_patterns", [])),
        tone_guides=dict(g_raw.get("tone_guides", {})),
        term_pairs=list(g_raw.get("term_pairs", [])),
        companies=list(c_raw.get("companies", [])),
    )


def reset_cache() -> None:
    """테스트 또는 hot-reload 용."""
    load_glossary.cache_clear()


def build_system_block(src_lang: str, tgt_lang: str) -> str:
    """언어쌍에 맞춰 Claude system prompt 의 'GLOSSARY' 섹션을 생성."""
    g = load_glossary()
    pair = f"{src_lang}_to_{tgt_lang}"
    tone = g.tone_guides.get(pair, "- Maintain professional business register.")

    acronyms = ", ".join(g.preserved_acronyms)
    company_names: list[str] = []
    for c in g.companies:
        company_names.append(c["name"])
        company_names.extend(c.get("aliases", []))
    companies_block = ", ".join(sorted(set(company_names)))

    term_lines = []
    for pair_dict in g.term_pairs:
        ko = pair_dict.get("ko", "")
        en = pair_dict.get("en", "")
        if src_lang == "ko" and tgt_lang == "en":
            term_lines.append(f"  • {ko} → {en}")
        elif src_lang == "en" and tgt_lang == "ko":
            term_lines.append(f"  • {en} → {ko}")
    term_section = ("\nDOMAIN TERMS (prefer these mappings when natural):\n"
                    + "\n".join(term_lines)) if term_lines else ""

    return f"""\
GLOSSARY:
- Preserve as-is (do NOT translate or transliterate): {acronyms}
- Preserve as-is (company / product names): {companies_block}
- Preserve as-is: version numbers, units (mm, µm, GHz, fps, …).
TONE:
{tone}{term_section}
"""
