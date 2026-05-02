"""전시자 ICP (Ideal Customer Profile) → 구조화 + 임베딩.

입력 (전시자 자유 텍스트): "바이오테크 R&D 책임자, 100~1000명 회사,
                          mass spec 또는 NMR 검토 중"
출력:
  ExhibitorICP{booth_id, target_roles, target_industries, target_company_size,
               target_keywords, embedding}

mock 모드: 결정론적 키워드 추출 + sha256 기반 fake 임베딩.
실 모드: Claude 구조화 + ada-002 임베딩 (lazy import).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
EMBED_DIM = 1536  # ada-002 와 동일


@dataclass
class ExhibitorICP:
    booth_id: str
    target_roles: list[str]
    target_industries: list[str]
    target_company_size: str          # "1-50" | "50-500" | "500-5000" | "5000+"
    target_keywords: list[str]
    raw_text: str
    embedding: list[float]            # 1536-d
    model: str                        # "claude+ada" | "mock"


# ---------------------------------------------------------------------------
# Mock — 룰 기반 키워드 추출
# ---------------------------------------------------------------------------
_INDUSTRY_KEYWORDS = {
    "biotech": ["바이오", "biotech", "pharma", "phaarma", "생명공학"],
    "fintech": ["fintech", "핀테크", "은행", "결제"],
    "manufacturing": ["제조", "manufacturing", "factory", "공장"],
    "automotive": ["자동차", "automotive", "EV", "전기차"],
    "healthcare": ["의료", "healthcare", "헬스케어", "병원"],
    "ai": ["AI", "ML", "머신러닝", "인공지능", "LLM"],
    "robotics": ["로봇", "robotics", "automation", "자동화"],
    "semiconductor": ["반도체", "semiconductor", "wafer", "fab"],
}
_ROLE_KEYWORDS = {
    "cto": ["CTO", "기술이사"],
    "ceo": ["CEO", "대표"],
    "vp": ["VP", "부사장", "head of"],
    "rd": ["R&D", "연구", "개발"],
    "engineer": ["engineer", "엔지니어", "개발자"],
    "buyer": ["구매", "buyer", "purchasing"],
    "manager": ["manager", "매니저", "팀장"],
}
_SIZE_PATTERNS = [
    (re.compile(r"5,?000\+|over 5,?000"), "5000+"),
    (re.compile(r"500[\s\-~]+5,?000"), "500-5000"),
    (re.compile(r"50[\s\-~]+500"), "50-500"),
    (re.compile(r"1[\s\-~]+50|under 50"), "1-50"),
    (re.compile(r"100[\s\-~]+1,?000"), "50-500"),  # 흔한 표현 매핑
]


def _extract_industries(text: str) -> list[str]:
    out = []
    low = text.lower()
    for ind, kws in _INDUSTRY_KEYWORDS.items():
        if any(kw.lower() in low for kw in kws):
            out.append(ind)
    return out


def _extract_roles(text: str) -> list[str]:
    out = []
    low = text.lower()
    for role, kws in _ROLE_KEYWORDS.items():
        if any(kw.lower() in low for kw in kws):
            out.append(role)
    return out


def _extract_size(text: str) -> str:
    for pat, label in _SIZE_PATTERNS:
        if pat.search(text):
            return label
    return "any"


def _extract_keywords(text: str) -> list[str]:
    # 영문 PascalCase, 한국어 명사 후보, 약어 추출
    tokens = re.findall(r"[A-Z][a-zA-Z0-9]+|[가-힣]{2,}", text)
    # 중복 제거 + 길이 제한
    seen: set[str] = set()
    out = []
    for t in tokens:
        if t in seen or len(t) < 2:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= 12:
            break
    return out


def _fake_embedding(text: str) -> list[float]:
    """SHA256 기반 결정론 1536-d 벡터. 실제 의미 없음, 같은 입력→같은 출력."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (h * ((EMBED_DIM // len(h)) + 1))[:EMBED_DIM]
    return [(b / 127.5) - 1.0 for b in raw]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def embed_icp(*, booth_id: str, raw_text: str) -> ExhibitorICP:
    if USE_MOCK:
        return ExhibitorICP(
            booth_id=booth_id,
            target_roles=_extract_roles(raw_text),
            target_industries=_extract_industries(raw_text),
            target_company_size=_extract_size(raw_text),
            target_keywords=_extract_keywords(raw_text),
            raw_text=raw_text,
            embedding=_fake_embedding(raw_text),
            model="mock",
        )
    # 실 호출 — Claude structure + OpenAI ada
    structured = _claude_structure(raw_text)
    vec = _ada_embed(raw_text + "\n" + json.dumps(structured, ensure_ascii=False))
    return ExhibitorICP(
        booth_id=booth_id,
        target_roles=structured.get("roles", []),
        target_industries=structured.get("industries", []),
        target_company_size=structured.get("company_size", "any"),
        target_keywords=structured.get("keywords", []),
        raw_text=raw_text,
        embedding=vec,
        model="claude+ada",
    )


# ---------------------------------------------------------------------------
# 실 모드 — lazy import
# ---------------------------------------------------------------------------
_STRUCTURE_PROMPT = """\
You receive a Korean/English free-text description of an Ideal Customer Profile
for a MICE booth. Extract structured fields. Return ONLY JSON:

{{
  "roles": ["..."],            // 1-5 short labels
  "industries": ["..."],       // 1-3
  "company_size": "1-50 | 50-500 | 500-5000 | 5000+ | any",
  "keywords": ["..."]          // 3-10 concrete search keywords
}}

Description:
{text}
"""


def _claude_structure(text: str) -> dict:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=400,
        messages=[
            {"role": "user", "content": _STRUCTURE_PROMPT.format(text=text)}
        ],
    )
    raw = resp.content[0].text  # type: ignore[index]
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def _ada_embed(text: str) -> list[float]:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return list(resp.data[0].embedding)


def to_dict(icp: ExhibitorICP) -> dict:
    return asdict(icp)
