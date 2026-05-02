"""참가자 프로필 → 임베딩.

GDPR 게이트: consent.matching=True · consent.location=True 둘 다 만족해야 임베딩 생성.
PII (이름·이메일·전화) 는 입력 단계에서 strip 후 텍스트만 사용.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass

from matching.governance import Consent, is_eligible_for_matching

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
EMBED_DIM = 1536


@dataclass
class VisitorProfile:
    visitor_hash: str
    role: str | None
    industry: str | None
    interests: list[str]               # PII-free 키워드만
    embedding: list[float]
    model: str
    consent_snapshot: dict             # 어떤 동의 상태에서 임베딩 됐는지 audit


class ConsentDeniedError(RuntimeError):
    """opt-in 안 된 visitor 의 임베딩 시도. 호출 측은 412 반환 권장."""


def _strip_pii(text: str) -> str:
    """이메일·전화 패턴 제거 (이름은 클라이언트에서 strip 후 호출 가정)."""
    import re
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    text = re.sub(r"\b\d{2,3}-\d{3,4}-\d{4}\b", "[phone]", text)
    text = re.sub(r"\b01\d{8,9}\b", "[phone]", text)
    return text


def _fake_embedding(text: str) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (h * ((EMBED_DIM // len(h)) + 1))[:EMBED_DIM]
    return [(b / 127.5) - 1.0 for b in raw]


def profile_visitor(
    *,
    visitor_hash: str,
    role: str | None,
    industry: str | None,
    interests: list[str],
    consent: Consent,
) -> VisitorProfile:
    if not is_eligible_for_matching(consent):
        raise ConsentDeniedError(
            "visitor not opted-in for matching+location"
        )

    sanitized = [_strip_pii(s) for s in interests]
    text = " | ".join(filter(None, [
        role or "",
        industry or "",
        ", ".join(sanitized),
    ]))

    if USE_MOCK:
        vec = _fake_embedding(text)
        model = "mock"
    else:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.embeddings.create(
            model="text-embedding-3-small", input=text,
        )
        vec = list(resp.data[0].embedding)
        model = "ada"

    return VisitorProfile(
        visitor_hash=visitor_hash,
        role=role,
        industry=industry,
        interests=sanitized,
        embedding=vec,
        model=model,
        consent_snapshot={
            "matching": consent.matching,
            "location": consent.location,
            "analytics": consent.analytics,
        },
    )


def to_dict(p: VisitorProfile) -> dict:
    return asdict(p)
