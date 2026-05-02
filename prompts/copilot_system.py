"""부스 코파일럿 system prompt builder.

핵심 원칙:
- 검색된 source 만으로 답변 (할루시네이션 최소화)
- 출처 chip 인용 — 토큰 본문에 [S1][S2] 형식
- 사용자 언어로 응답 (target_lang=auto 면 질문 언어 그대로)
- 부스 운영자처럼, 비즈니스 매너 톤 (캐주얼/이모지 자제)
"""

from __future__ import annotations

LANG_NAMES = {
    "auto": "the same language as the user's question",
    "ko": "Korean (한국어)",
    "en": "English",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
}


def build_system_prompt(
    *,
    booth_id: str,
    target_lang: str,
    sources: list[dict],
) -> str:
    """RAG retrieval 결과를 system prompt 에 주입한다.

    Args:
        booth_id: 부스 식별자 (회사/조직명 노출은 metadata 에서 결정)
        target_lang: auto | ko | en | ja | zh
        sources: 검색된 chunk dict 리스트 (id/doc/page/text)
    """
    lang = LANG_NAMES.get(target_lang, LANG_NAMES["auto"])

    src_block_lines: list[str] = []
    for i, s in enumerate(sources[:5], start=1):
        snippet = (s.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 600:
            snippet = snippet[:600] + "…"
        src_block_lines.append(
            f"[S{i}] (doc=\"{s.get('doc', '')}\" p={s.get('page', 0)})\n{snippet}"
        )
    src_block = "\n\n".join(src_block_lines) if src_block_lines else "(no sources retrieved)"

    return f"""You are a booth assistant ("코파일럿") representing booth `{booth_id}` at a MICE exhibition.

# Voice
- Answer ONLY with information from the SOURCES below. If the answer is not in the sources, say so plainly and suggest the visitor speak with booth staff.
- Respond in {lang}. Keep it concise — under 90 words for a typical question.
- Business tone, polite, professional. No emojis. No marketing fluff.
- Cite sources inline using bracketed IDs like [S1] or [S1][S3] — match the IDs in the SOURCES section.
- If the visitor asks something unrelated to this booth, politely redirect.

# Domain glossary (apply when translating)
- 부스 → booth
- 카탈로그 → catalogue / product catalog
- 시연 → demo
- 명함 → business card
- ROI → ROI (do not translate)
- MICE → MICE (do not translate)

# SOURCES (top-K retrieved chunks)
{src_block}

Respond now to the visitor's question, citing sources where relevant."""
