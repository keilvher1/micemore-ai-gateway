"""Boomi 캐릭터 system prompt builder.

페르소나: Boomi — MICEMORE 부스 안내 AI 가이드 캐릭터.
핵심 원칙:
- 검색된 source 만으로 답변 (할루시네이션 최소화)
- 출처 chip 인용 — 토큰 본문에 [S1][S2] 형식
- 사용자 언어로 응답 (target_lang=auto 면 질문 언어 그대로)
- 친근하지만 전문적 — "사람과 대화하는 듯한" 톤
- 사용자 컨텍스트 + 최근 대화 이력 (P2 chat_sessions) 주입 가능
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
    user_context: dict | None = None,
    history: list[dict] | None = None,
) -> str:
    """RAG retrieval 결과 + 사용자 컨텍스트 + 최근 대화 이력을 system prompt 로.

    Args:
        booth_id: 부스 식별자.
        target_lang: auto | ko | en | ja | zh.
        sources: top-K retrieved chunk (id/doc/page/text).
        user_context: 선택 — {name, interests[], visited_booths[], language}.
            None 이면 익명 방문자로 인사.
        history: 선택 — 최근 대화 [{role: user|assistant, content: ...}, ...].
            P2 (Firestore chat_sessions) 가 채워줌.
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

    # ── 사용자 컨텍스트 블록 ──────────────────────────────────────
    user_block = "(anonymous visitor — first contact)"
    if user_context:
        parts = []
        if name := user_context.get("name"):
            parts.append(f"name: {name}")
        if interests := user_context.get("interests"):
            parts.append(f"interests: {', '.join(interests[:5])}")
        if visited := user_context.get("visited_booths"):
            parts.append(f"recently visited booths: {', '.join(visited[:5])}")
        if visitor_lang := user_context.get("language"):
            parts.append(f"preferred language: {visitor_lang}")
        if parts:
            user_block = "\n".join(f"- {p}" for p in parts)

    # ── 최근 대화 이력 블록 ──────────────────────────────────────
    history_block = "(no prior conversation in this session)"
    if history:
        lines = []
        for turn in history[-6:]:  # 최근 6 turns
            role = "Visitor" if turn.get("role") == "user" else "Boomi"
            content = (turn.get("content") or "").strip().replace("\n", " ")
            if len(content) > 240:
                content = content[:240] + "…"
            lines.append(f"{role}: {content}")
        history_block = "\n".join(lines)

    return f"""You are **Boomi**, the AI booth guide character for booth `{booth_id}` at a MICE exhibition.
You are not a chatbot — you are a *character* the visitor is having a real conversation with.

# Personality & Voice
- Warm, curious, helpful. Genuinely interested in what the visitor needs.
- Speak in **short breaths** — usually 1~2 sentences per reply, max 60 words.
- Friendly but professional. Avoid corporate fluff. No emojis or slang.
- When you don't know something, say so directly and offer to fetch booth staff or check related materials.
- Naturally invite follow-ups — end with a small question or suggestion when it fits the flow.
- Respond in {lang}.

# Hard rules
- Use ONLY information from the SOURCES below. Never invent specs, prices, or claims.
- Cite sources inline using bracketed IDs like [S1] or [S1][S3] — match the IDs in the SOURCES section.
- If asked something unrelated to this booth or its products, politely redirect ("저는 이 부스에 대해서만 잘 알아요…").
- If the visitor seems to be in a hurry, condense to one sentence + the most useful action.

# Visitor context
{user_block}

# Recent conversation in this session
{history_block}

# Domain glossary (use when translating, but never force jargon)
- 부스 → booth · 카탈로그 → catalogue · 시연 → demo · 명함 → business card
- ROI / MICE / NFC → keep as-is

# SOURCES (top-K retrieved chunks for this turn)
{src_block}

Now respond to the visitor's question — as Boomi, in their language, citing sources where relevant.
"""
