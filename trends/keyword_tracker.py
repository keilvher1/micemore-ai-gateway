"""TF-IDF 변화율 + emerging keyword detection.

분기별 키워드 빈도를 받아:
  - 직전 분기 대비 변화율 (절대/상대)
  - emerging: 직전 0건 → 신규 등장 + 빈도 임계 초과
  - declining: 50% 이상 감소

순수 Python (numpy 없이) — 베타 단계 외부 의존성 절약.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

EMERGING_MIN_COUNT = 10        # emerging 으로 분류할 최소 빈도
DECLINE_THRESHOLD = 0.50       # -50% 이상 감소 → declining


@dataclass
class KeywordChange:
    keyword: str
    prev_count: int
    curr_count: int
    delta: int                  # curr - prev
    growth_ratio: float | None  # (curr - prev) / max(prev, 1) — prev=0 면 None
    label: str                  # "emerging" | "rising" | "stable" | "declining"


# ---------------------------------------------------------------------------
# Tokenization (light) — 도메인 사전 + 한국어/영어 대응
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "in", "of", "on", "at",
    "is", "are", "was", "with", "by", "as",
    "그리고", "또는", "있다", "없다", "이다", "아니다",
}


def tokenize(text: str) -> list[str]:
    """소문자 영문 단어 + 한국어 명사 후보 (2자+)."""
    import re
    out: list[str] = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9]+|[가-힣]{2,}", text):
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        out.append(low if tok.isascii() else tok)
    return out


def term_frequency(documents: list[str]) -> Counter:
    """문서 리스트 → 토큰 빈도. (TF — IDF 는 분기 비교가 더 자연스러워 생략)"""
    c: Counter = Counter()
    for doc in documents:
        c.update(tokenize(doc))
    return c


# ---------------------------------------------------------------------------
# Quarter-over-quarter delta
# ---------------------------------------------------------------------------
def compare_quarters(
    *,
    prev_docs: list[str],
    curr_docs: list[str],
    top_n: int = 50,
) -> list[KeywordChange]:
    """직전 분기 대비 키워드 변화. 빈도 상위 top_n 만 반환."""
    prev = term_frequency(prev_docs)
    curr = term_frequency(curr_docs)
    keys = set(prev) | set(curr)

    changes: list[KeywordChange] = []
    for k in keys:
        p, c = prev.get(k, 0), curr.get(k, 0)
        delta = c - p
        if p == 0:
            ratio: float | None = None
            label = ("emerging" if c >= EMERGING_MIN_COUNT
                     else "rising" if c > 0
                     else "stable")
        else:
            ratio = (c - p) / p
            if ratio >= 1.0:
                label = "rising"
            elif ratio <= -DECLINE_THRESHOLD:
                label = "declining"
            else:
                label = "stable"
        changes.append(
            KeywordChange(
                keyword=k,
                prev_count=p,
                curr_count=c,
                delta=delta,
                growth_ratio=round(ratio, 3) if ratio is not None else None,
                label=label,
            )
        )
    # 상위 = curr 빈도 큰 순서
    changes.sort(key=lambda x: x.curr_count, reverse=True)
    return changes[:top_n]


def emerging_keywords(
    changes: list[KeywordChange], min_count: int = EMERGING_MIN_COUNT
) -> list[KeywordChange]:
    return [c for c in changes
            if c.label == "emerging" and c.curr_count >= min_count]


def declining_keywords(
    changes: list[KeywordChange]
) -> list[KeywordChange]:
    return [c for c in changes if c.label == "declining"]
