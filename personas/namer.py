"""LLM 으로 클러스터 → 페르소나 이름 + 설명 자동 생성.

입력: ClusterCentroid + 전체 평균 (대비 기준)
출력: PersonaName{name(<=12자), tagline(<=24자), description(3~4 문장),
                  feature_importance: [(feature, signed_contribution)]}

명명 규칙 (LLM prompt 에 강제):
  - 한국어, 행동 기반 명사구 (예: "적극 정보 수집형")
  - 광고/마케팅 어조 금지
  - 전체 평균 대비 두드러진 피처 2~3개 인용 — narrator 식 explainability
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from personas.clusterer import ClusterCentroid, ClusteringResult

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


@dataclass
class PersonaName:
    cluster_id: int
    name: str                                  # ≤12자
    tagline: str                               # ≤24자, 한 줄 요약
    description: str                           # 3~4문장
    feature_importance: list[tuple[str, float]]  # signed contribution
    model: str                                 # "claude" | "gpt-4o" | "mock"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM = """\
당신은 MICE 행사 행동 데이터로 페르소나 클러스터에 이름을 붙입니다.

INPUT (사실값):
  - cluster_id: {cluster_id}
  - 클러스터 크기: {size}명
  - 클러스터 평균 피처 (FEATURE_NAMES 순서):
{cluster_avg}
  - 전체 평균 (대비 기준):
{global_avg}
  - 두드러진 피처 (cluster - global, 절대값 큰 순):
{deltas}

규칙:
1. 페르소나 이름 — 한국어, 12자 이내, 행동 기반 명사구.
   예시: "적극 정보 수집형", "네트워킹 우선형", "단일 솔루션 탐색형",
         "외국인 비즈니스", "호기심 관광형".
2. tagline — 24자 이내, 한 줄 핵심 행동.
3. description — 3~4문장, 두드러진 피처를 구체 수치로 인용.
   광고성 표현 금지 ("최고", "혁신적인", "탁월한").
4. feature_importance — 두드러진 피처 상위 4개 + signed contribution
   (cluster_avg - global_avg).

OUTPUT (정확히 JSON):
{{"name": "...", "tagline": "...", "description": "...",
  "feature_importance": [["dwell_min", 3.2], ["questions_count", 2.1], ...]}}
"""


def _format_features(values: list[float], names: list[str]) -> str:
    return "\n".join(
        f"    {n}: {v}" for n, v in zip(names, values)
    )


def _signed_deltas(
    centroid: ClusterCentroid,
    global_avg: list[float],
    feature_names: list[str],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """클러스터 평균 - 전체 평균. 절대값 큰 순."""
    deltas = [
        (feature_names[i], round(centroid.avg_features[i] - global_avg[i], 3))
        for i in range(len(feature_names))
    ]
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    return deltas[:top_n]


def _format_deltas(deltas: list[tuple[str, float]]) -> str:
    return "\n".join(
        f"    {n}: {'+' if v > 0 else ''}{v}" for n, v in deltas
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def name_clusters(result: ClusteringResult) -> list[PersonaName]:
    """모든 클러스터에 이름 부여."""
    if not result.centroids:
        return []
    # 전체 평균 — 클러스터 가중 평균
    n_feat = len(result.feature_names)
    total = sum(c.size for c in result.centroids) or 1
    global_avg = [
        round(
            sum(c.avg_features[f] * c.size for c in result.centroids) / total,
            3,
        )
        for f in range(n_feat)
    ]
    return [
        _name_one(c, global_avg, result.feature_names)
        for c in result.centroids
    ]


def _name_one(
    centroid: ClusterCentroid,
    global_avg: list[float],
    feature_names: list[str],
) -> PersonaName:
    deltas = _signed_deltas(centroid, global_avg, feature_names)
    if USE_MOCK:
        return _mock_name(centroid, deltas)
    # 실 LLM 호출
    prompt = _SYSTEM.format(
        cluster_id=centroid.cluster_id,
        size=centroid.size,
        cluster_avg=_format_features(centroid.avg_features, feature_names),
        global_avg=_format_features(global_avg, feature_names),
        deltas=_format_deltas(deltas),
    )
    try:
        raw, model = _call_claude(prompt)
    except Exception:  # noqa: BLE001
        raw, model = _call_gpt(prompt)
    parsed = _parse_json(raw)
    return PersonaName(
        cluster_id=centroid.cluster_id,
        name=str(parsed.get("name", f"cluster {centroid.cluster_id}"))[:12],
        tagline=str(parsed.get("tagline", ""))[:24],
        description=str(parsed.get("description", "")),
        feature_importance=[
            (str(k), float(v)) for k, v in parsed.get("feature_importance", [])
        ],
        model=model,
    )


# ---------------------------------------------------------------------------
# Mock — 결정론적, 두드러진 피처에 따라 이름 매핑
# ---------------------------------------------------------------------------
_PATTERN_RULES: list[tuple[set[str], str, str]] = [
    # (필요한 양의 delta 피처들, 이름, tagline)
    ({"questions_count", "dwell_min"}, "적극 정보 수집형",
     "긴 체류 + 다수 질문"),
    ({"translation_min", "business_card_saved"}, "외국인 비즈니스",
     "통역 사용 + 명함 교환"),
    ({"business_card_saved", "competitor_booths_visited"}, "네트워킹 우선형",
     "명함 다수 + 경쟁사 비교"),
    ({"revisit_count", "dwell_min"}, "단일 솔루션 탐색형",
     "재방문 + 깊은 체류"),
    ({"other_booths_visited"}, "호기심 관광형",
     "다부스 빠른 둘러보기"),
    ({"pricing_topic_ratio", "questions_count"}, "구매 검토형",
     "가격 질문 집중"),
    ({"tech_topic_ratio"}, "기술 검토형",
     "기술 스펙 질문 집중"),
]


def _mock_name(
    centroid: ClusterCentroid, deltas: list[tuple[str, float]]
) -> PersonaName:
    pos_features = {n for n, v in deltas if v > 0}
    name, tagline = "일반 방문형", "평균적 행동"
    for must_have, n_, t_ in _PATTERN_RULES:
        if must_have.issubset(pos_features):
            name, tagline = n_, t_
            break
    feat_lines = ", ".join(
        f"{n} {'+' if v > 0 else ''}{v:.1f}"
        for n, v in deltas[:3]
    )
    desc = (
        f"클러스터 {centroid.cluster_id} (크기 {centroid.size}명) — "
        f"전체 평균 대비 두드러진 신호: {feat_lines}. "
        f"{tagline} 패턴이 강하게 관측됩니다."
    )
    return PersonaName(
        cluster_id=centroid.cluster_id,
        name=name[:12],
        tagline=tagline[:24],
        description=desc,
        feature_importance=[(n, float(v)) for n, v in deltas[:4]],
        model="mock",
    )


# ---------------------------------------------------------------------------
# LLM wrappers + JSON parser
# ---------------------------------------------------------------------------
def _call_claude(prompt: str) -> tuple[str, str]:
    from anthropic import Anthropic  # type: ignore
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=400,
        system=prompt,
        messages=[{"role": "user", "content": "Generate persona JSON only."}],
    )
    return msg.content[0].text, "claude"  # type: ignore[index]


def _call_gpt(prompt: str) -> tuple[str, str]:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.getenv("GPT_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate persona JSON only."},
        ],
        max_tokens=400,
    )
    return resp.choices[0].message.content or "", "gpt-4o"


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict:
    import json
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))
