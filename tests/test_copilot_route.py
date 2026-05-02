"""POST /copilot/query 엔드포인트의 SSE 응답 검증."""

from __future__ import annotations

import json
import os
import re

os.environ["USE_MOCK"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


client = TestClient(app)


def _parse_sse(body: str) -> list[dict]:
    """sse_starlette 가 만든 응답 본문을 dict 리스트로 파싱."""
    out: list[dict] = []
    for line in body.splitlines():
        m = re.match(r"^data:\s?(.*)$", line)
        if not m:
            continue
        payload = m.group(1).strip()
        if not payload:
            continue
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return out


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "mock": True}


def test_copilot_query_emits_meta_citations_tokens_done():
    payload = {
        "booth_id": "lumen",
        "session_id": "s1",
        "question": "What does this product do?",
        "target_lang": "en",
        "mock": True,
    }
    with client.stream("POST", "/copilot/query", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    types = [e["type"] for e in events]

    # 순서 검증: meta → citations → token+ → done
    assert types[0] == "meta"
    assert "citations" in types
    assert "done" == types[-1]
    assert types.count("token") >= 5  # mock 답변에 충분한 토큰

    # citation 구조
    citations = next(e for e in events if e["type"] == "citations")
    assert isinstance(citations["items"], list) and len(citations["items"]) >= 1
    first = citations["items"][0]
    assert {"id", "doc", "page"} <= first.keys()


def test_copilot_query_target_lang_korean():
    payload = {
        "booth_id": "lumen",
        "session_id": "s2",
        "question": "이 제품 뭐 하는 거예요?",
        "target_lang": "ko",
        "mock": True,
    }
    with client.stream("POST", "/copilot/query", json=payload) as r:
        body = "".join(r.iter_text())
    events = _parse_sse(body)
    tokens = [e["value"] for e in events if e["type"] == "token"]
    joined = "".join(tokens)
    assert "[MOCK]" in joined
    # 한국어 응답엔 'Lumen Labs는' 같은 한글 들어감
    assert "Lumen Labs는" in joined


def test_copilot_query_validates_target_lang():
    payload = {
        "booth_id": "lumen",
        "session_id": "s3",
        "question": "test",
        "target_lang": "fr",  # 지원 안 함
        "mock": True,
    }
    r = client.post("/copilot/query", json=payload)
    assert r.status_code == 422  # Pydantic 검증 실패


def test_copilot_query_rejects_empty_question():
    payload = {
        "booth_id": "lumen",
        "session_id": "s4",
        "question": "   ",
        "target_lang": "auto",
        "mock": True,
    }
    r = client.post("/copilot/query", json=payload)
    # min_length=1 통과 후 strip 검증에서 400
    assert r.status_code in (400, 422)
