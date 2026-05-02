"""WebSocket /translation/session 엔드포인트 검증 (Sprint 1 mock).

운영자/참가자 두 폰을 각각 TestClient.websocket_connect 로 흉내내서:
  1. session.start 양쪽 → session.ready 양쪽에 도착
  2. audio.chunk → segment.partial 상대편에 도착
  3. audio.end → segment.final 양쪽 + tts.audio 상대편에 도착
"""
from __future__ import annotations

import json
import os

os.environ["USE_MOCK"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from translation.session_manager import SessionRole  # noqa: E402

client = TestClient(app)


def _start(ws, *, session_id: str, role: str, booth_id: str = "lumen",
           lang_pair: list[str] | None = None) -> None:
    ws.send_json({
        "type": "session.start",
        "payload": {
            "session_id": session_id,
            "booth_id": booth_id,
            "lang_pair": lang_pair or ["ko", "en"],
            "role": role,
        },
    })


def test_health_endpoint():
    r = client.get("/translation/healthz")
    assert r.status_code == 200
    assert r.json()["module"] == "translation"
    assert r.json()["mock"] is True


def test_pairing_emits_session_ready_to_both():
    sid = "s_pair"
    with client.websocket_connect("/translation/session") as op, \
         client.websocket_connect("/translation/session") as vi:

        _start(op, session_id=sid, role="operator")
        # 첫 도착 시점에는 한 쪽만 — paired=False
        msg_op_first = op.receive_json()
        assert msg_op_first["type"] == "session.ready"
        assert msg_op_first["paired"] is False

        _start(vi, session_id=sid, role="visitor")
        # visitor 도착으로 paired=True 가 양쪽에 broadcast
        msg_vi = vi.receive_json()
        msg_op_2 = op.receive_json()
        for m in (msg_op_2, msg_vi):
            assert m["type"] == "session.ready"
            assert m["paired"] is True
            assert m["lang_pair"] == ["ko", "en"]

        op.send_json({"type": "session.end", "payload": {}})
        # session.summary broadcast 도착
        op.receive_json()  # summary
        vi.receive_json()  # summary


def test_audio_end_emits_segment_final_and_tts():
    sid = "s_audio"
    with client.websocket_connect("/translation/session") as op, \
         client.websocket_connect("/translation/session") as vi:
        _start(op, session_id=sid, role="operator")
        op.receive_json()  # session.ready (unpaired)
        _start(vi, session_id=sid, role="visitor")
        vi.receive_json()  # session.ready (paired)
        op.receive_json()  # session.ready (paired)

        # 운영자가 한 segment 끝
        op.send_json({"type": "audio.end", "payload": {}})

        # segment.final 은 양쪽 모두 받음
        op_msg = op.receive_json()
        vi_msg = vi.receive_json()
        for m in (op_msg, vi_msg):
            assert m["type"] == "segment.final"
            assert m["speaker"] == "operator"
            assert m["source_lang"] == "ko"
            assert m["target_lang"] == "en"
            assert "Lumen Labs" in m["source_text"] or "안녕" in m["source_text"]

        # tts.audio 는 상대방(visitor) 한테만
        tts = vi.receive_json()
        assert tts["type"] == "tts.audio"
        assert tts["voice_id"] == "mock_en"

        op.send_json({"type": "session.end", "payload": {}})


def test_audio_chunk_broadcasts_partial():
    sid = "s_chunk"
    with client.websocket_connect("/translation/session") as op, \
         client.websocket_connect("/translation/session") as vi:
        _start(op, session_id=sid, role="operator")
        op.receive_json()
        _start(vi, session_id=sid, role="visitor")
        vi.receive_json()
        op.receive_json()

        op.send_json({
            "type": "audio.chunk",
            "payload": {"data": "AAAA", "format": "opus", "seq": 0},
        })
        # partial 양쪽 모두 도착
        for sock in (op, vi):
            msg = sock.receive_json()
            assert msg["type"] == "segment.partial"
            assert msg["speaker"] == "operator"
            assert msg["lang"] == "ko"

        op.send_json({"type": "session.end", "payload": {}})


def test_invalid_role_returns_schema_error():
    with client.websocket_connect("/translation/session") as ws:
        ws.send_json({
            "type": "session.start",
            "payload": {
                "session_id": "s_bad",
                "booth_id": "lumen",
                "lang_pair": ["ko", "en"],
                "role": "bystander",  # invalid
            },
        })
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "schema"
