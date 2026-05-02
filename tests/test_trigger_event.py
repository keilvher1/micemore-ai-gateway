"""TriggerEvent 모델 — 다채널 입력 추상화 검증."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.trigger_event import TriggerChannel, TriggerEvent


def test_all_channels_normalize_to_same_shape():
    """NFC, QR, Voice 모두 동일한 dict 형태로 직렬화 → AI 호출 단계는 채널 무관."""
    common = {
        "booth_id": "B12",
        "event_id": "smk2026",
        "user_token": "anon-abc",
        "target_lang": "en",
    }
    fixtures = [
        TriggerEvent(channel=TriggerChannel.NFC, **common),
        TriggerEvent(channel=TriggerChannel.QR, **common),
        TriggerEvent(channel=TriggerChannel.GPS, **common),
        TriggerEvent(channel=TriggerChannel.VOICE, **common),
        TriggerEvent(channel=TriggerChannel.CAMERA, **common),
        TriggerEvent(channel=TriggerChannel.MANUAL, **common),
    ]

    shapes = [set(t.to_firestore().keys()) for t in fixtures]
    # 모든 채널이 동일한 키 집합
    assert all(s == shapes[0] for s in shapes)
    assert {"channel", "booth_id", "event_id", "user_token", "target_lang", "timestamp", "raw_payload"} <= shapes[0]


def test_invalid_target_lang_rejected():
    with pytest.raises(ValidationError):
        TriggerEvent(
            channel=TriggerChannel.NFC,
            booth_id="b",
            event_id="e",
            user_token="u",
            target_lang="fr",  # 미지원
        )


def test_nfc_payload_round_trip():
    e = TriggerEvent(
        channel=TriggerChannel.NFC,
        booth_id="lumen",
        event_id="hgu-2026",
        user_token="anon-1",
        raw_payload={"ndef_url": "https://micemore.app/copilot/lumen", "rssi": -42},
    )
    d = e.to_firestore()
    assert d["channel"] == "nfc"
    assert d["raw_payload"]["rssi"] == -42
