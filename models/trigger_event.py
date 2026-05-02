"""다채널 입력 추상화 — TriggerEvent 모델.

핵심 원칙:
NFC, QR, GPS, Voice, Camera, Manual 모두 단일 TriggerEvent 로 정규화하고
이후 AI 코파일럿 호출은 채널 무관 동일 처리한다. NFC 는 6채널 중 1개일 뿐.

Firestore `user_action` 컬렉션에 그대로 매핑된다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TriggerChannel(str, Enum):
    NFC = "nfc"
    QR = "qr"
    GPS = "gps"
    VOICE = "voice"
    CAMERA = "camera"
    MANUAL = "manual"


class TriggerEvent(BaseModel):
    """모든 입력 채널이 동일한 형태로 들어온다."""

    channel: TriggerChannel
    booth_id: str = Field(..., min_length=1, max_length=128)
    event_id: str = Field(..., description="MICE 행사 식별자")
    user_token: str = Field(
        ...,
        description="행사별 익명 토큰 (개인 식별 정보 아님)",
        min_length=1,
        max_length=128,
    )
    target_lang: str = Field(default="auto", pattern=r"^(auto|ko|en|ja|zh)$")

    # 선택 필드 — 채널별 raw payload 보존
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_payload: Optional[dict] = None

    def to_firestore(self) -> dict:
        return {
            "channel": self.channel.value,
            "booth_id": self.booth_id,
            "event_id": self.event_id,
            "user_token": self.user_token,
            "target_lang": self.target_lang,
            "timestamp": self.timestamp.isoformat(),
            "raw_payload": self.raw_payload,
        }
