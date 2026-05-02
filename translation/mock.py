"""Sprint 1 더미 응답 — STT/TTS 호출 없이 양쪽 폰 동기화 검증.

Sprint 2 에서 audio.chunk 를 받으면 SttPipeline 으로 위임하고, 본 모듈은 테스트
fixtures 로만 남는다.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from translation.session_manager import SessionRole, TranslationSession

# 데모 시연 멘트 (8월 베타 검증 스크립트의 7-turn 대본)
_MOCK_PAIRS_KO_TO_EN = [
    (
        "안녕하세요, Lumen Labs 부스를 찾아주셔서 감사합니다. 무엇을 도와드릴까요?",
        "Hello, thank you for visiting the Lumen Labs booth. How can I help you?",
    ),
    (
        "산업용이시군요. 정확도와 속도 중 어느 쪽이 더 중요하세요?",
        "For industrial use — which matters more, accuracy or speed?",
    ),
    (
        "0.05mm 정확도, 시간당 1평방미터 스캔이 가능합니다.",
        "0.05mm accuracy, scanning one square meter per hour.",
    ),
    (
        "물론입니다. 명함 받으시면 다음 주 데모 일정 잡아드리겠습니다.",
        "Of course. If you share your business card, I'll schedule a demo for next week.",
    ),
]

_MOCK_PAIRS_EN_TO_KO = [
    (
        "Hi, I'm looking for a 3D scanning solution for industrial use cases.",
        "안녕하세요, 산업용 3D 스캐닝 솔루션을 찾고 있습니다.",
    ),
    (
        "Both, but accuracy is critical. What's your typical resolution?",
        "둘 다 중요하지만 정확도가 가장 핵심입니다. 일반적인 해상도가 어떻게 되나요?",
    ),
    (
        "Impressive. Can I get a demo and pricing?",
        "인상적이네요. 데모와 가격 안내 받을 수 있을까요?",
    ),
]


@dataclass
class MockSegment:
    segment_id: str
    speaker: SessionRole
    source_lang: str
    source_text: str
    target_lang: str
    target_text: str

    def as_segment_final(self) -> dict:
        return {
            "type": "segment.final",
            "segment_id": self.segment_id,
            "speaker": self.speaker.value,
            "source_lang": self.source_lang,
            "source_text": self.source_text,
            "target_lang": self.target_lang,
            "target_text": self.target_text,
            "confidence": 0.92,
            "ts": int(time.time() * 1000),
        }

    def as_tts_audio(self) -> dict:
        return {
            "type": "tts.audio",
            "segment_id": self.segment_id,
            "audio_url": None,
            "audio_b64": None,  # Sprint 1 mock — Sprint 3 에서 ElevenLabs 결과
            "duration_ms": 1800,
            "voice_id": f"mock_{self.target_lang}",
        }


def build_mock_segment(
    *, role: SessionRole, session: TranslationSession
) -> MockSegment:
    src = session.src_lang_for(role)
    tgt = session.tgt_lang_for(role)
    pairs = (
        _MOCK_PAIRS_KO_TO_EN if (src, tgt) == ("ko", "en")
        else _MOCK_PAIRS_EN_TO_KO if (src, tgt) == ("en", "ko")
        else [("(mock source)", "(mock translation)")]
    )
    # 같은 role 의 이전 segment 갯수로 round-robin
    own_segments = [s for s in session.segments
                    if s.get("speaker") == role.value]
    idx = len(own_segments) % len(pairs)
    s_text, t_text = pairs[idx]
    seg = MockSegment(
        segment_id=str(uuid.uuid4()),
        speaker=role,
        source_lang=src,
        source_text=s_text,
        target_lang=tgt,
        target_text=t_text,
    )
    session.segments.append(seg.as_segment_final())
    return seg
