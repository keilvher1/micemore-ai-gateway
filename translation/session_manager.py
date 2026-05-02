"""세션 broker — 운영자/참가자 두 WebSocket 을 같은 session_id 로 묶는다.

Sprint 1: 인-메모리 dict (단일 프로세스).
Sprint 2: Redis pub/sub 로 이전해 모바일 재연결 + 멀티 인스턴스 지원.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import WebSocket

log = logging.getLogger("session-mgr")


class SessionRole(str, Enum):
    OPERATOR = "operator"
    VISITOR = "visitor"


@dataclass
class _PeerSocket:
    ws: WebSocket
    user_id: str
    role: SessionRole


@dataclass
class TranslationSession:
    session_id: str
    booth_id: str
    lang_pair: tuple[str, str]
    started_at: float = field(default_factory=time.time)
    peers: dict[SessionRole, _PeerSocket] = field(default_factory=dict)
    segments: list[dict] = field(default_factory=list)
    ended: bool = False

    def is_paired(self) -> bool:
        return len(self.peers) == 2

    def src_lang_for(self, role: SessionRole) -> str:
        # operator 는 lang_pair[0], visitor 는 lang_pair[1]
        return self.lang_pair[0] if role == SessionRole.OPERATOR else self.lang_pair[1]

    def tgt_lang_for(self, role: SessionRole) -> str:
        return self.lang_pair[1] if role == SessionRole.OPERATOR else self.lang_pair[0]


class SessionManager:
    """단일 SessionManager 인스턴스가 모든 WS 를 중개."""

    def __init__(self) -> None:
        self._sessions: dict[str, TranslationSession] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ join
    async def attach(
        self,
        session_id: str,
        role: SessionRole,
        ws: WebSocket,
        user_id: str,
        booth_id: str,
        lang_pair: list[str],
    ) -> TranslationSession:
        async with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = TranslationSession(
                    session_id=session_id,
                    booth_id=booth_id,
                    lang_pair=(lang_pair[0], lang_pair[1]),
                )
                self._sessions[session_id] = sess
            sess.peers[role] = _PeerSocket(ws=ws, user_id=user_id, role=role)
        log.info(
            "attach | session=%s role=%s user=%s paired=%s",
            session_id, role.value, user_id, sess.is_paired(),
        )
        return sess

    # ----------------------------------------------------------------- leave
    async def detach(
        self, session_id: str | None, role: SessionRole | None, ws: WebSocket
    ) -> None:
        if not session_id or role is None:
            return
        async with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return
            peer = sess.peers.get(role)
            if peer and peer.ws is ws:
                sess.peers.pop(role, None)
                if not sess.peers:
                    self._sessions.pop(session_id, None)

    # ------------------------------------------------------------ broadcast
    async def broadcast(self, session_id: str | None, payload: dict[str, Any]) -> None:
        if not session_id:
            return
        sess = self._sessions.get(session_id)
        if not sess:
            return
        await asyncio.gather(
            *[p.ws.send_json(payload) for p in sess.peers.values()],
            return_exceptions=True,
        )

    async def send_to(
        self, session_id: str, role: SessionRole, payload: dict[str, Any]
    ) -> None:
        sess = self._sessions.get(session_id)
        if not sess:
            return
        peer = sess.peers.get(role)
        if peer:
            await peer.ws.send_json(payload)

    # ----------------------------------------------------------------- query
    def get(self, session_id: str) -> TranslationSession | None:
        return self._sessions.get(session_id)

    def active_count(self) -> int:
        return len(self._sessions)

    # --------------------------------------------------------------- finalize
    async def finalize(self, session_id: str) -> None:
        sess = self._sessions.get(session_id)
        if not sess or sess.ended:
            return
        sess.ended = True
        await self.broadcast(
            session_id,
            {
                "type": "session.summary",
                "session_id": session_id,
                "duration_sec": int(time.time() - sess.started_at),
                "segments_count": len(sess.segments),
                "transcript_url": None,  # Sprint 4: Firestore 저장 후 발급
            },
        )

    # ----------------------------------------------------------------- util
    @staticmethod
    def gen_id() -> str:
        return str(uuid.uuid4())
