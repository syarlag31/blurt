"""Session manager for tracking active WebSocket audio sessions.

Manages the lifecycle of AudioSession objects, enforces concurrency
limits, and provides lookup/cleanup operations.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from blurt.audio.buffer import AudioBuffer
from blurt.models.audio import AudioSession, SessionState

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages active audio streaming sessions.

    Thread-safe management of AudioSession instances and their
    associated AudioBuffers. Supports concurrent session limits
    and idle timeout cleanup.
    """

    def __init__(
        self,
        max_concurrent: int = 1000,
        max_audio_buffer_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self._sessions: dict[str, AudioSession] = {}
        self._buffers: dict[str, AudioBuffer] = {}
        self._lock = asyncio.Lock()
        self._max_concurrent = max_concurrent
        self._max_audio_buffer_bytes = max_audio_buffer_bytes

    async def create_session(
        self,
        user_id: str,
        **kwargs: object,
    ) -> AudioSession:
        """Create and register a new audio session.

        Args:
            user_id: The authenticated user's ID.
            **kwargs: Additional AudioSession fields (audio_config, device_id, etc.)

        Returns:
            The newly created AudioSession.

        Raises:
            SessionLimitError: If the max concurrent session limit is reached.
        """
        async with self._lock:
            if len(self._sessions) >= self._max_concurrent:
                raise SessionLimitError(
                    f"Maximum concurrent sessions ({self._max_concurrent}) reached"
                )

            session = AudioSession(user_id=user_id, **kwargs)  # type: ignore[arg-type]
            session.state = SessionState.ACTIVE
            self._sessions[session.session_id] = session
            self._buffers[session.session_id] = AudioBuffer(
                max_bytes=self._max_audio_buffer_bytes
            )

            logger.info(
                "Session created: %s for user %s",
                session.session_id,
                user_id,
            )
            return session

    async def get_session(self, session_id: str) -> AudioSession | None:
        """Get an active session by ID."""
        return self._sessions.get(session_id)

    async def get_buffer(self, session_id: str) -> AudioBuffer | None:
        """Get the audio buffer for a session."""
        return self._buffers.get(session_id)

    async def update_state(
        self, session_id: str, state: SessionState
    ) -> AudioSession | None:
        """Update the state of a session."""
        session = self._sessions.get(session_id)
        if session:
            session.state = state
            session.touch()
        return session

    async def end_session(self, session_id: str) -> AudioSession | None:
        """End and remove a session.

        Returns the session if it existed, None otherwise.
        The associated audio buffer is also cleaned up.
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            buffer = self._buffers.pop(session_id, None)

            if session:
                session.state = SessionState.CLOSED
                logger.info(
                    "Session ended: %s (chunks=%d, bytes=%d, utterances=%d)",
                    session_id,
                    session.chunks_received,
                    session.bytes_received,
                    session.utterances_processed,
                )

            if buffer and not buffer.is_empty:
                await buffer.discard()

            return session

    async def get_sessions_for_user(self, user_id: str) -> list[AudioSession]:
        """Get all active sessions for a given user."""
        return [s for s in self._sessions.values() if s.user_id == user_id]

    async def cleanup_idle(self, idle_timeout: float) -> list[str]:
        """Remove sessions that have been idle longer than idle_timeout seconds.

        Returns list of cleaned-up session IDs.
        """
        now = datetime.now(timezone.utc)
        to_remove: list[str] = []

        for sid, session in self._sessions.items():
            idle_secs = (now - session.last_activity).total_seconds()
            if idle_secs > idle_timeout:
                to_remove.append(sid)

        for sid in to_remove:
            await self.end_session(sid)
            logger.info("Cleaned up idle session: %s", sid)

        return to_remove

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    @property
    def session_ids(self) -> list[str]:
        """List of active session IDs."""
        return list(self._sessions.keys())


class SessionLimitError(Exception):
    """Raised when the maximum number of concurrent sessions is exceeded."""
