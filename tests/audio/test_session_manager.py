"""Tests for SessionManager."""

import pytest

from blurt.audio.session_manager import SessionLimitError, SessionManager
from blurt.models.audio import SessionState


@pytest.fixture
def manager():
    return SessionManager(max_concurrent=5, max_audio_buffer_bytes=1024)


class TestSessionManager:
    async def test_create_session(self, manager):
        session = await manager.create_session(user_id="user-1")
        assert session.user_id == "user-1"
        assert session.state == SessionState.ACTIVE
        assert manager.active_count == 1

    async def test_get_session(self, manager):
        session = await manager.create_session(user_id="user-1")
        found = await manager.get_session(session.session_id)
        assert found is not None
        assert found.session_id == session.session_id

    async def test_get_session_not_found(self, manager):
        found = await manager.get_session("nonexistent")
        assert found is None

    async def test_get_buffer(self, manager):
        session = await manager.create_session(user_id="user-1")
        buffer = await manager.get_buffer(session.session_id)
        assert buffer is not None
        assert buffer.max_bytes == 1024

    async def test_end_session(self, manager):
        session = await manager.create_session(user_id="user-1")
        ended = await manager.end_session(session.session_id)
        assert ended is not None
        assert ended.state == SessionState.CLOSED
        assert manager.active_count == 0

    async def test_end_nonexistent_session(self, manager):
        result = await manager.end_session("nonexistent")
        assert result is None

    async def test_session_limit(self, manager):
        for i in range(5):
            await manager.create_session(user_id=f"user-{i}")
        with pytest.raises(SessionLimitError):
            await manager.create_session(user_id="user-overflow")

    async def test_update_state(self, manager):
        session = await manager.create_session(user_id="user-1")
        updated = await manager.update_state(session.session_id, SessionState.PAUSED)
        assert updated is not None
        assert updated.state == SessionState.PAUSED

    async def test_get_sessions_for_user(self, manager):
        await manager.create_session(user_id="user-1")
        await manager.create_session(user_id="user-1")
        await manager.create_session(user_id="user-2")
        sessions = await manager.get_sessions_for_user("user-1")
        assert len(sessions) == 2

    async def test_cleanup_idle(self, manager):
        session = await manager.create_session(user_id="user-1")
        # Force the session to appear idle by backdating last_activity
        from datetime import datetime, timedelta, timezone
        session.last_activity = datetime.now(timezone.utc) - timedelta(seconds=600)
        removed = await manager.cleanup_idle(idle_timeout=300)
        assert len(removed) == 1
        assert manager.active_count == 0
