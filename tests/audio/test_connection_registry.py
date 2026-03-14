"""Tests for the ConnectionRegistry and task nudge server-push framing."""

import time

import pytest

from blurt.audio.connection_registry import ConnectionRegistry, TaskNudgePayload
from blurt.models.audio import ServerMessageType


class FakeHandler:
    """Minimal fake of AudioWebSocketHandler for registry tests."""

    def __init__(self, user_id: str = "test-user"):
        self._user_id = user_id
        self.pushed: list[tuple[ServerMessageType, dict]] = []

    @property
    def user_id(self) -> str:
        return self._user_id

    async def push_message(self, msg_type: ServerMessageType, payload: dict) -> None:
        self.pushed.append((msg_type, payload))


class TestTaskNudgePayload:
    """Tests for TaskNudgePayload data class."""

    def test_to_dict_minimal(self):
        nudge = TaskNudgePayload(task_id="t1", content="Buy groceries")
        d = nudge.to_dict()
        assert d["task_id"] == "t1"
        assert d["content"] == "Buy groceries"
        assert d["intent"] == "task"
        assert d["reason"] == "scheduled"
        assert d["priority"] == 0.5
        assert "nudge_ts" in d
        assert isinstance(d["nudge_ts"], float)
        # Optional fields omitted when empty/default
        assert "due_at" not in d
        assert "entity_names" not in d
        assert "surface_count" not in d

    def test_to_dict_full(self):
        nudge = TaskNudgePayload(
            task_id="t2",
            content="Call Alice about project",
            intent="reminder",
            due_at="2026-03-15T09:00:00Z",
            reason="due_soon",
            priority=0.9,
            entity_names=["Alice", "Project X"],
            surface_count=3,
        )
        d = nudge.to_dict()
        assert d["task_id"] == "t2"
        assert d["content"] == "Call Alice about project"
        assert d["intent"] == "reminder"
        assert d["due_at"] == "2026-03-15T09:00:00Z"
        assert d["reason"] == "due_soon"
        assert d["priority"] == 0.9
        assert d["entity_names"] == ["Alice", "Project X"]
        assert d["surface_count"] == 3

    def test_nudge_ts_is_recent(self):
        before = time.time()
        nudge = TaskNudgePayload(task_id="t1", content="test")
        d = nudge.to_dict()
        after = time.time()
        assert before <= d["nudge_ts"] <= after


class TestConnectionRegistry:
    """Tests for the ConnectionRegistry."""

    def test_register_unregister(self):
        registry = ConnectionRegistry()
        h = FakeHandler()
        assert registry.connection_count == 0

        registry.register(h)
        assert registry.connection_count == 1

        registry.unregister(h)
        assert registry.connection_count == 0

    def test_unregister_missing_is_safe(self):
        registry = ConnectionRegistry()
        h = FakeHandler()
        # Should not raise
        registry.unregister(h)
        assert registry.connection_count == 0

    def test_get_handlers_for_user(self):
        registry = ConnectionRegistry()
        h1 = FakeHandler("user-a")
        h2 = FakeHandler("user-b")
        h3 = FakeHandler("user-a")
        registry.register(h1)
        registry.register(h2)
        registry.register(h3)

        user_a_handlers = registry.get_handlers_for_user("user-a")
        assert len(user_a_handlers) == 2
        assert h1 in user_a_handlers
        assert h3 in user_a_handlers

        user_b_handlers = registry.get_handlers_for_user("user-b")
        assert len(user_b_handlers) == 1
        assert h2 in user_b_handlers

    def test_get_handlers_unknown_user(self):
        registry = ConnectionRegistry()
        assert registry.get_handlers_for_user("nobody") == []

    @pytest.mark.asyncio
    async def test_send_task_nudge_to_user(self):
        registry = ConnectionRegistry()
        h = FakeHandler("user-a")
        registry.register(h)

        nudge = TaskNudgePayload(task_id="t1", content="Buy milk", reason="context_match")
        sent = await registry.send_task_nudge("user-a", nudge)

        assert sent == 1
        assert len(h.pushed) == 1
        msg_type, payload = h.pushed[0]
        assert msg_type == ServerMessageType.TASK_NUDGE
        assert payload["task_id"] == "t1"
        assert payload["content"] == "Buy milk"
        assert payload["reason"] == "context_match"

    @pytest.mark.asyncio
    async def test_send_task_nudge_no_handlers(self):
        registry = ConnectionRegistry()
        nudge = TaskNudgePayload(task_id="t1", content="test")
        sent = await registry.send_task_nudge("nobody", nudge)
        assert sent == 0

    @pytest.mark.asyncio
    async def test_send_task_nudge_multiple_connections(self):
        registry = ConnectionRegistry()
        h1 = FakeHandler("user-a")
        h2 = FakeHandler("user-a")
        registry.register(h1)
        registry.register(h2)

        nudge = TaskNudgePayload(task_id="t1", content="test")
        sent = await registry.send_task_nudge("user-a", nudge)

        assert sent == 2
        assert len(h1.pushed) == 1
        assert len(h2.pushed) == 1

    @pytest.mark.asyncio
    async def test_send_nudge_only_targets_correct_user(self):
        registry = ConnectionRegistry()
        h_a = FakeHandler("user-a")
        h_b = FakeHandler("user-b")
        registry.register(h_a)
        registry.register(h_b)

        nudge = TaskNudgePayload(task_id="t1", content="test")
        await registry.send_task_nudge("user-a", nudge)

        assert len(h_a.pushed) == 1
        assert len(h_b.pushed) == 0

    @pytest.mark.asyncio
    async def test_broadcast_task_nudge(self):
        registry = ConnectionRegistry()
        h_a = FakeHandler("user-a")
        h_b = FakeHandler("user-b")
        registry.register(h_a)
        registry.register(h_b)

        nudge = TaskNudgePayload(task_id="t1", content="system alert")
        sent = await registry.broadcast_task_nudge(nudge)

        assert sent == 2
        assert len(h_a.pushed) == 1
        assert len(h_b.pushed) == 1

    @pytest.mark.asyncio
    async def test_push_to_user_arbitrary_type(self):
        registry = ConnectionRegistry()
        h = FakeHandler("user-a")
        registry.register(h)

        sent = await registry.push_to_user(
            "user-a",
            ServerMessageType.RESPONSE_TEXT,
            {"text": "hello"},
        )
        assert sent == 1
        msg_type, payload = h.pushed[0]
        assert msg_type == ServerMessageType.RESPONSE_TEXT
        assert payload["text"] == "hello"

    @pytest.mark.asyncio
    async def test_push_handles_handler_error_gracefully(self):
        """If a handler's push_message raises, the error is caught."""
        registry = ConnectionRegistry()

        class BrokenHandler(FakeHandler):
            async def push_message(self, msg_type, payload):
                raise RuntimeError("connection lost")

        h_ok = FakeHandler("user-a")
        h_broken = BrokenHandler("user-a")
        registry.register(h_ok)
        registry.register(h_broken)

        nudge = TaskNudgePayload(task_id="t1", content="test")
        sent = await registry.send_task_nudge("user-a", nudge)

        # One succeeded, one failed
        assert sent == 1
        assert len(h_ok.pushed) == 1
