"""Integration tests for task nudge server-push via WebSocket.

Verifies that task nudges pushed through the ConnectionRegistry are
delivered to connected clients through the persistent WebSocket at
appropriate intervals.
"""

import json
import threading
import time

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from blurt.audio.connection_registry import ConnectionRegistry, TaskNudgePayload
from blurt.config.settings import BlurtConfig, GeminiConfig, WebSocketConfig
from blurt.core.app import create_app
from blurt.models.audio import ClientMessageType, ServerMessageType


@pytest.fixture
def config():
    return BlurtConfig(
        gemini=GeminiConfig(api_key="test-key"),
        websocket=WebSocketConfig(
            idle_timeout=30.0,
            max_audio_buffer_bytes=4096,
            audio_chunk_max_bytes=1024,
            max_concurrent_sessions=5,
        ),
    )


@pytest.fixture
def app(config):
    """Create app with a test-only endpoint that triggers nudge pushes.

    This endpoint runs in the same event loop as the WebSocket handlers,
    ensuring that asyncio.Queue operations work correctly.
    """
    application = create_app(config)

    @application.post("/test/push-nudge")
    async def push_nudge(request: Request):
        body = await request.json()
        registry: ConnectionRegistry = request.app.state.connection_registry
        nudge = TaskNudgePayload(
            task_id=body.get("task_id", "test-task"),
            content=body.get("content", "Test nudge"),
            intent=body.get("intent", "task"),
            due_at=body.get("due_at"),
            reason=body.get("reason", "scheduled"),
            priority=body.get("priority", 0.5),
            entity_names=body.get("entity_names", []),
            surface_count=body.get("surface_count", 0),
        )
        user_id = body.get("user_id", "test-user")
        sent = await registry.send_task_nudge(user_id, nudge)
        return {"sent": sent}

    @application.post("/test/broadcast-nudge")
    async def broadcast_nudge(request: Request):
        body = await request.json()
        registry: ConnectionRegistry = request.app.state.connection_registry
        nudge = TaskNudgePayload(
            task_id=body.get("task_id", "test-task"),
            content=body.get("content", "Test nudge"),
        )
        sent = await registry.broadcast_task_nudge(nudge)
        return {"sent": sent}

    return application


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def _init_msg(user_id: str = "test-user", sequence: int = 0) -> str:
    return json.dumps({
        "type": ClientMessageType.SESSION_INIT.value,
        "sequence": sequence,
        "payload": {
            "user_id": user_id,
            "audio_config": {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "language": "en-US",
            },
            "timezone": "UTC",
        },
    })


def _msg(msg_type: str, sequence: int, payload: dict | None = None) -> str:
    return json.dumps({
        "type": msg_type,
        "sequence": sequence,
        "payload": payload or {},
    })


def _trigger_nudge(base_url, **kwargs):
    """Trigger a nudge via the test endpoint from a background thread.

    Uses httpx to make a POST request to the test nudge endpoint.
    This runs the push in the app's event loop (same as WebSocket handlers).
    """
    import httpx
    url = f"{base_url}/test/push-nudge"
    resp = httpx.post(url, json=kwargs)
    return resp.json()


class TestTaskNudgePush:
    """Tests that task nudges are pushed to connected WebSocket clients."""

    def test_nudge_received_by_connected_client(self, client, app):
        """A connected client receives a task nudge pushed via the registry."""
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_CREATED.value

            # Trigger nudge from a separate thread (via HTTP endpoint in same event loop)
            result = {"sent": None}

            def trigger():
                time.sleep(0.05)
                r = client.post("/test/push-nudge", json={
                    "user_id": "test-user",
                    "task_id": "task-001",
                    "content": "Buy groceries",
                    "intent": "task",
                    "reason": "due_soon",
                    "priority": 0.8,
                    "entity_names": ["groceries"],
                })
                result["sent"] = r.json()["sent"]

            t = threading.Thread(target=trigger)
            t.start()

            # Receive the nudge
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.TASK_NUDGE.value
            assert resp["payload"]["task_id"] == "task-001"
            assert resp["payload"]["content"] == "Buy groceries"
            assert resp["payload"]["intent"] == "task"
            assert resp["payload"]["reason"] == "due_soon"
            assert resp["payload"]["priority"] == 0.8
            assert "nudge_ts" in resp["payload"]

            t.join(timeout=5)
            assert result["sent"] == 1

            # Clean up
            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_ENDED.value

    def test_multiple_nudges_at_intervals(self, client, app):
        """Multiple nudges sent at intervals are all received in order."""
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_CREATED.value

            nudge_count = 3

            def push_multiple():
                time.sleep(0.05)
                for i in range(nudge_count):
                    client.post("/test/push-nudge", json={
                        "user_id": "test-user",
                        "task_id": f"task-{i:03d}",
                        "content": f"Task number {i}",
                    })
                    time.sleep(0.05)

            t = threading.Thread(target=push_multiple)
            t.start()

            # Receive all nudges in order
            received_ids = []
            for _ in range(nudge_count):
                resp = json.loads(ws.receive_text())
                assert resp["type"] == ServerMessageType.TASK_NUDGE.value
                received_ids.append(resp["payload"]["task_id"])

            assert received_ids == ["task-000", "task-001", "task-002"]
            t.join(timeout=10)

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))

    def test_nudge_between_capture_sessions(self, client, app):
        """Nudges work between audio sessions on a persistent connection."""
        with client.websocket_connect("/ws/audio") as ws:
            # First session
            ws.send_text(_init_msg())
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_CREATED.value

            # End first session (connection stays alive)
            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_ENDED.value
            assert resp["payload"]["persistent"] is True

            # Push nudge between sessions
            def push_nudge():
                time.sleep(0.05)
                client.post("/test/push-nudge", json={
                    "user_id": "test-user",
                    "task_id": "between-sessions",
                    "content": "Check email",
                })

            t = threading.Thread(target=push_nudge)
            t.start()

            # Should receive nudge even without active session
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.TASK_NUDGE.value
            assert resp["payload"]["task_id"] == "between-sessions"
            assert resp["payload"]["content"] == "Check email"

            t.join(timeout=5)

            # Can start a new session after receiving nudge
            ws.send_text(_init_msg(sequence=2))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_CREATED.value

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 3))

    def test_nudge_only_sent_to_target_user(self, client, app):
        """Nudges are only delivered to the correct user's connections."""
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg(user_id="user-A"))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_CREATED.value

            # Push nudge to a DIFFERENT user
            r = client.post("/test/push-nudge", json={
                "user_id": "user-B",
                "task_id": "wrong-user",
                "content": "Not for you",
            })
            # Should report 0 handlers reached
            assert r.json()["sent"] == 0

            # Send ping to verify connection is still alive (no nudge was received)
            ws.send_text(json.dumps({
                "type": ClientMessageType.PING.value,
                "sequence": 0,
            }))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.PONG.value

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))

    def test_nudge_during_active_audio_session(self, client, app):
        """Nudges arrive even while an audio session is actively streaming."""
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_CREATED.value

            # Send some audio data
            ws.send_bytes(b"\x00\x01" * 100)

            # Push nudge while audio session is active
            def push_nudge():
                time.sleep(0.05)
                client.post("/test/push-nudge", json={
                    "user_id": "test-user",
                    "task_id": "mid-stream",
                    "content": "Don't forget milk",
                })

            t = threading.Thread(target=push_nudge)
            t.start()

            # Should receive nudge even during audio streaming
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.TASK_NUDGE.value
            assert resp["payload"]["task_id"] == "mid-stream"

            t.join(timeout=5)

            # Audio session still works — commit and get response
            ws.send_text(_msg(ClientMessageType.AUDIO_COMMIT.value, 1))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.TRANSCRIPT_FINAL.value

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 2))

    def test_nudge_payload_entity_names_and_due_at(self, client, app):
        """Nudge payload includes optional fields when present."""
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            json.loads(ws.receive_text())  # session.created

            def push_nudge():
                time.sleep(0.05)
                client.post("/test/push-nudge", json={
                    "user_id": "test-user",
                    "task_id": "task-full",
                    "content": "Team standup",
                    "intent": "event",
                    "due_at": "2026-03-14T10:00:00Z",
                    "reason": "due_soon",
                    "priority": 0.95,
                    "entity_names": ["standup", "team"],
                    "surface_count": 3,
                })

            t = threading.Thread(target=push_nudge)
            t.start()

            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.TASK_NUDGE.value
            payload = resp["payload"]
            assert payload["task_id"] == "task-full"
            assert payload["content"] == "Team standup"
            assert payload["intent"] == "event"
            assert payload["due_at"] == "2026-03-14T10:00:00Z"
            assert payload["reason"] == "due_soon"
            assert payload["priority"] == 0.95
            assert payload["entity_names"] == ["standup", "team"]
            assert payload["surface_count"] == 3

            t.join(timeout=5)
            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))

    def test_broadcast_nudge_to_all_connected(self, client, app):
        """Broadcast nudge reaches all connected clients."""
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg(user_id="user-A"))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_CREATED.value

            def push_broadcast():
                time.sleep(0.05)
                r = client.post("/test/broadcast-nudge", json={
                    "task_id": "broadcast-task",
                    "content": "Server maintenance",
                })
                assert r.json()["sent"] == 1

            t = threading.Thread(target=push_broadcast)
            t.start()

            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.TASK_NUDGE.value
            assert resp["payload"]["task_id"] == "broadcast-task"
            assert resp["payload"]["content"] == "Server maintenance"

            t.join(timeout=5)
            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))


class TestTaskNudgePayloadSerialization:
    """Tests for TaskNudgePayload structure and serialization."""

    def test_basic_payload(self):
        nudge = TaskNudgePayload(task_id="t1", content="Do laundry")
        d = nudge.to_dict()
        assert d["task_id"] == "t1"
        assert d["content"] == "Do laundry"
        assert d["intent"] == "task"
        assert d["reason"] == "scheduled"
        assert d["priority"] == 0.5
        assert "nudge_ts" in d
        assert isinstance(d["nudge_ts"], float)
        assert "due_at" not in d
        assert "entity_names" not in d
        assert "surface_count" not in d

    def test_full_payload(self):
        nudge = TaskNudgePayload(
            task_id="t2",
            content="Team standup",
            intent="event",
            due_at="2026-03-14T10:00:00Z",
            reason="context_match",
            priority=0.95,
            entity_names=["standup", "team"],
            surface_count=3,
        )
        d = nudge.to_dict()
        assert d["task_id"] == "t2"
        assert d["content"] == "Team standup"
        assert d["intent"] == "event"
        assert d["due_at"] == "2026-03-14T10:00:00Z"
        assert d["reason"] == "context_match"
        assert d["priority"] == 0.95
        assert d["entity_names"] == ["standup", "team"]
        assert d["surface_count"] == 3

    def test_default_values(self):
        nudge = TaskNudgePayload(task_id="t3", content="Default test")
        assert nudge.intent == "task"
        assert nudge.due_at is None
        assert nudge.reason == "scheduled"
        assert nudge.priority == 0.5
        assert nudge.entity_names == []
        assert nudge.surface_count == 0


class TestConnectionRegistryUnit:
    """Unit tests for the ConnectionRegistry."""

    def test_register_and_unregister(self):
        registry = ConnectionRegistry()
        assert registry.connection_count == 0

        handler = _MockHandler(user_id="user-1")
        registry.register(handler)
        assert registry.connection_count == 1

        registry.unregister(handler)
        assert registry.connection_count == 0

    def test_unregister_unknown_handler(self):
        """Unregistering a handler not in the registry does not raise."""
        registry = ConnectionRegistry()
        handler = _MockHandler(user_id="user-1")
        registry.unregister(handler)  # should not raise
        assert registry.connection_count == 0

    def test_get_handlers_for_user(self):
        registry = ConnectionRegistry()
        h1 = _MockHandler(user_id="user-1")
        h2 = _MockHandler(user_id="user-2")
        h3 = _MockHandler(user_id="user-1")

        registry.register(h1)
        registry.register(h2)
        registry.register(h3)

        user1_handlers = registry.get_handlers_for_user("user-1")
        assert len(user1_handlers) == 2
        assert h1 in user1_handlers
        assert h3 in user1_handlers

        user2_handlers = registry.get_handlers_for_user("user-2")
        assert len(user2_handlers) == 1
        assert h2 in user2_handlers

    def test_get_handlers_for_unknown_user(self):
        registry = ConnectionRegistry()
        handlers = registry.get_handlers_for_user("nobody")
        assert handlers == []

    @pytest.mark.asyncio
    async def test_send_nudge_no_handlers(self):
        registry = ConnectionRegistry()
        nudge = TaskNudgePayload(task_id="t", content="x")
        sent = await registry.send_task_nudge("nobody", nudge)
        assert sent == 0

    @pytest.mark.asyncio
    async def test_send_nudge_to_correct_user(self):
        registry = ConnectionRegistry()
        h1 = _MockHandler(user_id="alice")
        h2 = _MockHandler(user_id="bob")
        registry.register(h1)
        registry.register(h2)

        nudge = TaskNudgePayload(task_id="alice-task", content="Alice's task")
        sent = await registry.send_task_nudge("alice", nudge)
        assert sent == 1
        assert len(h1.pushed) == 1
        assert h1.pushed[0][0] == ServerMessageType.TASK_NUDGE
        assert h1.pushed[0][1]["task_id"] == "alice-task"
        assert len(h2.pushed) == 0

    @pytest.mark.asyncio
    async def test_send_nudge_to_multiple_connections_same_user(self):
        """When a user has multiple connections, nudge goes to all."""
        registry = ConnectionRegistry()
        h1 = _MockHandler(user_id="alice")
        h2 = _MockHandler(user_id="alice")
        registry.register(h1)
        registry.register(h2)

        nudge = TaskNudgePayload(task_id="shared-task", content="Shared")
        sent = await registry.send_task_nudge("alice", nudge)
        assert sent == 2
        assert len(h1.pushed) == 1
        assert len(h2.pushed) == 1

    @pytest.mark.asyncio
    async def test_broadcast_nudge_to_all(self):
        registry = ConnectionRegistry()
        h1 = _MockHandler(user_id="alice")
        h2 = _MockHandler(user_id="bob")
        registry.register(h1)
        registry.register(h2)

        nudge = TaskNudgePayload(task_id="broadcast-task", content="All")
        sent = await registry.broadcast_task_nudge(nudge)
        assert sent == 2
        assert len(h1.pushed) == 1
        assert len(h2.pushed) == 1

    @pytest.mark.asyncio
    async def test_push_to_user_arbitrary_type(self):
        registry = ConnectionRegistry()
        h1 = _MockHandler(user_id="alice")
        registry.register(h1)

        sent = await registry.push_to_user(
            "alice",
            ServerMessageType.RESPONSE_TEXT,
            {"text": "hello"},
        )
        assert sent == 1
        assert h1.pushed[0][0] == ServerMessageType.RESPONSE_TEXT

    @pytest.mark.asyncio
    async def test_push_handler_error_is_swallowed(self):
        """Errors in push_message are caught, not propagated."""
        registry = ConnectionRegistry()
        h1 = _MockHandler(user_id="alice", fail=True)
        h2 = _MockHandler(user_id="alice")
        registry.register(h1)
        registry.register(h2)

        nudge = TaskNudgePayload(task_id="t", content="x")
        sent = await registry.send_task_nudge("alice", nudge)
        # h1 fails, h2 succeeds
        assert sent == 1
        assert len(h2.pushed) == 1


class _MockHandler:
    """Minimal mock for AudioWebSocketHandler for unit tests."""

    def __init__(self, user_id: str = "test", fail: bool = False):
        self._user_id = user_id
        self._fail = fail
        self.pushed: list[tuple] = []

    @property
    def user_id(self) -> str | None:
        return self._user_id

    async def push_message(self, msg_type, payload):
        if self._fail:
            raise RuntimeError("Mock push failure")
        self.pushed.append((msg_type, payload))
