"""Tests for the WebSocket full-duplex audio streaming endpoint.

Uses FastAPI's TestClient WebSocket support for integration testing
of the complete protocol flow.
"""

import json

import pytest
from fastapi.testclient import TestClient

from blurt.config.settings import BlurtConfig, GeminiConfig, WebSocketConfig
from blurt.core.app import create_app
from blurt.models.audio import ClientMessageType, ServerMessageType


@pytest.fixture
def config():
    return BlurtConfig(
        gemini=GeminiConfig(api_key="test-key"),
        websocket=WebSocketConfig(
            idle_timeout=10.0,
            max_audio_buffer_bytes=4096,
            audio_chunk_max_bytes=1024,
            max_concurrent_sessions=5,
        ),
    )


@pytest.fixture
def app(config):
    return create_app(config)


@pytest.fixture
def client(app):
    # Use as context manager to trigger lifespan (startup/shutdown)
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
            "timezone": "America/New_York",
        },
    })


def _msg(msg_type: str, sequence: int, payload: dict | None = None) -> str:
    return json.dumps({
        "type": msg_type,
        "sequence": sequence,
        "payload": payload or {},
    })


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "active_sessions" in data


class TestWebSocketSessionInit:
    def test_successful_session_init(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_CREATED.value
            assert resp["payload"]["user_id"] == "test-user"
            assert resp["session_id"] != "none"

            # Close gracefully
            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_ENDED.value

    def test_invalid_first_message(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(json.dumps({"type": "audio.chunk", "sequence": 0}))
            # Server sends error then closes
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.ERROR.value
            assert resp["payload"]["code"] == "protocol_error"

    def test_invalid_json_first_message(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text("not json at all")
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.ERROR.value
            assert resp["payload"]["code"] == "invalid_json"

    def test_missing_user_id(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(json.dumps({
                "type": ClientMessageType.SESSION_INIT.value,
                "sequence": 0,
                "payload": {},
            }))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.ERROR.value
            assert resp["payload"]["code"] == "invalid_payload"


class TestWebSocketAudioStreaming:
    def test_audio_chunk_and_commit(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            # Init
            ws.send_text(_init_msg())
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_CREATED.value

            # Send audio chunks (simulate PCM data)
            audio_chunk = b"\x00\x01" * 500  # 1000 bytes of fake audio
            ws.send_bytes(audio_chunk)

            # Commit
            ws.send_text(_msg(ClientMessageType.AUDIO_COMMIT.value, 1))

            # Should receive transcript.final
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.TRANSCRIPT_FINAL.value
            assert resp["payload"]["audio_bytes"] == 1000

            # Should receive response.text acknowledgment
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.RESPONSE_TEXT.value
            assert resp["payload"]["text"] == "Got it."

            # End session
            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 2))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_ENDED.value
            assert resp["payload"]["utterances_processed"] == 1

    def test_multiple_utterances(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            json.loads(ws.receive_text())  # session.created

            # First utterance
            ws.send_bytes(b"\x00" * 200)
            ws.send_text(_msg(ClientMessageType.AUDIO_COMMIT.value, 1))
            json.loads(ws.receive_text())  # transcript.final
            json.loads(ws.receive_text())  # response.text

            # Second utterance
            ws.send_bytes(b"\x01" * 300)
            ws.send_text(_msg(ClientMessageType.AUDIO_COMMIT.value, 2))
            resp = json.loads(ws.receive_text())  # transcript.final
            assert resp["payload"]["utterance_id"] == 2
            json.loads(ws.receive_text())  # response.text

            # End
            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 3))
            resp = json.loads(ws.receive_text())
            assert resp["payload"]["utterances_processed"] == 2

    def test_audio_ack_every_10_chunks(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            json.loads(ws.receive_text())  # session.created

            # Send 10 small chunks to trigger ACK
            for _ in range(10):
                ws.send_bytes(b"\x00" * 50)

            # Should receive an audio.ack after 10 chunks
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.AUDIO_ACK.value
            assert resp["payload"]["chunks_received"] == 10

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))
            # Commit remaining audio first, then session ended
            resp = json.loads(ws.receive_text())
            # Could be transcript or session ended depending on buffer state


class TestWebSocketTextInput:
    def test_text_input(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            json.loads(ws.receive_text())  # session.created

            ws.send_text(_msg(
                ClientMessageType.TEXT_INPUT.value,
                1,
                {"text": "Pick up groceries"},
            ))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.RESPONSE_TEXT.value
            assert resp["payload"]["input_text"] == "Pick up groceries"

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 2))


class TestWebSocketPauseResume:
    def test_pause_and_resume(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            json.loads(ws.receive_text())  # session.created

            # Pause
            ws.send_text(_msg(ClientMessageType.SESSION_PAUSE.value, 1))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_UPDATED.value
            assert resp["payload"]["state"] == "paused"

            # Audio sent while paused should be silently dropped
            ws.send_bytes(b"\x00" * 100)

            # Resume
            ws.send_text(_msg(ClientMessageType.SESSION_RESUME.value, 2))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.SESSION_UPDATED.value
            assert resp["payload"]["state"] == "active"

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 3))


class TestWebSocketPing:
    def test_ping_pong(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            json.loads(ws.receive_text())  # session.created

            ws.send_text(json.dumps({
                "type": ClientMessageType.PING.value,
                "sequence": 0,  # Ping doesn't require sequence ordering
            }))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.PONG.value
            assert "server_time" in resp["payload"]

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))


class TestWebSocketErrorCases:
    def test_chunk_too_large(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            json.loads(ws.receive_text())  # session.created

            # Send chunk larger than max (1024)
            ws.send_bytes(b"\x00" * 2000)
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.ERROR.value
            assert resp["payload"]["code"] == "chunk_too_large"

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 1))

    def test_commit_empty_buffer(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            json.loads(ws.receive_text())  # session.created

            ws.send_text(_msg(ClientMessageType.AUDIO_COMMIT.value, 1))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.ERROR.value
            assert resp["payload"]["code"] == "empty_buffer"

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 2))

    def test_unknown_message_type(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg())
            json.loads(ws.receive_text())  # session.created

            ws.send_text(json.dumps({
                "type": "bogus.type",
                "sequence": 1,
            }))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.ERROR.value
            assert resp["payload"]["code"] == "unknown_message_type"

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 2))

    def test_sequence_error(self, client):
        with client.websocket_connect("/ws/audio") as ws:
            ws.send_text(_init_msg(sequence=5))
            json.loads(ws.receive_text())  # session.created

            # Send with sequence <= init sequence (5) to trigger error
            ws.send_text(_msg(ClientMessageType.AUDIO_COMMIT.value, 4))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == ServerMessageType.ERROR.value
            assert resp["payload"]["code"] == "sequence_error"

            ws.send_text(_msg(ClientMessageType.SESSION_END.value, 6))


class TestWebSocketDurationEstimate:
    def test_pcm_duration_estimate(self):
        """Test PCM duration estimation with larger buffer limits."""
        large_config = BlurtConfig(
            gemini=GeminiConfig(api_key="test-key"),
            websocket=WebSocketConfig(
                idle_timeout=10.0,
                max_audio_buffer_bytes=64 * 1024,
                audio_chunk_max_bytes=64 * 1024,
                max_concurrent_sessions=5,
            ),
        )
        app = create_app(large_config)
        with TestClient(app) as client:
            with client.websocket_connect("/ws/audio") as ws:
                ws.send_text(_init_msg())
                json.loads(ws.receive_text())  # session.created

                # 16kHz, 16-bit mono = 32000 bytes/sec
                # 32000 bytes = 1 second = 1000ms
                ws.send_bytes(b"\x00" * 32000)
                ws.send_text(_msg(ClientMessageType.AUDIO_COMMIT.value, 1))

                resp = json.loads(ws.receive_text())  # transcript.final
                assert resp["payload"]["audio_duration_ms"] == 1000

                ws.send_text(_msg(ClientMessageType.SESSION_END.value, 2))
