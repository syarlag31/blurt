"""Tests for the capture API endpoints.

Covers:
- POST /api/v1/blurt — generic blurt capture
- POST /api/v1/blurt/voice — voice-specific capture
- POST /api/v1/blurt/text — text-specific capture
- GET  /api/v1/blurt/stats — pipeline statistics
- Casual remarks captured via API
- Zero-drop guarantee via HTTP endpoints
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blurt.api.capture import router, set_pipeline
from blurt.api.episodes import router as episodes_router, set_store
from blurt.memory.episodic import InMemoryEpisodicStore
from blurt.services.capture import BlurtCapturePipeline


# Reuse mock functions from service tests
async def _mock_classify(text: str) -> tuple[str, float]:
    text_lower = text.lower().strip()
    casual = ["huh", "hmm", "oh", "nice", "cool", "yeah", "okay", "ok"]
    if not text_lower or any(text_lower.startswith(p) for p in casual):
        return ("journal", 0.90)
    if any(w in text_lower for w in ["need to", "buy", "call"]):
        return ("task", 0.92)
    if any(w in text_lower for w in ["meeting", "dinner"]):
        return ("event", 0.89)
    return ("journal", 0.85)


@pytest.fixture
def store() -> InMemoryEpisodicStore:
    return InMemoryEpisodicStore()


@pytest.fixture
def app(store: InMemoryEpisodicStore) -> FastAPI:
    application = FastAPI()
    pipeline = BlurtCapturePipeline(store, classifier=_mock_classify)
    set_pipeline(pipeline)
    set_store(store)
    application.include_router(router)
    application.include_router(episodes_router)
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/v1/blurt — generic capture
# ---------------------------------------------------------------------------


class TestCaptureEndpoint:
    def test_capture_casual_remark(self, client: TestClient):
        """Casual remarks are captured via the API."""
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "huh, interesting"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["captured"] is True
        assert data["episode"]["raw_text"] == "huh, interesting"
        assert data["intent"] == "journal"

    def test_capture_empty_string(self, client: TestClient):
        """Empty strings are captured, not rejected."""
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": ""},
        )
        assert resp.status_code == 201
        assert resp.json()["captured"] is True

    def test_capture_task(self, client: TestClient):
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "I need to buy groceries"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["captured"] is True
        assert data["intent"] == "task"

    def test_capture_weather_comment(self, client: TestClient):
        """Off-hand weather comments are captured."""
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "nice weather today"},
        )
        assert resp.status_code == 201
        assert resp.json()["captured"] is True
        assert resp.json()["episode"]["raw_text"] == "nice weather today"

    def test_capture_returns_episode_with_id(self, client: TestClient):
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "test"},
        )
        data = resp.json()
        assert "id" in data["episode"]
        assert "timestamp" in data["episode"]

    def test_capture_with_session_context(self, client: TestClient):
        resp = client.post(
            "/api/v1/blurt",
            json={
                "user_id": "u1",
                "raw_text": "whatever",
                "session_id": "sess-1",
                "time_of_day": "evening",
                "day_of_week": "friday",
            },
        )
        assert resp.status_code == 201
        ctx = resp.json()["episode"]["context"]
        assert ctx["session_id"] == "sess-1"
        assert ctx["time_of_day"] == "evening"
        assert ctx["day_of_week"] == "friday"

    def test_capture_text_modality(self, client: TestClient):
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "edit", "modality": "text"},
        )
        assert resp.status_code == 201
        assert resp.json()["episode"]["modality"] == "text"


# ---------------------------------------------------------------------------
# POST /api/v1/blurt/voice — voice-specific capture
# ---------------------------------------------------------------------------


class TestVoiceCaptureEndpoint:
    def test_voice_capture(self, client: TestClient):
        resp = client.post(
            "/api/v1/blurt/voice",
            json={
                "user_id": "u1",
                "raw_text": "hmm, that's cool",
                "audio_duration_ms": 1500,
                "transcription_confidence": 0.92,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["captured"] is True
        assert data["episode"]["modality"] == "voice"

    def test_voice_capture_casual(self, client: TestClient):
        resp = client.post(
            "/api/v1/blurt/voice",
            json={"user_id": "u1", "raw_text": "oh well"},
        )
        assert resp.status_code == 201
        assert resp.json()["episode"]["raw_text"] == "oh well"


# ---------------------------------------------------------------------------
# POST /api/v1/blurt/text — text-specific capture
# ---------------------------------------------------------------------------


class TestTextCaptureEndpoint:
    def test_text_capture(self, client: TestClient):
        resp = client.post(
            "/api/v1/blurt/text",
            json={"user_id": "u1", "raw_text": "correction: 4pm not 3pm"},
        )
        assert resp.status_code == 201
        assert resp.json()["episode"]["modality"] == "text"


# ---------------------------------------------------------------------------
# GET /api/v1/blurt/stats — pipeline statistics
# ---------------------------------------------------------------------------


class TestStatsEndpoint:
    def test_stats_empty(self, client: TestClient):
        resp = client.get("/api/v1/blurt/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_captured"] == 0
        assert data["drop_rate"] == 0.0

    def test_stats_after_captures(self, client: TestClient):
        # Capture some blurts
        for text in ["hmm", "buy groceries", "nice weather"]:
            client.post(
                "/api/v1/blurt",
                json={"user_id": "u1", "raw_text": text},
            )

        resp = client.get("/api/v1/blurt/stats")
        data = resp.json()
        assert data["total_captured"] == 3
        assert data["drop_rate"] == 0.0
        assert data["casual_capture_count"] >= 2  # hmm + nice weather

    def test_stats_tracks_intent_distribution(self, client: TestClient):
        client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "I need to call someone"},
        )
        client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "hmm"},
        )

        resp = client.get("/api/v1/blurt/stats")
        dist = resp.json()["intent_distribution"]
        assert "task" in dist
        assert "journal" in dist


# ---------------------------------------------------------------------------
# Zero-drop guarantee through API
# ---------------------------------------------------------------------------


class TestZeroDropAPI:
    def test_all_input_types_captured(self, client: TestClient):
        """Every type of input is captured via the API."""
        inputs = [
            "I need to buy groceries",       # task
            "meeting tomorrow at 3",          # event
            "hmm",                            # casual
            "nice weather",                   # casual
            "",                               # empty
            "oh well",                        # filler
            "whatever",                       # dismissive
            "cool cool cool",                 # reaction
        ]

        episode_ids = set()
        for text in inputs:
            resp = client.post(
                "/api/v1/blurt",
                json={"user_id": "u1", "raw_text": text},
            )
            assert resp.status_code == 201, f"Failed to capture '{text}'"
            assert resp.json()["captured"] is True
            episode_ids.add(resp.json()["episode"]["id"])

        # All unique — no drops, no duplicates
        assert len(episode_ids) == len(inputs)

    def test_captured_blurts_retrievable_via_episodes_api(self, client: TestClient):
        """Blurts captured via /blurt are retrievable via /episodes."""
        # Capture a casual remark
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "nice weather today"},
        )
        episode_id = resp.json()["episode"]["id"]

        # Retrieve via episodes API
        get_resp = client.get(f"/api/v1/episodes/{episode_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["raw_text"] == "nice weather today"

    def test_casual_remarks_appear_in_recall(self, client: TestClient):
        """Casual remarks show up in full recall."""
        client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "huh, interesting"},
        )
        client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "I need to buy groceries"},
        )

        resp = client.get("/api/v1/episodes/recall/u1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_count"] == 2
        texts = [e["episode"]["raw_text"] for e in data["entries"]]
        assert "huh, interesting" in texts
        assert "I need to buy groceries" in texts
