"""API-level tests for the clear-state endpoints.

Validates:
- POST /api/v1/status/check — returns clear message when no tasks
- GET  /api/v1/status/clear — always returns affirmation
- Anti-shame design at the API layer
- Emotion-aware tone in responses
- Integration with task surfacing engine
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blurt.api.clear_state import (
    get_clear_service,
    get_scoring_engine,
    router,
    set_clear_service,
    set_scoring_engine,
)
from blurt.services.clear_state import (
    ClearStateService,
    ClearTone,
    _CALM_MESSAGES,
    _CELEBRATORY_MESSAGES,
    _GENTLE_MESSAGES,
    _SHAME_WORDS,
    _WARM_MESSAGES,
    _MESSAGE_POOLS,
)
from blurt.services.task_surfacing import TaskScoringEngine


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_services():
    """Ensure clean DI state for each test."""
    set_clear_service(ClearStateService())
    set_scoring_engine(TaskScoringEngine())
    yield
    set_clear_service(None)  # type: ignore
    set_scoring_engine(None)  # type: ignore


# ---------------------------------------------------------------------------
# GET /api/v1/status/clear
# ---------------------------------------------------------------------------


class TestGetClearEndpoint:
    """Tests for the GET /status/clear endpoint."""

    def test_returns_200(self, client):
        resp = client.get("/api/v1/status/clear")
        assert resp.status_code == 200

    def test_returns_is_clear_true(self, client):
        resp = client.get("/api/v1/status/clear")
        data = resp.json()
        assert data["is_clear"] is True

    def test_returns_non_empty_message(self, client):
        resp = client.get("/api/v1/status/clear")
        data = resp.json()
        assert data["message"] != ""
        assert len(data["message"]) > 0

    def test_returns_valid_tone(self, client):
        resp = client.get("/api/v1/status/clear")
        data = resp.json()
        valid_tones = {t.value for t in ClearTone}
        assert data["tone"] in valid_tones

    def test_message_from_calm_pool(self, client):
        """Default (no emotion context) uses calm tone."""
        resp = client.get("/api/v1/status/clear")
        data = resp.json()
        assert data["tone"] == "calm"
        assert data["message"] in _CALM_MESSAGES

    def test_varied_messages_across_calls(self, client):
        """Multiple calls produce varied messages."""
        messages = set()
        for _ in range(30):
            resp = client.get("/api/v1/status/clear")
            messages.add(resp.json()["message"])
        assert len(messages) >= 2, "Clear messages were all identical"

    def test_no_shame_language_in_response(self, client):
        """Responses never contain shame/guilt language."""
        import re

        for _ in range(50):
            resp = client.get("/api/v1/status/clear")
            message = resp.json()["message"].lower()
            for shame_word in _SHAME_WORDS:
                pattern = r"\b" + re.escape(shame_word) + r"\b"
                assert not re.search(pattern, message), (
                    f"Shame word '{shame_word}' in API response: '{message}'"
                )

    def test_message_is_brief(self, client):
        """Messages are suitable for TTS — under 12 words."""
        for _ in range(50):
            resp = client.get("/api/v1/status/clear")
            message = resp.json()["message"]
            word_count = len(message.split())
            assert word_count <= 12, (
                f"Message too long: '{message}' ({word_count} words)"
            )


# ---------------------------------------------------------------------------
# POST /api/v1/status/check
# ---------------------------------------------------------------------------


class TestCheckStatusEndpoint:
    """Tests for the POST /status/check endpoint."""

    def test_empty_store_returns_clear(self, client):
        """With no tasks in the store, status is clear."""
        resp = client.post(
            "/api/v1/status/check",
            json={"user_id": "test-user"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_clear"] is True
        assert data["clear_message"] is not None
        assert data["clear_message"]["is_clear"] is True
        assert data["clear_message"]["message"] != ""
        assert data["tasks_to_surface"] == 0

    def test_clear_message_has_tone(self, client):
        resp = client.post(
            "/api/v1/status/check",
            json={"user_id": "test-user"},
        )
        data = resp.json()
        valid_tones = {t.value for t in ClearTone}
        assert data["clear_message"]["tone"] in valid_tones

    def test_energy_accepted(self, client):
        """Energy levels are accepted in the request."""
        for energy in ["low", "medium", "high"]:
            resp = client.post(
                "/api/v1/status/check",
                json={"user_id": "test-user", "energy": energy},
            )
            assert resp.status_code == 200

    def test_valence_accepted(self, client):
        """Mood valence is accepted in the request."""
        resp = client.post(
            "/api/v1/status/check",
            json={
                "user_id": "test-user",
                "current_valence": -0.5,
                "current_arousal": 0.3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_clear"] is True

    def test_time_of_day_passed_through(self, client):
        resp = client.post(
            "/api/v1/status/check",
            json={"user_id": "test-user", "time_of_day": "evening"},
        )
        data = resp.json()
        assert data["clear_message"]["time_of_day"] == "evening"

    def test_no_shame_in_check_response(self, client):
        """POST check responses are always shame-free."""
        import re

        for _ in range(30):
            resp = client.post(
                "/api/v1/status/check",
                json={"user_id": "test-user"},
            )
            data = resp.json()
            message = data["clear_message"]["message"].lower()
            for shame_word in _SHAME_WORDS:
                pattern = r"\b" + re.escape(shame_word) + r"\b"
                assert not re.search(pattern, message), (
                    f"Shame word '{shame_word}' in check response: '{message}'"
                )


# ---------------------------------------------------------------------------
# Integration: surfacing empty → clear message flow
# ---------------------------------------------------------------------------


class TestSurfacingClearIntegration:
    """End-to-end: surfacing returns no tasks → clear message."""

    def test_surfacing_empty_triggers_clear_in_check(self, client):
        """When surfacing returns empty, /status/check returns clear."""
        # Check status — no tasks exist
        resp = client.post(
            "/api/v1/status/check",
            json={"user_id": "user-1"},
        )
        data = resp.json()
        assert data["is_clear"] is True
        assert data["clear_message"]["message"] != ""
        assert data["tasks_to_surface"] == 0

    def test_clear_endpoint_always_affirms(self, client):
        """GET /status/clear always returns a positive affirmation."""
        for _ in range(10):
            resp = client.get("/api/v1/status/clear")
            data = resp.json()
            assert data["is_clear"] is True
            assert len(data["message"]) > 0

    def test_clear_message_response_schema(self, client):
        """Verify the full response schema."""
        resp = client.get("/api/v1/status/clear")
        data = resp.json()
        assert "is_clear" in data
        assert "message" in data
        assert "tone" in data
        assert "total_tasks_checked" in data
        # time_of_day may be null
        assert "time_of_day" in data
