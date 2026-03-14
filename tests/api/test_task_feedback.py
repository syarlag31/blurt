"""Tests for the task feedback API endpoints.

Validates:
- POST /api/v1/tasks/{task_id}/feedback records feedback and returns shame-free ack
- GET /api/v1/tasks/{task_id}/feedback returns aggregated summary
- GET /api/v1/feedback/recent returns recent events for a user
- Thompson Sampling params are updated through the API
- Anti-shame: response messages contain no guilt language
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from blurt.api.task_feedback import (
    router,
    set_feedback_service,
)
from blurt.services.feedback import (
    InMemoryFeedbackStore,
    TaskFeedbackService,
)


# ── Shame word list — none of these should appear in API responses ──

SHAME_WORDS = [
    "overdue", "late", "failed", "guilt", "shame",
    "behind", "missed", "neglected", "forgot", "lazy",
    "procrastinat", "should have", "haven't completed",
]


@pytest.fixture
def app() -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def service() -> TaskFeedbackService:
    svc = TaskFeedbackService(store=InMemoryFeedbackStore())
    set_feedback_service(svc)
    return svc


@pytest.fixture
def client(app: FastAPI, service: TaskFeedbackService) -> TestClient:
    return TestClient(app)


class TestRecordFeedbackEndpoint:
    def test_accept_returns_201(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/tasks/task-1/feedback",
            json={"user_id": "u1", "action": "accept"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["task_id"] == "task-1"
        assert data["action"] == "accept"
        assert data["thompson_update_applied"] is True
        assert data["message"]  # Non-empty acknowledgment

    def test_dismiss_returns_201(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/tasks/task-1/feedback",
            json={"user_id": "u1", "action": "dismiss"},
        )
        assert resp.status_code == 201
        assert resp.json()["action"] == "dismiss"

    def test_snooze_returns_201(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/tasks/task-1/feedback",
            json={
                "user_id": "u1",
                "action": "snooze",
                "snooze_minutes": 45,
            },
        )
        assert resp.status_code == 201
        assert resp.json()["action"] == "snooze"

    def test_snooze_defaults_to_30_minutes(self, client: TestClient) -> None:
        """Snooze without explicit minutes defaults to 30."""
        resp = client.post(
            "/api/v1/tasks/task-1/feedback",
            json={"user_id": "u1", "action": "snooze"},
        )
        assert resp.status_code == 201

    def test_complete_returns_201(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/tasks/task-1/feedback",
            json={"user_id": "u1", "action": "complete"},
        )
        assert resp.status_code == 201
        assert resp.json()["action"] == "complete"

    def test_with_context(self, client: TestClient) -> None:
        """Feedback with full context generates proper context key."""
        resp = client.post(
            "/api/v1/tasks/task-1/feedback",
            json={
                "user_id": "u1",
                "action": "accept",
                "mood_valence": 0.7,
                "energy_level": 0.9,
                "time_of_day": "morning",
                "intent": "reminder",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "morning" in data["context_key"]
        assert "high_energy" in data["context_key"]
        assert "positive_mood" in data["context_key"]

    def test_invalid_action_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/tasks/task-1/feedback",
            json={"user_id": "u1", "action": "invalid_action"},
        )
        assert resp.status_code == 422

    def test_shame_free_responses(self, client: TestClient) -> None:
        """All response messages must be shame-free."""
        for action in ["accept", "dismiss", "snooze", "complete"]:
            resp = client.post(
                "/api/v1/tasks/task-1/feedback",
                json={"user_id": "u1", "action": action},
            )
            message = resp.json()["message"].lower()
            for word in SHAME_WORDS:
                assert word not in message, (
                    f"Shame word '{word}' found in {action} response: {message}"
                )

    def test_thompson_params_updated_via_api(
        self, client: TestClient, service: TaskFeedbackService
    ) -> None:
        """API endpoint actually triggers Thompson Sampling param updates."""
        # Accept the task
        client.post(
            "/api/v1/tasks/task-ts/feedback",
            json={"user_id": "u1", "action": "accept"},
        )
        params = service.store.get_params("task:task-ts")
        assert params.alpha == 2.0  # 1 prior + 1 accept
        assert params.beta == 1.0

        # Dismiss the task
        client.post(
            "/api/v1/tasks/task-ts/feedback",
            json={"user_id": "u1", "action": "dismiss"},
        )
        params = service.store.get_params("task:task-ts")
        assert params.alpha == 2.0
        assert params.beta == 2.0   # 1 prior + 1 dismiss


class TestFeedbackSummaryEndpoint:
    def test_empty_summary(self, client: TestClient) -> None:
        resp = client.get("/api/v1/tasks/nonexistent/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] == 0
        assert data["acceptance_rate"] == 0.0

    def test_summary_after_feedback(self, client: TestClient) -> None:
        # Record some feedback
        for action in ["accept", "dismiss", "complete", "snooze"]:
            client.post(
                "/api/v1/tasks/summary-task/feedback",
                json={"user_id": "u1", "action": action},
            )

        resp = client.get("/api/v1/tasks/summary-task/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] == 4
        assert data["accept_count"] == 1
        assert data["dismiss_count"] == 1
        assert data["snooze_count"] == 1
        assert data["complete_count"] == 1
        assert data["acceptance_rate"] == 0.5  # 2 positive / 4 total


class TestRecentFeedbackEndpoint:
    def test_empty_recent(self, client: TestClient) -> None:
        resp = client.get("/api/v1/feedback/recent", params={"user_id": "u1"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_recent_feedback(self, client: TestClient) -> None:
        client.post(
            "/api/v1/tasks/t1/feedback",
            json={"user_id": "u1", "action": "accept"},
        )
        client.post(
            "/api/v1/tasks/t2/feedback",
            json={"user_id": "u1", "action": "dismiss"},
        )
        client.post(
            "/api/v1/tasks/t3/feedback",
            json={"user_id": "u2", "action": "accept"},
        )

        resp = client.get("/api/v1/feedback/recent", params={"user_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(e["task_id"] in ("t1", "t2") for e in data)

    def test_recent_limit(self, client: TestClient) -> None:
        for i in range(10):
            client.post(
                f"/api/v1/tasks/t{i}/feedback",
                json={"user_id": "u1", "action": "accept"},
            )

        resp = client.get(
            "/api/v1/feedback/recent",
            params={"user_id": "u1", "limit": 3},
        )
        assert len(resp.json()) == 3
