"""Tests for the task surfacing API endpoints.

Validates:
- POST /api/v1/tasks/surface — full query with context
- GET  /api/v1/tasks/surface — quick query with query params
- POST /api/v1/tasks — add task
- GET  /api/v1/tasks/{task_id} — get task
- Anti-shame responses
- Score breakdown in responses
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from blurt.api.task_surfacing import (
    router,
    set_surfacing_service,
)
from blurt.services.task_surfacing import (
    EnergyLevel,
    SurfaceableTask,
)
from blurt.services.task_surfacing_query import (
    InMemoryTaskStore,
    TaskSurfacingQueryService,
)

NOW = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def app():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def service():
    store = InMemoryTaskStore()
    svc = TaskSurfacingQueryService(store=store)
    set_surfacing_service(svc)
    yield svc
    set_surfacing_service(None)  # type: ignore


@pytest.fixture
def service_with_tasks(service):
    """Service pre-loaded with test tasks."""
    tasks = [
        SurfaceableTask(
            content="Review PR for auth module",
            intent="task",
            estimated_energy=EnergyLevel.HIGH,
            entity_names=["auth-module", "Alice"],
            project="Backend",
            created_at=NOW,
        ),
        SurfaceableTask(
            content="Buy groceries",
            intent="task",
            estimated_energy=EnergyLevel.LOW,
            created_at=NOW,
        ),
        SurfaceableTask(
            content="Team standup",
            intent="event",
            estimated_energy=EnergyLevel.LOW,
            due_at=NOW + timedelta(hours=2),
            created_at=NOW,
        ),
    ]
    for t in tasks:
        service.add_task(t)
    return service


# ---------------------------------------------------------------------------
# POST /api/v1/tasks/surface
# ---------------------------------------------------------------------------


class TestSurfaceTasksEndpoint:
    """Test the POST surface endpoint."""

    def test_surface_empty_store(self, client, service):
        resp = client.post("/api/v1/tasks/surface", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["returned_count"] == 0
        assert data["tasks"] == []
        assert data["message"] != ""  # Anti-shame message

    def test_surface_returns_ranked_tasks(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={
            "energy": "medium",
            "mood_valence": 0.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["returned_count"] > 0

        # Tasks should be in descending score order
        scores = [t["composite_score"] for t in data["tasks"]]
        assert scores == sorted(scores, reverse=True)

    def test_surface_includes_score_breakdown(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={})
        assert resp.status_code == 200
        data = resp.json()

        for task in data["tasks"]:
            assert "signal_scores" in task
            assert len(task["signal_scores"]) == 6
            for signal in task["signal_scores"]:
                assert "signal" in signal
                assert "value" in signal
                assert "reason" in signal
                assert 0.0 <= signal["value"] <= 1.0

    def test_surface_with_context(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={
            "energy": "high",
            "mood_valence": 0.5,
            "active_entity_names": ["auth-module"],
            "active_project": "Backend",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["returned_count"] > 0

        # The auth module task should rank higher with matching context
        task_contents = [t["content"] for t in data["tasks"]]
        if len(task_contents) >= 2:
            assert task_contents[0] == "Review PR for auth module"

    def test_surface_with_intent_filter(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={
            "include_intents": ["task"],
        })
        assert resp.status_code == 200
        data = resp.json()

        for task in data["tasks"]:
            assert task["intent"] == "task"

    def test_surface_with_energy_cap(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={
            "max_energy": "low",
        })
        assert resp.status_code == 200
        data = resp.json()

        for task in data["tasks"]:
            assert task["estimated_energy"] == "low"

    def test_surface_max_results(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={
            "max_results": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["returned_count"] <= 1

    def test_surface_weights_reported(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "weights_used" in data
        assert "time_relevance" in data["weights_used"]

    def test_surface_custom_weights(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={
            "weights": {
                "time_relevance": 0.5,
                "energy_match": 0.1,
                "context_relevance": 0.1,
                "emotional_alignment": 0.1,
                "momentum": 0.1,
                "freshness": 0.1,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["weights_used"]["time_relevance"] == 0.5

    def test_surface_has_query_id(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        assert "query_id" in data
        assert data["query_id"] != ""

    def test_surface_metadata_counts(self, client, service_with_tasks):
        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        assert data["total_in_store"] == 3
        assert data["total_eligible"] > 0


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/surface
# ---------------------------------------------------------------------------


class TestSurfaceTasksQuickEndpoint:
    """Test the GET quick-surface endpoint."""

    def test_quick_surface_empty(self, client, service):
        resp = client.get("/api/v1/tasks/surface")
        assert resp.status_code == 200
        data = resp.json()
        assert data["returned_count"] == 0

    def test_quick_surface_with_params(self, client, service_with_tasks):
        resp = client.get(
            "/api/v1/tasks/surface",
            params={"energy": "low", "max_results": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["returned_count"] <= 2

    def test_quick_surface_returns_scores(self, client, service_with_tasks):
        resp = client.get("/api/v1/tasks/surface")
        assert resp.status_code == 200
        data = resp.json()
        for task in data["tasks"]:
            assert "composite_score" in task
            assert "signal_scores" in task


# ---------------------------------------------------------------------------
# POST /api/v1/tasks
# ---------------------------------------------------------------------------


class TestAddTaskEndpoint:
    """Test the add-task endpoint."""

    def test_add_task(self, client, service):
        resp = client.post("/api/v1/tasks", json={
            "content": "New task",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["content"] == "New task"
        assert data["status"] == "active"
        assert "task_id" in data

    def test_add_task_then_surface(self, client, service):
        # Add a task
        add_resp = client.post("/api/v1/tasks", json={
            "content": "Surfaceable task",
        })
        assert add_resp.status_code == 201

        # Surface should find it
        surf_resp = client.post("/api/v1/tasks/surface", json={})
        assert surf_resp.status_code == 200
        data = surf_resp.json()
        assert data["returned_count"] == 1
        assert data["tasks"][0]["content"] == "Surfaceable task"

    def test_add_task_with_metadata(self, client, service):
        resp = client.post("/api/v1/tasks", json={
            "content": "Detailed task",
            "intent": "task",
            "estimated_energy": "high",
            "entity_names": ["Alice"],
            "project": "Alpha",
            "tags": ["work", "urgent"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["content"] == "Detailed task"


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/{task_id}
# ---------------------------------------------------------------------------


class TestGetTaskEndpoint:
    """Test the get-task endpoint."""

    def test_get_existing_task(self, client, service):
        # Add a task
        add_resp = client.post("/api/v1/tasks", json={
            "content": "Find me",
        })
        task_id = add_resp.json()["task_id"]

        # Get it
        resp = client.get(f"/api/v1/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Find me"
        assert data["status"] == "active"

    def test_get_nonexistent_task(self, client, service):
        resp = client.get("/api/v1/tasks/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Anti-shame API tests
# ---------------------------------------------------------------------------


class TestAntiShameAPI:
    """Test anti-shame design in API responses."""

    def test_empty_result_no_guilt_language(self, client, service):
        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        message = data["message"].lower()
        guilt_words = ["overdue", "missed", "failed", "behind", "late", "forgotten", "lazy"]
        for word in guilt_words:
            assert word not in message

    def test_no_forced_surfacing(self, client, service):
        """Even with tasks, high min_score can result in empty — that's OK."""
        service.add_task(SurfaceableTask(content="Some task", created_at=NOW))
        resp = client.post("/api/v1/tasks/surface", json={
            "min_score": 0.99,
        })
        data = resp.json()
        # It's fine to return empty — no guilt
        assert data["returned_count"] == 0
