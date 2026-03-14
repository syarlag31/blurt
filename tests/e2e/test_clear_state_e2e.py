"""End-to-end tests for the you-are-clear message flow.

Validates the full pipeline: task surfacing engine finds no tasks →
clear-state service generates an affirming message → API returns it.

Anti-shame design principles verified end-to-end:
- No streaks, guilt, overdue, or forced engagement
- "No tasks pending" is celebrated, not treated as absence
- Messages vary, are brief, and match emotional context
- Completed/deferred/dropped tasks don't haunt the user
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blurt.api.clear_state import (
    router as clear_state_router,
    set_clear_service,
    set_scoring_engine,
)
from blurt.api.task_surfacing import (
    router as surfacing_router,
    set_surfacing_service,
)
from blurt.services.clear_state import (
    ClearStateService,
    _MESSAGE_POOLS,
    _SHAME_WORDS,
)
from blurt.services.task_surfacing import (
    SurfaceableTask,
    TaskScoringEngine,
    TaskStatus,
)
from blurt.services.task_surfacing_query import (
    InMemoryTaskStore,
    TaskSurfacingQueryService,
)

NOW = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(clear_state_router)
    app.include_router(surfacing_router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def store():
    return InMemoryTaskStore()


@pytest.fixture
def surfacing_service(store):
    svc = TaskSurfacingQueryService(store=store)
    set_surfacing_service(svc)
    yield svc
    set_surfacing_service(None)  # type: ignore


@pytest.fixture
def clear_service():
    svc = ClearStateService()
    set_clear_service(svc)
    yield svc
    set_clear_service(None)  # type: ignore


@pytest.fixture
def scoring_engine():
    engine = TaskScoringEngine()
    set_scoring_engine(engine)
    yield engine
    set_scoring_engine(None)  # type: ignore


@pytest.fixture(autouse=True)
def setup_all(surfacing_service, clear_service, scoring_engine):
    """Ensure all services are wired up for each test."""
    pass


# ── Helpers ──────────────────────────────────────────────────────


ALL_CLEAR_MESSAGES: set[str] = set()
for pool in _MESSAGE_POOLS.values():
    ALL_CLEAR_MESSAGES.update(pool)


def _assert_no_shame(text: str) -> None:
    """Assert that text contains no shame/guilt language."""
    text_lower = text.lower()
    for shame_word in _SHAME_WORDS:
        pattern = r"\b" + re.escape(shame_word) + r"\b"
        assert not re.search(pattern, text_lower), (
            f"Shame word '{shame_word}' found in: '{text}'"
        )


# ── E2E: Empty store → clear message ─────────────────────────────


class TestEmptyStoreClearFlow:
    """When no tasks exist, the user gets a positive clear message."""

    def test_surface_empty_store_returns_clear_message(self, client):
        """POST /tasks/surface with empty store → anti-shame message."""
        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        assert data["returned_count"] == 0
        assert data["tasks"] == []
        assert data["message"] != ""
        _assert_no_shame(data["message"])

    def test_quick_surface_empty_returns_message(self, client):
        """GET /tasks/surface with empty store → anti-shame message."""
        resp = client.get("/api/v1/tasks/surface")
        data = resp.json()
        assert data["returned_count"] == 0
        assert data["message"] != ""

    def test_status_check_empty_returns_clear(self, client):
        """POST /status/check with empty store → is_clear=True."""
        resp = client.post(
            "/api/v1/status/check",
            json={"user_id": "test-user"},
        )
        data = resp.json()
        assert data["is_clear"] is True
        assert data["clear_message"] is not None
        _assert_no_shame(data["clear_message"]["message"])

    def test_status_clear_always_affirms(self, client):
        """GET /status/clear → always a positive affirmation."""
        resp = client.get("/api/v1/status/clear")
        data = resp.json()
        assert data["is_clear"] is True
        assert data["message"] in ALL_CLEAR_MESSAGES
        _assert_no_shame(data["message"])


# ── E2E: All tasks completed → clear message ──────────────────────


class TestCompletedTasksClearFlow:
    """When all tasks are completed, the user is all clear."""

    def test_completed_tasks_produce_clear(self, client, store):
        """Completed tasks should not surface — user is clear."""
        task = SurfaceableTask(
            content="Already done",
            status=TaskStatus.COMPLETED,
            created_at=NOW,
        )
        store.add_task(task)

        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        assert data["returned_count"] == 0
        assert data["message"] != ""
        _assert_no_shame(data["message"])

    def test_dropped_tasks_produce_clear(self, client, store):
        """Dropped tasks should not surface — shame-free drop."""
        task = SurfaceableTask(
            content="Changed my mind",
            status=TaskStatus.DROPPED,
            created_at=NOW,
        )
        store.add_task(task)

        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        assert data["returned_count"] == 0
        assert data["message"] != ""

    def test_deferred_tasks_produce_clear(self, client, store):
        """Deferred tasks should not surface — user chose to defer."""
        task = SurfaceableTask(
            content="Not right now",
            status=TaskStatus.DEFERRED,
            created_at=NOW,
        )
        store.add_task(task)

        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        assert data["returned_count"] == 0
        assert data["message"] != ""

    def test_mixed_completed_dropped_deferred_all_clear(self, client, store):
        """Mix of non-active statuses → still clear."""
        for status in [TaskStatus.COMPLETED, TaskStatus.DROPPED, TaskStatus.DEFERRED]:
            store.add_task(
                SurfaceableTask(
                    content=f"Task with status {status.value}",
                    status=status,
                    created_at=NOW,
                )
            )

        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        assert data["returned_count"] == 0
        assert data["message"] != ""
        _assert_no_shame(data["message"])


# ── E2E: All tasks below threshold → clear ────────────────────────


class TestBelowThresholdClearFlow:
    """When all tasks score below the surfacing threshold, user is clear."""

    def test_high_threshold_no_surface(self, client, store):
        """With a very high min_score, no tasks pass → clear."""
        store.add_task(
            SurfaceableTask(
                content="Low priority",
                created_at=NOW,
            )
        )

        resp = client.post(
            "/api/v1/tasks/surface",
            json={"min_score": 0.99},
        )
        data = resp.json()
        assert data["returned_count"] == 0
        assert data["message"] != ""
        _assert_no_shame(data["message"])


# ── E2E: Shame-free design validation ────────────────────────────


class TestShameFreeDesignE2E:
    """Comprehensive anti-shame validation across the full pipeline."""

    def test_no_overdue_language_anywhere(self, client, store):
        """Past-due tasks don't generate 'overdue' language."""
        # Task with a due date in the past
        store.add_task(
            SurfaceableTask(
                content="Past due task",
                due_at=NOW - timedelta(days=3),
                created_at=NOW - timedelta(days=5),
            )
        )

        # Surface (task might appear, but with gentle language)
        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        if data["returned_count"] > 0:
            for task in data["tasks"]:
                _assert_no_shame(task["surfacing_reason"])
        else:
            _assert_no_shame(data["message"])

    def test_no_streak_counting(self, client):
        """No streak or consecutive-day counting in clear messages."""
        resp = client.get("/api/v1/status/clear")
        data = resp.json()
        message = data["message"].lower()
        streak_words = ["streak", "consecutive", "day count", "in a row"]
        for word in streak_words:
            assert word not in message

    def test_no_forced_engagement(self, client):
        """Clear state never pushes user to create/find tasks."""
        forced_phrases = [
            "you could",
            "you should",
            "why not",
            "try adding",
            "create a task",
            "get started",
        ]
        for _ in range(50):
            resp = client.get("/api/v1/status/clear")
            message = resp.json()["message"].lower()
            for phrase in forced_phrases:
                assert phrase not in message, (
                    f"Forced engagement '{phrase}' in: '{message}'"
                )

    def test_clear_state_is_always_valid(self, client):
        """Zero pending tasks is a valid, celebrated state."""
        resp = client.post(
            "/api/v1/status/check",
            json={"user_id": "happy-user"},
        )
        data = resp.json()
        # The response explicitly affirms clarity
        assert data["is_clear"] is True
        msg = data["clear_message"]["message"]
        # Message should be a positive affirmation
        assert len(msg) > 0
        _assert_no_shame(msg)


# ── E2E: Variety across multiple calls ────────────────────────────


class TestClearMessageVarietyE2E:
    """Messages vary to feel human, not robotic."""

    def test_surfacing_messages_vary(self, client):
        """Multiple surfacing calls with empty store → varied messages."""
        messages = set()
        for _ in range(30):
            resp = client.post("/api/v1/tasks/surface", json={})
            messages.add(resp.json()["message"])
        assert len(messages) >= 2

    def test_clear_endpoint_messages_vary(self, client):
        """Multiple GET /status/clear calls → varied messages."""
        messages = set()
        for _ in range(30):
            resp = client.get("/api/v1/status/clear")
            messages.add(resp.json()["message"])
        assert len(messages) >= 2

    def test_check_endpoint_messages_vary(self, client):
        """Multiple POST /status/check calls → varied messages."""
        messages = set()
        for _ in range(30):
            resp = client.post(
                "/api/v1/status/check",
                json={"user_id": "test-user"},
            )
            data = resp.json()
            messages.add(data["clear_message"]["message"])
        assert len(messages) >= 2


# ── E2E: Transition from tasks → clear ───────────────────────────


class TestTaskToClearTransition:
    """When tasks are completed and the user checks again, they get a clear message."""

    def test_add_complete_then_clear(self, client, surfacing_service, store):
        """Add a task, complete it, surface → clear message."""
        # Add a task
        add_resp = client.post("/api/v1/tasks", json={
            "content": "Finish report",
        })
        assert add_resp.status_code == 201
        task_id = add_resp.json()["task_id"]

        # Surface — should find the task
        surf_resp = client.post("/api/v1/tasks/surface", json={})
        assert surf_resp.json()["returned_count"] == 1

        # Complete the task
        task = store.get_task(task_id)
        assert task is not None
        task.status = TaskStatus.COMPLETED
        store.update_task(task)

        # Surface again — should be clear
        clear_resp = client.post("/api/v1/tasks/surface", json={})
        data = clear_resp.json()
        assert data["returned_count"] == 0
        assert data["message"] != ""
        _assert_no_shame(data["message"])

    def test_add_drop_then_clear(self, client, surfacing_service, store):
        """Add a task, drop it (shame-free), surface → clear message."""
        add_resp = client.post("/api/v1/tasks", json={
            "content": "Maybe later",
        })
        task_id = add_resp.json()["task_id"]

        # Drop the task — this is a valid, shame-free choice
        task = store.get_task(task_id)
        assert task is not None
        task.status = TaskStatus.DROPPED
        store.update_task(task)

        # Surface — should be clear
        resp = client.post("/api/v1/tasks/surface", json={})
        data = resp.json()
        assert data["returned_count"] == 0
        assert data["message"] != ""
        _assert_no_shame(data["message"])
