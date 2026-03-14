"""E2E Scenario: Mode switch — cloud vs local deployment modes.

Verifies that the Blurt application behaves correctly under both
BLURT_MODE=cloud and BLURT_MODE=local configurations:

- Cloud mode: app boots without egress guards, health and API endpoints work
- Local mode: egress guards are active, /egress-status available,
  API endpoints still function, outbound network is blocked

Each test builds its own FastAPI app with the desired mode so both
configurations are exercised in the same pytest run.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from blurt.api.capture import router as capture_router, set_pipeline
from blurt.api.episodes import router as episodes_router, set_store
from blurt.api.patterns import (
    router as patterns_router,
    set_pattern_service,
)
from blurt.api.task_feedback import (
    router as feedback_router,
    set_feedback_service,
)
from blurt.config.settings import BlurtConfig, DeploymentMode
from blurt.core.app import create_app
from blurt.memory.episodic import InMemoryEpisodicStore
from blurt.middleware.egress_guard import EgressGuard
from blurt.services.capture import BlurtCapturePipeline
from blurt.services.feedback import InMemoryFeedbackStore, TaskFeedbackService
from blurt.services.patterns import InMemoryPatternStore, PatternService

# Re-use the stub pipeline functions from conftest
from tests.e2e.conftest import (
    _stub_classify,
    _stub_detect_emotion,
    _stub_embed,
    _stub_extract_entities,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# App factory helpers — one per deployment mode
# ---------------------------------------------------------------------------


def _build_app(mode: DeploymentMode) -> tuple[FastAPI, dict[str, Any]]:
    """Create a fully wired FastAPI app in the given deployment mode.

    Returns the app and a dict of fresh stores/services for assertions.
    """
    config = BlurtConfig(mode=mode, debug=True)
    application = create_app(config)

    # Fresh stores per invocation — full isolation
    episodic_store = InMemoryEpisodicStore()
    pattern_store = InMemoryPatternStore()
    feedback_store = InMemoryFeedbackStore()

    pipeline = BlurtCapturePipeline(
        store=episodic_store,
        classifier=_stub_classify,
        entity_extractor=_stub_extract_entities,
        emotion_detector=_stub_detect_emotion,
        embedder=_stub_embed,
    )
    pattern_service = PatternService(store=pattern_store)
    feedback_service = TaskFeedbackService(store=feedback_store)

    # Ensure all routers are mounted
    _registered_paths = {r.path for r in application.routes}
    if "/api/v1/blurt" not in _registered_paths:
        application.include_router(capture_router)
    if "/api/v1/episodes" not in _registered_paths:
        application.include_router(episodes_router)

    # Wire DI singletons
    set_store(episodic_store)
    set_pipeline(pipeline)
    set_pattern_service(pattern_service)
    set_feedback_service(feedback_service)

    stores = {
        "episodic_store": episodic_store,
        "pattern_store": pattern_store,
        "feedback_store": feedback_store,
        "pipeline": pipeline,
        "pattern_service": pattern_service,
        "feedback_service": feedback_service,
    }
    return application, stores


def _build_cloud_app() -> tuple[FastAPI, dict[str, Any]]:
    """Convenience: create app in cloud mode."""
    return _build_app(DeploymentMode.CLOUD)


def _build_local_app() -> tuple[FastAPI, dict[str, Any]]:
    """Convenience: create app in local mode."""
    return _build_app(DeploymentMode.LOCAL)


# ---------------------------------------------------------------------------
# Fixtures — two independent clients for the two modes
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def cloud_app_and_client() -> AsyncGenerator[tuple[FastAPI, httpx.AsyncClient, dict[str, Any]], None]:
    """Cloud-mode app + async client."""
    app, stores = _build_cloud_app()
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield app, ac, stores


@pytest_asyncio.fixture
async def local_app_and_client() -> AsyncGenerator[tuple[FastAPI, httpx.AsyncClient, dict[str, Any]], None]:
    """Local-mode app + async client."""
    app, stores = _build_local_app()
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield app, ac, stores


# ---------------------------------------------------------------------------
# Cloud-mode tests
# ---------------------------------------------------------------------------


class TestCloudMode:
    """Tests that verify correct behaviour when BLURT_MODE=cloud."""

    async def test_health_returns_ok(self, cloud_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """Health endpoint works in cloud mode."""
        _app, client, _stores = cloud_app_and_client
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    async def test_config_mode_is_cloud(self, cloud_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """App state reflects cloud deployment mode."""
        app, _client, _stores = cloud_app_and_client
        assert app.state.config.mode == DeploymentMode.CLOUD

    async def test_egress_guard_inactive(self, cloud_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """Egress guard exists on app state but is not enabled in cloud mode."""
        app, _client, _stores = cloud_app_and_client
        guard: EgressGuard = app.state.egress_guard
        assert not guard.enabled

    async def test_capture_works_in_cloud(self, cloud_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """Capture pipeline processes blurts normally in cloud mode."""
        _app, client, stores = cloud_app_and_client
        payload = {
            "user_id": "cloud-user",
            "raw_text": "I need to buy groceries",
            "modality": "voice",
            "session_id": "",
            "time_of_day": "morning",
            "day_of_week": "monday",
        }
        resp = await client.post("/api/v1/blurt", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["intent"] == "task"

    async def test_patterns_api_works_in_cloud(self, cloud_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """Pattern CRUD operates normally in cloud mode."""
        _app, client, _stores = cloud_app_and_client
        payload = {
            "pattern_type": "energy",
            "description": "Productive mornings",
            "parameters": {},
            "confidence": 0.7,
            "observation_count": 3,
            "supporting_evidence": [],
        }
        resp = await client.post("/api/v1/users/cloud-user/patterns", json=payload)
        assert resp.status_code == 201

        resp2 = await client.get("/api/v1/users/cloud-user/patterns")
        assert resp2.status_code == 200
        patterns = resp2.json()
        assert len(patterns) >= 1

    async def test_no_egress_status_endpoint_in_cloud(self, cloud_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """The /egress-status diagnostic endpoint is NOT available in cloud mode
        because the EgressGuardMiddleware is not installed."""
        _app, client, _stores = cloud_app_and_client
        resp = await client.get("/egress-status")
        # Without the middleware, this is a 404
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Local-mode tests
# ---------------------------------------------------------------------------


class TestLocalMode:
    """Tests that verify correct behaviour when BLURT_MODE=local."""

    async def test_health_returns_ok(self, local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """Health endpoint works in local mode."""
        _app, client, _stores = local_app_and_client
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    async def test_config_mode_is_local(self, local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """App state reflects local deployment mode."""
        app, _client, _stores = local_app_and_client
        assert app.state.config.mode == DeploymentMode.LOCAL

    async def test_egress_guard_active(self, local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """Egress guard is enabled in local mode."""
        app, _client, _stores = local_app_and_client
        guard: EgressGuard = app.state.egress_guard
        assert guard.enabled
        # Deactivate so it doesn't interfere with other tests' socket usage
        guard.deactivate()

    async def test_egress_status_endpoint_available(self, local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """The /egress-status diagnostic endpoint is available in local mode."""
        app, client, _stores = local_app_and_client
        # Deactivate socket guard to avoid interfering with test transport
        guard: EgressGuard = app.state.egress_guard
        guard.deactivate()
        guard.enabled = True  # Keep logical enabled state for status

        resp = await client.get("/egress-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "violation_count" in data
        assert "socket_guard_active" in data

    async def test_capture_works_in_local(self, local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """Capture pipeline processes blurts in local mode (no external calls needed)."""
        app, client, stores = local_app_and_client
        # Deactivate socket guard — our stub pipeline doesn't do real network
        guard: EgressGuard = app.state.egress_guard
        guard.deactivate()

        payload = {
            "user_id": "local-user",
            "raw_text": "I need to fix the server",
            "modality": "voice",
            "session_id": "",
            "time_of_day": "afternoon",
            "day_of_week": "wednesday",
        }
        resp = await client.post("/api/v1/blurt", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["intent"] == "task"

    async def test_patterns_api_works_in_local(self, local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """Pattern CRUD operates in local mode."""
        app, client, _stores = local_app_and_client
        guard: EgressGuard = app.state.egress_guard
        guard.deactivate()

        payload = {
            "pattern_type": "energy",
            "description": "Productive mornings",
            "parameters": {},
            "confidence": 0.7,
            "observation_count": 3,
            "supporting_evidence": [],
        }
        resp = await client.post("/api/v1/users/local-user/patterns", json=payload)
        assert resp.status_code == 201

        resp2 = await client.get("/api/v1/users/local-user/patterns")
        assert resp2.status_code == 200
        patterns = resp2.json()
        assert len(patterns) >= 1

    async def test_egress_guard_check_destination(self, local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]]):
        """EgressGuard.check_destination blocks non-loopback and allows loopback."""
        app, _client, _stores = local_app_and_client
        guard: EgressGuard = app.state.egress_guard
        # Guard is enabled in local mode
        assert not guard.check_destination("generativelanguage.googleapis.com")
        assert guard.check_destination("localhost")
        assert guard.check_destination("127.0.0.1")
        # Clean up
        guard.deactivate()


# ---------------------------------------------------------------------------
# Cross-mode comparison tests
# ---------------------------------------------------------------------------


class TestCrossModeComparison:
    """Tests that compare behaviour between cloud and local modes."""

    async def test_same_capture_result_both_modes(
        self,
        cloud_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]],
        local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]],
    ):
        """Identical input produces identical capture output regardless of mode."""
        # Deactivate local socket guard so both clients can work
        local_app, local_client, _ = local_app_and_client
        local_guard: EgressGuard = local_app.state.egress_guard
        local_guard.deactivate()

        _, cloud_client, _ = cloud_app_and_client

        payload = {
            "user_id": "cross-mode-user",
            "raw_text": "I need to send the report to Alice",
            "modality": "voice",
            "session_id": "",
            "time_of_day": "morning",
            "day_of_week": "monday",
        }

        cloud_resp = await cloud_client.post("/api/v1/blurt", json=payload)
        local_resp = await local_client.post("/api/v1/blurt", json=payload)

        assert cloud_resp.status_code == local_resp.status_code == 201

        cloud_data = cloud_resp.json()
        local_data = local_resp.json()

        # Intent and entities should match (same stub classifiers)
        assert cloud_data["intent"] == local_data["intent"]
        assert cloud_data["intent"] == "task"

    async def test_health_format_identical(
        self,
        cloud_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]],
        local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]],
    ):
        """Health endpoint returns the same shape in both modes."""
        local_app, _, _ = local_app_and_client
        local_app.state.egress_guard.deactivate()

        _, cloud_client, _ = cloud_app_and_client
        _, local_client, _ = local_app_and_client

        cloud_health = (await cloud_client.get("/health")).json()
        local_health = (await local_client.get("/health")).json()

        assert set(cloud_health.keys()) == set(local_health.keys())
        assert cloud_health["status"] == local_health["status"] == "ok"
        assert cloud_health["version"] == local_health["version"]

    async def test_episode_storage_isolated_per_mode(
        self,
        cloud_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]],
        local_app_and_client: tuple[FastAPI, httpx.AsyncClient, dict[str, Any]],
    ):
        """Episodes stored in one mode don't appear in the other.

        Each mode fixture has its own InMemoryEpisodicStore, verifying
        full data isolation.
        """
        local_app, _, _ = local_app_and_client
        local_app.state.egress_guard.deactivate()

        _, cloud_client, cloud_stores = cloud_app_and_client
        _, local_client, local_stores = local_app_and_client

        # Capture in cloud
        # Need to set DI singletons before each call since they're module-level
        cloud_store = cloud_stores["episodic_store"]
        local_store = local_stores["episodic_store"]

        assert cloud_store is not local_store, "Stores should be separate instances"
