"""E2E Scenario: Egress guard activation on mode switch.

Validates that:
- Egress guard is inactive in cloud mode (default)
- Egress guard activates in local mode
- Outbound requests are blocked in local mode
- The /egress-status diagnostic endpoint returns correct violation counts
- Mode switching correctly toggles guard state

Each test creates its own app instance with the appropriate mode
configuration to test cross-cutting egress guard behavior.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from blurt.config.settings import BlurtConfig, DeploymentMode
from blurt.core.app import create_app
from blurt.middleware.egress_guard import (
    EgressBlockedError,
    EgressGuard,
    GuardedTransport,
    install_egress_guards,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures — separate apps for cloud and local modes
# ---------------------------------------------------------------------------


@pytest.fixture
def cloud_app() -> FastAPI:
    """App configured with BLURT_MODE=cloud (egress guard inactive)."""
    config = BlurtConfig(mode=DeploymentMode.CLOUD, debug=True)
    return create_app(config)


@pytest.fixture
def local_app() -> FastAPI:
    """App configured with BLURT_MODE=local (egress guard active)."""
    config = BlurtConfig(mode=DeploymentMode.LOCAL, debug=True)
    app = create_app(config)
    # Deactivate socket guard to avoid test interference with other tests
    # The socket-level guard monkey-patches socket.connect globally which
    # would break the ASGI test transport. We test it separately.
    guard: EgressGuard = app.state.egress_guard
    if guard._socket_guard and guard._socket_guard.active:
        guard._socket_guard.deactivate()
    return app


@pytest_asyncio.fixture
async def cloud_client(cloud_app: FastAPI):
    """httpx AsyncClient for the cloud-mode app."""
    transport = httpx.ASGITransport(app=cloud_app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def local_client(local_app: FastAPI):
    """httpx AsyncClient for the local-mode app."""
    transport = httpx.ASGITransport(app=local_app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Cloud mode — egress guard INACTIVE
# ---------------------------------------------------------------------------


async def test_cloud_mode_egress_guard_is_not_enabled(cloud_app: FastAPI):
    """In cloud mode, the egress guard should exist but not be enabled."""
    guard: EgressGuard = cloud_app.state.egress_guard
    assert guard.enabled is False


async def test_cloud_mode_health_endpoint_accessible(
    cloud_client: httpx.AsyncClient,
):
    """Cloud mode app serves health endpoint normally."""
    resp = await cloud_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


async def test_cloud_mode_check_destination_allows_external(
    cloud_app: FastAPI,
):
    """In cloud mode, check_destination allows any host (guard inactive)."""
    guard: EgressGuard = cloud_app.state.egress_guard
    assert guard.check_destination("api.gemini.google.com") is True
    assert guard.check_destination("calendar.googleapis.com") is True
    assert guard.check_destination("example.com") is True
    assert guard.check_destination("localhost") is True


async def test_cloud_mode_status_shows_disabled(cloud_app: FastAPI):
    """In cloud mode, guard status shows disabled with zero violations."""
    guard: EgressGuard = cloud_app.state.egress_guard
    status = guard.status()
    assert status["enabled"] is False
    assert status["socket_guard_active"] is False
    assert status["violation_count"] == 0
    assert status["recent_violations"] == []


# ---------------------------------------------------------------------------
# Local mode — egress guard ACTIVE
# ---------------------------------------------------------------------------


async def test_local_mode_egress_guard_is_enabled(local_app: FastAPI):
    """In local mode, the egress guard should be active."""
    guard: EgressGuard = local_app.state.egress_guard
    assert guard.enabled is True


async def test_local_mode_health_endpoint_still_accessible(
    local_client: httpx.AsyncClient,
):
    """Local mode app serves health endpoint (internal routes still work)."""
    resp = await local_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


async def test_local_mode_check_destination_blocks_external(
    local_app: FastAPI,
):
    """In local mode, check_destination blocks non-loopback hosts."""
    guard: EgressGuard = local_app.state.egress_guard
    assert guard.check_destination("api.gemini.google.com") is False
    assert guard.check_destination("calendar.googleapis.com") is False
    assert guard.check_destination("example.com") is False


async def test_local_mode_check_destination_allows_localhost(
    local_app: FastAPI,
):
    """In local mode, loopback addresses are still allowed."""
    guard: EgressGuard = local_app.state.egress_guard
    assert guard.check_destination("localhost") is True
    assert guard.check_destination("127.0.0.1") is True
    assert guard.check_destination("::1") is True
    assert guard.check_destination("0.0.0.0") is True


# ---------------------------------------------------------------------------
# /egress-status diagnostic endpoint
# ---------------------------------------------------------------------------


async def test_egress_status_endpoint_returns_guard_state_local(
    local_client: httpx.AsyncClient,
):
    """The /egress-status diagnostic endpoint returns guard status in local mode."""
    resp = await local_client.get("/egress-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert "violation_count" in data
    assert "recent_violations" in data
    assert isinstance(data["violation_count"], int)


async def test_egress_status_violation_count_starts_at_zero(
    local_client: httpx.AsyncClient,
):
    """Fresh local-mode app starts with zero violations."""
    resp = await local_client.get("/egress-status")
    data = resp.json()
    assert data["violation_count"] == 0
    assert data["recent_violations"] == []


async def test_egress_status_reflects_violations_after_blocked_attempt(
    local_app: FastAPI,
    local_client: httpx.AsyncClient,
):
    """After a blocked egress attempt, /egress-status reflects the violation count."""
    guard: EgressGuard = local_app.state.egress_guard

    # Simulate blocked outbound requests using the guarded transport
    transport = guard.create_guarded_transport()
    async with httpx.AsyncClient(transport=transport) as blocked_client:
        for i in range(3):
            with pytest.raises(EgressBlockedError):
                await blocked_client.get(f"https://external-api-{i}.example.com/data")

    # Now check /egress-status via the real API
    resp = await local_client.get("/egress-status")
    data = resp.json()
    assert data["violation_count"] == 3
    assert len(data["recent_violations"]) == 3

    # Each violation should have required fields
    for v in data["recent_violations"]:
        assert "timestamp" in v
        assert "destination" in v
        assert "layer" in v
        assert v["severity"] == "blocked"
        assert "example.com" in v["destination"]


async def test_egress_status_caps_recent_violations(
    local_app: FastAPI,
    local_client: httpx.AsyncClient,
):
    """The recent_violations list in /egress-status is capped at 10 entries."""
    guard: EgressGuard = local_app.state.egress_guard

    # Generate 15 violations
    transport = guard.create_guarded_transport()
    async with httpx.AsyncClient(transport=transport) as blocked_client:
        for i in range(15):
            with pytest.raises(EgressBlockedError):
                await blocked_client.get(f"https://host-{i}.example.com/")

    resp = await local_client.get("/egress-status")
    data = resp.json()
    assert data["violation_count"] == 15
    # recent_violations is capped at the 10 most recent
    assert len(data["recent_violations"]) == 10


# ---------------------------------------------------------------------------
# Guarded transport — blocks outbound requests
# ---------------------------------------------------------------------------


async def test_guarded_transport_blocks_external_requests(
    local_app: FastAPI,
):
    """GuardedTransport raises EgressBlockedError for external hosts."""
    guard: EgressGuard = local_app.state.egress_guard
    transport = guard.create_guarded_transport()

    async with httpx.AsyncClient(transport=transport) as blocked_client:
        with pytest.raises(EgressBlockedError) as exc_info:
            await blocked_client.get("https://api.gemini.google.com/v1/models")

        assert exc_info.value.layer == "httpx"
        assert "gemini.google.com" in exc_info.value.destination


async def test_guarded_transport_records_violation_details(
    local_app: FastAPI,
):
    """Violations recorded through GuardedTransport have correct metadata."""
    guard: EgressGuard = local_app.state.egress_guard
    guard.clear_violations()

    transport = guard.create_guarded_transport()
    async with httpx.AsyncClient(transport=transport) as blocked_client:
        with pytest.raises(EgressBlockedError):
            await blocked_client.post(
                "https://calendar.googleapis.com/v3/events",
                json={"summary": "test"},
            )

    assert guard.violation_count == 1
    v = guard.violations[0]
    assert v.layer == "httpx"
    assert v.severity.value == "blocked"
    assert "calendar.googleapis.com" in v.destination
    assert v.timestamp > 0


async def test_guarded_transport_multiple_violations_tracked(
    local_app: FastAPI,
):
    """Multiple blocked requests are tracked individually."""
    guard: EgressGuard = local_app.state.egress_guard
    guard.clear_violations()

    transport = guard.create_guarded_transport()
    targets = [
        "https://api.gemini.google.com/classify",
        "https://calendar.googleapis.com/events",
        "https://storage.googleapis.com/upload",
    ]

    async with httpx.AsyncClient(transport=transport) as blocked_client:
        for url in targets:
            with pytest.raises(EgressBlockedError):
                await blocked_client.get(url)

    assert guard.violation_count == 3
    destinations = [v.destination for v in guard.violations]
    assert any("gemini" in d for d in destinations)
    assert any("calendar" in d for d in destinations)
    assert any("storage" in d for d in destinations)


# ---------------------------------------------------------------------------
# Guard activation / deactivation (mode switching)
# ---------------------------------------------------------------------------


async def test_guard_deactivation_allows_destination_checks():
    """After deactivating the guard, check_destination allows all hosts."""
    guard = EgressGuard()
    guard.activate()
    assert guard.check_destination("example.com") is False

    guard.deactivate()
    assert guard.check_destination("example.com") is True
    assert guard.enabled is False


async def test_guard_reactivation_resumes_blocking():
    """Guard can be deactivated and reactivated (simulating mode switch)."""
    guard = EgressGuard()

    # Start active (local mode)
    guard.activate()
    assert guard.enabled is True
    assert guard.check_destination("external.com") is False

    # Switch to cloud mode
    guard.deactivate()
    assert guard.enabled is False
    assert guard.check_destination("external.com") is True

    # Switch back to local mode
    guard.activate()
    assert guard.enabled is True
    assert guard.check_destination("external.com") is False

    # Cleanup
    guard.deactivate()


async def test_guard_status_reflects_mode_transitions():
    """Guard status dict reflects activation state across transitions."""
    guard = EgressGuard()

    # Initially inactive
    status = guard.status()
    assert status["enabled"] is False
    assert status["violation_count"] == 0

    # Activate (local mode)
    guard.activate()
    status = guard.status()
    assert status["enabled"] is True
    assert status["socket_guard_active"] is True

    # Deactivate (cloud mode)
    guard.deactivate()
    status = guard.status()
    assert status["enabled"] is False
    assert status["socket_guard_active"] is False


async def test_clear_violations_resets_count(local_app: FastAPI):
    """clear_violations resets the violation count and list."""
    guard: EgressGuard = local_app.state.egress_guard

    # Generate some violations
    transport = guard.create_guarded_transport()
    async with httpx.AsyncClient(transport=transport) as blocked_client:
        for i in range(5):
            with pytest.raises(EgressBlockedError):
                await blocked_client.get(f"https://host-{i}.test.com/")

    assert guard.violation_count == 5

    guard.clear_violations()
    assert guard.violation_count == 0
    assert guard.violations == []

    # Status also shows zero
    status = guard.status()
    assert status["violation_count"] == 0
    assert status["recent_violations"] == []


# ---------------------------------------------------------------------------
# install_egress_guards integration
# ---------------------------------------------------------------------------


async def test_install_egress_guards_cloud_mode():
    """install_egress_guards in cloud mode creates inactive guard on app.state."""
    app = FastAPI()
    guard = install_egress_guards(app, local_mode=False)

    assert guard.enabled is False
    assert hasattr(app.state, "egress_guard")
    assert app.state.egress_guard is guard


async def test_install_egress_guards_local_mode():
    """install_egress_guards in local mode creates active guard with middleware."""
    app = FastAPI()
    guard = install_egress_guards(app, local_mode=True)

    assert guard.enabled is True
    assert hasattr(app.state, "egress_guard")
    assert app.state.egress_guard is guard

    # Cleanup socket guard to avoid polluting other tests
    guard.deactivate()


async def test_double_activate_is_idempotent():
    """Calling activate() twice does not error or double-install."""
    guard = EgressGuard()
    guard.activate()
    guard.activate()  # Should be a no-op
    assert guard.enabled is True

    guard.deactivate()
    assert guard.enabled is False

    # Double deactivate is also safe
    guard.deactivate()
    assert guard.enabled is False
