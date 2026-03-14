"""E2E Scenario 1: Health check and app lifecycle.

Verifies the app boots correctly, health endpoint responds,
and basic connectivity works through the full ASGI stack.

Cross-cutting concerns exercised:
- ASGI lifecycle: startup/shutdown hooks fire correctly
- Middleware chain: request flows through all registered middleware layers
- Configuration loading: BlurtConfig is resolved from environment (cloud/local)
- HTTP transport: health endpoint responds with version and session metadata
- Deployment mode agnosticism: health check works under both BLURT_MODE values
"""

from __future__ import annotations

import httpx
import pytest


pytestmark = pytest.mark.asyncio


async def test_health_endpoint_returns_ok(client: httpx.AsyncClient):
    """Health endpoint returns status ok with version and session count."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "active_sessions" in data


async def test_health_active_sessions_default_zero(client: httpx.AsyncClient):
    """No active WebSocket sessions on fresh app start."""
    resp = await client.get("/health")
    data = resp.json()
    assert data["active_sessions"] == 0


async def test_nonexistent_route_returns_404(client: httpx.AsyncClient):
    """Unknown routes return 404, not 500."""
    resp = await client.get("/api/v1/nonexistent")
    assert resp.status_code == 404
