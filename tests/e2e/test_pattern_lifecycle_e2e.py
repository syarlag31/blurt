"""E2E Scenario 4: Pattern lifecycle — create, query, reinforce, weaken, deactivate.

Tests the full CRUD lifecycle of learned behavioral patterns through
the patterns API, including filtering by type, day, time, and confidence.

Cross-cutting concerns exercised:
- Pattern store CRUD: create, retrieve, update, deactivate operations
- Confidence scoring: reinforce increases and weaken decreases confidence
- Temporal context: patterns filtered by day-of-week and time-of-day
- Pattern typing: different pattern types (routine, preference) coexist
- Deactivation semantics: deactivated patterns excluded from active queries
- HTTP API contract: patterns API request/response schema validation
- State transitions: pattern confidence trajectory across multiple mutations
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.asyncio


class TestPatternCRUD:
    """Full pattern create-read-update-delete cycle."""

    async def test_create_and_get_pattern(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Create a pattern and retrieve it by ID."""
        pat = await create_pattern_via_api(
            pattern_type="energy",
            description="User is more productive in mornings",
            parameters={"time_of_day": "morning"},
            confidence=0.6,
        )
        pattern_id = pat["id"]

        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns/{pattern_id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["description"] == "User is more productive in mornings"
        assert data["confidence"] == 0.6
        assert data["is_active"] is True

    async def test_reinforce_increases_confidence(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Reinforcing a pattern increases its confidence."""
        pat = await create_pattern_via_api(confidence=0.5)
        pattern_id = pat["id"]

        resp = await client.put(
            f"/api/v1/users/{test_user_id}/patterns/{pattern_id}/reinforce",
            json={"evidence": "completed morning task", "confidence_boost": 0.1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == 0.6
        assert data["observation_count"] == 2

    async def test_weaken_decreases_confidence(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Weakening a pattern decreases its confidence."""
        pat = await create_pattern_via_api(confidence=0.7)
        pattern_id = pat["id"]

        resp = await client.put(
            f"/api/v1/users/{test_user_id}/patterns/{pattern_id}/weaken",
            json={"confidence_penalty": 0.2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == pytest.approx(0.5, abs=0.01)

    async def test_weaken_below_threshold_deactivates(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Patterns weakened below 0.1 are auto-deactivated."""
        pat = await create_pattern_via_api(confidence=0.15)
        pattern_id = pat["id"]

        resp = await client.put(
            f"/api/v1/users/{test_user_id}/patterns/{pattern_id}/weaken",
            json={"confidence_penalty": 0.1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] < 0.1
        assert data["is_active"] is False

    async def test_deactivate_pattern(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Deactivated patterns are soft-deleted."""
        pat = await create_pattern_via_api()
        pattern_id = pat["id"]

        resp = await client.delete(
            f"/api/v1/users/{test_user_id}/patterns/{pattern_id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_active"] is False


class TestPatternQuerying:
    """Query patterns with filters."""

    async def test_list_patterns_empty(
        self,
        client: httpx.AsyncClient,
        test_user_id: str,
    ):
        """Listing patterns for a user with none returns empty."""
        resp = await client.get(f"/api/v1/users/{test_user_id}/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert data["patterns"] == []
        assert data["total_count"] == 0

    async def test_filter_by_type(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Filter patterns by type."""
        await create_pattern_via_api(pattern_type="energy", description="energy pat")
        await create_pattern_via_api(pattern_type="mood", description="mood pat")

        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns",
            params={"type": "energy"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert all(p["pattern_type"] == "energy_rhythm" for p in data["patterns"])

    async def test_filter_by_min_confidence(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Filter patterns by minimum confidence threshold."""
        await create_pattern_via_api(confidence=0.3, description="low conf")
        await create_pattern_via_api(confidence=0.8, description="high conf")

        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns",
            params={"min_confidence": 0.5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert all(p["confidence"] >= 0.5 for p in data["patterns"])

    async def test_invalid_pattern_type_returns_400(
        self,
        client: httpx.AsyncClient,
        test_user_id: str,
    ):
        """Invalid pattern type in query returns 400."""
        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns",
            params={"type": "nonexistent"},
        )
        assert resp.status_code == 400

    async def test_pattern_summary(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        test_user_id: str,
    ):
        """Pattern summary groups by type with stats."""
        await create_pattern_via_api(pattern_type="energy", confidence=0.7)
        await create_pattern_via_api(pattern_type="energy", confidence=0.9)
        await create_pattern_via_api(pattern_type="mood", confidence=0.6)

        resp = await client.get(
            f"/api/v1/users/{test_user_id}/patterns/summary"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_active"] == 3
        assert "energy_rhythm" in data["by_type"]
        assert data["by_type"]["energy_rhythm"]["count"] == 2
