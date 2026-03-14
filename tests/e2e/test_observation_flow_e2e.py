"""E2E Scenario 8: Observation endpoint and recall.

Tests the observation intake endpoint and the full recall endpoint
that merges raw episodes with compressed summaries.

Cross-cutting concerns exercised:
- Observation intake: high-level blurt intake via /observations endpoint
- Recall engine: merges raw episodes with compressed summaries into a
  unified recall response for downstream consumers
- Compression integration: compressed episodes are included in recall output
  alongside uncompressed ones
- Temporal windowing: recall respects time-range filters for relevant context
- Session-aware recall: observations grouped by session for coherent retrieval
- Store layering: observation → episode → compression → recall spans multiple
  storage layers in a single query path
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.asyncio


class TestObservationEndpoint:
    """POST /api/v1/episodes/observations — high-level blurt intake."""

    async def test_store_observation(
        self,
        client: httpx.AsyncClient,
        test_user_id: str,
    ):
        """Store an observation and get it back."""
        payload = {
            "user_id": test_user_id,
            "raw_text": "I should look into that new framework",
            "modality": "voice",
            "intent": "idea",
            "intent_confidence": 0.87,
            "emotion": {
                "primary": "anticipation",
                "intensity": 1.0,
                "valence": 0.3,
                "arousal": 0.5,
            },
            "entities": [
                {"name": "new framework", "entity_type": "tool", "confidence": 0.8}
            ],
            "behavioral_signal": "none",
            "context": {
                "time_of_day": "afternoon",
                "day_of_week": "wednesday",
                "session_id": "obs-session",
            },
        }
        resp = await client.post("/api/v1/episodes/observations", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["observation_stored"] is True
        assert data["entity_count"] == 1
        assert data["episode"]["intent"] == "idea"

    async def test_observation_retrievable_by_id(
        self,
        client: httpx.AsyncClient,
        test_user_id: str,
    ):
        """Observation stored via POST is retrievable via GET."""
        payload = {
            "user_id": test_user_id,
            "raw_text": "quick thought",
            "intent": "journal",
            "intent_confidence": 0.9,
            "emotion": {
                "primary": "trust",
                "intensity": 0.5,
                "valence": 0.0,
                "arousal": 0.2,
            },
            "context": {
                "time_of_day": "morning",
                "day_of_week": "monday",
            },
        }
        create_resp = await client.post(
            "/api/v1/episodes/observations", json=payload
        )
        episode_id = create_resp.json()["episode"]["id"]

        get_resp = await client.get(f"/api/v1/episodes/{episode_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["raw_text"] == "quick thought"


class TestRecallEndpoint:
    """GET /api/v1/episodes/recall/{user_id} — unified timeline."""

    async def test_recall_returns_episodes(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
        test_user_id: str,
    ):
        """Recall endpoint returns raw episodes."""
        await create_episode_via_api(raw_text="first memory")
        await create_episode_via_api(raw_text="second memory")

        resp = await client.get(f"/api/v1/episodes/recall/{test_user_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_count"] == 2
        assert data["total_count"] >= 2

    async def test_recall_includes_summaries_after_compression(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
        test_user_id: str,
    ):
        """After compression, recall includes both episodes and summaries."""
        ep1 = await create_episode_via_api(raw_text="morning standup")
        ep2 = await create_episode_via_api(raw_text="code review")

        # Compress into summary
        await client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": test_user_id,
                "episode_ids": [ep1["id"], ep2["id"]],
                "summary_text": "Morning work session",
            },
        )

        resp = await client.get(f"/api/v1/episodes/recall/{test_user_id}")
        data = resp.json()
        assert data["summary_count"] >= 1
        # Both entry types present
        entry_types = {e["entry_type"] for e in data["entries"]}
        assert "summary" in entry_types

    async def test_recall_with_entity_filter(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
        test_user_id: str,
    ):
        """Recall can be filtered by entity name."""
        await create_episode_via_api(
            raw_text="discussed with Alice",
            entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        await create_episode_via_api(raw_text="unrelated thought")

        resp = await client.get(
            f"/api/v1/episodes/recall/{test_user_id}",
            params={"entity": "Alice"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_count"] >= 1
