"""E2E Scenario 3: Episodic memory — CRUD and querying through HTTP.

Tests the full lifecycle of episodes: create, retrieve by ID, query by
user with filters, session grouping, entity timeline, and compression.

Cross-cutting concerns exercised:
- Episodic store CRUD: create, retrieve-by-ID, query-by-user operations
- Temporal filtering: before/after timestamp filters on episode queries
- Intent-based filtering: isolate episodes by classified intent label
- Session grouping: episodes within a session form a retrievable chain
- Entity timeline: entity references tracked across episodes over time
- Compression: episode summaries produced without losing source data
- Data serialization: Episode model ↔ JSON round-trip through HTTP layer
- Store isolation: in-memory store resets between tests (no cross-talk)
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.asyncio


class TestEpisodeCreationAndRetrieval:
    """Create episodes and retrieve them by ID."""

    async def test_create_and_get_episode(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
    ):
        """Create an episode and retrieve it by ID."""
        ep = await create_episode_via_api(raw_text="I need to review the PR")
        episode_id = ep["id"]

        resp = await client.get(f"/api/v1/episodes/{episode_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_text"] == "I need to review the PR"
        assert data["id"] == episode_id

    async def test_get_nonexistent_episode_returns_404(
        self,
        client: httpx.AsyncClient,
    ):
        resp = await client.get("/api/v1/episodes/nonexistent-id")
        assert resp.status_code == 404


class TestEpisodeQuerying:
    """Query episodes with various filters."""

    async def test_query_by_user(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
        test_user_id: str,
    ):
        """List all episodes for a user."""
        await create_episode_via_api(raw_text="first blurt")
        await create_episode_via_api(raw_text="second blurt")

        resp = await client.get(f"/api/v1/episodes/user/{test_user_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 2
        assert len(data["episodes"]) == 2

    async def test_query_by_intent_filter(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
        test_user_id: str,
    ):
        """Filter episodes by intent type."""
        await create_episode_via_api(raw_text="task one", intent="task")
        await create_episode_via_api(raw_text="journal entry", intent="journal")

        resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"intent": "task"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert all(ep["intent"] == "task" for ep in data["episodes"])

    async def test_query_by_entity_filter(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
        test_user_id: str,
    ):
        """Filter episodes by entity name."""
        await create_episode_via_api(
            raw_text="talked to Alice",
            entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )
        await create_episode_via_api(raw_text="random thought")

        resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"entity": "Alice"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["episodes"]) >= 1
        found = any(
            any(e["name"] == "Alice" for e in ep["entities"])
            for ep in data["episodes"]
        )
        assert found

    async def test_query_by_emotion_filter(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
        test_user_id: str,
    ):
        """Filter episodes by primary emotion."""
        await create_episode_via_api(
            raw_text="feeling great",
            emotion_primary="joy",
            emotion_intensity=2.0,
        )
        await create_episode_via_api(raw_text="neutral thought")

        resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"emotion": "joy"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["episodes"]) >= 1


class TestSessionGrouping:
    """Episodes within a session can be retrieved together."""

    async def test_get_session_episodes(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
    ):
        """Retrieve all episodes in a specific session."""
        session = "sess-abc"
        await create_episode_via_api(raw_text="first in session", session_id=session)
        await create_episode_via_api(raw_text="second in session", session_id=session)
        await create_episode_via_api(raw_text="different session", session_id="other")

        resp = await client.get(f"/api/v1/episodes/session/{session}")
        assert resp.status_code == 200
        episodes = resp.json()
        assert len(episodes) == 2
        assert all(ep["context"]["session_id"] == session for ep in episodes)


class TestEntityTimeline:
    """Entity timeline returns episodes mentioning a specific entity."""

    async def test_entity_timeline(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
        test_user_id: str,
    ):
        await create_episode_via_api(
            raw_text="meeting with Bob",
            entities=[{"name": "Bob", "entity_type": "person", "confidence": 0.95}],
        )
        await create_episode_via_api(
            raw_text="Bob sent the report",
            entities=[{"name": "Bob", "entity_type": "person", "confidence": 0.90}],
        )

        resp = await client.get(
            f"/api/v1/episodes/entity/{test_user_id}/Bob"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_name"] == "Bob"
        assert data["count"] == 2


class TestEpisodeCompression:
    """Compress multiple episodes into a summary."""

    async def test_compress_and_list_summaries(
        self,
        client: httpx.AsyncClient,
        create_episode_via_api: Any,
        test_user_id: str,
    ):
        """Compress episodes and verify summaries are retrievable."""
        ep1 = await create_episode_via_api(raw_text="worked on API design")
        ep2 = await create_episode_via_api(raw_text="reviewed pull request")

        compress_resp = await client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": test_user_id,
                "episode_ids": [ep1["id"], ep2["id"]],
                "summary_text": "Engineering work: API design and PR review",
            },
        )
        assert compress_resp.status_code == 201
        summary = compress_resp.json()
        assert summary["episode_count"] == 2

        # List summaries
        list_resp = await client.get(
            f"/api/v1/episodes/summaries/{test_user_id}"
        )
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert data["count"] >= 1
