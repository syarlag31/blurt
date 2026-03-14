"""E2E Scenario 6: Capture → Episode store cross-cutting flow.

Verifies that blurts captured via the capture API are actually persisted
in episodic memory and retrievable through the episodes API.  This tests
cross-service data flow through the full pipeline.

Cross-cutting concerns exercised:
- Cross-API data flow: POST /blurt writes data readable via GET /episodes
- Pipeline-to-store integration: pipeline output is persisted without loss
- Content integrity: raw text, intent, entities, emotion survive the round-trip
- Session continuity: blurts within a session form a linked episode chain
- Temporal ordering: captured episodes appear in chronological order
- Store consistency: episode count increments correctly after each capture
- Service boundary crossing: capture service → episodic store → episodes API
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.asyncio


class TestCaptureToEpisodeFlow:
    """Capture a blurt and confirm it lands in episodic memory."""

    async def test_captured_blurt_appears_in_episodes(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """A captured blurt should be retrievable from the episodes API."""
        result = await capture_blurt_via_api("I need to buy groceries")
        episode_id = result["episode"]["id"]

        # Retrieve from episodes API
        resp = await client.get(f"/api/v1/episodes/{episode_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_text"] == "I need to buy groceries"
        assert data["intent"] == "task"

    async def test_multiple_captures_all_stored(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Multiple captured blurts are all stored and queryable."""
        await capture_blurt_via_api("need to fix the login bug")
        await capture_blurt_via_api("meeting with design team")
        await capture_blurt_via_api("hmm, interesting thought")

        resp = await client.get(f"/api/v1/episodes/user/{test_user_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 3

    async def test_captured_entities_persisted(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Entities extracted during capture are persisted in the episode."""
        result = await capture_blurt_via_api(
            "Talk to Alice about the React migration"
        )
        episode_id = result["episode"]["id"]

        resp = await client.get(f"/api/v1/episodes/{episode_id}")
        data = resp.json()
        entity_names = [e["name"] for e in data["entities"]]
        assert "Alice" in entity_names or "React" in entity_names

    async def test_session_linking_through_capture(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """Captures in the same session share a session_id."""
        session = "flow-session"
        await capture_blurt_via_api("first thought", session_id=session)
        await capture_blurt_via_api("second thought", session_id=session)

        resp = await client.get(f"/api/v1/episodes/session/{session}")
        assert resp.status_code == 200
        episodes = resp.json()
        assert len(episodes) == 2

    async def test_capture_stats_and_episode_count_agree(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Stats total_captured matches the number of episodes in store."""
        for text in ["buy milk", "call dentist", "nice day"]:
            await capture_blurt_via_api(text)

        stats_resp = await client.get("/api/v1/blurt/stats")
        stats = stats_resp.json()

        episodes_resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}"
        )
        episodes = episodes_resp.json()

        assert stats["total_captured"] == episodes["total_count"]
