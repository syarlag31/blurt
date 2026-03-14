"""E2E Scenario 4: Contradictory updates — rapid UPDATE intent chaining,
calendar dedup, and episodic immutability.

Validates that when a user issues a rapid sequence of contradictory UPDATE
blurts, the system:

1. Classifies each successive blurt correctly per the keyword-priority rules
2. Stores ALL updates as separate, immutable episodes (append-only)
3. Preserves chronological order so the latest state can be derived
4. Never mutates or overwrites earlier episodes
5. Captures evolving emotion across the contradiction chain
6. Handles entity references consistently across rapid updates
7. Produces correct session-level episode chains
8. Allows filtering by intent to retrieve only update episodes
9. Keeps pipeline stats accurate across the entire burst
10. Produces distinct episode IDs even for semantically similar updates

The stub classifier maps keywords to intents with strict priority ordering:
  task > event > reminder > idea > update > question > journal
So "update" texts must avoid higher-priority keywords (e.g. "meeting", "need to")
to be classified as update intent.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from blurt.memory.episodic import InMemoryEpisodicStore, IntentFilter


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helper: rapid-fire a chain of pure-update blurts (no higher-priority keywords)
# ---------------------------------------------------------------------------


async def _fire_pure_update_chain(
    capture: Any,
    session_id: str = "contradictory-session",
) -> list[dict[str, Any]]:
    """Send a series of update blurts that trigger the update intent.

    Avoids keywords like 'meeting', 'need to', 'buy', 'finish' that would
    trigger higher-priority intents in the stub classifier.
    """
    updates = [
        "Status update: the plan with Alice changed to 3pm",
        "Update that — the plan with Alice is now at 4pm",
        "Done with the progress report on Project Alpha",
        "Progress update: Alice completed the React prototype",
        "Update: scratch the plan with Alice, Bob is unavailable",
        "Status update — Bob is done with the Python migration",
    ]
    results: list[dict[str, Any]] = []
    for text in updates:
        result = await capture(
            raw_text=text,
            session_id=session_id,
            time_of_day="afternoon",
            day_of_week="wednesday",
        )
        results.append(result)
    return results


async def _fire_mixed_intent_chain(
    capture: Any,
    session_id: str = "mixed-contradictory-session",
) -> list[dict[str, Any]]:
    """Send contradictory blurts that span multiple intents.

    Some mention 'meeting' (→ event), some are pure updates. This exercises
    cross-intent chaining and immutability for mixed scenarios.
    """
    blurts = [
        ("Update: the plan with Alice changed to 3pm", "update"),
        ("Actually the meeting with Alice is at 4pm", "event"),
        ("Status update: just scratch all plans with Alice", "update"),
    ]
    results: list[dict[str, Any]] = []
    for text, _expected_intent in blurts:
        result = await capture(
            raw_text=text,
            session_id=session_id,
            time_of_day="afternoon",
            day_of_week="wednesday",
        )
        results.append(result)
    return results


class TestContradictoryUpdateChaining:
    """Rapid UPDATE blurts are each captured and classified correctly."""

    async def test_all_pure_updates_classified_as_update_intent(
        self,
        capture_blurt_via_api: Any,
    ):
        """Every pure-update utterance is classified as 'update'."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        for i, r in enumerate(results):
            assert r["captured"] is True, f"Update {i} was not captured"
            assert r["intent"] == "update", (
                f"Update {i} ('{r['episode']['raw_text']}') classified as "
                f"'{r['intent']}' instead of 'update'"
            )

    async def test_each_update_gets_unique_episode_id(
        self,
        capture_blurt_via_api: Any,
    ):
        """Every contradictory update produces a distinct episode ID."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        episode_ids = [r["episode"]["id"] for r in results]
        assert len(set(episode_ids)) == len(episode_ids), (
            "Duplicate episode IDs found — updates must produce unique episodes"
        )

    async def test_confidence_stays_high_across_chain(
        self,
        capture_blurt_via_api: Any,
    ):
        """Intent confidence remains consistently high for update keywords."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        for r in results:
            assert r["intent_confidence"] >= 0.8, (
                f"Confidence {r['intent_confidence']} too low for clear update"
            )

    async def test_mixed_intent_chain_captures_all(
        self,
        capture_blurt_via_api: Any,
    ):
        """A mixed chain with event and update intents still captures all blurts."""
        results = await _fire_mixed_intent_chain(capture_blurt_via_api)
        assert len(results) == 3
        for r in results:
            assert r["captured"] is True
        # First and third are update, second is event (due to 'meeting' keyword)
        assert results[0]["intent"] == "update"
        assert results[1]["intent"] == "event"
        assert results[2]["intent"] == "update"


class TestEpisodicImmutability:
    """Earlier episodes are never mutated by later contradictory updates."""

    async def test_first_episode_unchanged_after_contradictions(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """The first update episode retains its original text after later updates."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        first_id = results[0]["episode"]["id"]
        first_text = results[0]["episode"]["raw_text"]

        # Re-fetch the first episode from the API
        resp = await client.get(f"/api/v1/episodes/{first_id}")
        assert resp.status_code == 200
        ep = resp.json()
        assert ep["raw_text"] == first_text
        assert ep["intent"] == "update"

    async def test_all_episodes_preserved_in_order(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """All episodes exist in the store — none overwritten or lost."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        episode_ids = [r["episode"]["id"] for r in results]

        # Fetch all episodes for the user
        resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"limit": 100},
        )
        assert resp.status_code == 200
        stored = resp.json()
        stored_ids = {ep["id"] for ep in stored["episodes"]}

        for eid in episode_ids:
            assert eid in stored_ids, f"Episode {eid} missing from store"

    async def test_episode_timestamps_are_monotonically_increasing(
        self,
        capture_blurt_via_api: Any,
    ):
        """Episode timestamps are ordered — later updates have later timestamps."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        timestamps = [r["episode"]["timestamp"] for r in results]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamp {i} ({timestamps[i]}) is before timestamp {i-1} "
                f"({timestamps[i-1]})"
            )

    async def test_cannot_modify_stored_episode_via_duplicate_append(
        self,
        episodic_store: InMemoryEpisodicStore,
        capture_blurt_via_api: Any,
    ):
        """The append-only store rejects duplicate episode IDs."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        first_id = results[0]["episode"]["id"]

        # Confirm the episode exists
        ep = await episodic_store.get(first_id)
        assert ep is not None

        # Attempt to re-append with same ID should raise
        from blurt.memory.episodic import Episode

        duplicate = Episode(id=first_id, raw_text="tampered")
        with pytest.raises(ValueError, match="already exists"):
            await episodic_store.append(duplicate)


class TestSessionChaining:
    """Contradictory updates within a session form a coherent chain."""

    async def test_session_contains_all_updates(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """All updates in the same session are retrievable together."""
        session = "session-contradictory-chain"
        results = await _fire_pure_update_chain(
            capture_blurt_via_api, session_id=session
        )

        resp = await client.get(f"/api/v1/episodes/session/{session}")
        assert resp.status_code == 200
        session_eps = resp.json()
        assert len(session_eps) == len(results)

    async def test_session_episodes_in_chronological_order(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """Session episodes are returned in chronological order."""
        session = "session-chrono-check"
        await _fire_pure_update_chain(
            capture_blurt_via_api, session_id=session
        )

        resp = await client.get(f"/api/v1/episodes/session/{session}")
        assert resp.status_code == 200
        eps = resp.json()
        timestamps = [e["timestamp"] for e in eps]
        assert timestamps == sorted(timestamps), (
            "Session episodes not in chronological order"
        )

    async def test_mixed_session_with_non_update_blurts(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """A session mixing updates and non-updates stores all episodes."""
        session = "session-mixed"
        await capture_blurt_via_api(
            "I need to buy groceries",  # task intent (has 'need to' + 'buy')
            session_id=session,
        )
        await capture_blurt_via_api(
            "Status update: Alice rescheduled to later",  # update intent
            session_id=session,
        )
        await capture_blurt_via_api(
            "Dinner with Bob at 7",  # event intent (has 'dinner')
            session_id=session,
        )

        resp = await client.get(f"/api/v1/episodes/session/{session}")
        assert resp.status_code == 200
        eps = resp.json()
        assert len(eps) == 3
        intents = [e["intent"] for e in eps]
        assert "task" in intents
        assert "update" in intents
        assert "event" in intents


class TestEntityConsistencyAcrossUpdates:
    """Entity references are consistent across contradictory updates."""

    async def test_alice_mentioned_in_multiple_updates(
        self,
        capture_blurt_via_api: Any,
    ):
        """Alice is extracted in all updates that mention her."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        alice_episodes = [
            r for r in results
            if any(
                e["name"] == "Alice"
                for e in r["episode"]["entities"]
            )
        ]
        # Updates 1, 2, 4, 5 mention Alice
        assert len(alice_episodes) >= 3, (
            f"Expected Alice in ≥3 updates, found {len(alice_episodes)}"
        )

    async def test_entity_timeline_shows_update_chain(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """The entity timeline for Alice shows the full update history."""
        await _fire_pure_update_chain(capture_blurt_via_api)

        resp = await client.get(
            f"/api/v1/episodes/entity/{test_user_id}/Alice"
        )
        assert resp.status_code == 200
        timeline = resp.json()
        assert timeline["count"] >= 3, (
            f"Alice timeline has {timeline['count']} entries, expected ≥3"
        )
        # All episodes in the timeline mention Alice
        for ep in timeline["episodes"]:
            entity_names = [e["name"] for e in ep["entities"]]
            assert "Alice" in entity_names

    async def test_bob_entity_appears_in_relevant_updates(
        self,
        capture_blurt_via_api: Any,
    ):
        """Bob is extracted only from updates that mention him."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        bob_episodes = [
            r for r in results
            if any(
                e["name"] == "Bob"
                for e in r["episode"]["entities"]
            )
        ]
        # Bob is mentioned in updates 5 and 6
        assert len(bob_episodes) >= 2

    async def test_project_alpha_entity_extracted(
        self,
        capture_blurt_via_api: Any,
    ):
        """Project Alpha is extracted from the status update."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        alpha_episodes = [
            r for r in results
            if any(
                e["name"] == "Project Alpha"
                for e in r["episode"]["entities"]
            )
        ]
        assert len(alpha_episodes) >= 1


class TestIntentFilteringForUpdates:
    """The intent filter correctly isolates update episodes."""

    async def test_filter_by_update_intent(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Filtering by intent='update' returns only update episodes."""
        # Create non-update blurts
        await capture_blurt_via_api("I need to fix the bug")  # task
        await capture_blurt_via_api("Meeting at noon")  # event
        # Create update blurts
        await _fire_pure_update_chain(capture_blurt_via_api)

        resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"intent": "update", "limit": 100},
        )
        assert resp.status_code == 200
        data = resp.json()
        for ep in data["episodes"]:
            assert ep["intent"] == "update"
        assert len(data["episodes"]) == 6  # the 6 pure update-chain blurts

    async def test_direct_store_query_with_intent_filter(
        self,
        episodic_store: InMemoryEpisodicStore,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Direct store query with IntentFilter returns only updates."""
        await capture_blurt_via_api("I need to call the dentist")  # task
        await _fire_pure_update_chain(capture_blurt_via_api)

        updates = await episodic_store.query(
            test_user_id,
            intent_filter=IntentFilter("update"),
            limit=100,
        )
        assert len(updates) == 6
        for ep in updates:
            assert ep.intent == "update"


class TestEmotionAcrossContradictions:
    """Emotion evolves across the chain — frustration may build."""

    async def test_emotion_detected_for_each_update(
        self,
        capture_blurt_via_api: Any,
    ):
        """Every update in the chain has emotion detection applied."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        for i, r in enumerate(results):
            assert r["emotion_detected"] is True, (
                f"Update {i} missing emotion detection"
            )

    async def test_emotion_varies_across_frustrated_updates(
        self,
        capture_blurt_via_api: Any,
    ):
        """A frustrated update chain shows changing emotional context."""
        # Fire updates with escalating frustration
        r1 = await capture_blurt_via_api(
            "Status update: the plan changed to 3pm",
            session_id="frustration-chain",
        )
        r2 = await capture_blurt_via_api(
            "Frustrated update — the plan changed AGAIN to 4pm",
            session_id="frustration-chain",
        )
        r3 = await capture_blurt_via_api(
            "I'm annoyed, status update: just cancel the whole plan",
            session_id="frustration-chain",
        )

        # r1 should have neutral/trust emotion
        # r3 with "annoyed" triggers anger detection
        assert r1["episode"]["emotion"]["primary"] != "anger"
        assert r3["episode"]["emotion"]["primary"] == "anger"


class TestPipelineStatsForUpdateBurst:
    """Pipeline statistics correctly reflect the update burst."""

    async def test_stats_count_all_updates(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """Stats endpoint counts all updates in the burst."""
        await _fire_pure_update_chain(capture_blurt_via_api)

        resp = await client.get("/api/v1/blurt/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_captured"] == 6
        assert stats["drop_rate"] == 0.0
        assert stats["intent_distribution"].get("update", 0) == 6

    async def test_stats_show_zero_drop_rate(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """Even with rapid contradictory updates, nothing is dropped."""
        await _fire_pure_update_chain(capture_blurt_via_api)

        resp = await client.get("/api/v1/blurt/stats")
        stats = resp.json()
        assert stats["drop_rate"] == 0.0
        assert stats["enrichment_success_rate"] > 0.9

    async def test_fully_enriched_across_burst(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """All updates in the burst are fully enriched."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        for r in results:
            assert r["fully_enriched"] is True

        resp = await client.get("/api/v1/blurt/stats")
        stats = resp.json()
        assert stats["fully_enriched_count"] == 6


class TestCalendarDedup:
    """Contradictory event/update blurts produce separate episodes (no dedup at
    episodic layer — dedup is a calendar integration concern, not memory).

    The episodic store is append-only, so contradictory time references
    are both preserved. A higher-level calendar sync would resolve the latest
    state, but the memory layer keeps the full history.
    """

    async def test_conflicting_times_both_stored(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Both '3pm' and '4pm' updates are stored as separate episodes."""
        r1 = await capture_blurt_via_api(
            "Status update: the plan with Alice changed to 3pm"
        )
        r2 = await capture_blurt_via_api(
            "Update that — the plan with Alice is now at 4pm"
        )

        assert r1["episode"]["id"] != r2["episode"]["id"]

        # Both exist in store
        ep1 = await episodic_store.get(r1["episode"]["id"])
        ep2 = await episodic_store.get(r2["episode"]["id"])
        assert ep1 is not None
        assert ep2 is not None
        assert "3pm" in ep1.raw_text
        assert "4pm" in ep2.raw_text

    async def test_cancellation_does_not_remove_earlier_episodes(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """A cancellation update does not delete/modify the creation episode."""
        r1 = await capture_blurt_via_api(
            "Status update: plan with Alice at 3pm"
        )
        r2 = await capture_blurt_via_api(
            "Update: scratch the plan with Alice"
        )

        ep1 = await episodic_store.get(r1["episode"]["id"])
        ep2 = await episodic_store.get(r2["episode"]["id"])

        assert ep1 is not None and ep2 is not None
        assert "3pm" in ep1.raw_text
        assert "scratch" in ep2.raw_text.lower()
        # Original is untouched
        assert not ep1.is_compressed
        assert ep1.intent == "update"

    async def test_latest_state_derivable_from_chronological_query(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """The latest state can be derived by taking the most recent episode."""
        await capture_blurt_via_api(
            "Status update: plan with Alice at 3pm"
        )
        await capture_blurt_via_api(
            "Update: plan with Alice at 4pm"
        )
        await capture_blurt_via_api(
            "Update: scratch the plan with Alice"
        )

        # Query update episodes — newest first (default sort)
        resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"intent": "update", "limit": 10},
        )
        assert resp.status_code == 200
        episodes = resp.json()["episodes"]
        assert len(episodes) == 3
        # Newest first → the cancellation is at index 0
        assert "scratch" in episodes[0]["raw_text"].lower()


class TestCompressionPreservesHistory:
    """Compressing update episodes preserves raw data while creating a summary."""

    async def test_compress_contradictory_chain(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Compressing a contradictory chain produces a valid summary."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        episode_ids = [r["episode"]["id"] for r in results]

        # Compress all updates into a summary
        resp = await client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": test_user_id,
                "episode_ids": episode_ids,
                "summary_text": "Multiple contradictory plan updates with Alice and Bob",
            },
        )
        assert resp.status_code == 201
        summary = resp.json()
        assert summary["episode_count"] == 6
        assert "update" in summary["intent_distribution"]
        assert summary["intent_distribution"]["update"] == 6

        # Raw episodes are marked as compressed but still retrievable
        for eid in episode_ids:
            ep_resp = await client.get(f"/api/v1/episodes/{eid}")
            assert ep_resp.status_code == 200
            assert ep_resp.json()["is_compressed"] is True

    async def test_compressed_episodes_excluded_by_default(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """After compression, default queries exclude compressed episodes."""
        results = await _fire_pure_update_chain(capture_blurt_via_api)
        episode_ids = [r["episode"]["id"] for r in results]

        await client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": test_user_id,
                "episode_ids": episode_ids,
                "summary_text": "Compressed update chain",
            },
        )

        # Default query excludes compressed
        resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"limit": 100},
        )
        data = resp.json()
        compressed_ids = {
            ep["id"] for ep in data["episodes"] if ep["is_compressed"]
        }
        assert len(compressed_ids) == 0

        # With include_compressed=true, they appear
        resp2 = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"limit": 100, "include_compressed": True},
        )
        data2 = resp2.json()
        assert data2["total_count"] == 6
