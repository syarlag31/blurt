"""E2E Scenario 9: Disambiguation chain — low-confidence fallback, correction
flow, entity aliasing, and future auto-resolution.

Exercises what happens when the pipeline encounters ambiguous input:
1. Low-confidence classification triggers fallback to a safe default (journal)
2. User corrects the misclassification via a follow-up blurt
3. Entity aliasing — the same real-world entity appears under different names
   and the system tracks both mentions
4. After corrections accumulate, future blurts with similar patterns resolve
   to the correct intent with higher confidence

All tests flow through the full HTTP stack: POST /api/v1/blurt and
GET /api/v1/episodes endpoints with the real FastAPI app, in-memory stores,
and stub classifiers.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from blurt.memory.episodic import InMemoryEpisodicStore


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _capture(
    capture_blurt_via_api: Any,
    text: str,
    *,
    modality: str = "voice",
    session_id: str = "disambig-session",
    time_of_day: str = "morning",
    day_of_week: str = "monday",
) -> dict[str, Any]:
    """Shortcut to capture a blurt with session context."""
    return await capture_blurt_via_api(
        text,
        modality=modality,
        session_id=session_id,
        time_of_day=time_of_day,
        day_of_week=day_of_week,
    )


# ===========================================================================
# Test group 1: Low-confidence fallback
# ===========================================================================


class TestLowConfidenceFallback:
    """When the classifier is uncertain, the pipeline should still capture
    everything (zero-drop) and fall back to the safest intent."""

    async def test_ambiguous_text_is_still_captured(
        self,
        capture_blurt_via_api: Any,
    ):
        """Ambiguous text that matches no strong keyword pattern is captured
        as journal with a reasonable confidence — never dropped."""
        result = await _capture(capture_blurt_via_api, "yeah so that thing")
        assert result["captured"] is True
        assert result["intent"] == "journal"  # fallback intent
        assert result["episode"]["raw_text"] == "yeah so that thing"

    async def test_fallback_intent_has_stable_confidence(
        self,
        capture_blurt_via_api: Any,
    ):
        """Fallback classification should produce a confidence value rather
        than 0 or 1 — indicating the classifier ran but was uncertain."""
        result = await _capture(capture_blurt_via_api, "hmm not sure about that")
        assert 0.0 < result["intent_confidence"] <= 1.0
        # The stub classifier returns 0.85 for journal fallback
        assert result["intent_confidence"] >= 0.5

    async def test_fallback_episode_stored_with_metadata(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """A fallback episode is stored with full metadata (emotion, context)
        even though classification was ambiguous."""
        result = await _capture(capture_blurt_via_api, "oh sure thing ok")
        episode_id = result["episode"]["id"]

        # Verify stored episode via GET
        resp = await client.get(f"/api/v1/episodes/{episode_id}")
        assert resp.status_code == 200
        episode = resp.json()
        assert episode["intent"] == "journal"
        assert episode["emotion"]["primary"] is not None
        assert episode["context"]["session_id"] == "disambig-session"

    async def test_multiple_ambiguous_blurts_all_captured(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Multiple ambiguous blurts in a row should all be captured — no
        deduplication or rate limiting on uncertain inputs."""
        texts = [
            "hmm",
            "ok then",
            "right right",
            "mmhm sure",
        ]
        episode_ids = []
        for text in texts:
            result = await _capture(capture_blurt_via_api, text)
            assert result["captured"] is True
            episode_ids.append(result["episode"]["id"])

        # All episodes should be retrievable
        for eid in episode_ids:
            resp = await client.get(f"/api/v1/episodes/{eid}")
            assert resp.status_code == 200

        # User timeline should contain all episodes
        resp = await client.get(f"/api/v1/episodes/user/{test_user_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] >= len(texts)


# ===========================================================================
# Test group 2: Correction flow
# ===========================================================================


class TestCorrectionFlow:
    """User corrects a misclassified blurt by providing clarification via
    a follow-up blurt. The system should capture the correction as a new
    episode that can be linked back via the session."""

    async def test_initial_misclassification_then_correction(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """An ambiguous blurt followed by a correction creates two linked
        episodes in the same session."""
        session = "correction-session-1"

        # Step 1: Ambiguous input classified as journal (fallback)
        r1 = await capture_blurt_via_api(
            "that dentist thing",
            session_id=session,
        )
        assert r1["captured"] is True
        # "dentist thing" has no strong keyword match -> journal
        first_intent = r1["intent"]

        # Step 2: User corrects with explicit task language
        r2 = await capture_blurt_via_api(
            "I need to call the dentist",
            session_id=session,
        )
        assert r2["captured"] is True
        assert r2["intent"] == "task"

        # Both episodes are in the same session
        resp = await client.get(f"/api/v1/episodes/session/{session}")
        assert resp.status_code == 200
        session_episodes = resp.json()
        assert len(session_episodes) == 2
        # Correction episode should have task intent
        intents = [ep["intent"] for ep in session_episodes]
        assert "task" in intents

    async def test_correction_preserves_original_episode(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """The original misclassified episode is preserved (append-only) —
        corrections do not overwrite the original."""
        session = "correction-session-2"

        r1 = await capture_blurt_via_api(
            "something about the meeting I think",
            session_id=session,
        )
        original_id = r1["episode"]["id"]
        original_text = r1["episode"]["raw_text"]

        # Correction
        await capture_blurt_via_api(
            "I have a meeting with Bob at 3pm",
            session_id=session,
        )

        # Original episode unchanged
        resp = await client.get(f"/api/v1/episodes/{original_id}")
        assert resp.status_code == 200
        original = resp.json()
        assert original["raw_text"] == original_text
        assert original["id"] == original_id

    async def test_correction_chain_three_steps(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """A three-step correction chain: ambiguous → partial → fully
        specified, all captured in the same session."""
        session = "correction-session-3"

        # Step 1: Very vague
        r1 = await capture_blurt_via_api("that thing", session_id=session)
        assert r1["intent"] == "journal"

        # Step 2: Partial clarification (still ambiguous for the stub)
        r2 = await capture_blurt_via_api(
            "the Alice thing", session_id=session
        )
        # Contains entity "alice" but no task/event keyword -> journal
        assert r2["captured"] is True

        # Step 3: Fully specified
        r3 = await capture_blurt_via_api(
            "I need to send Alice the report", session_id=session
        )
        assert r3["intent"] == "task"

        # All three episodes in session
        resp = await client.get(f"/api/v1/episodes/session/{session}")
        assert resp.status_code == 200
        episodes = resp.json()
        assert len(episodes) == 3


# ===========================================================================
# Test group 3: Entity aliasing
# ===========================================================================


class TestEntityAliasing:
    """The same real-world entity can appear under different names or forms
    (e.g., "Bob", "Robert", "Bob from Acme"). The system should extract
    entities consistently and allow querying by any alias."""

    async def test_same_entity_mentioned_with_different_context(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Mentioning 'Bob' in different contexts stores multiple episodes
        that are all retrievable via entity timeline."""
        session = "alias-session-1"

        await capture_blurt_via_api(
            "I need to call Bob about the deadline",
            session_id=session,
        )
        await capture_blurt_via_api(
            "Bob mentioned the meeting is moved to Friday",
            session_id=session,
        )

        # Entity timeline should show both mentions
        resp = await client.get(
            f"/api/v1/episodes/entity/{test_user_id}/Bob"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_name"] == "Bob"
        assert data["count"] >= 2

    async def test_multiple_entities_in_single_blurt(
        self,
        capture_blurt_via_api: Any,
    ):
        """A blurt mentioning multiple entities extracts all of them."""
        result = await _capture(
            capture_blurt_via_api,
            "Alice and Bob need to finish Project Alpha",
        )
        assert result["captured"] is True
        assert result["entities_extracted"] >= 2

        # Check entity names in the episode
        entity_names = [
            e["name"] for e in result["episode"]["entities"]
        ]
        assert "Alice" in entity_names
        assert "Bob" in entity_names

    async def test_entity_appears_across_sessions(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """The same entity mentioned in different sessions should all appear
        in the entity timeline."""
        # Session A
        await capture_blurt_via_api(
            "Alice said the project is on track",
            session_id="alias-sess-a",
        )
        # Session B
        await capture_blurt_via_api(
            "I need to tell Alice about the budget update",
            session_id="alias-sess-b",
        )

        resp = await client.get(
            f"/api/v1/episodes/entity/{test_user_id}/Alice"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 2

        # Episodes span different sessions
        session_ids = {
            ep["context"]["session_id"] for ep in data["episodes"]
        }
        assert len(session_ids) >= 2

    async def test_entity_and_organization_co_occurrence(
        self,
        capture_blurt_via_api: Any,
        client: httpx.AsyncClient,
        test_user_id: str,
    ):
        """When a person entity and organization entity co-occur in the same
        blurt, both should be extracted and independently queryable."""
        result = await _capture(
            capture_blurt_via_api,
            "Bob from Acme sent me the proposal",
        )
        assert result["entities_extracted"] >= 2

        entity_names = [
            e["name"] for e in result["episode"]["entities"]
        ]
        assert "Bob" in entity_names
        assert "Acme Corp" in entity_names

        # Both should be independently queryable via entity timeline
        for entity in ["Bob", "Acme Corp"]:
            resp = await client.get(
                f"/api/v1/episodes/entity/{test_user_id}/{entity}"
            )
            assert resp.status_code == 200
            assert resp.json()["count"] >= 1


# ===========================================================================
# Test group 4: Future auto-resolution (pattern accumulation)
# ===========================================================================


class TestFutureAutoResolution:
    """After multiple blurts with the same intent are captured in a session,
    the system builds enough context that later queries can leverage the
    episode history for resolution. This tests the data foundation for
    auto-resolution rather than an ML model."""

    async def test_repeated_intent_builds_session_context(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Multiple task blurts in a session create a rich session context
        that future resolution can leverage."""
        session = "auto-resolve-1"

        tasks = [
            "I need to buy groceries",
            "I need to call the dentist",
            "I need to finish the report",
        ]
        for t in tasks:
            result = await capture_blurt_via_api(t, session_id=session)
            assert result["intent"] == "task"

        # Session should have all task episodes
        resp = await client.get(f"/api/v1/episodes/session/{session}")
        assert resp.status_code == 200
        episodes = resp.json()
        assert len(episodes) == 3
        assert all(ep["intent"] == "task" for ep in episodes)

    async def test_entity_mention_frequency_for_disambiguation(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """When an entity is mentioned frequently, entity timeline data can
        be used to disambiguate future references to that entity."""
        # Build up a history of mentions for Alice
        mentions = [
            "Alice is working on the frontend",
            "Alice sent the design doc",
            "need to tell Alice about the bug",
            "Alice mentioned the deployment",
        ]
        for text in mentions:
            await capture_blurt_via_api(text, session_id="freq-session")

        # Entity timeline should show all mentions
        resp = await client.get(
            f"/api/v1/episodes/entity/{test_user_id}/Alice"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 4

        # All episodes should reference Alice with consistent entity_type
        for ep in data["episodes"]:
            alice_entities = [
                e for e in ep["entities"] if e["name"] == "Alice"
            ]
            assert len(alice_entities) >= 1
            assert alice_entities[0]["entity_type"] == "person"

    async def test_stats_reflect_disambiguation_chain(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """Pipeline stats should accurately reflect all blurts in the
        disambiguation chain — nothing dropped, intent distribution updated."""
        # Capture a mix of ambiguous and clear blurts
        await _capture(capture_blurt_via_api, "hmm something")  # journal
        await _capture(capture_blurt_via_api, "I need to buy milk")  # task
        await _capture(capture_blurt_via_api, "dinner at seven pm")  # event
        await _capture(capture_blurt_via_api, "yeah ok sure")  # journal

        resp = await client.get("/api/v1/blurt/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_captured"] == 4
        assert stats["drop_rate"] == 0.0
        # Intent distribution should reflect the mix
        dist = stats["intent_distribution"]
        assert "journal" in dist
        assert dist["journal"] >= 2  # at least the ambiguous ones

    async def test_session_thread_enables_correction_lookup(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """The session-based thread of episodes enables future auto-resolution
        by providing a chain of context: ambiguous → correction → confirmed."""
        session = "auto-resolve-chain"

        # Ambiguous blurt
        r1 = await capture_blurt_via_api(
            "that project thing with the budget",
            session_id=session,
        )

        # Correction with entities
        r2 = await capture_blurt_via_api(
            "I need to update Project Alpha budget for Acme",
            session_id=session,
        )

        # Confirmation
        r3 = await capture_blurt_via_api(
            "yeah send the Project Alpha update to Bob at Acme",
            session_id=session,
        )

        # Session thread should contain the full chain
        resp = await client.get(f"/api/v1/episodes/session/{session}")
        assert resp.status_code == 200
        episodes = resp.json()
        assert len(episodes) == 3

        # First episode: ambiguous (no strong entities)
        # Later episodes: progressively more entities
        entity_counts = [len(ep["entities"]) for ep in episodes]
        # The correction and confirmation should have more entities
        assert entity_counts[-1] >= entity_counts[0]

    async def test_intent_filter_separates_resolved_from_unresolved(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """After a disambiguation chain, intent-based filtering correctly
        separates resolved (task/event) episodes from unresolved (journal)."""
        session = "filter-session"

        await capture_blurt_via_api("hmm that thing", session_id=session)  # journal
        await capture_blurt_via_api("the meeting tomorrow", session_id=session)  # event
        await capture_blurt_via_api("I need to fix the bug", session_id=session)  # task

        # Filter by task intent
        resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"intent": "task"},
        )
        assert resp.status_code == 200
        task_episodes = resp.json()["episodes"]
        assert all(ep["intent"] == "task" for ep in task_episodes)
        assert len(task_episodes) >= 1

        # Filter by journal intent
        resp = await client.get(
            f"/api/v1/episodes/user/{test_user_id}",
            params={"intent": "journal"},
        )
        assert resp.status_code == 200
        journal_episodes = resp.json()["episodes"]
        assert all(ep["intent"] == "journal" for ep in journal_episodes)
        assert len(journal_episodes) >= 1

    async def test_semantic_search_finds_related_disambiguation_episodes(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
        episodic_store: InMemoryEpisodicStore,
    ):
        """Semantic search across disambiguation chain episodes should return
        related episodes based on embedding similarity."""
        # Capture related blurts
        r1 = await capture_blurt_via_api(
            "something about the project budget",
            session_id="semantic-disambig",
        )
        r2 = await capture_blurt_via_api(
            "I need to update Project Alpha budget",
            session_id="semantic-disambig",
        )

        # Use the embedding from r2 to search
        ep2 = await episodic_store.get(r2["episode"]["id"])
        assert ep2 is not None

        if ep2.embedding:
            resp = await client.post(
                "/api/v1/episodes/search/semantic",
                json={
                    "user_id": test_user_id,
                    "query_embedding": ep2.embedding,
                    "limit": 10,
                    "min_similarity": 0.5,
                },
            )
            assert resp.status_code == 200
            results = resp.json()
            # Should find at least the original episode
            assert results["count"] >= 1
