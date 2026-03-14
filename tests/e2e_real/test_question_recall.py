"""Real E2E test: QUESTION intent recall after storing several blurts.

Validates that the recall engine can retrieve relevant episodes when
the user asks a question about previously captured blurts. All Gemini
API calls are real — no mocks.

Flow:
1. Capture several diverse blurts via ``BlurtCapturePipeline`` (real Gemini).
2. Ask a QUESTION query via ``PersonalHistoryRecallEngine``.
3. Assert the recall returns relevant results ranked by similarity.

No mocks. Skipped automatically when ``GEMINI_API_KEY`` is not set.
"""

from __future__ import annotations

import pytest

from blurt.clients.embeddings import GeminiEmbeddingProvider
from blurt.memory.episodic import InMemoryEpisodicStore, IntentFilter
from blurt.services.capture import BlurtCapturePipeline, CaptureResult
from blurt.services.recall import PersonalHistoryRecallEngine

pytestmark = pytest.mark.asyncio

TEST_USER = "real-e2e-test-user"
TEST_SESSION = "real-e2e-test-session"

# A set of topically distinct blurts to store before querying.
BLURTS = [
    "Met with Sarah about the Q2 marketing deck and she loved the new design",
    "I need to buy groceries this weekend, we're out of milk and eggs",
    "Had a great workout at the gym this morning, ran five miles",
    "The Q2 marketing deck needs a new section on competitive analysis",
]

# A question that should match the Q2-deck-related blurts.
QUESTION_QUERY = "what did I say about the Q2 deck?"


async def _capture_all(
    pipeline: BlurtCapturePipeline,
    texts: list[str],
) -> list[CaptureResult]:
    """Capture a list of blurts sequentially, returning their results."""
    results: list[CaptureResult] = []
    for text in texts:
        result = await pipeline.capture_text(
            TEST_USER,
            text,
            session_id=TEST_SESSION,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_recall_returns_results_after_storing_blurts(
    capture_pipeline: BlurtCapturePipeline,
    episodic_store: InMemoryEpisodicStore,
    embedding_provider: GeminiEmbeddingProvider,
) -> None:
    """After storing several blurts, a recall query should return results."""
    # Store blurts
    results = await _capture_all(capture_pipeline, BLURTS)
    stored_count = sum(1 for r in results if r.was_stored)
    assert stored_count == len(BLURTS), (
        f"Expected all {len(BLURTS)} blurts stored, only {stored_count} succeeded"
    )

    # Build recall engine backed by the same episodic store
    engine = PersonalHistoryRecallEngine(
        episodic_store=episodic_store,
        embedding_provider=embedding_provider,
    )

    response = await engine.recall(TEST_USER, QUESTION_QUERY)

    assert response.results, (
        f"Recall returned no results for '{QUESTION_QUERY}'. "
        f"Sources searched: {response.sources_searched}"
    )
    assert response.total_results >= 1, (
        f"Expected at least 1 result, got {response.total_results}"
    )


async def test_recall_ranks_relevant_blurt_highest(
    capture_pipeline: BlurtCapturePipeline,
    episodic_store: InMemoryEpisodicStore,
    embedding_provider: GeminiEmbeddingProvider,
) -> None:
    """The most relevant blurt (about Q2 deck) should rank above unrelated ones."""
    results = await _capture_all(capture_pipeline, BLURTS)
    assert all(r.was_stored for r in results), "All blurts should be stored"

    engine = PersonalHistoryRecallEngine(
        episodic_store=episodic_store,
        embedding_provider=embedding_provider,
    )

    response = await engine.recall(TEST_USER, QUESTION_QUERY)
    assert response.results, "Recall returned no results"

    # The top result's content should mention "Q2" or "deck" or "marketing"
    top = response.results[0]
    top_content_lower = top.content.lower()
    assert any(kw in top_content_lower for kw in ("q2", "deck", "marketing")), (
        f"Top recall result should be about the Q2 deck, but got: '{top.content}' "
        f"(score={top.relevance_score:.3f})"
    )


async def test_recall_does_not_return_unrelated_as_top(
    capture_pipeline: BlurtCapturePipeline,
    episodic_store: InMemoryEpisodicStore,
    embedding_provider: GeminiEmbeddingProvider,
) -> None:
    """Unrelated blurts (groceries, gym) should NOT be the top result for Q2 query."""
    await _capture_all(capture_pipeline, BLURTS)

    engine = PersonalHistoryRecallEngine(
        episodic_store=episodic_store,
        embedding_provider=embedding_provider,
    )

    response = await engine.recall(TEST_USER, QUESTION_QUERY)
    assert response.results, "Recall returned no results"

    top_content_lower = response.results[0].content.lower()
    # The top result should NOT be about groceries or gym
    assert "groceries" not in top_content_lower, (
        f"Top result is about groceries, not Q2 deck: '{response.results[0].content}'"
    )
    assert "gym" not in top_content_lower, (
        f"Top result is about gym, not Q2 deck: '{response.results[0].content}'"
    )


async def test_recall_embedding_was_generated(
    capture_pipeline: BlurtCapturePipeline,
    episodic_store: InMemoryEpisodicStore,
    embedding_provider: GeminiEmbeddingProvider,
) -> None:
    """The recall engine should generate a query embedding for semantic search."""
    await _capture_all(capture_pipeline, BLURTS)

    engine = PersonalHistoryRecallEngine(
        episodic_store=episodic_store,
        embedding_provider=embedding_provider,
    )

    response = await engine.recall(TEST_USER, QUESTION_QUERY)
    assert response.query_embedding_generated, (
        "Recall engine did not generate a query embedding — semantic search may not have run"
    )


async def test_episodic_store_query_by_intent_after_capture(
    capture_pipeline: BlurtCapturePipeline,
    episodic_store: InMemoryEpisodicStore,
) -> None:
    """After capturing several blurts, filtering by intent should return matching episodes."""
    results = await _capture_all(capture_pipeline, BLURTS)
    assert all(r.was_stored for r in results)

    # Collect all intents that were assigned
    intents_assigned = {r.episode.intent for r in results if r.episode.intent}
    assert intents_assigned, "No intents were assigned to any episode"

    # Query for each intent — at least one should return results
    total_found = 0
    for intent_val in intents_assigned:
        episodes = await episodic_store.query(
            TEST_USER,
            intent_filter=IntentFilter(intent=intent_val),
        )
        total_found += len(episodes)

    assert total_found >= len(BLURTS), (
        f"Expected at least {len(BLURTS)} total episodes across intent filters, "
        f"got {total_found}. Intents: {intents_assigned}"
    )
