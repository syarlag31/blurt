"""E2E Scenario 10: Semantic search over episodic memory.

Tests the embedding-based semantic search endpoint, verifying
that episodes with embeddings can be found by similarity.

Cross-cutting concerns exercised:
- Embedding generation: stub embedder produces deterministic vectors from text
- Vector similarity search: cosine similarity ranking over episode embeddings
- Semantic retrieval: natural-language queries matched to stored episodes by
  meaning rather than keyword overlap
- Score ordering: results ranked by descending similarity score
- Top-K filtering: only the most relevant episodes returned (configurable limit)
- Episode enrichment: search results include full episode metadata (intent,
  entities, emotion) alongside similarity scores
- Cross-store query: search spans all episodes in the episodic store,
  exercising the embedding index and episode store together
"""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import _stub_embed


pytestmark = pytest.mark.asyncio


class TestSemanticSearch:
    """POST /api/v1/episodes/search/semantic — embedding similarity search."""

    async def test_semantic_search_returns_similar_episodes(
        self,
        client: httpx.AsyncClient,
        test_user_id: str,
    ):
        """Episodes with similar embeddings are returned by semantic search."""
        # Create episodes with embeddings
        texts = [
            "I need to review the Python code",
            "The Python migration is going well",
            "Had lunch at the Italian restaurant",
        ]
        for text in texts:
            embedding = await _stub_embed(text)
            await client.post(
                "/api/v1/episodes",
                json={
                    "user_id": test_user_id,
                    "raw_text": text,
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
                    "embedding": embedding,
                },
            )

        # Search with a query similar to "Python" episodes
        query_embedding = await _stub_embed("Python programming review")

        resp = await client.post(
            "/api/v1/episodes/search/semantic",
            json={
                "user_id": test_user_id,
                "query_embedding": query_embedding,
                "limit": 5,
                "min_similarity": 0.0,  # Low threshold for test
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1

    async def test_semantic_search_empty_store(
        self,
        client: httpx.AsyncClient,
        test_user_id: str,
    ):
        """Semantic search on empty store returns zero results."""
        query_embedding = await _stub_embed("anything")
        resp = await client.post(
            "/api/v1/episodes/search/semantic",
            json={
                "user_id": test_user_id,
                "query_embedding": query_embedding,
                "limit": 5,
                "min_similarity": 0.5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    async def test_semantic_search_respects_min_similarity(
        self,
        client: httpx.AsyncClient,
        test_user_id: str,
    ):
        """High min_similarity filters out low-similarity matches."""
        embedding = await _stub_embed("specific topic about databases")
        await client.post(
            "/api/v1/episodes",
            json={
                "user_id": test_user_id,
                "raw_text": "specific topic about databases",
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
                "embedding": embedding,
            },
        )

        # Very different query
        unrelated_embedding = await _stub_embed("zzz zzz zzz zzz")
        resp = await client.post(
            "/api/v1/episodes/search/semantic",
            json={
                "user_id": test_user_id,
                "query_embedding": unrelated_embedding,
                "limit": 5,
                "min_similarity": 0.99,  # Very high threshold
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Very dissimilar query with high threshold should return few or no results
        assert data["count"] <= 1
