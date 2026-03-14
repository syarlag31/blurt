"""Real E2E tests for Gemini embeddings.

Validates that:
- Real Gemini embedding API returns 768-dimension vectors.
- Semantically similar texts produce high cosine similarity.
- Dissimilar texts produce lower cosine similarity.
- Batch embedding works correctly.

All calls hit the **real** Gemini API — no mocks.
"""

from __future__ import annotations

import pytest

from blurt.clients.embeddings import GeminiEmbeddingProvider, cosine_similarity
from blurt.services.embedding import EmbeddingService

pytestmark = pytest.mark.real_e2e

EXPECTED_DIMENSION = 768


# ---------------------------------------------------------------------------
# Dimension & shape tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_embedding_returns_768_dimensions(
    embedding_provider: GeminiEmbeddingProvider,
) -> None:
    """A single text should produce a 768-dimension float vector."""
    vec = await embedding_provider.embed("hello world")
    assert len(vec) == EXPECTED_DIMENSION
    # Every element should be a float
    assert all(isinstance(v, (int, float)) for v in vec)


@pytest.mark.asyncio
async def test_batch_embeddings_each_768_dimensions(
    embedding_provider: GeminiEmbeddingProvider,
) -> None:
    """Batching multiple texts should return one 768-d vector per input."""
    texts = [
        "The cat sat on the mat.",
        "Machine learning is fascinating.",
        "I need to buy groceries.",
    ]
    vectors = await embedding_provider.embed_batch(texts)
    assert len(vectors) == len(texts)
    for vec in vectors:
        assert len(vec) == EXPECTED_DIMENSION


# ---------------------------------------------------------------------------
# Cosine similarity — similar vs. dissimilar texts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_similar_texts_high_cosine_similarity(
    embedding_provider: GeminiEmbeddingProvider,
) -> None:
    """Two semantically similar sentences should have cosine similarity > 0.75."""
    vec_a = await embedding_provider.embed(
        "I have a dentist appointment on Friday at 3pm."
    )
    vec_b = await embedding_provider.embed(
        "My dental visit is scheduled for Friday afternoon."
    )
    sim = cosine_similarity(vec_a, vec_b)
    assert sim > 0.75, f"Expected high similarity for near-synonyms, got {sim:.4f}"


@pytest.mark.asyncio
async def test_dissimilar_texts_lower_cosine_similarity(
    embedding_provider: GeminiEmbeddingProvider,
) -> None:
    """Semantically unrelated texts should have notably lower similarity than related ones."""
    texts = [
        "I have a dentist appointment on Friday at 3pm.",
        "The quantum physics lecture covered wave-particle duality.",
        "My dental visit is scheduled for Friday afternoon.",
    ]
    vectors = await embedding_provider.embed_batch(texts)

    sim_similar = cosine_similarity(vectors[0], vectors[2])
    sim_different = cosine_similarity(vectors[0], vectors[1])

    # The related pair must score meaningfully higher
    assert sim_similar > sim_different, (
        f"Similar pair ({sim_similar:.4f}) should beat dissimilar pair ({sim_different:.4f})"
    )
    # And the gap should be non-trivial
    assert sim_similar - sim_different > 0.05, (
        f"Gap too small: {sim_similar - sim_different:.4f}"
    )


# ---------------------------------------------------------------------------
# EmbeddingService higher-level API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_service_embed_text(
    embedding_service: EmbeddingService,
) -> None:
    """EmbeddingService.embed_text should return an EmbeddingResult with 768-d vector."""
    result = await embedding_service.embed_text("remind me to call Mom tonight")
    assert result is not None
    assert len(result.values) == EXPECTED_DIMENSION
    assert result.dimension == EXPECTED_DIMENSION


@pytest.mark.asyncio
async def test_embedding_service_compute_similarity(
    embedding_service: EmbeddingService,
) -> None:
    """EmbeddingService.compute_similarity should agree with raw cosine for similar texts."""
    score = await embedding_service.compute_similarity(
        "Pick up groceries after work",
        "Buy food items on the way home from the office",
    )
    assert isinstance(score, float)
    assert score > 0.60, f"Expected meaningful similarity, got {score:.4f}"
