"""Blurt API client wrappers."""

from blurt.clients.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingResult,
    GeminiEmbeddingProvider,
    LocalEmbeddingProvider,
    MockEmbeddingProvider,
    cosine_similarity,
)

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingResult",
    "GeminiEmbeddingProvider",
    "LocalEmbeddingProvider",
    "MockEmbeddingProvider",
    "cosine_similarity",
]
