"""Embedding service for semantic memory and search.

High-level service that wraps embedding providers with:
- Automatic provider selection (cloud vs local)
- API key validation and management
- Batch embedding with optimization
- Embedding cache management
- Similarity search utilities

Usage::

    from blurt.services.embedding import EmbeddingService
    from blurt.config.settings import GeminiConfig

    config = GeminiConfig.from_env()
    service = EmbeddingService.from_config(config)

    async with service:
        vec = await service.embed_text("Remember to call Alice")
        vecs = await service.embed_texts(["text1", "text2"])
        results = await service.find_similar("call someone", stored_items)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from blurt.clients.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    GeminiEmbeddingProvider,
    LocalEmbeddingProvider,
    MockEmbeddingProvider,
    cosine_similarity,
)
from blurt.config.settings import DeploymentMode, GeminiConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SimilarityMatch:
    """A search result from similarity matching.

    Attributes:
        index: Index in the original items list.
        score: Cosine similarity score (-1.0 to 1.0).
        item: The matched item (if items were provided).
    """

    index: int
    score: float
    item: Any = None


class EmbeddingService:
    """High-level embedding service for Blurt's semantic memory.

    Provides a unified interface for generating embeddings and performing
    similarity searches, regardless of whether running in cloud or local mode.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        mode: DeploymentMode = DeploymentMode.CLOUD,
    ) -> None:
        self._provider = provider
        self._mode = mode
        self._is_open = False

    @classmethod
    def from_config(
        cls,
        config: GeminiConfig,
        *,
        mode: DeploymentMode = DeploymentMode.CLOUD,
        cache_size: int = 1024,
    ) -> EmbeddingService:
        """Create an EmbeddingService from a GeminiConfig.

        In cloud mode, creates a GeminiEmbeddingProvider.
        In local mode, creates a LocalEmbeddingProvider (or Mock fallback).

        Args:
            config: Gemini configuration with API key and model settings.
            mode: Deployment mode (cloud or local).
            cache_size: Max cache entries for cloud provider.

        Returns:
            Configured EmbeddingService.
        """
        if mode == DeploymentMode.LOCAL:
            try:
                provider: EmbeddingProvider = LocalEmbeddingProvider()
                logger.info("EmbeddingService: using local provider (sentence-transformers)")
            except ImportError:
                provider = MockEmbeddingProvider()
                logger.info("EmbeddingService: using mock provider (no sentence-transformers)")
        else:
            errors = []
            if not config.api_key:
                errors.append("GEMINI_API_KEY is required for cloud embedding")
            if errors:
                raise ValueError("; ".join(errors))

            provider = GeminiEmbeddingProvider(
                config=config,
                cache_size=cache_size,
            )
            logger.info(
                "EmbeddingService: using Gemini cloud provider (model=%s, dim=%d)",
                config.embedding_model,
                config.embedding_dimensions,
            )

        return cls(provider, mode=mode)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def provider(self) -> EmbeddingProvider:
        """The underlying embedding provider."""
        return self._provider

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self._provider.dimension

    @property
    def mode(self) -> DeploymentMode:
        """Current deployment mode."""
        return self._mode

    @property
    def is_cloud(self) -> bool:
        """Whether using cloud embeddings."""
        return self._mode == DeploymentMode.CLOUD

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def open(self) -> None:
        """Initialize the service (called automatically via context manager)."""
        if isinstance(self._provider, GeminiEmbeddingProvider):
            await self._provider._ensure_http()
        self._is_open = True

    async def close(self) -> None:
        """Shut down the service and release resources."""
        if isinstance(self._provider, GeminiEmbeddingProvider):
            await self._provider.close()
        self._is_open = False

    async def __aenter__(self) -> EmbeddingService:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Core Methods ──────────────────────────────────────────────────

    async def embed_text(
        self,
        text: str,
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> EmbeddingResult:
        """Generate an embedding for a single text.

        Args:
            text: Input text to embed.
            task_type: Embedding task type for the Gemini API.

        Returns:
            EmbeddingResult with the vector and metadata.
        """
        if isinstance(self._provider, GeminiEmbeddingProvider):
            values = await self._provider.embed(text, task_type=task_type)
        else:
            values = await self._provider.embed(text)

        return EmbeddingResult(
            values=values,
            model=self._get_model_name(),
            dimension=len(values),
        )

    async def embed_texts(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Uses batch API when available for efficiency.

        Args:
            texts: List of input texts.
            task_type: Embedding task type.

        Returns:
            List of EmbeddingResult objects.
        """
        if not texts:
            return []

        if isinstance(self._provider, GeminiEmbeddingProvider):
            vectors = await self._provider.embed_batch(texts, task_type=task_type)
        else:
            vectors = await self._provider.embed_batch(texts)

        model = self._get_model_name()
        return [
            EmbeddingResult(
                values=v,
                model=model,
                dimension=len(v),
            )
            for v in vectors
        ]

    async def embed_for_search(self, query: str) -> EmbeddingResult:
        """Generate an embedding optimized for search queries.

        Uses RETRIEVAL_QUERY task type for better search results.
        """
        return await self.embed_text(query, task_type="RETRIEVAL_QUERY")

    async def embed_for_storage(self, text: str) -> EmbeddingResult:
        """Generate an embedding optimized for document storage.

        Uses RETRIEVAL_DOCUMENT task type for indexed content.
        """
        return await self.embed_text(text, task_type="RETRIEVAL_DOCUMENT")

    # ── Similarity Search ─────────────────────────────────────────────

    async def find_similar(
        self,
        query: str,
        candidates: Sequence[tuple[str, list[float]]],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SimilarityMatch]:
        """Find the most similar items to a query.

        Args:
            query: Query text to search for.
            candidates: Sequence of (label, embedding_vector) tuples.
            top_k: Maximum number of results.
            min_score: Minimum similarity score threshold.

        Returns:
            Top-k SimilarityMatch results sorted by descending score.
        """
        query_result = await self.embed_for_search(query)
        query_vec = query_result.values

        scored: list[SimilarityMatch] = []
        for i, (label, candidate_vec) in enumerate(candidates):
            if len(candidate_vec) != len(query_vec):
                continue
            score = cosine_similarity(query_vec, candidate_vec)
            if score >= min_score:
                scored.append(SimilarityMatch(index=i, score=score, item=label))

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    async def compute_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Cosine similarity score (-1.0 to 1.0).
        """
        results = await self.embed_texts(
            [text_a, text_b], task_type="SEMANTIC_SIMILARITY"
        )
        if len(results) < 2:
            return 0.0
        return cosine_similarity(results[0].values, results[1].values)

    # ── Cache Management ──────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Clear the embedding cache (cloud provider only)."""
        if isinstance(self._provider, GeminiEmbeddingProvider):
            self._provider.clear_cache()

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        base = {
            "mode": self._mode.value,
            "dimension": self.dimension,
            "provider": type(self._provider).__name__,
        }
        if isinstance(self._provider, GeminiEmbeddingProvider):
            base["provider_stats"] = self._provider.stats
        return base

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_model_name(self) -> str:
        """Get the model name from the provider."""
        if isinstance(self._provider, GeminiEmbeddingProvider):
            return self._provider.model
        if isinstance(self._provider, LocalEmbeddingProvider):
            return self._provider._model_name
        return type(self._provider).__name__
