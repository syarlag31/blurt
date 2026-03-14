"""Embedding providers for semantic memory.

Uses Gemini 2 embeddings for vector representation of entities, facts,
and patterns. Provides cosine similarity search for semantic retrieval.
Falls back to a local embedding strategy for local-only mode (feature parity).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import httpx

from blurt.config.settings import GeminiConfig

logger = logging.getLogger(__name__)


# ── Data Types ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Result of an embedding operation.

    Attributes:
        values: The embedding vector.
        model: Model that produced the embedding.
        dimension: Dimensionality of the vector.
        latency_ms: Time taken in milliseconds.
        cached: Whether this result came from cache.
        token_count: Estimated input token count (if available).
    """

    values: list[float]
    model: str
    dimension: int
    latency_ms: float = 0.0
    cached: bool = False
    token_count: int = 0


# ── Abstract Base ─────────────────────────────────────────────────────


class EmbeddingProvider(ABC):
    """Abstract embedding provider for pluggable backends."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


# ── Gemini Cloud Provider ─────────────────────────────────────────────


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini 2 embedding provider for cloud mode.

    Uses the text-embedding model from the Gemini API via httpx.
    Supports configurable dimensions, task types, batch processing
    with chunking, and an in-memory LRU cache.
    """

    # Gemini text-embedding-004 supports 768 dimensions by default
    DEFAULT_DIMENSION = 768
    # Max texts per batchEmbedContents call
    BATCH_CHUNK_SIZE = 100

    def __init__(
        self,
        config: GeminiConfig | None = None,
        *,
        api_key: str = "",
        model: str = "",
        dimensions: int = 0,
        cache_size: int = 1024,
    ) -> None:
        """Initialize the Gemini embedding provider.

        Args:
            config: Full GeminiConfig. If provided, other kwargs are ignored.
            api_key: API key (if config not provided).
            model: Embedding model ID (default: text-embedding-004).
            dimensions: Output embedding dimensions (default: 768).
            cache_size: Max entries in the LRU embedding cache (0 to disable).
        """
        if config is not None:
            self._api_key = config.api_key
            self._model = config.embedding_model
            self._dimensions = config.embedding_dimensions
            self._base_url = config.base_url
            self._connect_timeout = config.connect_timeout
            self._read_timeout = config.read_timeout
            self._max_retries = config.max_retries
            self._retry_backoff_base = config.retry_backoff_base
            self._retry_backoff_max = config.retry_backoff_max
        else:
            self._api_key = api_key
            self._model = model or "gemini-embedding-001"
            self._dimensions = dimensions or self.DEFAULT_DIMENSION
            self._base_url = "https://generativelanguage.googleapis.com/v1beta"
            self._connect_timeout = 10.0
            self._read_timeout = 60.0
            self._max_retries = 3
            self._retry_backoff_base = 0.5
            self._retry_backoff_max = 30.0

        self._http: httpx.AsyncClient | None = None
        self._owns_http = False

        # LRU cache for embeddings
        self._cache_size = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

        # Stats
        self._request_count = 0
        self._cache_hits = 0
        self._total_latency_ms = 0.0

    @property
    def dimension(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        """The embedding model ID."""
        return self._model

    @property
    def api_key(self) -> str:
        """The configured API key (masked for display)."""
        return self._api_key

    @property
    def api_key_masked(self) -> str:
        """Masked API key for safe logging."""
        if not self._api_key:
            return "<not set>"
        if len(self._api_key) <= 8:
            return "****"
        return f"{self._api_key[:4]}...{self._api_key[-4:]}"

    @property
    def stats(self) -> dict[str, Any]:
        """Provider statistics."""
        return {
            "model": self._model,
            "dimension": self._dimensions,
            "request_count": self._request_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_max": self._cache_size,
            "avg_latency_ms": (
                round(self._total_latency_ms / self._request_count, 2)
                if self._request_count > 0
                else 0.0
            ),
        }

    def validate(self) -> list[str]:
        """Validate configuration, returning error messages."""
        errors: list[str] = []
        if not self._api_key:
            errors.append("API key is required (set GEMINI_API_KEY)")
        if self._dimensions <= 0:
            errors.append("embedding dimensions must be positive")
        if not self._model:
            errors.append("embedding model must be specified")
        return errors

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def _ensure_http(self) -> httpx.AsyncClient:
        """Lazily initialize the HTTP client."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=self._connect_timeout,
                    read=self._read_timeout,
                    write=self._read_timeout,
                    pool=self._connect_timeout,
                ),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=30,
                ),
                headers={"Content-Type": "application/json"},
            )
            self._owns_http = True
        return self._http

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._http is not None and self._owns_http:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> GeminiEmbeddingProvider:
        await self._ensure_http()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Cache ─────────────────────────────────────────────────────────

    def _cache_key(self, text: str, task_type: str) -> str:
        """Generate a cache key from text and task type."""
        h = hashlib.sha256(f"{task_type}:{text}".encode()).hexdigest()
        return h

    def _cache_get(self, key: str) -> list[float] | None:
        """Look up cached embedding, moving to front on hit."""
        if self._cache_size <= 0:
            return None
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return self._cache[key]
        return None

    def _cache_put(self, key: str, values: list[float]) -> None:
        """Store embedding in cache, evicting oldest if full."""
        if self._cache_size <= 0:
            return
        self._cache[key] = values
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    # ── Core Methods ──────────────────────────────────────────────────

    async def embed(
        self,
        text: str,
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[float]:
        """Generate an embedding vector for text via Gemini API.

        Args:
            text: Input text to embed.
            task_type: Embedding task type. One of:
                SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY,
                CLASSIFICATION, CLUSTERING.

        Returns:
            Embedding vector as a list of floats.
        """
        # Check cache
        cache_key = self._cache_key(text, task_type)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        result = await self._embed_single(text, task_type)
        self._cache_put(cache_key, result)
        return result

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts via Gemini API.

        Processes in chunks of BATCH_CHUNK_SIZE. Uses the batchEmbedContents
        endpoint for efficiency. Checks cache first for each text.

        Args:
            texts: List of input texts.
            task_type: Embedding task type.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text, task_type)
            cached = self._cache_get(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        # Batch-embed uncached texts
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            embeddings = await self._embed_batch_raw(uncached_texts, task_type)

            for idx, embedding in zip(uncached_indices, embeddings):
                results[idx] = embedding
                cache_key = self._cache_key(texts[idx], task_type)
                self._cache_put(cache_key, embedding)

        return [r for r in results if r is not None]

    async def embed_for_query(self, text: str) -> list[float]:
        """Generate an embedding optimized for search queries.

        Uses RETRIEVAL_QUERY task type for better search results.
        """
        return await self.embed(text, task_type="RETRIEVAL_QUERY")

    async def embed_for_document(self, text: str) -> list[float]:
        """Generate an embedding optimized for document storage.

        Uses RETRIEVAL_DOCUMENT task type for indexed content.
        """
        return await self.embed(text, task_type="RETRIEVAL_DOCUMENT")

    async def embed_for_similarity(self, text: str) -> list[float]:
        """Generate an embedding for semantic similarity comparison."""
        return await self.embed(text, task_type="SEMANTIC_SIMILARITY")

    # ── Internal API Calls ────────────────────────────────────────────

    async def _embed_single(self, text: str, task_type: str) -> list[float]:
        """Make a single embedContent API call with retries."""
        http = await self._ensure_http()
        url = f"/models/{self._model}:embedContent?key={self._api_key}"
        body = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self._dimensions,
        }

        start = time.monotonic()
        data = await self._request_with_retry(http, "POST", url, json=body)
        latency = (time.monotonic() - start) * 1000
        self._total_latency_ms += latency
        self._request_count += 1

        embedding = data.get("embedding", {})
        values = embedding.get("values", [])

        logger.debug(
            "Embedding generated: model=%s dim=%d latency=%.1fms",
            self._model,
            len(values),
            latency,
        )
        return values

    async def _embed_batch_raw(
        self, texts: list[str], task_type: str
    ) -> list[list[float]]:
        """Batch-embed texts using batchEmbedContents, chunked."""
        all_embeddings: list[list[float]] = []

        for chunk_start in range(0, len(texts), self.BATCH_CHUNK_SIZE):
            chunk = texts[chunk_start : chunk_start + self.BATCH_CHUNK_SIZE]
            embeddings = await self._batch_embed_chunk(chunk, task_type)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _batch_embed_chunk(
        self, texts: list[str], task_type: str
    ) -> list[list[float]]:
        """Embed a single chunk via batchEmbedContents."""
        http = await self._ensure_http()
        url = f"/models/{self._model}:batchEmbedContents?key={self._api_key}"
        body = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self._dimensions,
                }
                for text in texts
            ]
        }

        start = time.monotonic()
        data = await self._request_with_retry(http, "POST", url, json=body)
        latency = (time.monotonic() - start) * 1000
        self._total_latency_ms += latency
        self._request_count += 1

        embeddings_data = data.get("embeddings", [])
        result = [e.get("values", []) for e in embeddings_data]

        logger.debug(
            "Batch embedding: count=%d model=%s latency=%.1fms",
            len(result),
            self._model,
            latency,
        )
        return result

    async def _request_with_retry(
        self,
        http: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """HTTP request with exponential backoff retries."""
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await http.request(method, url, **kwargs)

                if response.status_code == 200:
                    return response.json()  # type: ignore[no-any-return]

                body_text = response.text

                # Non-retryable
                if response.status_code in (400, 401, 403):
                    raise EmbeddingError(
                        f"Gemini API error {response.status_code}: {body_text}",
                        status_code=response.status_code,
                        retryable=False,
                    )

                # Retryable
                last_error = EmbeddingError(
                    f"Gemini API error {response.status_code}: {body_text}",
                    status_code=response.status_code,
                    retryable=True,
                )

            except EmbeddingError:
                raise
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_error = EmbeddingError(
                    f"Connection error: {e}", retryable=True
                )

            # Backoff before retry
            if attempt < self._max_retries:
                delay = min(
                    self._retry_backoff_base * (2**attempt),
                    self._retry_backoff_max,
                )
                logger.warning(
                    "Embedding request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

        assert last_error is not None
        raise last_error


class EmbeddingError(Exception):
    """Error from embedding operations."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


# ── Local Provider ────────────────────────────────────────────────────


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider for offline/local-only mode.

    Uses a lightweight local model (e.g., sentence-transformers) so
    local-only mode has full feature parity with no data leakage.
    """

    DIMENSION = 384  # all-MiniLM-L6-v2 dimension

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer(self._model_name)

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    async def embed(self, text: str) -> list[float]:
        self._load_model()
        vector = self._model.encode(text)
        return vector.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        vectors = self._model.encode(texts)
        return [v.tolist() for v in vectors]


# ── Mock Provider ─────────────────────────────────────────────────────


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing.

    Generates deterministic embeddings based on text hash for
    reproducible tests without requiring API keys or model downloads.
    """

    DIMENSION = 64

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    async def embed(self, text: str) -> list[float]:
        return self._hash_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-embedding from text."""
        h = hash(text)
        vector = []
        for i in range(self.DIMENSION):
            val = math.sin(h * (i + 1) * 0.1) * 0.5
            vector.append(val)
        # Normalize to unit vector
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        return vector


# ── Similarity Utilities ──────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    # Clamp to [-1, 1] to handle floating point precision
    return max(-1.0, min(1.0, dot / (mag_a * mag_b)))
