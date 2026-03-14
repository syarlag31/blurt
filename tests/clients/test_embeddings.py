"""Tests for the Gemini 2 embedding service client.

Tests cover:
- Configuration and API key management
- GeminiEmbeddingProvider with httpx mocking
- Single and batch embedding generation
- LRU cache behavior
- Task type routing (query, document, similarity)
- Retry logic on transient failures
- Error handling (auth, bad request, network)
- LocalEmbeddingProvider and MockEmbeddingProvider
- EmbeddingService high-level API
- Similarity search utilities
- cosine_similarity function
"""

from __future__ import annotations

import math

import httpx
import pytest

from blurt.clients.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingResult,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    cosine_similarity,
)
from blurt.config.settings import DeploymentMode, GeminiConfig
from blurt.services.embedding import EmbeddingService, SimilarityMatch


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def gemini_config() -> GeminiConfig:
    """Valid test configuration."""
    return GeminiConfig(
        api_key="test-embed-key-12345",
        base_url="https://test.googleapis.com/v1beta",
        embedding_model="gemini-embedding-001",
        embedding_dimensions=768,
        max_retries=1,
        retry_backoff_base=0.01,
    )


@pytest.fixture
def provider(gemini_config: GeminiConfig) -> GeminiEmbeddingProvider:
    """GeminiEmbeddingProvider with test config."""
    return GeminiEmbeddingProvider(config=gemini_config, cache_size=10)


@pytest.fixture
def mock_provider() -> MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    return MockEmbeddingProvider()


def _make_embed_response(values: list[float]) -> dict:
    """Build a mock embedContent response."""
    return {"embedding": {"values": values}}


def _make_batch_response(embeddings: list[list[float]]) -> dict:
    """Build a mock batchEmbedContents response."""
    return {"embeddings": [{"values": v} for v in embeddings]}


def _mock_transport(responses: list[httpx.Response]) -> httpx.AsyncBaseTransport:
    """Create a mock transport that returns responses in order."""
    call_count = 0

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return responses[idx]

    return MockTransport()


# ── GeminiEmbeddingProvider Tests ─────────────────────────────────


class TestGeminiEmbeddingProviderConfig:
    """Configuration and API key management tests."""

    def test_from_config(self, gemini_config: GeminiConfig):
        provider = GeminiEmbeddingProvider(config=gemini_config)
        assert provider.dimension == 768
        assert provider.model == "gemini-embedding-001"
        assert provider.api_key == "test-embed-key-12345"

    def test_from_kwargs(self):
        provider = GeminiEmbeddingProvider(
            api_key="my-key",
            model="custom-model",
            dimensions=256,
        )
        assert provider.dimension == 256
        assert provider.model == "custom-model"
        assert provider.api_key == "my-key"

    def test_defaults_without_config(self):
        provider = GeminiEmbeddingProvider(api_key="k")
        assert provider.dimension == 768
        assert provider.model == "gemini-embedding-001"

    def test_api_key_masked(self):
        provider = GeminiEmbeddingProvider(api_key="abcd1234efgh5678")
        assert provider.api_key_masked == "abcd...5678"

    def test_api_key_masked_short(self):
        provider = GeminiEmbeddingProvider(api_key="short")
        assert provider.api_key_masked == "****"

    def test_api_key_masked_empty(self):
        provider = GeminiEmbeddingProvider(api_key="")
        assert provider.api_key_masked == "<not set>"

    def test_validate_missing_key(self):
        provider = GeminiEmbeddingProvider(api_key="")
        errors = provider.validate()
        assert any("API key" in e for e in errors)

    def test_validate_valid(self, gemini_config: GeminiConfig):
        provider = GeminiEmbeddingProvider(config=gemini_config)
        assert provider.validate() == []

    def test_validate_bad_dimensions(self):
        provider = GeminiEmbeddingProvider(api_key="k", dimensions=-1)
        errors = provider.validate()
        assert any("dimensions" in e for e in errors)

    def test_stats_initial(self, provider: GeminiEmbeddingProvider):
        stats = provider.stats
        assert stats["request_count"] == 0
        assert stats["cache_hits"] == 0
        assert stats["model"] == "gemini-embedding-001"
        assert stats["dimension"] == 768


class TestGeminiEmbeddingProviderEmbed:
    """Single embedding generation tests."""

    @pytest.mark.asyncio
    async def test_embed_single(self, provider: GeminiEmbeddingProvider):
        """Test single text embedding via Gemini API."""
        fake_vec = [0.1] * 768
        transport = _mock_transport([
            httpx.Response(200, json=_make_embed_response(fake_vec)),
        ])
        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=transport,
        )
        provider._owns_http = True

        result = await provider.embed("hello world")
        assert result == fake_vec
        assert provider.stats["request_count"] == 1

    @pytest.mark.asyncio
    async def test_embed_caches_result(self, provider: GeminiEmbeddingProvider):
        """Test that repeated embed calls use cache."""
        fake_vec = [0.2] * 768
        transport = _mock_transport([
            httpx.Response(200, json=_make_embed_response(fake_vec)),
        ])
        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=transport,
        )
        provider._owns_http = True

        result1 = await provider.embed("cached text")
        result2 = await provider.embed("cached text")
        assert result1 == result2
        assert provider.stats["request_count"] == 1  # only 1 API call
        assert provider.stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_embed_different_task_types_not_cached(
        self, provider: GeminiEmbeddingProvider
    ):
        """Different task types should produce separate cache entries."""
        vec1 = [0.1] * 768
        vec2 = [0.2] * 768
        transport = _mock_transport([
            httpx.Response(200, json=_make_embed_response(vec1)),
            httpx.Response(200, json=_make_embed_response(vec2)),
        ])
        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=transport,
        )
        provider._owns_http = True

        r1 = await provider.embed("text", task_type="RETRIEVAL_DOCUMENT")
        r2 = await provider.embed("text", task_type="RETRIEVAL_QUERY")
        assert r1 == vec1
        assert r2 == vec2
        assert provider.stats["request_count"] == 2

    @pytest.mark.asyncio
    async def test_embed_cache_eviction(self):
        """Test LRU cache evicts oldest entry when full."""
        config = GeminiConfig(
            api_key="k",
            base_url="https://test.googleapis.com/v1beta",
            max_retries=0,
        )
        provider = GeminiEmbeddingProvider(config=config, cache_size=2)

        call_count = 0

        class CountTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                nonlocal call_count
                call_count += 1
                return httpx.Response(200, json=_make_embed_response([float(call_count)] * 768))

        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=CountTransport(),
        )
        provider._owns_http = True

        await provider.embed("a")  # call 1, cache: {a}
        await provider.embed("b")  # call 2, cache: {a, b}
        await provider.embed("c")  # call 3, cache: {b, c} — a evicted

        # "a" should require a new API call (was evicted)
        call_count_before = call_count
        await provider.embed("a")  # call 4, cache: {c, a} — b evicted
        assert call_count == call_count_before + 1

        # "c" should still be cached (was not evicted)
        call_count_before = call_count
        await provider.embed("c")
        assert call_count == call_count_before  # no new call


class TestGeminiEmbeddingProviderBatch:
    """Batch embedding tests."""

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider: GeminiEmbeddingProvider):
        """Test batch embedding with batchEmbedContents."""
        vecs = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        transport = _mock_transport([
            httpx.Response(200, json=_make_batch_response(vecs)),
        ])
        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=transport,
        )
        provider._owns_http = True

        results = await provider.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert results[0] == vecs[0]
        assert results[2] == vecs[2]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, provider: GeminiEmbeddingProvider):
        """Empty input returns empty output."""
        results = await provider.embed_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_batch_uses_cache(self, provider: GeminiEmbeddingProvider):
        """Batch should use cache for previously embedded texts."""
        vec_a = [0.1] * 768
        vec_b = [0.2] * 768

        # Pre-populate cache via single embed
        transport = _mock_transport([
            httpx.Response(200, json=_make_embed_response(vec_a)),
            httpx.Response(200, json=_make_batch_response([vec_b])),
        ])
        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=transport,
        )
        provider._owns_http = True

        await provider.embed("a")  # caches a
        results = await provider.embed_batch(["a", "b"])  # a from cache, b from API
        assert len(results) == 2
        assert results[0] == vec_a
        assert results[1] == vec_b


class TestGeminiEmbeddingProviderRetries:
    """Retry and error handling tests."""

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self, provider: GeminiEmbeddingProvider):
        """Test retry on 500 server error."""
        fake_vec = [0.5] * 768
        transport = _mock_transport([
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(200, json=_make_embed_response(fake_vec)),
        ])
        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=transport,
        )
        provider._owns_http = True

        result = await provider.embed("retry text")
        assert result == fake_vec

    @pytest.mark.asyncio
    async def test_error_on_auth_failure(self, provider: GeminiEmbeddingProvider):
        """Test non-retryable 401 error."""
        transport = _mock_transport([
            httpx.Response(401, text="Unauthorized"),
        ])
        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=transport,
        )
        provider._owns_http = True

        with pytest.raises(EmbeddingError) as exc_info:
            await provider.embed("bad key")
        assert exc_info.value.status_code == 401
        assert not exc_info.value.retryable

    @pytest.mark.asyncio
    async def test_error_on_bad_request(self, provider: GeminiEmbeddingProvider):
        """Test non-retryable 400 error."""
        transport = _mock_transport([
            httpx.Response(400, text="Bad Request"),
        ])
        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=transport,
        )
        provider._owns_http = True

        with pytest.raises(EmbeddingError) as exc_info:
            await provider.embed("bad input")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test that retries exhaust and raise."""
        config = GeminiConfig(
            api_key="k",
            base_url="https://test.googleapis.com/v1beta",
            max_retries=2,
            retry_backoff_base=0.001,
        )
        provider = GeminiEmbeddingProvider(config=config)
        transport = _mock_transport([
            httpx.Response(500, text="fail"),
            httpx.Response(500, text="fail"),
            httpx.Response(500, text="fail"),
        ])
        provider._http = httpx.AsyncClient(
            base_url="https://test.googleapis.com/v1beta",
            transport=transport,
        )
        provider._owns_http = True

        with pytest.raises(EmbeddingError) as exc_info:
            await provider.embed("doomed")
        assert exc_info.value.retryable


class TestGeminiEmbeddingProviderLifecycle:
    """Lifecycle management tests."""

    @pytest.mark.asyncio
    async def test_context_manager(self, gemini_config: GeminiConfig):
        """Test async context manager creates and closes HTTP client."""
        provider = GeminiEmbeddingProvider(config=gemini_config)
        assert provider._http is None

        async with provider:
            assert provider._http is not None
            assert not provider._http.is_closed

        assert provider._http is None

    @pytest.mark.asyncio
    async def test_clear_cache(self, provider: GeminiEmbeddingProvider):
        """Test cache clearing."""
        provider._cache["key1"] = [0.1]
        provider._cache["key2"] = [0.2]
        assert len(provider._cache) == 2

        provider.clear_cache()
        assert len(provider._cache) == 0

    def test_convenience_methods_exist(self, provider: GeminiEmbeddingProvider):
        """Test that convenience embedding methods exist."""
        assert hasattr(provider, "embed_for_query")
        assert hasattr(provider, "embed_for_document")
        assert hasattr(provider, "embed_for_similarity")


# ── MockEmbeddingProvider Tests ───────────────────────────────────


class TestMockEmbeddingProvider:
    """Tests for the deterministic mock provider."""

    @pytest.mark.asyncio
    async def test_embed_deterministic(self, mock_provider: MockEmbeddingProvider):
        """Same text always produces the same embedding."""
        v1 = await mock_provider.embed("hello")
        v2 = await mock_provider.embed("hello")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_embed_different_texts(self, mock_provider: MockEmbeddingProvider):
        """Different texts produce different embeddings."""
        v1 = await mock_provider.embed("hello")
        v2 = await mock_provider.embed("world")
        assert v1 != v2

    @pytest.mark.asyncio
    async def test_embed_dimension(self, mock_provider: MockEmbeddingProvider):
        """Embeddings have the correct dimension."""
        v = await mock_provider.embed("test")
        assert len(v) == 64
        assert mock_provider.dimension == 64

    @pytest.mark.asyncio
    async def test_embed_normalized(self, mock_provider: MockEmbeddingProvider):
        """Embeddings are unit vectors."""
        v = await mock_provider.embed("test normalization")
        magnitude = math.sqrt(sum(x * x for x in v))
        assert abs(magnitude - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_provider: MockEmbeddingProvider):
        """Batch embedding works correctly."""
        results = await mock_provider.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(v) == 64 for v in results)


# ── cosine_similarity Tests ───────────────────────────────────────


class TestCosineSimilarity:

    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1, 0], [1, 0, 0])

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 0]) == 0.0


# ── EmbeddingService Tests ────────────────────────────────────────


class TestEmbeddingService:
    """Tests for the high-level embedding service."""

    @pytest.fixture
    def service(self, mock_provider: MockEmbeddingProvider) -> EmbeddingService:
        return EmbeddingService(mock_provider, mode=DeploymentMode.LOCAL)

    @pytest.mark.asyncio
    async def test_embed_text(self, service: EmbeddingService):
        result = await service.embed_text("test input")
        assert isinstance(result, EmbeddingResult)
        assert len(result.values) == 64
        assert result.dimension == 64

    @pytest.mark.asyncio
    async def test_embed_texts(self, service: EmbeddingService):
        results = await service.embed_texts(["a", "b", "c"])
        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResult) for r in results)

    @pytest.mark.asyncio
    async def test_embed_texts_empty(self, service: EmbeddingService):
        results = await service.embed_texts([])
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_for_search(self, service: EmbeddingService):
        result = await service.embed_for_search("find something")
        assert isinstance(result, EmbeddingResult)
        assert len(result.values) == 64

    @pytest.mark.asyncio
    async def test_embed_for_storage(self, service: EmbeddingService):
        result = await service.embed_for_storage("store this")
        assert isinstance(result, EmbeddingResult)

    @pytest.mark.asyncio
    async def test_compute_similarity(self, service: EmbeddingService):
        score = await service.compute_similarity("hello", "hello")
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_compute_similarity_different(self, service: EmbeddingService):
        score = await service.compute_similarity("hello", "completely different text")
        assert -1.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_find_similar(self, service: EmbeddingService):
        # Build some candidate embeddings
        candidates = []
        for text in ["apple pie", "banana split", "apple sauce"]:
            vec = await service.embed_text(text)
            candidates.append((text, vec.values))

        results = await service.find_similar("apple", candidates, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SimilarityMatch) for r in results)
        # Results should be sorted by score descending
        if len(results) > 1:
            assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_find_similar_with_min_score(self, service: EmbeddingService):
        candidates = [("x", [0.0] * 64)]  # zero vector -> 0 similarity
        results = await service.find_similar(
            "query", candidates, min_score=0.5
        )
        # Zero vector should be filtered out
        assert len(results) == 0

    def test_dimension(self, service: EmbeddingService):
        assert service.dimension == 64

    def test_mode(self, service: EmbeddingService):
        assert service.mode == DeploymentMode.LOCAL
        assert not service.is_cloud

    def test_get_stats(self, service: EmbeddingService):
        stats = service.get_stats()
        assert stats["mode"] == "local"
        assert stats["dimension"] == 64
        assert stats["provider"] == "MockEmbeddingProvider"


class TestEmbeddingServiceFromConfig:
    """Tests for EmbeddingService.from_config factory."""

    def test_cloud_mode_requires_api_key(self):
        config = GeminiConfig(api_key="")
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            EmbeddingService.from_config(config, mode=DeploymentMode.CLOUD)

    def test_cloud_mode_with_key(self):
        config = GeminiConfig(api_key="test-key")
        service = EmbeddingService.from_config(config, mode=DeploymentMode.CLOUD)
        assert service.is_cloud
        assert isinstance(service.provider, GeminiEmbeddingProvider)

    def test_local_mode_fallback_to_mock(self):
        config = GeminiConfig()
        service = EmbeddingService.from_config(config, mode=DeploymentMode.LOCAL)
        assert not service.is_cloud
        # Should be either Local or Mock provider
        assert isinstance(service.provider, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        config = GeminiConfig(api_key="test-key")
        service = EmbeddingService.from_config(config, mode=DeploymentMode.CLOUD)
        async with service:
            assert service._is_open
        assert not service._is_open

    def test_clear_cache_cloud(self):
        config = GeminiConfig(api_key="test-key")
        service = EmbeddingService.from_config(config, mode=DeploymentMode.CLOUD)
        # Should not raise
        service.clear_cache()

    def test_clear_cache_local(self):
        config = GeminiConfig()
        service = EmbeddingService.from_config(config, mode=DeploymentMode.LOCAL)
        # Should not raise (no-op for non-Gemini providers)
        service.clear_cache()
