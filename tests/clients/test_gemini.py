"""Tests for the Gemini 2 multimodal API client wrapper.

Tests cover authentication, configuration, connection lifecycle,
retry logic, and all API methods using httpx mocking.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx
import pytest

from blurt.clients.gemini import (
    ClientState,
    GeminiAuthError,
    GeminiClient,
    GeminiConnectionError,
    GeminiError,
    GeminiModelError,
    GeminiRateLimitError,
    GeminiResponse,
    ModelTier,
)
from blurt.config.settings import GeminiConfig

BASE_URL = "https://test.googleapis.com/v1beta"


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def config() -> GeminiConfig:
    """Valid test configuration."""
    return GeminiConfig(
        api_key="test-api-key-12345",
        base_url=BASE_URL,
        connect_timeout=5.0,
        read_timeout=30.0,
        max_retries=2,
        retry_backoff_base=0.01,  # fast retries for tests
    )


@pytest.fixture
def mock_generate_response() -> dict:
    """Standard Gemini generateContent response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "This is a task intent."}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
    }


@pytest.fixture
def mock_embedding_response() -> dict:
    """Standard embedding response."""
    return {
        "embedding": {
            "values": [0.1] * 768,
        }
    }


@pytest.fixture
def mock_models_list_response() -> dict:
    """Models list response for health check."""
    return {
        "models": [{"name": "models/gemini-2.5-flash-lite"}],
    }


def _make_mock_client(handler, base_url: str = BASE_URL) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with a mock transport and base_url."""
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url=base_url)


def _setup_client(config: GeminiConfig, handler) -> GeminiClient:
    """Create a GeminiClient in CONNECTED state with a mock transport."""
    client = GeminiClient(config)
    client._http = _make_mock_client(handler, config.base_url)
    client._state = ClientState.CONNECTED
    return client


# ── Configuration Tests ─────────────────────────────────────────────


class TestGeminiConfig:
    def test_default_config(self):
        cfg = GeminiConfig()
        assert cfg.api_key == ""
        assert "googleapis.com" in cfg.base_url
        assert cfg.flash_lite_model == "gemini-2.5-flash-lite"
        assert cfg.flash_model == "gemini-2.5-flash"
        assert cfg.max_retries == 3

    def test_from_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key-123"}):
            cfg = GeminiConfig.from_env()
            assert cfg.api_key == "env-key-123"

    def test_from_env_with_overrides(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            cfg = GeminiConfig.from_env(api_key="override-key")
            assert cfg.api_key == "override-key"

    def test_from_env_all_vars(self):
        env = {
            "GEMINI_API_KEY": "k",
            "GEMINI_BASE_URL": "https://custom.api/v1",
            "GEMINI_CONNECT_TIMEOUT": "15",
            "GEMINI_READ_TIMEOUT": "120",
            "GEMINI_STREAM_TIMEOUT": "240",
            "GEMINI_MAX_RETRIES": "5",
        }
        with patch.dict(os.environ, env):
            cfg = GeminiConfig.from_env()
            assert cfg.api_key == "k"
            assert cfg.base_url == "https://custom.api/v1"
            assert cfg.connect_timeout == 15.0
            assert cfg.read_timeout == 120.0
            assert cfg.stream_timeout == 240.0
            assert cfg.max_retries == 5

    def test_validate_missing_key(self):
        cfg = GeminiConfig()
        errors = cfg.validate()
        assert any("api_key" in e for e in errors)

    def test_validate_invalid_timeout(self):
        cfg = GeminiConfig(api_key="key", connect_timeout=-1)
        errors = cfg.validate()
        assert any("connect_timeout" in e for e in errors)

    def test_validate_valid(self):
        cfg = GeminiConfig(api_key="key")
        assert cfg.validate() == []

    def test_frozen(self):
        cfg = GeminiConfig(api_key="key")
        with pytest.raises(AttributeError):
            cfg.api_key = "new"  # type: ignore[misc]


# ── GeminiResponse Tests ────────────────────────────────────────────


class TestGeminiResponse:
    def test_from_api_response(self, mock_generate_response):
        resp = GeminiResponse.from_api_response(
            mock_generate_response, "gemini-2.5-flash-lite", 42.5
        )
        assert resp.text == "This is a task intent."
        assert resp.model == "gemini-2.5-flash-lite"
        assert resp.latency_ms == 42.5
        assert resp.finish_reason == "STOP"
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 8
        assert resp.usage["total_tokens"] == 18

    def test_from_empty_response(self):
        resp = GeminiResponse.from_api_response({}, "model", 0.0)
        assert resp.text == ""
        assert resp.finish_reason == ""

    def test_from_no_text_parts(self):
        data = {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}
        resp = GeminiResponse.from_api_response(data, "model", 0.0)
        assert resp.text == ""

    def test_multi_part_text(self):
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello "}, {"text": "world"}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        resp = GeminiResponse.from_api_response(data, "model", 0.0)
        assert resp.text == "Hello world"


# ── Client Lifecycle Tests ───────────────────────────────────────────


class TestClientLifecycle:
    def test_initial_state(self, config):
        client = GeminiClient(config)
        assert client.state == ClientState.CREATED
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_connect_and_close(self, config, mock_models_list_response):
        client = GeminiClient(config)
        client._http = _make_mock_client(
            lambda req: httpx.Response(200, json=mock_models_list_response),
            config.base_url,
        )
        client._state = ClientState.CREATED
        await client._health_check()
        client._state = ClientState.CONNECTED

        assert client.state == ClientState.CONNECTED
        assert client.is_connected

        await client.close()
        assert client.state == ClientState.CLOSED
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_context_manager(self, config, mock_models_list_response):
        """Test async with lifecycle."""
        mock_client = _make_mock_client(
            lambda req: httpx.Response(200, json=mock_models_list_response),
            config.base_url,
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            async with GeminiClient(config) as client:
                assert client.is_connected
            assert client.state == ClientState.CLOSED

    @pytest.mark.asyncio
    async def test_connect_without_api_key(self):
        cfg = GeminiConfig(api_key="")
        client = GeminiClient(cfg)
        with pytest.raises(GeminiAuthError):
            await client.connect()

    @pytest.mark.asyncio
    async def test_connect_degraded_on_health_failure(self, config):
        """Client enters DEGRADED state if health check fails at connect."""
        mock_client = _make_mock_client(
            lambda req: httpx.Response(500, json={"error": "down"}),
            config.base_url,
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            client = GeminiClient(config)
            await client.connect()
            assert client.state == ClientState.DEGRADED
            assert client.is_connected
            await client.close()

    @pytest.mark.asyncio
    async def test_double_connect_noop(self, config, mock_models_list_response):
        mock_client = _make_mock_client(
            lambda req: httpx.Response(200, json=mock_models_list_response),
            config.base_url,
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            client = GeminiClient(config)
            await client.connect()
            await client.connect()  # Should be a no-op
            assert client.state == ClientState.CONNECTED
            await client.close()

    @pytest.mark.asyncio
    async def test_ensure_connected_raises_when_closed(self, config):
        client = GeminiClient(config)
        client._state = ClientState.CLOSED
        with pytest.raises(GeminiError, match="closed"):
            client._ensure_connected()

    @pytest.mark.asyncio
    async def test_ensure_connected_raises_when_created(self, config):
        client = GeminiClient(config)
        with pytest.raises(GeminiError, match="not connected"):
            client._ensure_connected()


# ── Stats Tests ──────────────────────────────────────────────────────


class TestClientStats:
    def test_initial_stats(self, config):
        client = GeminiClient(config)
        stats = client.stats
        assert stats["request_count"] == 0
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["state"] == "created"

    def test_config_readonly(self, config):
        client = GeminiClient(config)
        assert client.config is config


# ── Generate Tests ───────────────────────────────────────────────────


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_text(self, config, mock_generate_response):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "generateContent" in str(request.url)
            assert "key=test-api-key-12345" in str(request.url)
            body = json.loads(request.content)
            assert body["contents"][0]["parts"][0]["text"] == "Classify this"
            return httpx.Response(200, json=mock_generate_response)

        client = _setup_client(config, handler)
        resp = await client.generate("Classify this", tier=ModelTier.FAST)

        assert resp.text == "This is a task intent."
        assert resp.model == "gemini-2.5-flash-lite"
        assert resp.latency_ms > 0
        assert client.stats["request_count"] == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_generate_with_system_instruction(self, config, mock_generate_response):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert "systemInstruction" in body
            assert body["systemInstruction"]["parts"][0]["text"] == "You are a classifier"
            return httpx.Response(200, json=mock_generate_response)

        client = _setup_client(config, handler)
        await client.generate("Classify this", system_instruction="You are a classifier")
        await client.close()

    @pytest.mark.asyncio
    async def test_generate_json_mode(self, config, mock_generate_response):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["generationConfig"]["responseMimeType"] == "application/json"
            return httpx.Response(200, json=mock_generate_response)

        client = _setup_client(config, handler)
        await client.generate("Return JSON", response_mime_type="application/json")
        await client.close()

    @pytest.mark.asyncio
    async def test_generate_smart_tier(self, config, mock_generate_response):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "gemini-2.5-flash:" in str(request.url)
            assert "flash-lite" not in str(request.url)
            return httpx.Response(200, json=mock_generate_response)

        client = _setup_client(config, handler)
        await client.generate("Think deeply", tier=ModelTier.SMART)
        await client.close()

    @pytest.mark.asyncio
    async def test_model_tier_resolution(self, config):
        client = GeminiClient(config)
        assert client._resolve_model(ModelTier.FAST) == "gemini-2.5-flash-lite"
        assert client._resolve_model(ModelTier.SMART) == "gemini-2.5-flash"


# ── Audio (Multimodal) Tests ────────────────────────────────────────


class TestGenerateFromAudio:
    @pytest.mark.asyncio
    async def test_audio_input(self, config, mock_generate_response):
        audio_bytes = b"\x00\x01\x02\x03" * 100

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            parts = body["contents"][0]["parts"]
            assert "inline_data" in parts[0]
            assert parts[0]["inline_data"]["mime_type"] == "audio/webm"
            import base64
            decoded = base64.b64decode(parts[0]["inline_data"]["data"])
            assert decoded == audio_bytes
            return httpx.Response(200, json=mock_generate_response)

        client = _setup_client(config, handler)
        resp = await client.generate_from_audio(audio_bytes)
        assert resp.text == "This is a task intent."
        await client.close()

    @pytest.mark.asyncio
    async def test_audio_with_prompt(self, config, mock_generate_response):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            parts = body["contents"][0]["parts"]
            assert len(parts) == 2  # audio + text
            assert "inline_data" in parts[0]
            assert parts[1]["text"] == "Classify this audio"
            return httpx.Response(200, json=mock_generate_response)

        client = _setup_client(config, handler)
        await client.generate_from_audio(b"\x00" * 100, prompt="Classify this audio")
        await client.close()

    @pytest.mark.asyncio
    async def test_audio_custom_mime(self, config, mock_generate_response):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "audio/wav"
            return httpx.Response(200, json=mock_generate_response)

        client = _setup_client(config, handler)
        await client.generate_from_audio(b"\x00", mime_type="audio/wav")
        await client.close()


# ── Embedding Tests ──────────────────────────────────────────────────


class TestEmbed:
    @pytest.mark.asyncio
    async def test_embed_text(self, config, mock_embedding_response):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "embedContent" in str(request.url)
            body = json.loads(request.content)
            assert body["content"]["parts"][0]["text"] == "Hello world"
            assert body["taskType"] == "SEMANTIC_SIMILARITY"
            return httpx.Response(200, json=mock_embedding_response)

        client = _setup_client(config, handler)
        resp = await client.embed("Hello world")
        assert len(resp.values) == 768
        assert resp.model == "gemini-embedding-001"
        assert resp.latency_ms > 0
        await client.close()

    @pytest.mark.asyncio
    async def test_embed_batch(self, config, mock_embedding_response):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=mock_embedding_response)

        client = _setup_client(config, handler)
        results = await client.embed_batch(["text1", "text2", "text3"])
        assert len(results) == 3
        assert all(len(r.values) == 768 for r in results)
        await client.close()

    @pytest.mark.asyncio
    async def test_embed_custom_task_type(self, config, mock_embedding_response):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["taskType"] == "RETRIEVAL_QUERY"
            return httpx.Response(200, json=mock_embedding_response)

        client = _setup_client(config, handler)
        await client.embed("search query", task_type="RETRIEVAL_QUERY")
        await client.close()


# ── Multi-turn Tests ─────────────────────────────────────────────────


class TestMultiTurn:
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, config, mock_generate_response):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert len(body["contents"]) == 3
            assert body["contents"][0]["role"] == "user"
            assert body["contents"][1]["role"] == "model"
            assert body["contents"][2]["role"] == "user"
            return httpx.Response(200, json=mock_generate_response)

        client = _setup_client(config, handler)
        messages = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there"}]},
            {"role": "user", "parts": [{"text": "What about Q2?"}]},
        ]
        resp = await client.generate_multi_turn(messages)
        assert resp.text == "This is a task intent."
        await client.close()


# ── Error Handling Tests ─────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_auth_error_401(self, config):
        client = _setup_client(
            config, lambda req: httpx.Response(401, json={"error": "unauthorized"})
        )
        with pytest.raises(GeminiAuthError):
            await client.generate("test")
        await client.close()

    @pytest.mark.asyncio
    async def test_auth_error_403(self, config):
        client = _setup_client(
            config, lambda req: httpx.Response(403, json={"error": "forbidden"})
        )
        with pytest.raises(GeminiAuthError, match="permissions"):
            await client.generate("test")
        await client.close()

    @pytest.mark.asyncio
    async def test_bad_request_400(self, config):
        client = _setup_client(
            config, lambda req: httpx.Response(400, json={"error": "bad"})
        )
        with pytest.raises(GeminiModelError):
            await client.generate("test")
        await client.close()

    @pytest.mark.asyncio
    async def test_rate_limit_429(self, config):
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(429, json={"error": "rate limited"})

        client = _setup_client(config, handler)
        with pytest.raises(GeminiRateLimitError):
            await client.generate("test")

        # Should have retried max_retries times
        assert call_count == config.max_retries + 1
        await client.close()

    @pytest.mark.asyncio
    async def test_server_error_retries(self, config, mock_generate_response):
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(503, json={"error": "unavailable"})
            return httpx.Response(200, json=mock_generate_response)

        client = _setup_client(config, handler)
        resp = await client.generate("test")
        assert resp.text == "This is a task intent."
        assert call_count == 3  # 2 failures + 1 success
        await client.close()

    @pytest.mark.asyncio
    async def test_error_count_tracked(self, config):
        client = _setup_client(
            config, lambda req: httpx.Response(401, json={"error": "unauthorized"})
        )
        with pytest.raises(GeminiAuthError):
            await client.generate("test")

        assert client.stats["error_count"] == 1
        assert client.stats["request_count"] == 1
        await client.close()


# ── Health Check Tests ───────────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self, config, mock_models_list_response):
        client = _setup_client(
            config, lambda req: httpx.Response(200, json=mock_models_list_response)
        )
        result = await client.health_check()
        assert result["healthy"] is True
        assert result["state"] == "connected"
        assert result["latency_ms"] >= 0
        await client.close()

    @pytest.mark.asyncio
    async def test_unhealthy_degrades(self, config):
        client = _setup_client(
            config, lambda req: httpx.Response(500, json={"error": "down"})
        )
        result = await client.health_check()
        assert result["healthy"] is False
        assert result["state"] == "degraded"
        assert "error" in result
        await client.close()

    @pytest.mark.asyncio
    async def test_recovery_from_degraded(self, config, mock_models_list_response):
        client = _setup_client(
            config, lambda req: httpx.Response(200, json=mock_models_list_response)
        )
        client._state = ClientState.DEGRADED

        result = await client.health_check()
        assert result["healthy"] is True
        assert result["state"] == "connected"
        assert client.state == ClientState.CONNECTED
        await client.close()


# ── Exception Hierarchy Tests ────────────────────────────────────────


class TestExceptions:
    def test_base_error(self):
        err = GeminiError("test", status_code=500, retryable=True)
        assert str(err) == "test"
        assert err.status_code == 500
        assert err.retryable is True

    def test_auth_error(self):
        err = GeminiAuthError()
        assert err.status_code == 401
        assert err.retryable is False

    def test_rate_limit_error(self):
        err = GeminiRateLimitError(retry_after=5.0)
        assert err.status_code == 429
        assert err.retryable is True
        assert err.retry_after == 5.0

    def test_connection_error(self):
        err = GeminiConnectionError("timeout")
        assert err.retryable is True

    def test_model_error(self):
        err = GeminiModelError("bad input", status_code=400)
        assert err.status_code == 400

    def test_inheritance(self):
        assert issubclass(GeminiAuthError, GeminiError)
        assert issubclass(GeminiRateLimitError, GeminiError)
        assert issubclass(GeminiConnectionError, GeminiError)
        assert issubclass(GeminiModelError, GeminiError)
