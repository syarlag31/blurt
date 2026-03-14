"""Gemini 2 multimodal API client wrapper.

Provides authenticated, lifecycle-managed access to Gemini 2 models with:
- Two-model strategy (Flash-Lite for classification, Flash for reasoning)
- Raw audio input support for voice-first interaction
- Streaming and non-streaming response modes
- Embedding generation for semantic memory
- Automatic retries with exponential backoff
- Connection pooling and graceful shutdown
- Health checks and readiness probes
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Self

import httpx

from blurt.config.settings import GeminiConfig

logger = logging.getLogger(__name__)


class ClientState(str, Enum):
    """Lifecycle states for the Gemini client."""

    CREATED = "created"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    CLOSED = "closed"


class GeminiError(Exception):
    """Base exception for Gemini client errors."""

    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class GeminiAuthError(GeminiError):
    """Authentication or authorization failure."""

    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message, status_code=401, retryable=False)


class GeminiRateLimitError(GeminiError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429, retryable=True)
        self.retry_after = retry_after


class GeminiConnectionError(GeminiError):
    """Network or connection failure."""

    def __init__(self, message: str = "Connection failed") -> None:
        super().__init__(message, retryable=True)


class GeminiModelError(GeminiError):
    """Model returned an error response."""

    pass


@dataclass
class GeminiResponse:
    """Parsed response from a Gemini API call.

    Attributes:
        text: Extracted text content from the response.
        raw: Full raw response dictionary.
        model: Model that generated this response.
        usage: Token usage statistics.
        latency_ms: Request latency in milliseconds.
        finish_reason: Why generation stopped.
    """

    text: str
    raw: dict[str, Any]
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    finish_reason: str = ""

    @classmethod
    def from_api_response(cls, data: dict[str, Any], model: str, latency_ms: float) -> Self:
        """Parse a Gemini API response into a GeminiResponse."""
        text = ""
        finish_reason = ""

        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "")
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            text_parts = [p["text"] for p in parts if "text" in p]
            text = "".join(text_parts)

        usage_meta = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_meta.get("promptTokenCount", 0),
            "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
            "total_tokens": usage_meta.get("totalTokenCount", 0),
        }

        return cls(
            text=text,
            raw=data,
            model=model,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )


@dataclass
class EmbeddingResponse:
    """Response from an embedding request.

    Attributes:
        values: The embedding vector.
        model: Model used for embedding.
        latency_ms: Request latency in milliseconds.
    """

    values: list[float]
    model: str
    latency_ms: float = 0.0


class ModelTier(str, Enum):
    """Which model tier to use for a request.

    FAST: Flash-Lite — classification, extraction, emotion detection.
    SMART: Flash — reasoning, insights, journaling, Q&A.
    """

    FAST = "fast"
    SMART = "smart"


class GeminiClient:
    """Async Gemini 2 multimodal API client with lifecycle management.

    Usage::

        config = GeminiConfig.from_env()
        async with GeminiClient(config) as client:
            response = await client.generate(
                prompt="Classify this intent",
                tier=ModelTier.FAST,
            )
            print(response.text)

    The client manages HTTP connection pooling, authentication, retries,
    and graceful shutdown. Use as an async context manager for proper
    lifecycle management.
    """

    def __init__(self, config: GeminiConfig) -> None:
        self._config = config
        self._state = ClientState.CREATED
        self._http: httpx.AsyncClient | None = None
        self._request_count = 0
        self._error_count = 0
        self._last_health_check: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize HTTP client and validate connection.

        Raises:
            GeminiAuthError: If the API key is invalid.
            GeminiConnectionError: If the API is unreachable.
        """
        if self._state == ClientState.CONNECTED:
            return

        errors = self._config.validate()
        if errors:
            raise GeminiAuthError("; ".join(errors))

        self._http = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(
                connect=self._config.connect_timeout,
                read=self._config.read_timeout,
                write=self._config.read_timeout,
                pool=self._config.connect_timeout,
            ),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30,
            ),
            headers={
                "Content-Type": "application/json",
            },
        )

        # Validate connectivity with a lightweight models.list call
        try:
            await self._health_check()
            self._state = ClientState.CONNECTED
            logger.info("Gemini client connected successfully")
        except Exception:
            self._state = ClientState.DEGRADED
            logger.warning("Gemini client connected in degraded mode (health check failed)")

    async def close(self) -> None:
        """Gracefully shut down the client and release resources."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._state = ClientState.CLOSED
        logger.info(
            "Gemini client closed (requests=%d, errors=%d)",
            self._request_count,
            self._error_count,
        )

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── Properties ─────────────────────────────────────────────────

    @property
    def state(self) -> ClientState:
        """Current client lifecycle state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether the client is ready to make requests."""
        return self._state in (ClientState.CONNECTED, ClientState.DEGRADED)

    @property
    def config(self) -> GeminiConfig:
        """Client configuration (read-only)."""
        return self._config

    @property
    def stats(self) -> dict[str, Any]:
        """Client usage statistics."""
        return {
            "state": self._state.value,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._request_count
                if self._request_count > 0
                else 0.0
            ),
        }

    # ── Core API Methods ───────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        *,
        tier: ModelTier = ModelTier.FAST,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        response_mime_type: str | None = None,
    ) -> GeminiResponse:
        """Generate text content from a text prompt.

        Args:
            prompt: The text prompt.
            tier: Model tier — FAST (Flash-Lite) or SMART (Flash).
            system_instruction: Optional system instruction.
            temperature: Sampling temperature (0.0–2.0).
            max_output_tokens: Maximum tokens in the response.
            response_mime_type: Optional MIME type for structured output
                (e.g., "application/json").

        Returns:
            Parsed GeminiResponse with text and metadata.
        """
        model = self._resolve_model(tier)
        body = self._build_generate_body(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
        )
        return await self._generate_request(model, body)

    async def generate_from_audio(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
        *,
        prompt: str = "",
        tier: ModelTier = ModelTier.FAST,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        response_mime_type: str | None = None,
    ) -> GeminiResponse:
        """Generate content from raw audio input (multimodal).

        Sends raw audio bytes directly to Gemini 2 for understanding.
        This is the primary voice input path — audio in, understanding out.

        Args:
            audio_data: Raw audio bytes.
            mime_type: Audio MIME type (audio/webm, audio/wav, audio/mp3, etc.).
            prompt: Optional text prompt alongside the audio.
            tier: Model tier — FAST or SMART.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature.
            max_output_tokens: Maximum response tokens.
            response_mime_type: Optional structured output MIME type.

        Returns:
            GeminiResponse with transcription/understanding.
        """
        model = self._resolve_model(tier)

        parts: list[dict[str, Any]] = [
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(audio_data).decode("ascii"),
                }
            }
        ]
        if prompt:
            parts.append({"text": prompt})

        body = self._build_generate_body(
            contents=[{"role": "user", "parts": parts}],
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
        )
        return await self._generate_request(model, body)

    async def generate_multi_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        tier: ModelTier = ModelTier.SMART,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        response_mime_type: str | None = None,
    ) -> GeminiResponse:
        """Generate from a multi-turn conversation history.

        Args:
            messages: List of message dicts with 'role' and 'parts'.
            tier: Model tier (defaults to SMART for conversations).
            system_instruction: Optional system instruction.
            temperature: Sampling temperature.
            max_output_tokens: Maximum response tokens.
            response_mime_type: Optional structured output MIME type.

        Returns:
            GeminiResponse.
        """
        model = self._resolve_model(tier)
        body = self._build_generate_body(
            contents=messages,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
        )
        return await self._generate_request(model, body)

    async def embed(
        self,
        text: str,
        *,
        task_type: str = "SEMANTIC_SIMILARITY",
    ) -> EmbeddingResponse:
        """Generate embeddings for semantic memory.

        Args:
            text: Text to embed.
            task_type: Embedding task type (SEMANTIC_SIMILARITY,
                RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, CLASSIFICATION,
                CLUSTERING).

        Returns:
            EmbeddingResponse with the embedding vector.
        """
        self._ensure_connected()

        model = self._config.embedding_model
        url = f"/models/{model}:embedContent?key={self._config.api_key}"
        body = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self._config.embedding_dimensions,
        }

        start = time.monotonic()
        data = await self._request_with_retry("POST", url, json=body)
        latency = (time.monotonic() - start) * 1000

        embedding = data.get("embedding", {})
        values = embedding.get("values", [])

        return EmbeddingResponse(
            values=values,
            model=model,
            latency_ms=latency,
        )

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task_type: str = "SEMANTIC_SIMILARITY",
    ) -> list[EmbeddingResponse]:
        """Generate embeddings for multiple texts concurrently.

        Args:
            texts: List of texts to embed.
            task_type: Embedding task type.

        Returns:
            List of EmbeddingResponse objects.
        """
        tasks = [self.embed(text, task_type=task_type) for text in texts]
        return list(await asyncio.gather(*tasks))

    @asynccontextmanager
    async def stream(
        self,
        prompt: str,
        *,
        tier: ModelTier = ModelTier.SMART,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
    ) -> AsyncIterator[AsyncIterator[str]]:
        """Stream a text generation response.

        Yields text chunks as they arrive. Use for real-time voice responses.

        Usage::

            async with client.stream("Summarize this", tier=ModelTier.SMART) as chunks:
                async for chunk in chunks:
                    print(chunk, end="")

        Args:
            prompt: Text prompt.
            tier: Model tier.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature.
            max_output_tokens: Maximum response tokens.

        Yields:
            Text chunks as they arrive from the API.
        """
        self._ensure_connected()
        assert self._http is not None

        model = self._resolve_model(tier)
        url = f"/models/{model}:streamGenerateContent?alt=sse&key={self._config.api_key}"
        body = self._build_generate_body(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        async def _stream_chunks() -> AsyncIterator[str]:
            async with self._http.stream(  # type: ignore[union-attr]
                "POST",
                url,
                json=body,
                timeout=httpx.Timeout(
                    connect=self._config.connect_timeout,
                    read=self._config.stream_timeout,
                    write=self._config.read_timeout,
                    pool=self._config.connect_timeout,
                ),
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    self._handle_error_status(response.status_code, error_body.decode())
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        import json as _json

                        try:
                            data = _json.loads(line[6:])
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        yield part["text"]
                        except _json.JSONDecodeError:
                            continue

        yield _stream_chunks()

    # ── Health & Diagnostics ───────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Run a health check and return status information.

        Returns:
            Dict with 'healthy', 'state', 'latency_ms', and 'stats'.
        """
        start = time.monotonic()
        healthy = False
        error_msg = None

        try:
            await self._health_check()
            healthy = True
            if self._state == ClientState.DEGRADED:
                self._state = ClientState.CONNECTED
        except Exception as e:
            error_msg = str(e)
            if self._state == ClientState.CONNECTED:
                self._state = ClientState.DEGRADED

        latency = (time.monotonic() - start) * 1000

        result: dict[str, Any] = {
            "healthy": healthy,
            "state": self._state.value,
            "latency_ms": round(latency, 2),
            **self.stats,
        }
        if error_msg:
            result["error"] = error_msg

        return result

    # ── Internal Helpers ───────────────────────────────────────────

    def _resolve_model(self, tier: ModelTier) -> str:
        """Map model tier to actual model ID."""
        if tier == ModelTier.FAST:
            return self._config.flash_lite_model
        return self._config.flash_model

    def _build_generate_body(
        self,
        contents: list[dict[str, Any]],
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        response_mime_type: str | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        """Build the request body for generateContent.

        Args:
            contents: Conversation content parts.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature.
            max_output_tokens: Maximum output tokens.
            response_mime_type: Optional MIME type for structured output.
            thinking_budget: If set, controls thinking token budget for
                models that support it (e.g. gemini-2.5-flash).
                Set to 0 to disable thinking entirely.
        """
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        if response_mime_type:
            body["generationConfig"]["responseMimeType"] = response_mime_type
            # Auto-disable thinking for structured output to prevent thinking
            # tokens from consuming the output budget (affects gemini-2.5+)
            if thinking_budget is None:
                thinking_budget = 0
        if thinking_budget is not None:
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": thinking_budget,
            }
        return body

    async def _generate_request(self, model: str, body: dict[str, Any]) -> GeminiResponse:
        """Execute a generateContent request with retries."""
        self._ensure_connected()
        url = f"/models/{model}:generateContent?key={self._config.api_key}"

        start = time.monotonic()
        data = await self._request_with_retry("POST", url, json=body)
        latency = (time.monotonic() - start) * 1000

        return GeminiResponse.from_api_response(data, model, latency)

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request with exponential backoff retries.

        Retries on 429, 500, 502, 503, 504, and connection errors.
        """
        assert self._http is not None
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                self._request_count += 1
                response = await self._http.request(method, url, **kwargs)

                if response.status_code == 200:
                    return response.json()  # type: ignore[no-any-return]

                body_text = response.text
                self._error_count += 1

                # Non-retryable errors
                if response.status_code == 401:
                    raise GeminiAuthError()
                if response.status_code == 403:
                    raise GeminiAuthError("API key lacks required permissions")
                if response.status_code == 400:
                    raise GeminiModelError(
                        f"Bad request: {body_text}",
                        status_code=400,
                        retryable=False,
                    )

                # Retryable errors
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else None
                    last_error = GeminiRateLimitError(retry_after=delay)
                elif response.status_code >= 500:
                    last_error = GeminiError(
                        f"Server error {response.status_code}: {body_text}",
                        status_code=response.status_code,
                        retryable=True,
                    )
                else:
                    raise GeminiError(
                        f"Unexpected status {response.status_code}: {body_text}",
                        status_code=response.status_code,
                    )

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                self._error_count += 1
                last_error = GeminiConnectionError(str(e))
            except GeminiError:
                raise
            except Exception as e:
                self._error_count += 1
                raise GeminiError(f"Unexpected error: {e}") from e

            # Backoff before retry
            if attempt < self._config.max_retries:
                delay = min(
                    self._config.retry_backoff_base * (2 ** attempt),
                    self._config.retry_backoff_max,
                )
                # For rate limits, use server-suggested delay if available
                if isinstance(last_error, GeminiRateLimitError) and last_error.retry_after:
                    delay = max(delay, last_error.retry_after)

                logger.warning(
                    "Request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._config.max_retries + 1,
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        assert last_error is not None
        raise last_error

    async def _health_check(self) -> None:
        """Lightweight health check via models.list."""
        assert self._http is not None
        response = await self._http.get(
            f"/models?key={self._config.api_key}&pageSize=1",
        )
        if response.status_code == 401:
            raise GeminiAuthError()
        if response.status_code != 200:
            raise GeminiConnectionError(f"Health check returned {response.status_code}")
        self._last_health_check = time.monotonic()

    def _ensure_connected(self) -> None:
        """Raise if client is not in a usable state."""
        if self._state == ClientState.CLOSED:
            raise GeminiError("Client is closed", retryable=False)
        if self._state == ClientState.CREATED:
            raise GeminiError("Client not connected — call connect() or use async with", retryable=False)

    def _handle_error_status(self, status_code: int, body: str) -> None:
        """Raise the appropriate error for a non-200 status code."""
        if status_code == 401:
            raise GeminiAuthError()
        if status_code == 429:
            raise GeminiRateLimitError()
        if status_code >= 500:
            raise GeminiError(f"Server error {status_code}", status_code=status_code, retryable=True)
        raise GeminiError(f"HTTP {status_code}: {body}", status_code=status_code)
