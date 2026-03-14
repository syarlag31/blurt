"""Blurt configuration settings.

Centralized configuration for all Blurt services. Supports environment
variables, .env files, and programmatic overrides. Cloud-first with
local-only option.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Self


class DeploymentMode(str, Enum):
    """Deployment mode — cloud-first or local-only."""

    CLOUD = "cloud"
    LOCAL = "local"


class GeminiModel(str, Enum):
    """Gemini model identifiers for the two-model strategy.

    Flash-Lite: classification, extraction, emotion detection (cheap, fast).
    Flash: reasoning, insights, journaling, Q&A (smarter, costlier).
    """

    FLASH_LITE = "gemini-2.5-flash-lite"
    FLASH = "gemini-2.5-flash"


# Default Gemini API endpoint
_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Default timeouts (seconds)
_DEFAULT_CONNECT_TIMEOUT = 10.0
_DEFAULT_READ_TIMEOUT = 60.0
_DEFAULT_STREAM_TIMEOUT = 120.0

# Default retry settings
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF_BASE = 0.5
_DEFAULT_RETRY_BACKOFF_MAX = 30.0


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    """Configuration for the Gemini API client.

    Attributes:
        api_key: Gemini API key. Read from GEMINI_API_KEY env var if not set.
        base_url: API base URL. Overridable for local proxies.
        flash_lite_model: Model ID for classification/extraction tasks.
        flash_model: Model ID for reasoning/insight tasks.
        connect_timeout: TCP connection timeout in seconds.
        read_timeout: Response read timeout in seconds.
        stream_timeout: Streaming response timeout in seconds.
        max_retries: Maximum number of retry attempts for transient failures.
        retry_backoff_base: Base delay for exponential backoff (seconds).
        retry_backoff_max: Maximum backoff delay cap (seconds).
        embedding_model: Model ID for generating embeddings.
        embedding_dimensions: Dimensionality for embedding vectors.
    """

    api_key: str = ""
    base_url: str = _DEFAULT_GEMINI_BASE_URL
    flash_lite_model: str = GeminiModel.FLASH_LITE.value
    flash_model: str = GeminiModel.FLASH.value
    connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT
    read_timeout: float = _DEFAULT_READ_TIMEOUT
    stream_timeout: float = _DEFAULT_STREAM_TIMEOUT
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_backoff_base: float = _DEFAULT_RETRY_BACKOFF_BASE
    retry_backoff_max: float = _DEFAULT_RETRY_BACKOFF_MAX
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    @classmethod
    def from_env(cls, **overrides: object) -> Self:
        """Build config from environment variables with optional overrides.

        Environment variables:
            GEMINI_API_KEY: API key (required for cloud mode).
            GEMINI_BASE_URL: Override API base URL.
            GEMINI_CONNECT_TIMEOUT: Connection timeout seconds.
            GEMINI_READ_TIMEOUT: Read timeout seconds.
            GEMINI_STREAM_TIMEOUT: Stream timeout seconds.
            GEMINI_MAX_RETRIES: Max retry count.
        """
        env_map: dict[str, object] = {}

        if api_key := os.environ.get("GEMINI_API_KEY"):
            env_map["api_key"] = api_key
        if base_url := os.environ.get("GEMINI_BASE_URL"):
            env_map["base_url"] = base_url
        if connect_timeout := os.environ.get("GEMINI_CONNECT_TIMEOUT"):
            env_map["connect_timeout"] = float(connect_timeout)
        if read_timeout := os.environ.get("GEMINI_READ_TIMEOUT"):
            env_map["read_timeout"] = float(read_timeout)
        if stream_timeout := os.environ.get("GEMINI_STREAM_TIMEOUT"):
            env_map["stream_timeout"] = float(stream_timeout)
        if max_retries := os.environ.get("GEMINI_MAX_RETRIES"):
            env_map["max_retries"] = int(max_retries)

        # Overrides take precedence over env vars
        env_map.update(overrides)
        return cls(**env_map)  # type: ignore[arg-type]

    def validate(self) -> list[str]:
        """Validate configuration, returning a list of error messages."""
        errors: list[str] = []
        if not self.api_key:
            errors.append("api_key is required (set GEMINI_API_KEY)")
        if self.connect_timeout <= 0:
            errors.append("connect_timeout must be positive")
        if self.read_timeout <= 0:
            errors.append("read_timeout must be positive")
        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")
        return errors


@dataclass(frozen=True, slots=True)
class WebSocketConfig:
    """WebSocket server configuration."""

    max_message_size: int = 1024 * 1024  # 1MB max per WS message
    ping_interval: float = 30.0  # seconds
    ping_timeout: float = 10.0  # seconds
    idle_timeout: float = 300.0  # 5 min idle disconnect
    max_audio_buffer_bytes: int = 10 * 1024 * 1024  # 10MB max buffered audio per session
    audio_chunk_max_bytes: int = 32 * 1024  # 32KB max per audio chunk
    max_concurrent_sessions: int = 1000

    @classmethod
    def from_env(cls, **overrides: object) -> Self:
        """Build from environment."""
        env_map: dict[str, object] = {}
        if v := os.environ.get("BLURT_WS_MAX_MESSAGE_SIZE"):
            env_map["max_message_size"] = int(v)
        if v := os.environ.get("BLURT_WS_PING_INTERVAL"):
            env_map["ping_interval"] = float(v)
        if v := os.environ.get("BLURT_WS_IDLE_TIMEOUT"):
            env_map["idle_timeout"] = float(v)
        if v := os.environ.get("BLURT_WS_MAX_AUDIO_BUFFER"):
            env_map["max_audio_buffer_bytes"] = int(v)
        env_map.update(overrides)
        return cls(**env_map)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BlurtConfig:
    """Top-level Blurt configuration."""

    mode: DeploymentMode = DeploymentMode.CLOUD
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    data_dir: Path = field(default_factory=lambda: Path.home() / ".blurt")
    encryption_enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @classmethod
    def from_env(cls, **overrides: object) -> Self:
        """Build full config from environment."""
        mode_str = os.environ.get("BLURT_MODE", "cloud").lower()
        mode = DeploymentMode(mode_str) if mode_str in ("cloud", "local") else DeploymentMode.CLOUD

        data_dir_str = os.environ.get("BLURT_DATA_DIR")
        data_dir = Path(data_dir_str) if data_dir_str else Path.home() / ".blurt"

        encryption = os.environ.get("BLURT_ENCRYPTION", "true").lower() != "false"

        gemini_overrides = {
            k: v for k, v in overrides.items()
            if k.startswith("gemini_") or k == "api_key"
        }
        remaining = {
            k: v for k, v in overrides.items()
            if k not in gemini_overrides
        }

        host = os.environ.get("BLURT_HOST", "0.0.0.0")
        port = int(os.environ.get("BLURT_PORT", "8000"))
        debug = os.environ.get("BLURT_DEBUG", "false").lower() == "true"

        ws_overrides = {
            k.removeprefix("ws_"): v for k, v in overrides.items()
            if k.startswith("ws_")
        }

        return cls(
            mode=remaining.get("mode", mode),  # type: ignore[arg-type]
            gemini=GeminiConfig.from_env(**gemini_overrides),
            websocket=WebSocketConfig.from_env(**ws_overrides),
            data_dir=remaining.get("data_dir", data_dir),  # type: ignore[arg-type]
            encryption_enabled=remaining.get("encryption_enabled", encryption),  # type: ignore[arg-type]
            host=remaining.get("host", host),  # type: ignore[arg-type]
            port=remaining.get("port", port),  # type: ignore[arg-type]
            debug=remaining.get("debug", debug),  # type: ignore[arg-type]
        )
