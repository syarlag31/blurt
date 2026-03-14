"""Service provider registry for local-only mode support.

Detects deployment mode (cloud vs local) and provides the correct
implementations of all external service clients. In local mode,
all external services (LLM, Google Calendar, Notion) use local/mock
implementations with full feature parity and zero data leakage.

Environment detection:
    BLURT_MODE=local  → local-only mode (no external API calls)
    BLURT_MODE=cloud  → cloud mode (default, uses real APIs)

Usage::

    from blurt.services.provider import ServiceProvider, get_provider

    provider = get_provider()  # auto-detects mode from env
    llm = provider.llm_client()
    calendar = provider.calendar_client()
    notion = provider.notion_client()
    embeddings = provider.embedding_provider()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from blurt.clients.embeddings import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    LocalEmbeddingProvider,
    MockEmbeddingProvider,
)
from blurt.config.settings import BlurtConfig, DeploymentMode

logger = logging.getLogger(__name__)


# ── Abstract Client Protocols ──────────────────────────────────────


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM clients (cloud Gemini or local mock)."""

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate text from a prompt."""
        ...

    async def generate_from_audio(
        self, audio_data: bytes, mime_type: str, **kwargs: Any
    ) -> Any:
        """Generate content from raw audio input."""
        ...

    async def connect(self) -> None:
        """Initialize the client."""
        ...

    async def close(self) -> None:
        """Shut down the client."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the client is ready."""
        ...


@runtime_checkable
class CalendarClient(Protocol):
    """Protocol for calendar clients (Google Calendar or local mock)."""

    def create_event(self, event: Any, **kwargs: Any) -> Any:
        """Create a calendar event."""
        ...

    def update_event(self, event: Any, **kwargs: Any) -> Any:
        """Update a calendar event."""
        ...

    def create_or_update_event(self, event: Any, **kwargs: Any) -> Any:
        """Create or update depending on whether event has an external ID."""
        ...


@runtime_checkable
class NotionClient(Protocol):
    """Protocol for Notion clients (real API or local mock)."""

    async def create_page(self, database_id: str, properties: dict[str, Any]) -> dict[str, Any]:
        """Create a page in a Notion database."""
        ...

    async def update_page(self, page_id: str, properties: dict[str, Any]) -> dict[str, Any]:
        """Update an existing Notion page."""
        ...

    async def query_database(
        self, database_id: str, filter_params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Query a Notion database."""
        ...

    async def connect(self) -> None:
        """Initialize the client."""
        ...

    async def close(self) -> None:
        """Shut down the client."""
        ...


# ── Environment Detection ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Detected environment information for mode selection."""

    mode: DeploymentMode
    has_gemini_key: bool
    has_google_credentials: bool
    has_notion_token: bool
    detection_source: str  # "env_var", "auto_detect", "explicit"

    @property
    def is_local(self) -> bool:
        return self.mode == DeploymentMode.LOCAL

    @property
    def is_cloud(self) -> bool:
        return self.mode == DeploymentMode.CLOUD

    @property
    def missing_cloud_credentials(self) -> list[str]:
        """List credentials missing for full cloud mode."""
        missing = []
        if not self.has_gemini_key:
            missing.append("GEMINI_API_KEY")
        if not self.has_google_credentials:
            missing.append("BLURT_GOOGLE_CLIENT_ID / GOOGLE_APPLICATION_CREDENTIALS")
        if not self.has_notion_token:
            missing.append("NOTION_API_TOKEN")
        return missing


def detect_environment() -> EnvironmentInfo:
    """Detect deployment mode and available credentials from environment.

    Priority:
        1. Explicit BLURT_MODE env var (cloud/local)
        2. Auto-detect: if no API keys are set, default to local mode
        3. Fall back to cloud mode if keys are present
    """
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_google = bool(
        os.environ.get("BLURT_GOOGLE_CLIENT_ID")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    has_notion = bool(os.environ.get("NOTION_API_TOKEN"))

    explicit_mode = os.environ.get("BLURT_MODE", "").strip().lower()

    if explicit_mode == "local":
        return EnvironmentInfo(
            mode=DeploymentMode.LOCAL,
            has_gemini_key=has_gemini,
            has_google_credentials=has_google,
            has_notion_token=has_notion,
            detection_source="env_var",
        )

    if explicit_mode == "cloud":
        return EnvironmentInfo(
            mode=DeploymentMode.CLOUD,
            has_gemini_key=has_gemini,
            has_google_credentials=has_google,
            has_notion_token=has_notion,
            detection_source="env_var",
        )

    # Auto-detect: if no cloud credentials at all, default local
    if not has_gemini and not has_google and not has_notion:
        logger.info(
            "No cloud API credentials detected — defaulting to local-only mode. "
            "Set BLURT_MODE=cloud to override."
        )
        return EnvironmentInfo(
            mode=DeploymentMode.LOCAL,
            has_gemini_key=False,
            has_google_credentials=False,
            has_notion_token=False,
            detection_source="auto_detect",
        )

    return EnvironmentInfo(
        mode=DeploymentMode.CLOUD,
        has_gemini_key=has_gemini,
        has_google_credentials=has_google,
        has_notion_token=has_notion,
        detection_source="auto_detect",
    )


# ── Service Provider ───────────────────────────────────────────────


class ServiceProvider:
    """Central registry that provides the correct service implementations
    based on deployment mode.

    In cloud mode: returns real API clients (Gemini, Google Calendar, Notion).
    In local mode: returns local/mock implementations with feature parity.

    Usage::

        provider = ServiceProvider(config)
        llm = provider.llm_client()
        calendar = provider.calendar_client()
        embeddings = provider.embedding_provider()
    """

    def __init__(
        self,
        config: BlurtConfig | None = None,
        env_info: EnvironmentInfo | None = None,
    ) -> None:
        self._config = config or BlurtConfig.from_env()
        self._env_info = env_info or detect_environment()

        # Override config mode if environment detection says local
        if self._env_info.is_local and self._config.mode != DeploymentMode.LOCAL:
            # Re-create config with local mode
            self._config = BlurtConfig(
                mode=DeploymentMode.LOCAL,
                gemini=self._config.gemini,
                websocket=self._config.websocket,
                data_dir=self._config.data_dir,
                encryption_enabled=self._config.encryption_enabled,
                host=self._config.host,
                port=self._config.port,
                debug=self._config.debug,
            )

        # Cached instances
        self._llm: LLMClient | None = None
        self._calendar: CalendarClient | None = None
        self._notion: NotionClient | None = None
        self._embeddings: EmbeddingProvider | None = None

        logger.info(
            "ServiceProvider initialized (mode=%s, source=%s)",
            self.mode.value,
            self._env_info.detection_source,
        )

    @property
    def mode(self) -> DeploymentMode:
        """Current deployment mode."""
        return self._config.mode

    @property
    def is_local(self) -> bool:
        """Whether running in local-only mode."""
        return self._config.mode == DeploymentMode.LOCAL

    @property
    def config(self) -> BlurtConfig:
        """Current configuration."""
        return self._config

    @property
    def environment(self) -> EnvironmentInfo:
        """Detected environment info."""
        return self._env_info

    def llm_client(self) -> LLMClient:
        """Get the LLM client for the current mode.

        Cloud: GeminiClient (real API)
        Local: LocalLLMClient (mock/local inference)
        """
        if self._llm is not None:
            return self._llm

        if self.is_local:
            from blurt.clients.local_llm import LocalLLMClient

            self._llm = LocalLLMClient(data_dir=self._config.data_dir)
            logger.info("Using local LLM client (no external API calls)")
        else:
            from blurt.clients.gemini import GeminiClient

            self._llm = GeminiClient(self._config.gemini)  # type: ignore[assignment]
            logger.info("Using Gemini cloud LLM client")

        assert self._llm is not None
        return self._llm

    def calendar_client(self) -> CalendarClient:
        """Get the calendar client for the current mode.

        Cloud: GoogleCalendarClient (real Google API)
        Local: LocalCalendarClient (file-backed local store)
        """
        if self._calendar is not None:
            return self._calendar

        if self.is_local:
            from blurt.integrations.google_calendar.local_client import LocalCalendarClient

            self._calendar = LocalCalendarClient(data_dir=self._config.data_dir)
            logger.info("Using local calendar client (no Google API calls)")
        else:
            from blurt.integrations.google_calendar.client import GoogleCalendarClient
            from blurt.integrations.google_calendar.auth import GoogleCalendarAuth

            auth = GoogleCalendarAuth()
            self._calendar = GoogleCalendarClient(auth)  # type: ignore[assignment]
            logger.info("Using Google Calendar cloud client")

        assert self._calendar is not None
        return self._calendar

    def notion_client(self) -> NotionClient:
        """Get the Notion client for the current mode.

        Cloud: NotionAPIClient (real Notion API)
        Local: LocalNotionClient (file-backed local store)
        """
        if self._notion is not None:
            return self._notion

        if self.is_local:
            from blurt.integrations.notion.local_client import LocalNotionClient

            self._notion = LocalNotionClient(data_dir=self._config.data_dir)
            logger.info("Using local Notion client (no Notion API calls)")
        else:
            from blurt.integrations.notion.client import NotionAPIClient

            token = os.environ.get("NOTION_API_TOKEN", "")
            self._notion = NotionAPIClient(api_token=token)
            logger.info("Using Notion cloud client")

        return self._notion

    def embedding_provider(self) -> EmbeddingProvider:
        """Get the embedding provider for the current mode.

        Cloud: GeminiEmbeddingProvider (Gemini API)
        Local: LocalEmbeddingProvider (sentence-transformers) or
               MockEmbeddingProvider (deterministic, no model download)
        """
        if self._embeddings is not None:
            return self._embeddings

        if self.is_local:
            try:
                # Try local embedding model first
                self._embeddings = LocalEmbeddingProvider()
                logger.info("Using local embedding provider (sentence-transformers)")
            except ImportError:
                # Fall back to mock if sentence-transformers not installed
                self._embeddings = MockEmbeddingProvider()
                logger.info(
                    "Using mock embedding provider "
                    "(install sentence-transformers for better local embeddings)"
                )
        else:
            api_key = self._config.gemini.api_key
            self._embeddings = GeminiEmbeddingProvider(api_key=api_key)
            logger.info("Using Gemini cloud embedding provider")

        return self._embeddings

    async def startup(self) -> None:
        """Initialize all service clients that need async startup."""
        llm = self.llm_client()
        if hasattr(llm, "connect"):
            await llm.connect()

        notion = self.notion_client()
        if hasattr(notion, "connect"):
            await notion.connect()

        logger.info("All service clients started (mode=%s)", self.mode.value)

    async def shutdown(self) -> None:
        """Gracefully shut down all service clients."""
        if self._llm is not None and hasattr(self._llm, "close"):
            await self._llm.close()
        if self._notion is not None and hasattr(self._notion, "close"):
            await self._notion.close()

        logger.info("All service clients shut down")

    def status(self) -> dict[str, Any]:
        """Return status information about all service providers."""
        return {
            "mode": self.mode.value,
            "detection_source": self._env_info.detection_source,
            "is_local": self.is_local,
            "services": {
                "llm": {
                    "type": type(self._llm).__name__ if self._llm else "not_initialized",
                    "connected": (
                        self._llm.is_connected if self._llm and hasattr(self._llm, "is_connected") else None
                    ),
                },
                "calendar": {
                    "type": type(self._calendar).__name__ if self._calendar else "not_initialized",
                },
                "notion": {
                    "type": type(self._notion).__name__ if self._notion else "not_initialized",
                },
                "embeddings": {
                    "type": type(self._embeddings).__name__ if self._embeddings else "not_initialized",
                    "dimension": self._embeddings.dimension if self._embeddings else None,
                },
            },
            "missing_cloud_credentials": self._env_info.missing_cloud_credentials,
        }


# ── Module-level singleton ────────────────────────────────────────

_provider: ServiceProvider | None = None


def get_provider(config: BlurtConfig | None = None) -> ServiceProvider:
    """Get or create the global ServiceProvider singleton.

    Args:
        config: Optional config override. If provided on first call,
                uses this config. Ignored on subsequent calls.

    Returns:
        The global ServiceProvider instance.
    """
    global _provider
    if _provider is None:
        _provider = ServiceProvider(config=config)
    return _provider


def reset_provider() -> None:
    """Reset the global provider (useful for testing)."""
    global _provider
    _provider = None
