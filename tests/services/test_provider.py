"""Tests for the ServiceProvider and local-only mode detection."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from blurt.clients.local_llm import LocalLLMClient, LocalLLMResponse
from blurt.config.settings import BlurtConfig, DeploymentMode
from blurt.integrations.google_calendar.local_client import LocalCalendarClient
from blurt.integrations.notion.local_client import LocalNotionClient
from blurt.services.provider import (
    EnvironmentInfo,
    ServiceProvider,
    detect_environment,
    get_provider,
    reset_provider,
)


# ── Environment Detection Tests ──────────────────────────────────


class TestDetectEnvironment:
    """Tests for detect_environment() function."""

    def test_explicit_local_mode(self):
        """BLURT_MODE=local forces local mode regardless of available keys."""
        with patch.dict(os.environ, {"BLURT_MODE": "local", "GEMINI_API_KEY": "test-key"}, clear=False):
            env = detect_environment()
            assert env.mode == DeploymentMode.LOCAL
            assert env.is_local is True
            assert env.is_cloud is False
            assert env.detection_source == "env_var"
            # Key is detected but mode is still local
            assert env.has_gemini_key is True

    def test_explicit_cloud_mode(self):
        """BLURT_MODE=cloud forces cloud mode."""
        with patch.dict(os.environ, {"BLURT_MODE": "cloud"}, clear=False):
            # Clear any keys that might be set
            env_clean = {k: v for k, v in os.environ.items() if k not in (
                "GEMINI_API_KEY", "BLURT_GOOGLE_CLIENT_ID",
                "GOOGLE_APPLICATION_CREDENTIALS", "NOTION_API_TOKEN"
            )}
            with patch.dict(os.environ, env_clean, clear=True):
                os.environ["BLURT_MODE"] = "cloud"
                env = detect_environment()
                assert env.mode == DeploymentMode.CLOUD
                assert env.detection_source == "env_var"

    def test_auto_detect_local_no_credentials(self):
        """Auto-detect defaults to local when no API keys are set."""
        env_clean = {k: v for k, v in os.environ.items() if k not in (
            "BLURT_MODE", "GEMINI_API_KEY", "BLURT_GOOGLE_CLIENT_ID",
            "GOOGLE_APPLICATION_CREDENTIALS", "NOTION_API_TOKEN"
        )}
        with patch.dict(os.environ, env_clean, clear=True):
            env = detect_environment()
            assert env.mode == DeploymentMode.LOCAL
            assert env.detection_source == "auto_detect"
            assert env.has_gemini_key is False
            assert env.has_google_credentials is False
            assert env.has_notion_token is False

    def test_auto_detect_cloud_with_gemini_key(self):
        """Auto-detect uses cloud mode when Gemini API key is set."""
        env_clean = {k: v for k, v in os.environ.items() if k not in (
            "BLURT_MODE", "BLURT_GOOGLE_CLIENT_ID",
            "GOOGLE_APPLICATION_CREDENTIALS", "NOTION_API_TOKEN"
        )}
        with patch.dict(os.environ, env_clean, clear=True):
            os.environ["GEMINI_API_KEY"] = "test-key-123"
            env = detect_environment()
            assert env.mode == DeploymentMode.CLOUD
            assert env.detection_source == "auto_detect"
            assert env.has_gemini_key is True

    def test_missing_cloud_credentials(self):
        """missing_cloud_credentials lists what's missing for full cloud."""
        env = EnvironmentInfo(
            mode=DeploymentMode.CLOUD,
            has_gemini_key=True,
            has_google_credentials=False,
            has_notion_token=False,
            detection_source="env_var",
        )
        missing = env.missing_cloud_credentials
        assert "GEMINI_API_KEY" not in missing
        assert "BLURT_GOOGLE_CLIENT_ID / GOOGLE_APPLICATION_CREDENTIALS" in missing
        assert "NOTION_API_TOKEN" in missing

    def test_no_missing_credentials(self):
        """No missing credentials when everything is set."""
        env = EnvironmentInfo(
            mode=DeploymentMode.CLOUD,
            has_gemini_key=True,
            has_google_credentials=True,
            has_notion_token=True,
            detection_source="env_var",
        )
        assert env.missing_cloud_credentials == []


# ── ServiceProvider Tests ────────────────────────────────────────


class TestServiceProvider:
    """Tests for ServiceProvider mode switching."""

    def _make_local_provider(self, tmp_path: Path) -> ServiceProvider:
        """Create a provider in local mode."""
        env = EnvironmentInfo(
            mode=DeploymentMode.LOCAL,
            has_gemini_key=False,
            has_google_credentials=False,
            has_notion_token=False,
            detection_source="env_var",
        )
        config = BlurtConfig(
            mode=DeploymentMode.LOCAL,
            data_dir=tmp_path,
        )
        return ServiceProvider(config=config, env_info=env)

    def _make_cloud_provider(self, tmp_path: Path) -> ServiceProvider:
        """Create a provider in cloud mode."""
        env = EnvironmentInfo(
            mode=DeploymentMode.CLOUD,
            has_gemini_key=True,
            has_google_credentials=True,
            has_notion_token=True,
            detection_source="env_var",
        )
        config = BlurtConfig(
            mode=DeploymentMode.CLOUD,
            data_dir=tmp_path,
        )
        return ServiceProvider(config=config, env_info=env)

    def test_local_mode_properties(self, tmp_path):
        provider = self._make_local_provider(tmp_path)
        assert provider.is_local is True
        assert provider.mode == DeploymentMode.LOCAL

    def test_cloud_mode_properties(self, tmp_path):
        provider = self._make_cloud_provider(tmp_path)
        assert provider.is_local is False
        assert provider.mode == DeploymentMode.CLOUD

    def test_local_llm_client(self, tmp_path):
        """Local mode returns LocalLLMClient."""
        provider = self._make_local_provider(tmp_path)
        llm = provider.llm_client()
        assert isinstance(llm, LocalLLMClient)

    def test_local_calendar_client(self, tmp_path):
        """Local mode returns LocalCalendarClient."""
        provider = self._make_local_provider(tmp_path)
        cal = provider.calendar_client()
        assert isinstance(cal, LocalCalendarClient)

    def test_local_notion_client(self, tmp_path):
        """Local mode returns LocalNotionClient."""
        provider = self._make_local_provider(tmp_path)
        notion = provider.notion_client()
        assert isinstance(notion, LocalNotionClient)

    def test_local_embedding_provider(self, tmp_path):
        """Local mode returns a local/mock embedding provider."""
        provider = self._make_local_provider(tmp_path)
        embeddings = provider.embedding_provider()
        # Should be either LocalEmbeddingProvider or MockEmbeddingProvider
        assert hasattr(embeddings, "embed")
        assert hasattr(embeddings, "dimension")
        assert embeddings.dimension > 0

    def test_clients_are_cached(self, tmp_path):
        """Same client instance returned on repeated calls."""
        provider = self._make_local_provider(tmp_path)
        llm1 = provider.llm_client()
        llm2 = provider.llm_client()
        assert llm1 is llm2

        cal1 = provider.calendar_client()
        cal2 = provider.calendar_client()
        assert cal1 is cal2

    def test_status_report(self, tmp_path):
        """Status returns mode and service info."""
        provider = self._make_local_provider(tmp_path)
        # Initialize some clients
        provider.llm_client()
        provider.calendar_client()

        status = provider.status()
        assert status["mode"] == "local"
        assert status["is_local"] is True
        assert status["services"]["llm"]["type"] == "LocalLLMClient"
        assert status["services"]["calendar"]["type"] == "LocalCalendarClient"
        assert status["services"]["notion"]["type"] == "not_initialized"

    def test_env_detection_overrides_config_mode(self, tmp_path):
        """If env says local but config says cloud, provider uses local."""
        env = EnvironmentInfo(
            mode=DeploymentMode.LOCAL,
            has_gemini_key=False,
            has_google_credentials=False,
            has_notion_token=False,
            detection_source="auto_detect",
        )
        config = BlurtConfig(mode=DeploymentMode.CLOUD, data_dir=tmp_path)
        provider = ServiceProvider(config=config, env_info=env)
        assert provider.is_local is True
        assert provider.mode == DeploymentMode.LOCAL

    @pytest.mark.asyncio
    async def test_startup_shutdown(self, tmp_path):
        """Startup and shutdown lifecycle works for local mode."""
        provider = self._make_local_provider(tmp_path)
        await provider.startup()

        llm = provider.llm_client()
        assert llm.is_connected is True

        await provider.shutdown()
        assert llm.is_connected is False


# ── LocalLLMClient Tests ─────────────────────────────────────────


class TestLocalLLMClient:
    """Tests for the local LLM client."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        client = LocalLLMClient()
        assert client.is_connected is False

        await client.connect()
        assert client.is_connected is True

        await client.close()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with LocalLLMClient() as client:
            assert client.is_connected is True
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_generate_text(self):
        async with LocalLLMClient() as client:
            response = await client.generate("Hello world")
            assert isinstance(response, LocalLLMResponse)
            assert response.text
            assert response.model == "local-mock"
            assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_json_classification(self):
        """JSON mode with classification system instruction."""
        async with LocalLLMClient() as client:
            response = await client.generate(
                "I need to schedule a meeting tomorrow",
                system_instruction="Classify the intent of this text",
                response_mime_type="application/json",
            )
            import json
            data = json.loads(response.text)
            assert "intent" in data
            assert "confidence" in data
            assert data["confidence"] > 0

    @pytest.mark.asyncio
    async def test_generate_json_entity_extraction(self):
        """JSON mode with entity extraction."""
        async with LocalLLMClient() as client:
            response = await client.generate(
                "Meeting with John at Google headquarters",
                system_instruction="Extract entities from this text",
                response_mime_type="application/json",
            )
            import json
            data = json.loads(response.text)
            assert "entities" in data

    @pytest.mark.asyncio
    async def test_generate_json_emotion(self):
        """JSON mode with emotion detection."""
        async with LocalLLMClient() as client:
            response = await client.generate(
                "I'm feeling really happy today!",
                system_instruction="Detect the emotion in this text",
                response_mime_type="application/json",
            )
            import json
            data = json.loads(response.text)
            assert "primary_emotion" in data
            assert data["primary_emotion"] == "joy"

    @pytest.mark.asyncio
    async def test_generate_from_audio(self):
        """Audio input returns acknowledgment in local mode."""
        async with LocalLLMClient() as client:
            response = await client.generate_from_audio(
                b"\x00" * 16000,
                "audio/webm",
                prompt="What did I say?",
            )
            assert "16000 bytes" in response.text
            assert "audio/webm" in response.text

    @pytest.mark.asyncio
    async def test_health_check(self):
        async with LocalLLMClient() as client:
            health = await client.health_check()
            assert health["healthy"] is True
            assert health["mode"] == "local"

    @pytest.mark.asyncio
    async def test_intent_classification_task(self):
        """Classifies 'todo' keyword as task intent."""
        async with LocalLLMClient() as client:
            response = await client.generate(
                "todo: buy groceries",
                system_instruction="Classify the intent",
                response_mime_type="application/json",
            )
            import json
            data = json.loads(response.text)
            assert data["intent"] == "task"

    @pytest.mark.asyncio
    async def test_intent_classification_event(self):
        """Classifies 'meeting' keyword as event intent."""
        async with LocalLLMClient() as client:
            response = await client.generate(
                "meeting with sarah at 3pm",
                system_instruction="Classify the intent",
                response_mime_type="application/json",
            )
            import json
            data = json.loads(response.text)
            assert data["intent"] == "event"


# ── LocalCalendarClient Tests ────────────────────────────────────


class TestLocalCalendarClient:
    """Tests for the local calendar client."""

    def test_create_event(self, tmp_path):
        from dataclasses import dataclass, field
        from datetime import datetime

        @dataclass
        class FakeEvent:
            title: str = "Test Meeting"
            start_time: datetime = field(default_factory=datetime.now)
            end_time: datetime | None = None
            description: str = ""
            google_calendar_id: str | None = None
            synced_at: datetime | None = None

        client = LocalCalendarClient(data_dir=tmp_path)
        event = FakeEvent(title="Test Meeting")
        result = client.create_event(event)

        assert result.google_calendar_id is not None
        assert result.google_calendar_id.startswith("local_")
        assert result.synced_at is not None

    def test_update_event(self, tmp_path):
        from dataclasses import dataclass, field
        from datetime import datetime

        @dataclass
        class FakeEvent:
            title: str = "Test"
            start_time: datetime = field(default_factory=datetime.now)
            end_time: datetime | None = None
            description: str = ""
            google_calendar_id: str | None = None
            synced_at: datetime | None = None

        client = LocalCalendarClient(data_dir=tmp_path)
        event = FakeEvent(title="Original")
        client.create_event(event)

        event.title = "Updated"
        result = client.update_event(event)
        assert result.title == "Updated"

    def test_create_or_update(self, tmp_path):
        from dataclasses import dataclass, field
        from datetime import datetime

        @dataclass
        class FakeEvent:
            title: str = "Test"
            start_time: datetime = field(default_factory=datetime.now)
            end_time: datetime | None = None
            description: str = ""
            google_calendar_id: str | None = None
            synced_at: datetime | None = None

        client = LocalCalendarClient(data_dir=tmp_path)

        # Without ID → create
        event = FakeEvent(title="New Event")
        result = client.create_or_update_event(event)
        assert result.google_calendar_id is not None

        # With ID → update
        event.title = "Updated Event"
        result = client.create_or_update_event(event)
        assert result.title == "Updated Event"

    def test_list_events(self, tmp_path):
        from dataclasses import dataclass, field
        from datetime import datetime

        @dataclass
        class FakeEvent:
            title: str = "Test"
            start_time: datetime = field(default_factory=datetime.now)
            end_time: datetime | None = None
            description: str = ""
            google_calendar_id: str | None = None
            synced_at: datetime | None = None

        client = LocalCalendarClient(data_dir=tmp_path)
        client.create_event(FakeEvent(title="Event 1"))
        client.create_event(FakeEvent(title="Event 2"))

        events = client.list_events()
        assert len(events) == 2

    def test_delete_event(self, tmp_path):
        from dataclasses import dataclass, field
        from datetime import datetime

        @dataclass
        class FakeEvent:
            title: str = "Test"
            start_time: datetime = field(default_factory=datetime.now)
            end_time: datetime | None = None
            description: str = ""
            google_calendar_id: str | None = None
            synced_at: datetime | None = None

        client = LocalCalendarClient(data_dir=tmp_path)
        event = FakeEvent(title="To Delete")
        client.create_event(event)

        assert event.google_calendar_id is not None
        assert client.delete_event(event.google_calendar_id) is True
        assert client.list_events() == []
        assert client.delete_event("nonexistent") is False


# ── LocalNotionClient Tests ──────────────────────────────────────


class TestLocalNotionClient:
    """Tests for the local Notion client."""

    @pytest.mark.asyncio
    async def test_create_page(self, tmp_path):
        client = LocalNotionClient(data_dir=tmp_path)
        await client.connect()

        page = await client.create_page(
            database_id="test-db",
            properties={"Name": {"title": [{"text": {"content": "Test Page"}}]}},
        )

        assert page["id"].startswith("local_page_")
        assert page["parent"]["database_id"] == "test-db"
        assert page["properties"]["Name"]["title"][0]["text"]["content"] == "Test Page"

    @pytest.mark.asyncio
    async def test_update_page(self, tmp_path):
        client = LocalNotionClient(data_dir=tmp_path)
        await client.connect()

        page = await client.create_page(
            database_id="test-db",
            properties={"Name": {"title": [{"text": {"content": "Original"}}]}},
        )

        updated = await client.update_page(
            page_id=page["id"],
            properties={"Status": {"select": {"name": "Done"}}},
        )

        assert "Status" in updated["properties"]

    @pytest.mark.asyncio
    async def test_query_database(self, tmp_path):
        client = LocalNotionClient(data_dir=tmp_path)
        await client.connect()

        await client.create_page("db-1", {"title": "Page 1"})
        await client.create_page("db-1", {"title": "Page 2"})
        await client.create_page("db-2", {"title": "Page 3"})

        results = await client.query_database("db-1")
        assert len(results) == 2

        results = await client.query_database("db-2")
        assert len(results) == 1

        results = await client.query_database("nonexistent")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_update_nonexistent_page_raises(self, tmp_path):
        client = LocalNotionClient(data_dir=tmp_path)
        await client.connect()

        with pytest.raises(KeyError):
            await client.update_page("nonexistent", {"foo": "bar"})


# ── Singleton Tests ──────────────────────────────────────────────


class TestProviderSingleton:
    """Tests for the global provider singleton."""

    def test_get_provider_returns_same_instance(self):
        reset_provider()
        try:
            env_clean = {k: v for k, v in os.environ.items() if k not in (
                "BLURT_MODE", "GEMINI_API_KEY", "BLURT_GOOGLE_CLIENT_ID",
                "GOOGLE_APPLICATION_CREDENTIALS", "NOTION_API_TOKEN"
            )}
            with patch.dict(os.environ, env_clean, clear=True):
                p1 = get_provider()
                p2 = get_provider()
                assert p1 is p2
        finally:
            reset_provider()

    def test_reset_provider(self):
        reset_provider()
        try:
            env_clean = {k: v for k, v in os.environ.items() if k not in (
                "BLURT_MODE", "GEMINI_API_KEY", "BLURT_GOOGLE_CLIENT_ID",
                "GOOGLE_APPLICATION_CREDENTIALS", "NOTION_API_TOKEN"
            )}
            with patch.dict(os.environ, env_clean, clear=True):
                p1 = get_provider()
                reset_provider()
                p2 = get_provider()
                assert p1 is not p2
        finally:
            reset_provider()
