"""Shared fixtures for real E2E tests (no mocks).

These tests call the REAL Gemini API — every test is skipped automatically
if ``GEMINI_API_KEY`` is not present in the environment.

Provides:
- gemini_config: ``GeminiConfig`` built from the environment.
- gemini_client: Connected ``GeminiClient`` (async, auto-closed).
- intent_classifier: ``IntentClassifier`` wired to the real client.
- entity_extractor: ``EntityExtractor`` wired to the real client.
- emotion_service: ``EmotionDetectionService`` wired to the real client.
- embedding_provider: ``GeminiEmbeddingProvider`` for real embeddings.
- embedding_service: ``EmbeddingService`` wrapping the real provider.
- episodic_store: Fresh ``InMemoryEpisodicStore`` per test.
- capture_pipeline: ``BlurtCapturePipeline`` using real Gemini classifiers.

Run selectively::

    pytest -m real_e2e
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv

# Load .env from project root before anything reads os.environ
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import pytest
import pytest_asyncio

from blurt.classification.classifier import IntentClassifier
from blurt.clients.embeddings import GeminiEmbeddingProvider
from blurt.clients.gemini import GeminiClient
from blurt.config.settings import GeminiConfig
from blurt.core.entity_extractor import EntityExtractor
from blurt.memory.episodic import EmotionSnapshot, EntityRef, InMemoryEpisodicStore
from blurt.services.capture import BlurtCapturePipeline
from blurt.services.embedding import EmbeddingService
from blurt.services.emotion import EmotionDetectionService

# ---------------------------------------------------------------------------
# Auto-skip when GEMINI_API_KEY is absent
# ---------------------------------------------------------------------------

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def _skip_if_no_key() -> None:
    """Raise ``pytest.skip`` when the API key is missing."""
    if not _GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not set — skipping real E2E test")


# Apply the ``real_e2e`` marker to every test in this directory automatically.
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the ``real_e2e`` marker to all tests collected from this package."""
    for item in items:
        if "e2e_real" in str(item.fspath):
            item.add_marker(pytest.mark.real_e2e)


# ---------------------------------------------------------------------------
# Rate-limit guard — pause between tests to avoid 429s
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(autouse=True)
async def _rate_limit_pause() -> AsyncGenerator[None, None]:
    """Yield control then sleep after each test to avoid Gemini rate limits."""
    yield
    await asyncio.sleep(3)


# ---------------------------------------------------------------------------
# Gemini infrastructure fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gemini_config() -> GeminiConfig:
    """``GeminiConfig`` populated from the environment.

    Skips the test if ``GEMINI_API_KEY`` is not set.
    Uses higher retry/backoff settings to handle free-tier rate limits.
    """
    _skip_if_no_key()
    return GeminiConfig.from_env(
        max_retries=2,
        retry_backoff_base=2.0,
        retry_backoff_max=15.0,
    )


@pytest_asyncio.fixture
async def gemini_client(gemini_config: GeminiConfig) -> AsyncGenerator[GeminiClient, None]:
    """Connected ``GeminiClient`` — torn down after each test."""
    client = GeminiClient(gemini_config)
    await client.connect()
    try:
        yield client
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# NLP service fixtures (all backed by real Gemini)
# ---------------------------------------------------------------------------


@pytest.fixture
def intent_classifier(gemini_client: GeminiClient) -> IntentClassifier:
    """Real ``IntentClassifier`` using Gemini Flash-Lite."""
    return IntentClassifier(gemini_client)


@pytest.fixture
def entity_extractor(gemini_client: GeminiClient) -> EntityExtractor:
    """Real ``EntityExtractor`` using Gemini Flash-Lite."""
    return EntityExtractor(gemini_client)


@pytest.fixture
def emotion_service(gemini_client: GeminiClient) -> EmotionDetectionService:
    """Real ``EmotionDetectionService`` using Gemini Flash-Lite."""
    return EmotionDetectionService(gemini_client)


@pytest.fixture
def embedding_provider(gemini_config: GeminiConfig) -> GeminiEmbeddingProvider:
    """Real ``GeminiEmbeddingProvider`` using gemini-embedding-001."""
    return GeminiEmbeddingProvider(gemini_config)


@pytest.fixture
def embedding_service(embedding_provider: GeminiEmbeddingProvider) -> EmbeddingService:
    """Real ``EmbeddingService`` wrapping the Gemini provider."""
    return EmbeddingService(embedding_provider)


# ---------------------------------------------------------------------------
# Storage fixtures (in-memory — tests are isolated)
# ---------------------------------------------------------------------------


@pytest.fixture
def episodic_store() -> InMemoryEpisodicStore:
    """Fresh in-memory episodic store for each test."""
    return InMemoryEpisodicStore()


# ---------------------------------------------------------------------------
# Capture pipeline (real classifiers, in-memory storage)
# ---------------------------------------------------------------------------


@pytest.fixture
def capture_pipeline(
    episodic_store: InMemoryEpisodicStore,
    intent_classifier: IntentClassifier,
    entity_extractor: EntityExtractor,
    emotion_service: EmotionDetectionService,
    embedding_provider: GeminiEmbeddingProvider,
) -> BlurtCapturePipeline:
    """``BlurtCapturePipeline`` wired to real Gemini services.

    Classification, entity extraction, emotion detection, and embedding
    all call the real Gemini API.  Storage is in-memory so tests stay
    isolated and idempotent.
    """

    async def _classify(text: str) -> tuple[str, float]:
        scores = await intent_classifier.classify(text)
        if scores:
            top = scores[0]
            return (top.intent.value, top.confidence)
        return ("journal", 0.5)

    async def _extract_entities(text: str) -> list[EntityRef]:
        result = await entity_extractor.extract(text)
        return [
            EntityRef(
                name=e.name,
                entity_type=e.entity_type if isinstance(e.entity_type, str) else e.entity_type.value,
                entity_id=str(hash(e.name)),
                confidence=0.9,
            )
            for e in result.entities
        ]

    async def _detect_emotion(text: str) -> EmotionSnapshot:
        result = await emotion_service.detect(text)
        return EmotionSnapshot(
            primary=result.primary.emotion.value if hasattr(result.primary.emotion, "value") else str(result.primary.emotion),
            intensity=result.primary.intensity,
            valence=result.valence,
            arousal=result.arousal,
        )

    async def _embed(text: str) -> list[float]:
        if not text.strip():
            return []
        return await embedding_provider.embed(text)

    return BlurtCapturePipeline(
        store=episodic_store,
        classifier=_classify,
        entity_extractor=_extract_entities,
        emotion_detector=_detect_emotion,
        embedder=_embed,
    )


# ---------------------------------------------------------------------------
# Test identifiers
# ---------------------------------------------------------------------------

TEST_USER_ID = "real-e2e-test-user"
TEST_SESSION_ID = "real-e2e-test-session"


@pytest.fixture
def test_user_id() -> str:
    """Stable test user identifier."""
    return TEST_USER_ID


@pytest.fixture
def test_session_id() -> str:
    """Stable test session identifier."""
    return TEST_SESSION_ID
