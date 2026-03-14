"""Shared fixtures for E2E tests.

Provides:
- app: Fully configured FastAPI application with all routers registered
- client: httpx AsyncClient wired to the app (no network)
- test_user_id / test_session_id: Stable identifiers for test scenarios
- memory_reset: Auto-use fixture that resets all DI singletons between tests
- create_episode_via_api: Helper to create episodes through the API
- create_pattern_via_api: Helper to create patterns through the API
- record_feedback_via_api: Helper to record task feedback through the API

All fixtures are self-contained and work with both BLURT_MODE=cloud and
BLURT_MODE=local.  External APIs (Gemini, Google Calendar) are never called;
the capture pipeline uses lightweight stub classifiers that run in-process.
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from blurt.api.capture import router as capture_router, set_pipeline
from blurt.api.episodes import router as episodes_router, set_store
from blurt.api.patterns import (
    router as patterns_router,
    set_pattern_service,
)
from blurt.api.task_feedback import (
    router as feedback_router,
    set_feedback_service,
)
from blurt.config.settings import BlurtConfig, DeploymentMode
from blurt.core.app import create_app
from blurt.memory.episodic import (
    EmotionSnapshot,
    EntityRef,
    InMemoryEpisodicStore,
)
from blurt.services.capture import BlurtCapturePipeline
from blurt.services.feedback import InMemoryFeedbackStore, TaskFeedbackService
from blurt.services.patterns import InMemoryPatternStore, PatternService


# ---------------------------------------------------------------------------
# Lightweight stub pipeline functions (no external API calls)
# ---------------------------------------------------------------------------


async def _stub_classify(text: str) -> tuple[str, float]:
    """Deterministic intent classifier for E2E tests.

    Maps keywords to intents so test assertions are stable.
    """
    lower = text.lower().strip()

    if not lower:
        return ("journal", 1.0)

    keyword_map: list[tuple[list[str], str, float]] = [
        (["need to", "buy", "call", "finish", "todo", "fix", "send"], "task", 0.92),
        (["meeting", "dinner", "appointment", "lunch", "event"], "event", 0.89),
        (["remind", "reminder", "don't forget", "remember to"], "reminder", 0.88),
        (["idea", "what if", "maybe we could", "brainstorm"], "idea", 0.87),
        (["update", "progress", "status", "done with"], "update", 0.86),
        (["how", "what", "why", "when", "where", "?"], "question", 0.85),
    ]

    for keywords, intent, confidence in keyword_map:
        if any(kw in lower for kw in keywords):
            return (intent, confidence)

    return ("journal", 0.85)


async def _stub_extract_entities(text: str) -> list[EntityRef]:
    """Deterministic entity extractor for E2E tests."""
    entities: list[EntityRef] = []
    lower = text.lower()

    entity_map: dict[str, tuple[str, str]] = {
        "alice": ("Alice", "person"),
        "bob": ("Bob", "person"),
        "acme": ("Acme Corp", "organization"),
        "project alpha": ("Project Alpha", "project"),
        "new york": ("New York", "place"),
        "python": ("Python", "tool"),
        "react": ("React", "tool"),
    }

    for keyword, (name, etype) in entity_map.items():
        if keyword in lower:
            entities.append(
                EntityRef(
                    name=name,
                    entity_type=etype,
                    entity_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, name)),
                    confidence=0.95,
                )
            )

    return entities


async def _stub_detect_emotion(text: str) -> EmotionSnapshot:
    """Deterministic emotion detector for E2E tests."""
    lower = text.lower()

    if any(w in lower for w in ["happy", "great", "excited", "love"]):
        return EmotionSnapshot(primary="joy", intensity=2.0, valence=0.8, arousal=0.7)
    if any(w in lower for w in ["angry", "frustrated", "annoyed"]):
        return EmotionSnapshot(primary="anger", intensity=1.5, valence=-0.6, arousal=0.8)
    if any(w in lower for w in ["sad", "down", "depressed"]):
        return EmotionSnapshot(primary="sadness", intensity=1.5, valence=-0.7, arousal=0.3)
    if any(w in lower for w in ["worried", "anxious", "nervous"]):
        return EmotionSnapshot(primary="fear", intensity=1.0, valence=-0.4, arousal=0.6)

    return EmotionSnapshot(primary="trust", intensity=0.5, valence=0.0, arousal=0.2)


async def _stub_embed(text: str) -> list[float]:
    """Deterministic embedder that produces a simple hash-based vector.

    Returns a 16-dimensional vector (small for tests) derived from the text
    so that similar inputs yield somewhat similar vectors.
    """
    if not text.strip():
        return []

    vec = [0.0] * 16
    for i, ch in enumerate(text.lower()):
        vec[i % 16] += ord(ch) / 1000.0

    # Normalize
    magnitude = sum(v * v for v in vec) ** 0.5
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

TEST_USER_ID = "e2e-test-user"
TEST_SESSION_ID = "e2e-test-session"


@pytest.fixture
def test_user_id() -> str:
    """Stable test user identifier."""
    return TEST_USER_ID


@pytest.fixture
def test_session_id() -> str:
    """Stable test session identifier."""
    return TEST_SESSION_ID


@pytest.fixture
def episodic_store() -> InMemoryEpisodicStore:
    """Fresh in-memory episodic store for each test."""
    return InMemoryEpisodicStore()


@pytest.fixture
def pattern_store() -> InMemoryPatternStore:
    """Fresh in-memory pattern store for each test."""
    return InMemoryPatternStore()


@pytest.fixture
def feedback_store() -> InMemoryFeedbackStore:
    """Fresh in-memory feedback store for each test."""
    return InMemoryFeedbackStore()


@pytest.fixture
def capture_pipeline(episodic_store: InMemoryEpisodicStore) -> BlurtCapturePipeline:
    """Capture pipeline wired to stub classifiers (no external calls)."""
    return BlurtCapturePipeline(
        store=episodic_store,
        classifier=_stub_classify,
        entity_extractor=_stub_extract_entities,
        emotion_detector=_stub_detect_emotion,
        embedder=_stub_embed,
    )


@pytest.fixture
def pattern_service(pattern_store: InMemoryPatternStore) -> PatternService:
    """Pattern service backed by fresh in-memory store."""
    return PatternService(store=pattern_store)


@pytest.fixture
def feedback_service(feedback_store: InMemoryFeedbackStore) -> TaskFeedbackService:
    """Feedback service backed by fresh in-memory store."""
    return TaskFeedbackService(store=feedback_store)


@pytest.fixture
def app(
    episodic_store: InMemoryEpisodicStore,
    capture_pipeline: BlurtCapturePipeline,
    pattern_service: PatternService,
    feedback_service: TaskFeedbackService,
) -> FastAPI:
    """Fully configured FastAPI app with all routers and DI overrides.

    Uses the real FastAPI application factory from blurt.core.app but
    overrides DI singletons with fresh in-memory stores so tests are
    isolated. Registers routers not included in the default app (capture,
    episodes) since they are needed for E2E flows.
    """
    config = BlurtConfig(
        mode=DeploymentMode.CLOUD,
        debug=True,
    )
    application = create_app(config)

    # Register additional routers that are not in the default app
    # (capture and episodes are used via the pipeline but may not be
    # mounted by create_app; include them idempotently)
    _registered_paths = {r.path for r in application.routes}

    if "/api/v1/blurt" not in _registered_paths:
        application.include_router(capture_router)
    if "/api/v1/episodes" not in _registered_paths:
        application.include_router(episodes_router)

    # Override DI singletons with test instances
    set_store(episodic_store)
    set_pipeline(capture_pipeline)
    set_pattern_service(pattern_service)
    set_feedback_service(feedback_service)

    return application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """httpx AsyncClient bound to the test FastAPI app.

    Uses ASGITransport so requests flow through the full ASGI stack
    (middleware, lifespan, dependency injection) without network I/O.
    """
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def memory_reset(
    episodic_store: InMemoryEpisodicStore,
    pattern_service: PatternService,
    feedback_service: TaskFeedbackService,
    capture_pipeline: BlurtCapturePipeline,
) -> None:
    """Auto-use fixture that ensures DI singletons are reset for every test.

    This runs before each test (via the dependency chain) and guarantees
    that module-level singletons point to the fresh per-test stores
    created by the other fixtures.  No state leaks between tests.
    """
    # The DI overrides are already applied by the `app` fixture's
    # set_store / set_pipeline / set_pattern_service / set_feedback_service
    # calls.  This fixture exists to make the reset *explicit* and ensure
    # it fires even for tests that don't request the `app` fixture directly.
    set_store(episodic_store)
    set_pipeline(capture_pipeline)
    set_pattern_service(pattern_service)
    set_feedback_service(feedback_service)


# ---------------------------------------------------------------------------
# API helper factories — convenience for multi-step test scenarios
# ---------------------------------------------------------------------------


@pytest.fixture
def create_episode_via_api(
    client: httpx.AsyncClient,
    test_user_id: str,
) -> Any:
    """Factory fixture: create an episode through the episodes API.

    Returns an async callable:
        episode = await create_episode_via_api(raw_text="hello", intent="journal")
    """

    async def _create(
        raw_text: str = "test blurt",
        intent: str = "task",
        intent_confidence: float = 0.9,
        emotion_primary: str = "trust",
        emotion_intensity: float = 0.5,
        emotion_valence: float = 0.0,
        emotion_arousal: float = 0.2,
        entities: list[dict[str, Any]] | None = None,
        behavioral_signal: str = "none",
        session_id: str = "",
        time_of_day: str = "morning",
        day_of_week: str = "monday",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id or test_user_id,
            "raw_text": raw_text,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "emotion": {
                "primary": emotion_primary,
                "intensity": emotion_intensity,
                "valence": emotion_valence,
                "arousal": emotion_arousal,
            },
            "entities": entities or [],
            "behavioral_signal": behavioral_signal,
            "context": {
                "time_of_day": time_of_day,
                "day_of_week": day_of_week,
                "session_id": session_id,
            },
        }
        resp = await client.post("/api/v1/episodes", json=payload)
        assert resp.status_code == 201, f"Episode creation failed: {resp.text}"
        return resp.json()

    return _create


@pytest.fixture
def create_pattern_via_api(
    client: httpx.AsyncClient,
    test_user_id: str,
) -> Any:
    """Factory fixture: create a pattern through the patterns API.

    Returns an async callable:
        pattern = await create_pattern_via_api(
            pattern_type="energy",
            description="User is more productive in mornings",
        )
    """

    async def _create(
        pattern_type: str = "energy",
        description: str = "Test pattern",
        parameters: dict[str, Any] | None = None,
        confidence: float = 0.5,
        observation_count: int = 1,
        supporting_evidence: list[str] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        uid = user_id or test_user_id
        payload = {
            "pattern_type": pattern_type,
            "description": description,
            "parameters": parameters or {},
            "confidence": confidence,
            "observation_count": observation_count,
            "supporting_evidence": supporting_evidence or [],
        }
        resp = await client.post(
            f"/api/v1/users/{uid}/patterns",
            json=payload,
        )
        assert resp.status_code == 201, f"Pattern creation failed: {resp.text}"
        return resp.json()

    return _create


@pytest.fixture
def record_feedback_via_api(
    client: httpx.AsyncClient,
    test_user_id: str,
) -> Any:
    """Factory fixture: record task feedback through the API.

    Returns an async callable:
        feedback = await record_feedback_via_api(
            task_id="task-1",
            action="accept",
        )
    """

    async def _record(
        task_id: str = "task-1",
        action: str = "accept",
        mood_valence: float = 0.0,
        energy_level: float = 0.5,
        time_of_day: str = "morning",
        snooze_minutes: int | None = None,
        intent: str = "task",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "user_id": user_id or test_user_id,
            "action": action,
            "mood_valence": mood_valence,
            "energy_level": energy_level,
            "time_of_day": time_of_day,
            "intent": intent,
        }
        if snooze_minutes is not None:
            payload["snooze_minutes"] = snooze_minutes
        resp = await client.post(
            f"/api/v1/tasks/{task_id}/feedback",
            json=payload,
        )
        assert resp.status_code == 201, f"Feedback recording failed: {resp.text}"
        return resp.json()

    return _record


@pytest.fixture
def capture_blurt_via_api(
    client: httpx.AsyncClient,
    test_user_id: str,
) -> Any:
    """Factory fixture: capture a blurt through the capture API.

    Returns an async callable:
        result = await capture_blurt_via_api("I need to call the dentist")
    """

    async def _capture(
        raw_text: str = "test blurt",
        modality: str = "voice",
        session_id: str = "",
        time_of_day: str = "morning",
        day_of_week: str = "monday",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id or test_user_id,
            "raw_text": raw_text,
            "modality": modality,
            "session_id": session_id,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
        }
        resp = await client.post("/api/v1/blurt", json=payload)
        assert resp.status_code == 201, f"Blurt capture failed: {resp.text}"
        return resp.json()

    return _capture
