"""Real E2E test: 3-tier memory system stores and retrieves a classified blurt.

Validates the full memory lifecycle with REAL Gemini API calls:
1. Capture a blurt via the pipeline (Tier 2 — episodic store)
2. Retrieve it by ID, by entity, by intent, and by session
3. Ingest into the promotion pipeline (Tier 1 — working memory)
4. Run promotion cycle to move through working → episodic → semantic (Tier 3)
5. Verify entity nodes and relationships in semantic memory

No mocks. Skipped automatically when ``GEMINI_API_KEY`` is not set.
"""

from __future__ import annotations

import pytest

from blurt.core.memory.models import (
    Entity,
    IntentType,
    MemoryTier,
    WorkingMemoryItem,
)
from blurt.core.memory.promotion import MemoryPromotionPipeline, MemoryStore, PromotionThresholds
from blurt.memory.episodic import (
    EntityFilter,
    InMemoryEpisodicStore,
)
from blurt.memory.working import WorkingMemory
from blurt.services.capture import BlurtCapturePipeline, CaptureResult

pytestmark = pytest.mark.asyncio

# Input text with clear entities and scheduling intent
INPUT_TEXT = "Tell Sarah we need to reschedule the product launch meeting to next Thursday"
TEST_USER = "real-e2e-test-user"
TEST_SESSION = "real-e2e-test-session"


# ---------------------------------------------------------------------------
# Tier 2: Episodic store — capture and retrieval
# ---------------------------------------------------------------------------


async def test_episodic_store_and_retrieve_by_id(
    capture_pipeline: BlurtCapturePipeline,
    episodic_store: InMemoryEpisodicStore,
) -> None:
    """Capture a blurt, then retrieve the stored episode by its ID."""
    result: CaptureResult = await capture_pipeline.capture_text(
        TEST_USER,
        INPUT_TEXT,
        session_id=TEST_SESSION,
    )

    assert result.was_stored, (
        f"Episode was NOT stored. Warnings: {result.warnings}"
    )

    # Retrieve by ID
    episode = await episodic_store.get(result.episode.id)
    assert episode is not None, (
        f"Could not retrieve episode by ID '{result.episode.id}'"
    )
    assert episode.raw_text == INPUT_TEXT, (
        f"Retrieved episode text mismatch: expected '{INPUT_TEXT}', got '{episode.raw_text}'"
    )
    assert episode.user_id == TEST_USER
    assert episode.intent, "Episode should have a classified intent"


async def test_episodic_query_by_entity(
    capture_pipeline: BlurtCapturePipeline,
    episodic_store: InMemoryEpisodicStore,
) -> None:
    """Capture a blurt mentioning 'Sarah', then query episodic store by entity name."""
    result: CaptureResult = await capture_pipeline.capture_text(
        TEST_USER,
        INPUT_TEXT,
        session_id=TEST_SESSION,
    )

    assert result.was_stored
    assert result.entities_extracted >= 1, (
        f"Expected at least 1 entity extracted, got {result.entities_extracted}"
    )

    # Query by entity name "Sarah"
    episodes = await episodic_store.query(
        TEST_USER,
        entity_filter=EntityFilter(entity_name="Sarah"),
    )
    assert len(episodes) >= 1, (
        f"Expected at least 1 episode mentioning 'Sarah', got {len(episodes)}. "
        f"Stored entities: {[e.name for e in result.episode.entities]}"
    )
    assert any(INPUT_TEXT in ep.raw_text for ep in episodes)


async def test_episodic_query_by_session(
    capture_pipeline: BlurtCapturePipeline,
    episodic_store: InMemoryEpisodicStore,
) -> None:
    """Capture a blurt, then retrieve it via session filter."""
    result: CaptureResult = await capture_pipeline.capture_text(
        TEST_USER,
        INPUT_TEXT,
        session_id=TEST_SESSION,
    )

    assert result.was_stored

    # Query by session
    session_episodes = await episodic_store.get_session_episodes(TEST_SESSION)
    assert len(session_episodes) >= 1, (
        f"Expected at least 1 episode in session '{TEST_SESSION}', got {len(session_episodes)}"
    )
    assert any(ep.raw_text == INPUT_TEXT for ep in session_episodes)


# ---------------------------------------------------------------------------
# Tier 1: Working memory — add and context aggregation
# ---------------------------------------------------------------------------


async def test_working_memory_stores_classified_blurt(
    intent_classifier,  # noqa: ANN001
) -> None:
    """Classify a blurt with real Gemini, store in working memory, verify retrieval."""
    from blurt.memory.working import IntentType as WMIntentType

    # Classify with real Gemini
    scores = await intent_classifier.classify(INPUT_TEXT)
    assert scores, "IntentClassifier returned empty scores"
    top = scores[0]

    # Map to working memory IntentType
    wm_intent = WMIntentType(top.intent.value)

    # Store in working memory
    wm = WorkingMemory(session_id=TEST_SESSION)
    entry = await wm.add(
        INPUT_TEXT,
        intent=wm_intent,
        confidence=top.confidence,
        source="text",
    )

    # Retrieve and verify
    retrieved = await wm.get_entry(entry.id)
    assert retrieved is not None, f"Could not retrieve working memory entry '{entry.id}'"
    assert retrieved.content == INPUT_TEXT
    assert retrieved.intent == wm_intent
    assert retrieved.confidence == top.confidence

    # Context should reflect the entry
    ctx = await wm.get_context()
    assert ctx.entry_count == 1
    assert wm_intent in ctx.recent_intents


# ---------------------------------------------------------------------------
# Tier 3: Promotion pipeline — working → episodic → semantic
# ---------------------------------------------------------------------------


async def test_promotion_working_to_episodic(
    intent_classifier,  # noqa: ANN001
    entity_extractor,  # noqa: ANN001
) -> None:
    """Classify + extract entities with real Gemini, ingest into promotion pipeline,
    and verify working → episodic promotion."""
    # Classify
    scores = await intent_classifier.classify(INPUT_TEXT)
    assert scores
    top = scores[0]
    intent = IntentType(top.intent.value)

    # Extract entities
    extraction = await entity_extractor.extract(INPUT_TEXT)
    entities = [
        Entity(name=e.name, entity_type=e.entity_type if isinstance(e.entity_type, str) else e.entity_type.value)
        for e in extraction.entities
    ]

    # Build a working memory item
    item = WorkingMemoryItem(
        content=INPUT_TEXT,
        intent=intent,
        confidence=top.confidence,
        entities=entities,
    )

    # Ingest into the promotion pipeline
    store = MemoryStore()
    pipeline = MemoryPromotionPipeline(store=store)
    pipeline.ingest(item)

    assert len(store.working) == 1, "Item should be in working memory"
    assert len(store.episodic) == 0, "Episodic should be empty before promotion"

    # Run promotion cycle — score threshold is 0.3, task/event intents score high
    events = pipeline.run_promotion_cycle()

    # The item should have been promoted (high relevance from intent + entities)
    assert len(store.episodic) >= 1, (
        f"Expected at least 1 episodic item after promotion. "
        f"Working still has {len(store.working)} items. "
        f"Promotion events: {[(e.reason, e.score) for e in events]}"
    )
    assert len(store.working) == 0, "Working memory should be empty after promotion"

    # Verify the episodic item content
    ep_item = store.episodic[0]
    assert ep_item.content == INPUT_TEXT
    assert ep_item.intent == intent
    assert len(ep_item.entities) == len(entities)


async def test_full_promotion_to_semantic(
    intent_classifier,  # noqa: ANN001
    entity_extractor,  # noqa: ANN001
) -> None:
    """Full 3-tier promotion: working → episodic → semantic with real Gemini classification.

    Requires ≥2 mentions and ≥1 entity for semantic promotion, so we ingest
    the same content twice and bump mention counts.
    """
    # Classify
    scores = await intent_classifier.classify(INPUT_TEXT)
    assert scores
    top = scores[0]
    intent = IntentType(top.intent.value)

    # Extract entities
    extraction = await entity_extractor.extract(INPUT_TEXT)
    assert extraction.entities, "Expected at least 1 entity for semantic promotion test"
    entities = [
        Entity(name=e.name, entity_type=e.entity_type if isinstance(e.entity_type, str) else e.entity_type.value)
        for e in extraction.entities
    ]

    # Build and ingest a working memory item
    item = WorkingMemoryItem(
        content=INPUT_TEXT,
        intent=intent,
        confidence=top.confidence,
        entities=entities,
    )

    store = MemoryStore()
    # Use a lower semantic threshold so the test focuses on promotion mechanics,
    # not scoring calibration.
    thresholds = PromotionThresholds(episodic_to_semantic=0.1)
    pipeline = MemoryPromotionPipeline(store=store, thresholds=thresholds)
    pipeline.ingest(item)

    # Phase 1: promote working → episodic
    pipeline.run_promotion_cycle()
    assert len(store.episodic) >= 1, "Should have at least 1 episodic item"

    # Bump mention count to meet semantic promotion threshold (≥2 mentions)
    # Touch multiple times to also increase access_count / importance score.
    ep_item = store.episodic[0]
    for _ in range(3):
        pipeline.touch_episodic_item(ep_item.id)
    assert ep_item.mention_count >= 2, (
        f"Expected mention_count ≥ 2, got {ep_item.mention_count}"
    )

    # Phase 2: promote episodic → semantic
    events = pipeline.run_promotion_cycle()

    semantic_events = [
        e for e in events if e.target_tier == MemoryTier.SEMANTIC
    ]
    assert len(semantic_events) >= 1, (
        f"Expected at least 1 semantic promotion event. "
        f"Events: {[(e.source_tier.value, e.target_tier.value, e.reason, e.score) for e in events]}"
    )

    # Verify semantic memory has entity nodes
    assert len(store.semantic) >= 1, "Should have at least 1 semantic entity node"

    # Find "Sarah" in semantic memory
    sarah_node = store.find_semantic_by_entity("Sarah")
    assert sarah_node is not None, (
        f"Expected 'Sarah' in semantic memory. "
        f"Semantic entities: {[s.entity.name for s in store.semantic if s.entity]}"
    )
    assert sarah_node.entity is not None
    assert sarah_node.entity.name.lower() == "sarah"
    assert sarah_node.mention_count >= 1

    # Verify tier counts
    counts = pipeline.get_tier_counts()
    assert counts[MemoryTier.WORKING.value] == 0, "Working should be empty"
    assert counts[MemoryTier.EPISODIC.value] >= 1, "Episodic should have items"
    assert counts[MemoryTier.SEMANTIC.value] >= 1, "Semantic should have items"

    # Verify promotion history is tracked
    history = pipeline.get_promotion_history()
    assert len(history) >= 2, (
        f"Expected at least 2 promotion events (w→e, e→s), got {len(history)}"
    )
