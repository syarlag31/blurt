"""Real E2E test: full capture pipeline (text → intent + entities + emotion).

Validates the complete ``BlurtCapturePipeline`` with REAL Gemini API calls.
Sends text through ``capture_text()`` and asserts that the returned
``CaptureResult`` contains a classified intent, extracted entities, and
detected emotion — all produced by the live Gemini model.

No mocks. Skipped automatically when ``GEMINI_API_KEY`` is not set.
"""

from __future__ import annotations

import pytest

from blurt.services.capture import BlurtCapturePipeline, CaptureResult, CaptureStage

pytestmark = pytest.mark.asyncio

# A rich sentence that should trigger all enrichment stages:
#   - Intent: EVENT or REMINDER (scheduling language)
#   - Entities: "Sarah" (person), "product launch" (project/event)
#   - Emotion: mild positive or neutral (anticipation)
INPUT_TEXT = "Tell Sarah we need to reschedule the product launch meeting to next Thursday"

TEST_USER = "real-e2e-test-user"
TEST_SESSION = "real-e2e-test-session"


async def test_full_pipeline_stores_episode(
    capture_pipeline: BlurtCapturePipeline,
) -> None:
    """The pipeline should store the episode (zero-drop guarantee)."""
    result: CaptureResult = await capture_pipeline.capture_text(
        TEST_USER,
        INPUT_TEXT,
        session_id=TEST_SESSION,
    )

    assert result.was_stored, (
        f"Episode was NOT stored. Stages completed: "
        f"{[s.value for s in result.stages_completed]}, warnings: {result.warnings}"
    )
    assert CaptureStage.STORED in result.stages_completed


async def test_full_pipeline_classifies_intent(
    capture_pipeline: BlurtCapturePipeline,
) -> None:
    """The pipeline should classify the input with a recognised intent."""
    result: CaptureResult = await capture_pipeline.capture_text(
        TEST_USER,
        INPUT_TEXT,
        session_id=TEST_SESSION,
    )

    assert result.classification_applied, (
        f"Classification was not applied. Warnings: {result.warnings}"
    )
    assert CaptureStage.CLASSIFIED in result.stages_completed

    # The stored episode should carry an intent label
    episode = result.episode
    assert episode.intent, (
        f"Episode has no intent set. Episode: {episode}"
    )
    # For a rescheduling sentence Gemini may classify as event, task, reminder,
    # or update — all are valid. We accept any of the 7 intents as proof the pipeline ran.
    valid_intents = {"event", "reminder", "task", "journal", "update", "idea", "question"}
    assert episode.intent in valid_intents, (
        f"Unexpected intent '{episode.intent}' — not one of the 7 Blurt intents"
    )


async def test_full_pipeline_extracts_entities(
    capture_pipeline: BlurtCapturePipeline,
) -> None:
    """The pipeline should extract at least one entity (e.g. Sarah, product launch)."""
    result: CaptureResult = await capture_pipeline.capture_text(
        TEST_USER,
        INPUT_TEXT,
        session_id=TEST_SESSION,
    )

    assert CaptureStage.ENTITIES_EXTRACTED in result.stages_completed, (
        f"Entity extraction stage not completed. Warnings: {result.warnings}"
    )
    assert result.entities_extracted >= 1, (
        f"Expected at least 1 entity, got {result.entities_extracted}"
    )

    # Check the stored episode has entities
    episode = result.episode
    assert episode.entities, (
        f"Episode has no entities despite extraction claiming {result.entities_extracted} entities"
    )

    # "Sarah" should appear among entity names
    entity_names_lower = [e.name.lower() for e in episode.entities]
    assert any("sarah" in n for n in entity_names_lower), (
        f"Expected 'Sarah' among entities, got: {entity_names_lower}"
    )


async def test_full_pipeline_detects_emotion(
    capture_pipeline: BlurtCapturePipeline,
) -> None:
    """The pipeline should detect emotion for the input text."""
    result: CaptureResult = await capture_pipeline.capture_text(
        TEST_USER,
        INPUT_TEXT,
        session_id=TEST_SESSION,
    )

    assert result.emotion_detected, (
        f"Emotion was not detected. Warnings: {result.warnings}"
    )
    assert CaptureStage.EMOTION_DETECTED in result.stages_completed

    # The episode should carry an emotion snapshot
    episode = result.episode
    assert episode.emotion, (
        f"Episode has no emotion snapshot despite emotion_detected=True"
    )
    assert episode.emotion.primary, (
        f"Emotion snapshot has no primary emotion"
    )
    # Intensity should be a meaningful value
    assert 0.0 <= episode.emotion.intensity <= 1.0, (
        f"Emotion intensity out of [0,1] range: {episode.emotion.intensity}"
    )


async def test_full_pipeline_generates_embedding(
    capture_pipeline: BlurtCapturePipeline,
) -> None:
    """The pipeline should generate a real embedding vector."""
    result: CaptureResult = await capture_pipeline.capture_text(
        TEST_USER,
        INPUT_TEXT,
        session_id=TEST_SESSION,
    )

    assert result.embedding_generated, (
        f"Embedding was not generated. Warnings: {result.warnings}"
    )
    assert CaptureStage.EMBEDDED in result.stages_completed

    # The episode should have a non-empty embedding
    episode = result.episode
    assert episode.embedding, (
        f"Episode has no embedding despite embedding_generated=True"
    )
    assert len(episode.embedding) == 768, (
        f"Expected 768-dim embedding, got {len(episode.embedding)}"
    )


async def test_full_pipeline_fully_enriched(
    capture_pipeline: BlurtCapturePipeline,
) -> None:
    """A well-formed sentence should result in a fully enriched capture."""
    result: CaptureResult = await capture_pipeline.capture_text(
        TEST_USER,
        INPUT_TEXT,
        session_id=TEST_SESSION,
    )

    assert result.fully_enriched, (
        f"Expected fully_enriched=True but got False. "
        f"classification={result.classification_applied}, "
        f"emotion={result.emotion_detected}, "
        f"embedding={result.embedding_generated}. "
        f"Warnings: {result.warnings}"
    )
    assert not result.warnings, (
        f"Expected no warnings for a clean capture, got: {result.warnings}"
    )
    assert result.latency_ms > 0, "Latency should be a positive value"
