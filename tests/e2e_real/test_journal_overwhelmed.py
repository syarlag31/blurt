"""Real E2E test: "I feel overwhelmed today" → JOURNAL intent + negative emotion.

Uses REAL Gemini API calls — no mocks.
Skipped automatically when ``GEMINI_API_KEY`` is not set.
"""

from __future__ import annotations

import pytest

from blurt.models.intents import BlurtIntent
from blurt.models.emotions import PrimaryEmotion

pytestmark = pytest.mark.asyncio

INPUT_TEXT = "I feel overwhelmed today"

# Negative emotions we'd accept for "overwhelmed" — primarily sadness/fear,
# but anger or disgust are also plausible from a real LLM.
_NEGATIVE_EMOTIONS = {
    PrimaryEmotion.SADNESS,
    PrimaryEmotion.FEAR,
    PrimaryEmotion.ANGER,
    PrimaryEmotion.DISGUST,
}


async def test_overwhelmed_classified_as_journal(intent_classifier) -> None:
    """Gemini should classify 'I feel overwhelmed today' as JOURNAL."""
    scores = await intent_classifier.classify(INPUT_TEXT)

    assert scores, "Expected non-empty classification scores"

    top = scores[0]
    assert top.intent == BlurtIntent.JOURNAL, (
        f"Expected JOURNAL as top intent, got {top.intent.value} "
        f"(confidence={top.confidence:.2f})"
    )
    # The phrase is a clear personal reflection — confidence should be solid.
    assert top.confidence >= 0.5, (
        f"Expected confidence ≥ 0.50 for JOURNAL, got {top.confidence:.2f}"
    )


async def test_overwhelmed_has_negative_valence(emotion_service) -> None:
    """Emotion detection should report negative valence for 'I feel overwhelmed today'."""
    result = await emotion_service.detect(INPUT_TEXT)

    # "overwhelmed" is unambiguously negative.
    assert result.valence < 0, (
        f"Expected negative valence for '{INPUT_TEXT}', got {result.valence:.2f}"
    )


async def test_overwhelmed_primary_emotion_is_negative(emotion_service) -> None:
    """Primary emotion should be one of the negative Plutchik emotions."""
    result = await emotion_service.detect(INPUT_TEXT)

    assert result.primary.emotion in _NEGATIVE_EMOTIONS, (
        f"Expected a negative primary emotion (sadness/fear/anger/disgust), "
        f"got {result.primary.emotion.value} with intensity {result.primary.intensity:.2f}"
    )
    # Intensity should be meaningful, not negligible.
    assert result.primary.intensity >= 0.3, (
        f"Expected meaningful intensity (≥ 0.30), got {result.primary.intensity:.2f}"
    )
