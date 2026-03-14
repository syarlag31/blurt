"""Real E2E test: Gemini classifies 'dentist Friday 3pm' as EVENT intent.

This test calls the real Gemini API — no mocks. It validates that the
IntentClassifier correctly identifies a scheduling utterance as an EVENT
with high confidence (>=0.70).
"""

from __future__ import annotations

import pytest

from blurt.classification.classifier import IntentClassifier
from blurt.models.intents import BlurtIntent


@pytest.mark.real_e2e
@pytest.mark.asyncio
async def test_classify_dentist_friday_as_event(
    intent_classifier: IntentClassifier,
) -> None:
    """Gemini should classify 'dentist Friday 3pm' as EVENT with confidence >= 0.70."""
    scores = await intent_classifier.classify("dentist Friday 3pm")

    # Scores are sorted by confidence descending; the top one is the primary intent
    assert len(scores) > 0, "Expected at least one IntentScore from the classifier"

    top = scores[0]
    assert top.intent == BlurtIntent.EVENT, (
        f"Expected EVENT as top intent, got {top.intent.value} "
        f"(confidence={top.confidence:.2f})"
    )
    assert top.confidence >= 0.70, (
        f"Expected EVENT confidence >= 0.70, got {top.confidence:.2f}"
    )
