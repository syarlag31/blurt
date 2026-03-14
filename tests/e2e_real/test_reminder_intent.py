"""Real E2E test: Gemini classifies "remind me to call Mom tonight" as REMINDER.

Uses the REAL Gemini API — no mocks. Skipped automatically when
``GEMINI_API_KEY`` is not set in the environment.
"""

from __future__ import annotations

import pytest

from blurt.classification.classifier import IntentClassifier
from blurt.models.intents import BlurtIntent


@pytest.mark.real_e2e
@pytest.mark.asyncio
async def test_remind_me_classified_as_reminder(
    intent_classifier: IntentClassifier,
) -> None:
    """Gemini should classify 'remind me to call Mom tonight' as REMINDER intent."""
    scores = await intent_classifier.classify("remind me to call Mom tonight")

    assert scores, "Expected non-empty scores from classifier"

    top = scores[0]
    assert top.intent == BlurtIntent.REMINDER, (
        f"Expected REMINDER as top intent, got {top.intent.value} "
        f"(confidence={top.confidence:.2f})"
    )
    # The phrase is unambiguous — confidence should be reasonably high
    assert top.confidence >= 0.5, (
        f"Expected confidence >= 0.5 for REMINDER, got {top.confidence:.2f}"
    )
