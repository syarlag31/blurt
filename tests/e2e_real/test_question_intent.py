"""Real E2E test: Gemini classifies question-type input as QUESTION intent.

Uses the real Gemini API — skipped when ``GEMINI_API_KEY`` is absent.
"""

from __future__ import annotations

import pytest

from blurt.models.intents import BlurtIntent


@pytest.mark.asyncio
async def test_question_intent_q2_deck(intent_classifier):
    """Gemini should classify 'what did I say about the Q2 deck' as QUESTION.

    This is a retrieval-style question about the user's own past input,
    which maps directly to the QUESTION intent in the Blurt taxonomy.
    """
    text = "what did I say about the Q2 deck"
    scores = await intent_classifier.classify(text)

    # Should return a non-empty list of scored intents
    assert scores, "Expected at least one IntentScore from classify()"

    # The top-scored intent should be QUESTION
    top = scores[0]
    assert top.intent == BlurtIntent.QUESTION, (
        f"Expected QUESTION as top intent, got {top.intent.value} "
        f"(confidence={top.confidence:.2f})"
    )

    # Confidence for QUESTION should be reasonably high
    assert top.confidence >= 0.5, (
        f"Expected confidence >= 0.5 for QUESTION, got {top.confidence:.2f}"
    )
