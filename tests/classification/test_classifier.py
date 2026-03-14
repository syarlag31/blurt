"""Tests for the intent classifier."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.classification.classifier import ClassificationError, IntentClassifier
from blurt.clients.gemini import GeminiClient, GeminiResponse, ModelTier
from blurt.models.intents import BlurtIntent


def _make_gemini_response(text: str, model: str = "flash-lite") -> GeminiResponse:
    """Helper to create a mock GeminiResponse."""
    return GeminiResponse(text=text, raw={}, model=model)


def _make_classification_json(
    primary: str = "task",
    scores: dict[str, float] | None = None,
) -> str:
    """Helper to build classifier JSON output."""
    if scores is None:
        scores = {
            "task": 0.92,
            "event": 0.02,
            "reminder": 0.02,
            "idea": 0.01,
            "journal": 0.01,
            "update": 0.01,
            "question": 0.01,
        }
    return json.dumps({"primary_intent": primary, "confidence_scores": scores})


@pytest.fixture
def mock_client() -> GeminiClient:
    """Create a mock GeminiClient."""
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock()
    return client


@pytest.fixture
def classifier(mock_client: GeminiClient) -> IntentClassifier:
    return IntentClassifier(mock_client)


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    async def test_classify_clear_task(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Test classification of a clear task intent."""
        mock_client.generate.return_value = _make_gemini_response(
            _make_classification_json("task", {
                "task": 0.92, "event": 0.02, "reminder": 0.02,
                "idea": 0.01, "journal": 0.01, "update": 0.01, "question": 0.01,
            })
        )

        scores = await classifier.classify("I need to buy groceries")

        assert len(scores) == 7
        assert scores[0].intent == BlurtIntent.TASK
        assert scores[0].confidence > 0.85

        # Verify it used the FAST tier
        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs.get("tier") == ModelTier.FAST

    async def test_classify_journal_entry(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Test classification of a journal entry."""
        mock_client.generate.return_value = _make_gemini_response(
            _make_classification_json("journal", {
                "task": 0.02, "event": 0.01, "reminder": 0.01,
                "idea": 0.03, "journal": 0.89, "update": 0.02, "question": 0.02,
            })
        )

        scores = await classifier.classify("Today was a great day at work")
        assert scores[0].intent == BlurtIntent.JOURNAL
        assert scores[0].confidence > 0.85

    async def test_classify_event(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Test classification of an event."""
        mock_client.generate.return_value = _make_gemini_response(
            _make_classification_json("event", {
                "task": 0.03, "event": 0.90, "reminder": 0.03,
                "idea": 0.01, "journal": 0.01, "update": 0.01, "question": 0.01,
            })
        )

        scores = await classifier.classify("Meeting with Sarah at 3pm tomorrow")
        assert scores[0].intent == BlurtIntent.EVENT

    async def test_classify_ambiguous_input(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Test classification where scores are close."""
        mock_client.generate.return_value = _make_gemini_response(
            _make_classification_json("task", {
                "task": 0.35, "event": 0.30, "reminder": 0.15,
                "idea": 0.05, "journal": 0.05, "update": 0.05, "question": 0.05,
            })
        )

        scores = await classifier.classify("Doctor appointment next week")
        assert len(scores) == 7
        # Top score should be below threshold
        assert scores[0].confidence < 0.85

    async def test_classify_all_intents_present(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Ensure all 7 intents are always returned even if model omits some."""
        # Response with only 3 intents
        mock_client.generate.return_value = _make_gemini_response(
            json.dumps({
                "primary_intent": "idea",
                "confidence_scores": {"idea": 0.8, "journal": 0.1, "task": 0.1},
            })
        )

        scores = await classifier.classify("What if we built a spaceship?")
        assert len(scores) == 7

        intent_set = {s.intent for s in scores}
        for intent in BlurtIntent:
            assert intent in intent_set

    async def test_classify_normalizes_scores(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Scores should be normalized to sum to ~1.0."""
        mock_client.generate.return_value = _make_gemini_response(
            json.dumps({
                "primary_intent": "task",
                "confidence_scores": {
                    "task": 5.0, "event": 1.0, "reminder": 1.0,
                    "idea": 1.0, "journal": 1.0, "update": 0.5, "question": 0.5,
                },
            })
        )

        scores = await classifier.classify("Do the laundry")
        total = sum(s.confidence for s in scores)
        assert abs(total - 1.0) < 0.02

    async def test_classify_invalid_json_raises(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Invalid JSON response should raise ClassificationError."""
        mock_client.generate.return_value = _make_gemini_response("not json at all")

        with pytest.raises(ClassificationError, match="Invalid JSON"):
            await classifier.classify("hello")

    async def test_classify_fallback_primary_only(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """If confidence_scores is missing, build from primary_intent alone."""
        mock_client.generate.return_value = _make_gemini_response(
            json.dumps({"primary_intent": "reminder"})
        )

        scores = await classifier.classify("Remind me to call mom")
        assert scores[0].intent == BlurtIntent.REMINDER
        assert scores[0].confidence == pytest.approx(0.85, abs=0.01)
        assert len(scores) == 7

    async def test_classify_unknown_intent_defaults_to_journal(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Unknown intent name defaults to journal."""
        mock_client.generate.return_value = _make_gemini_response(
            json.dumps({"primary_intent": "unknown_nonsense"})
        )

        scores = await classifier.classify("blah blah")
        assert scores[0].intent == BlurtIntent.JOURNAL

    async def test_resolve_ambiguity(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Test ambiguity resolution with the smart model."""
        mock_client.generate.return_value = _make_gemini_response(
            json.dumps({
                "primary_intent": "event",
                "confidence": 0.92,
                "multi_intent": False,
                "intents": [{"intent": "event", "confidence": 0.92, "segment": "full"}],
                "reasoning": "Time reference indicates an event",
            }),
            model="flash",
        )

        result = await classifier.resolve_ambiguity("Doctor at 3pm next Tuesday")

        assert result["primary_intent"] == BlurtIntent.EVENT
        assert result["confidence"] == 0.92
        assert result["multi_intent"] is False

        # Should use SMART tier
        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs.get("tier") == ModelTier.SMART

    async def test_resolve_ambiguity_multi_intent(
        self, classifier: IntentClassifier, mock_client: MagicMock
    ) -> None:
        """Test resolution detecting multiple intents."""
        mock_client.generate.return_value = _make_gemini_response(
            json.dumps({
                "primary_intent": "task",
                "confidence": 0.88,
                "multi_intent": True,
                "intents": [
                    {"intent": "task", "confidence": 0.88, "segment": "buy groceries"},
                    {"intent": "reminder", "confidence": 0.85, "segment": "remind me at 5"},
                ],
                "reasoning": "Contains both a task and a time-bound reminder",
            })
        )

        result = await classifier.resolve_ambiguity("Buy groceries and remind me at 5")
        assert result["multi_intent"] is True
        assert len(result["intents"]) == 2


class TestIntentScore:
    """Tests for IntentScore clamping."""

    def test_confidence_clamped_to_0_1(self) -> None:
        from blurt.classification.models import IntentScore

        score = IntentScore(intent=BlurtIntent.TASK, confidence=1.5)
        assert score.confidence == 1.0

        score2 = IntentScore(intent=BlurtIntent.TASK, confidence=-0.5)
        assert score2.confidence == 0.0
