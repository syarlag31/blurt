"""Tests for the classification pipeline controller."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.classification.classifier import ClassificationError
from blurt.classification.models import (
    CONFIDENCE_THRESHOLD,
    ClassificationResult,
    ClassificationStatus,
    FallbackStrategy,
    IntentScore,
)
from blurt.classification.pipeline import ClassificationPipeline
from blurt.clients.gemini import GeminiClient, GeminiResponse
from blurt.models.intents import BlurtIntent


def _make_response(text: str, model: str = "flash-lite") -> GeminiResponse:
    return GeminiResponse(text=text, raw={}, model=model)


def _high_confidence_scores(intent: str = "task") -> str:
    """Build JSON with a high-confidence primary intent."""
    scores = {i.value: 0.02 for i in BlurtIntent}
    scores[intent] = 0.90
    # Normalize remainder
    remainder = 1.0 - 0.90
    others = [k for k in scores if k != intent]
    for k in others:
        scores[k] = remainder / len(others)
    return json.dumps({"primary_intent": intent, "confidence_scores": scores})


def _low_confidence_scores() -> str:
    """Build JSON with low confidence across all intents."""
    return json.dumps({
        "primary_intent": "task",
        "confidence_scores": {
            "task": 0.25, "event": 0.20, "reminder": 0.15,
            "idea": 0.15, "journal": 0.10, "update": 0.10, "question": 0.05,
        },
    })


def _ambiguous_scores() -> str:
    """Build JSON where top-2 are close (ambiguous)."""
    return json.dumps({
        "primary_intent": "task",
        "confidence_scores": {
            "task": 0.38, "event": 0.35, "reminder": 0.10,
            "idea": 0.05, "journal": 0.05, "update": 0.04, "question": 0.03,
        },
    })


def _resolution_response(
    intent: str = "event", confidence: float = 0.92, multi: bool = False
) -> str:
    return json.dumps({
        "primary_intent": intent,
        "confidence": confidence,
        "multi_intent": multi,
        "intents": [{"intent": intent, "confidence": confidence, "segment": "full"}],
        "reasoning": "Resolved by smart model",
    })


@pytest.fixture
def mock_client() -> GeminiClient:
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock()
    return client


@pytest.fixture
def pipeline(mock_client: GeminiClient) -> ClassificationPipeline:
    return ClassificationPipeline(mock_client)


@pytest.fixture
def pipeline_no_escalation(mock_client: GeminiClient) -> ClassificationPipeline:
    return ClassificationPipeline(mock_client, enable_escalation=False)


class TestClassificationPipelineConfident:
    """Tests for confident classifications (above 85% threshold)."""

    async def test_confident_task_classification(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """High-confidence task should route directly."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("task")
        )

        result = await pipeline.classify("Buy milk from the store")

        assert result.primary_intent == BlurtIntent.TASK
        assert result.confidence >= CONFIDENCE_THRESHOLD
        assert result.status == ClassificationStatus.CONFIDENT
        assert result.is_confident
        assert not result.was_ambiguous
        assert result.resolution is None

    async def test_confident_event(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("event")
        )
        result = await pipeline.classify("Lunch with team at noon")
        assert result.primary_intent == BlurtIntent.EVENT
        assert result.status == ClassificationStatus.CONFIDENT

    async def test_confident_reminder(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("reminder")
        )
        result = await pipeline.classify("Remind me to take meds at 9pm")
        assert result.primary_intent == BlurtIntent.REMINDER
        assert result.status == ClassificationStatus.CONFIDENT

    async def test_confident_idea(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("idea")
        )
        result = await pipeline.classify("What if we used AI for gardening")
        assert result.primary_intent == BlurtIntent.IDEA
        assert result.status == ClassificationStatus.CONFIDENT

    async def test_confident_journal(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("journal")
        )
        result = await pipeline.classify("Today was a really long day")
        assert result.primary_intent == BlurtIntent.JOURNAL
        assert result.status == ClassificationStatus.CONFIDENT

    async def test_confident_update(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("update")
        )
        result = await pipeline.classify("Finished the report")
        assert result.primary_intent == BlurtIntent.UPDATE
        assert result.status == ClassificationStatus.CONFIDENT

    async def test_confident_question(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("question")
        )
        result = await pipeline.classify("When is the deadline for the project?")
        assert result.primary_intent == BlurtIntent.QUESTION
        assert result.status == ClassificationStatus.CONFIDENT

    async def test_all_seven_intents_classified(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """All 7 intent types should be present in all_scores."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("task")
        )
        result = await pipeline.classify("Do something")
        intent_set = {s.intent for s in result.all_scores}
        for intent in BlurtIntent:
            assert intent in intent_set


class TestClassificationPipelineFallback:
    """Tests for fallback/ambiguity logic when below 85% threshold."""

    async def test_low_confidence_escalates_to_smart(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Below threshold should escalate to the smart model."""
        # First call: low confidence from Flash-Lite
        # Second call: high confidence from Flash (resolution)
        mock_client.generate.side_effect = [
            _make_response(_low_confidence_scores()),
            _make_response(_resolution_response("task", 0.91)),
        ]

        result = await pipeline.classify("Something about groceries maybe")

        assert result.primary_intent == BlurtIntent.TASK
        assert result.status == ClassificationStatus.RESOLVED
        assert result.is_confident
        assert result.was_ambiguous
        assert result.resolution is not None
        assert result.resolution.strategy_used == FallbackStrategy.ESCALATE_TO_SMART
        assert result.resolution.resolution_model == "flash"

        # Two generate calls: FAST then SMART
        assert mock_client.generate.call_count == 2

    async def test_ambiguous_escalates_to_smart(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Ambiguous (close top-2 scores) should escalate."""
        mock_client.generate.side_effect = [
            _make_response(_ambiguous_scores()),
            _make_response(_resolution_response("event", 0.93)),
        ]

        result = await pipeline.classify("Doctor at 3pm")

        assert result.primary_intent == BlurtIntent.EVENT
        assert result.status == ClassificationStatus.RESOLVED
        assert result.resolution is not None

    async def test_escalation_still_unsure_defaults_to_journal(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """If smart model is also unsure, default to journal (safe fallback)."""
        mock_client.generate.side_effect = [
            _make_response(_low_confidence_scores()),
            _make_response(_resolution_response("task", 0.60)),  # Still below threshold
        ]

        result = await pipeline.classify("Hmm not sure what to say")

        assert result.primary_intent == BlurtIntent.JOURNAL
        assert result.confidence == 1.0  # Safe default is 100% confident
        assert result.status == ClassificationStatus.LOW_CONFIDENCE

    async def test_no_escalation_defaults_to_journal(
        self, pipeline_no_escalation: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """With escalation disabled, low confidence defaults to journal immediately."""
        mock_client.generate.return_value = _make_response(_low_confidence_scores())

        result = await pipeline_no_escalation.classify("Unclear input")

        assert result.primary_intent == BlurtIntent.JOURNAL
        assert result.confidence == 1.0
        # Only one generate call (no escalation)
        assert mock_client.generate.call_count == 1

    async def test_multi_intent_detection(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Smart model can detect multiple intents in one input."""
        mock_client.generate.side_effect = [
            _make_response(_ambiguous_scores()),
            _make_response(json.dumps({
                "primary_intent": "task",
                "confidence": 0.88,
                "multi_intent": True,
                "intents": [
                    {"intent": "task", "confidence": 0.88, "segment": "buy groceries"},
                    {"intent": "reminder", "confidence": 0.85, "segment": "remind at 5pm"},
                ],
                "reasoning": "Both task and reminder present",
            })),
        ]

        result = await pipeline.classify("Buy groceries and remind me at 5pm")

        assert result.status == ClassificationStatus.MULTI_INTENT
        assert result.is_multi_intent
        assert result.primary_intent == BlurtIntent.TASK
        assert "sub_intents" in result.metadata

    async def test_escalation_failure_defaults_to_journal(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """If escalation fails entirely, fall back to journal."""
        mock_client.generate.side_effect = [
            _make_response(_low_confidence_scores()),
            Exception("API error"),
        ]

        result = await pipeline.classify("Something went wrong")

        assert result.primary_intent == BlurtIntent.JOURNAL
        assert result.confidence == 1.0
        assert result.resolution is not None
        assert result.resolution.strategy_used == FallbackStrategy.DEFAULT_JOURNAL


class TestClassificationPipelineError:
    """Tests for error handling in the pipeline."""

    async def test_classification_error_defaults_to_journal(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Classification errors should produce a journal fallback, not crash."""
        mock_client.generate.side_effect = ClassificationError("Model unavailable")

        result = await pipeline.classify("Test input")

        assert result.primary_intent == BlurtIntent.JOURNAL
        assert result.status == ClassificationStatus.ERROR
        assert "error" in result.metadata

    async def test_unexpected_error_defaults_to_journal(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Unexpected exceptions should produce a journal fallback."""
        mock_client.generate.side_effect = RuntimeError("Something unexpected")

        result = await pipeline.classify("Test input")

        assert result.primary_intent == BlurtIntent.JOURNAL
        assert result.status == ClassificationStatus.ERROR


class TestClassificationPipelineDownstream:
    """Tests for downstream handler routing."""

    async def test_handler_called_on_success(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Downstream handler should be called with the result."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("task")
        )

        handler = AsyncMock()
        pipeline.register_handler(handler)

        result = await pipeline.classify("Buy groceries")

        handler.assert_called_once_with(result)

    async def test_multiple_handlers_called(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """All registered handlers should be called."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("task")
        )

        handler1 = AsyncMock()
        handler2 = AsyncMock()
        pipeline.register_handler(handler1)
        pipeline.register_handler(handler2)

        await pipeline.classify("Buy groceries")

        handler1.assert_called_once()
        handler2.assert_called_once()

    async def test_handler_error_does_not_break_pipeline(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """A failing handler should not prevent other handlers or crash."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("task")
        )

        failing_handler = AsyncMock(side_effect=RuntimeError("handler broke"))
        good_handler = AsyncMock()
        pipeline.register_handler(failing_handler)
        pipeline.register_handler(good_handler)

        result = await pipeline.classify("Buy groceries")

        # Good handler should still be called
        good_handler.assert_called_once()
        assert result.primary_intent == BlurtIntent.TASK

    async def test_handler_called_on_error_fallback(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Handlers should be called even when classification falls back to journal."""
        mock_client.generate.side_effect = ClassificationError("fail")

        handler = AsyncMock()
        pipeline.register_handler(handler)

        result = await pipeline.classify("Test")

        handler.assert_called_once_with(result)
        assert result.primary_intent == BlurtIntent.JOURNAL

    async def test_unregister_handler(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Unregistered handlers should not be called."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("task")
        )

        handler = AsyncMock()
        pipeline.register_handler(handler)
        pipeline.unregister_handler(handler)

        await pipeline.classify("Test")

        handler.assert_not_called()


class TestClassificationPipelineStats:
    """Tests for pipeline statistics tracking."""

    async def test_stats_increment_on_confident(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("task")
        )

        await pipeline.classify("Buy groceries")

        assert pipeline.stats.total_classified == 1
        assert pipeline.stats.confident_count == 1
        assert pipeline.stats.error_count == 0

    async def test_stats_increment_on_escalation(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.side_effect = [
            _make_response(_low_confidence_scores()),
            _make_response(_resolution_response("task", 0.91)),
        ]

        await pipeline.classify("Maybe a task")

        assert pipeline.stats.total_classified == 1
        assert pipeline.stats.escalated_count == 1

    async def test_stats_increment_on_error(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.side_effect = ClassificationError("boom")

        await pipeline.classify("Crash")

        assert pipeline.stats.total_classified == 1
        assert pipeline.stats.error_count == 1

    async def test_stats_avg_latency(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("task")
        )

        await pipeline.classify("A")
        await pipeline.classify("B")

        assert pipeline.stats.total_classified == 2
        assert pipeline.stats.avg_latency_ms >= 0

    async def test_stats_to_dict(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_scores("task")
        )
        await pipeline.classify("Test")

        d = pipeline.stats.to_dict()
        assert "total_classified" in d
        assert "confident_rate" in d
        assert "avg_latency_ms" in d


class TestClassificationResult:
    """Tests for ClassificationResult properties."""

    def test_is_confident(self) -> None:
        r = ClassificationResult(
            status=ClassificationStatus.CONFIDENT,
            primary_intent=BlurtIntent.TASK,
            confidence=0.92,
        )
        assert r.is_confident

    def test_resolved_is_confident(self) -> None:
        r = ClassificationResult(
            status=ClassificationStatus.RESOLVED,
            primary_intent=BlurtIntent.EVENT,
            confidence=0.91,
        )
        assert r.is_confident

    def test_low_confidence_not_confident(self) -> None:
        r = ClassificationResult(
            status=ClassificationStatus.LOW_CONFIDENCE,
            primary_intent=BlurtIntent.JOURNAL,
            confidence=0.40,
        )
        assert not r.is_confident

    def test_secondary_intent(self) -> None:
        r = ClassificationResult(
            all_scores=[
                IntentScore(intent=BlurtIntent.TASK, confidence=0.5),
                IntentScore(intent=BlurtIntent.EVENT, confidence=0.3),
            ]
        )
        assert r.secondary_intent == BlurtIntent.EVENT

    def test_secondary_intent_none_when_single(self) -> None:
        r = ClassificationResult(
            all_scores=[IntentScore(intent=BlurtIntent.TASK, confidence=1.0)]
        )
        assert r.secondary_intent is None

    def test_metadata_attached(self) -> None:
        r = ClassificationResult(metadata={"session_id": "abc123"})
        assert r.metadata["session_id"] == "abc123"

    def test_has_id_and_timestamp(self) -> None:
        r = ClassificationResult()
        assert r.id  # Non-empty UUID
        assert r.timestamp is not None
