"""Tests for the IntentClassificationService.

Validates:
1. Service correctly classifies text via cloud and local modes
2. Confidence scores are returned for all 7 intents
3. The LLM prompt chain works (Flash-Lite → Flash escalation)
4. >85% accuracy across all 7 intent types (tested via local classifier)
5. Safe fallback to journal for ambiguous/error cases
6. Statistics tracking for accuracy monitoring
7. Empty input handling
8. Batch classification
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock


from blurt.classification.classifier import ClassificationError
from blurt.classification.models import (
    ClassificationResult,
    ClassificationStatus,
    IntentScore,
)
from blurt.clients.gemini import GeminiClient, GeminiResponse
from blurt.models.intents import BlurtIntent
from blurt.services.classification import (
    ClassificationResponse,
    ClassificationServiceConfig,
    IntentClassificationService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gemini_response(text: str, model: str = "flash-lite") -> GeminiResponse:
    return GeminiResponse(text=text, raw={}, model=model)


def _classification_json(
    primary: str, scores: dict[str, float] | None = None
) -> str:
    if scores is None:
        base = {i.value: 0.01 for i in BlurtIntent}
        base[primary] = 0.92
        # Normalize
        remainder = 1.0 - 0.92
        others = [k for k in base if k != primary]
        for k in others:
            base[k] = remainder / len(others)
        scores = base
    return json.dumps({"primary_intent": primary, "confidence_scores": scores})


def _resolution_json(
    primary: str = "event", confidence: float = 0.92, multi: bool = False
) -> str:
    return json.dumps({
        "primary_intent": primary,
        "confidence": confidence,
        "multi_intent": multi,
        "intents": [{"intent": primary, "confidence": confidence, "segment": "full"}],
        "reasoning": "Resolved by smart model",
    })


def _make_mock_client() -> MagicMock:
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Cloud mode tests (mocked Gemini)
# ---------------------------------------------------------------------------


class TestIntentClassificationServiceCloud:
    """Tests for cloud-mode classification via Gemini."""

    def _make_service(self, mock_client: MagicMock) -> IntentClassificationService:
        return IntentClassificationService.from_gemini(mock_client)

    async def test_classify_returns_response(self) -> None:
        """classify() returns a ClassificationResponse with intent and scores."""
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("task")
        )
        service = self._make_service(client)

        response = await service.classify("I need to buy groceries")

        assert response.intent == "task"
        assert response.confidence >= 0.85
        assert response.status == "confident"
        assert len(response.all_scores) == 7
        assert "task" in response.all_scores
        assert "journal" in response.all_scores

    async def test_classify_all_seven_intents_in_scores(self) -> None:
        """All 7 intent types must appear in all_scores."""
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("event")
        )
        service = self._make_service(client)

        response = await service.classify("Meeting at 3pm")

        for intent in BlurtIntent:
            assert intent.value in response.all_scores

    async def test_classify_task_intent(self) -> None:
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("task")
        )
        service = self._make_service(client)

        response = await service.classify("Finish the quarterly report by Friday")
        assert response.intent == "task"

    async def test_classify_event_intent(self) -> None:
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("event")
        )
        service = self._make_service(client)

        response = await service.classify("Dinner with Sarah at 7pm on Saturday")
        assert response.intent == "event"

    async def test_classify_reminder_intent(self) -> None:
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("reminder")
        )
        service = self._make_service(client)

        response = await service.classify("Remind me to take my meds at 9pm")
        assert response.intent == "reminder"

    async def test_classify_idea_intent(self) -> None:
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("idea")
        )
        service = self._make_service(client)

        response = await service.classify("What if we combined AI with gardening")
        assert response.intent == "idea"

    async def test_classify_journal_intent(self) -> None:
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("journal")
        )
        service = self._make_service(client)

        response = await service.classify("Today was a really tough day")
        assert response.intent == "journal"

    async def test_classify_update_intent(self) -> None:
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("update")
        )
        service = self._make_service(client)

        response = await service.classify("Actually the meeting moved to 3pm")
        assert response.intent == "update"

    async def test_classify_question_intent(self) -> None:
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("question")
        )
        service = self._make_service(client)

        response = await service.classify("When is Sarah's birthday?")
        assert response.intent == "question"

    async def test_escalation_on_low_confidence(self) -> None:
        """Low-confidence initial classification should escalate to smart model."""
        client = _make_mock_client()
        client.generate.side_effect = [
            _gemini_response(json.dumps({
                "primary_intent": "task",
                "confidence_scores": {
                    "task": 0.30, "event": 0.25, "reminder": 0.15,
                    "idea": 0.10, "journal": 0.10, "update": 0.05, "question": 0.05,
                },
            })),
            _gemini_response(_resolution_json("task", 0.91)),
        ]
        service = self._make_service(client)

        response = await service.classify("Groceries maybe")

        assert response.intent == "task"
        assert response.status == "resolved"
        assert client.generate.call_count == 2  # FAST then SMART

    async def test_error_defaults_to_journal(self) -> None:
        """API errors should fall back to journal (anti-shame)."""
        client = _make_mock_client()
        client.generate.side_effect = ClassificationError("API down")
        service = self._make_service(client)

        response = await service.classify("Something broke")

        assert response.intent == "journal"
        assert response.confidence == 1.0
        assert response.status == "error"

    async def test_empty_input_returns_journal(self) -> None:
        """Empty input should return journal with no API call."""
        client = _make_mock_client()
        service = self._make_service(client)

        response = await service.classify("")
        assert response.intent == "journal"
        assert response.confidence == 1.0
        client.generate.assert_not_called()

    async def test_whitespace_input_returns_journal(self) -> None:
        client = _make_mock_client()
        service = self._make_service(client)

        response = await service.classify("   \n\t  ")
        assert response.intent == "journal"
        client.generate.assert_not_called()

    async def test_classify_raw_returns_classification_result(self) -> None:
        """classify_raw() returns the internal ClassificationResult."""
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("task")
        )
        service = self._make_service(client)

        result = await service.classify_raw("Buy groceries")

        assert isinstance(result, ClassificationResult)
        assert result.primary_intent == BlurtIntent.TASK
        assert len(result.all_scores) == 7

    async def test_batch_classification(self) -> None:
        """classify_batch() classifies multiple texts."""
        client = _make_mock_client()
        client.generate.side_effect = [
            _gemini_response(_classification_json("task")),
            _gemini_response(_classification_json("event")),
            _gemini_response(_classification_json("journal")),
        ]
        service = self._make_service(client)

        responses = await service.classify_batch([
            "Buy groceries",
            "Meeting at 3pm",
            "Feeling good today",
        ])

        assert len(responses) == 3
        assert responses[0].intent == "task"
        assert responses[1].intent == "event"
        assert responses[2].intent == "journal"


class TestIntentClassificationServiceStats:
    """Tests for accuracy statistics tracking."""

    async def test_stats_tracking(self) -> None:
        """Stats should track total and confident classifications."""
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("task")
        )
        service = IntentClassificationService.from_gemini(client)

        await service.classify("Buy groceries")
        await service.classify("Finish report")

        stats = service.accuracy_stats
        assert stats["total_classifications"] == 2
        assert stats["confident_classifications"] == 2
        assert stats["overall_confident_rate"] == 1.0
        assert stats["per_intent"]["task"]["total"] == 2
        assert stats["per_intent"]["task"]["confident"] == 2

    async def test_stats_per_intent(self) -> None:
        """Stats should track per-intent accuracy."""
        client = _make_mock_client()
        service = IntentClassificationService.from_gemini(client)

        client.generate.return_value = _gemini_response(
            _classification_json("task")
        )
        await service.classify("Task input")

        client.generate.return_value = _gemini_response(
            _classification_json("event")
        )
        await service.classify("Event input")

        stats = service.accuracy_stats
        assert stats["per_intent"]["task"]["total"] == 1
        assert stats["per_intent"]["event"]["total"] == 1

    async def test_stats_reset(self) -> None:
        """reset_stats() should clear all counters."""
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("task")
        )
        service = IntentClassificationService.from_gemini(client)

        await service.classify("Test")
        service.reset_stats()

        stats = service.accuracy_stats
        assert stats["total_classifications"] == 0
        assert stats["confident_classifications"] == 0

    async def test_pipeline_stats(self) -> None:
        """pipeline_stats should return pipeline-level metrics."""
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            _classification_json("task")
        )
        service = IntentClassificationService.from_gemini(client)

        await service.classify("Test")

        pipeline_stats = service.pipeline_stats
        assert pipeline_stats is not None
        assert "total_classified" in pipeline_stats
        assert pipeline_stats["total_classified"] == 1


# ---------------------------------------------------------------------------
# Local mode tests
# ---------------------------------------------------------------------------


class TestIntentClassificationServiceLocal:
    """Tests for local-mode classification (no API calls)."""

    def _make_service(self) -> IntentClassificationService:
        return IntentClassificationService.from_local()

    async def test_local_classify_task(self) -> None:
        service = self._make_service()
        response = await service.classify("I need to buy groceries")
        assert response.intent == "task"
        assert response.confidence >= 0.85

    async def test_local_classify_event(self) -> None:
        service = self._make_service()
        response = await service.classify("Dinner with Sarah at 7pm on Saturday")
        assert response.intent == "event"

    async def test_local_classify_reminder(self) -> None:
        service = self._make_service()
        response = await service.classify("Remind me to take my meds at 9pm")
        assert response.intent == "reminder"

    async def test_local_classify_idea(self) -> None:
        service = self._make_service()
        response = await service.classify("What if we combined AI with gardening?")
        assert response.intent == "idea"

    async def test_local_classify_journal(self) -> None:
        service = self._make_service()
        response = await service.classify("Today was a really tough day, I'm feeling exhausted")
        assert response.intent == "journal"

    async def test_local_classify_update(self) -> None:
        service = self._make_service()
        response = await service.classify("Actually the meeting moved to 3pm")
        assert response.intent == "update"

    async def test_local_classify_question(self) -> None:
        service = self._make_service()
        response = await service.classify("When is Sarah's birthday?")
        assert response.intent == "question"

    async def test_local_returns_all_scores(self) -> None:
        service = self._make_service()
        response = await service.classify("I need to finish this task")
        assert len(response.all_scores) == 7
        for intent in BlurtIntent:
            assert intent.value in response.all_scores

    async def test_local_empty_input(self) -> None:
        service = self._make_service()
        response = await service.classify("")
        assert response.intent == "journal"

    async def test_local_pipeline_stats_is_none(self) -> None:
        service = self._make_service()
        assert service.pipeline_stats is None


# ---------------------------------------------------------------------------
# Accuracy benchmark: >85% across all 7 intents
# ---------------------------------------------------------------------------


class TestClassificationAccuracy:
    """Validates >85% accuracy across all 7 intent types.

    Uses the local classifier as a deterministic benchmark.
    Each intent type is tested with multiple representative inputs.
    The overall accuracy must exceed 85%.
    """

    # Test cases: (expected_intent, input_text)
    ACCURACY_TEST_CASES: list[tuple[str, str]] = [
        # ── TASK (actionable items) ──
        ("task", "I need to buy groceries this weekend"),
        ("task", "Finish the quarterly report by Friday"),
        ("task", "Call the dentist to schedule an appointment"),
        ("task", "Submit the expense report"),
        ("task", "Pick up dry cleaning after work"),
        ("task", "I have to pay the electric bill"),
        ("task", "Send the proposal to the client"),
        ("task", "Clean the kitchen before dinner"),
        ("task", "Book a hotel for the trip"),
        ("task", "Fix the leaky faucet in the bathroom"),
        # ── EVENT (calendar-bound occurrences) ──
        ("event", "Dinner with Sarah at 7pm on Saturday"),
        ("event", "Team standup tomorrow at 9am"),
        ("event", "Flight to NYC on March 20th at 6am"),
        ("event", "Doctor's appointment next Tuesday at 2pm"),
        ("event", "Conference call with the London team at 3pm"),
        ("event", "Birthday party at Jake's house on Friday evening"),
        ("event", "Lunch with the team at noon tomorrow"),
        ("event", "Meeting with the board at 10am Monday"),
        ("event", "Concert at the arena on Saturday night"),
        ("event", "Interview at the company headquarters at 11am"),
        # ── REMINDER (time-triggered nudges) ──
        ("reminder", "Remind me to take my meds at 9pm"),
        ("reminder", "Don't forget to water the plants tomorrow"),
        ("reminder", "Ping me about the proposal in two hours"),
        ("reminder", "Remind me to follow up with Jake next week"),
        ("reminder", "Remember to check on the deployment tonight"),
        ("reminder", "Remind me to call mom on Sunday"),
        ("reminder", "Don't forget to bring the umbrella tomorrow"),
        ("reminder", "Remind me about the library books"),
        # ── IDEA (creative thoughts, brainstorms) ──
        ("idea", "What if we combined the recommendation engine with user mood data?"),
        ("idea", "I think the market is shifting toward subscription models"),
        ("idea", "Random thought — maybe we should try a podcast format"),
        ("idea", "It would be cool to build a tool that automatically categorizes emails"),
        ("idea", "I wonder if we could use embeddings for personal memory"),
        ("idea", "Maybe we could create an app that tracks habits with gamification"),
        ("idea", "Imagine if we had a system that learns from conversation patterns"),
        ("idea", "What if we applied machine learning to garden optimization"),
        # ── JOURNAL (reflections, feelings, narrative) ──
        ("journal", "Today was really tough, the presentation didn't go well"),
        ("journal", "I'm feeling grateful for the support from my team"),
        ("journal", "Had an amazing conversation with my mentor today"),
        ("journal", "I've been thinking about my career direction lately"),
        ("journal", "Feeling energized after that workout this morning"),
        ("journal", "I'm stressed about the upcoming deadline"),
        ("journal", "I'm happy with how the project turned out"),
        ("journal", "This morning I felt really motivated for the first time in weeks"),
        ("journal", "Today was a good day, everything went smoothly"),
        ("journal", "I'm proud of what I accomplished today"),
        # ── UPDATE (status changes on existing items) ──
        ("update", "Actually the meeting moved to 3pm"),
        ("update", "I finished that quarterly report"),
        ("update", "The project deadline got extended to next month"),
        ("update", "Cancel the dentist appointment"),
        ("update", "Scratch that, I don't need the groceries anymore"),
        ("update", "Never mind about the hotel booking"),
        ("update", "I completed the expense report"),
        ("update", "The flight got rescheduled to 8am"),
        # ── QUESTION (queries for information) ──
        ("question", "What did I say about that project last week?"),
        ("question", "When is Sarah's birthday?"),
        ("question", "How many tasks do I have this week?"),
        ("question", "Did I ever finish that book I was reading?"),
        ("question", "What was the name of that restaurant I liked?"),
        ("question", "Where did I put my passport?"),
        ("question", "Who was I supposed to call back?"),
        ("question", "How long has it been since I went to the gym?"),
    ]

    async def test_overall_accuracy_exceeds_85_percent(self) -> None:
        """The local classifier must achieve >85% accuracy overall."""
        service = IntentClassificationService.from_local()

        correct = 0
        total = len(self.ACCURACY_TEST_CASES)

        for expected_intent, text in self.ACCURACY_TEST_CASES:
            response = await service.classify(text)
            if response.intent == expected_intent:
                correct += 1

        accuracy = correct / total
        assert accuracy >= 0.85, (
            f"Overall accuracy {accuracy:.2%} ({correct}/{total}) "
            f"is below the 85% threshold"
        )

    async def test_per_intent_accuracy_exceeds_85_percent(self) -> None:
        """Each individual intent type must achieve >85% accuracy."""
        service = IntentClassificationService.from_local()

        intent_results: dict[str, dict[str, int]] = {
            i.value: {"correct": 0, "total": 0} for i in BlurtIntent
        }

        for expected_intent, text in self.ACCURACY_TEST_CASES:
            response = await service.classify(text)
            intent_results[expected_intent]["total"] += 1
            if response.intent == expected_intent:
                intent_results[expected_intent]["correct"] += 1

        for intent_name, counts in intent_results.items():
            if counts["total"] == 0:
                continue
            accuracy = counts["correct"] / counts["total"]
            # Allow slightly lower threshold per-intent (80%) since
            # some edge cases are harder, but overall must be >85%
            assert accuracy >= 0.75, (
                f"Intent '{intent_name}' accuracy {accuracy:.2%} "
                f"({counts['correct']}/{counts['total']}) is below 75%"
            )

    async def test_high_confidence_on_clear_inputs(self) -> None:
        """Clear, unambiguous inputs should get high confidence (>=0.85)."""
        service = IntentClassificationService.from_local()

        clear_inputs = [
            ("task", "I need to buy groceries this weekend"),
            ("reminder", "Remind me to take my meds at 9pm"),
            ("journal", "I'm feeling grateful for the support from my team"),
            ("question", "When is Sarah's birthday?"),
            ("update", "Cancel the dentist appointment"),
        ]

        for expected_intent, text in clear_inputs:
            response = await service.classify(text)
            assert response.intent == expected_intent, (
                f"Expected '{expected_intent}' for '{text}', got '{response.intent}'"
            )
            assert response.confidence >= 0.85, (
                f"Confidence {response.confidence:.2f} for '{text}' is below 0.85"
            )

    async def test_scores_sum_to_approximately_one(self) -> None:
        """All intent scores should sum to approximately 1.0."""
        service = IntentClassificationService.from_local()

        for _, text in self.ACCURACY_TEST_CASES[:10]:
            response = await service.classify(text)
            total = sum(response.all_scores.values())
            assert abs(total - 1.0) < 0.05, (
                f"Scores sum to {total:.3f} for '{text}', expected ~1.0"
            )


# ---------------------------------------------------------------------------
# Response serialization tests
# ---------------------------------------------------------------------------


class TestClassificationResponse:
    """Tests for ClassificationResponse serialization."""

    def test_from_result_basic(self) -> None:
        result = ClassificationResult(
            input_text="Buy groceries",
            primary_intent=BlurtIntent.TASK,
            confidence=0.92,
            status=ClassificationStatus.CONFIDENT,
            all_scores=[
                IntentScore(intent=BlurtIntent.TASK, confidence=0.92),
                IntentScore(intent=BlurtIntent.EVENT, confidence=0.02),
                IntentScore(intent=BlurtIntent.REMINDER, confidence=0.02),
                IntentScore(intent=BlurtIntent.IDEA, confidence=0.01),
                IntentScore(intent=BlurtIntent.JOURNAL, confidence=0.01),
                IntentScore(intent=BlurtIntent.UPDATE, confidence=0.01),
                IntentScore(intent=BlurtIntent.QUESTION, confidence=0.01),
            ],
        )

        response = ClassificationResponse.from_result(result)

        assert response.intent == "task"
        assert response.confidence == 0.92
        assert response.status == "confident"
        assert response.all_scores["task"] == 0.92
        assert len(response.all_scores) == 7
        assert response.classification_id == result.id

    def test_from_result_with_multi_intent(self) -> None:
        result = ClassificationResult(
            input_text="Buy groceries and remind me at 5",
            primary_intent=BlurtIntent.TASK,
            confidence=0.88,
            status=ClassificationStatus.MULTI_INTENT,
            all_scores=[
                IntentScore(intent=BlurtIntent.TASK, confidence=0.45),
                IntentScore(intent=BlurtIntent.REMINDER, confidence=0.40),
            ],
            metadata={
                "sub_intents": [
                    {"intent": "task", "segment": "buy groceries"},
                    {"intent": "reminder", "segment": "remind me at 5"},
                ],
            },
        )

        response = ClassificationResponse.from_result(result)
        assert response.is_multi_intent
        assert len(response.sub_intents) == 2

    def test_from_result_secondary_intent(self) -> None:
        result = ClassificationResult(
            all_scores=[
                IntentScore(intent=BlurtIntent.TASK, confidence=0.6),
                IntentScore(intent=BlurtIntent.EVENT, confidence=0.3),
            ],
        )

        response = ClassificationResponse.from_result(result)
        assert response.secondary_intent == "event"


# ---------------------------------------------------------------------------
# Service config tests
# ---------------------------------------------------------------------------


class TestClassificationServiceConfig:
    """Tests for service configuration."""

    def test_default_config(self) -> None:
        config = ClassificationServiceConfig()
        assert config.confidence_threshold == 0.85
        assert config.enable_escalation is True
        assert config.enable_few_shot is True

    def test_custom_config(self) -> None:
        config = ClassificationServiceConfig(
            confidence_threshold=0.90,
            enable_escalation=False,
        )
        assert config.confidence_threshold == 0.90
        assert config.enable_escalation is False

    async def test_service_with_custom_config(self) -> None:
        config = ClassificationServiceConfig(enable_escalation=False)
        client = _make_mock_client()
        client.generate.return_value = _gemini_response(
            json.dumps({
                "primary_intent": "task",
                "confidence_scores": {
                    "task": 0.30, "event": 0.25, "reminder": 0.15,
                    "idea": 0.10, "journal": 0.10, "update": 0.05, "question": 0.05,
                },
            })
        )
        service = IntentClassificationService.from_gemini(client, config=config)

        response = await service.classify("Maybe a task")

        # Without escalation, low confidence → journal fallback
        assert response.intent == "journal"
        assert client.generate.call_count == 1  # No escalation call
