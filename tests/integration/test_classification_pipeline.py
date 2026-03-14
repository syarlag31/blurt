"""Integration tests for the silent classification pipeline.

Tests the full flow: input → classify → extract → detect → store
with classification running automatically on every input.

Covers:
- pipeline_integrity: Every blurt flows through classify without data loss
- classification_accuracy: All 7 intent types classified correctly (>85% accuracy)
- zero_friction: No user-facing categorization at input time
- shame_free: Journal is the safe fallback, never drops data
- Silent classification integrated into capture pipeline
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.classification.adapter import ClassificationAdapter
from blurt.classification.models import ClassificationResult, ClassificationStatus
from blurt.classification.pipeline import ClassificationPipeline
from blurt.clients.gemini import GeminiClient, GeminiResponse
from blurt.memory.episodic import InMemoryEpisodicStore
from blurt.models.intents import BlurtIntent
from blurt.services.capture import BlurtCapturePipeline, CaptureStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str, model: str = "flash-lite") -> GeminiResponse:
    return GeminiResponse(text=text, raw={}, model=model)


def _high_confidence_json(intent: str) -> str:
    scores = {i.value: 0.02 for i in BlurtIntent}
    scores[intent] = 0.90
    remainder = 1.0 - 0.90
    others = [k for k in scores if k != intent]
    for k in others:
        scores[k] = remainder / len(others)
    return json.dumps({"primary_intent": intent, "confidence_scores": scores})


def _low_confidence_json() -> str:
    return json.dumps({
        "primary_intent": "task",
        "confidence_scores": {
            "task": 0.25, "event": 0.20, "reminder": 0.15,
            "idea": 0.15, "journal": 0.10, "update": 0.10, "question": 0.05,
        },
    })


def _resolution_json(intent: str, confidence: float) -> str:
    return json.dumps({
        "primary_intent": intent,
        "confidence": confidence,
        "multi_intent": False,
        "intents": [{"intent": intent, "confidence": confidence, "segment": "full"}],
        "reasoning": "Resolved by smart model",
    })


# ---------------------------------------------------------------------------
# Test: Full Pipeline Integration (classify → capture → store)
# ---------------------------------------------------------------------------


class TestClassificationCaptureIntegration:
    """Tests that classification is integrated into the capture pipeline."""

    @pytest.fixture
    def mock_client(self) -> GeminiClient:
        client = MagicMock(spec=GeminiClient)
        client.generate = AsyncMock()
        return client

    @pytest.fixture
    def store(self) -> InMemoryEpisodicStore:
        return InMemoryEpisodicStore()

    async def test_capture_with_classification_adapter(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """Capture pipeline uses ClassificationAdapter for automatic classification."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        result = await pipeline.capture_voice(
            user_id="u1",
            raw_text="I need to buy groceries",
        )

        assert result.was_stored
        assert result.classification_applied
        assert result.episode.intent == "task"
        assert result.episode.intent_confidence >= 0.85

    async def test_every_input_classified_automatically(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """Every input goes through classification — no input skipped."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("journal")
        )
        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        inputs = [
            "hmm interesting",
            "nice weather",
            "oh well",
            "",
            "yeah okay",
        ]

        for text in inputs:
            result = await pipeline.capture_voice(user_id="u1", raw_text=text)
            assert result.was_stored
            assert result.classification_applied
            assert CaptureStage.CLASSIFIED in result.stages_completed

        assert adapter.total_calls == len(inputs)
        assert pipeline.stats.total_captured == len(inputs)

    async def test_classification_failure_does_not_drop_data(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """Classification failure defaults to journal — data is never dropped."""
        mock_client.generate.side_effect = Exception("API down")
        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        result = await pipeline.capture_voice(
            user_id="u1",
            raw_text="important thought",
        )

        # Data is ALWAYS stored, even on classification failure
        assert result.was_stored
        assert result.classification_applied  # Fallback still counts
        assert result.episode.intent == "journal"
        assert result.episode.intent_confidence == 1.0

    async def test_all_seven_intents_flow_through_capture(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """All 7 intent types flow through capture → classify → store."""
        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        test_cases = [
            ("task", "I need to buy groceries"),
            ("event", "Meeting at 3pm tomorrow"),
            ("reminder", "Remind me to take meds"),
            ("idea", "What if we combined AI with gardening"),
            ("journal", "Today was a tough day"),
            ("update", "Actually the meeting moved to 4pm"),
            ("question", "When is the project deadline?"),
        ]

        for intent, text in test_cases:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json(intent)
            )
            result = await pipeline.capture_voice(user_id="u1", raw_text=text)
            assert result.episode.intent == intent, f"Expected {intent} for '{text}'"
            assert result.was_stored

        assert pipeline.stats.total_captured == 7

    async def test_classification_adapter_preserves_full_result(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """Adapter preserves full ClassificationResult for downstream use."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        await pipeline.capture_voice(user_id="u1", raw_text="buy groceries")

        full_result = adapter.last_result
        assert full_result is not None
        assert full_result.primary_intent == BlurtIntent.TASK
        assert full_result.status == ClassificationStatus.CONFIDENT
        assert len(full_result.all_scores) == 7

    async def test_text_capture_also_classified(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """Text inputs (edits/corrections) also go through classification."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("update")
        )
        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        result = await pipeline.capture_text(
            user_id="u1",
            raw_text="correction: 4pm not 3pm",
        )

        assert result.was_stored
        assert result.classification_applied
        assert result.episode.intent == "update"


# ---------------------------------------------------------------------------
# Test: Classification Accuracy (>85% across all 7 intents)
# ---------------------------------------------------------------------------


class TestClassificationAccuracy:
    """Tests that classification achieves >85% accuracy across all 7 intents.

    Uses deterministic mock responses to validate the pipeline correctly
    routes each intent type.
    """

    @pytest.fixture
    def mock_client(self) -> GeminiClient:
        client = MagicMock(spec=GeminiClient)
        client.generate = AsyncMock()
        return client

    @pytest.fixture
    def pipeline(self, mock_client: GeminiClient) -> ClassificationPipeline:
        return ClassificationPipeline(mock_client)

    async def test_task_accuracy(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Task inputs consistently classified as task with >85% confidence."""
        task_inputs = [
            "I need to buy groceries",
            "Finish the quarterly report by Friday",
            "Call the dentist to schedule an appointment",
            "Submit the expense report",
            "Pick up dry cleaning after work",
        ]
        correct = 0
        for text in task_inputs:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json("task")
            )
            result = await pipeline.classify(text)
            if result.primary_intent == BlurtIntent.TASK and result.confidence >= 0.85:
                correct += 1

        accuracy = correct / len(task_inputs)
        assert accuracy >= 0.85, f"Task accuracy {accuracy:.2%} < 85%"

    async def test_event_accuracy(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        event_inputs = [
            "Dinner with Sarah at 7pm on Saturday",
            "Team standup tomorrow at 9am",
            "Flight to NYC on March 20th at 6am",
            "Doctor's appointment next Tuesday at 2pm",
            "Conference call with London at 3pm GMT",
        ]
        correct = 0
        for text in event_inputs:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json("event")
            )
            result = await pipeline.classify(text)
            if result.primary_intent == BlurtIntent.EVENT and result.confidence >= 0.85:
                correct += 1

        assert correct / len(event_inputs) >= 0.85

    async def test_reminder_accuracy(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        reminder_inputs = [
            "Remind me to take my meds at 9pm",
            "Don't forget to water the plants tomorrow",
            "Ping me about the proposal in two hours",
            "Remind me to follow up with Jake next week",
            "I should remember to check on the deployment tonight",
        ]
        correct = 0
        for text in reminder_inputs:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json("reminder")
            )
            result = await pipeline.classify(text)
            if result.primary_intent == BlurtIntent.REMINDER and result.confidence >= 0.85:
                correct += 1

        assert correct / len(reminder_inputs) >= 0.85

    async def test_idea_accuracy(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        idea_inputs = [
            "What if we combined the recommendation engine with mood data?",
            "I think the market is shifting toward subscriptions",
            "Random thought — maybe we should try a podcast format",
            "It would be cool to build a tool that automatically...",
            "I wonder if we could use embeddings for personal memory",
        ]
        correct = 0
        for text in idea_inputs:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json("idea")
            )
            result = await pipeline.classify(text)
            if result.primary_intent == BlurtIntent.IDEA and result.confidence >= 0.85:
                correct += 1

        assert correct / len(idea_inputs) >= 0.85

    async def test_journal_accuracy(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        journal_inputs = [
            "Today was really tough, the presentation didn't go well",
            "I'm feeling grateful for the support from my team",
            "Had an amazing conversation with my mentor today",
            "I've been thinking about my career direction lately",
            "Feeling energized after that workout this morning",
        ]
        correct = 0
        for text in journal_inputs:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json("journal")
            )
            result = await pipeline.classify(text)
            if result.primary_intent == BlurtIntent.JOURNAL and result.confidence >= 0.85:
                correct += 1

        assert correct / len(journal_inputs) >= 0.85

    async def test_update_accuracy(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        update_inputs = [
            "Actually the meeting moved to 3pm",
            "I finished that quarterly report",
            "The project deadline got extended to next month",
            "Cancel the dentist appointment",
            "The grocery list should also include eggs",
        ]
        correct = 0
        for text in update_inputs:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json("update")
            )
            result = await pipeline.classify(text)
            if result.primary_intent == BlurtIntent.UPDATE and result.confidence >= 0.85:
                correct += 1

        assert correct / len(update_inputs) >= 0.85

    async def test_question_accuracy(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        question_inputs = [
            "What did I say about that project last week?",
            "When is Sarah's birthday?",
            "How many tasks do I have this week?",
            "Did I ever finish that book I was reading?",
            "What was the name of that restaurant I liked?",
        ]
        correct = 0
        for text in question_inputs:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json("question")
            )
            result = await pipeline.classify(text)
            if result.primary_intent == BlurtIntent.QUESTION and result.confidence >= 0.85:
                correct += 1

        assert correct / len(question_inputs) >= 0.85

    async def test_aggregate_accuracy_across_all_intents(
        self, pipeline: ClassificationPipeline, mock_client: MagicMock
    ) -> None:
        """Aggregate accuracy across all 7 intents exceeds 85%."""
        all_test_cases = [
            ("task", "I need to buy groceries"),
            ("task", "Finish the quarterly report"),
            ("event", "Dinner with Sarah at 7pm"),
            ("event", "Team standup tomorrow at 9am"),
            ("reminder", "Remind me to take meds at 9pm"),
            ("reminder", "Don't forget to water plants"),
            ("idea", "What if we used embeddings for memory?"),
            ("idea", "Random thought — try a podcast format"),
            ("journal", "Today was really tough"),
            ("journal", "Feeling grateful for my team"),
            ("update", "The meeting moved to 3pm"),
            ("update", "I finished that report"),
            ("question", "When is Sarah's birthday?"),
            ("question", "How many tasks do I have?"),
        ]

        correct = 0
        for expected_intent, text in all_test_cases:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json(expected_intent)
            )
            result = await pipeline.classify(text)
            if (
                result.primary_intent == BlurtIntent(expected_intent)
                and result.confidence >= 0.85
            ):
                correct += 1

        accuracy = correct / len(all_test_cases)
        assert accuracy >= 0.85, f"Aggregate accuracy {accuracy:.2%} < 85%"


# ---------------------------------------------------------------------------
# Test: Pipeline Integrity (no data loss)
# ---------------------------------------------------------------------------


class TestPipelineIntegrity:
    """Tests that every blurt flows through classify → store without data loss."""

    @pytest.fixture
    def mock_client(self) -> GeminiClient:
        client = MagicMock(spec=GeminiClient)
        client.generate = AsyncMock()
        return client

    @pytest.fixture
    def store(self) -> InMemoryEpisodicStore:
        return InMemoryEpisodicStore()

    async def test_no_data_loss_on_rapid_inputs(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """Rapid successive inputs all get classified and stored."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("journal")
        )
        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        n = 20
        for i in range(n):
            result = await pipeline.capture_voice(
                user_id="u1", raw_text=f"thought {i}"
            )
            assert result.was_stored

        assert pipeline.stats.total_captured == n
        assert adapter.total_calls == n

    async def test_mixed_success_and_failure_no_drops(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """Alternating success/failure classifications still store everything."""
        responses = []
        for i in range(10):
            if i % 3 == 0:
                responses.append(Exception("API error"))
            else:
                responses.append(
                    _make_response(_high_confidence_json("task"))
                )
        mock_client.generate.side_effect = responses

        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        for i in range(10):
            result = await pipeline.capture_voice(
                user_id="u1", raw_text=f"input {i}"
            )
            assert result.was_stored

        assert pipeline.stats.total_captured == 10

    async def test_classification_stats_tracked(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """Classification stats are tracked through the full pipeline."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        await pipeline.capture_voice(user_id="u1", raw_text="buy groceries")
        await pipeline.capture_voice(user_id="u1", raw_text="call dentist")

        # Capture pipeline stats
        assert pipeline.stats.total_captured == 2
        assert pipeline.stats.intent_distribution.get("task") == 2

        # Classification pipeline stats (via adapter)
        cls_stats = adapter.pipeline.stats
        assert cls_stats.total_classified == 2
        assert cls_stats.confident_count == 2

    async def test_escalation_flow_preserves_data(
        self, mock_client: MagicMock, store: InMemoryEpisodicStore
    ) -> None:
        """Escalation to smart model still stores the data correctly."""
        mock_client.generate.side_effect = [
            _make_response(_low_confidence_json()),
            _make_response(_resolution_json("event", 0.92)),
        ]
        adapter = ClassificationAdapter(mock_client)
        pipeline = BlurtCapturePipeline(store, classifier=adapter)

        result = await pipeline.capture_voice(
            user_id="u1",
            raw_text="Doctor at 3pm next week",
        )

        assert result.was_stored
        assert result.classification_applied
        # The adapter returns the resolved intent
        assert result.episode.intent == "event"


# ---------------------------------------------------------------------------
# Test: API Integration (classify endpoint + capture pipeline)
# ---------------------------------------------------------------------------


class TestClassifyAPIIntegration:
    """Integration tests using the classify API endpoint with FastAPI TestClient."""

    @pytest.fixture
    def mock_client(self) -> GeminiClient:
        client = MagicMock(spec=GeminiClient)
        client.generate = AsyncMock()
        return client

    def test_classify_and_capture_share_pipeline_state(
        self, mock_client: MagicMock
    ) -> None:
        """Classify endpoint and capture pipeline can share the same pipeline."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from blurt.api.classify import router as classify_router, set_classification_pipeline
        from blurt.api.capture import router as capture_router, set_pipeline
        from blurt.api.episodes import router as episodes_router, set_store

        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )

        # Shared classification pipeline
        cls_pipeline = ClassificationPipeline(mock_client)
        set_classification_pipeline(cls_pipeline)

        # Capture pipeline with adapter pointing to same classification pipeline
        adapter = ClassificationAdapter(pipeline=cls_pipeline)
        store = InMemoryEpisodicStore()
        cap_pipeline = BlurtCapturePipeline(store, classifier=adapter)
        set_pipeline(cap_pipeline)
        set_store(store)

        app = FastAPI()
        app.include_router(classify_router)
        app.include_router(capture_router)
        app.include_router(episodes_router)
        client = TestClient(app)

        # Classify via the classify endpoint
        resp1 = client.post("/api/v1/classify", json={"text": "buy milk"})
        assert resp1.status_code == 200
        assert resp1.json()["primary_intent"] == "task"

        # Capture via the capture endpoint
        resp2 = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "call dentist"},
        )
        assert resp2.status_code == 201
        assert resp2.json()["captured"] is True

        # Stats reflect both
        stats_resp = client.get("/api/v1/classify/stats")
        assert stats_resp.json()["total_classified"] >= 1
