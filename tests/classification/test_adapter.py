"""Tests for the ClassificationAdapter.

Covers:
- Adapter bridges ClassificationPipeline to capture pipeline protocol
- Every call returns (intent, confidence) tuple
- Full ClassificationResult accessible via last_result
- Callback invoked on every classification
- Error handling and safe fallback to journal
- Silent operation — no user-facing interaction
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.classification.adapter import ClassificationAdapter, create_classification_adapter
from blurt.classification.models import ClassificationResult, ClassificationStatus
from blurt.classification.pipeline import ClassificationPipeline
from blurt.clients.gemini import GeminiClient, GeminiResponse
from blurt.models.intents import BlurtIntent


def _make_response(text: str, model: str = "flash-lite") -> GeminiResponse:
    return GeminiResponse(text=text, raw={}, model=model)


def _high_confidence_json(intent: str = "task") -> str:
    scores = {i.value: 0.02 for i in BlurtIntent}
    scores[intent] = 0.90
    remainder = 1.0 - 0.90
    others = [k for k in scores if k != intent]
    for k in others:
        scores[k] = remainder / len(others)
    return json.dumps({"primary_intent": intent, "confidence_scores": scores})


@pytest.fixture
def mock_client() -> GeminiClient:
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock()
    return client


@pytest.fixture
def adapter(mock_client: GeminiClient) -> ClassificationAdapter:
    return ClassificationAdapter(mock_client)


class TestClassificationAdapter:
    """Tests for the ClassificationAdapter callable interface."""

    async def test_returns_intent_confidence_tuple(
        self, adapter: ClassificationAdapter, mock_client: MagicMock
    ) -> None:
        """Adapter returns (intent, confidence) matching capture pipeline protocol."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        intent, confidence = await adapter("I need to buy groceries")
        assert intent == "task"
        assert confidence >= 0.85

    async def test_last_result_contains_full_classification(
        self, adapter: ClassificationAdapter, mock_client: MagicMock
    ) -> None:
        """last_result provides the full ClassificationResult."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("event")
        )
        await adapter("Meeting at noon")
        result = adapter.last_result
        assert result is not None
        assert result.primary_intent == BlurtIntent.EVENT
        assert result.status == ClassificationStatus.CONFIDENT
        assert len(result.all_scores) == 7

    async def test_total_calls_increments(
        self, adapter: ClassificationAdapter, mock_client: MagicMock
    ) -> None:
        """total_calls tracks number of classifications."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        assert adapter.total_calls == 0
        await adapter("first")
        assert adapter.total_calls == 1
        await adapter("second")
        assert adapter.total_calls == 2

    async def test_callback_invoked(
        self, mock_client: MagicMock
    ) -> None:
        """on_classified callback receives the full ClassificationResult."""
        callback = AsyncMock()
        adapter = ClassificationAdapter(mock_client, on_classified=callback)

        mock_client.generate.return_value = _make_response(
            _high_confidence_json("idea")
        )
        await adapter("What if we built a spaceship?")

        callback.assert_called_once()
        result = callback.call_args[0][0]
        assert isinstance(result, ClassificationResult)
        assert result.primary_intent == BlurtIntent.IDEA

    async def test_callback_error_does_not_break_adapter(
        self, mock_client: MagicMock
    ) -> None:
        """Callback errors don't prevent classification from completing."""
        callback = AsyncMock(side_effect=RuntimeError("callback broke"))
        adapter = ClassificationAdapter(mock_client, on_classified=callback)

        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        intent, confidence = await adapter("buy groceries")

        # Still returns valid classification
        assert intent == "task"
        assert confidence >= 0.85

    async def test_error_falls_back_to_journal(
        self, adapter: ClassificationAdapter, mock_client: MagicMock
    ) -> None:
        """Classification errors fall back to journal (anti-shame)."""
        mock_client.generate.side_effect = Exception("API down")
        intent, confidence = await adapter("broken input")
        assert intent == "journal"
        assert confidence == 1.0

    async def test_all_seven_intents_classifiable(
        self, adapter: ClassificationAdapter, mock_client: MagicMock
    ) -> None:
        """All 7 intent types can be returned through the adapter."""
        for intent_type in BlurtIntent:
            mock_client.generate.return_value = _make_response(
                _high_confidence_json(intent_type.value)
            )
            intent, confidence = await adapter(f"test {intent_type.value}")
            assert intent == intent_type.value
            assert confidence >= 0.85

    async def test_pipeline_accessible(
        self, adapter: ClassificationAdapter
    ) -> None:
        """Underlying pipeline is accessible."""
        assert isinstance(adapter.pipeline, ClassificationPipeline)

    def test_create_with_pipeline(self, mock_client: MagicMock) -> None:
        """Can create adapter with an existing pipeline."""
        pipeline = ClassificationPipeline(mock_client)
        adapter = ClassificationAdapter(pipeline=pipeline)
        assert adapter.pipeline is pipeline

    def test_create_requires_client_or_pipeline(self) -> None:
        """Must provide either client or pipeline."""
        with pytest.raises(ValueError, match="Either client or pipeline"):
            ClassificationAdapter()  # type: ignore[call-arg]


class TestCreateClassificationAdapter:
    """Tests for the factory function."""

    def test_factory_creates_adapter(self, mock_client: MagicMock) -> None:
        adapter = create_classification_adapter(mock_client)
        assert isinstance(adapter, ClassificationAdapter)

    def test_factory_with_callback(self, mock_client: MagicMock) -> None:
        callback = AsyncMock()
        adapter = create_classification_adapter(mock_client, on_classified=callback)
        assert adapter is not None
