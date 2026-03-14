"""Tests for the classification API endpoints.

Covers:
- POST /api/v1/classify — single text classification
- POST /api/v1/classify/batch — batch classification
- GET  /api/v1/classify/stats — classification pipeline statistics
- GET  /api/v1/classify/health — classification pipeline health
- Silent classification (no user-facing categorization)
- All 7 intent types classified correctly
- Pipeline behavior: confidence thresholds, escalation, fallback
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blurt.api.classify import router, set_classification_pipeline
from blurt.classification.pipeline import ClassificationPipeline
from blurt.clients.gemini import GeminiClient, GeminiResponse
from blurt.models.intents import BlurtIntent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str, model: str = "flash-lite") -> GeminiResponse:
    return GeminiResponse(text=text, raw={}, model=model)


def _high_confidence_json(intent: str = "task") -> str:
    """Build JSON with a high-confidence primary intent."""
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


def _resolution_json(intent: str = "event", confidence: float = 0.92) -> str:
    return json.dumps({
        "primary_intent": intent,
        "confidence": confidence,
        "multi_intent": False,
        "intents": [{"intent": intent, "confidence": confidence, "segment": "full"}],
        "reasoning": "Resolved by smart model",
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> GeminiClient:
    client = MagicMock(spec=GeminiClient)
    client.generate = AsyncMock()
    return client


@pytest.fixture
def pipeline(mock_client: GeminiClient) -> ClassificationPipeline:
    return ClassificationPipeline(mock_client)


@pytest.fixture
def app(pipeline: ClassificationPipeline) -> FastAPI:
    application = FastAPI()
    set_classification_pipeline(pipeline)
    application.include_router(router)
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/v1/classify — single classification
# ---------------------------------------------------------------------------


class TestClassifyEndpoint:
    """Tests for the single classification endpoint."""

    def test_classify_task(self, client: TestClient, mock_client: MagicMock) -> None:
        """Clear task input classified correctly."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "I need to buy groceries"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["primary_intent"] == "task"
        assert data["confidence"] >= 0.85
        assert data["status"] == "confident"

    def test_classify_event(self, client: TestClient, mock_client: MagicMock) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("event")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "Meeting with Sarah at 3pm"},
        )
        assert resp.status_code == 200
        assert resp.json()["primary_intent"] == "event"

    def test_classify_reminder(self, client: TestClient, mock_client: MagicMock) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("reminder")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "Remind me to take meds at 9pm"},
        )
        assert resp.status_code == 200
        assert resp.json()["primary_intent"] == "reminder"

    def test_classify_idea(self, client: TestClient, mock_client: MagicMock) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("idea")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "What if we used AI for gardening?"},
        )
        assert resp.status_code == 200
        assert resp.json()["primary_intent"] == "idea"

    def test_classify_journal(self, client: TestClient, mock_client: MagicMock) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("journal")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "Today was a really long day"},
        )
        assert resp.status_code == 200
        assert resp.json()["primary_intent"] == "journal"

    def test_classify_update(self, client: TestClient, mock_client: MagicMock) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("update")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "Actually the meeting moved to 3pm"},
        )
        assert resp.status_code == 200
        assert resp.json()["primary_intent"] == "update"

    def test_classify_question(self, client: TestClient, mock_client: MagicMock) -> None:
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("question")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "When is the project deadline?"},
        )
        assert resp.status_code == 200
        assert resp.json()["primary_intent"] == "question"

    def test_classify_returns_all_seven_scores(
        self, client: TestClient, mock_client: MagicMock
    ) -> None:
        """All 7 intent scores are present in the response."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "Do something"},
        )
        scores = resp.json()["all_scores"]
        for intent in BlurtIntent:
            assert intent.value in scores

    def test_classify_silent_no_user_interaction(
        self, client: TestClient, mock_client: MagicMock
    ) -> None:
        """Classification is silent — no user-facing categorization needed."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "random input here"},
        )
        # Response always succeeds — never asks user for clarification
        assert resp.status_code == 200
        assert "primary_intent" in resp.json()

    def test_classify_with_metadata(
        self, client: TestClient, mock_client: MagicMock
    ) -> None:
        """Metadata is passed through to the classification result."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        resp = client.post(
            "/api/v1/classify",
            json={"text": "test", "metadata": {"session_id": "sess-1"}},
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"].get("session_id") == "sess-1"

    def test_classify_low_confidence_escalates(
        self, client: TestClient, mock_client: MagicMock
    ) -> None:
        """Low confidence triggers escalation to smart model."""
        mock_client.generate.side_effect = [
            _make_response(_low_confidence_json()),
            _make_response(_resolution_json("task", 0.91)),
        ]
        resp = client.post(
            "/api/v1/classify",
            json={"text": "something ambiguous"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "resolved"
        assert data["was_ambiguous"] is True

    def test_classify_error_falls_back_to_journal(
        self, client: TestClient, mock_client: MagicMock
    ) -> None:
        """Classification errors default to journal (shame-free fallback)."""
        mock_client.generate.side_effect = Exception("API error")
        resp = client.post(
            "/api/v1/classify",
            json={"text": "broken input"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["primary_intent"] == "journal"
        assert data["status"] == "error"


# ---------------------------------------------------------------------------
# POST /api/v1/classify/batch — batch classification
# ---------------------------------------------------------------------------


class TestBatchClassifyEndpoint:
    def test_batch_classify(self, client: TestClient, mock_client: MagicMock) -> None:
        """Batch classification classifies all texts."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        resp = client.post(
            "/api/v1/classify/batch",
            json={"texts": ["buy groceries", "call dentist", "finish report"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3
        for result in data["results"]:
            assert result["primary_intent"] == "task"

    def test_batch_classify_mixed_intents(
        self, client: TestClient, mock_client: MagicMock
    ) -> None:
        """Batch can classify different intents in one request."""
        mock_client.generate.side_effect = [
            _make_response(_high_confidence_json("task")),
            _make_response(_high_confidence_json("event")),
            _make_response(_high_confidence_json("journal")),
        ]
        resp = client.post(
            "/api/v1/classify/batch",
            json={"texts": ["buy milk", "lunch at noon", "feeling tired"]},
        )
        data = resp.json()
        intents = [r["primary_intent"] for r in data["results"]]
        assert "task" in intents
        assert "event" in intents
        assert "journal" in intents

    def test_batch_empty_list_rejected(self, client: TestClient) -> None:
        """Empty batch request is rejected."""
        resp = client.post(
            "/api/v1/classify/batch",
            json={"texts": []},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/classify/stats — pipeline statistics
# ---------------------------------------------------------------------------


class TestClassifyStatsEndpoint:
    def test_stats_empty(self, client: TestClient) -> None:
        resp = client.get("/api/v1/classify/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_classified"] == 0
        assert data["confident_rate"] == 0.0

    def test_stats_after_classifications(
        self, client: TestClient, mock_client: MagicMock
    ) -> None:
        """Stats update after classifications."""
        mock_client.generate.return_value = _make_response(
            _high_confidence_json("task")
        )
        # Classify a few items
        for text in ["buy milk", "call dentist"]:
            client.post("/api/v1/classify", json={"text": text})

        resp = client.get("/api/v1/classify/stats")
        data = resp.json()
        assert data["total_classified"] == 2
        assert data["confident_count"] == 2
        assert data["confident_rate"] > 0


# ---------------------------------------------------------------------------
# GET /api/v1/classify/health — pipeline health
# ---------------------------------------------------------------------------


class TestClassifyHealthEndpoint:
    def test_health_initialized(self, client: TestClient) -> None:
        resp = client.get("/api/v1/classify/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_initialized"] is True
        assert data["healthy"] is True

    def test_health_not_initialized(self) -> None:
        """Health reports unhealthy when pipeline not initialized."""
        set_classification_pipeline(None)  # type: ignore[arg-type]

        app = FastAPI()
        app.include_router(router)
        test_client = TestClient(app)

        resp = test_client.get("/api/v1/classify/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_initialized"] is False
        assert data["healthy"] is False
