"""E2E Scenario 2: Capture pipeline — zero-drop guarantee through HTTP.

Exercises the full POST /api/v1/blurt flow: HTTP request → FastAPI routing →
capture pipeline (classify → extract → detect → embed) → episodic store →
HTTP response.  Verifies the zero-drop contract and classification output.

Cross-cutting concerns exercised:
- Pipeline orchestration: classify → entity-extract → emotion-detect → embed
  stages execute in order with data flowing between them
- Zero-drop contract: every accepted blurt produces exactly one episode
- Intent classification: stub classifier maps keywords to intents correctly
- Entity extraction: NER-like extraction runs on raw text within the pipeline
- Emotion detection: valence/arousal detected and attached to the episode
- Embedding generation: vector embedding computed and stored alongside content
- HTTP serialization: pipeline output is correctly serialized to JSON response
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.asyncio


class TestCaptureVoiceE2E:
    """Voice capture through the full HTTP stack."""

    async def test_capture_task_utterance(
        self,
        capture_blurt_via_api: Any,
    ):
        """A task-like utterance is captured, classified, and stored."""
        result = await capture_blurt_via_api("I need to call the dentist")
        assert result["captured"] is True
        assert result["intent"] == "task"
        assert result["intent_confidence"] > 0.8
        assert result["episode"]["raw_text"] == "I need to call the dentist"

    async def test_capture_casual_remark(
        self,
        capture_blurt_via_api: Any,
    ):
        """Casual remarks are captured as journal — never dropped."""
        result = await capture_blurt_via_api("huh, nice weather today")
        assert result["captured"] is True
        assert result["intent"] == "journal"

    async def test_capture_empty_string(
        self,
        capture_blurt_via_api: Any,
    ):
        """Empty strings are still captured (zero-drop)."""
        result = await capture_blurt_via_api("")
        assert result["captured"] is True

    async def test_capture_event_utterance(
        self,
        capture_blurt_via_api: Any,
    ):
        """An event utterance is classified correctly."""
        result = await capture_blurt_via_api("I have a meeting at 3pm")
        assert result["captured"] is True
        assert result["intent"] == "event"

    async def test_capture_produces_entities(
        self,
        capture_blurt_via_api: Any,
    ):
        """Entity extraction runs and produces results."""
        result = await capture_blurt_via_api(
            "I need to call Alice about Project Alpha"
        )
        assert result["entities_extracted"] >= 1

    async def test_capture_produces_emotion(
        self,
        capture_blurt_via_api: Any,
    ):
        """Emotion detection runs and flags detection."""
        result = await capture_blurt_via_api("I'm so happy about this!")
        assert result["emotion_detected"] is True

    async def test_capture_text_modality(
        self,
        capture_blurt_via_api: Any,
    ):
        """Text modality capture works through the API."""
        result = await capture_blurt_via_api(
            "buy groceries", modality="text"
        )
        assert result["captured"] is True
        assert result["intent"] == "task"


class TestCaptureStatsE2E:
    """Pipeline statistics endpoint after captures."""

    async def test_stats_reflect_captures(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """Stats endpoint reflects captured blurts."""
        await capture_blurt_via_api("need to fix the bug")
        await capture_blurt_via_api("hmm interesting", modality="text")

        resp = await client.get("/api/v1/blurt/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_captured"] == 2
        assert stats["voice_count"] == 1
        assert stats["text_count"] == 1
        assert stats["drop_rate"] == 0.0
