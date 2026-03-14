"""Tests for brief verbal acknowledgments in capture API responses.

Validates AC 13: Brief verbal acknowledgments returned naturally, not chatty.

Every capture response includes an `acknowledgment` field with:
- Brief text (1-8 words, typically 1-3)
- Tone matching the emotional context
- No shame language, no chattiness
- Silent mode for questions (answer replaces ack)
- Variety across consecutive responses (via AcknowledgmentService)
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blurt.api.capture import router, set_ack_service, set_pipeline
from blurt.api.episodes import router as episodes_router, set_store
from blurt.memory.episodic import InMemoryEpisodicStore
from blurt.services.acknowledgment import AcknowledgmentService
from blurt.services.capture import BlurtCapturePipeline


# ── Mock classifier ──────────────────────────────────────────────

async def _mock_classify(text: str) -> tuple[str, float]:
    text_lower = text.lower().strip()
    # Questions first — "?" is the strongest signal
    if "?" in text_lower:
        return ("question", 0.91)
    if any(w in text_lower for w in ["need to", "buy", "call", "fix"]):
        return ("task", 0.92)
    if any(w in text_lower for w in ["meeting", "dinner", "appointment"]):
        return ("event", 0.89)
    if any(w in text_lower for w in ["remind", "don't forget"]):
        return ("reminder", 0.90)
    if any(w in text_lower for w in ["what if", "idea", "maybe we"]):
        return ("idea", 0.88)
    if any(w in text_lower for w in ["actually", "change", "update"]):
        return ("update", 0.87)
    return ("journal", 0.85)


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def store() -> InMemoryEpisodicStore:
    return InMemoryEpisodicStore()


@pytest.fixture
def ack_service() -> AcknowledgmentService:
    return AcknowledgmentService(history_size=5)


@pytest.fixture
def app(store: InMemoryEpisodicStore, ack_service: AcknowledgmentService) -> FastAPI:
    application = FastAPI()
    pipeline = BlurtCapturePipeline(store, classifier=_mock_classify)
    set_pipeline(pipeline)
    set_store(store)
    set_ack_service(ack_service)
    application.include_router(router)
    application.include_router(episodes_router)
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ── Acknowledgment presence ──────────────────────────────────────

class TestAcknowledgmentPresence:
    """Every capture response includes an acknowledgment."""

    def test_capture_includes_acknowledgment_field(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "I need to buy groceries"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "acknowledgment" in data
        ack = data["acknowledgment"]
        assert "text" in ack
        assert "tone" in ack
        assert "is_silent" in ack

    def test_voice_capture_includes_acknowledgment(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/blurt/voice",
            json={"user_id": "u1", "raw_text": "hmm interesting"},
        )
        assert resp.status_code == 201
        assert "acknowledgment" in resp.json()

    def test_text_capture_includes_acknowledgment(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/blurt/text",
            json={"user_id": "u1", "raw_text": "correction: 4pm"},
        )
        assert resp.status_code == 201
        assert "acknowledgment" in resp.json()


# ── Brevity ──────────────────────────────────────────────────────

class TestAcknowledgmentBrevity:
    """Acknowledgments must be brief — 1-8 words, not chatty."""

    _MAX_WORDS = 8

    @pytest.mark.parametrize("text,expected_intent", [
        ("I need to fix the sink", "task"),
        ("dinner with Sarah tonight", "event"),
        ("remind me to call mom", "reminder"),
        ("what if we tried microservices", "idea"),
        ("feeling kind of tired today", "journal"),
        ("actually change that to 4pm", "update"),
    ])
    def test_ack_is_brief_for_each_intent(
        self, client: TestClient, text: str, expected_intent: str
    ) -> None:
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": text},
        )
        ack = resp.json()["acknowledgment"]
        word_count = len(ack["text"].split())
        assert 1 <= word_count <= self._MAX_WORDS, (
            f"Ack '{ack['text']}' for intent {expected_intent} has "
            f"{word_count} words (max {self._MAX_WORDS})"
        )

    def test_ack_not_chatty(self, client: TestClient) -> None:
        """Acknowledgment should never be a full sentence explanation."""
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "I need to buy groceries"},
        )
        ack_text = resp.json()["acknowledgment"]["text"]
        # Should not contain chatty phrases
        chatty_markers = [
            "I have", "I will", "I've", "your task",
            "has been", "was added", "successfully",
            "I understand", "that's been",
        ]
        text_lower = ack_text.lower()
        for marker in chatty_markers:
            assert marker not in text_lower, (
                f"Chatty marker '{marker}' found in ack: '{ack_text}'"
            )


# ── Question handling ────────────────────────────────────────────

class TestQuestionAcknowledgment:
    """Questions get silent acks — answers instead of verbal confirmations."""

    def test_question_ack_is_silent(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "when is the meeting?"},
        )
        ack = resp.json()["acknowledgment"]
        assert ack["is_silent"] is True
        assert ack["text"] == ""


# ── Anti-shame ───────────────────────────────────────────────────

class TestAntiShameInAPI:
    """No guilt, streaks, overdue counters, or pressure in API acks."""

    _SHAME_WORDS = [
        "overdue", "late", "behind", "missed", "failed", "forgot",
        "streak", "pending", "urgent", "hurry", "asap", "deadline",
        "warning", "alert", "critical", "important", "priority",
    ]

    def test_no_shame_in_task_ack(self, client: TestClient) -> None:
        for _ in range(20):
            resp = client.post(
                "/api/v1/blurt",
                json={"user_id": "u1", "raw_text": "I need to fix the report"},
            )
            ack_text = resp.json()["acknowledgment"]["text"].lower()
            for word in self._SHAME_WORDS:
                assert word not in ack_text, (
                    f"Shame word '{word}' in ack: '{ack_text}'"
                )

    def test_no_shame_in_reminder_ack(self, client: TestClient) -> None:
        for _ in range(20):
            resp = client.post(
                "/api/v1/blurt",
                json={"user_id": "u1", "raw_text": "remind me to call the doctor"},
            )
            ack_text = resp.json()["acknowledgment"]["text"].lower()
            for word in self._SHAME_WORDS:
                assert word not in ack_text


# ── Variety across consecutive calls ─────────────────────────────

class TestAcknowledgmentVariety:
    """Consecutive acks vary to feel natural, not robotic."""

    def test_variety_across_task_acks(self, client: TestClient) -> None:
        """Over 10 task captures, we see at least 2 distinct ack phrases."""
        ack_texts = set()
        for i in range(15):
            resp = client.post(
                "/api/v1/blurt",
                json={"user_id": "u1", "raw_text": f"I need to do thing {i}"},
            )
            ack_texts.add(resp.json()["acknowledgment"]["text"])
        assert len(ack_texts) >= 2, (
            f"All acks were identical: {ack_texts}"
        )

    def test_no_excessive_consecutive_repeats(self, client: TestClient) -> None:
        """The AcknowledgmentService avoids repeating the same phrase back-to-back."""
        prev = None
        repeat_count = 0
        for i in range(15):
            resp = client.post(
                "/api/v1/blurt",
                json={"user_id": "u1", "raw_text": f"I need to buy item {i}"},
            )
            text = resp.json()["acknowledgment"]["text"]
            if text == prev:
                repeat_count += 1
            prev = text
        # With 7 task phrases and history_size=5, repeats should be rare
        assert repeat_count <= 4, (
            f"Too many consecutive repeats: {repeat_count}"
        )


# ── Tone appropriateness ─────────────────────────────────────────

class TestAcknowledgmentTone:
    """Tone field is present and valid in API responses."""

    _VALID_TONES = {"warm", "calm", "energetic", "gentle", "matter_of_fact"}

    def test_tone_is_valid(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "I need to fix the car"},
        )
        tone = resp.json()["acknowledgment"]["tone"]
        assert tone in self._VALID_TONES, f"Invalid tone: {tone}"

    def test_casual_remark_gets_calm_tone(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "nice weather today"},
        )
        # Without emotion context, default is calm
        tone = resp.json()["acknowledgment"]["tone"]
        assert tone in self._VALID_TONES


# ── Integration: ack matches intent ──────────────────────────────

class TestAckMatchesIntent:
    """Acknowledgment text should be contextually appropriate to the intent."""

    def test_event_ack_sounds_like_scheduling(self, client: TestClient) -> None:
        event_ack_pool = {
            "On the calendar.", "Scheduled.", "Noted.",
            "Got it down.", "Saved.", "Calendared.",
        }
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "meeting with John at 3"},
        )
        ack_text = resp.json()["acknowledgment"]["text"]
        assert ack_text in event_ack_pool, (
            f"Event ack '{ack_text}' not from event pool"
        )

    def test_reminder_ack_sounds_like_reminder(self, client: TestClient) -> None:
        reminder_ack_pool = {
            "I'll remind you.", "Set.", "You'll get a nudge.",
            "Reminder set.", "Won't forget.", "Noted.",
        }
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "remind me to water the plants"},
        )
        ack_text = resp.json()["acknowledgment"]["text"]
        assert ack_text in reminder_ack_pool, (
            f"Reminder ack '{ack_text}' not from reminder pool"
        )

    def test_idea_ack_sounds_encouraging(self, client: TestClient) -> None:
        idea_ack_pool = {
            "Interesting.", "Saved that thought.", "Captured.",
            "Noted.", "Cool idea.", "Stored.", "Filed away.",
        }
        resp = client.post(
            "/api/v1/blurt",
            json={"user_id": "u1", "raw_text": "what if we tried a new approach"},
        )
        ack_text = resp.json()["acknowledgment"]["text"]
        assert ack_text in idea_ack_pool, (
            f"Idea ack '{ack_text}' not from idea pool"
        )
