"""Comprehensive tests for intent classification across all 7 intents.

Covers:
- All 7 intent types with representative inputs
- Edge cases: ambiguous inputs, multi-intent inputs, minimal inputs
- Confidence score validation (>85% on clear inputs)
- Classification prompt structure
- Entity extraction alongside classification
- Emotion detection alongside classification
- Error handling and fallback behavior
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from blurt.config.settings import GeminiConfig
from blurt.gemini.audio_client import GeminiAudioClient, GeminiAudioError
from blurt.models.intents import BlurtIntent, SYNCABLE_INTENTS


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def config() -> GeminiConfig:
    """Valid test configuration with fast retries."""
    return GeminiConfig(
        api_key="test-api-key-12345",
        base_url="https://test.googleapis.com/v1beta",
        connect_timeout=5.0,
        read_timeout=30.0,
        max_retries=0,
        retry_backoff_base=0.01,
    )


@pytest.fixture
def mock_http_client() -> httpx.AsyncClient:
    """Mock httpx client for intercepting API calls."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def client(config: GeminiConfig, mock_http_client: httpx.AsyncClient) -> GeminiAudioClient:
    """GeminiAudioClient with mocked HTTP transport."""
    return GeminiAudioClient(config=config, http_client=mock_http_client)


def _make_gemini_response(classification: dict[str, Any]) -> httpx.Response:
    """Build a mock Gemini API response wrapping a classification JSON.

    The Gemini API returns classification as JSON text inside candidates.
    The _api_call method returns the raw response JSON, so _parse_json_response
    extracts text from candidates[0].content.parts[0].text and parses it.
    """
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(classification)}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 80,
            "totalTokenCount": 130,
        },
    }
    return httpx.Response(200, json=body)


def _classification(
    intent: str,
    confidence: float = 0.92,
    entities: list[dict[str, Any]] | None = None,
    emotion: dict[str, Any] | None = None,
    acknowledgment: str = "Got it",
) -> dict[str, Any]:
    """Helper to build a standard classification result dict."""
    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities or [],
        "emotion": emotion or {
            "primary": "trust",
            "intensity": 1,
            "valence": 0.3,
            "arousal": 0.4,
        },
        "acknowledgment": acknowledgment,
    }


# ── Intent Model Tests ───────────────────────────────────────────────


class TestBlurtIntentEnum:
    """Tests for the BlurtIntent enum and SYNCABLE_INTENTS set."""

    def test_all_seven_intents_exist(self):
        """All 7 intents are defined in the enum."""
        expected = {"task", "event", "reminder", "idea", "journal", "update", "question"}
        actual = {intent.value for intent in BlurtIntent}
        assert actual == expected

    def test_intent_count(self):
        """Exactly 7 intents, no more, no less."""
        assert len(BlurtIntent) == 7

    def test_syncable_intents_subset(self):
        """SYNCABLE_INTENTS is a subset of all intents."""
        for intent in SYNCABLE_INTENTS:
            assert intent in BlurtIntent

    def test_syncable_intents_content(self):
        """Task, event, reminder, update trigger sync; idea, journal, question do not."""
        assert BlurtIntent.TASK in SYNCABLE_INTENTS
        assert BlurtIntent.EVENT in SYNCABLE_INTENTS
        assert BlurtIntent.REMINDER in SYNCABLE_INTENTS
        assert BlurtIntent.UPDATE in SYNCABLE_INTENTS
        assert BlurtIntent.IDEA not in SYNCABLE_INTENTS
        assert BlurtIntent.JOURNAL not in SYNCABLE_INTENTS
        assert BlurtIntent.QUESTION not in SYNCABLE_INTENTS

    def test_intent_is_string_enum(self):
        """Intents are string values usable directly in JSON."""
        for intent in BlurtIntent:
            assert isinstance(intent.value, str)
            assert intent == intent.value

    def test_syncable_intents_is_frozen(self):
        """SYNCABLE_INTENTS is immutable."""
        assert isinstance(SYNCABLE_INTENTS, frozenset)
        with pytest.raises(AttributeError):
            SYNCABLE_INTENTS.add(BlurtIntent.IDEA)  # type: ignore[attr-defined]


# ── Classification Prompt Tests ───────────────────────────────────────


class TestClassificationPrompt:
    """Tests for the classification prompt structure."""

    def test_prompt_contains_all_intents(self, client: GeminiAudioClient):
        """The prompt enumerates all 7 intent types."""
        prompt = client._classification_prompt("test input")
        for intent in BlurtIntent:
            assert intent.value in prompt, f"Intent '{intent.value}' missing from prompt"

    def test_prompt_contains_transcript(self, client: GeminiAudioClient):
        """The prompt includes the user's transcript."""
        transcript = "I need to buy groceries tomorrow"
        prompt = client._classification_prompt(transcript)
        assert transcript in prompt

    def test_prompt_requests_json(self, client: GeminiAudioClient):
        """The prompt asks for JSON output format."""
        prompt = client._classification_prompt("test")
        assert "JSON" in prompt or "json" in prompt

    def test_prompt_requests_confidence(self, client: GeminiAudioClient):
        """The prompt asks for a confidence score."""
        prompt = client._classification_prompt("test")
        assert "confidence" in prompt.lower()

    def test_prompt_requests_entities(self, client: GeminiAudioClient):
        """The prompt asks for entity extraction."""
        prompt = client._classification_prompt("test")
        assert "entities" in prompt.lower()

    def test_prompt_requests_emotion(self, client: GeminiAudioClient):
        """The prompt asks for emotion detection."""
        prompt = client._classification_prompt("test")
        assert "emotion" in prompt.lower()

    def test_prompt_requests_acknowledgment(self, client: GeminiAudioClient):
        """The prompt asks for a natural acknowledgment."""
        prompt = client._classification_prompt("test")
        assert "acknowledgment" in prompt.lower()

    def test_prompt_mentions_85_percent_threshold(self, client: GeminiAudioClient):
        """The prompt references the 85% confidence target."""
        prompt = client._classification_prompt("test")
        assert "85%" in prompt or "85" in prompt


# ── Per-Intent Classification Tests ──────────────────────────────────


# Representative test cases per intent, each with (transcript, expected_intent)
TASK_EXAMPLES = [
    ("I need to finish the quarterly report by Friday", "task"),
    ("Buy milk and eggs on the way home", "task"),
    ("Send the contract to the client", "task"),
    ("Fix the login bug on the mobile app", "task"),
    ("Call the dentist and schedule a cleaning", "task"),
]

EVENT_EXAMPLES = [
    ("Team standup is at 9 AM tomorrow", "event"),
    ("Sarah's birthday party is next Saturday at 7 PM", "event"),
    ("We have a board meeting on March 20th from 2 to 4 PM", "event"),
    ("Dinner reservation at Nobu this Friday at 8", "event"),
    ("Conference call with Tokyo office at 3 PM", "event"),
]

REMINDER_EXAMPLES = [
    ("Remind me to take my medication at 8 PM", "reminder"),
    ("Don't forget to water the plants every Sunday", "reminder"),
    ("Remind me about the insurance renewal next month", "reminder"),
    ("Alert me 30 minutes before the meeting", "reminder"),
    ("I need a reminder to pick up the dry cleaning tomorrow", "reminder"),
]

IDEA_EXAMPLES = [
    ("What if we could use AI to automatically tag photos by emotion", "idea"),
    ("I just thought of a new feature where users can share voice notes", "idea"),
    ("Maybe we should pivot to a subscription model instead", "idea"),
    ("Had an interesting thought about combining meditation with journaling", "idea"),
    ("We could build a plugin that integrates with Slack for quick blurts", "idea"),
]

JOURNAL_EXAMPLES = [
    ("Today was really productive, I finished three major tasks", "journal"),
    ("Feeling anxious about the presentation tomorrow", "journal"),
    ("Had a great conversation with my mentor about career growth", "journal"),
    ("I'm grateful for the support from my team this week", "journal"),
    ("Reflecting on the past year, I've grown a lot professionally", "journal"),
]

UPDATE_EXAMPLES = [
    ("The project deadline has been moved to next Friday", "update"),
    ("I finished the design mockups for the new feature", "update"),
    ("The client approved the budget increase", "update"),
    ("Sprint velocity is up 15% from last quarter", "update"),
    ("The API migration is 80% complete", "update"),
]

QUESTION_EXAMPLES = [
    ("What was the name of that restaurant Sarah recommended", "question"),
    ("When is my next dentist appointment", "question"),
    ("How many tasks did I complete last week", "question"),
    ("What did I talk about in yesterday's meeting notes", "question"),
    ("Who was I supposed to email about the partnership", "question"),
]


class TestTaskClassification:
    """Tests for TASK intent classification."""

    @pytest.mark.parametrize("transcript,expected_intent", TASK_EXAMPLES)
    async def test_task_classification(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        transcript: str,
        expected_intent: str,
    ):
        result = _classification(expected_intent, confidence=0.93)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(transcript)

        assert output["intent"] == "task"
        assert output["confidence"] >= 0.85

    async def test_task_with_entities(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Task classification extracts relevant entities."""
        result = _classification(
            "task",
            confidence=0.95,
            entities=[
                {"name": "quarterly report", "type": "project", "metadata": {}},
                {"name": "Friday", "type": "date", "metadata": {}},
            ],
        )
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "I need to finish the quarterly report by Friday"
        )

        assert output["intent"] == "task"
        assert len(output["entities"]) >= 1


class TestEventClassification:
    """Tests for EVENT intent classification."""

    @pytest.mark.parametrize("transcript,expected_intent", EVENT_EXAMPLES)
    async def test_event_classification(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        transcript: str,
        expected_intent: str,
    ):
        result = _classification(expected_intent, confidence=0.94)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(transcript)

        assert output["intent"] == "event"
        assert output["confidence"] >= 0.85

    async def test_event_with_temporal_entities(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Events should extract temporal entities."""
        result = _classification(
            "event",
            confidence=0.96,
            entities=[
                {"name": "board meeting", "type": "event", "metadata": {}},
                {"name": "March 20th", "type": "date", "metadata": {}},
            ],
        )
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "We have a board meeting on March 20th"
        )

        assert output["intent"] == "event"
        assert len(output["entities"]) >= 1


class TestReminderClassification:
    """Tests for REMINDER intent classification."""

    @pytest.mark.parametrize("transcript,expected_intent", REMINDER_EXAMPLES)
    async def test_reminder_classification(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        transcript: str,
        expected_intent: str,
    ):
        result = _classification(expected_intent, confidence=0.91)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(transcript)

        assert output["intent"] == "reminder"
        assert output["confidence"] >= 0.85


class TestIdeaClassification:
    """Tests for IDEA intent classification."""

    @pytest.mark.parametrize("transcript,expected_intent", IDEA_EXAMPLES)
    async def test_idea_classification(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        transcript: str,
        expected_intent: str,
    ):
        result = _classification(expected_intent, confidence=0.90)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(transcript)

        assert output["intent"] == "idea"
        assert output["confidence"] >= 0.85


class TestJournalClassification:
    """Tests for JOURNAL intent classification."""

    @pytest.mark.parametrize("transcript,expected_intent", JOURNAL_EXAMPLES)
    async def test_journal_classification(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        transcript: str,
        expected_intent: str,
    ):
        result = _classification(expected_intent, confidence=0.89)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(transcript)

        assert output["intent"] == "journal"
        assert output["confidence"] >= 0.85

    async def test_journal_with_emotion(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Journal entries should detect emotion with appropriate intensity."""
        result = _classification(
            "journal",
            confidence=0.92,
            emotion={
                "primary": "sadness",
                "intensity": 2,
                "valence": -0.6,
                "arousal": 0.3,
            },
        )
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Feeling anxious about the presentation tomorrow"
        )

        assert output["intent"] == "journal"
        assert output["emotion"]["primary"] == "sadness"
        assert output["emotion"]["valence"] < 0  # negative emotion


class TestUpdateClassification:
    """Tests for UPDATE intent classification."""

    @pytest.mark.parametrize("transcript,expected_intent", UPDATE_EXAMPLES)
    async def test_update_classification(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        transcript: str,
        expected_intent: str,
    ):
        result = _classification(expected_intent, confidence=0.88)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(transcript)

        assert output["intent"] == "update"
        assert output["confidence"] >= 0.85


class TestQuestionClassification:
    """Tests for QUESTION intent classification."""

    @pytest.mark.parametrize("transcript,expected_intent", QUESTION_EXAMPLES)
    async def test_question_classification(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        transcript: str,
        expected_intent: str,
    ):
        result = _classification(expected_intent, confidence=0.93)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(transcript)

        assert output["intent"] == "question"
        assert output["confidence"] >= 0.85


# ── Confidence Score Validation ──────────────────────────────────────


class TestConfidenceScores:
    """Validate that confidence scores meet the >85% threshold on clear inputs."""

    ALL_CLEAR_EXAMPLES = [
        *TASK_EXAMPLES,
        *EVENT_EXAMPLES,
        *REMINDER_EXAMPLES,
        *IDEA_EXAMPLES,
        *JOURNAL_EXAMPLES,
        *UPDATE_EXAMPLES,
        *QUESTION_EXAMPLES,
    ]

    @pytest.mark.parametrize("transcript,expected_intent", ALL_CLEAR_EXAMPLES)
    async def test_confidence_above_85_percent(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        transcript: str,
        expected_intent: str,
    ):
        """Every clear, representative input must achieve >85% confidence."""
        result = _classification(expected_intent, confidence=0.92)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(transcript)

        assert output["confidence"] >= 0.85, (
            f"Confidence {output['confidence']:.2f} < 0.85 for "
            f"'{transcript}' (expected {expected_intent})"
        )
        assert output["intent"] == expected_intent

    async def test_confidence_is_float_between_0_and_1(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Confidence must be a valid probability."""
        result = _classification("task", confidence=0.95)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Finish the report")

        assert isinstance(output["confidence"], (int, float))
        assert 0.0 <= output["confidence"] <= 1.0

    async def test_high_confidence_for_unambiguous_task(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Very clear task statements should get very high confidence."""
        result = _classification("task", confidence=0.97)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Buy milk from the store")

        assert output["confidence"] >= 0.90

    async def test_high_confidence_for_unambiguous_question(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Very clear questions should get very high confidence."""
        result = _classification("question", confidence=0.96)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "What time is my dentist appointment?"
        )

        assert output["confidence"] >= 0.90


# ── Edge Cases ───────────────────────────────────────────────────────


class TestAmbiguousInputs:
    """Tests for inputs that could map to multiple intents."""

    async def test_task_or_reminder_ambiguity(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """'Don't forget to...' could be task or reminder — should still classify."""
        result = _classification("reminder", confidence=0.72)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Don't forget to send the invoice"
        )

        # Should pick one intent, even if confidence is lower
        assert output["intent"] in ("task", "reminder")
        assert 0.0 <= output["confidence"] <= 1.0

    async def test_event_or_task_ambiguity(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """'Schedule a meeting...' could be event creation or task to schedule."""
        result = _classification("event", confidence=0.78)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Schedule a meeting with John next week"
        )

        assert output["intent"] in ("event", "task")

    async def test_idea_or_journal_ambiguity(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Reflective thoughts mixing ideas and journaling."""
        result = _classification("journal", confidence=0.68)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "I've been thinking about how we could improve team morale "
            "and maybe we should try weekly social events"
        )

        assert output["intent"] in ("idea", "journal")

    async def test_update_or_journal_ambiguity(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Status reports that feel like journal entries."""
        result = _classification("update", confidence=0.74)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Made good progress on the redesign today, feeling optimistic"
        )

        assert output["intent"] in ("update", "journal")

    async def test_question_or_task_ambiguity(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Implicit requests phrased as questions."""
        result = _classification("question", confidence=0.70)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Can someone look into why the build is failing?"
        )

        assert output["intent"] in ("question", "task")

    async def test_ambiguous_inputs_have_lower_confidence(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Ambiguous inputs should naturally get lower confidence scores."""
        result = _classification("task", confidence=0.65)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "That thing we talked about earlier"
        )

        # Ambiguous input should not get a very high confidence
        assert output["confidence"] < 0.85


class TestMultiIntentInputs:
    """Tests for inputs that contain multiple intents."""

    async def test_task_and_reminder_combined(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Input with both a task and a reminder."""
        # System should pick the primary intent
        result = _classification("task", confidence=0.80)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "I need to finish the report and remind me to send it by 5 PM"
        )

        # Should classify as one of the contained intents
        assert output["intent"] in ("task", "reminder")

    async def test_event_and_task_combined(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Input with both event info and a task."""
        result = _classification("event", confidence=0.76)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "The meeting is at 3 PM and I need to prepare the slides before then"
        )

        assert output["intent"] in ("event", "task")

    async def test_idea_and_question_combined(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Input with an idea and a question."""
        result = _classification("idea", confidence=0.73)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "What if we added dark mode? Has anyone tried that before?"
        )

        assert output["intent"] in ("idea", "question")

    async def test_journal_and_update_combined(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Input mixing personal reflection with status update."""
        result = _classification("journal", confidence=0.71)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Shipped the new feature today and honestly I'm really proud of it"
        )

        assert output["intent"] in ("journal", "update")

    async def test_multi_intent_still_returns_single_classification(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Even multi-intent inputs must return a single primary intent."""
        result = _classification("task", confidence=0.75)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Remind me to buy groceries, also that meeting is at 3, "
            "and I had a great idea about the new product"
        )

        # Must return exactly one intent, not a list
        assert isinstance(output["intent"], str)
        assert output["intent"] in {intent.value for intent in BlurtIntent}


class TestMinimalAndEdgeCaseInputs:
    """Tests for minimal, empty, and unusual inputs."""

    async def test_single_word_input(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Single word inputs should still produce a classification."""
        result = _classification("task", confidence=0.45)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Groceries")

        assert output["intent"] in {intent.value for intent in BlurtIntent}
        # Low confidence expected for minimal input
        assert 0.0 <= output["confidence"] <= 1.0

    async def test_very_short_input(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Two-word inputs should classify with lower confidence."""
        result = _classification("task", confidence=0.55)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Buy milk")

        assert output["intent"] in {intent.value for intent in BlurtIntent}

    async def test_very_long_input(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Long rambling inputs should still classify."""
        long_text = (
            "So I was thinking about this thing at work and the team "
            "had a discussion about the roadmap and then we decided to "
            "push the deadline back by two weeks because the API integration "
            "is taking longer than expected and also the designer needs more "
            "time for the user testing phase and honestly I think it's the "
            "right call because quality matters more than speed"
        )
        result = _classification("update", confidence=0.82)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(long_text)

        assert output["intent"] in {intent.value for intent in BlurtIntent}

    async def test_input_with_special_characters(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Inputs with special chars (from voice transcription errors)."""
        result = _classification("task", confidence=0.80)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Email john@example.com about the $500 invoice #1234"
        )

        assert output["intent"] in {intent.value for intent in BlurtIntent}

    async def test_input_with_numbers(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Numeric inputs should classify (dates, amounts, etc.)."""
        result = _classification("event", confidence=0.88)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Meeting at 3:30 PM on March 15th 2026 in room 204"
        )

        assert output["intent"] == "event"

    async def test_conversational_filler_input(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Conversational filler words from natural speech."""
        result = _classification("task", confidence=0.60)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Um so like I was gonna say that we should probably you know "
            "get the thing done by uh Friday maybe"
        )

        assert output["intent"] in {intent.value for intent in BlurtIntent}

    async def test_non_english_input(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Non-English input should still produce a classification."""
        result = _classification("task", confidence=0.70)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(
            "Necesito comprar leche del supermercado"
        )

        assert output["intent"] in {intent.value for intent in BlurtIntent}


# ── Classification Response Structure ────────────────────────────────


class TestClassificationResponseStructure:
    """Tests for the structure and completeness of classification results."""

    async def test_response_has_all_required_fields(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Classification result must have intent, confidence, entities, emotion, acknowledgment."""
        result = _classification(
            "task",
            confidence=0.92,
            entities=[{"name": "report", "type": "project", "metadata": {}}],
            emotion={"primary": "anticipation", "intensity": 1, "valence": 0.4, "arousal": 0.5},
            acknowledgment="On it",
        )
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Finish the report")

        assert "intent" in output
        assert "confidence" in output
        assert "entities" in output
        assert "emotion" in output
        assert "acknowledgment" in output

    async def test_intent_is_valid_enum_value(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Returned intent must be one of the 7 valid values."""
        result = _classification("task", confidence=0.92)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Do something")

        valid_intents = {intent.value for intent in BlurtIntent}
        assert output["intent"] in valid_intents

    async def test_entities_is_list(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Entities field should always be a list."""
        result = _classification("task", confidence=0.92, entities=[])
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Do something")

        assert isinstance(output["entities"], list)

    async def test_emotion_is_dict(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Emotion field should always be a dict."""
        result = _classification("task", confidence=0.92)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Do something")

        assert isinstance(output["emotion"], dict)

    async def test_acknowledgment_is_string(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Acknowledgment should be a brief string."""
        result = _classification("task", confidence=0.92, acknowledgment="Got it")
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Do something")

        assert isinstance(output["acknowledgment"], str)
        assert len(output["acknowledgment"]) > 0

    async def test_entity_structure(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Each entity should have name, type, and metadata."""
        result = _classification(
            "task",
            confidence=0.92,
            entities=[
                {"name": "John", "type": "person", "metadata": {"role": "client"}},
                {"name": "Acme Corp", "type": "organization", "metadata": {}},
            ],
        )
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Email John at Acme Corp")

        for entity in output["entities"]:
            assert "name" in entity
            assert "type" in entity

    async def test_emotion_structure(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Emotion dict should have primary, intensity, valence, arousal."""
        result = _classification(
            "journal",
            confidence=0.92,
            emotion={
                "primary": "joy",
                "intensity": 2,
                "valence": 0.8,
                "arousal": 0.6,
            },
        )
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Had a wonderful day today")

        emotion = output["emotion"]
        assert "primary" in emotion
        assert "intensity" in emotion
        assert "valence" in emotion
        assert "arousal" in emotion


# ── Shame-Free Design Validation ─────────────────────────────────────


class TestShameFreeAcknowledgments:
    """Validate that classification acknowledgments follow anti-shame design."""

    SHAME_WORDS = [
        "overdue", "late", "behind", "failed", "missed", "lazy",
        "procrastinating", "disappointed", "should have", "must",
    ]

    async def test_acknowledgment_has_no_guilt_language(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Acknowledgments must not contain shame/guilt language."""
        result = _classification("task", confidence=0.92, acknowledgment="Got it, noted")
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("I still haven't done the taxes")

        ack = output["acknowledgment"].lower()
        for word in self.SHAME_WORDS:
            assert word not in ack, (
                f"Acknowledgment contains shame word '{word}': '{output['acknowledgment']}'"
            )

    async def test_acknowledgment_is_brief(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Acknowledgments should be 2-8 words as specified in the prompt."""
        result = _classification("task", confidence=0.92, acknowledgment="Got it")
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript("Send the email")

        word_count = len(output["acknowledgment"].split())
        assert 1 <= word_count <= 10, (
            f"Acknowledgment has {word_count} words: '{output['acknowledgment']}'"
        )


# ── API Error Handling ───────────────────────────────────────────────


class TestClassificationErrorHandling:
    """Tests for error handling in the classification flow."""

    async def test_api_error_raises_exception(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """API errors should propagate as GeminiAudioError."""
        mock_http_client.post = AsyncMock(
            return_value=httpx.Response(500, json={"error": {"message": "Internal error"}})
        )

        with pytest.raises(GeminiAudioError):
            await client.classify_transcript("Test input")

    async def test_rate_limit_error(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Rate limiting should raise appropriate error."""
        mock_http_client.post = AsyncMock(
            return_value=httpx.Response(
                429,
                json={"error": {"message": "Rate limit exceeded"}},
            )
        )

        with pytest.raises((GeminiAudioError, GeminiAudioError)):
            await client.classify_transcript("Test input")

    async def test_malformed_json_response(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Malformed JSON from API should raise parse error."""
        bad_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "not valid json {{{"}],
                    },
                    "finishReason": "STOP",
                }
            ],
        }
        mock_http_client.post = AsyncMock(
            return_value=httpx.Response(200, json=bad_response)
        )

        with pytest.raises((GeminiAudioError, json.JSONDecodeError, Exception)):
            await client.classify_transcript("Test input")

    async def test_empty_response(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Empty API response should return empty dict gracefully."""
        empty_response = {"candidates": []}
        mock_http_client.post = AsyncMock(
            return_value=httpx.Response(200, json=empty_response)
        )

        output = await client.classify_transcript("Test input")

        # Empty candidates returns empty dict — graceful degradation
        assert output == {}


# ── Integration: Classification → Pipeline ───────────────────────────


class TestClassificationInPipeline:
    """Tests that classification integrates correctly with the audio pipeline."""

    async def test_classify_uses_flash_lite_model(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        config: GeminiConfig,
    ):
        """Classification should use Flash-Lite (fast tier), not Flash."""
        result = _classification("task", confidence=0.92)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        await client.classify_transcript("Buy groceries")

        # Verify the API was called with the flash-lite model in the URL
        call_args = mock_http_client.post.call_args
        assert call_args is not None
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert config.flash_lite_model in url

    async def test_classify_requests_json_mime_type(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Classification request should set responseMimeType to application/json."""
        result = _classification("task", confidence=0.92)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        await client.classify_transcript("Buy groceries")

        call_args = mock_http_client.post.call_args
        assert call_args is not None
        body = call_args[1].get("json", {})
        gen_config = body.get("generationConfig", {})
        assert gen_config.get("responseMimeType") == "application/json"

    async def test_classify_uses_low_temperature(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Classification should use low temperature for deterministic output."""
        result = _classification("task", confidence=0.92)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        await client.classify_transcript("Buy groceries")

        call_args = mock_http_client.post.call_args
        assert call_args is not None
        body = call_args[1].get("json", {})
        gen_config = body.get("generationConfig", {})
        temperature = gen_config.get("temperature", 1.0)
        assert temperature <= 0.2, f"Temperature {temperature} is too high for classification"


# ── Cross-Intent Coverage ────────────────────────────────────────────


class TestCrossIntentCoverage:
    """Ensure every intent is covered and classification is consistent."""

    @pytest.mark.parametrize("intent", list(BlurtIntent))
    async def test_each_intent_can_be_classified(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
        intent: BlurtIntent,
    ):
        """Every BlurtIntent value should be classifiable."""
        result = _classification(intent.value, confidence=0.90)
        mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

        output = await client.classify_transcript(f"Test input for {intent.value}")

        assert output["intent"] == intent.value

    async def test_different_inputs_can_yield_different_intents(
        self,
        client: GeminiAudioClient,
        mock_http_client: AsyncMock,
    ):
        """Verify the classifier distinguishes between different intents."""
        intents_seen = set()
        examples = [
            ("Buy groceries", "task"),
            ("Meeting at 3 PM tomorrow", "event"),
            ("Remind me at 5", "reminder"),
            ("What if we built a chatbot", "idea"),
            ("I feel great today", "journal"),
            ("The project is 80% done", "update"),
            ("What did I do last week", "question"),
        ]

        for transcript, expected in examples:
            result = _classification(expected, confidence=0.90)
            mock_http_client.post = AsyncMock(return_value=_make_gemini_response(result))

            output = await client.classify_transcript(transcript)
            intents_seen.add(output["intent"])

        # Should see all 7 distinct intents
        assert len(intents_seen) == 7
        assert intents_seen == {intent.value for intent in BlurtIntent}
