"""Tests for the acknowledgment service.

Validates that acknowledgments are:
- Brief (under 8 words)
- Natural and varied
- Intent-appropriate
- Emotion-aware
- Anti-shame (no guilt language)
- Silent for questions (answers instead)
"""

from __future__ import annotations

import pytest

from blurt.classification.models import (
    ClassificationResult,
    ClassificationStatus,
    IntentScore,
)
from blurt.models.emotions import (
    EmotionResult,
    EmotionScore,
    PrimaryEmotion,
)
from blurt.models.intents import BlurtIntent
from blurt.services.acknowledgment import (
    SILENT_ACK,
    Acknowledgment,
    AcknowledgmentService,
    AcknowledgmentTone,
    _select_tone,
    generate_acknowledgment,
    generate_acknowledgment_for_error,
)


# ── Helpers ────────────────────────────────────────────────────────

def _make_classification(
    intent: BlurtIntent,
    confidence: float = 0.92,
    status: ClassificationStatus = ClassificationStatus.CONFIDENT,
    text: str = "test input",
) -> ClassificationResult:
    """Create a ClassificationResult for testing."""
    return ClassificationResult(
        input_text=text,
        primary_intent=intent,
        confidence=confidence,
        status=status,
        all_scores=[IntentScore(intent=intent, confidence=confidence)],
        model_used="flash-lite",
    )


def _make_emotion(
    emotion: PrimaryEmotion = PrimaryEmotion.JOY,
    intensity: float = 0.5,
    valence: float = 0.0,
    arousal: float = 0.5,
) -> EmotionResult:
    """Create an EmotionResult for testing."""
    return EmotionResult(
        primary=EmotionScore(emotion=emotion, intensity=intensity),
        valence=valence,
        arousal=arousal,
    )


# ── Brevity Tests ──────────────────────────────────────────────────

class TestBrevity:
    """Acknowledgments must be brief — 1-8 words max."""

    @pytest.mark.parametrize("intent", list(BlurtIntent))
    def test_all_intents_produce_brief_acks(self, intent: BlurtIntent) -> None:
        """Every intent's acknowledgment is under the word limit."""
        classification = _make_classification(intent)
        ack = generate_acknowledgment(classification)

        if intent == BlurtIntent.QUESTION:
            assert ack.is_silent
            assert ack.text == ""
        else:
            assert ack.word_count <= 8, (
                f"Ack for {intent.value} too long: '{ack.text}' "
                f"({ack.word_count} words)"
            )
            assert ack.word_count >= 1, f"Ack for {intent.value} is empty"

    def test_task_ack_is_short(self) -> None:
        classification = _make_classification(BlurtIntent.TASK)
        ack = generate_acknowledgment(classification)
        assert ack.word_count <= 3

    def test_event_ack_is_short(self) -> None:
        classification = _make_classification(BlurtIntent.EVENT)
        ack = generate_acknowledgment(classification)
        assert ack.word_count <= 4

    def test_error_ack_is_brief(self) -> None:
        ack = generate_acknowledgment_for_error()
        assert ack.word_count <= 3


# ── Intent Appropriateness ─────────────────────────────────────────

class TestIntentAppropriateness:
    """Each intent gets a contextually appropriate acknowledgment."""

    def test_question_is_silent(self) -> None:
        """Questions get answers, not acks."""
        classification = _make_classification(BlurtIntent.QUESTION)
        ack = generate_acknowledgment(classification, answer_text="Sarah's birthday is March 5th")
        assert ack.is_silent
        assert ack.answer == "Sarah's birthday is March 5th"
        assert ack.text == ""

    def test_task_acknowledged(self) -> None:
        classification = _make_classification(BlurtIntent.TASK)
        ack = generate_acknowledgment(classification)
        assert not ack.is_silent
        assert ack.intent == BlurtIntent.TASK

    def test_event_acknowledged(self) -> None:
        classification = _make_classification(BlurtIntent.EVENT)
        ack = generate_acknowledgment(classification)
        assert not ack.is_silent
        assert ack.intent == BlurtIntent.EVENT

    def test_reminder_acknowledged(self) -> None:
        classification = _make_classification(BlurtIntent.REMINDER)
        ack = generate_acknowledgment(classification)
        assert not ack.is_silent
        assert ack.intent == BlurtIntent.REMINDER

    def test_idea_acknowledged(self) -> None:
        classification = _make_classification(BlurtIntent.IDEA)
        ack = generate_acknowledgment(classification)
        assert not ack.is_silent
        assert ack.intent == BlurtIntent.IDEA

    def test_journal_acknowledged(self) -> None:
        classification = _make_classification(BlurtIntent.JOURNAL)
        ack = generate_acknowledgment(classification)
        assert not ack.is_silent
        assert ack.intent == BlurtIntent.JOURNAL

    def test_update_acknowledged(self) -> None:
        classification = _make_classification(BlurtIntent.UPDATE)
        ack = generate_acknowledgment(classification)
        assert not ack.is_silent
        assert ack.intent == BlurtIntent.UPDATE


# ── Emotion Awareness ──────────────────────────────────────────────

class TestEmotionAwareness:
    """Tone adapts to the user's emotional state."""

    def test_high_energy_positive_is_energetic(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.JOY, intensity=0.8, valence=0.7, arousal=0.9
        )
        tone = _select_tone(emotion)
        assert tone == AcknowledgmentTone.ENERGETIC

    def test_negative_valence_is_gentle(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.SADNESS, intensity=0.7, valence=-0.5, arousal=0.3
        )
        tone = _select_tone(emotion)
        assert tone == AcknowledgmentTone.GENTLE

    def test_positive_joy_is_warm(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.JOY, intensity=0.5, valence=0.5, arousal=0.5
        )
        tone = _select_tone(emotion)
        assert tone == AcknowledgmentTone.WARM

    def test_low_arousal_is_matter_of_fact(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.ANTICIPATION, intensity=0.3, valence=0.1, arousal=0.2
        )
        tone = _select_tone(emotion)
        assert tone == AcknowledgmentTone.MATTER_OF_FACT

    def test_no_emotion_is_calm(self) -> None:
        tone = _select_tone(None)
        assert tone == AcknowledgmentTone.CALM

    def test_gentle_journal_uses_warm_phrases(self) -> None:
        """Sad journal entry gets empathetic phrasing."""
        classification = _make_classification(BlurtIntent.JOURNAL)
        emotion = _make_emotion(
            PrimaryEmotion.SADNESS, intensity=0.7, valence=-0.5, arousal=0.3
        )
        ack = generate_acknowledgment(classification, emotion)
        assert ack.tone == AcknowledgmentTone.GENTLE
        # Should use gentle pool
        assert ack.text in [
            "I hear you.",
            "Thanks for sharing that.",
            "Noted.",
            "Got it.",
        ]

    def test_energetic_idea_uses_enthusiastic_phrases(self) -> None:
        """Excited idea gets matching energy."""
        classification = _make_classification(BlurtIntent.IDEA)
        emotion = _make_emotion(
            PrimaryEmotion.JOY, intensity=0.8, valence=0.7, arousal=0.9
        )
        ack = generate_acknowledgment(classification, emotion)
        assert ack.tone == AcknowledgmentTone.ENERGETIC
        assert ack.text in [
            "Love it.",
            "Nice one.",
            "Ooh, captured.",
            "Saved that.",
        ]


# ── Anti-Shame Design ─────────────────────────────────────────────

class TestAntiShame:
    """No guilt, no streaks, no overdue, no pressure."""

    _SHAME_WORDS = {
        "overdue", "late", "behind", "missed", "failed", "forgot",
        "streak", "pending", "urgent", "hurry", "asap", "deadline",
        "warning", "alert", "critical", "important", "priority",
        "you should", "you need to", "don't forget",
    }

    @pytest.mark.parametrize("intent", list(BlurtIntent))
    def test_no_shame_language_in_any_intent(self, intent: BlurtIntent) -> None:
        """No acknowledgment contains guilt/pressure language."""
        classification = _make_classification(intent)
        # Generate many acks to cover the pool
        for _ in range(50):
            ack = generate_acknowledgment(classification)
            text_lower = ack.text.lower()
            for shame_word in self._SHAME_WORDS:
                assert shame_word not in text_lower, (
                    f"Shame word '{shame_word}' found in ack "
                    f"for {intent.value}: '{ack.text}'"
                )

    def test_error_ack_is_not_alarming(self) -> None:
        """Even errors produce calm, simple acks."""
        ack = generate_acknowledgment_for_error()
        assert ack.tone == AcknowledgmentTone.CALM
        text_lower = ack.text.lower()
        assert "error" not in text_lower
        assert "fail" not in text_lower
        assert "wrong" not in text_lower


# ── Variety / Non-Robotic ──────────────────────────────────────────

class TestVariety:
    """Responses should vary to feel natural."""

    def test_multiple_acks_are_not_all_identical(self) -> None:
        """Over 20 acks, we should see at least 2 distinct phrases."""
        classification = _make_classification(BlurtIntent.TASK)
        acks = set()
        for _ in range(30):
            ack = generate_acknowledgment(classification)
            acks.add(ack.text)
        assert len(acks) >= 2, "All acknowledgments were identical"

    def test_service_avoids_consecutive_repeats(self) -> None:
        """AcknowledgmentService tracks history to avoid repetition."""
        service = AcknowledgmentService(history_size=5)
        classification = _make_classification(BlurtIntent.TASK)
        prev = None
        repeat_count = 0
        for _ in range(20):
            ack = service.acknowledge(classification)
            if ack.text == prev:
                repeat_count += 1
            prev = ack.text
        # With 7 task phrases and history of 5, consecutive repeats should be rare
        assert repeat_count <= 3, (
            f"Too many consecutive repeats: {repeat_count}"
        )

    def test_service_reset_clears_history(self) -> None:
        service = AcknowledgmentService(history_size=3)
        classification = _make_classification(BlurtIntent.TASK)
        for _ in range(5):
            service.acknowledge(classification)
        service.reset()
        # After reset, all phrases are available again
        ack = service.acknowledge(classification)
        assert ack.text  # Should produce something


# ── AcknowledgmentService ──────────────────────────────────────────

class TestAcknowledgmentService:
    """Tests for the stateful AcknowledgmentService."""

    def test_question_bypasses_variety_logic(self) -> None:
        service = AcknowledgmentService()
        classification = _make_classification(BlurtIntent.QUESTION)
        ack = service.acknowledge(classification, answer_text="42")
        assert ack.is_silent
        assert ack.answer == "42"

    def test_error_status_gets_calm_ack(self) -> None:
        service = AcknowledgmentService()
        classification = _make_classification(
            BlurtIntent.JOURNAL,
            status=ClassificationStatus.ERROR,
        )
        ack = service.acknowledge(classification)
        assert ack.tone == AcknowledgmentTone.CALM
        assert ack.text == "Got it."

    def test_emotion_affects_tone_through_service(self) -> None:
        service = AcknowledgmentService()
        classification = _make_classification(BlurtIntent.JOURNAL)
        emotion = _make_emotion(
            PrimaryEmotion.SADNESS, intensity=0.7, valence=-0.5, arousal=0.3
        )
        ack = service.acknowledge(classification, emotion)
        assert ack.tone == AcknowledgmentTone.GENTLE


# ── Acknowledgment Model ──────────────────────────────────────────

class TestAcknowledgmentModel:
    """Tests for the Acknowledgment dataclass."""

    def test_word_count_empty(self) -> None:
        ack = Acknowledgment(text="", is_silent=True)
        assert ack.word_count == 0

    def test_word_count_single(self) -> None:
        ack = Acknowledgment(text="Got it.")
        assert ack.word_count == 2

    def test_silent_ack_singleton(self) -> None:
        assert SILENT_ACK.is_silent
        assert SILENT_ACK.text == ""
        assert SILENT_ACK.intent == BlurtIntent.QUESTION

    def test_frozen_dataclass(self) -> None:
        ack = Acknowledgment(text="Noted.")
        with pytest.raises(AttributeError):
            ack.text = "Changed"  # type: ignore[misc]
