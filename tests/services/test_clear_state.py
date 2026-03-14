"""Tests for the clear-state message service.

Validates that when no tasks need attention, Blurt responds with:
- Positive, affirming messages (not "nothing to do")
- Emotion-aware tone adjustment
- Varied phrases (no robotic repetition)
- Anti-shame language (no guilt, streaks, overdue, etc.)
- Brief messages suitable for TTS
- Time-of-day awareness when available
"""

from __future__ import annotations

import re

import pytest

from blurt.models.emotions import EmotionResult, EmotionScore, PrimaryEmotion
from blurt.services.clear_state import (
    ClearStateMessage,
    ClearStateService,
    ClearTone,
    _CALM_MESSAGES,
    _CELEBRATORY_MESSAGES,
    _GENTLE_MESSAGES,
    _MAX_WORDS,
    _MESSAGE_POOLS,
    _SHAME_WORDS,
    _WARM_MESSAGES,
    generate_clear_message,
    select_clear_tone,
)
from blurt.services.task_surfacing import (
    ScoredTask,
    SignalScore,
    SignalType,
    SurfaceableTask,
    SurfacingResult,
    UserContext,
)


# ── Helpers ────────────────────────────────────────────────────────


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


def _empty_surfacing_result(total_eligible: int = 0) -> SurfacingResult:
    """Create an empty SurfacingResult (no tasks to surface)."""
    return SurfacingResult(
        tasks=[],
        total_eligible=total_eligible,
        total_filtered=0,
    )


def _nonempty_surfacing_result() -> SurfacingResult:
    """Create a SurfacingResult with one scored task."""
    task = SurfaceableTask(content="Test task")
    scored = ScoredTask(
        task=task,
        composite_score=0.7,
        signal_scores=(
            SignalScore(signal=SignalType.FRESHNESS, value=0.8, reason="fresh"),
        ),
        surfacing_reason="test",
    )
    return SurfacingResult(
        tasks=[scored],
        total_eligible=1,
        total_filtered=0,
    )


# ── Tone Selection ────────────────────────────────────────────────


class TestToneSelection:
    """Tests for select_clear_tone()."""

    def test_no_emotion_returns_calm(self) -> None:
        assert select_clear_tone(None) == ClearTone.CALM

    def test_high_energy_positive_returns_celebratory(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.JOY, intensity=0.9, valence=0.8, arousal=0.9
        )
        assert select_clear_tone(emotion) == ClearTone.CELEBRATORY

    def test_negative_valence_returns_gentle(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.SADNESS, intensity=0.6, valence=-0.5, arousal=0.3
        )
        assert select_clear_tone(emotion) == ClearTone.GENTLE

    def test_positive_valence_returns_warm(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.JOY, intensity=0.5, valence=0.4, arousal=0.4
        )
        assert select_clear_tone(emotion) == ClearTone.WARM

    def test_neutral_returns_calm(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.ANTICIPATION, intensity=0.3, valence=0.0, arousal=0.4
        )
        assert select_clear_tone(emotion) == ClearTone.CALM


# ── Message Generation ────────────────────────────────────────────


class TestMessageGeneration:
    """Tests for generate_clear_message()."""

    def test_generates_message_without_surfacing_result(self) -> None:
        msg = generate_clear_message()
        assert isinstance(msg, ClearStateMessage)
        assert msg.text != ""
        assert msg.has_active_tasks is False

    def test_generates_message_with_empty_surfacing(self) -> None:
        result = _empty_surfacing_result(total_eligible=5)
        msg = generate_clear_message(surfacing_result=result)
        assert msg.has_active_tasks is False
        assert msg.total_tasks_checked == 5

    def test_calm_tone_uses_calm_pool(self) -> None:
        msg = generate_clear_message(emotion=None)
        assert msg.tone == ClearTone.CALM
        assert msg.text in _CALM_MESSAGES

    def test_gentle_tone_uses_gentle_pool(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.SADNESS, intensity=0.7, valence=-0.5, arousal=0.3
        )
        msg = generate_clear_message(emotion=emotion)
        assert msg.tone == ClearTone.GENTLE
        assert msg.text in _GENTLE_MESSAGES

    def test_celebratory_tone_uses_celebratory_pool(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.JOY, intensity=0.9, valence=0.8, arousal=0.9
        )
        msg = generate_clear_message(emotion=emotion)
        assert msg.tone == ClearTone.CELEBRATORY
        assert msg.text in _CELEBRATORY_MESSAGES

    def test_warm_tone_uses_warm_pool(self) -> None:
        emotion = _make_emotion(
            PrimaryEmotion.TRUST, intensity=0.5, valence=0.4, arousal=0.4
        )
        msg = generate_clear_message(emotion=emotion)
        assert msg.tone == ClearTone.WARM
        assert msg.text in _WARM_MESSAGES

    def test_time_of_day_stored(self) -> None:
        msg = generate_clear_message(time_of_day="evening")
        assert msg.time_of_day == "evening"


# ── Brevity ───────────────────────────────────────────────────────


class TestBrevity:
    """Clear-state messages must be brief — suitable for TTS."""

    @pytest.mark.parametrize("tone", list(ClearTone))
    def test_all_pool_messages_under_word_limit(self, tone: ClearTone) -> None:
        """Every message in every pool is under the word limit."""
        pool = _MESSAGE_POOLS[tone]
        for text in pool:
            word_count = len(text.split())
            assert word_count <= _MAX_WORDS, (
                f"Message too long in {tone.value} pool: '{text}' "
                f"({word_count} words, max {_MAX_WORDS})"
            )

    @pytest.mark.parametrize("tone", list(ClearTone))
    def test_all_pool_messages_not_empty(self, tone: ClearTone) -> None:
        """Every message in every pool has at least 1 word."""
        pool = _MESSAGE_POOLS[tone]
        for text in pool:
            assert len(text.strip()) > 0, f"Empty message in {tone.value} pool"

    def test_generated_message_under_word_limit(self) -> None:
        for _ in range(100):
            msg = generate_clear_message()
            assert msg.word_count <= _MAX_WORDS


# ── Anti-Shame Design ────────────────────────────────────────────


class TestAntiShame:
    """No guilt, no streaks, no overdue, no pressure — ever."""

    @pytest.mark.parametrize("tone", list(ClearTone))
    def test_no_shame_language_in_pool(self, tone: ClearTone) -> None:
        """No message pool contains guilt/pressure language."""
        pool = _MESSAGE_POOLS[tone]
        for text in pool:
            text_lower = text.lower()
            for shame_word in _SHAME_WORDS:
                # Use word boundary matching to avoid false positives
                # like "plate" matching "late"
                pattern = r'\b' + re.escape(shame_word) + r'\b'
                assert not re.search(pattern, text_lower), (
                    f"Shame word '{shame_word}' found in {tone.value} "
                    f"pool: '{text}'"
                )

    def test_no_shame_in_generated_messages(self) -> None:
        """Generate many messages and verify none contain shame language."""
        emotions = [
            None,
            _make_emotion(PrimaryEmotion.JOY, valence=0.8, arousal=0.9),
            _make_emotion(PrimaryEmotion.SADNESS, valence=-0.5, arousal=0.3),
            _make_emotion(PrimaryEmotion.TRUST, valence=0.4, arousal=0.4),
        ]
        for emotion in emotions:
            for _ in range(50):
                msg = generate_clear_message(emotion=emotion)
                text_lower = msg.text.lower()
                for shame_word in _SHAME_WORDS:
                    pattern = r'\b' + re.escape(shame_word) + r'\b'
                    assert not re.search(pattern, text_lower), (
                        f"Shame word '{shame_word}' in message: '{msg.text}'"
                    )

    def test_no_negative_framing(self) -> None:
        """Messages should not frame emptiness negatively."""
        negative_framings = [
            "nothing to do",
            "no work",
            "empty list",
            "zero tasks",
            "no tasks",
            "idle",
        ]
        for tone in ClearTone:
            pool = _MESSAGE_POOLS[tone]
            for text in pool:
                text_lower = text.lower()
                for neg in negative_framings:
                    assert neg not in text_lower, (
                        f"Negative framing '{neg}' found in "
                        f"{tone.value}: '{text}'"
                    )

    def test_clear_state_is_positive(self) -> None:
        """The has_active_tasks flag is always False for clear messages."""
        for _ in range(20):
            msg = generate_clear_message()
            assert msg.has_active_tasks is False


# ── Variety / Non-Robotic ─────────────────────────────────────────


class TestVariety:
    """Responses should vary to feel natural."""

    def test_multiple_messages_not_all_identical(self) -> None:
        """Over 30 calls, we should see at least 2 distinct phrases."""
        messages = set()
        for _ in range(30):
            msg = generate_clear_message()
            messages.add(msg.text)
        assert len(messages) >= 2, "All clear messages were identical"

    def test_each_pool_has_variety(self) -> None:
        """Each tone pool has at least 4 distinct messages."""
        for tone in ClearTone:
            pool = _MESSAGE_POOLS[tone]
            assert len(pool) >= 4, (
                f"{tone.value} pool has only {len(pool)} messages"
            )
            assert len(set(pool)) == len(pool), (
                f"{tone.value} pool has duplicate messages"
            )

    def test_service_avoids_consecutive_repeats(self) -> None:
        """ClearStateService tracks history to avoid repetition."""
        service = ClearStateService(history_size=5)
        result = _empty_surfacing_result()
        prev = None
        repeat_count = 0
        for _ in range(20):
            msg = service.check_and_respond(result)
            assert msg is not None
            if msg.text == prev:
                repeat_count += 1
            prev = msg.text
        # With 8 calm phrases and history of 5, repeats should be rare
        assert repeat_count <= 3, (
            f"Too many consecutive repeats: {repeat_count}"
        )

    def test_service_reset_clears_history(self) -> None:
        service = ClearStateService(history_size=3)
        result = _empty_surfacing_result()
        for _ in range(5):
            service.check_and_respond(result)
        service.reset()
        msg = service.check_and_respond(result)
        assert msg is not None
        assert msg.text != ""


# ── ClearStateService ──────────────────────────────────────────────


class TestClearStateService:
    """Tests for the stateful ClearStateService."""

    def test_returns_message_when_no_tasks(self) -> None:
        service = ClearStateService()
        result = _empty_surfacing_result()
        msg = service.check_and_respond(result)
        assert msg is not None
        assert isinstance(msg, ClearStateMessage)
        assert msg.has_active_tasks is False

    def test_returns_none_when_tasks_exist(self) -> None:
        service = ClearStateService()
        result = _nonempty_surfacing_result()
        msg = service.check_and_respond(result)
        assert msg is None

    def test_emotion_affects_tone(self) -> None:
        service = ClearStateService()
        result = _empty_surfacing_result()
        emotion = _make_emotion(
            PrimaryEmotion.SADNESS, intensity=0.7, valence=-0.5, arousal=0.3
        )
        msg = service.check_and_respond(result, emotion=emotion)
        assert msg is not None
        assert msg.tone == ClearTone.GENTLE

    def test_celebratory_tone_on_high_energy(self) -> None:
        service = ClearStateService()
        result = _empty_surfacing_result()
        emotion = _make_emotion(
            PrimaryEmotion.JOY, intensity=0.9, valence=0.8, arousal=0.9
        )
        msg = service.check_and_respond(result, emotion=emotion)
        assert msg is not None
        assert msg.tone == ClearTone.CELEBRATORY

    def test_time_of_day_passed_through(self) -> None:
        service = ClearStateService()
        result = _empty_surfacing_result()
        msg = service.check_and_respond(result, time_of_day="morning")
        assert msg is not None
        assert msg.time_of_day == "morning"

    def test_generate_direct(self) -> None:
        """generate() works without a surfacing result."""
        service = ClearStateService()
        msg = service.generate(total_checked=3)
        assert msg.has_active_tasks is False
        assert msg.total_tasks_checked == 3

    def test_total_tasks_checked_from_surfacing(self) -> None:
        service = ClearStateService()
        result = _empty_surfacing_result(total_eligible=10)
        msg = service.check_and_respond(result)
        assert msg is not None
        assert msg.total_tasks_checked == 10


# ── ClearStateMessage Model ──────────────────────────────────────


class TestClearStateMessageModel:
    """Tests for the ClearStateMessage dataclass."""

    def test_word_count_empty(self) -> None:
        msg = ClearStateMessage(text="")
        assert msg.word_count == 0

    def test_word_count_single(self) -> None:
        msg = ClearStateMessage(text="Clear.")
        assert msg.word_count == 1

    def test_word_count_multi(self) -> None:
        msg = ClearStateMessage(text="You're all clear.")
        assert msg.word_count == 3

    def test_frozen_dataclass(self) -> None:
        msg = ClearStateMessage(text="All clear.")
        with pytest.raises(AttributeError):
            msg.text = "Changed"  # type: ignore[misc]

    def test_has_active_tasks_always_false(self) -> None:
        msg = ClearStateMessage(text="All clear.", has_active_tasks=False)
        assert msg.has_active_tasks is False

    def test_default_tone_is_calm(self) -> None:
        msg = ClearStateMessage(text="All clear.")
        assert msg.tone == ClearTone.CALM


# ── Integration with TaskScoringEngine ────────────────────────────


class TestIntegrationWithSurfacing:
    """Tests that clear-state integrates with the task surfacing engine."""

    def test_empty_task_list_produces_clear_message(self) -> None:
        """When scoring engine receives no tasks, clear state fires."""
        from blurt.services.task_surfacing import TaskScoringEngine

        engine = TaskScoringEngine()
        context = UserContext()
        result = engine.score_and_rank(tasks=[], context=context)

        service = ClearStateService()
        msg = service.check_and_respond(result)
        assert msg is not None
        assert msg.has_active_tasks is False
        assert msg.text in _CALM_MESSAGES

    def test_all_tasks_below_threshold_produces_clear(self) -> None:
        """When all tasks score below threshold, no tasks surface → clear."""
        from blurt.services.task_surfacing import TaskScoringEngine

        engine = TaskScoringEngine(min_score=0.99)  # Very high threshold
        context = UserContext()
        task = SurfaceableTask(content="Low priority thing")
        result = engine.score_and_rank(tasks=[task], context=context)

        service = ClearStateService()
        msg = service.check_and_respond(result)
        assert msg is not None
        assert msg.has_active_tasks is False

    def test_completed_tasks_dont_surface(self) -> None:
        """Completed tasks are filtered out — user is clear."""
        from blurt.services.task_surfacing import TaskScoringEngine, TaskStatus

        engine = TaskScoringEngine()
        context = UserContext()
        task = SurfaceableTask(
            content="Already done",
            status=TaskStatus.COMPLETED,
        )
        result = engine.score_and_rank(tasks=[task], context=context)

        service = ClearStateService()
        msg = service.check_and_respond(result)
        assert msg is not None

    def test_dropped_tasks_dont_surface(self) -> None:
        """Dropped tasks are filtered out — shame-free drop."""
        from blurt.services.task_surfacing import TaskScoringEngine, TaskStatus

        engine = TaskScoringEngine()
        context = UserContext()
        task = SurfaceableTask(
            content="Decided against this",
            status=TaskStatus.DROPPED,
        )
        result = engine.score_and_rank(tasks=[task], context=context)

        service = ClearStateService()
        msg = service.check_and_respond(result)
        assert msg is not None
