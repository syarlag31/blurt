"""Brief verbal acknowledgment generator for Blurt.

Produces short, natural acknowledgments after processing each blurt.
Designed to feel conversational — like a trusted friend who heard you,
not a chatbot that narrates everything it did.

Design principles:
- Brief: 1-5 words max for simple confirmations, 1 short sentence for questions
- Natural: Varies phrasing to avoid sounding robotic
- Anti-shame: Never guilt, never nag, never count overdue items
- Intent-aware: Acknowledgment tone matches what the user said
- Emotion-aware: Adjusts warmth/energy based on detected emotion
- No-tasks-pending is valid: Never force-surface or push tasks
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from blurt.classification.models import ClassificationResult, ClassificationStatus
from blurt.models.emotions import EmotionResult, PrimaryEmotion
from blurt.models.intents import BlurtIntent


class AcknowledgmentTone(str, Enum):
    """Tone modifiers for acknowledgments based on emotional context."""

    WARM = "warm"           # User is sharing something personal/emotional
    CALM = "calm"           # Neutral, everyday input
    ENERGETIC = "energetic" # User sounds excited or enthusiastic
    GENTLE = "gentle"       # User sounds down or stressed
    MATTER_OF_FACT = "matter_of_fact"  # Straightforward operational input


# ── Per-intent acknowledgment pools ────────────────────────────────
# Each pool contains short, natural phrases. The system randomly picks
# one per blurt so responses feel varied and human.

_TASK_ACKS = [
    "Got it.",
    "On it.",
    "Noted.",
    "Captured.",
    "Done.",
    "Tracked.",
    "Added.",
]

_EVENT_ACKS = [
    "On the calendar.",
    "Scheduled.",
    "Noted.",
    "Got it down.",
    "Saved.",
    "Calendared.",
]

_REMINDER_ACKS = [
    "I'll remind you.",
    "Set.",
    "You'll get a nudge.",
    "Reminder set.",
    "Won't forget.",
    "Noted.",
]

_IDEA_ACKS = [
    "Interesting.",
    "Saved that thought.",
    "Captured.",
    "Noted.",
    "Cool idea.",
    "Stored.",
    "Filed away.",
]

_JOURNAL_ACKS = [
    "Heard you.",
    "Got it.",
    "Noted.",
    "Captured.",
    "Thanks for sharing.",
    "Recorded.",
]

_UPDATE_ACKS = [
    "Updated.",
    "Got it.",
    "Changed.",
    "Noted.",
    "Done.",
    "Adjusted.",
]

_QUESTION_ACKS: list[str] = []  # Questions get answers, not acks

# Pools indexed by intent
_ACK_POOLS: dict[BlurtIntent, list[str]] = {
    BlurtIntent.TASK: _TASK_ACKS,
    BlurtIntent.EVENT: _EVENT_ACKS,
    BlurtIntent.REMINDER: _REMINDER_ACKS,
    BlurtIntent.IDEA: _IDEA_ACKS,
    BlurtIntent.JOURNAL: _JOURNAL_ACKS,
    BlurtIntent.UPDATE: _UPDATE_ACKS,
    BlurtIntent.QUESTION: _QUESTION_ACKS,
}

# ── Emotion-modulated overlays ─────────────────────────────────────
# When emotion is detected with high intensity, these softer/warmer
# variants may replace the default pool selection.

_GENTLE_JOURNAL_ACKS = [
    "I hear you.",
    "Thanks for sharing that.",
    "Noted.",
    "Got it.",
]

_ENERGETIC_IDEA_ACKS = [
    "Love it.",
    "Nice one.",
    "Ooh, captured.",
    "Saved that.",
]

# Maximum word count for an acknowledgment (safety rail)
_MAX_WORDS = 8


@dataclass(frozen=True, slots=True)
class Acknowledgment:
    """A brief verbal acknowledgment to return to the user.

    Attributes:
        text: The short acknowledgment string.
        tone: The emotional tone used.
        intent: The classified intent this acknowledges.
        is_silent: If True, no verbal ack is needed (e.g., question gets an answer instead).
        answer: For QUESTION intents, the actual answer text (ack is silent).
    """

    text: str
    tone: AcknowledgmentTone = AcknowledgmentTone.CALM
    intent: BlurtIntent = BlurtIntent.JOURNAL
    is_silent: bool = False
    answer: str | None = None

    @property
    def word_count(self) -> int:
        """Number of words in the acknowledgment."""
        return len(self.text.split()) if self.text else 0


# ── Singleton silent ack for questions ─────────────────────────────
SILENT_ACK = Acknowledgment(
    text="",
    tone=AcknowledgmentTone.CALM,
    intent=BlurtIntent.QUESTION,
    is_silent=True,
)


def _select_tone(emotion: EmotionResult | None) -> AcknowledgmentTone:
    """Derive acknowledgment tone from emotion detection result.

    Args:
        emotion: Emotion detection result, or None if not available.

    Returns:
        Appropriate AcknowledgmentTone for the response.
    """
    if emotion is None:
        return AcknowledgmentTone.CALM

    # High arousal + positive valence → energetic
    if emotion.arousal >= 0.7 and emotion.valence > 0.2:
        return AcknowledgmentTone.ENERGETIC

    # Negative valence + sadness/fear → gentle
    if emotion.valence < -0.3:
        return AcknowledgmentTone.GENTLE

    # Positive valence with trust/joy → warm
    if emotion.valence > 0.3 and emotion.primary.emotion in (
        PrimaryEmotion.JOY,
        PrimaryEmotion.TRUST,
    ):
        return AcknowledgmentTone.WARM

    # Low arousal → calm/matter-of-fact
    if emotion.arousal < 0.3:
        return AcknowledgmentTone.MATTER_OF_FACT

    return AcknowledgmentTone.CALM


def _pick_from_pool(
    intent: BlurtIntent,
    tone: AcknowledgmentTone,
) -> str:
    """Select a random acknowledgment from the appropriate pool.

    Uses emotion-aware overlays when tone warrants it, otherwise
    falls back to the standard intent pool.

    Args:
        intent: Classified intent.
        tone: Derived emotional tone.

    Returns:
        A brief acknowledgment string.
    """
    # Emotion-aware pool overrides
    if tone == AcknowledgmentTone.GENTLE and intent == BlurtIntent.JOURNAL:
        return random.choice(_GENTLE_JOURNAL_ACKS)

    if tone == AcknowledgmentTone.ENERGETIC and intent == BlurtIntent.IDEA:
        return random.choice(_ENERGETIC_IDEA_ACKS)

    pool = _ACK_POOLS.get(intent, _JOURNAL_ACKS)
    if not pool:
        return ""

    return random.choice(pool)


def generate_acknowledgment(
    classification: ClassificationResult,
    emotion: EmotionResult | None = None,
    *,
    answer_text: str | None = None,
) -> Acknowledgment:
    """Generate a brief verbal acknowledgment for a classified blurt.

    This is the main entry point. Every processed blurt passes through
    here to get its acknowledgment. The result is designed to be spoken
    aloud via TTS or displayed as a brief text confirmation.

    Args:
        classification: The classification result from the pipeline.
        emotion: Optional emotion detection result for tone adjustment.
        answer_text: For QUESTION intents, the generated answer.

    Returns:
        An Acknowledgment with brief, natural text.
    """
    intent = classification.primary_intent

    # Questions get answers, not acknowledgments
    if intent == BlurtIntent.QUESTION:
        return Acknowledgment(
            text="",
            tone=AcknowledgmentTone.CALM,
            intent=intent,
            is_silent=True,
            answer=answer_text,
        )

    tone = _select_tone(emotion)
    text = _pick_from_pool(intent, tone)

    return Acknowledgment(
        text=text,
        tone=tone,
        intent=intent,
        is_silent=False,
    )


def generate_acknowledgment_for_error() -> Acknowledgment:
    """Generate an acknowledgment when processing had an error.

    Even on error, we don't alarm the user. The blurt was still
    captured (journal fallback), so we acknowledge simply.

    Returns:
        A calm, brief acknowledgment.
    """
    return Acknowledgment(
        text="Got it.",
        tone=AcknowledgmentTone.CALM,
        intent=BlurtIntent.JOURNAL,
        is_silent=False,
    )


class AcknowledgmentService:
    """Stateful acknowledgment generator with variety tracking.

    Tracks recent acknowledgments to avoid repeating the same phrase
    consecutively, making interactions feel more natural.
    """

    def __init__(self, *, history_size: int = 5) -> None:
        """Initialize the acknowledgment service.

        Args:
            history_size: Number of recent acks to track for variety.
        """
        self._history: list[str] = []
        self._history_size = history_size

    def acknowledge(
        self,
        classification: ClassificationResult,
        emotion: EmotionResult | None = None,
        *,
        answer_text: str | None = None,
    ) -> Acknowledgment:
        """Generate an acknowledgment, avoiding recent repetition.

        Args:
            classification: Classification result from the pipeline.
            emotion: Optional emotion result for tone adjustment.
            answer_text: For QUESTION intents, the answer text.

        Returns:
            A brief, natural Acknowledgment.
        """
        intent = classification.primary_intent

        # Questions bypass the variety logic
        if intent == BlurtIntent.QUESTION:
            return generate_acknowledgment(
                classification, emotion, answer_text=answer_text
            )

        # Error classifications still get a calm ack
        if classification.status == ClassificationStatus.ERROR:
            ack = generate_acknowledgment_for_error()
            self._record(ack.text)
            return ack

        tone = _select_tone(emotion)
        pool = self._get_pool(intent, tone)

        # Pick a phrase not in recent history
        text = self._pick_varied(pool)
        ack = Acknowledgment(text=text, tone=tone, intent=intent)
        self._record(text)
        return ack

    def _get_pool(
        self, intent: BlurtIntent, tone: AcknowledgmentTone
    ) -> list[str]:
        """Get the appropriate pool, considering tone overlays."""
        if tone == AcknowledgmentTone.GENTLE and intent == BlurtIntent.JOURNAL:
            return list(_GENTLE_JOURNAL_ACKS)
        if tone == AcknowledgmentTone.ENERGETIC and intent == BlurtIntent.IDEA:
            return list(_ENERGETIC_IDEA_ACKS)
        pool = _ACK_POOLS.get(intent, _JOURNAL_ACKS)
        return list(pool) if pool else ["Got it."]

    def _pick_varied(self, pool: list[str]) -> str:
        """Pick a phrase from the pool, preferring ones not recently used."""
        candidates = [p for p in pool if p not in self._history]
        if not candidates:
            # All options exhausted — reset and pick any
            candidates = pool
        return random.choice(candidates) if candidates else "Got it."

    def _record(self, text: str) -> None:
        """Record a used phrase in history, maintaining window size."""
        self._history.append(text)
        if len(self._history) > self._history_size:
            self._history.pop(0)

    def reset(self) -> None:
        """Clear acknowledgment history."""
        self._history.clear()
