"""Clear-state message generator for Blurt.

When no tasks need attention, Blurt responds with a positive, affirming
"you-are-clear" message. This is a core anti-shame design feature:
no-tasks-pending is always a valid, celebrated state.

Design principles:
- Affirming: "You're all clear" not "Nothing to do"
- Emotion-aware: Tone adapts to the user's current mood
- Varied: Multiple phrases to feel natural, not robotic
- Anti-shame: No guilt, no forced engagement, no "you could be doing..."
- Brief: Short enough for TTS, under 12 words
- Context-aware: Messages reflect time-of-day when available
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from blurt.models.emotions import EmotionResult
from blurt.services.task_surfacing import SurfacingResult


class ClearTone(str, Enum):
    """Tone variants for clear-state messages."""

    WARM = "warm"               # User feeling good, reflect it back
    CALM = "calm"               # Default neutral tone
    GENTLE = "gentle"           # User feeling down — reassuring, not dismissive
    CELEBRATORY = "celebratory" # User in high spirits — match the energy


# ── Message pools ────────────────────────────────────────────────

_CALM_MESSAGES = [
    "You're all clear.",
    "Nothing needs your attention.",
    "All clear.",
    "You're free.",
    "Nothing pending.",
    "Desk is clean.",
    "All good here.",
    "You're caught up.",
]

_WARM_MESSAGES = [
    "You're all set.",
    "Everything's taken care of.",
    "Nothing needs you.",
    "You're in great shape.",
    "All caught up.",
    "Smooth sailing.",
    "You're golden.",
    "All wrapped up.",
]

_GENTLE_MESSAGES = [
    "Nothing needs you right now.",
    "You're all clear. Rest easy.",
    "Nothing waiting on you.",
    "All good. Take your time.",
    "Nothing pressing.",
    "You're free to just be.",
    "All clear. No rush.",
    "Nothing needs your attention right now.",
]

_CELEBRATORY_MESSAGES = [
    "You're totally clear!",
    "All done!",
    "Clean slate!",
    "Nothing pending — nice!",
    "You crushed it. All clear.",
    "Zero inbox. Well done.",
    "All clear — enjoy it!",
    "Wide open — you earned it.",
]

_TIME_AWARE_SUFFIXES: dict[str, list[str]] = {
    "morning": [
        "Fresh start.",
        "Good morning to a clear plate.",
        "",
    ],
    "afternoon": [
        "Enjoy the afternoon.",
        "",
        "",
    ],
    "evening": [
        "Nice way to end the day.",
        "Relax tonight.",
        "",
    ],
    "night": [
        "Rest well.",
        "Sleep easy.",
        "",
    ],
}

_MESSAGE_POOLS: dict[ClearTone, list[str]] = {
    ClearTone.CALM: _CALM_MESSAGES,
    ClearTone.WARM: _WARM_MESSAGES,
    ClearTone.GENTLE: _GENTLE_MESSAGES,
    ClearTone.CELEBRATORY: _CELEBRATORY_MESSAGES,
}

# Maximum word count for a clear message (safety rail)
_MAX_WORDS = 12

# Words that must NEVER appear in clear-state messages
_SHAME_WORDS = frozenset({
    "overdue", "late", "behind", "missed", "failed", "forgot",
    "streak", "urgent", "hurry", "asap", "deadline",
    "warning", "alert", "critical", "important", "priority",
    "you should", "you need to", "don't forget", "you could",
    "idle", "lazy", "unproductive", "wasting",
})


@dataclass(frozen=True, slots=True)
class ClearStateMessage:
    """A 'you-are-clear' message when no tasks need attention.

    Attributes:
        text: The affirming message string.
        tone: The emotional tone used.
        has_active_tasks: Always False for clear-state messages.
        total_tasks_checked: How many tasks were evaluated before determining clear.
        time_of_day: Optional time context used for message selection.
    """

    text: str
    tone: ClearTone = ClearTone.CALM
    has_active_tasks: bool = False
    total_tasks_checked: int = 0
    time_of_day: str | None = None

    @property
    def word_count(self) -> int:
        """Number of words in the message."""
        return len(self.text.split()) if self.text else 0


def select_clear_tone(emotion: EmotionResult | None) -> ClearTone:
    """Derive clear-state tone from emotion detection result.

    Args:
        emotion: Emotion detection result, or None if not available.

    Returns:
        Appropriate ClearTone for the response.
    """
    if emotion is None:
        return ClearTone.CALM

    # High arousal + positive valence → celebratory
    if emotion.arousal >= 0.7 and emotion.valence > 0.2:
        return ClearTone.CELEBRATORY

    # Negative valence → gentle (reassuring, not dismissive)
    if emotion.valence < -0.3:
        return ClearTone.GENTLE

    # Positive valence → warm
    if emotion.valence > 0.2:
        return ClearTone.WARM

    return ClearTone.CALM


def generate_clear_message(
    surfacing_result: SurfacingResult | None = None,
    emotion: EmotionResult | None = None,
    *,
    time_of_day: str | None = None,
) -> ClearStateMessage:
    """Generate a 'you-are-clear' message.

    Called when the task surfacing engine returns no tasks to surface,
    or when the user explicitly asks about pending tasks and there are none.

    Args:
        surfacing_result: The surfacing result (empty or None).
        emotion: Optional emotion detection result for tone adjustment.
        time_of_day: Optional time context (morning/afternoon/evening/night).

    Returns:
        A ClearStateMessage with an affirming, shame-free message.
    """
    tone = select_clear_tone(emotion)
    pool = _MESSAGE_POOLS[tone]
    text = random.choice(pool)

    total_checked = 0
    if surfacing_result is not None:
        total_checked = surfacing_result.total_eligible

    return ClearStateMessage(
        text=text,
        tone=tone,
        has_active_tasks=False,
        total_tasks_checked=total_checked,
        time_of_day=time_of_day,
    )


class ClearStateService:
    """Stateful clear-state message generator with variety tracking.

    Tracks recent messages to avoid repeating the same phrase
    consecutively, making interactions feel natural.
    """

    def __init__(self, *, history_size: int = 5) -> None:
        """Initialize the clear-state service.

        Args:
            history_size: Number of recent messages to track for variety.
        """
        self._history: list[str] = []
        self._history_size = history_size

    def check_and_respond(
        self,
        surfacing_result: SurfacingResult,
        emotion: EmotionResult | None = None,
        *,
        time_of_day: str | None = None,
    ) -> ClearStateMessage | None:
        """Check surfacing result and return a clear message if no tasks.

        This is the main integration point. After the task surfacing engine
        runs, pass its result here. If there are tasks to surface, returns
        None. If clear, returns a positive message.

        Args:
            surfacing_result: Result from TaskScoringEngine.score_and_rank().
            emotion: Optional emotion for tone adjustment.
            time_of_day: Optional time context.

        Returns:
            ClearStateMessage if no tasks need attention, None otherwise.
        """
        if surfacing_result.has_tasks:
            return None

        return self._generate_varied(
            surfacing_result=surfacing_result,
            emotion=emotion,
            time_of_day=time_of_day,
        )

    def generate(
        self,
        emotion: EmotionResult | None = None,
        *,
        time_of_day: str | None = None,
        total_checked: int = 0,
    ) -> ClearStateMessage:
        """Generate a clear-state message directly (without a surfacing result).

        Useful when the caller has already determined there are no tasks.

        Args:
            emotion: Optional emotion for tone adjustment.
            time_of_day: Optional time context.
            total_checked: How many tasks were evaluated.

        Returns:
            A ClearStateMessage with an affirming message.
        """
        dummy_result = SurfacingResult(
            tasks=[],
            total_eligible=total_checked,
            total_filtered=0,
        )
        return self._generate_varied(
            surfacing_result=dummy_result,
            emotion=emotion,
            time_of_day=time_of_day,
        )

    def _generate_varied(
        self,
        surfacing_result: SurfacingResult,
        emotion: EmotionResult | None,
        time_of_day: str | None,
    ) -> ClearStateMessage:
        """Generate a varied clear-state message avoiding recent repeats."""
        tone = select_clear_tone(emotion)
        pool = _MESSAGE_POOLS[tone]

        text = self._pick_varied(pool)
        self._record(text)

        total_checked = surfacing_result.total_eligible

        return ClearStateMessage(
            text=text,
            tone=tone,
            has_active_tasks=False,
            total_tasks_checked=total_checked,
            time_of_day=time_of_day,
        )

    def _pick_varied(self, pool: list[str]) -> str:
        """Pick a phrase from the pool, preferring ones not recently used."""
        candidates = [p for p in pool if p not in self._history]
        if not candidates:
            candidates = pool
        return random.choice(candidates) if candidates else "All clear."

    def _record(self, text: str) -> None:
        """Record a used phrase in history, maintaining window size."""
        self._history.append(text)
        if len(self._history) > self._history_size:
            self._history.pop(0)

    def reset(self) -> None:
        """Clear message history."""
        self._history.clear()
