"""Task feedback service with Thompson Sampling parameter updates.

Records user interactions with surfaced tasks (accept, dismiss, snooze, complete)
and updates Thompson Sampling Beta distribution parameters to improve future
task surfacing decisions.

Thompson Sampling approach:
- Each task-context combination maintains alpha (successes) and beta (failures)
  parameters of a Beta distribution.
- "Accept" and "Complete" increment alpha (positive signal).
- "Dismiss" increments beta (negative signal).
- "Snooze" is treated as a weak negative — partial beta increment.
- When surfacing, we sample from the Beta distribution to decide ranking,
  balancing exploration (uncertain tasks) vs exploitation (proven good matches).

Anti-shame design:
- No guilt language in responses.
- Snooze is explicitly supported and respected.
- Dismiss is shame-free — the task just gets deprioritized.
- No "overdue" or "you haven't completed" messaging.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol


class FeedbackAction(str, Enum):
    """User actions on surfaced tasks. All actions are shame-free."""

    ACCEPT = "accept"      # User accepts the surfaced task (starts working)
    DISMISS = "dismiss"    # User dismisses — respect the choice, no guilt
    SNOOZE = "snooze"      # User defers — will surface again later
    COMPLETE = "complete"  # User marks as done


# Map actions to Thompson Sampling updates (alpha_delta, beta_delta)
_ACTION_UPDATES: dict[FeedbackAction, tuple[float, float]] = {
    FeedbackAction.ACCEPT: (1.0, 0.0),     # Strong positive
    FeedbackAction.COMPLETE: (2.0, 0.0),    # Strongest positive (accepted AND finished)
    FeedbackAction.DISMISS: (0.0, 1.0),     # Negative signal
    FeedbackAction.SNOOZE: (0.0, 0.3),      # Weak negative — they want it, just not now
}


@dataclass
class FeedbackEvent:
    """A single feedback event on a surfaced task."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    user_id: str = ""
    action: FeedbackAction = FeedbackAction.DISMISS
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Context at the time of feedback — used for contextual Thompson Sampling
    context_key: str = ""  # e.g., "morning_high_energy" — bucket key
    mood_valence: float = 0.0
    energy_level: float = 0.5
    time_of_day: str = ""

    # Snooze-specific
    snooze_minutes: int | None = None  # How long to snooze

    # Optional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThompsonParams:
    """Beta distribution parameters for Thompson Sampling.

    alpha = successes + 1 (prior)
    beta = failures + 1 (prior)

    We use a (1, 1) prior (uniform distribution) for new items.
    """

    alpha: float = 1.0  # Prior: 1 success
    beta: float = 1.0   # Prior: 1 failure
    total_observations: int = 0
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @property
    def mean(self) -> float:
        """Expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the Beta distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    def sample(self) -> float:
        """Draw a sample from the Beta distribution.

        This is the core of Thompson Sampling — random sampling
        naturally balances exploration vs exploitation.
        """
        return random.betavariate(self.alpha, self.beta)

    def update(self, alpha_delta: float, beta_delta: float) -> None:
        """Update parameters with new evidence."""
        self.alpha += alpha_delta
        self.beta += beta_delta
        self.total_observations += 1
        self.last_updated = datetime.now(timezone.utc)


@dataclass
class FeedbackSummary:
    """Summary of feedback for a task across all contexts."""

    task_id: str
    total_events: int = 0
    accept_count: int = 0
    dismiss_count: int = 0
    snooze_count: int = 0
    complete_count: int = 0
    acceptance_rate: float = 0.0
    thompson_mean: float = 0.5  # From the global Thompson params
    last_feedback_at: datetime | None = None


class FeedbackStore(Protocol):
    """Protocol for persisting feedback events and Thompson parameters."""

    def store_event(self, event: FeedbackEvent) -> None:
        """Store a feedback event."""
        ...

    def get_events(
        self,
        task_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackEvent]:
        """Retrieve feedback events with optional filters."""
        ...

    def get_params(self, key: str) -> ThompsonParams:
        """Get Thompson params for a key (creates with priors if missing)."""
        ...

    def set_params(self, key: str, params: ThompsonParams) -> None:
        """Persist updated Thompson params."""
        ...

    def get_task_summary(self, task_id: str) -> FeedbackSummary:
        """Get aggregated feedback summary for a task."""
        ...


class InMemoryFeedbackStore:
    """In-memory implementation of FeedbackStore for development/testing."""

    def __init__(self) -> None:
        self._events: list[FeedbackEvent] = []
        self._params: dict[str, ThompsonParams] = {}

    def store_event(self, event: FeedbackEvent) -> None:
        self._events.append(event)

    def get_events(
        self,
        task_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackEvent]:
        result = self._events
        if task_id is not None:
            result = [e for e in result if e.task_id == task_id]
        if user_id is not None:
            result = [e for e in result if e.user_id == user_id]
        # Most recent first
        result = sorted(result, key=lambda e: e.timestamp, reverse=True)
        return result[:limit]

    def get_params(self, key: str) -> ThompsonParams:
        if key not in self._params:
            self._params[key] = ThompsonParams()
        return self._params[key]

    def set_params(self, key: str, params: ThompsonParams) -> None:
        self._params[key] = params

    def get_task_summary(self, task_id: str) -> FeedbackSummary:
        events = self.get_events(task_id=task_id, limit=10000)
        if not events:
            return FeedbackSummary(task_id=task_id)

        accept_count = sum(1 for e in events if e.action == FeedbackAction.ACCEPT)
        dismiss_count = sum(1 for e in events if e.action == FeedbackAction.DISMISS)
        snooze_count = sum(1 for e in events if e.action == FeedbackAction.SNOOZE)
        complete_count = sum(1 for e in events if e.action == FeedbackAction.COMPLETE)
        total = len(events)
        positive = accept_count + complete_count
        acceptance_rate = positive / total if total > 0 else 0.0

        # Get global Thompson params for this task
        global_key = f"task:{task_id}"
        params = self.get_params(global_key)

        return FeedbackSummary(
            task_id=task_id,
            total_events=total,
            accept_count=accept_count,
            dismiss_count=dismiss_count,
            snooze_count=snooze_count,
            complete_count=complete_count,
            acceptance_rate=round(acceptance_rate, 4),
            thompson_mean=round(params.mean, 4),
            last_feedback_at=events[0].timestamp if events else None,
        )


def _build_context_key(
    time_of_day: str = "",
    energy_bucket: str = "",
    mood_bucket: str = "",
) -> str:
    """Build a composite context key for contextual Thompson Sampling.

    The key segments context so the model learns different preferences
    for different situations (e.g., morning+high_energy vs evening+low_energy).
    """
    parts = []
    if time_of_day:
        parts.append(time_of_day)
    if energy_bucket:
        parts.append(energy_bucket)
    if mood_bucket:
        parts.append(mood_bucket)
    return ":".join(parts) if parts else "global"


def _energy_bucket(energy: float) -> str:
    """Map energy (0-1) to a bucket label."""
    if energy < 0.33:
        return "low_energy"
    elif energy < 0.66:
        return "med_energy"
    else:
        return "high_energy"


def _mood_bucket(valence: float) -> str:
    """Map mood valence (-1 to 1) to a bucket label."""
    if valence < -0.3:
        return "negative_mood"
    elif valence > 0.3:
        return "positive_mood"
    else:
        return "neutral_mood"


class TaskFeedbackService:
    """Service for recording task feedback and updating Thompson Sampling params.

    Manages the feedback loop:
    1. User interacts with a surfaced task (accept/dismiss/snooze/complete)
    2. Feedback is recorded as an event
    3. Thompson Sampling parameters are updated at multiple granularities:
       - Global task level (task:{task_id})
       - Context-specific (task:{task_id}:context:{context_key})
       - Intent-level (intent:{intent}:context:{context_key})
    4. Updated parameters influence future surfacing decisions
    """

    def __init__(self, store: FeedbackStore | None = None) -> None:
        self._store = store or InMemoryFeedbackStore()

    @property
    def store(self) -> FeedbackStore:
        return self._store

    def record_feedback(
        self,
        task_id: str,
        user_id: str,
        action: FeedbackAction,
        mood_valence: float = 0.0,
        energy_level: float = 0.5,
        time_of_day: str = "",
        snooze_minutes: int | None = None,
        intent: str = "task",
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEvent:
        """Record a user's feedback on a surfaced task and update Thompson params.

        Args:
            task_id: The surfaced task's ID.
            user_id: The user's ID.
            action: What the user did (accept, dismiss, snooze, complete).
            mood_valence: User's current mood (-1 to 1).
            energy_level: User's current energy (0 to 1).
            time_of_day: Time bucket (morning, afternoon, evening, night).
            snooze_minutes: Duration to snooze (only for snooze action).
            intent: The task's classified intent type.
            metadata: Additional metadata.

        Returns:
            The recorded FeedbackEvent.
        """
        # Build context key for contextual Thompson Sampling
        ctx_key = _build_context_key(
            time_of_day=time_of_day,
            energy_bucket=_energy_bucket(energy_level),
            mood_bucket=_mood_bucket(mood_valence),
        )

        event = FeedbackEvent(
            task_id=task_id,
            user_id=user_id,
            action=action,
            context_key=ctx_key,
            mood_valence=mood_valence,
            energy_level=energy_level,
            time_of_day=time_of_day,
            snooze_minutes=snooze_minutes if action == FeedbackAction.SNOOZE else None,
            metadata=metadata or {},
        )

        # Store the event
        self._store.store_event(event)

        # Update Thompson Sampling parameters at multiple granularities
        alpha_delta, beta_delta = _ACTION_UPDATES[action]
        self._update_thompson_params(
            task_id=task_id,
            intent=intent,
            context_key=ctx_key,
            alpha_delta=alpha_delta,
            beta_delta=beta_delta,
        )

        return event

    def _update_thompson_params(
        self,
        task_id: str,
        intent: str,
        context_key: str,
        alpha_delta: float,
        beta_delta: float,
    ) -> None:
        """Update Thompson Sampling params at multiple levels.

        We maintain params at three granularities:
        1. Global task level — overall quality signal for this specific task
        2. Task + context — how well this task fits in this specific context
        3. Intent + context — how well this intent type works in this context
           (generalizes learning across tasks of the same type)
        """
        keys = [
            f"task:{task_id}",
            f"task:{task_id}:ctx:{context_key}",
            f"intent:{intent}:ctx:{context_key}",
        ]

        for key in keys:
            params = self._store.get_params(key)
            params.update(alpha_delta, beta_delta)
            self._store.set_params(key, params)

    def sample_score(
        self,
        task_id: str,
        intent: str = "task",
        context_key: str = "global",
    ) -> float:
        """Sample a Thompson Sampling score for a task in a given context.

        Combines task-level and intent-level signals.
        Used by the surfacing engine to rank tasks with exploration/exploitation.

        Returns:
            A sampled score in [0, 1] from the Beta distribution.
        """
        # Get task-context params
        task_ctx_key = f"task:{task_id}:ctx:{context_key}"
        task_params = self._store.get_params(task_ctx_key)

        # Get intent-context params (generalized)
        intent_ctx_key = f"intent:{intent}:ctx:{context_key}"
        intent_params = self._store.get_params(intent_ctx_key)

        # Blend: weight task-specific more if we have enough observations
        task_weight = min(1.0, task_params.total_observations / 10.0)
        intent_weight = 1.0 - task_weight

        task_sample = task_params.sample()
        intent_sample = intent_params.sample()

        return task_weight * task_sample + intent_weight * intent_sample

    def get_task_summary(self, task_id: str) -> FeedbackSummary:
        """Get aggregated feedback summary for a task."""
        return self._store.get_task_summary(task_id)

    def get_recent_feedback(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[FeedbackEvent]:
        """Get recent feedback events for a user."""
        return self._store.get_events(user_id=user_id, limit=limit)
