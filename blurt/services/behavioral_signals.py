"""Behavioral signal collector for adaptive task surfacing.

Tracks user interactions (task acceptance, dismissal, completion, deferral)
and converts them into structured reward/penalty signals that feed the
Thompson Sampler for weight learning and the ThompsonSamplingEngine for
category-level preference learning.

The collector bridges raw FeedbackEvents → structured signals by:
1. Aggregating interaction history per user into behavioral profiles
2. Computing reward/penalty magnitudes based on interaction type + context
3. Deriving contextual signal contributions for the Thompson Sampler arms
4. Tracking behavioral velocity (trend in engagement over time)
5. Building BehavioralProfile snapshots for the scoring engine

Anti-shame design:
- Dismissals are neutral feedback signals, not punishment
- Deferrals (snooze) are respected — partial positive for intent
- No streaks, guilt counters, or forced engagement metrics
- "No interactions" is a valid state — zero signals, not negative signals
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from blurt.services.feedback import (
    FeedbackAction,
    FeedbackEvent,
    FeedbackSummary,
    InMemoryFeedbackStore,
    TaskFeedbackService,
)
from blurt.services.surfacing.models import BehavioralProfile
from blurt.services.surfacing.thompson import ThompsonSampler


# ---------------------------------------------------------------------------
# Signal types & reward configuration
# ---------------------------------------------------------------------------


class SignalKind(str, Enum):
    """Types of behavioral signals emitted by the collector."""

    REWARD = "reward"       # Positive signal (accept, complete)
    PENALTY = "penalty"     # Negative signal (dismiss)
    DEFERRAL = "deferral"   # Weak negative — intent to engage later (snooze)
    NEUTRAL = "neutral"     # No interaction — informational only


@dataclass
class RewardConfig:
    """Configuration for converting actions to reward/penalty magnitudes.

    Anti-shame: penalty magnitudes are deliberately smaller than rewards
    to prevent negative spirals from occasional dismissals.
    """

    accept_reward: float = 1.0       # Standard positive signal
    complete_reward: float = 2.0     # Strong positive — task was finished
    dismiss_penalty: float = 0.5     # Gentle negative — respect the choice
    snooze_penalty: float = 0.15     # Very gentle — they want it, just not now
    snooze_reward: float = 0.3       # Partial positive — intent to engage

    # Context multipliers — amplify signal in high-confidence contexts
    high_energy_multiplier: float = 1.2   # Clearer signal when user is energetic
    low_energy_multiplier: float = 0.8    # Gentler when user is tired
    repeat_dismiss_decay: float = 0.7     # Decay penalty for repeated dismissals
    time_match_bonus: float = 0.2         # Bonus when task matches preferred time


@dataclass
class BehavioralSignal:
    """A single behavioral signal derived from a user interaction.

    This is the output of the collector — a structured representation
    of what the interaction means for the surfacing model.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    task_id: str = ""
    kind: SignalKind = SignalKind.NEUTRAL
    magnitude: float = 0.0  # Absolute strength of the signal (always >= 0)

    # Reward value for the Thompson Sampler (-1 to +1 normalized)
    reward_value: float = 0.0

    # Which signal dimensions were most relevant at the time
    signal_contributions: dict[str, float] = field(default_factory=dict)

    # Context at interaction time
    context: SignalContext = field(default_factory=lambda: SignalContext())

    # Source action that generated this signal
    source_action: FeedbackAction = FeedbackAction.DISMISS
    source_event_id: str = ""

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalContext:
    """Contextual information captured at signal time."""

    time_of_day: str = ""         # morning, afternoon, evening, night
    energy_level: float = 0.5     # 0.0–1.0
    mood_valence: float = 0.0     # -1.0 to 1.0
    intent: str = "task"          # The task's classified intent
    cognitive_load: float = 0.5   # Task difficulty 0.0–1.0
    tags: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)


@dataclass
class InteractionStats:
    """Aggregated interaction statistics for a user over a time window."""

    user_id: str = ""
    window_hours: float = 24.0
    total_interactions: int = 0
    accepts: int = 0
    completions: int = 0
    dismissals: int = 0
    snoozes: int = 0

    # Derived metrics
    acceptance_rate: float = 0.0  # (accepts + completions) / total
    completion_rate: float = 0.0  # completions / total
    engagement_score: float = 0.5  # Weighted engagement (0-1)

    # Behavioral velocity — is engagement trending up or down?
    velocity: float = 0.0  # Positive = improving, negative = declining

    # Per-context breakdowns
    by_time_of_day: dict[str, float] = field(default_factory=dict)
    by_intent: dict[str, float] = field(default_factory=dict)
    by_energy_bucket: dict[str, float] = field(default_factory=dict)


@dataclass
class SignalBatch:
    """A batch of behavioral signals with summary metadata."""

    signals: list[BehavioralSignal] = field(default_factory=list)
    net_reward: float = 0.0    # Sum of all reward values
    total_magnitude: float = 0.0
    signal_count: int = 0

    # Per-dimension aggregate contributions
    dimension_contributions: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Signal store protocol
# ---------------------------------------------------------------------------


class BehavioralSignalStore(Protocol):
    """Protocol for persisting behavioral signals."""

    def store_signal(self, signal: BehavioralSignal) -> None: ...

    def get_signals(
        self,
        user_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[BehavioralSignal]: ...

    def get_interaction_count(
        self,
        user_id: str,
        task_id: str,
        action: FeedbackAction | None = None,
    ) -> int: ...


class InMemorySignalStore:
    """In-memory implementation for development/testing."""

    def __init__(self) -> None:
        self._signals: list[BehavioralSignal] = []

    def store_signal(self, signal: BehavioralSignal) -> None:
        self._signals.append(signal)

    def get_signals(
        self,
        user_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[BehavioralSignal]:
        result = [s for s in self._signals if s.user_id == user_id]
        if since is not None:
            result = [s for s in result if s.timestamp >= since]
        result.sort(key=lambda s: s.timestamp, reverse=True)
        return result[:limit]

    def get_interaction_count(
        self,
        user_id: str,
        task_id: str,
        action: FeedbackAction | None = None,
    ) -> int:
        count = 0
        for s in self._signals:
            if s.user_id == user_id and s.task_id == task_id:
                if action is None or s.source_action == action:
                    count += 1
        return count


# ---------------------------------------------------------------------------
# Behavioral signal collector
# ---------------------------------------------------------------------------


class BehavioralSignalCollector:
    """Collects user interactions and converts them to reward/penalty signals.

    The collector sits between the feedback API and the Thompson Sampler:

        User action → FeedbackEvent → [Collector] → BehavioralSignal → ThompsonSampler

    It enriches raw feedback with:
    - Context-aware reward magnitudes (energy, time-of-day adjustments)
    - Repeat-dismissal decay (diminishing penalty for same task)
    - Signal dimension contributions (which scoring signals led to this outcome)
    - Aggregated behavioral profiles for the scoring engine

    Usage:
        collector = BehavioralSignalCollector(sampler=my_sampler)
        signal = collector.collect(feedback_event, signal_contributions={...})
        # signal is stored and sampler is updated automatically
    """

    def __init__(
        self,
        sampler: ThompsonSampler | None = None,
        store: BehavioralSignalStore | None = None,
        config: RewardConfig | None = None,
    ) -> None:
        self._sampler = sampler or ThompsonSampler()
        self._store = store or InMemorySignalStore()
        self._config = config or RewardConfig()

    @property
    def sampler(self) -> ThompsonSampler:
        return self._sampler

    @property
    def store(self) -> BehavioralSignalStore:
        return self._store

    @property
    def config(self) -> RewardConfig:
        return self._config

    def collect(
        self,
        event: FeedbackEvent,
        signal_contributions: dict[str, float] | None = None,
        cognitive_load: float = 0.5,
        tags: list[str] | None = None,
        entity_ids: list[str] | None = None,
        intent: str = "task",
    ) -> BehavioralSignal:
        """Convert a feedback event into a behavioral signal and update the sampler.

        Args:
            event: The raw feedback event from the user.
            signal_contributions: How much each scoring dimension contributed
                to surfacing this task. Used to assign credit/blame to signals.
            cognitive_load: Task difficulty (0.0 trivial to 1.0 deep work).
            tags: Task tags for per-tag learning.
            entity_ids: Entity IDs linked to the task.
            intent: Task's classified intent type.

        Returns:
            The generated BehavioralSignal.
        """
        contributions = signal_contributions or {}

        # Build context
        ctx = SignalContext(
            time_of_day=event.time_of_day,
            energy_level=event.energy_level,
            mood_valence=event.mood_valence,
            intent=intent,
            cognitive_load=cognitive_load,
            tags=tags or [],
            entity_ids=entity_ids or [],
        )

        # Compute reward magnitude with context adjustments
        kind, magnitude, reward_value = self._compute_reward(
            action=event.action,
            context=ctx,
            user_id=event.user_id,
            task_id=event.task_id,
        )

        signal = BehavioralSignal(
            user_id=event.user_id,
            task_id=event.task_id,
            kind=kind,
            magnitude=magnitude,
            reward_value=reward_value,
            signal_contributions=contributions,
            context=ctx,
            source_action=event.action,
            source_event_id=event.id,
            timestamp=event.timestamp,
        )

        # Store the signal
        self._store.store_signal(signal)

        # Update Thompson Sampler with signal contributions
        if contributions:
            is_reward = reward_value > 0
            learning_rate = magnitude  # Scale learning by signal strength
            self._sampler.record_feedback(
                signal_contributions=contributions,
                reward=is_reward,
                learning_rate=learning_rate,
            )

        return signal

    def collect_batch(
        self,
        events: list[FeedbackEvent],
        contributions_map: dict[str, dict[str, float]] | None = None,
        intent: str = "task",
    ) -> SignalBatch:
        """Convert multiple feedback events into signals in batch.

        Args:
            events: List of feedback events.
            contributions_map: Map of event_id → signal contributions.
            intent: Default intent for all events.

        Returns:
            SignalBatch with all generated signals and summary metrics.
        """
        contrib_map = contributions_map or {}
        signals: list[BehavioralSignal] = []
        net_reward = 0.0
        total_magnitude = 0.0
        dim_contributions: dict[str, float] = defaultdict(float)

        for event in events:
            contribs = contrib_map.get(event.id, {})
            signal = self.collect(
                event=event,
                signal_contributions=contribs,
                intent=intent,
            )
            signals.append(signal)
            net_reward += signal.reward_value
            total_magnitude += signal.magnitude

            for dim, contrib in signal.signal_contributions.items():
                dim_contributions[dim] += contrib * signal.reward_value

        return SignalBatch(
            signals=signals,
            net_reward=net_reward,
            total_magnitude=total_magnitude,
            signal_count=len(signals),
            dimension_contributions=dict(dim_contributions),
        )

    def get_interaction_stats(
        self,
        user_id: str,
        window_hours: float = 24.0,
    ) -> InteractionStats:
        """Compute aggregated interaction statistics for a user.

        Args:
            user_id: The user to analyze.
            window_hours: Look-back window in hours.

        Returns:
            InteractionStats with counts, rates, and breakdowns.
        """
        since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        signals = self._store.get_signals(user_id=user_id, since=since, limit=10000)

        stats = InteractionStats(
            user_id=user_id,
            window_hours=window_hours,
            total_interactions=len(signals),
        )

        if not signals:
            return stats

        # Count by action type
        by_time: dict[str, list[float]] = defaultdict(list)
        by_intent: dict[str, list[float]] = defaultdict(list)
        by_energy: dict[str, list[float]] = defaultdict(list)

        for s in signals:
            if s.source_action == FeedbackAction.ACCEPT:
                stats.accepts += 1
            elif s.source_action == FeedbackAction.COMPLETE:
                stats.completions += 1
            elif s.source_action == FeedbackAction.DISMISS:
                stats.dismissals += 1
            elif s.source_action == FeedbackAction.SNOOZE:
                stats.snoozes += 1

            # Track reward by context
            rv = s.reward_value
            if s.context.time_of_day:
                by_time[s.context.time_of_day].append(rv)
            if s.context.intent:
                by_intent[s.context.intent].append(rv)
            energy_bucket = _energy_bucket(s.context.energy_level)
            by_energy[energy_bucket].append(rv)

        total = stats.total_interactions
        positive = stats.accepts + stats.completions
        stats.acceptance_rate = positive / total if total > 0 else 0.0
        stats.completion_rate = stats.completions / total if total > 0 else 0.0

        # Engagement score: weighted combination
        stats.engagement_score = min(1.0, (
            0.4 * stats.acceptance_rate
            + 0.4 * stats.completion_rate
            + 0.2 * (1.0 - stats.dismissals / total if total > 0 else 0.5)
        ))

        # Compute velocity: compare first half vs second half engagement
        stats.velocity = self._compute_velocity(signals)

        # Context breakdowns (average reward per context)
        stats.by_time_of_day = {
            k: sum(v) / len(v) for k, v in by_time.items()
        }
        stats.by_intent = {
            k: sum(v) / len(v) for k, v in by_intent.items()
        }
        stats.by_energy_bucket = {
            k: sum(v) / len(v) for k, v in by_energy.items()
        }

        return stats

    def build_behavioral_profile(
        self,
        user_id: str,
        window_hours: float = 168.0,  # 1 week default
    ) -> BehavioralProfile:
        """Build a BehavioralProfile from collected signals.

        This profile feeds directly into the surfacing scorers to inform
        context-aware task scoring decisions.

        Args:
            user_id: The user to build a profile for.
            window_hours: Look-back window in hours.

        Returns:
            BehavioralProfile for the scoring engine.
        """
        since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        signals = self._store.get_signals(user_id=user_id, since=since, limit=10000)

        profile = BehavioralProfile()

        if not signals:
            return profile

        # Compute completion rates by time-of-day
        time_totals: dict[str, int] = defaultdict(int)
        time_completions: dict[str, int] = defaultdict(int)

        # Completion rates by cognitive load bucket
        load_totals: dict[str, int] = defaultdict(int)
        load_completions: dict[str, int] = defaultdict(int)

        # Completion rates by tag
        tag_totals: dict[str, int] = defaultdict(int)
        tag_completions: dict[str, int] = defaultdict(int)

        # Hourly activity
        hourly_counts: dict[int, int] = defaultdict(int)

        # Daily completions
        today = datetime.now(timezone.utc).date()
        completions_today = 0
        total_days = max(1.0, window_hours / 24.0)
        total_completions = 0

        for s in signals:
            tod = s.context.time_of_day or "unknown"
            time_totals[tod] += 1

            load_bucket = _load_bucket(s.context.cognitive_load)
            load_totals[load_bucket] += 1

            for tag in s.context.tags:
                tag_totals[tag] += 1

            hour = s.timestamp.hour
            hourly_counts[hour] += 1

            is_completion = s.source_action == FeedbackAction.COMPLETE
            is_accept = s.source_action == FeedbackAction.ACCEPT

            if is_completion or is_accept:
                time_completions[tod] += 1
                load_completions[load_bucket] += 1
                for tag in s.context.tags:
                    tag_completions[tag] += 1

            if is_completion:
                total_completions += 1
                if s.timestamp.date() == today:
                    completions_today += 1

        # Build completion rates
        profile.completion_by_time = {
            k: time_completions.get(k, 0) / v
            for k, v in time_totals.items()
            if v > 0
        }
        profile.completion_by_load = {
            k: load_completions.get(k, 0) / v
            for k, v in load_totals.items()
            if v > 0
        }
        profile.completion_by_tag = {
            k: tag_completions.get(k, 0) / v
            for k, v in tag_totals.items()
            if v > 0
        }

        # Normalize hourly activity
        max_hourly = max(hourly_counts.values()) if hourly_counts else 1
        profile.hourly_activity = {
            h: c / max_hourly for h, c in hourly_counts.items()
        }

        profile.avg_daily_completions = total_completions / total_days
        profile.tasks_completed_today = completions_today

        # Hours since last interaction
        if signals:
            latest = max(s.timestamp for s in signals)
            delta = datetime.now(timezone.utc) - latest
            profile.hours_since_last_interaction = delta.total_seconds() / 3600.0

        return profile

    def get_sampler_state(self) -> dict[str, dict[str, float]]:
        """Get the current Thompson Sampler state for inspection."""
        return self._sampler.get_state()

    def get_expected_weights(self) -> dict[str, float]:
        """Get the sampler's expected (mean) weights."""
        return self._sampler.get_expected_weights()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _compute_reward(
        self,
        action: FeedbackAction,
        context: SignalContext,
        user_id: str,
        task_id: str,
    ) -> tuple[SignalKind, float, float]:
        """Compute the signal kind, magnitude, and reward value.

        Returns:
            (SignalKind, magnitude, reward_value) where:
            - magnitude is always >= 0 (absolute strength)
            - reward_value is positive for rewards, negative for penalties
        """
        cfg = self._config

        # Base magnitudes per action
        if action == FeedbackAction.ACCEPT:
            kind = SignalKind.REWARD
            base_magnitude = cfg.accept_reward
        elif action == FeedbackAction.COMPLETE:
            kind = SignalKind.REWARD
            base_magnitude = cfg.complete_reward
        elif action == FeedbackAction.DISMISS:
            kind = SignalKind.PENALTY
            base_magnitude = cfg.dismiss_penalty
        elif action == FeedbackAction.SNOOZE:
            kind = SignalKind.DEFERRAL
            base_magnitude = cfg.snooze_penalty
        else:
            return SignalKind.NEUTRAL, 0.0, 0.0

        # Context-aware adjustments
        multiplier = 1.0

        # Energy-based multiplier: clearer signals when user is alert
        if context.energy_level > 0.7:
            multiplier *= cfg.high_energy_multiplier
        elif context.energy_level < 0.3:
            multiplier *= cfg.low_energy_multiplier

        # Repeat-dismissal decay: diminish penalty for repeated dismissals
        # of the same task (anti-shame: don't punish for not wanting something)
        if action == FeedbackAction.DISMISS:
            prev_dismissals = self._store.get_interaction_count(
                user_id=user_id,
                task_id=task_id,
                action=FeedbackAction.DISMISS,
            )
            if prev_dismissals > 0:
                # Each repeat reduces magnitude by decay factor
                multiplier *= cfg.repeat_dismiss_decay ** min(prev_dismissals, 5)

        magnitude = base_magnitude * multiplier

        # Convert to reward value (positive for rewards, negative for penalties)
        if kind == SignalKind.REWARD:
            reward_value = magnitude
        elif kind == SignalKind.PENALTY:
            reward_value = -magnitude
        elif kind == SignalKind.DEFERRAL:
            # Snooze: net small negative (penalty) + partial positive (intent)
            reward_value = cfg.snooze_reward - magnitude
        else:
            reward_value = 0.0

        return kind, magnitude, reward_value

    def _compute_velocity(
        self,
        signals: list[BehavioralSignal],
    ) -> float:
        """Compute engagement velocity: trend in reward over time.

        Splits signals into first half and second half by time,
        compares average reward. Positive = improving, negative = declining.

        Returns:
            Velocity in [-1, 1] range.
        """
        if len(signals) < 4:
            return 0.0

        # Sort by time ascending
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)
        mid = len(sorted_signals) // 2

        first_half = sorted_signals[:mid]
        second_half = sorted_signals[mid:]

        avg_first = sum(s.reward_value for s in first_half) / len(first_half)
        avg_second = sum(s.reward_value for s in second_half) / len(second_half)

        # Normalize to [-1, 1]
        diff = avg_second - avg_first
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, diff))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _energy_bucket(energy: float) -> str:
    """Map energy (0-1) to a bucket label."""
    if energy < 0.33:
        return "low_energy"
    elif energy < 0.66:
        return "med_energy"
    else:
        return "high_energy"


def _load_bucket(load: float) -> str:
    """Map cognitive load (0-1) to a bucket label."""
    if load < 0.33:
        return "low"
    elif load < 0.66:
        return "medium"
    else:
        return "high"
