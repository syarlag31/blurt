"""Composite task scoring and surfacing engine for Blurt.

Surfaces the right task at the right time based on user context — mood,
energy, time of day, related entities, and emotional state. Follows
anti-shame design: no overdue counters, no guilt, no forced engagement.
"No tasks pending" is always a valid state.

The engine combines individual signal scores using configurable weights,
ranks eligible tasks, and returns a sorted list of surfaced tasks.

Signal dimensions:
- Time relevance: how time-sensitive is this task right now?
- Energy match: does the task's energy requirement match current energy?
- Context relevance: does the task relate to current entities/topics?
- Emotional alignment: is the user in the right emotional state for this?
- Momentum: has the user been working on related things recently?
- Freshness: how recently was this task created or mentioned?
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from blurt.services.surfacing.thompson import ThompsonSampler


class EnergyLevel(str, Enum):
    """User's current energy level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskStatus(str, Enum):
    """Task lifecycle status."""

    ACTIVE = "active"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    DROPPED = "dropped"  # user explicitly chose to drop — shame-free


class SignalType(str, Enum):
    """The individual signal dimensions that compose a task's surfacing score."""

    TIME_RELEVANCE = "time_relevance"
    ENERGY_MATCH = "energy_match"
    CONTEXT_RELEVANCE = "context_relevance"
    EMOTIONAL_ALIGNMENT = "emotional_alignment"
    MOMENTUM = "momentum"
    FRESHNESS = "freshness"


@dataclass(frozen=True, slots=True)
class SignalScore:
    """A single scored signal for a task.

    Attributes:
        signal: Which signal dimension this score belongs to.
        value: The score value in [0.0, 1.0].
        reason: Human-readable reason for the score (for debugging/transparency).
    """

    signal: SignalType
    value: float  # 0.0–1.0
    reason: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                f"Signal score must be between 0.0 and 1.0, got {self.value}"
            )


@dataclass
class SurfacingWeights:
    """Configurable weights for combining signal scores.

    Weights are normalized internally — they express relative importance,
    not absolute values. E.g., time_relevance=0.3 and energy_match=0.2
    means time is 50% more important than energy in the composite score.
    """

    time_relevance: float = 0.25
    energy_match: float = 0.20
    context_relevance: float = 0.20
    emotional_alignment: float = 0.15
    momentum: float = 0.10
    freshness: float = 0.10

    def as_dict(self) -> dict[SignalType, float]:
        """Return weights as a {SignalType: weight} mapping."""
        return {
            SignalType.TIME_RELEVANCE: self.time_relevance,
            SignalType.ENERGY_MATCH: self.energy_match,
            SignalType.CONTEXT_RELEVANCE: self.context_relevance,
            SignalType.EMOTIONAL_ALIGNMENT: self.emotional_alignment,
            SignalType.MOMENTUM: self.momentum,
            SignalType.FRESHNESS: self.freshness,
        }

    @property
    def total(self) -> float:
        """Sum of all weights for normalization."""
        return sum(self.as_dict().values())


@dataclass
class SurfaceableTask:
    """A task eligible for surfacing with its metadata.

    This is the input representation — a task from memory/storage that
    the scoring engine evaluates.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    status: TaskStatus = TaskStatus.ACTIVE
    intent: str = "task"  # original classified intent

    # Time context
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    due_at: datetime | None = None  # soft due — never used for guilt
    last_mentioned_at: datetime | None = None

    # Energy/effort metadata
    estimated_energy: EnergyLevel = EnergyLevel.MEDIUM
    estimated_duration_minutes: int | None = None

    # Entity/context linkage
    entity_ids: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    project: str | None = None

    # Emotional context from when this was captured
    capture_valence: float = 0.0  # -1.0 to 1.0
    capture_arousal: float = 0.5  # 0.0 to 1.0

    # Interaction history
    times_surfaced: int = 0
    times_deferred: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserContext:
    """The user's current state — used to score tasks contextually.

    This represents the "right now" snapshot: energy, mood, what they're
    working on, time of day, etc.
    """

    energy: EnergyLevel = EnergyLevel.MEDIUM
    current_valence: float = 0.0  # -1.0 to 1.0
    current_arousal: float = 0.5  # 0.0 to 1.0
    active_entity_ids: list[str] = field(default_factory=list)
    active_entity_names: list[str] = field(default_factory=list)
    active_project: str | None = None
    recent_task_ids: list[str] = field(default_factory=list)
    now: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass(frozen=True, slots=True)
class ScoredTask:
    """A task with its composite score and individual signal breakdown.

    This is the output of the scoring engine — tasks sorted by score,
    each with full transparency into why they scored the way they did.
    """

    task: SurfaceableTask
    composite_score: float  # 0.0–1.0
    signal_scores: tuple[SignalScore, ...]  # individual signal breakdown
    surfacing_reason: str = ""  # short human-readable reason

    @property
    def signal_breakdown(self) -> dict[str, float]:
        """Return {signal_name: value} for easy inspection."""
        return {s.signal.value: s.value for s in self.signal_scores}


@dataclass
class SurfacingResult:
    """Complete result of a task surfacing request.

    Contains the ranked list of tasks to surface, plus metadata about
    the surfacing decision.
    """

    tasks: list[ScoredTask] = field(default_factory=list)
    total_eligible: int = 0
    total_filtered: int = 0
    context_snapshot: UserContext | None = None
    thompson_weights: dict[str, float] | None = None

    @property
    def has_tasks(self) -> bool:
        """Whether there are any tasks to surface. Empty is valid."""
        return len(self.tasks) > 0

    @property
    def top_task(self) -> ScoredTask | None:
        """The highest-scored task, or None if empty."""
        return self.tasks[0] if self.tasks else None


class TaskScoringEngine:
    """Composite task scoring engine.

    Evaluates each task against the user's current context across multiple
    signal dimensions, combines them with configurable weights, and returns
    a ranked list of tasks to surface.

    Design principles:
    - No task is ever "overdue" — time relevance decays gently, not punitively
    - Empty results are valid — never force-surface tasks
    - Scores are transparent — every signal is individually visible
    - Anti-shame: deferred tasks don't accumulate negative signals
    """

    # Minimum composite score to include in surfacing results
    DEFAULT_MIN_SCORE: float = 0.15

    # Maximum number of tasks to surface at once
    DEFAULT_MAX_RESULTS: int = 5

    def __init__(
        self,
        weights: SurfacingWeights | None = None,
        min_score: float | None = None,
        max_results: int | None = None,
        thompson_sampler: ThompsonSampler | None = None,
    ) -> None:
        self.weights = weights or SurfacingWeights()
        self.min_score = min_score if min_score is not None else self.DEFAULT_MIN_SCORE
        self.max_results = max_results if max_results is not None else self.DEFAULT_MAX_RESULTS
        self.thompson_sampler = thompson_sampler

    def score_and_rank(
        self,
        tasks: list[SurfaceableTask],
        context: UserContext,
    ) -> SurfacingResult:
        """Score all eligible tasks and return ranked results.

        When a ThompsonSampler is attached, sampled weights modulate the
        base signal weights for this ranking pass. Each call samples fresh
        weights, introducing controlled exploration. The sampled weights
        are stored on the result for downstream feedback recording.

        Args:
            tasks: All candidate tasks (will be filtered for eligibility).
            context: The user's current context snapshot.

        Returns:
            SurfacingResult with tasks sorted by composite score descending.
        """
        # Filter to eligible tasks only
        eligible = [t for t in tasks if self._is_eligible(t)]
        total_eligible = len(eligible)

        # Sample Thompson weights once per ranking pass (consistent
        # within a single surfacing call, different across calls)
        thompson_weights: dict[str, float] | None = None
        if self.thompson_sampler is not None:
            thompson_weights = self.thompson_sampler.sample_weights()

        # Score each task
        scored: list[ScoredTask] = []
        for task in eligible:
            signals = self._compute_signals(task, context)
            composite = self._composite_score(signals, thompson_weights)

            if composite >= self.min_score:
                reason = self._generate_reason(signals, task)
                scored.append(
                    ScoredTask(
                        task=task,
                        composite_score=composite,
                        signal_scores=tuple(signals),
                        surfacing_reason=reason,
                    )
                )

        # Sort by composite score descending
        scored.sort(key=lambda s: s.composite_score, reverse=True)

        # Limit results
        filtered = scored[: self.max_results]

        result = SurfacingResult(
            tasks=filtered,
            total_eligible=total_eligible,
            total_filtered=len(scored) - len(filtered),
            context_snapshot=context,
        )

        # Attach Thompson weights to the result for feedback recording
        if thompson_weights is not None:
            result.thompson_weights = thompson_weights

        return result

    def score_single(
        self,
        task: SurfaceableTask,
        context: UserContext,
        use_thompson: bool = True,
    ) -> ScoredTask:
        """Score a single task against the current context.

        Useful for on-demand scoring without full ranking.

        Args:
            task: The task to score.
            context: Current user context.
            use_thompson: Whether to apply Thompson Sampling weights.
                         Set to False for deterministic scoring.
        """
        thompson_weights: dict[str, float] | None = None
        if use_thompson and self.thompson_sampler is not None:
            thompson_weights = self.thompson_sampler.sample_weights()

        signals = self._compute_signals(task, context)
        composite = self._composite_score(signals, thompson_weights)
        reason = self._generate_reason(signals, task)

        return ScoredTask(
            task=task,
            composite_score=composite,
            signal_scores=tuple(signals),
            surfacing_reason=reason,
        )

    def _is_eligible(self, task: SurfaceableTask) -> bool:
        """Check if a task is eligible for surfacing.

        Only active tasks are surfaced. Completed, deferred, and dropped
        tasks are excluded.
        """
        return task.status == TaskStatus.ACTIVE

    def _compute_signals(
        self,
        task: SurfaceableTask,
        context: UserContext,
    ) -> list[SignalScore]:
        """Compute all individual signal scores for a task."""
        return [
            self._score_time_relevance(task, context),
            self._score_energy_match(task, context),
            self._score_context_relevance(task, context),
            self._score_emotional_alignment(task, context),
            self._score_momentum(task, context),
            self._score_freshness(task, context),
        ]

    def _composite_score(
        self,
        signals: list[SignalScore],
        thompson_weights: dict[str, float] | None = None,
    ) -> float:
        """Combine individual signals into a weighted composite score.

        When Thompson Sampling is active, the sampled weights modulate the
        base configuration weights — effectively learning which signal
        dimensions best predict user engagement over time.

        Weights are normalized so they sum to 1.0 regardless of config.

        Args:
            signals: Individual signal scores for the task.
            thompson_weights: Optional Thompson-sampled weights keyed by
                signal name. When present, each base weight is multiplied
                by the corresponding sampled weight before normalization.
        """
        weight_map = self.weights.as_dict()

        if thompson_weights:
            # Modulate base weights with Thompson Sampling weights.
            # Thompson weights are normalized to sum to 1.0, with N signals
            # uniform would be 1/N each. We multiply base weight by
            # (thompson_weight * N) so that uniform Thompson weights
            # leave base weights unchanged.
            n_signals = len(weight_map)
            modulated = {}
            for signal_type, base_w in weight_map.items():
                signal_name = signal_type.value
                ts_w = thompson_weights.get(signal_name, 1.0 / n_signals)
                modulated[signal_type] = base_w * (ts_w * n_signals)
            weight_map = modulated

        total_weight = sum(weight_map.values())

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            weight_map.get(s.signal, 0.0) * s.value for s in signals
        )

        score = weighted_sum / total_weight
        return max(0.0, min(1.0, score))

    # --- Individual signal scorers ---

    def _score_time_relevance(
        self, task: SurfaceableTask, context: UserContext
    ) -> SignalScore:
        """Score how time-relevant this task is right now.

        Uses gentle decay — tasks without due dates get a moderate baseline.
        Tasks approaching their due date score higher. Tasks past due date
        do NOT get penalized extra — anti-shame design.
        """
        if task.due_at is None:
            # No due date: moderate baseline — not urgent, but not irrelevant
            return SignalScore(
                signal=SignalType.TIME_RELEVANCE,
                value=0.4,
                reason="no due date — moderate baseline",
            )

        hours_until = (task.due_at - context.now).total_seconds() / 3600.0

        if hours_until < 0:
            # Past due — gentle constant, NOT increasing penalty
            # Anti-shame: past-due tasks don't get worse over time
            value = 0.6
            reason = "past suggested time — gentle relevance"
        elif hours_until <= 2:
            # Within 2 hours — high relevance
            value = 0.9
            reason = "within 2 hours"
        elif hours_until <= 24:
            # Within a day — smooth decay
            value = 0.5 + 0.4 * math.exp(-hours_until / 8.0)
            reason = f"within {hours_until:.0f} hours"
        elif hours_until <= 168:  # 1 week
            # Within a week — gentle relevance
            value = 0.2 + 0.3 * math.exp(-hours_until / 48.0)
            reason = f"within {hours_until / 24:.0f} days"
        else:
            # Far future — low but nonzero
            value = 0.15
            reason = "far future"

        return SignalScore(
            signal=SignalType.TIME_RELEVANCE,
            value=max(0.0, min(1.0, value)),
            reason=reason,
        )

    def _score_energy_match(
        self, task: SurfaceableTask, context: UserContext
    ) -> SignalScore:
        """Score how well the task's energy requirement matches user's energy.

        Perfect match = 1.0, one level off = 0.5, two levels off = 0.2.
        """
        energy_order = {
            EnergyLevel.LOW: 0,
            EnergyLevel.MEDIUM: 1,
            EnergyLevel.HIGH: 2,
        }

        user_level = energy_order[context.energy]
        task_level = energy_order[task.estimated_energy]
        diff = abs(user_level - task_level)

        if diff == 0:
            value = 1.0
            reason = "perfect energy match"
        elif diff == 1:
            value = 0.5
            reason = "one energy level off"
        else:
            value = 0.2
            reason = "energy mismatch"

        return SignalScore(
            signal=SignalType.ENERGY_MATCH,
            value=value,
            reason=reason,
        )

    def _score_context_relevance(
        self, task: SurfaceableTask, context: UserContext
    ) -> SignalScore:
        """Score how relevant the task is to the user's current context.

        Based on entity overlap (people, projects, topics currently active).
        """
        if not context.active_entity_ids and not context.active_entity_names:
            # No active context — moderate baseline for all tasks
            return SignalScore(
                signal=SignalType.CONTEXT_RELEVANCE,
                value=0.3,
                reason="no active context",
            )

        # Check entity ID overlap
        id_overlap = len(
            set(task.entity_ids) & set(context.active_entity_ids)
        )
        # Check entity name overlap (case-insensitive)
        task_names_lower = {n.lower() for n in task.entity_names}
        context_names_lower = {n.lower() for n in context.active_entity_names}
        name_overlap = len(task_names_lower & context_names_lower)

        total_overlap = id_overlap + name_overlap

        # Check project match
        project_match = (
            task.project is not None
            and context.active_project is not None
            and task.project.lower() == context.active_project.lower()
        )

        if project_match and total_overlap > 0:
            value = min(1.0, 0.7 + 0.1 * total_overlap)
            reason = f"same project + {total_overlap} shared entities"
        elif project_match:
            value = 0.7
            reason = "same project"
        elif total_overlap > 0:
            value = min(0.8, 0.3 + 0.15 * total_overlap)
            reason = f"{total_overlap} shared entities"
        else:
            value = 0.1
            reason = "no context overlap"

        return SignalScore(
            signal=SignalType.CONTEXT_RELEVANCE,
            value=value,
            reason=reason,
        )

    def _score_emotional_alignment(
        self, task: SurfaceableTask, context: UserContext
    ) -> SignalScore:
        """Score whether the user's current emotional state aligns with the task.

        Avoids surfacing high-stress tasks when user is already stressed.
        Prefers lighter tasks when valence is low.
        """
        valence_diff = abs(context.current_valence - task.capture_valence)
        arousal_diff = abs(context.current_arousal - task.capture_arousal)

        # Low valence (user is down) — prefer low-arousal/positive tasks
        if context.current_valence < -0.3:
            if task.capture_arousal > 0.7:
                value = 0.2
                reason = "user mood low — avoiding high-intensity task"
            elif task.capture_valence >= 0:
                value = 0.7
                reason = "positive-capture task for low mood"
            else:
                value = 0.3
                reason = "both negative — low alignment"
        elif context.current_valence > 0.3:
            # User is in good mood — all tasks score well
            value = 0.6 + 0.2 * (1.0 - valence_diff)
            reason = "user in positive mood"
        else:
            # Neutral mood — moderate alignment across the board
            value = 0.5
            reason = "neutral mood — moderate alignment"

        # Slight boost if arousal levels are similar (flow state matching)
        if arousal_diff < 0.2:
            value = min(1.0, value + 0.1)
            reason += " + arousal match"

        return SignalScore(
            signal=SignalType.EMOTIONAL_ALIGNMENT,
            value=max(0.0, min(1.0, value)),
            reason=reason,
        )

    def _score_momentum(
        self, task: SurfaceableTask, context: UserContext
    ) -> SignalScore:
        """Score whether surfacing this task continues the user's current momentum.

        If the user has been working on related tasks recently, surfacing
        a related task supports flow state.
        """
        if not context.recent_task_ids:
            return SignalScore(
                signal=SignalType.MOMENTUM,
                value=0.3,
                reason="no recent task history",
            )

        # Check if this task's entities overlap with recently-worked tasks' context
        # For now, use a simple "is this task related to recent work?" check
        # via entity overlap with active context (momentum implies continuation)
        if task.id in context.recent_task_ids:
            # Already recently worked on — moderate score (don't spam same task)
            return SignalScore(
                signal=SignalType.MOMENTUM,
                value=0.4,
                reason="recently worked on — moderate momentum",
            )

        # Check entity/project overlap for momentum
        has_project_overlap = (
            task.project is not None
            and context.active_project is not None
            and task.project.lower() == context.active_project.lower()
        )

        task_names_lower = {n.lower() for n in task.entity_names}
        context_names_lower = {n.lower() for n in context.active_entity_names}
        entity_overlap = len(task_names_lower & context_names_lower)

        if has_project_overlap and entity_overlap > 0:
            value = 0.9
            reason = "strong momentum — same project + shared entities"
        elif has_project_overlap:
            value = 0.7
            reason = "project momentum"
        elif entity_overlap > 0:
            value = 0.6
            reason = f"entity momentum ({entity_overlap} shared)"
        else:
            value = 0.2
            reason = "no momentum signal"

        return SignalScore(
            signal=SignalType.MOMENTUM,
            value=value,
            reason=reason,
        )

    def _score_freshness(
        self, task: SurfaceableTask, context: UserContext
    ) -> SignalScore:
        """Score how recently the task was created or mentioned.

        Newer tasks and recently-mentioned tasks score higher.
        Uses gentle decay — old tasks don't disappear, they just
        score lower on this one dimension.
        """
        # Use most recent timestamp
        reference_time = task.last_mentioned_at or task.created_at
        age_hours = (context.now - reference_time).total_seconds() / 3600.0

        if age_hours < 0:
            age_hours = 0.0

        if age_hours <= 1:
            value = 1.0
            reason = "very fresh (< 1 hour)"
        elif age_hours <= 24:
            # Smooth decay over first day
            value = 0.5 + 0.5 * math.exp(-age_hours / 8.0)
            reason = f"fresh ({age_hours:.0f} hours)"
        elif age_hours <= 168:  # 1 week
            value = 0.2 + 0.3 * math.exp(-(age_hours - 24) / 72.0)
            reason = f"{age_hours / 24:.0f} days old"
        else:
            # Old — low but nonzero freshness
            value = 0.15
            reason = "older task"

        return SignalScore(
            signal=SignalType.FRESHNESS,
            value=max(0.0, min(1.0, value)),
            reason=reason,
        )

    def record_task_feedback(
        self,
        scored_task: ScoredTask,
        engaged: bool,
        learning_rate: float = 1.0,
    ) -> None:
        """Record user feedback for a surfaced task.

        Updates the Thompson Sampler's arm distributions based on whether
        the user engaged with the task. This closes the feedback loop:
        surface → user response → weight adaptation.

        Anti-shame: non-engagement is neutral feedback ("not the right
        moment"), never a penalty.

        Args:
            scored_task: The scored task the user interacted with.
            engaged: True if user completed/started the task.
                    False if surfaced but not engaged.
            learning_rate: Strength of the feedback (default 1.0).

        No-op if no ThompsonSampler is attached.
        """
        if self.thompson_sampler is None:
            return

        signal_scores = scored_task.signal_breakdown
        composite = scored_task.composite_score

        if engaged:
            self.thompson_sampler.record_completion(
                signal_scores, composite, learning_rate
            )
        else:
            self.thompson_sampler.record_dismissal(
                signal_scores, composite, learning_rate
            )

    def _generate_reason(
        self, signals: list[SignalScore], task: SurfaceableTask
    ) -> str:
        """Generate a short human-readable reason for why this task was surfaced.

        Picks the top-scoring signal as the primary reason.
        Anti-shame: never uses guilt language.
        """
        if not signals:
            return "available task"

        # Find highest-scoring signal
        best = max(signals, key=lambda s: s.value)
        return best.reason
