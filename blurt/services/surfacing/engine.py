"""Composite task scoring engine for the surfacing package.

Combines the 6 individual scoring functions (mood, energy, time_of_day,
calendar_availability, entity_relevance, behavioral) into a single
weighted composite score per task.  Supports configurable weights per
signal dimension and optional Thompson Sampling for adaptive learning.

Design principles (anti-shame):
- No-tasks-pending is always valid — empty results are fine.
- No forced surfacing — minimum score threshold filters noise.
- No guilt language — all reasons are neutral/positive.
- Scores are transparent — every dimension is individually visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from blurt.services.surfacing.models import (
    ScoredTask,
    SurfacingContext,
    TaskItem,
)
from blurt.services.surfacing.scorers import (
    score_behavioral,
    score_calendar_availability,
    score_energy,
    score_entity_relevance,
    score_mood,
    score_time_of_day,
)
from blurt.services.surfacing.thompson import ThompsonSampler


class SignalDimension(str, Enum):
    """The six signal dimensions that compose the composite score."""

    MOOD = "mood"
    ENERGY = "energy"
    TIME_OF_DAY = "time_of_day"
    CALENDAR = "calendar"
    ENTITY_RELEVANCE = "entity_relevance"
    BEHAVIORAL = "behavioral"


@dataclass(frozen=True, slots=True)
class DimensionScore:
    """Score for a single signal dimension.

    Attributes:
        dimension: Which signal dimension this score is for.
        value: Score in [0.0, 1.0].
        weight: The configured weight for this dimension (before normalization).
        effective_weight: The weight after normalization and optional Thompson modulation.
        weighted_contribution: value * effective_weight (the actual contribution to composite).
    """

    dimension: SignalDimension
    value: float
    weight: float
    effective_weight: float
    weighted_contribution: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                f"Score value must be in [0.0, 1.0], got {self.value}"
            )


@dataclass
class SignalWeights:
    """Configurable weights for each signal dimension.

    Weights express relative importance — they are normalized internally
    so the absolute values don't matter, only their ratios.

    Example: mood=0.3, energy=0.2 means mood is 50% more important than
    energy in the final composite score.
    """

    mood: float = 0.20
    energy: float = 0.20
    time_of_day: float = 0.15
    calendar: float = 0.10
    entity_relevance: float = 0.20
    behavioral: float = 0.15

    def as_dict(self) -> dict[SignalDimension, float]:
        """Return weights as {SignalDimension: weight}."""
        return {
            SignalDimension.MOOD: self.mood,
            SignalDimension.ENERGY: self.energy,
            SignalDimension.TIME_OF_DAY: self.time_of_day,
            SignalDimension.CALENDAR: self.calendar,
            SignalDimension.ENTITY_RELEVANCE: self.entity_relevance,
            SignalDimension.BEHAVIORAL: self.behavioral,
        }

    @property
    def total(self) -> float:
        """Sum of all weights."""
        return sum(self.as_dict().values())

    def normalized(self) -> dict[SignalDimension, float]:
        """Return weights normalized to sum to 1.0."""
        t = self.total
        if t == 0:
            n = len(SignalDimension)
            return {d: 1.0 / n for d in SignalDimension}
        d = self.as_dict()
        return {k: v / t for k, v in d.items()}

    @classmethod
    def uniform(cls) -> SignalWeights:
        """Create uniform weights (equal importance for all dimensions)."""
        w = 1.0 / len(SignalDimension)
        return cls(
            mood=w,
            energy=w,
            time_of_day=w,
            calendar=w,
            entity_relevance=w,
            behavioral=w,
        )

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> SignalWeights:
        """Create SignalWeights from a {dimension_name: weight} dict.

        Unrecognized keys are ignored. Missing keys use the default value.
        """
        field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        valid = {k: v for k, v in d.items() if k in field_names}
        return cls(**valid)  # type: ignore[arg-type]


# Type alias for scorer functions
ScorerFn = Callable[[TaskItem, SurfacingContext], float]

# Default scorer registry: maps dimension → scorer function
DEFAULT_SCORERS: dict[SignalDimension, ScorerFn] = {
    SignalDimension.MOOD: score_mood,
    SignalDimension.ENERGY: score_energy,
    SignalDimension.TIME_OF_DAY: score_time_of_day,
    SignalDimension.CALENDAR: score_calendar_availability,
    SignalDimension.ENTITY_RELEVANCE: score_entity_relevance,
    SignalDimension.BEHAVIORAL: score_behavioral,
}


@dataclass
class CompositeResult:
    """Result of scoring a single task with full breakdown.

    Provides the composite score plus per-dimension detail for
    transparency and debugging.
    """

    task: TaskItem
    composite_score: float
    dimension_scores: tuple[DimensionScore, ...]

    @property
    def breakdown(self) -> dict[str, float]:
        """Return {dimension_name: value} for easy inspection."""
        return {ds.dimension.value: ds.value for ds in self.dimension_scores}

    @property
    def weighted_breakdown(self) -> dict[str, float]:
        """Return {dimension_name: weighted_contribution}."""
        return {
            ds.dimension.value: ds.weighted_contribution
            for ds in self.dimension_scores
        }

    @property
    def effective_weights(self) -> dict[str, float]:
        """Return {dimension_name: effective_weight}."""
        return {
            ds.dimension.value: ds.effective_weight
            for ds in self.dimension_scores
        }

    def to_scored_task(self) -> ScoredTask:
        """Convert to the ScoredTask model from surfacing.models."""
        scores = self.breakdown
        return ScoredTask(
            task=self.task,
            mood_score=scores.get("mood", 0.0),
            energy_score=scores.get("energy", 0.0),
            time_of_day_score=scores.get("time_of_day", 0.0),
            calendar_score=scores.get("calendar", 0.0),
            entity_relevance_score=scores.get("entity_relevance", 0.0),
            behavioral_score=scores.get("behavioral", 0.0),
            composite_score=self.composite_score,
        )


@dataclass
class RankingResult:
    """Result of ranking multiple tasks.

    Contains scored tasks sorted by composite score (descending),
    metadata about the ranking pass, and any Thompson weights used.
    """

    scored_tasks: list[CompositeResult] = field(default_factory=list)
    total_candidates: int = 0
    total_filtered: int = 0
    thompson_weights: dict[str, float] | None = None

    @property
    def has_tasks(self) -> bool:
        """Whether there are any tasks to surface. Empty is valid."""
        return len(self.scored_tasks) > 0

    @property
    def top(self) -> CompositeResult | None:
        """The highest-scored task, or None if empty."""
        return self.scored_tasks[0] if self.scored_tasks else None

    @property
    def top_n(self) -> list[CompositeResult]:
        """Alias for scored_tasks (already limited by max_results)."""
        return self.scored_tasks


class CompositeScoringEngine:
    """Combines individual signal scores using weighted aggregation.

    The engine:
    1. Evaluates each task against the current context using 6 scorer functions.
    2. Combines the raw scores via configurable weights per dimension.
    3. Optionally modulates weights with Thompson Sampling for adaptive learning.
    4. Returns ranked tasks with full score transparency.

    Anti-shame design:
    - Empty results are valid — no forced surfacing.
    - Completed tasks are excluded — no guilt about past items.
    - All scores use gentle decay — nothing penalizes the user.
    - Minimum score threshold prevents noise.

    Args:
        weights: Configurable signal weights. Defaults to balanced preset.
        min_score: Minimum composite score to include in results.
        max_results: Maximum number of tasks to surface per ranking pass.
        thompson_sampler: Optional adaptive weight learner.
        scorers: Optional custom scorer registry. Defaults to all 6 built-in scorers.
    """

    DEFAULT_MIN_SCORE: float = 0.15
    DEFAULT_MAX_RESULTS: int = 5

    def __init__(
        self,
        weights: SignalWeights | None = None,
        min_score: float | None = None,
        max_results: int | None = None,
        thompson_sampler: ThompsonSampler | None = None,
        scorers: dict[SignalDimension, ScorerFn] | None = None,
    ) -> None:
        self.weights = weights or SignalWeights()
        self.min_score = (
            min_score if min_score is not None else self.DEFAULT_MIN_SCORE
        )
        self.max_results = (
            max_results if max_results is not None else self.DEFAULT_MAX_RESULTS
        )
        self.thompson_sampler = thompson_sampler
        self._scorers = scorers or dict(DEFAULT_SCORERS)

    # ── Public API ─────────────────────────────────────────────────────

    def score_task(
        self,
        task: TaskItem,
        context: SurfacingContext,
        *,
        use_thompson: bool = True,
    ) -> CompositeResult:
        """Score a single task against the current context.

        Args:
            task: The task to score.
            context: Current user context snapshot.
            use_thompson: Whether to apply Thompson Sampling modulation.
                         Set to False for deterministic scoring.

        Returns:
            CompositeResult with full score breakdown.
        """
        thompson_weights = None
        if use_thompson and self.thompson_sampler is not None:
            thompson_weights = self.thompson_sampler.sample_weights()

        raw_scores = self._compute_raw_scores(task, context)
        return self._aggregate(task, raw_scores, thompson_weights)

    def rank_tasks(
        self,
        tasks: list[TaskItem],
        context: SurfacingContext,
    ) -> RankingResult:
        """Score and rank multiple tasks.

        Filters completed tasks, applies minimum score threshold,
        limits results, and returns sorted by composite score descending.

        Args:
            tasks: Candidate tasks.
            context: Current user context.

        Returns:
            RankingResult with ranked tasks and metadata.
        """
        # Filter out completed tasks
        eligible = [t for t in tasks if not t.completed]
        total_candidates = len(eligible)

        # Sample Thompson weights once per ranking pass for consistency
        thompson_weights: dict[str, float] | None = None
        if self.thompson_sampler is not None:
            thompson_weights = self.thompson_sampler.sample_weights()

        # Score each task
        scored: list[CompositeResult] = []
        for task in eligible:
            raw_scores = self._compute_raw_scores(task, context)
            result = self._aggregate(task, raw_scores, thompson_weights)
            if result.composite_score >= self.min_score:
                scored.append(result)

        # Sort by composite score descending
        scored.sort(key=lambda r: r.composite_score, reverse=True)

        # Limit results
        limited = scored[: self.max_results]
        total_filtered = len(scored) - len(limited)

        return RankingResult(
            scored_tasks=limited,
            total_candidates=total_candidates,
            total_filtered=total_filtered,
            thompson_weights=thompson_weights,
        )

    def update_weights(self, new_weights: SignalWeights) -> None:
        """Update the signal weights at runtime.

        Useful for user preferences or A/B testing.
        """
        self.weights = new_weights

    def get_weights(self) -> SignalWeights:
        """Return the current signal weights."""
        return self.weights

    def register_scorer(
        self,
        dimension: SignalDimension,
        scorer: ScorerFn,
    ) -> None:
        """Register or replace a scorer for a signal dimension.

        Enables extending the engine with custom scoring logic.
        """
        self._scorers[dimension] = scorer

    def record_feedback(
        self,
        result: CompositeResult,
        engaged: bool,
        learning_rate: float = 1.0,
    ) -> None:
        """Record user feedback for Thompson Sampling adaptation.

        Anti-shame: non-engagement is neutral ("not the right moment"),
        never a penalty.

        Args:
            result: The scored task the user saw.
            engaged: True if the user acted on the task.
            learning_rate: Update strength (default 1.0).

        No-op if no Thompson Sampler is attached.
        """
        if self.thompson_sampler is None:
            return

        # Use raw score values as signal contributions
        contributions = result.breakdown
        if engaged:
            self.thompson_sampler.record_feedback(
                contributions, reward=True, learning_rate=learning_rate
            )
        else:
            self.thompson_sampler.record_feedback(
                contributions, reward=False, learning_rate=learning_rate
            )

    # ── Internal ───────────────────────────────────────────────────────

    def _compute_raw_scores(
        self,
        task: TaskItem,
        context: SurfacingContext,
    ) -> dict[SignalDimension, float]:
        """Compute raw scores for all registered dimensions."""
        scores: dict[SignalDimension, float] = {}
        for dimension, scorer_fn in self._scorers.items():
            raw = scorer_fn(task, context)
            # Clamp to [0, 1] for safety
            scores[dimension] = max(0.0, min(1.0, raw))
        return scores

    def _aggregate(
        self,
        task: TaskItem,
        raw_scores: dict[SignalDimension, float],
        thompson_weights: dict[str, float] | None = None,
    ) -> CompositeResult:
        """Combine raw scores into a weighted composite.

        When Thompson weights are present, they modulate the base weights:
        effective_weight = base_weight * (thompson_weight * N_dimensions)
        so uniform Thompson weights leave base weights unchanged.

        All weights are normalized to sum to 1.0 before computing the
        weighted average.
        """
        base_weights = self.weights.as_dict()
        n_dims = len(base_weights)

        # Modulate with Thompson weights if present
        if thompson_weights:
            modulated: dict[SignalDimension, float] = {}
            for dim, base_w in base_weights.items():
                ts_w = thompson_weights.get(dim.value, 1.0 / n_dims)
                modulated[dim] = base_w * (ts_w * n_dims)
            effective = modulated
        else:
            effective = dict(base_weights)

        # Normalize weights to sum to 1.0
        total_weight = sum(effective.values())
        if total_weight == 0:
            # All weights zero — return zero composite
            dim_scores = tuple(
                DimensionScore(
                    dimension=dim,
                    value=raw_scores.get(dim, 0.0),
                    weight=0.0,
                    effective_weight=0.0,
                    weighted_contribution=0.0,
                )
                for dim in SignalDimension
            )
            return CompositeResult(
                task=task,
                composite_score=0.0,
                dimension_scores=dim_scores,
            )

        normalized = {k: v / total_weight for k, v in effective.items()}

        # Compute weighted contributions
        dim_scores_list: list[DimensionScore] = []
        composite = 0.0

        for dim in SignalDimension:
            value = raw_scores.get(dim, 0.0)
            eff_w = normalized.get(dim, 0.0)
            contribution = value * eff_w
            composite += contribution

            dim_scores_list.append(
                DimensionScore(
                    dimension=dim,
                    value=value,
                    weight=base_weights.get(dim, 0.0),
                    effective_weight=eff_w,
                    weighted_contribution=contribution,
                )
            )

        # Clamp composite to [0, 1]
        composite = max(0.0, min(1.0, composite))

        return CompositeResult(
            task=task,
            composite_score=composite,
            dimension_scores=tuple(dim_scores_list),
        )
