"""Thompson Sampling ranking pipeline — unifies all Thompson layers with context scoring.

Integrates three levels of Thompson Sampling into the task surfacing pipeline:
1. Signal-level: ThompsonSampler modulates which scoring dimensions matter most
2. Category-level: ThompsonSamplingEngine samples per-intent engagement probability
3. Task-level: TaskFeedbackService samples per-task + per-context engagement

The pipeline produces a final score per task that combines:
    final_score = context_score * (1 - ts_blend) + thompson_score * ts_blend

Where:
    context_score = composite from TaskScoringEngine (with signal-level Thompson modulation)
    thompson_score = blend of category-level and task-level Thompson samples
    ts_blend = how much Thompson Sampling influences ranking (grows with observations)

AC 11 Sub-AC 3: Integrate Thompson Sampling weights into the surfacing/ranking pipeline.

Anti-shame design:
- No task is ever zeroed out by Thompson Sampling
- Minimum floor on all scores ensures exploration
- Empty results remain valid — never force-surface tasks
- Dismissals are neutral learning signals, not punishment
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from blurt.services.feedback import (
    FeedbackAction,
    TaskFeedbackService,
    _build_context_key,
    _energy_bucket,
    _mood_bucket,
)
from blurt.services.task_surfacing import (
    EnergyLevel,
    ScoredTask,
    SurfaceableTask,
    SurfacingResult,
    SurfacingWeights,
    TaskScoringEngine,
    TaskStatus,
    UserContext,
)
from blurt.services.surfacing.thompson import ThompsonSampler
from blurt.services.thompson_sampling import (
    ThompsonSamplingEngine,
    FeedbackType,
)


@dataclass(frozen=True, slots=True)
class ThompsonScoreBreakdown:
    """Detailed breakdown of how Thompson Sampling contributed to a task's score.

    Attributes:
        context_score: Base composite score from context-based scoring engine.
        signal_weights: Thompson-modulated signal weights used for context scoring.
        category_sample: Sampled engagement probability for the task's intent category.
        task_sample: Per-task Thompson sample (blended task-level + intent-level).
        thompson_blend: Final blended Thompson score (category + task).
        blend_factor: How much Thompson influenced the final score (0-1).
        final_score: Combined score: context × (1 - blend) + thompson × blend.
    """

    context_score: float
    signal_weights: dict[str, float] | None
    category_sample: float
    task_sample: float
    thompson_blend: float
    blend_factor: float
    final_score: float


@dataclass
class RankedTask:
    """A task with its final Thompson-integrated ranking score.

    Contains the scored task from the context engine plus the Thompson
    integration breakdown for full transparency.
    """

    scored_task: ScoredTask
    thompson_breakdown: ThompsonScoreBreakdown
    final_score: float

    @property
    def task(self) -> SurfaceableTask:
        return self.scored_task.task

    @property
    def task_id(self) -> str:
        return self.scored_task.task.id


@dataclass
class ThompsonRankingResult:
    """Complete result of a Thompson-integrated ranking pass.

    Contains ranked tasks sorted by final_score (descending),
    plus metadata about the ranking for transparency and debugging.
    """

    ranked_tasks: list[RankedTask] = field(default_factory=list)
    total_candidates: int = 0
    total_filtered: int = 0
    signal_thompson_weights: dict[str, float] | None = None
    category_samples: dict[str, float] | None = None
    blend_factor_used: float = 0.0

    @property
    def has_tasks(self) -> bool:
        """Whether there are any tasks to surface. Empty is valid."""
        return len(self.ranked_tasks) > 0

    @property
    def top(self) -> RankedTask | None:
        """The highest-ranked task, or None if empty."""
        return self.ranked_tasks[0] if self.ranked_tasks else None

    @property
    def tasks(self) -> list[ScoredTask]:
        """Return ScoredTasks for backward compatibility."""
        return [rt.scored_task for rt in self.ranked_tasks]


class ThompsonRankingPipeline:
    """Unified pipeline that integrates Thompson Sampling into task ranking.

    Combines three levels of Thompson Sampling with context-based scoring:

    1. **Signal-level** (via ThompsonSampler in the scoring engine):
       Learns which scoring dimensions (time, energy, mood, etc.) best
       predict user engagement. Modulates the weights of the composite
       scoring engine.

    2. **Category-level** (via ThompsonSamplingEngine):
       Learns per-intent-type engagement probability. Tasks from
       historically-engaged categories get a boost.

    3. **Task-level** (via TaskFeedbackService):
       Learns per-task and per-context engagement. Tasks the user has
       previously engaged with in similar contexts get a boost.

    The final score for each task:
        final = context_score × (1 - blend) + thompson_score × blend

    Where blend grows as we accumulate more observations (cold start =
    low blend, letting context dominate; more data = higher blend,
    letting learned preferences dominate).

    Anti-shame design:
    - Minimum floor on all Thompson scores prevents zero-out
    - Category suppression is bounded — no category is fully ignored
    - Empty results are valid and expected
    - Feedback is always gentle (dismissals are neutral signals)

    Args:
        scoring_engine: The context-based scoring engine (may include signal-level Thompson).
        category_engine: Optional category-level Thompson Sampling engine.
        feedback_service: Optional per-task feedback service with Thompson params.
        category_weight: Weight of category-level Thompson in the combined Thompson score.
        task_weight: Weight of task-level Thompson in the combined Thompson score.
        max_blend_factor: Maximum blend factor (caps how much Thompson can override context).
        blend_observations_scale: Number of observations needed for blend to reach ~63% of max.
        min_thompson_score: Floor for Thompson scores (ensures exploration).
        min_final_score: Minimum final score to include in results.
        max_results: Maximum number of tasks to return.
    """

    DEFAULT_CATEGORY_WEIGHT: float = 0.4
    DEFAULT_TASK_WEIGHT: float = 0.6
    DEFAULT_MAX_BLEND: float = 0.5
    DEFAULT_BLEND_SCALE: float = 20.0
    DEFAULT_MIN_THOMPSON: float = 0.1
    DEFAULT_MIN_FINAL: float = 0.10
    DEFAULT_MAX_RESULTS: int = 5

    def __init__(
        self,
        scoring_engine: TaskScoringEngine | None = None,
        category_engine: ThompsonSamplingEngine | None = None,
        feedback_service: TaskFeedbackService | None = None,
        *,
        category_weight: float | None = None,
        task_weight: float | None = None,
        max_blend_factor: float | None = None,
        blend_observations_scale: float | None = None,
        min_thompson_score: float | None = None,
        min_final_score: float | None = None,
        max_results: int | None = None,
    ) -> None:
        self.scoring_engine = scoring_engine or TaskScoringEngine()
        self.category_engine = category_engine
        self.feedback_service = feedback_service

        self.category_weight = category_weight if category_weight is not None else self.DEFAULT_CATEGORY_WEIGHT
        self.task_weight = task_weight if task_weight is not None else self.DEFAULT_TASK_WEIGHT
        self.max_blend_factor = max_blend_factor if max_blend_factor is not None else self.DEFAULT_MAX_BLEND
        self.blend_observations_scale = blend_observations_scale if blend_observations_scale is not None else self.DEFAULT_BLEND_SCALE
        self.min_thompson_score = min_thompson_score if min_thompson_score is not None else self.DEFAULT_MIN_THOMPSON
        self.min_final_score = min_final_score if min_final_score is not None else self.DEFAULT_MIN_FINAL
        self.max_results = max_results if max_results is not None else self.DEFAULT_MAX_RESULTS

    def rank(
        self,
        tasks: list[SurfaceableTask],
        context: UserContext,
        *,
        context_key: str | None = None,
    ) -> ThompsonRankingResult:
        """Score and rank tasks using all Thompson Sampling layers combined with context.

        This is the main entry point for the integrated ranking pipeline.

        Args:
            tasks: Candidate tasks to rank.
            context: Current user context (energy, mood, time, entities).
            context_key: Optional pre-built context key for feedback service.
                         If None, built from context automatically.

        Returns:
            ThompsonRankingResult with tasks ranked by final_score (descending).
        """
        # Filter to eligible tasks only
        eligible = [t for t in tasks if t.status == TaskStatus.ACTIVE]
        total_candidates = len(eligible)

        if not eligible:
            return ThompsonRankingResult(
                total_candidates=0,
                total_filtered=0,
            )

        # Build context key for feedback service if not provided
        if context_key is None:
            time_of_day = self._time_of_day_bucket(context.now)
            ctx_key = _build_context_key(
                time_of_day=time_of_day,
                energy_bucket=_energy_bucket(self._energy_to_float(context.energy)),
                mood_bucket=_mood_bucket(context.current_valence),
            )
        else:
            ctx_key = context_key

        # Step 1: Get context-based scores using the scoring engine
        # (includes signal-level Thompson modulation if sampler attached)
        base_result = self.scoring_engine.score_and_rank(eligible, context)
        signal_weights = base_result.thompson_weights

        # Step 2: Get category-level Thompson samples
        category_samples: dict[str, float] | None = None
        if self.category_engine is not None:
            try:
                sample_result = self.category_engine.sample(apply_decay=True)
                category_samples = sample_result.all_samples
            except Exception:
                category_samples = None

        # Step 3: Compute blend factor based on total observations
        blend_factor = self._compute_blend_factor()

        # Step 4: Score each task with full Thompson integration
        ranked: list[RankedTask] = []

        # Build a lookup from task ID to ScoredTask from base_result
        scored_lookup: dict[str, ScoredTask] = {
            st.task.id: st for st in base_result.tasks
        }

        # Also score tasks that didn't pass the scoring engine's threshold
        # but might pass with Thompson boost
        for task in eligible:
            # Get context score (from engine result or re-score)
            if task.id in scored_lookup:
                scored = scored_lookup[task.id]
                context_score = scored.composite_score
            else:
                # Task was below engine threshold — score it directly
                scored = self.scoring_engine.score_single(
                    task, context, use_thompson=True
                )
                context_score = scored.composite_score

            # Get category-level Thompson sample for this task's intent
            category_sample = 0.5  # neutral default
            if category_samples and task.intent in category_samples:
                category_sample = category_samples[task.intent]
            elif category_samples:
                # Unknown category — use average
                category_sample = sum(category_samples.values()) / len(category_samples)

            # Get task-level Thompson sample from feedback service
            task_sample = 0.5  # neutral default (uniform prior)
            if self.feedback_service is not None:
                try:
                    task_sample = self.feedback_service.sample_score(
                        task_id=task.id,
                        intent=task.intent,
                        context_key=ctx_key,
                    )
                except Exception:
                    task_sample = 0.5

            # Apply minimum floor (anti-shame: nothing zeroed out)
            category_sample = max(self.min_thompson_score, category_sample)
            task_sample = max(self.min_thompson_score, task_sample)

            # Combine category and task Thompson scores
            thompson_blend = (
                self.category_weight * category_sample
                + self.task_weight * task_sample
            )

            # Compute final score: weighted combination of context and Thompson
            final_score = (
                context_score * (1.0 - blend_factor)
                + thompson_blend * blend_factor
            )

            # Clamp to [0, 1]
            final_score = max(0.0, min(1.0, final_score))

            # Apply minimum final score filter
            if final_score < self.min_final_score:
                continue

            breakdown = ThompsonScoreBreakdown(
                context_score=context_score,
                signal_weights=signal_weights,
                category_sample=category_sample,
                task_sample=task_sample,
                thompson_blend=thompson_blend,
                blend_factor=blend_factor,
                final_score=final_score,
            )

            ranked.append(
                RankedTask(
                    scored_task=scored,
                    thompson_breakdown=breakdown,
                    final_score=final_score,
                )
            )

        # Sort by final score descending
        ranked.sort(key=lambda r: r.final_score, reverse=True)

        # Limit results
        total_above = len(ranked)
        ranked = ranked[: self.max_results]
        total_filtered = total_above - len(ranked)

        return ThompsonRankingResult(
            ranked_tasks=ranked,
            total_candidates=total_candidates,
            total_filtered=total_filtered,
            signal_thompson_weights=signal_weights,
            category_samples=category_samples,
            blend_factor_used=blend_factor,
        )

    def record_feedback(
        self,
        ranked_task: RankedTask,
        action: FeedbackAction,
        user_id: str = "",
        mood_valence: float = 0.0,
        energy_level: float = 0.5,
        time_of_day: str = "",
    ) -> None:
        """Record user feedback and update all Thompson Sampling layers.

        Closes the full feedback loop across all three Thompson layers:
        1. Signal-level: updates which scoring dimensions predict engagement
        2. Category-level: updates per-intent engagement probability
        3. Task-level: updates per-task and per-context engagement

        Anti-shame: all actions are respected. Dismissals are neutral
        learning signals, not punishment.

        Args:
            ranked_task: The ranked task the user interacted with.
            action: What the user did (accept, dismiss, snooze, complete).
            user_id: User identifier.
            mood_valence: Current mood at feedback time.
            energy_level: Current energy at feedback time.
            time_of_day: Time-of-day bucket.
        """
        engaged = action in (FeedbackAction.ACCEPT, FeedbackAction.COMPLETE)

        # 1. Signal-level feedback (via scoring engine)
        self.scoring_engine.record_task_feedback(
            ranked_task.scored_task,
            engaged=engaged,
        )

        # 2. Category-level feedback (via category engine)
        if self.category_engine is not None:
            intent = ranked_task.task.intent
            # Map FeedbackAction to ThompsonSamplingEngine's FeedbackType
            feedback_map = {
                FeedbackAction.ACCEPT: FeedbackType.ACCEPTED,
                FeedbackAction.COMPLETE: FeedbackType.COMPLETED,
                FeedbackAction.DISMISS: FeedbackType.DISMISSED,
                FeedbackAction.SNOOZE: FeedbackType.SNOOZED,
            }
            ts_feedback = feedback_map.get(action, FeedbackType.IGNORED)
            try:
                self.category_engine.update(intent, ts_feedback)
            except KeyError:
                # Unknown category — add it dynamically
                self.category_engine.add_category(intent)
                self.category_engine.update(intent, ts_feedback)

        # 3. Task-level feedback (via feedback service)
        if self.feedback_service is not None:
            self.feedback_service.record_feedback(
                task_id=ranked_task.task_id,
                user_id=user_id,
                action=action,
                mood_valence=mood_valence,
                energy_level=energy_level,
                time_of_day=time_of_day,
                intent=ranked_task.task.intent,
            )

    def _compute_blend_factor(self) -> float:
        """Compute how much Thompson Sampling should influence ranking.

        Starts low (context dominates) and grows as we accumulate
        observations. This ensures cold-start behavior is sensible
        while allowing learned preferences to take over with data.

        Formula: blend = max_blend × (1 - exp(-total_obs / scale))
        """
        total_obs = 0

        # Count observations from category engine
        if self.category_engine is not None:
            for params in self.category_engine.params.values():
                total_obs += params.total_observations

        # Count observations from signal-level sampler
        if self.scoring_engine.thompson_sampler is not None:
            for arm in self.scoring_engine.thompson_sampler.arms.values():
                total_obs += int(arm.total_observations)

        if total_obs == 0:
            return 0.0

        import math
        blend = self.max_blend_factor * (
            1.0 - math.exp(-total_obs / self.blend_observations_scale)
        )
        return min(self.max_blend_factor, blend)

    @staticmethod
    def _time_of_day_bucket(dt: datetime) -> str:
        """Map datetime to a time-of-day bucket string."""
        hour = dt.hour
        if 5 <= hour < 8:
            return "early_morning"
        elif 8 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    @staticmethod
    def _energy_to_float(energy: EnergyLevel) -> float:
        """Convert EnergyLevel enum to a float for context key building."""
        return {
            EnergyLevel.LOW: 0.2,
            EnergyLevel.MEDIUM: 0.5,
            EnergyLevel.HIGH: 0.8,
        }.get(energy, 0.5)
