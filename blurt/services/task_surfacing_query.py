"""Task surfacing query service — filters, scores, and ranks tasks for the user.

Provides the query interface that:
1. Maintains a store of surfaceable tasks
2. Filters to eligible tasks (active only, respecting anti-shame design)
3. Scores them via the composite TaskScoringEngine
4. Returns a ranked list with full score breakdowns

Anti-shame design:
- Empty results are valid and expected — never force-surface tasks
- No overdue penalties, no guilt, no streak counters
- Deferred/dropped tasks are excluded silently (user's choice is respected)
- "No tasks pending" is a healthy state
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from blurt.services.task_surfacing import (
    EnergyLevel,
    ScoredTask,
    SignalType,
    SurfaceableTask,
    SurfacingResult,
    SurfacingWeights,
    TaskScoringEngine,
    TaskStatus,
    UserContext,
)


class TaskStore(Protocol):
    """Protocol for persisting surfaceable tasks."""

    def add_task(self, task: SurfaceableTask) -> None: ...

    def get_task(self, task_id: str) -> SurfaceableTask | None: ...

    def get_all_tasks(self, user_id: str | None = None) -> list[SurfaceableTask]: ...

    def update_task(self, task: SurfaceableTask) -> None: ...

    def remove_task(self, task_id: str) -> bool: ...


class InMemoryTaskStore:
    """In-memory task store for development and testing.

    Tasks are keyed by (user_id, task_id). When user_id is None,
    tasks are stored under a default user.
    """

    DEFAULT_USER = "__default__"

    def __init__(self) -> None:
        # user_id -> task_id -> task
        self._tasks: dict[str, dict[str, SurfaceableTask]] = {}

    def add_task(
        self, task: SurfaceableTask, user_id: str | None = None
    ) -> None:
        uid = user_id or self.DEFAULT_USER
        if uid not in self._tasks:
            self._tasks[uid] = {}
        self._tasks[uid][task.id] = task

    def get_task(
        self, task_id: str, user_id: str | None = None
    ) -> SurfaceableTask | None:
        uid = user_id or self.DEFAULT_USER
        return self._tasks.get(uid, {}).get(task_id)

    def get_all_tasks(self, user_id: str | None = None) -> list[SurfaceableTask]:
        uid = user_id or self.DEFAULT_USER
        return list(self._tasks.get(uid, {}).values())

    def update_task(
        self, task: SurfaceableTask, user_id: str | None = None
    ) -> None:
        uid = user_id or self.DEFAULT_USER
        if uid in self._tasks and task.id in self._tasks[uid]:
            self._tasks[uid][task.id] = task

    def remove_task(
        self, task_id: str, user_id: str | None = None
    ) -> bool:
        uid = user_id or self.DEFAULT_USER
        if uid in self._tasks and task_id in self._tasks[uid]:
            del self._tasks[uid][task_id]
            return True
        return False

    def clear(self, user_id: str | None = None) -> None:
        """Clear all tasks for a user (or all users if None)."""
        if user_id:
            self._tasks.pop(user_id, None)
        else:
            self._tasks.clear()


@dataclass
class SurfacingQuery:
    """Query parameters for task surfacing.

    Defines the user's current context and any filters to apply
    before scoring.
    """

    # User identification
    user_id: str = ""

    # Current user context for scoring
    energy: EnergyLevel = EnergyLevel.MEDIUM
    mood_valence: float = 0.0  # -1.0 to 1.0
    mood_arousal: float = 0.5  # 0.0 to 1.0
    active_entity_ids: list[str] = field(default_factory=list)
    active_entity_names: list[str] = field(default_factory=list)
    active_project: str | None = None
    recent_task_ids: list[str] = field(default_factory=list)

    # Filtering options
    max_results: int = 5
    min_score: float = 0.15
    include_intents: list[str] | None = None  # None = all intents
    exclude_entity_ids: list[str] | None = None
    tags_filter: list[str] | None = None  # only tasks with any of these tags
    max_energy: EnergyLevel | None = None  # cap on task energy requirement

    # Time override (for testing)
    now: datetime | None = None

    # Custom weights override
    weights: SurfacingWeights | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SurfacingQueryResult:
    """Complete result of a surfacing query.

    Contains ranked tasks with full score breakdowns, plus metadata
    about the query execution.
    """

    # The ranked, scored tasks
    tasks: list[ScoredTask] = field(default_factory=list)

    # Query metadata
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_in_store: int = 0
    total_eligible: int = 0
    total_after_filters: int = 0
    total_above_threshold: int = 0
    returned_count: int = 0

    # Context used for scoring
    context_snapshot: UserContext | None = None
    weights_used: dict[str, float] | None = None

    # Anti-shame: empty is valid
    message: str = ""

    @property
    def has_tasks(self) -> bool:
        """Whether there are any tasks to surface. Empty is valid."""
        return len(self.tasks) > 0

    @property
    def top_task(self) -> ScoredTask | None:
        """The highest-scored task, or None if empty."""
        return self.tasks[0] if self.tasks else None

    def score_breakdown(self, index: int = 0) -> dict[str, float] | None:
        """Get the signal breakdown for the task at the given index."""
        if index < 0 or index >= len(self.tasks):
            return None
        return self.tasks[index].signal_breakdown


# Anti-shame messages for empty results
_EMPTY_MESSAGES = [
    "All clear — nothing needs your attention right now.",
    "You're caught up. Enjoy the moment.",
    "No tasks waiting. That's a good thing.",
    "Nothing to surface right now.",
    "All clear for now.",
]


class TaskSurfacingQueryService:
    """Service that handles task surfacing queries.

    Orchestrates the full flow:
    1. Retrieve tasks from the store
    2. Apply pre-scoring filters (intent, entity, tag, energy)
    3. Build UserContext from the query
    4. Score and rank via TaskScoringEngine (or ThompsonRankingPipeline)
    5. Return results with full transparency

    When a ThompsonRankingPipeline is provided, it integrates all three
    Thompson Sampling layers (signal-level, category-level, task-level)
    into the ranking. Otherwise falls back to direct engine scoring.

    Anti-shame design: empty results get a positive message,
    never guilt language.
    """

    def __init__(
        self,
        store: InMemoryTaskStore | None = None,
        engine: TaskScoringEngine | None = None,
        thompson_pipeline: Any | None = None,
    ) -> None:
        self.store = store or InMemoryTaskStore()
        self.engine = engine or TaskScoringEngine()
        self.thompson_pipeline = thompson_pipeline

    def query(self, query: SurfacingQuery) -> SurfacingQueryResult:
        """Execute a surfacing query and return ranked results.

        When a ThompsonRankingPipeline is attached, uses the full
        Thompson-integrated ranking. Otherwise uses the base scoring engine.

        Args:
            query: The surfacing query with user context and filters.

        Returns:
            SurfacingQueryResult with ranked tasks and metadata.
        """
        import random

        # 1. Get all tasks from the store
        all_tasks = self.store.get_all_tasks(
            user_id=query.user_id or None
        )
        total_in_store = len(all_tasks)

        # 2. Apply pre-scoring filters
        filtered = self._apply_filters(all_tasks, query)
        total_after_filters = len(filtered)

        # 3. Build UserContext from query
        context = UserContext(
            energy=query.energy,
            current_valence=query.mood_valence,
            current_arousal=query.mood_arousal,
            active_entity_ids=query.active_entity_ids,
            active_entity_names=query.active_entity_names,
            active_project=query.active_project,
            recent_task_ids=query.recent_task_ids,
            now=query.now or datetime.now(timezone.utc),
        )

        # 4. Route to Thompson pipeline or base engine
        if self.thompson_pipeline is not None:
            return self._query_with_thompson(
                filtered, context, query, total_in_store, total_after_filters
            )

        # Fallback: use base engine (original path)
        return self._query_with_engine(
            filtered, context, query, total_in_store, total_after_filters
        )

    def _query_with_thompson(
        self,
        filtered: list[SurfaceableTask],
        context: UserContext,
        query: SurfacingQuery,
        total_in_store: int,
        total_after_filters: int,
    ) -> SurfacingQueryResult:
        """Execute ranking via the full Thompson pipeline."""
        import random
        from blurt.services.thompson_ranking import ThompsonRankingPipeline

        pipeline: ThompsonRankingPipeline = self.thompson_pipeline
        result = pipeline.rank(filtered, context)

        # Increment surface counts
        for ranked_task in result.ranked_tasks:
            ranked_task.task.times_surfaced += 1
            self.store.update_task(
                ranked_task.task, user_id=query.user_id or None
            )

        # Build weights used (include Thompson metadata)
        weights_dict = (query.weights or self.engine.weights).as_dict()
        weights_used = {k.value: v for k, v in weights_dict.items()}

        if result.signal_thompson_weights:
            weights_used["_thompson_modulated"] = True  # type: ignore[assignment]
        weights_used["_blend_factor"] = result.blend_factor_used  # type: ignore[assignment]

        message = ""
        if not result.has_tasks:
            message = random.choice(_EMPTY_MESSAGES)

        return SurfacingQueryResult(
            tasks=result.tasks,
            total_in_store=total_in_store,
            total_eligible=result.total_candidates,
            total_after_filters=total_after_filters,
            total_above_threshold=len(result.ranked_tasks) + result.total_filtered,
            returned_count=len(result.ranked_tasks),
            context_snapshot=context,
            weights_used=weights_used,
            message=message,
        )

    def _query_with_engine(
        self,
        filtered: list[SurfaceableTask],
        context: UserContext,
        query: SurfacingQuery,
        total_in_store: int,
        total_after_filters: int,
    ) -> SurfacingQueryResult:
        """Execute ranking via the base scoring engine (original path)."""
        import random

        # Create engine with query-specific overrides if needed
        engine = self.engine
        if query.weights is not None or query.min_score != 0.15 or query.max_results != 5:
            engine = TaskScoringEngine(
                weights=query.weights or self.engine.weights,
                min_score=query.min_score,
                max_results=query.max_results,
                thompson_sampler=self.engine.thompson_sampler,
            )

        # Score and rank via the engine
        result = engine.score_and_rank(filtered, context)

        # Increment surface counts for returned tasks
        for scored_task in result.tasks:
            scored_task.task.times_surfaced += 1
            self.store.update_task(
                scored_task.task, user_id=query.user_id or None
            )

        # Build the query result
        weights_dict = (query.weights or self.engine.weights).as_dict()
        weights_used = {k.value: v for k, v in weights_dict.items()}

        if result.thompson_weights:
            weights_used["_thompson_modulated"] = True  # type: ignore[assignment]

        message = ""
        if not result.tasks:
            message = random.choice(_EMPTY_MESSAGES)

        return SurfacingQueryResult(
            tasks=result.tasks,
            total_in_store=total_in_store,
            total_eligible=result.total_eligible,
            total_after_filters=total_after_filters,
            total_above_threshold=len(result.tasks) + result.total_filtered,
            returned_count=len(result.tasks),
            context_snapshot=context,
            weights_used=weights_used,
            message=message,
        )

    def add_task(
        self, task: SurfaceableTask, user_id: str | None = None
    ) -> None:
        """Add a task to the store."""
        self.store.add_task(task, user_id=user_id)

    def get_task(
        self, task_id: str, user_id: str | None = None
    ) -> SurfaceableTask | None:
        """Get a task by ID."""
        return self.store.get_task(task_id, user_id=user_id)

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        user_id: str | None = None,
    ) -> SurfaceableTask | None:
        """Update a task's status. All status transitions are shame-free."""
        task = self.store.get_task(task_id, user_id=user_id)
        if task is None:
            return None
        task.status = status
        self.store.update_task(task, user_id=user_id)
        return task

    def _apply_filters(
        self,
        tasks: list[SurfaceableTask],
        query: SurfacingQuery,
    ) -> list[SurfaceableTask]:
        """Apply pre-scoring filters to narrow the candidate set.

        These are hard filters that remove tasks before scoring.
        The scoring engine applies its own eligibility check (active only).
        """
        filtered = list(tasks)

        # Intent filter
        if query.include_intents:
            intents_lower = {i.lower() for i in query.include_intents}
            filtered = [
                t for t in filtered if t.intent.lower() in intents_lower
            ]

        # Entity exclusion filter
        if query.exclude_entity_ids:
            exclude_set = set(query.exclude_entity_ids)
            filtered = [
                t
                for t in filtered
                if not (set(t.entity_ids) & exclude_set)
            ]

        # Tag filter (any match) — tags stored in metadata
        if query.tags_filter:
            tags_lower = {tag.lower() for tag in query.tags_filter}
            filtered = [
                t
                for t in filtered
                if any(
                    tag.lower() in tags_lower
                    for tag in t.metadata.get("tags", [])
                )
                or not t.metadata.get("tags", [])  # tasks without tags pass through
            ]

        # Energy cap filter
        if query.max_energy is not None:
            energy_order = {
                EnergyLevel.LOW: 0,
                EnergyLevel.MEDIUM: 1,
                EnergyLevel.HIGH: 2,
            }
            max_level = energy_order[query.max_energy]
            filtered = [
                t
                for t in filtered
                if energy_order[t.estimated_energy] <= max_level
            ]

        return filtered
