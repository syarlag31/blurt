"""Task nudge scheduler — periodically evaluates pending tasks and pushes nudges.

Runs as a background asyncio task during the app lifespan. On each tick:
1. Identifies connected users via the ConnectionRegistry
2. Queries the PgTaskStore for active tasks per user
3. Scores tasks using the TaskScoringEngine against a default context
4. Determines which tasks warrant a nudge (score threshold, cooldown, due_soon)
5. Sends nudges via ConnectionRegistry.send_task_nudge()

Anti-shame design:
- Nudge frequency is capped (default: max 3 nudges per cycle, 5-minute cooldown per task)
- Tasks that have been deferred/dismissed are never nudged
- Low-scoring tasks are not nudged — only genuinely relevant ones
- "No nudges" is the default healthy state
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from blurt.audio.connection_registry import ConnectionRegistry, TaskNudgePayload
from blurt.services.task_surfacing import (
    EnergyLevel,
    ScoredTask,
    SurfaceableTask,
    TaskScoringEngine,
    TaskStatus,
    UserContext,
)


class TaskStoreProtocol(Protocol):
    """Protocol for task stores used by the nudge scheduler.

    Both ``PgTaskStore`` and test fakes satisfy this interface.
    """

    async def get_all_tasks_async(
        self, user_id: str | None = None
    ) -> list[SurfaceableTask]:
        """Retrieve all tasks, optionally filtered by user_id."""
        ...

logger = logging.getLogger(__name__)


# Default nudge interval (seconds between evaluation cycles)
DEFAULT_NUDGE_INTERVAL = 120.0  # 2 minutes

# Minimum score threshold for a task to be nudge-worthy
DEFAULT_NUDGE_SCORE_THRESHOLD = 0.35

# Maximum nudges per user per cycle
DEFAULT_MAX_NUDGES_PER_CYCLE = 3

# Minimum seconds between nudges for the same task
DEFAULT_TASK_COOLDOWN = 300.0  # 5 minutes

# Score boost for tasks due within 2 hours
DUE_SOON_BOOST = 0.15


@dataclass
class NudgeSchedulerConfig:
    """Configuration for the task nudge scheduler.

    All values have sensible defaults and can be overridden
    without touching the scheduler logic.
    """

    interval_seconds: float = DEFAULT_NUDGE_INTERVAL
    score_threshold: float = DEFAULT_NUDGE_SCORE_THRESHOLD
    max_nudges_per_cycle: int = DEFAULT_MAX_NUDGES_PER_CYCLE
    task_cooldown_seconds: float = DEFAULT_TASK_COOLDOWN
    due_soon_hours: float = 2.0
    default_energy: EnergyLevel = EnergyLevel.MEDIUM


class TaskNudgeScheduler:
    """Background scheduler that evaluates tasks and pushes nudges.

    Integrates with:
    - ConnectionRegistry: knows who is connected
    - PgTaskStore: retrieves pending tasks from Postgres
    - TaskScoringEngine: scores tasks against a default user context

    Lifecycle: start() → runs in background → stop()
    """

    def __init__(
        self,
        registry: ConnectionRegistry,
        task_store: TaskStoreProtocol,
        config: NudgeSchedulerConfig | None = None,
        scoring_engine: TaskScoringEngine | None = None,
    ) -> None:
        self._registry = registry
        self._task_store = task_store
        self._config = config or NudgeSchedulerConfig()
        self._engine = scoring_engine or TaskScoringEngine()
        self._task: asyncio.Task[None] | None = None
        self._running = False

        # Track last nudge time per (user_id, task_id) to enforce cooldowns
        self._last_nudge_times: dict[tuple[str, str], float] = {}

    @property
    def is_running(self) -> bool:
        """Whether the scheduler background task is active."""
        return self._running and self._task is not None and not self._task.done()

    def start(self) -> asyncio.Task[None]:
        """Start the background nudge evaluation loop.

        Returns the asyncio.Task for the caller to manage (e.g. cancel on shutdown).
        """
        if self._task is not None and not self._task.done():
            logger.warning("Nudge scheduler already running")
            return self._task

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Task nudge scheduler started (interval=%.0fs, threshold=%.2f)",
            self._config.interval_seconds,
            self._config.score_threshold,
        )
        return self._task

    async def stop(self) -> None:
        """Stop the background nudge evaluation loop gracefully."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Task nudge scheduler stopped")

    async def _loop(self) -> None:
        """Main evaluation loop — runs periodically until stopped."""
        try:
            while self._running:
                try:
                    await self._evaluate_and_nudge()
                except Exception:
                    logger.exception("Error in nudge evaluation cycle")

                await asyncio.sleep(self._config.interval_seconds)
        except asyncio.CancelledError:
            logger.debug("Nudge scheduler loop cancelled")

    async def _evaluate_and_nudge(self) -> None:
        """Single evaluation cycle: score tasks, send nudges to connected users."""
        if self._registry.connection_count == 0:
            # No one connected — skip evaluation entirely
            return

        # Collect unique user IDs from connected handlers
        connected_users = self._get_connected_user_ids()
        if not connected_users:
            return

        now = time.time()
        total_nudges_sent = 0

        for user_id in connected_users:
            try:
                nudges_sent = await self._evaluate_user_tasks(user_id, now)
                total_nudges_sent += nudges_sent
            except Exception:
                logger.exception("Error evaluating tasks for user=%s", user_id)

        if total_nudges_sent > 0:
            logger.info(
                "Nudge cycle complete: %d nudges sent to %d users",
                total_nudges_sent,
                len(connected_users),
            )

    async def _evaluate_user_tasks(self, user_id: str, now: float) -> int:
        """Evaluate and nudge tasks for a single user.

        Returns the number of nudges sent.
        """
        # Fetch all active tasks from Postgres
        tasks = await self._task_store.get_all_tasks_async(user_id=user_id)

        # Filter to active tasks only
        active_tasks = [t for t in tasks if t.status == TaskStatus.ACTIVE]
        if not active_tasks:
            return 0

        # Build a default user context for scoring
        context = UserContext(
            energy=self._config.default_energy,
            now=datetime.now(timezone.utc),
        )

        # Score all tasks
        scored_tasks = self._score_tasks(active_tasks, context)

        # Filter by threshold and cooldown, then send nudges
        nudges_sent = 0
        for scored in scored_tasks:
            if nudges_sent >= self._config.max_nudges_per_cycle:
                break

            task = scored.task
            task_id = task.id

            # Check cooldown (per user + task)
            cooldown_key = (user_id, task_id)
            last_nudge = self._last_nudge_times.get(cooldown_key, 0.0)
            if (now - last_nudge) < self._config.task_cooldown_seconds:
                continue

            # Check score threshold
            effective_score = scored.composite_score

            # Boost score for tasks due soon
            if task.due_at is not None:
                hours_until = (task.due_at - context.now).total_seconds() / 3600.0
                if 0 < hours_until <= self._config.due_soon_hours:
                    effective_score = min(1.0, effective_score + DUE_SOON_BOOST)

            if effective_score < self._config.score_threshold:
                continue

            # Determine nudge reason
            reason = self._determine_reason(scored, task, context)

            # Build and send the nudge
            nudge = TaskNudgePayload(
                task_id=task_id,
                content=task.content,
                intent=task.intent,
                due_at=task.due_at.isoformat() if task.due_at else None,
                reason=reason,
                priority=effective_score,
                entity_names=task.entity_names,
                surface_count=task.times_surfaced,
            )

            sent = await self._registry.send_task_nudge(user_id, nudge)
            if sent > 0:
                self._last_nudge_times[cooldown_key] = now
                nudges_sent += sent

        return nudges_sent

    def _score_tasks(
        self,
        tasks: list[SurfaceableTask],
        context: UserContext,
    ) -> list[ScoredTask]:
        """Score and rank tasks using the TaskScoringEngine.

        Returns tasks sorted by composite score descending.
        """
        result = self._engine.score_and_rank(tasks, context)
        return result.tasks

    def _determine_reason(
        self,
        scored: ScoredTask,
        task: SurfaceableTask,
        context: UserContext,
    ) -> str:
        """Determine the human-readable reason for nudging this task."""
        if task.due_at is not None:
            hours_until = (task.due_at - context.now).total_seconds() / 3600.0
            if hours_until < 0:
                return "past_suggested_time"
            elif hours_until <= self._config.due_soon_hours:
                return "due_soon"
            elif hours_until <= 24:
                return "due_today"

        # Use the scoring engine's reason
        if scored.surfacing_reason:
            return scored.surfacing_reason

        return "scheduled"

    def _get_connected_user_ids(self) -> set[str]:
        """Extract unique user IDs from the connection registry."""
        user_ids: set[str] = set()
        for handler in list(self._registry._handlers):
            uid = handler.user_id
            if uid is not None:
                user_ids.add(uid)
        return user_ids

    def clear_cooldowns(self) -> None:
        """Clear all task cooldown timers (useful for testing)."""
        self._last_nudge_times.clear()

    async def force_evaluate(self) -> None:
        """Force an immediate evaluation cycle (useful for testing/debugging)."""
        await self._evaluate_and_nudge()
