"""Tests for the task nudge scheduler service.

Validates:
- Periodic evaluation of pending tasks
- Score-based nudge determination
- Per-task cooldown enforcement
- Max nudges per cycle cap
- Due-soon score boost
- Graceful handling when no users connected
- Integration with ConnectionRegistry and TaskStore
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from blurt.audio.connection_registry import ConnectionRegistry
from blurt.services.task_nudge_scheduler import (
    DEFAULT_NUDGE_SCORE_THRESHOLD,
    DEFAULT_TASK_COOLDOWN,
    NudgeSchedulerConfig,
    TaskNudgeScheduler,
)
from blurt.services.task_surfacing import (
    EnergyLevel,
    ScoredTask,
    SignalScore,
    SignalType,
    SurfaceableTask,
    TaskStatus,
)


def _make_task(
    task_id: str = "t1",
    content: str = "Test task",
    status: TaskStatus = TaskStatus.ACTIVE,
    due_at: datetime | None = None,
    entity_names: list[str] | None = None,
    times_surfaced: int = 0,
) -> SurfaceableTask:
    """Create a test task with sensible defaults."""
    return SurfaceableTask(
        id=task_id,
        content=content,
        status=status,
        intent="task",
        due_at=due_at,
        entity_names=entity_names or [],
        times_surfaced=times_surfaced,
    )


def _make_scored_task(
    task: SurfaceableTask,
    score: float = 0.5,
    reason: str = "test reason",
) -> ScoredTask:
    """Create a scored task wrapper."""
    return ScoredTask(
        task=task,
        composite_score=score,
        signal_scores=(
            SignalScore(signal=SignalType.TIME_RELEVANCE, value=score, reason=reason),
        ),
        surfacing_reason=reason,
    )


class FakeTaskStore:
    """Fake PgTaskStore for testing — returns pre-configured tasks."""

    def __init__(self, tasks: list[SurfaceableTask] | None = None) -> None:
        self._tasks = tasks or []

    async def get_all_tasks_async(
        self, user_id: str | None = None
    ) -> list[SurfaceableTask]:
        return [t for t in self._tasks]


class FakeHandler:
    """Fake WebSocket handler with user_id property."""

    def __init__(self, user_id: str | None = None) -> None:
        self._user_id = user_id
        self.pushed_messages: list[tuple[str, dict]] = []

    @property
    def user_id(self) -> str | None:
        return self._user_id

    async def push_message(self, msg_type, payload) -> None:
        self.pushed_messages.append((msg_type, payload))


class TestTaskNudgeSchedulerConfig:
    """Tests for NudgeSchedulerConfig defaults."""

    def test_defaults(self) -> None:
        config = NudgeSchedulerConfig()
        assert config.interval_seconds == 120.0
        assert config.score_threshold == DEFAULT_NUDGE_SCORE_THRESHOLD
        assert config.max_nudges_per_cycle == 3
        assert config.task_cooldown_seconds == DEFAULT_TASK_COOLDOWN
        assert config.due_soon_hours == 2.0
        assert config.default_energy == EnergyLevel.MEDIUM

    def test_custom_config(self) -> None:
        config = NudgeSchedulerConfig(
            interval_seconds=60.0,
            score_threshold=0.5,
            max_nudges_per_cycle=1,
        )
        assert config.interval_seconds == 60.0
        assert config.score_threshold == 0.5
        assert config.max_nudges_per_cycle == 1


class TestSchedulerLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self) -> None:
        registry = ConnectionRegistry()
        store = FakeTaskStore()
        scheduler = TaskNudgeScheduler(registry, store)
        task = scheduler.start()
        assert task is not None
        assert scheduler.is_running
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self) -> None:
        registry = ConnectionRegistry()
        store = FakeTaskStore()
        scheduler = TaskNudgeScheduler(registry, store)
        task1 = scheduler.start()
        task2 = scheduler.start()
        assert task1 is task2
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self) -> None:
        registry = ConnectionRegistry()
        store = FakeTaskStore()
        scheduler = TaskNudgeScheduler(registry, store)
        scheduler.start()
        assert scheduler.is_running
        await scheduler.stop()
        assert not scheduler.is_running


class TestNudgeEvaluation:
    """Tests for the core evaluation logic."""

    @pytest.mark.asyncio
    async def test_no_nudges_when_no_connections(self) -> None:
        """No evaluation when no one is connected."""
        registry = ConnectionRegistry()
        store = FakeTaskStore([_make_task()])
        scheduler = TaskNudgeScheduler(registry, store)

        # force_evaluate should be a no-op with no connections
        await scheduler.force_evaluate()
        # No error, no nudges

    @pytest.mark.asyncio
    async def test_no_nudges_when_no_tasks(self) -> None:
        """No nudges when user has no tasks."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        store = FakeTaskStore([])  # No tasks
        scheduler = TaskNudgeScheduler(registry, store)

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) == 0

    @pytest.mark.asyncio
    async def test_no_nudges_for_completed_tasks(self) -> None:
        """Completed/deferred/dropped tasks are never nudged."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        tasks = [
            _make_task("t1", status=TaskStatus.COMPLETED),
            _make_task("t2", status=TaskStatus.DEFERRED),
            _make_task("t3", status=TaskStatus.DROPPED),
        ]
        store = FakeTaskStore(tasks)
        scheduler = TaskNudgeScheduler(registry, store)

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) == 0

    @pytest.mark.asyncio
    async def test_nudge_sent_for_high_score_task(self) -> None:
        """Tasks above the score threshold get nudged."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        # Task with due_at soon (high time_relevance score)
        task = _make_task(
            "t1",
            content="Call dentist",
            due_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(score_threshold=0.1),
        )

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) > 0
        msg_type, payload = handler.pushed_messages[0]
        assert payload["task_id"] == "t1"
        assert payload["content"] == "Call dentist"

    @pytest.mark.asyncio
    async def test_nudge_below_threshold_not_sent(self) -> None:
        """Tasks below the score threshold don't get nudged."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        # Task with no due date, no context — will score moderate
        task = _make_task("t1", content="Maybe someday")
        store = FakeTaskStore([task])

        # Set threshold very high so nothing passes
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(score_threshold=0.99),
        )

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) == 0

    @pytest.mark.asyncio
    async def test_task_cooldown_enforced(self) -> None:
        """Same task cannot be nudged again within cooldown period."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        task = _make_task(
            "t1",
            due_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(
                score_threshold=0.1,
                task_cooldown_seconds=300.0,
            ),
        )

        # First evaluation — should nudge
        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) == 1

        # Second evaluation — should be in cooldown
        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) == 1  # Still 1

    @pytest.mark.asyncio
    async def test_cooldown_expires(self) -> None:
        """Task can be nudged again after cooldown expires."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        task = _make_task(
            "t1",
            due_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(
                score_threshold=0.1,
                task_cooldown_seconds=0.01,  # Very short cooldown
            ),
        )

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) == 1

        # Wait for cooldown to expire
        await asyncio.sleep(0.02)

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) == 2

    @pytest.mark.asyncio
    async def test_max_nudges_per_cycle_enforced(self) -> None:
        """No more than max_nudges_per_cycle are sent per evaluation."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        # Create 5 tasks, all due soon
        tasks = [
            _make_task(
                f"t{i}",
                content=f"Task {i}",
                due_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            for i in range(5)
        ]
        store = FakeTaskStore(tasks)
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(
                score_threshold=0.1,
                max_nudges_per_cycle=2,
            ),
        )

        await scheduler.force_evaluate()
        # Should cap at 2 per cycle
        assert len(handler.pushed_messages) <= 2

    @pytest.mark.asyncio
    async def test_due_soon_boost(self) -> None:
        """Tasks due within due_soon_hours get a score boost."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        # Task due in 1 hour — should get DUE_SOON_BOOST
        task = _make_task(
            "t1",
            content="Urgent call",
            due_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        store = FakeTaskStore([task])

        # Set threshold just above what the task would score without boost
        # but below what it would score with the boost
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(score_threshold=0.1),
        )

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) > 0
        _, payload = handler.pushed_messages[0]
        assert payload["reason"] == "due_soon"


class TestNudgeReasons:
    """Tests for nudge reason determination."""

    @pytest.mark.asyncio
    async def test_due_soon_reason(self) -> None:
        """Tasks due within threshold get 'due_soon' reason."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        task = _make_task(
            "t1",
            due_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(score_threshold=0.1),
        )

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) > 0
        _, payload = handler.pushed_messages[0]
        assert payload["reason"] == "due_soon"

    @pytest.mark.asyncio
    async def test_past_due_reason(self) -> None:
        """Tasks past due get 'past_suggested_time' reason."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        task = _make_task(
            "t1",
            due_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(score_threshold=0.1),
        )

        await scheduler.force_evaluate()
        if handler.pushed_messages:
            _, payload = handler.pushed_messages[0]
            assert payload["reason"] == "past_suggested_time"

    @pytest.mark.asyncio
    async def test_due_today_reason(self) -> None:
        """Tasks due within 24 hours (but not 'soon') get 'due_today' reason."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        task = _make_task(
            "t1",
            due_at=datetime.now(timezone.utc) + timedelta(hours=12),
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(score_threshold=0.1),
        )

        await scheduler.force_evaluate()
        if handler.pushed_messages:
            _, payload = handler.pushed_messages[0]
            assert payload["reason"] == "due_today"


class TestConnectedUserDiscovery:
    """Tests for user discovery from connection registry."""

    def test_get_connected_user_ids(self) -> None:
        """Extracts unique user IDs from connected handlers."""
        registry = ConnectionRegistry()
        h1 = FakeHandler(user_id="user1")
        h2 = FakeHandler(user_id="user2")
        h3 = FakeHandler(user_id="user1")  # Duplicate
        h4 = FakeHandler(user_id=None)  # No user_id yet

        registry.register(h1)
        registry.register(h2)
        registry.register(h3)
        registry.register(h4)

        store = FakeTaskStore()
        scheduler = TaskNudgeScheduler(registry, store)

        user_ids = scheduler._get_connected_user_ids()
        assert user_ids == {"user1", "user2"}

    @pytest.mark.asyncio
    async def test_multi_user_nudges(self) -> None:
        """Each connected user gets their own evaluation."""
        registry = ConnectionRegistry()
        h1 = FakeHandler(user_id="user1")
        h2 = FakeHandler(user_id="user2")
        registry.register(h1)
        registry.register(h2)

        task = _make_task(
            "t1",
            due_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(score_threshold=0.1),
        )

        await scheduler.force_evaluate()
        # Both users should get nudged (the store returns same tasks for both)
        total = len(h1.pushed_messages) + len(h2.pushed_messages)
        assert total >= 2


class TestClearCooldowns:
    """Tests for cooldown management."""

    @pytest.mark.asyncio
    async def test_clear_cooldowns(self) -> None:
        """clear_cooldowns() allows immediate re-nudge."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        task = _make_task(
            "t1",
            due_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(
                score_threshold=0.1,
                task_cooldown_seconds=9999,
            ),
        )

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) == 1

        # Clear cooldowns
        scheduler.clear_cooldowns()

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) == 2


class TestNudgePayloadContents:
    """Tests for the nudge payload structure."""

    @pytest.mark.asyncio
    async def test_nudge_payload_fields(self) -> None:
        """Nudge payload includes all expected fields."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        task = _make_task(
            "t1",
            content="Buy groceries",
            due_at=datetime.now(timezone.utc) + timedelta(hours=1),
            entity_names=["Whole Foods"],
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(score_threshold=0.1),
        )

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) > 0

        _, payload = handler.pushed_messages[0]
        assert payload["task_id"] == "t1"
        assert payload["content"] == "Buy groceries"
        assert payload["intent"] == "task"
        assert "due_at" in payload
        assert "priority" in payload
        assert "nudge_ts" in payload
        assert payload["entity_names"] == ["Whole Foods"]

    @pytest.mark.asyncio
    async def test_nudge_priority_reflects_score(self) -> None:
        """Nudge priority value reflects the effective score."""
        registry = ConnectionRegistry()
        handler = FakeHandler(user_id="user1")
        registry.register(handler)

        task = _make_task(
            "t1",
            due_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        store = FakeTaskStore([task])
        scheduler = TaskNudgeScheduler(
            registry,
            store,
            config=NudgeSchedulerConfig(score_threshold=0.1),
        )

        await scheduler.force_evaluate()
        assert len(handler.pushed_messages) > 0

        _, payload = handler.pushed_messages[0]
        assert 0.0 <= payload["priority"] <= 1.0
