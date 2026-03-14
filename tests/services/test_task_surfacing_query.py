"""Tests for the task surfacing query service.

Validates:
- Eligible task filtering (active only, respecting anti-shame design)
- Composite scoring via TaskScoringEngine
- Ranked results with score breakdowns
- Pre-scoring filters (intent, entity, tag, energy)
- Anti-shame: empty results are valid, no guilt language
- Multi-user isolation
- Surface count tracking
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


from blurt.services.task_surfacing import (
    EnergyLevel,
    SignalType,
    SurfaceableTask,
    SurfacingWeights,
    TaskStatus,
)
from blurt.services.task_surfacing_query import (
    InMemoryTaskStore,
    SurfacingQuery,
    SurfacingQueryResult,
    TaskSurfacingQueryService,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)


def _make_task(
    content: str = "test task",
    status: TaskStatus = TaskStatus.ACTIVE,
    intent: str = "task",
    energy: EnergyLevel = EnergyLevel.MEDIUM,
    entity_ids: list[str] | None = None,
    entity_names: list[str] | None = None,
    project: str | None = None,
    due_at: datetime | None = None,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
    capture_valence: float = 0.0,
    capture_arousal: float = 0.5,
    **kwargs,
) -> SurfaceableTask:
    task = SurfaceableTask(
        content=content,
        status=status,
        intent=intent,
        estimated_energy=energy,
        entity_ids=entity_ids or [],
        entity_names=entity_names or [],
        project=project,
        due_at=due_at,
        created_at=created_at or NOW,
        capture_valence=capture_valence,
        capture_arousal=capture_arousal,
        **kwargs,
    )
    if tags:
        task.metadata["tags"] = tags
    return task


def _make_service(tasks: list[SurfaceableTask] | None = None, user_id: str = "") -> TaskSurfacingQueryService:
    store = InMemoryTaskStore()
    for t in (tasks or []):
        store.add_task(t, user_id=user_id or None)
    return TaskSurfacingQueryService(store=store)


def _default_query(**kwargs) -> SurfacingQuery:
    defaults = {
        "now": NOW,
        "energy": EnergyLevel.MEDIUM,
        "mood_valence": 0.0,
        "mood_arousal": 0.5,
    }
    defaults.update(kwargs)
    return SurfacingQuery(**defaults)


# ---------------------------------------------------------------------------
# Basic query tests
# ---------------------------------------------------------------------------


class TestBasicQuery:
    """Test basic surfacing query functionality."""

    def test_empty_store_returns_empty_result(self):
        service = _make_service([])
        result = service.query(_default_query())

        assert not result.has_tasks
        assert result.returned_count == 0
        assert result.total_in_store == 0
        assert result.message != ""  # Anti-shame: always a positive message

    def test_single_task_returns_scored(self):
        task = _make_task(content="Buy groceries")
        service = _make_service([task])
        result = service.query(_default_query())

        assert result.has_tasks
        assert result.returned_count == 1
        assert result.tasks[0].task.content == "Buy groceries"
        assert 0.0 <= result.tasks[0].composite_score <= 1.0

    def test_score_breakdown_present(self):
        task = _make_task(content="Review PR")
        service = _make_service([task])
        result = service.query(_default_query())

        scored = result.tasks[0]
        breakdown = scored.signal_breakdown
        assert "time_relevance" in breakdown
        assert "energy_match" in breakdown
        assert "context_relevance" in breakdown
        assert "emotional_alignment" in breakdown
        assert "momentum" in breakdown
        assert "freshness" in breakdown

        # All values in [0, 1]
        for v in breakdown.values():
            assert 0.0 <= v <= 1.0

    def test_multiple_tasks_ranked_by_score(self):
        # Create tasks with different characteristics
        t1 = _make_task(
            content="Urgent meeting prep",
            due_at=NOW + timedelta(hours=1),
            energy=EnergyLevel.HIGH,
        )
        t2 = _make_task(
            content="Check email",
            energy=EnergyLevel.LOW,
        )
        t3 = _make_task(
            content="Plan project",
            due_at=NOW + timedelta(days=7),
            energy=EnergyLevel.HIGH,
        )
        service = _make_service([t1, t2, t3])
        result = service.query(_default_query())

        assert result.returned_count >= 2
        # Tasks should be in descending score order
        scores = [t.composite_score for t in result.tasks]
        assert scores == sorted(scores, reverse=True)

    def test_query_id_unique(self):
        service = _make_service([])
        r1 = service.query(_default_query())
        r2 = service.query(_default_query())
        assert r1.query_id != r2.query_id


# ---------------------------------------------------------------------------
# Eligibility filtering tests
# ---------------------------------------------------------------------------


class TestEligibilityFiltering:
    """Test that only eligible tasks are surfaced."""

    def test_only_active_tasks_surfaced(self):
        active = _make_task(content="Active task", status=TaskStatus.ACTIVE)
        completed = _make_task(content="Done", status=TaskStatus.COMPLETED)
        deferred = _make_task(content="Later", status=TaskStatus.DEFERRED)
        dropped = _make_task(content="Nah", status=TaskStatus.DROPPED)

        service = _make_service([active, completed, deferred, dropped])
        result = service.query(_default_query())

        task_contents = [t.task.content for t in result.tasks]
        assert "Active task" in task_contents
        assert "Done" not in task_contents
        assert "Later" not in task_contents
        assert "Nah" not in task_contents

    def test_completed_tasks_excluded(self):
        t = _make_task(status=TaskStatus.COMPLETED)
        service = _make_service([t])
        result = service.query(_default_query())
        assert not result.has_tasks

    def test_dropped_tasks_excluded_shame_free(self):
        """Dropped tasks are excluded silently — no judgment."""
        t = _make_task(status=TaskStatus.DROPPED)
        service = _make_service([t])
        result = service.query(_default_query())
        assert not result.has_tasks


# ---------------------------------------------------------------------------
# Pre-scoring filter tests
# ---------------------------------------------------------------------------


class TestPreScoringFilters:
    """Test pre-scoring filters (intent, entity, tag, energy)."""

    def test_intent_filter_includes_only_matching(self):
        task = _make_task(content="Buy milk", intent="task")
        event = _make_task(content="Meeting", intent="event")
        reminder = _make_task(content="Call mom", intent="reminder")

        service = _make_service([task, event, reminder])
        result = service.query(_default_query(include_intents=["task"]))

        contents = [t.task.content for t in result.tasks]
        assert "Buy milk" in contents
        assert "Meeting" not in contents
        assert "Call mom" not in contents

    def test_intent_filter_case_insensitive(self):
        task = _make_task(content="A task", intent="TASK")
        service = _make_service([task])
        result = service.query(_default_query(include_intents=["task"]))
        assert result.returned_count == 1

    def test_entity_exclusion_filter(self):
        t1 = _make_task(content="With entity", entity_ids=["e1"])
        t2 = _make_task(content="Without entity", entity_ids=["e2"])

        service = _make_service([t1, t2])
        result = service.query(_default_query(exclude_entity_ids=["e1"]))

        contents = [t.task.content for t in result.tasks]
        assert "Without entity" in contents
        assert "With entity" not in contents

    def test_tag_filter(self):
        t1 = _make_task(content="Tagged", tags=["work"])
        t2 = _make_task(content="Untagged")
        t3 = _make_task(content="Other tag", tags=["personal"])

        service = _make_service([t1, t2, t3])
        result = service.query(_default_query(tags_filter=["work"]))

        contents = [t.task.content for t in result.tasks]
        assert "Tagged" in contents
        assert "Untagged" in contents  # no tags pass through
        assert "Other tag" not in contents

    def test_energy_cap_filter(self):
        low = _make_task(content="Easy", energy=EnergyLevel.LOW)
        medium = _make_task(content="Medium", energy=EnergyLevel.MEDIUM)
        high = _make_task(content="Hard", energy=EnergyLevel.HIGH)

        service = _make_service([low, medium, high])
        result = service.query(_default_query(max_energy=EnergyLevel.MEDIUM))

        contents = [t.task.content for t in result.tasks]
        assert "Easy" in contents
        assert "Medium" in contents
        assert "Hard" not in contents

    def test_multiple_filters_combined(self):
        t1 = _make_task(content="Match", intent="task", energy=EnergyLevel.LOW)
        t2 = _make_task(content="Wrong intent", intent="event", energy=EnergyLevel.LOW)
        t3 = _make_task(content="Too hard", intent="task", energy=EnergyLevel.HIGH)

        service = _make_service([t1, t2, t3])
        result = service.query(
            _default_query(
                include_intents=["task"],
                max_energy=EnergyLevel.MEDIUM,
            )
        )

        contents = [t.task.content for t in result.tasks]
        assert "Match" in contents
        assert "Wrong intent" not in contents
        assert "Too hard" not in contents


# ---------------------------------------------------------------------------
# Scoring context tests
# ---------------------------------------------------------------------------


class TestScoringContext:
    """Test that user context affects scoring appropriately."""

    def test_energy_match_boosts_score(self):
        high_task = _make_task(content="Deep work", energy=EnergyLevel.HIGH)
        low_task = _make_task(content="Quick email", energy=EnergyLevel.LOW)

        service = _make_service([high_task, low_task])

        # High energy user should prefer high energy task
        high_result = service.query(_default_query(energy=EnergyLevel.HIGH))
        high_scores = {t.task.content: t.composite_score for t in high_result.tasks}

        # Low energy user should prefer low energy task
        low_result = service.query(_default_query(energy=EnergyLevel.LOW))
        low_scores = {t.task.content: t.composite_score for t in low_result.tasks}

        # High energy user scores deep work higher
        assert high_scores["Deep work"] > high_scores.get("Quick email", 0)
        # Low energy user scores quick email higher
        assert low_scores["Quick email"] > low_scores.get("Deep work", 0)

    def test_context_relevance_with_active_entities(self):
        relevant = _make_task(
            content="Related task",
            entity_names=["ProjectX"],
        )
        unrelated = _make_task(
            content="Unrelated task",
            entity_names=["ProjectY"],
        )

        service = _make_service([relevant, unrelated])
        result = service.query(
            _default_query(active_entity_names=["ProjectX"])
        )

        scores = {t.task.content: t.composite_score for t in result.tasks}
        assert scores["Related task"] > scores["Unrelated task"]

    def test_time_relevance_boosts_urgent_tasks(self):
        urgent = _make_task(
            content="Due soon",
            due_at=NOW + timedelta(hours=1),
        )
        far = _make_task(
            content="Due later",
            due_at=NOW + timedelta(days=14),
        )

        service = _make_service([urgent, far])
        result = service.query(_default_query())

        scores = {t.task.content: t.composite_score for t in result.tasks}
        assert scores["Due soon"] > scores["Due later"]

    def test_project_match_boosts_score(self):
        same_project = _make_task(
            content="Same project",
            project="Alpha",
            entity_names=["Widget"],
        )
        diff_project = _make_task(
            content="Diff project",
            project="Beta",
            entity_names=["Gadget"],
        )

        service = _make_service([same_project, diff_project])
        result = service.query(
            _default_query(
                active_project="Alpha",
                active_entity_names=["Widget"],
            )
        )

        scores = {t.task.content: t.composite_score for t in result.tasks}
        assert scores["Same project"] > scores["Diff project"]


# ---------------------------------------------------------------------------
# Anti-shame design tests
# ---------------------------------------------------------------------------


class TestAntiShameDesign:
    """Test anti-shame design principles."""

    def test_empty_result_has_positive_message(self):
        service = _make_service([])
        result = service.query(_default_query())

        assert result.message != ""
        # Should not contain guilt language
        guilt_words = ["overdue", "missed", "failed", "behind", "late", "forgotten"]
        for word in guilt_words:
            assert word.lower() not in result.message.lower()

    def test_no_overdue_penalty(self):
        """Past-due tasks should NOT score worse than tasks without due dates."""
        overdue = _make_task(
            content="Past due",
            due_at=NOW - timedelta(days=3),
        )
        no_due = _make_task(content="No deadline")

        service = _make_service([overdue, no_due])
        result = service.query(_default_query())

        overdue_score = next(
            t for t in result.tasks if t.task.content == "Past due"
        )
        # Past due should still have reasonable time_relevance (0.6 per engine)
        time_score = overdue_score.signal_breakdown.get("time_relevance", 0)
        assert time_score >= 0.5  # Not penalized

    def test_dropped_tasks_not_judged(self):
        """Dropped tasks just disappear. No shame."""
        active = _make_task(content="Active")
        dropped = _make_task(content="Dropped", status=TaskStatus.DROPPED)

        service = _make_service([active, dropped])
        result = service.query(_default_query())

        assert result.returned_count == 1
        assert result.tasks[0].task.content == "Active"

    def test_score_transparency(self):
        """Every score breakdown is fully visible — no hidden signals."""
        task = _make_task(content="Transparent task")
        service = _make_service([task])
        result = service.query(_default_query())

        scored = result.tasks[0]
        # Must have all 6 signal scores
        assert len(scored.signal_scores) == 6
        signals = {s.signal for s in scored.signal_scores}
        expected = {
            SignalType.TIME_RELEVANCE,
            SignalType.ENERGY_MATCH,
            SignalType.CONTEXT_RELEVANCE,
            SignalType.EMOTIONAL_ALIGNMENT,
            SignalType.MOMENTUM,
            SignalType.FRESHNESS,
        }
        assert signals == expected

        # Every signal has a reason
        for s in scored.signal_scores:
            assert s.reason != ""


# ---------------------------------------------------------------------------
# Max results and min score tests
# ---------------------------------------------------------------------------


class TestResultLimits:
    """Test max_results and min_score thresholds."""

    def test_max_results_limits_output(self):
        tasks = [_make_task(content=f"Task {i}") for i in range(10)]
        service = _make_service(tasks)
        result = service.query(_default_query(max_results=3))

        assert result.returned_count <= 3

    def test_min_score_filters_low_scorers(self):
        tasks = [_make_task(content=f"Task {i}") for i in range(5)]
        service = _make_service(tasks)

        # Very high min_score should filter most tasks
        result = service.query(_default_query(min_score=0.99))
        assert result.returned_count == 0

    def test_min_score_zero_returns_all_eligible(self):
        tasks = [_make_task(content=f"Task {i}") for i in range(3)]
        service = _make_service(tasks)
        result = service.query(_default_query(min_score=0.0))
        assert result.returned_count == 3


# ---------------------------------------------------------------------------
# Surface count tracking tests
# ---------------------------------------------------------------------------


class TestSurfaceCountTracking:
    """Test that surface counts are tracked."""

    def test_surface_count_increments(self):
        task = _make_task(content="Track me")
        service = _make_service([task])

        # Surface once
        result = service.query(_default_query())
        assert result.tasks[0].task.times_surfaced == 1

        # Surface again
        result = service.query(_default_query())
        assert result.tasks[0].task.times_surfaced == 2


# ---------------------------------------------------------------------------
# Multi-user isolation tests
# ---------------------------------------------------------------------------


class TestMultiUserIsolation:
    """Test that tasks are isolated per user."""

    def test_different_users_see_own_tasks(self):
        store = InMemoryTaskStore()
        t1 = _make_task(content="Alice task")
        t2 = _make_task(content="Bob task")
        store.add_task(t1, user_id="alice")
        store.add_task(t2, user_id="bob")

        service = TaskSurfacingQueryService(store=store)

        alice_result = service.query(_default_query(user_id="alice"))
        bob_result = service.query(_default_query(user_id="bob"))

        alice_contents = [t.task.content for t in alice_result.tasks]
        bob_contents = [t.task.content for t in bob_result.tasks]

        assert "Alice task" in alice_contents
        assert "Bob task" not in alice_contents
        assert "Bob task" in bob_contents
        assert "Alice task" not in bob_contents


# ---------------------------------------------------------------------------
# Weights override tests
# ---------------------------------------------------------------------------


class TestWeightsOverride:
    """Test custom weight overrides in queries."""

    def test_custom_weights_applied(self):
        task = _make_task(content="Test")
        service = _make_service([task])

        custom_weights = SurfacingWeights(
            time_relevance=0.5,
            energy_match=0.1,
            context_relevance=0.1,
            emotional_alignment=0.1,
            momentum=0.1,
            freshness=0.1,
        )
        result = service.query(_default_query(weights=custom_weights))

        assert result.weights_used is not None
        assert result.weights_used["time_relevance"] == 0.5

    def test_weights_reported_in_result(self):
        service = _make_service([_make_task()])
        result = service.query(_default_query())

        assert result.weights_used is not None
        assert "time_relevance" in result.weights_used
        assert "energy_match" in result.weights_used


# ---------------------------------------------------------------------------
# Task store tests
# ---------------------------------------------------------------------------


class TestInMemoryTaskStore:
    """Test the in-memory task store."""

    def test_add_and_get(self):
        store = InMemoryTaskStore()
        task = _make_task(content="Store me")
        store.add_task(task)

        retrieved = store.get_task(task.id)
        assert retrieved is not None
        assert retrieved.content == "Store me"

    def test_get_nonexistent_returns_none(self):
        store = InMemoryTaskStore()
        assert store.get_task("nonexistent") is None

    def test_update_task(self):
        store = InMemoryTaskStore()
        task = _make_task(content="Original")
        store.add_task(task)

        task.content = "Updated"
        store.update_task(task)

        retrieved = store.get_task(task.id)
        assert retrieved is not None
        assert retrieved.content == "Updated"

    def test_remove_task(self):
        store = InMemoryTaskStore()
        task = _make_task(content="Remove me")
        store.add_task(task)

        assert store.remove_task(task.id)
        assert store.get_task(task.id) is None

    def test_remove_nonexistent_returns_false(self):
        store = InMemoryTaskStore()
        assert not store.remove_task("nonexistent")

    def test_get_all_tasks(self):
        store = InMemoryTaskStore()
        t1 = _make_task(content="One")
        t2 = _make_task(content="Two")
        store.add_task(t1)
        store.add_task(t2)

        all_tasks = store.get_all_tasks()
        assert len(all_tasks) == 2

    def test_clear(self):
        store = InMemoryTaskStore()
        store.add_task(_make_task())
        store.add_task(_make_task())
        store.clear()
        assert len(store.get_all_tasks()) == 0


# ---------------------------------------------------------------------------
# Query result helper tests
# ---------------------------------------------------------------------------


class TestSurfacingQueryResult:
    """Test SurfacingQueryResult helpers."""

    def test_has_tasks_false_when_empty(self):
        result = SurfacingQueryResult()
        assert not result.has_tasks

    def test_top_task_none_when_empty(self):
        result = SurfacingQueryResult()
        assert result.top_task is None

    def test_score_breakdown_out_of_range(self):
        result = SurfacingQueryResult()
        assert result.score_breakdown(0) is None
        assert result.score_breakdown(-1) is None

    def test_context_snapshot_preserved(self):
        task = _make_task(content="Context check")
        service = _make_service([task])
        result = service.query(
            _default_query(
                energy=EnergyLevel.HIGH,
                mood_valence=0.8,
            )
        )

        assert result.context_snapshot is not None
        assert result.context_snapshot.energy == EnergyLevel.HIGH
        assert result.context_snapshot.current_valence == 0.8


# ---------------------------------------------------------------------------
# Service task management tests
# ---------------------------------------------------------------------------


class TestServiceTaskManagement:
    """Test add/get/update through the service layer."""

    def test_add_and_query(self):
        service = _make_service()
        task = _make_task(content="Added via service")
        service.add_task(task)

        result = service.query(_default_query())
        assert result.returned_count == 1
        assert result.tasks[0].task.content == "Added via service"

    def test_update_task_status(self):
        service = _make_service()
        task = _make_task(content="Complete me")
        service.add_task(task)

        updated = service.update_task_status(task.id, TaskStatus.COMPLETED)
        assert updated is not None
        assert updated.status == TaskStatus.COMPLETED

        # Completed task should not surface
        result = service.query(_default_query())
        assert result.returned_count == 0

    def test_update_nonexistent_task(self):
        service = _make_service()
        result = service.update_task_status("nonexistent", TaskStatus.COMPLETED)
        assert result is None


# ---------------------------------------------------------------------------
# Metadata reporting tests
# ---------------------------------------------------------------------------


class TestMetadataReporting:
    """Test that query result metadata is accurate."""

    def test_total_counts_accurate(self):
        tasks = [
            _make_task(content="Active 1"),
            _make_task(content="Active 2"),
            _make_task(content="Active 3"),
            _make_task(content="Completed", status=TaskStatus.COMPLETED),
        ]
        service = _make_service(tasks)
        result = service.query(_default_query(max_results=2))

        assert result.total_in_store == 4
        assert result.total_eligible == 3  # only active
        assert result.returned_count <= 2

    def test_filter_counts_accurate(self):
        tasks = [
            _make_task(content="Task 1", intent="task"),
            _make_task(content="Event 1", intent="event"),
            _make_task(content="Task 2", intent="task"),
        ]
        service = _make_service(tasks)
        result = service.query(_default_query(include_intents=["task"]))

        assert result.total_in_store == 3
        assert result.total_after_filters == 2  # only tasks
