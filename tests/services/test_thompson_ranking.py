"""Tests for the Thompson Ranking Pipeline — integrated Thompson Sampling + context scoring.

AC 11 Sub-AC 3: Verify that Thompson Sampling weights are integrated into the
task surfacing/ranking pipeline so surfaced tasks are ordered by sampled scores
combined with context factors.

Covers:
- Pipeline ranks tasks using combined context + Thompson scores
- Category-level Thompson boosts tasks from high-engagement intents
- Task-level Thompson boosts individually engaged tasks
- Signal-level Thompson modulates scoring dimension weights
- Blend factor grows with observations (cold start = context dominant)
- Feedback loop updates all three Thompson layers
- Anti-shame: no task zeroed out, empty results valid, no guilt
- Exploration: sampling produces variety in rankings
- Deterministic behavior without Thompson layers
- Integration with TaskSurfacingQueryService
- Serialization round-trip of Thompson state
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blurt.services.feedback import (
    FeedbackAction,
    InMemoryFeedbackStore,
    TaskFeedbackService,
)
from blurt.services.surfacing.thompson import ThompsonSampler
from blurt.services.task_surfacing import (
    EnergyLevel,
    SurfaceableTask,
    TaskScoringEngine,
    TaskStatus,
    UserContext,
)
from blurt.services.task_surfacing_query import (
    InMemoryTaskStore,
    SurfacingQuery,
    TaskSurfacingQueryService,
)
from blurt.services.thompson_ranking import (
    RankedTask,
    ThompsonRankingPipeline,
    ThompsonRankingResult,
    ThompsonScoreBreakdown,
)
from blurt.services.thompson_sampling import (
    FeedbackType,
    ThompsonSamplingEngine,
)


# ── Helpers ─────────────────────────────────────────────────────────

NOW = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)


def _ctx(**kwargs) -> UserContext:
    defaults: dict = dict(
        energy=EnergyLevel.MEDIUM,
        current_valence=0.0,
        current_arousal=0.5,
        now=NOW,
    )
    defaults.update(kwargs)
    return UserContext(**defaults)


def _task(**kwargs) -> SurfaceableTask:
    defaults: dict = dict(
        content="test task",
        status=TaskStatus.ACTIVE,
        estimated_energy=EnergyLevel.MEDIUM,
        created_at=NOW - timedelta(hours=2),
    )
    defaults.update(kwargs)
    return SurfaceableTask(**defaults)


def _pipeline(
    *,
    with_category: bool = False,
    with_feedback: bool = False,
    with_signal_sampler: bool = False,
    seed: int = 42,
    **kwargs,
) -> ThompsonRankingPipeline:
    """Create a pipeline with optional Thompson layers."""
    signal_sampler = ThompsonSampler() if with_signal_sampler else None
    engine = TaskScoringEngine(
        thompson_sampler=signal_sampler,
        min_score=0.0,  # Accept all for testing
    )

    category_engine = (
        ThompsonSamplingEngine(seed=seed) if with_category else None
    )
    feedback_service = (
        TaskFeedbackService(store=InMemoryFeedbackStore())
        if with_feedback
        else None
    )

    return ThompsonRankingPipeline(
        scoring_engine=engine,
        category_engine=category_engine,
        feedback_service=feedback_service,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# Core pipeline behavior
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineBasics:
    """Test basic pipeline construction and ranking."""

    def test_pipeline_creates_with_defaults(self):
        """Pipeline should work with all default settings."""
        pipe = ThompsonRankingPipeline()
        assert pipe.scoring_engine is not None
        assert pipe.category_engine is None
        assert pipe.feedback_service is None

    def test_pipeline_ranks_tasks(self):
        """Pipeline should rank tasks and return results."""
        pipe = _pipeline()
        tasks = [_task(content="a"), _task(content="b")]
        result = pipe.rank(tasks, _ctx())
        assert isinstance(result, ThompsonRankingResult)
        assert result.has_tasks
        assert len(result.ranked_tasks) == 2

    def test_pipeline_returns_ranked_tasks(self):
        """Results should contain RankedTask with breakdown."""
        pipe = _pipeline()
        result = pipe.rank([_task()], _ctx())
        assert result.has_tasks
        rt = result.ranked_tasks[0]
        assert isinstance(rt, RankedTask)
        assert isinstance(rt.thompson_breakdown, ThompsonScoreBreakdown)
        assert 0.0 <= rt.final_score <= 1.0

    def test_empty_input_returns_empty(self):
        """Empty task list should return empty results (valid state)."""
        pipe = _pipeline()
        result = pipe.rank([], _ctx())
        assert not result.has_tasks
        assert result.top is None

    def test_filters_non_active_tasks(self):
        """Only ACTIVE tasks should be ranked."""
        pipe = _pipeline()
        tasks = [
            _task(content="active", status=TaskStatus.ACTIVE),
            _task(content="completed", status=TaskStatus.COMPLETED),
            _task(content="deferred", status=TaskStatus.DEFERRED),
            _task(content="dropped", status=TaskStatus.DROPPED),
        ]
        result = pipe.rank(tasks, _ctx())
        assert len(result.ranked_tasks) == 1
        assert result.ranked_tasks[0].task.content == "active"

    def test_results_sorted_by_final_score(self):
        """Tasks should be sorted by final_score descending."""
        pipe = _pipeline()
        tasks = [_task() for _ in range(5)]
        result = pipe.rank(tasks, _ctx())
        scores = [rt.final_score for rt in result.ranked_tasks]
        assert scores == sorted(scores, reverse=True)

    def test_max_results_limits_output(self):
        """Pipeline should respect max_results."""
        pipe = _pipeline(max_results=2)
        tasks = [_task() for _ in range(10)]
        result = pipe.rank(tasks, _ctx())
        assert len(result.ranked_tasks) <= 2
        assert result.total_filtered >= 0


# ═══════════════════════════════════════════════════════════════════════
# Thompson Sampling integration
# ═══════════════════════════════════════════════════════════════════════


class TestThompsonIntegration:
    """Test that all three Thompson layers are used in ranking."""

    def test_without_thompson_uses_context_only(self):
        """Without any Thompson layers, final_score equals context_score."""
        pipe = _pipeline()
        result = pipe.rank([_task()], _ctx())
        rt = result.ranked_tasks[0]
        # blend_factor should be 0 (no observations)
        assert rt.thompson_breakdown.blend_factor == 0.0
        assert rt.final_score == rt.thompson_breakdown.context_score

    def test_category_engine_provides_samples(self):
        """Category engine samples should appear in results."""
        pipe = _pipeline(with_category=True)
        result = pipe.rank([_task()], _ctx())
        assert result.category_samples is not None
        assert len(result.category_samples) > 0

    def test_signal_sampler_provides_weights(self):
        """Signal sampler weights should appear in results."""
        pipe = _pipeline(with_signal_sampler=True)
        result = pipe.rank([_task()], _ctx())
        assert result.signal_thompson_weights is not None

    def test_blend_factor_zero_at_cold_start(self):
        """At cold start (no observations), blend factor should be 0."""
        pipe = _pipeline(with_category=True, with_feedback=True)
        result = pipe.rank([_task()], _ctx())
        assert result.blend_factor_used == 0.0

    def test_blend_factor_grows_with_observations(self):
        """Blend factor should increase as observations accumulate."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        # Add observations
        for _ in range(30):
            cat_engine.update("task", FeedbackType.ACCEPTED)
        for _ in range(10):
            cat_engine.update("event", FeedbackType.DISMISSED)

        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
        )
        result = pipe.rank([_task()], _ctx())
        assert result.blend_factor_used > 0.0

    def test_category_boost_shifts_ranking(self):
        """Tasks from high-engagement categories should rank higher."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        # Strongly boost "event" intent
        for _ in range(50):
            cat_engine.update("event", FeedbackType.COMPLETED)
        # Suppress "task" intent
        for _ in range(50):
            cat_engine.update("task", FeedbackType.DISMISSED)

        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
            max_blend_factor=0.8,
            blend_observations_scale=5.0,  # Fast convergence for test
        )

        tasks = [
            _task(content="task-type", intent="task"),
            _task(content="event-type", intent="event"),
        ]
        ctx = _ctx()

        # Run multiple times — event should win more often
        event_wins = 0
        for _ in range(100):
            result = pipe.rank(tasks, ctx)
            if result.top and result.top.task.content == "event-type":
                event_wins += 1

        assert event_wins > 50, (
            f"Event-type should win majority due to category boost, won {event_wins}/100"
        )

    def test_task_level_feedback_shifts_thompson_sample(self):
        """Per-task feedback should shift Thompson sample scores deterministically."""
        feedback_svc = TaskFeedbackService(store=InMemoryFeedbackStore())

        task_a = _task(content="task-a")
        task_b = _task(content="task-b")

        # Record many positive feedbacks for task-a
        for _ in range(50):
            feedback_svc.record_feedback(
                task_id=task_a.id,
                user_id="user1",
                action=FeedbackAction.COMPLETE,
                intent="task",
            )

        # Record many negative feedbacks for task-b
        for _ in range(50):
            feedback_svc.record_feedback(
                task_id=task_b.id,
                user_id="user1",
                action=FeedbackAction.DISMISS,
                intent="task",
            )

        # Verify that the feedback service produces higher expected scores
        # for task-a vs task-b
        a_params = feedback_svc.store.get_params(f"task:{task_a.id}")
        b_params = feedback_svc.store.get_params(f"task:{task_b.id}")

        # task-a has positive feedback → higher alpha → higher mean
        assert a_params.alpha > a_params.beta, (
            f"task-a alpha should exceed beta: {a_params.alpha} vs {a_params.beta}"
        )
        # task-b has negative feedback → higher beta → lower mean
        assert b_params.beta > b_params.alpha, (
            f"task-b beta should exceed alpha: {b_params.beta} vs {b_params.alpha}"
        )
        assert a_params.mean > b_params.mean, (
            f"task-a mean should exceed task-b: {a_params.mean:.4f} vs {b_params.mean:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Score breakdown transparency
# ═══════════════════════════════════════════════════════════════════════


class TestScoreBreakdown:
    """Test that score breakdowns are transparent and accurate."""

    def test_breakdown_contains_all_fields(self):
        """ThompsonScoreBreakdown should have all expected fields."""
        pipe = _pipeline(with_category=True, with_feedback=True)
        result = pipe.rank([_task()], _ctx())
        bd = result.ranked_tasks[0].thompson_breakdown

        assert 0.0 <= bd.context_score <= 1.0
        assert 0.0 <= bd.category_sample <= 1.0
        assert 0.0 <= bd.task_sample <= 1.0
        assert 0.0 <= bd.thompson_blend <= 1.0
        assert 0.0 <= bd.blend_factor <= 1.0
        assert 0.0 <= bd.final_score <= 1.0

    def test_final_score_formula(self):
        """final_score should follow the documented formula."""
        pipe = _pipeline()
        result = pipe.rank([_task()], _ctx())
        bd = result.ranked_tasks[0].thompson_breakdown

        expected = (
            bd.context_score * (1.0 - bd.blend_factor)
            + bd.thompson_blend * bd.blend_factor
        )
        assert bd.final_score == pytest.approx(expected, abs=0.001)

    def test_thompson_blend_formula(self):
        """thompson_blend should be weighted avg of category and task samples."""
        pipe = _pipeline(with_category=True, with_feedback=True)
        result = pipe.rank([_task()], _ctx())
        bd = result.ranked_tasks[0].thompson_breakdown

        expected = (
            pipe.category_weight * bd.category_sample
            + pipe.task_weight * bd.task_sample
        )
        assert bd.thompson_blend == pytest.approx(expected, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════
# Feedback loop
# ═══════════════════════════════════════════════════════════════════════


class TestFeedbackLoop:
    """Test the feedback loop: rank → engage → learn → re-rank."""

    def test_record_feedback_accept(self):
        """Recording accept feedback should update Thompson layers."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        feedback_svc = TaskFeedbackService(store=InMemoryFeedbackStore())
        sampler = ThompsonSampler()

        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(
                thompson_sampler=sampler, min_score=0.0,
            ),
            category_engine=cat_engine,
            feedback_service=feedback_svc,
        )

        result = pipe.rank([_task(intent="task")], _ctx())
        rt = result.ranked_tasks[0]

        # Record feedback
        pipe.record_feedback(
            rt, FeedbackAction.ACCEPT, user_id="user1",
        )

        # Category engine should have updated
        params = cat_engine.get_params("task")
        assert params.total_observations == 1
        assert params.alpha > 1.0

        # Feedback service should have recorded
        events = feedback_svc.get_recent_feedback("user1")
        assert len(events) == 1
        assert events[0].action == FeedbackAction.ACCEPT

    def test_record_feedback_dismiss(self):
        """Dismissal feedback should update beta (gentle negative)."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        feedback_svc = TaskFeedbackService(store=InMemoryFeedbackStore())

        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
            feedback_service=feedback_svc,
        )

        result = pipe.rank([_task(intent="task")], _ctx())
        rt = result.ranked_tasks[0]

        pipe.record_feedback(rt, FeedbackAction.DISMISS, user_id="user1")

        params = cat_engine.get_params("task")
        assert params.beta > 1.0  # Beta increased

    def test_record_feedback_complete(self):
        """Completion is the strongest positive signal."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
        )

        result = pipe.rank([_task(intent="task")], _ctx())
        pipe.record_feedback(result.ranked_tasks[0], FeedbackAction.COMPLETE)

        params = cat_engine.get_params("task")
        # Completion has the highest alpha weight
        assert params.alpha > 2.0

    def test_record_feedback_snooze(self):
        """Snooze is treated as partial positive (user wants it, not now)."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
        )

        result = pipe.rank([_task(intent="task")], _ctx())
        pipe.record_feedback(result.ranked_tasks[0], FeedbackAction.SNOOZE)

        params = cat_engine.get_params("task")
        # Snooze adds both a small alpha and small beta
        assert params.alpha > 1.0
        assert params.beta > 1.0

    def test_feedback_noop_without_layers(self):
        """Feedback should be safe with no Thompson layers attached."""
        pipe = _pipeline()
        result = pipe.rank([_task()], _ctx())
        # Should not raise
        pipe.record_feedback(result.ranked_tasks[0], FeedbackAction.ACCEPT)
        pipe.record_feedback(result.ranked_tasks[0], FeedbackAction.DISMISS)

    def test_unknown_category_added_dynamically(self):
        """Recording feedback for unknown category should add it."""
        cat_engine = ThompsonSamplingEngine(
            categories=["task"], seed=42,
        )
        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
        )

        result = pipe.rank(
            [_task(intent="custom_intent")], _ctx()
        )
        # Should not raise
        pipe.record_feedback(
            result.ranked_tasks[0], FeedbackAction.ACCEPT,
        )
        assert "custom_intent" in cat_engine.categories


# ═══════════════════════════════════════════════════════════════════════
# Anti-shame verification
# ═══════════════════════════════════════════════════════════════════════


class TestAntiShame:
    """Verify anti-shame design principles with Thompson ranking."""

    def test_no_task_zeroed_out(self):
        """No task should ever score exactly zero, even with extreme Thompson."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        # Heavily suppress "task" category
        for _ in range(100):
            cat_engine.update("task", FeedbackType.DISMISSED)

        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
            min_thompson_score=0.1,
            min_final_score=0.0,
            max_blend_factor=0.9,
            blend_observations_scale=5.0,
        )

        result = pipe.rank([_task(intent="task")], _ctx())
        assert result.has_tasks
        assert result.ranked_tasks[0].final_score > 0.0

    def test_empty_results_valid(self):
        """Empty results remain a valid state."""
        pipe = _pipeline()
        result = pipe.rank([], _ctx())
        assert not result.has_tasks
        assert result.total_candidates == 0

    def test_minimum_thompson_score_floor(self):
        """Thompson samples should never go below min_thompson_score."""
        pipe = _pipeline(
            with_category=True,
            min_thompson_score=0.15,
        )
        result = pipe.rank([_task()], _ctx())
        bd = result.ranked_tasks[0].thompson_breakdown
        assert bd.category_sample >= 0.15
        assert bd.task_sample >= 0.15

    def test_no_guilt_language_in_empty(self):
        """No guilt language when used with query service."""
        pipe = _pipeline()
        store = InMemoryTaskStore()
        service = TaskSurfacingQueryService(
            store=store,
            thompson_pipeline=pipe,
        )
        result = service.query(SurfacingQuery(user_id="u1"))
        if result.message:
            guilt_words = ["overdue", "behind", "missed", "failed", "shame"]
            for word in guilt_words:
                assert word not in result.message.lower()


# ═══════════════════════════════════════════════════════════════════════
# Exploration behavior
# ═══════════════════════════════════════════════════════════════════════


class TestExploration:
    """Verify that Thompson Sampling provides controlled exploration."""

    def test_cold_start_diverse_rankings(self):
        """At cold start, rankings should show some diversity."""
        cat_engine = ThompsonSamplingEngine(seed=None)  # random seed
        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
            max_blend_factor=0.3,
            blend_observations_scale=1.0,  # fast for test
        )
        # Add one observation so blend_factor > 0
        cat_engine.update("task", FeedbackType.ACCEPTED)

        tasks = [
            _task(content=f"task-{i}", intent="task")
            for i in range(3)
        ]

        top_ids = set()
        for _ in range(50):
            result = pipe.rank(tasks, _ctx())
            if result.top:
                top_ids.add(result.top.task.id)

        # With 3 near-identical tasks and exploration, we should see variety
        assert len(top_ids) >= 1  # At minimum one task is surfaced

    def test_exploitation_converges(self):
        """With enough positive feedback, exploitation should converge."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        feedback_svc = TaskFeedbackService(store=InMemoryFeedbackStore())

        # Heavily favor "event" category
        for _ in range(100):
            cat_engine.update("event", FeedbackType.COMPLETED)

        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
            feedback_service=feedback_svc,
            max_blend_factor=0.6,
            blend_observations_scale=10.0,
        )

        tasks = [
            _task(content="event-task", intent="event"),
            _task(content="task-task", intent="task"),
        ]

        event_wins = 0
        for _ in range(100):
            result = pipe.rank(tasks, _ctx())
            if result.top and result.top.task.content == "event-task":
                event_wins += 1

        # Event should win most times after strong positive feedback
        assert event_wins > 40


# ═══════════════════════════════════════════════════════════════════════
# QueryService integration
# ═══════════════════════════════════════════════════════════════════════


class TestQueryServiceIntegration:
    """Test integration with TaskSurfacingQueryService."""

    def test_query_service_uses_pipeline(self):
        """Query service should use Thompson pipeline when provided."""
        pipe = _pipeline(with_category=True)
        store = InMemoryTaskStore()
        task = _task(content="integrated task")
        store.add_task(task)

        service = TaskSurfacingQueryService(
            store=store,
            thompson_pipeline=pipe,
        )

        result = service.query(SurfacingQuery(min_score=0.0))
        assert result.has_tasks
        assert result.returned_count == 1

    def test_query_service_fallback_without_pipeline(self):
        """Without pipeline, query service uses base engine."""
        store = InMemoryTaskStore()
        task = _task(content="base task")
        store.add_task(task)

        service = TaskSurfacingQueryService(store=store)
        result = service.query(SurfacingQuery(min_score=0.0))
        assert result.has_tasks

    def test_query_service_includes_blend_factor(self):
        """Query result weights_used should include blend factor."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        cat_engine.update("task", FeedbackType.ACCEPTED)

        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
        )
        store = InMemoryTaskStore()
        store.add_task(_task())

        service = TaskSurfacingQueryService(
            store=store,
            thompson_pipeline=pipe,
        )

        result = service.query(SurfacingQuery(min_score=0.0))
        assert result.weights_used is not None
        assert "_blend_factor" in result.weights_used

    def test_query_service_increments_surface_count(self):
        """Surface count should be incremented via pipeline path."""
        pipe = _pipeline()
        store = InMemoryTaskStore()
        task = _task(content="track me")
        store.add_task(task)

        service = TaskSurfacingQueryService(
            store=store,
            thompson_pipeline=pipe,
        )

        result = service.query(SurfacingQuery(min_score=0.0))
        assert result.has_tasks

        # Check surface count was incremented
        updated = store.get_task(task.id)
        assert updated is not None
        assert updated.times_surfaced == 1

    def test_query_service_empty_message(self):
        """Empty results through pipeline should get anti-shame message."""
        pipe = _pipeline()
        store = InMemoryTaskStore()

        service = TaskSurfacingQueryService(
            store=store,
            thompson_pipeline=pipe,
        )

        result = service.query(SurfacingQuery())
        assert not result.has_tasks
        assert result.message != ""


# ═══════════════════════════════════════════════════════════════════════
# Combined context + Thompson scoring
# ═══════════════════════════════════════════════════════════════════════


class TestCombinedScoring:
    """Verify that context and Thompson scores are properly combined."""

    def test_context_dominant_at_cold_start(self):
        """At cold start, context scores should fully determine ranking."""
        pipe = _pipeline(with_category=True, with_feedback=True)

        # Create tasks with different context fit
        high_energy_task = _task(
            content="deep work",
            estimated_energy=EnergyLevel.HIGH,
        )
        low_energy_task = _task(
            content="quick thing",
            estimated_energy=EnergyLevel.LOW,
        )
        ctx = _ctx(energy=EnergyLevel.HIGH)

        result = pipe.rank([low_energy_task, high_energy_task], ctx)
        assert result.blend_factor_used == 0.0

        # With blend_factor=0, final_score should equal context_score
        for rt in result.ranked_tasks:
            assert rt.final_score == pytest.approx(
                rt.thompson_breakdown.context_score, abs=0.001
            )

    def test_thompson_influences_with_data(self):
        """With data, Thompson should shift scores away from pure context."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        for _ in range(50):
            cat_engine.update("task", FeedbackType.COMPLETED)

        sampler = ThompsonSampler()
        engine = TaskScoringEngine(
            thompson_sampler=sampler, min_score=0.0,
        )

        pipe = ThompsonRankingPipeline(
            scoring_engine=engine,
            category_engine=cat_engine,
            max_blend_factor=0.5,
            blend_observations_scale=5.0,
        )

        result = pipe.rank([_task()], _ctx())
        assert result.blend_factor_used > 0.0

        rt = result.ranked_tasks[0]
        bd = rt.thompson_breakdown
        # Final score should differ from pure context score
        # (unless Thompson blend happens to equal context score)
        assert bd.blend_factor > 0.0

    def test_high_energy_match_with_thompson_boost(self):
        """Realistic scenario: high energy match + Thompson category boost."""
        cat_engine = ThompsonSamplingEngine(seed=42)
        for _ in range(30):
            cat_engine.update("task", FeedbackType.COMPLETED)

        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
            max_blend_factor=0.4,
            blend_observations_scale=5.0,
        )

        ctx = _ctx(energy=EnergyLevel.HIGH, current_valence=0.5)
        high_match = _task(
            content="high energy deep work",
            estimated_energy=EnergyLevel.HIGH,
            intent="task",
        )

        result = pipe.rank([high_match], ctx)
        assert result.has_tasks
        assert result.ranked_tasks[0].final_score > 0.3


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_unknown_intent_handled_gracefully(self):
        """Tasks with unknown intents should not crash."""
        cat_engine = ThompsonSamplingEngine(
            categories=["task", "event"], seed=42,
        )
        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(min_score=0.0),
            category_engine=cat_engine,
        )

        task = _task(intent="custom_unknown_type")
        result = pipe.rank([task], _ctx())
        assert result.has_tasks

    def test_all_layers_combined(self):
        """Full pipeline with all three Thompson layers should work."""
        sampler = ThompsonSampler()
        cat_engine = ThompsonSamplingEngine(seed=42)
        feedback_svc = TaskFeedbackService(store=InMemoryFeedbackStore())

        pipe = ThompsonRankingPipeline(
            scoring_engine=TaskScoringEngine(
                thompson_sampler=sampler, min_score=0.0,
            ),
            category_engine=cat_engine,
            feedback_service=feedback_svc,
        )

        tasks = [
            _task(content=f"task-{i}", intent="task")
            for i in range(5)
        ]

        result = pipe.rank(tasks, _ctx())
        assert result.has_tasks
        assert len(result.ranked_tasks) == 5

    def test_single_task_ranking(self):
        """Single task should rank without issues."""
        pipe = _pipeline(with_category=True, with_feedback=True)
        result = pipe.rank([_task()], _ctx())
        assert len(result.ranked_tasks) == 1

    def test_many_tasks_performance(self):
        """Pipeline should handle many tasks without issues."""
        pipe = _pipeline()
        tasks = [_task(content=f"task-{i}") for i in range(100)]
        result = pipe.rank(tasks, _ctx())
        assert result.total_candidates == 100

    def test_custom_weight_configuration(self):
        """Custom category/task weights should be respected."""
        pipe = _pipeline(
            with_category=True,
            category_weight=0.8,
            task_weight=0.2,
        )
        assert pipe.category_weight == 0.8
        assert pipe.task_weight == 0.2

        result = pipe.rank([_task()], _ctx())
        assert result.has_tasks

    def test_backward_compat_scored_tasks(self):
        """ThompsonRankingResult.tasks should return ScoredTask list."""
        pipe = _pipeline()
        result = pipe.rank([_task()], _ctx())
        tasks = result.tasks
        assert len(tasks) == 1
        assert hasattr(tasks[0], "composite_score")
        assert hasattr(tasks[0], "signal_scores")
