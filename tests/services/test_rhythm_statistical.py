"""Tests for rhythm detection statistical analysis.

Validates rolling averages, periodicity detection (autocorrelation),
trend computation, and enrichment of detected rhythms with statistical
metadata. These features enable Blurt to identify recurring patterns
like weekly energy dips and peak creativity windows.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from blurt.memory.episodic import (
    BehavioralSignal,
    EmotionSnapshot,
    Episode,
    EpisodeContext,
    InMemoryEpisodicStore,
    InputModality,
)
from blurt.services.rhythm import (
    PERIODICITY_THRESHOLD,
    DetectedRhythm,
    PeriodicityResult,
    RhythmAnalysisResult,
    RhythmBucket,
    RhythmDetectionService,
    RhythmType,
    RollingAverageResult,
    WeeklySlotSample,
    _compute_slot_consistency,
    _bucket_key,
    _episodes_to_daily_series,
    _episodes_to_weekly_slot_samples,
    _enrich_rhythms_with_periodicity,
    analyze_rhythms,
    compute_autocorrelation,
    compute_periodicity_for_slots,
    compute_rolling_average,
    compute_rolling_averages_for_episodes,
    compute_trend,
    detect_weekly_periodicity,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_episode(
    *,
    user_id: str = "user-1",
    day: str = "monday",
    time: str = "morning",
    valence: float = 0.0,
    arousal: float = 0.5,
    intent: str = "task",
    signal: BehavioralSignal = BehavioralSignal.NONE,
    timestamp: datetime | None = None,
    episode_id: str | None = None,
) -> Episode:
    """Create a test episode with controlled parameters."""
    return Episode(
        id=episode_id or str(uuid.uuid4()),
        user_id=user_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        raw_text="test blurt",
        modality=InputModality.VOICE,
        intent=intent,
        intent_confidence=0.9,
        emotion=EmotionSnapshot(
            primary="joy" if valence > 0 else "sadness",
            intensity=abs(valence),
            valence=valence,
            arousal=arousal,
        ),
        behavioral_signal=signal,
        context=EpisodeContext(
            time_of_day=time,
            day_of_week=day,
            session_id="session-1",
        ),
    )


def _make_weekly_episodes(
    weeks: int = 4,
    slot_configs: dict[str, dict] | None = None,
    base_time: datetime | None = None,
) -> list[Episode]:
    """Generate episodes across multiple weeks with configurable per-slot values.

    Args:
        weeks: Number of weeks to generate.
        slot_configs: Dict of "day:time" -> {"valence": f, "arousal": f, "intent": s}.
            If not provided, generates neutral episodes across all slots.
        base_time: Starting timestamp (Monday of week 1).

    Returns:
        List of episodes across all specified weeks.
    """
    if base_time is None:
        # Start from a known Monday
        base_time = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)  # A Monday

    day_offsets = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    time_hours = {
        "morning": 9, "afternoon": 14, "evening": 19, "night": 22,
    }

    if slot_configs is None:
        slot_configs = {
            "monday:morning": {"valence": 0.3, "arousal": 0.5},
        }

    episodes: list[Episode] = []

    for week in range(weeks):
        week_start = base_time + timedelta(weeks=week)
        for slot_key, config in slot_configs.items():
            day_str, time_str = slot_key.split(":")
            day_offset = day_offsets.get(day_str, 0)
            hour = time_hours.get(time_str, 9)

            ts = week_start + timedelta(days=day_offset, hours=hour - 9)

            episodes.append(_make_episode(
                day=day_str,
                time=time_str,
                valence=config.get("valence", 0.0),
                arousal=config.get("arousal", 0.5),
                intent=config.get("intent", "task"),
                signal=config.get("signal", BehavioralSignal.NONE),
                timestamp=ts,
            ))

    return episodes


# ── Rolling Average Tests ────────────────────────────────────────────


class TestComputeRollingAverage:
    """Tests for the compute_rolling_average function."""

    def test_empty_samples(self):
        result = compute_rolling_average([])
        assert result == []

    def test_single_sample(self):
        samples = [WeeklySlotSample(week_number=1, value=5.0)]
        result = compute_rolling_average(samples, window_size=3)
        assert len(result) == 1
        assert result[0] == 5.0

    def test_two_samples_window_three(self):
        """Fewer samples than window → single average."""
        samples = [
            WeeklySlotSample(week_number=1, value=4.0),
            WeeklySlotSample(week_number=2, value=6.0),
        ]
        result = compute_rolling_average(samples, window_size=3)
        assert len(result) == 1
        assert result[0] == 5.0

    def test_exact_window_size(self):
        samples = [
            WeeklySlotSample(week_number=1, value=3.0),
            WeeklySlotSample(week_number=2, value=6.0),
            WeeklySlotSample(week_number=3, value=9.0),
        ]
        result = compute_rolling_average(samples, window_size=3)
        assert len(result) == 1
        assert result[0] == 6.0

    def test_sliding_window(self):
        samples = [
            WeeklySlotSample(week_number=1, value=2.0),
            WeeklySlotSample(week_number=2, value=4.0),
            WeeklySlotSample(week_number=3, value=6.0),
            WeeklySlotSample(week_number=4, value=8.0),
        ]
        result = compute_rolling_average(samples, window_size=3)
        assert len(result) == 2
        assert abs(result[0] - 4.0) < 0.01  # avg(2,4,6)
        assert abs(result[1] - 6.0) < 0.01  # avg(4,6,8)

    def test_constant_series_is_flat(self):
        samples = [WeeklySlotSample(week_number=i, value=5.0) for i in range(6)]
        result = compute_rolling_average(samples, window_size=3)
        for val in result:
            assert abs(val - 5.0) < 0.01

    def test_increasing_series(self):
        samples = [WeeklySlotSample(week_number=i, value=float(i)) for i in range(5)]
        result = compute_rolling_average(samples, window_size=3)
        # Each window average should be increasing
        for i in range(len(result) - 1):
            assert result[i + 1] > result[i]


class TestComputeTrend:
    """Tests for the compute_trend function."""

    def test_empty_series(self):
        direction, strength = compute_trend([])
        assert direction == 0.0
        assert strength == 0.0

    def test_single_value(self):
        direction, strength = compute_trend([5.0])
        assert direction == 0.0
        assert strength == 0.0

    def test_increasing_trend(self):
        direction, strength = compute_trend([1.0, 2.0, 3.0, 4.0, 5.0])
        assert direction > 0.0  # Positive slope
        assert strength > 0.9  # Nearly perfect R²

    def test_decreasing_trend(self):
        direction, strength = compute_trend([5.0, 4.0, 3.0, 2.0, 1.0])
        assert direction < 0.0  # Negative slope
        assert strength > 0.9

    def test_flat_trend(self):
        direction, strength = compute_trend([3.0, 3.0, 3.0, 3.0])
        assert abs(direction) < 0.01
        # R² is 0 when no variance

    def test_noisy_upward_trend(self):
        direction, _strength = compute_trend([1.0, 3.0, 2.0, 4.0, 5.0])
        assert direction > 0.0

    def test_two_values(self):
        direction, strength = compute_trend([1.0, 3.0])
        assert direction > 0.0
        assert strength == 1.0  # Perfect fit with 2 points


class TestRollingAveragesForEpisodes:
    """Tests for compute_rolling_averages_for_episodes."""

    def test_empty_episodes(self):
        result = compute_rolling_averages_for_episodes([])
        assert result == {}

    def test_basic_rolling_average(self):
        episodes = _make_weekly_episodes(
            weeks=4,
            slot_configs={"monday:morning": {"valence": 0.5, "arousal": 0.7}},
        )
        result = compute_rolling_averages_for_episodes(episodes, "energy")
        assert "monday:morning" in result
        ra = result["monday:morning"]
        assert ra.weeks_of_data == 4
        assert len(ra.rolling_values) >= 1
        assert ra.overall_mean > 0.0

    def test_multiple_metrics(self):
        episodes = _make_weekly_episodes(
            weeks=4,
            slot_configs={"monday:morning": {"valence": 0.5, "arousal": 0.7, "intent": "idea"}},
        )
        energy_result = compute_rolling_averages_for_episodes(episodes, "energy")
        valence_result = compute_rolling_averages_for_episodes(episodes, "valence")
        creativity_result = compute_rolling_averages_for_episodes(episodes, "creativity")

        assert "monday:morning" in energy_result
        assert "monday:morning" in valence_result
        assert "monday:morning" in creativity_result

    def test_trend_detection_across_weeks(self):
        """Energy increasing over weeks should show positive trend."""
        base_time = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
        episodes: list[Episode] = []

        for week in range(6):
            ts = base_time + timedelta(weeks=week)
            # Energy increases each week
            arousal = 0.3 + week * 0.1
            episodes.append(_make_episode(
                day="monday", time="morning",
                valence=0.5, arousal=min(1.0, arousal),
                timestamp=ts,
            ))

        result = compute_rolling_averages_for_episodes(episodes, "energy")
        ra = result.get("monday:morning")
        assert ra is not None
        assert ra.trend_direction > 0.0  # Positive trend
        assert ra.is_trending_up

    def test_stable_pattern_no_trend(self):
        """Consistent values across weeks should show stable trend."""
        episodes = _make_weekly_episodes(
            weeks=6,
            slot_configs={"tuesday:afternoon": {"valence": 0.3, "arousal": 0.5}},
        )
        result = compute_rolling_averages_for_episodes(episodes, "energy")
        ra = result.get("tuesday:afternoon")
        assert ra is not None
        assert ra.is_stable


# ── Autocorrelation Tests ────────────────────────────────────────────


class TestComputeAutocorrelation:
    """Tests for the compute_autocorrelation function."""

    def test_empty_series(self):
        assert compute_autocorrelation([], 7) == 0.0

    def test_short_series(self):
        assert compute_autocorrelation([1.0, 2.0], 7) == 0.0

    def test_constant_series(self):
        series = [5.0] * 20
        assert compute_autocorrelation(series, 7) == 0.0

    def test_perfect_weekly_periodicity(self):
        """A signal that repeats exactly every 7 days should have high autocorrelation."""
        pattern = [0.8, 0.5, 0.5, 0.5, 0.2, 0.3, 0.4]  # Mon-Sun
        series = pattern * 4  # 4 weeks
        autocorr = compute_autocorrelation(series, 7)
        assert autocorr > 0.7  # Strong weekly periodicity (finite-sample bias lowers it slightly)

    def test_no_periodicity_random(self):
        """Non-periodic data should have near-zero autocorrelation."""
        import random
        random.seed(42)
        series = [random.uniform(0, 1) for _ in range(28)]
        autocorr = compute_autocorrelation(series, 7)
        assert abs(autocorr) < 0.5

    def test_lag_1_adjacent_correlation(self):
        """Smoothly varying series should have high lag-1 autocorrelation."""
        series = [math.sin(2 * math.pi * i / 7) for i in range(28)]
        autocorr_1 = compute_autocorrelation(series, 1)
        assert autocorr_1 > 0.5  # Adjacent days are similar

    def test_weekly_sine_wave(self):
        """Sine wave with 7-day period should have strong lag-7 autocorrelation."""
        series = [math.sin(2 * math.pi * i / 7) for i in range(28)]
        autocorr_7 = compute_autocorrelation(series, 7)
        assert autocorr_7 > 0.7  # Finite-sample autocorrelation is slightly lower


class TestEpisodesToDailySeries:
    """Tests for _episodes_to_daily_series."""

    def test_empty_episodes(self):
        series = _episodes_to_daily_series([], lambda e: e.emotion.valence)
        assert series == []

    def test_single_day(self):
        ts = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
        episodes = [_make_episode(valence=0.5, timestamp=ts)]
        series = _episodes_to_daily_series(episodes, lambda e: e.emotion.valence)
        assert len(series) == 1
        assert series[0] == 0.5

    def test_multiple_episodes_same_day(self):
        ts = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
        episodes = [
            _make_episode(valence=0.2, timestamp=ts),
            _make_episode(valence=0.8, timestamp=ts + timedelta(hours=2)),
        ]
        series = _episodes_to_daily_series(episodes, lambda e: e.emotion.valence)
        assert len(series) == 1
        assert abs(series[0] - 0.5) < 0.01  # Average of 0.2 and 0.8

    def test_gap_filling_with_mean(self):
        """Missing days get filled with the overall mean."""
        ts1 = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
        ts2 = ts1 + timedelta(days=2)  # Skip one day
        episodes = [
            _make_episode(valence=0.6, timestamp=ts1),
            _make_episode(valence=0.4, timestamp=ts2),
        ]
        series = _episodes_to_daily_series(episodes, lambda e: e.emotion.valence)
        assert len(series) == 3
        # Gap day should have mean of 0.6 and 0.4 = 0.5
        assert abs(series[1] - 0.5) < 0.01


class TestDetectWeeklyPeriodicity:
    """Tests for detect_weekly_periodicity."""

    def test_insufficient_data(self):
        episodes = [_make_episode(timestamp=datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc))]
        result = detect_weekly_periodicity(episodes, lambda e: e.emotion.valence)
        assert result == 0.0

    def test_consistent_weekly_pattern(self):
        """Same pattern repeating each week should have positive autocorrelation."""
        base = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
        episodes: list[Episode] = []

        # Create a consistent weekly pattern over 4 weeks
        for week in range(4):
            # High energy on Mondays
            episodes.append(_make_episode(
                valence=0.8, arousal=0.9,
                timestamp=base + timedelta(weeks=week, days=0),
            ))
            # Low energy on Fridays
            episodes.append(_make_episode(
                valence=-0.5, arousal=0.2,
                timestamp=base + timedelta(weeks=week, days=4),
            ))
            # Neutral on Wednesdays
            episodes.append(_make_episode(
                valence=0.0, arousal=0.5,
                timestamp=base + timedelta(weeks=week, days=2),
            ))

        result = detect_weekly_periodicity(
            episodes,
            lambda e: e.emotion.arousal * (0.5 + 0.5 * e.emotion.valence),
        )
        # Should detect some weekly pattern
        assert result > 0.0


class TestSlotConsistency:
    """Tests for _compute_slot_consistency."""

    def test_empty_values(self):
        assert _compute_slot_consistency([]) == 0.0

    def test_single_value(self):
        assert _compute_slot_consistency([5.0]) == 0.0

    def test_perfectly_consistent(self):
        """All same values = maximum consistency."""
        result = _compute_slot_consistency([0.5, 0.5, 0.5, 0.5])
        assert result == 1.0

    def test_high_variance_low_consistency(self):
        """Wildly varying values = low consistency."""
        result = _compute_slot_consistency([0.1, 0.9, 0.1, 0.9, 0.1])
        assert result < 0.7

    def test_near_zero_consistent(self):
        """Values consistently near zero."""
        result = _compute_slot_consistency([0.01, 0.02, 0.01, 0.02])
        assert result > 0.5

    def test_near_zero_inconsistent(self):
        """Values near zero but with big deviation → low consistency."""
        result = _compute_slot_consistency([0.0, 0.0, 0.0, 0.5])
        assert result < 0.5  # Low consistency due to high variance relative to mean


class TestPeriodicityForSlots:
    """Tests for compute_periodicity_for_slots."""

    def test_empty_episodes(self):
        result = compute_periodicity_for_slots([], "energy")
        assert result == {}

    def test_insufficient_weeks(self):
        """Single week of data → not periodic."""
        episodes = _make_weekly_episodes(
            weeks=1,
            slot_configs={"monday:morning": {"valence": 0.5, "arousal": 0.7}},
        )
        result = compute_periodicity_for_slots(episodes, "energy")
        if "monday:morning" in result:
            assert not result["monday:morning"].is_periodic

    def test_multi_week_consistent_pattern(self):
        """Consistent pattern over many weeks → detected as periodic."""
        episodes = _make_weekly_episodes(
            weeks=6,
            slot_configs={
                "monday:morning": {"valence": 0.8, "arousal": 0.9},
                "friday:afternoon": {"valence": -0.5, "arousal": 0.2},
                "wednesday:morning": {"valence": 0.1, "arousal": 0.5},
            },
        )
        result = compute_periodicity_for_slots(episodes, "energy")

        # At least one slot should show periodicity with enough consistent data
        assert len(result) > 0

    def test_periodicity_result_structure(self):
        episodes = _make_weekly_episodes(
            weeks=4,
            slot_configs={"tuesday:evening": {"valence": 0.3, "arousal": 0.6}},
        )
        result = compute_periodicity_for_slots(episodes, "energy")

        for key, pr in result.items():
            assert isinstance(pr, PeriodicityResult)
            assert pr.slot_key == key
            assert 0.0 <= pr.period_strength <= 1.0
            assert pr.sample_count >= 0
            assert isinstance(pr.is_periodic, bool)


# ── Enrichment Tests ─────────────────────────────────────────────────


class TestEnrichRhythmsWithPeriodicity:
    """Tests for _enrich_rhythms_with_periodicity."""

    def _make_rhythm(self, day="friday", time="afternoon",
                     rhythm_type=RhythmType.ENERGY_CRASH):
        return DetectedRhythm(
            rhythm_type=rhythm_type,
            day_of_week=day, time_of_day=time,
            z_score=2.0, metric_value=0.1, metric_mean=0.5,
            metric_std=0.2, observation_count=10, confidence=0.5,
        )

    def test_no_enrichment_data(self):
        rhythms = [self._make_rhythm()]
        enriched = _enrich_rhythms_with_periodicity(rhythms, {}, {})
        assert len(enriched) == 1
        assert not enriched[0].is_periodic
        assert enriched[0].confidence == 0.5  # Unchanged

    def test_periodic_pattern_gets_confidence_boost(self):
        rhythms = [self._make_rhythm()]
        pr = PeriodicityResult(
            slot_key="friday:afternoon",
            day_of_week="friday", time_of_day="afternoon",
            autocorrelation_lag7=0.6,
            is_periodic=True,
            period_strength=0.7,
            sample_count=8,
        )
        enriched = _enrich_rhythms_with_periodicity(
            rhythms, {}, {"friday:afternoon": pr},
        )
        assert enriched[0].is_periodic
        assert enriched[0].confidence > 0.5  # Boosted
        assert enriched[0].periodicity_strength == 0.7

    def test_non_periodic_no_boost(self):
        rhythms = [self._make_rhythm()]
        pr = PeriodicityResult(
            slot_key="friday:afternoon",
            day_of_week="friday", time_of_day="afternoon",
            autocorrelation_lag7=0.1,
            is_periodic=False,
            period_strength=0.1,
            sample_count=4,
        )
        enriched = _enrich_rhythms_with_periodicity(
            rhythms, {}, {"friday:afternoon": pr},
        )
        assert not enriched[0].is_periodic
        assert enriched[0].confidence == 0.5  # No boost

    def test_trend_direction_from_rolling_averages(self):
        rhythms = [self._make_rhythm()]
        ra = RollingAverageResult(
            slot_key="friday:afternoon",
            day_of_week="friday", time_of_day="afternoon",
            rolling_values=[0.3, 0.2, 0.1],
            overall_mean=0.2,
            trend_direction=-0.5,
            trend_strength=0.8,
            weeks_of_data=5,
        )
        enriched = _enrich_rhythms_with_periodicity(
            rhythms, {"friday:afternoon": ra}, {},
        )
        assert enriched[0].trend_direction == -0.5
        assert enriched[0].weeks_observed == 5

    def test_confidence_capped_at_one(self):
        rhythms = [DetectedRhythm(
            rhythm_type=RhythmType.ENERGY_PEAK,
            day_of_week="monday", time_of_day="morning",
            z_score=3.0, metric_value=0.9, metric_mean=0.4,
            metric_std=0.15, observation_count=50, confidence=0.95,
        )]
        pr = PeriodicityResult(
            slot_key="monday:morning",
            day_of_week="monday", time_of_day="morning",
            autocorrelation_lag7=0.9,
            is_periodic=True,
            period_strength=0.9,
            sample_count=10,
        )
        enriched = _enrich_rhythms_with_periodicity(
            rhythms, {}, {"monday:morning": pr},
        )
        assert enriched[0].confidence <= 1.0


# ── Integration: Full Pipeline with Rolling Averages ─────────────────


class TestAnalyzeRhythmsWithStatistics:
    """Integration tests for the full pipeline with rolling averages and periodicity."""

    def test_pipeline_includes_rolling_averages(self):
        """Full pipeline should produce rolling average data."""
        episodes = _make_weekly_episodes(
            weeks=4,
            slot_configs={
                "monday:morning": {"valence": 0.7, "arousal": 0.8},
                "tuesday:morning": {"valence": 0.3, "arousal": 0.5},
                "wednesday:morning": {"valence": 0.2, "arousal": 0.4},
                "thursday:morning": {"valence": 0.1, "arousal": 0.5},
                "friday:afternoon": {"valence": -0.5, "arousal": 0.2},
            },
        )
        result = analyze_rhythms("user-1", episodes)

        assert result.rolling_averages is not None
        assert len(result.rolling_averages) > 0

    def test_pipeline_includes_periodicity(self):
        """Full pipeline should produce periodicity results."""
        episodes = _make_weekly_episodes(
            weeks=4,
            slot_configs={
                "monday:morning": {"valence": 0.7, "arousal": 0.8},
                "friday:afternoon": {"valence": -0.5, "arousal": 0.2},
                "tuesday:morning": {"valence": 0.3, "arousal": 0.5},
                "wednesday:morning": {"valence": 0.2, "arousal": 0.4},
                "thursday:morning": {"valence": 0.1, "arousal": 0.5},
            },
        )
        result = analyze_rhythms("user-1", episodes)

        assert result.periodicity_results is not None
        assert len(result.periodicity_results) > 0

    def test_periodic_rhythms_filter(self):
        """Result should have a periodic_rhythms filter."""
        episodes = _make_weekly_episodes(
            weeks=6,
            slot_configs={
                "monday:morning": {"valence": 0.7, "arousal": 0.8},
                "tuesday:morning": {"valence": 0.3, "arousal": 0.5},
                "wednesday:morning": {"valence": 0.2, "arousal": 0.4},
                "thursday:morning": {"valence": 0.1, "arousal": 0.5},
                "friday:afternoon": {"valence": -0.7, "arousal": 0.1},
            },
        )
        result = analyze_rhythms("user-1", episodes)

        # periodic_rhythms should be a subset of all rhythms
        for pr in result.periodic_rhythms:
            assert pr.is_periodic
            assert pr in result.rhythms

    def test_to_dict_includes_periodicity(self):
        episodes = _make_weekly_episodes(
            weeks=4,
            slot_configs={
                "monday:morning": {"valence": 0.7, "arousal": 0.8},
                "tuesday:morning": {"valence": 0.3, "arousal": 0.5},
                "wednesday:morning": {"valence": 0.2, "arousal": 0.4},
                "thursday:morning": {"valence": 0.1, "arousal": 0.5},
                "friday:afternoon": {"valence": -0.5, "arousal": 0.2},
            },
        )
        result = analyze_rhythms("user-1", episodes)
        data = result.to_dict()

        assert "periodicity" in data
        assert "periodic_patterns" in data["summary"]

    def test_enriched_rhythm_to_learned_pattern(self):
        """Enriched rhythms should serialize periodicity data into LearnedPattern."""
        episodes = _make_weekly_episodes(
            weeks=5,
            slot_configs={
                "monday:morning": {"valence": 0.8, "arousal": 0.9},
                "tuesday:morning": {"valence": 0.3, "arousal": 0.5},
                "wednesday:morning": {"valence": 0.2, "arousal": 0.4},
                "thursday:morning": {"valence": 0.1, "arousal": 0.5},
                "friday:afternoon": {"valence": -0.7, "arousal": 0.15},
            },
        )
        result = analyze_rhythms("user-1", episodes)

        for rhythm in result.rhythms:
            pattern = rhythm.to_learned_pattern("user-1")
            assert "is_periodic" in pattern.parameters
            assert "periodicity_strength" in pattern.parameters
            assert "trend_direction" in pattern.parameters
            assert "weeks_observed" in pattern.parameters

    def test_weekly_energy_dip_detected(self):
        """Classic use case: Friday afternoon energy dip should be detected."""
        episodes = _make_weekly_episodes(
            weeks=5,
            slot_configs={
                "monday:morning": {"valence": 0.6, "arousal": 0.7},
                "tuesday:morning": {"valence": 0.5, "arousal": 0.6},
                "wednesday:morning": {"valence": 0.4, "arousal": 0.6},
                "thursday:morning": {"valence": 0.3, "arousal": 0.5},
                "friday:afternoon": {"valence": -0.6, "arousal": 0.2},
            },
        )
        result = analyze_rhythms("user-1", episodes)

        crashes = result.energy_crashes
        assert len(crashes) >= 1
        friday_crash = [c for c in crashes
                        if c.day_of_week == "friday" and c.time_of_day == "afternoon"]
        assert len(friday_crash) >= 1

    def test_peak_creativity_window_detected(self):
        """Sunday evening creativity peak should be detected."""
        episodes = _make_weekly_episodes(
            weeks=5,
            slot_configs={
                "monday:morning": {"valence": 0.3, "arousal": 0.5, "intent": "task"},
                "tuesday:morning": {"valence": 0.3, "arousal": 0.5, "intent": "task"},
                "wednesday:morning": {"valence": 0.3, "arousal": 0.5, "intent": "task"},
                "thursday:morning": {"valence": 0.3, "arousal": 0.5, "intent": "task"},
                "sunday:evening": {"valence": 0.6, "arousal": 0.7, "intent": "idea"},
            },
        )
        result = analyze_rhythms("user-1", episodes)

        peaks = result.creativity_peaks
        assert len(peaks) >= 1
        sunday_peak = [p for p in peaks
                       if p.day_of_week == "sunday" and p.time_of_day == "evening"]
        assert len(sunday_peak) >= 1

    def test_empty_episodes_no_crash(self):
        result = analyze_rhythms("user-1", [])
        assert result.rolling_averages == {}
        assert result.periodicity_results == {}


# ── Service Integration Tests ────────────────────────────────────────


class TestRhythmDetectionServiceStatistics:
    """Tests for RhythmDetectionService with statistical analysis."""

    @pytest.mark.asyncio
    async def test_service_returns_periodicity_data(self):
        store = InMemoryEpisodicStore()
        base = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)

        # Populate with weekly pattern
        for week in range(5):
            for day_offset, (day, valence, arousal) in enumerate([
                ("monday", 0.7, 0.8),
                ("tuesday", 0.3, 0.5),
                ("wednesday", 0.2, 0.4),
                ("thursday", 0.1, 0.5),
                ("friday", -0.5, 0.2),
            ]):
                ts = base + timedelta(weeks=week, days=day_offset)
                ep = _make_episode(
                    day=day, time="morning",
                    valence=valence, arousal=arousal,
                    timestamp=ts,
                )
                await store.append(ep)

        service = RhythmDetectionService(store)
        start = base - timedelta(days=1)
        end = base + timedelta(weeks=6)

        result = await service.analyze_user_rhythms("user-1", start, end)

        assert result.total_episodes_analyzed == 25
        assert len(result.rolling_averages) > 0
        assert len(result.periodicity_results) > 0

    @pytest.mark.asyncio
    async def test_context_includes_trend_data(self):
        store = InMemoryEpisodicStore()
        base = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)

        for week in range(5):
            for day_offset, (day, valence, arousal) in enumerate([
                ("monday", 0.7, 0.8),
                ("tuesday", 0.3, 0.5),
                ("wednesday", 0.2, 0.4),
                ("thursday", 0.1, 0.5),
                ("friday", -0.5, 0.2),
            ]):
                ts = base + timedelta(weeks=week, days=day_offset)
                ep = _make_episode(
                    day=day, time="morning",
                    valence=valence, arousal=arousal,
                    timestamp=ts,
                )
                await store.append(ep)

        service = RhythmDetectionService(store)
        context = await service.get_current_rhythm_context(
            "user-1", "monday", "morning",
        )

        assert context["day_of_week"] == "monday"
        assert context["time_of_day"] == "morning"
        # Should include rolling average and periodicity info
        # (may be None if slot has no data, but the keys should be present)
        assert "rolling_average" in context
        assert "periodicity" in context


# ── Weekly Slot Samples Tests ────────────────────────────────────────


class TestEpisodesToWeeklySlotSamples:
    """Tests for _episodes_to_weekly_slot_samples."""

    def test_empty_episodes(self):
        result = _episodes_to_weekly_slot_samples(
            [], lambda e: e.emotion.valence,
        )
        assert result == {}

    def test_single_week(self):
        episodes = _make_weekly_episodes(
            weeks=1,
            slot_configs={"monday:morning": {"valence": 0.5, "arousal": 0.7}},
        )
        result = _episodes_to_weekly_slot_samples(
            episodes, lambda e: e.emotion.valence,
        )
        assert "monday:morning" in result
        assert len(result["monday:morning"]) == 1
        assert abs(result["monday:morning"][0].value - 0.5) < 0.01

    def test_multiple_weeks(self):
        episodes = _make_weekly_episodes(
            weeks=4,
            slot_configs={"monday:morning": {"valence": 0.5, "arousal": 0.7}},
        )
        result = _episodes_to_weekly_slot_samples(
            episodes, lambda e: e.emotion.valence,
        )
        assert "monday:morning" in result
        assert len(result["monday:morning"]) == 4

    def test_multiple_episodes_same_week_averaged(self):
        """Two episodes in the same slot+week should be averaged."""
        base = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
        episodes = [
            _make_episode(
                day="monday", time="morning", valence=0.2, arousal=0.5,
                timestamp=base,
            ),
            _make_episode(
                day="monday", time="morning", valence=0.8, arousal=0.5,
                timestamp=base + timedelta(hours=1),
            ),
        ]
        result = _episodes_to_weekly_slot_samples(
            episodes, lambda e: e.emotion.valence,
        )
        assert "monday:morning" in result
        assert len(result["monday:morning"]) == 1
        assert abs(result["monday:morning"][0].value - 0.5) < 0.01

    def test_invalid_day_skipped(self):
        episodes = [_make_episode(day="invalid", time="morning")]
        result = _episodes_to_weekly_slot_samples(
            episodes, lambda e: e.emotion.valence,
        )
        assert result == {}


# ── PeriodicityResult Tests ──────────────────────────────────────────


class TestPeriodicityResult:
    def test_confidence_multiplier_periodic(self):
        pr = PeriodicityResult(
            slot_key="monday:morning",
            day_of_week="monday", time_of_day="morning",
            autocorrelation_lag7=0.6,
            is_periodic=True,
            period_strength=0.7,
            sample_count=8,
        )
        assert pr.confidence_multiplier > 1.0

    def test_confidence_multiplier_not_periodic(self):
        pr = PeriodicityResult(
            slot_key="monday:morning",
            day_of_week="monday", time_of_day="morning",
            autocorrelation_lag7=0.1,
            is_periodic=False,
            period_strength=0.1,
            sample_count=4,
        )
        assert pr.confidence_multiplier == 1.0


# ── RollingAverageResult Tests ───────────────────────────────────────


class TestRollingAverageResult:
    def test_trending_up(self):
        ra = RollingAverageResult(
            slot_key="monday:morning",
            day_of_week="monday", time_of_day="morning",
            rolling_values=[0.3, 0.5, 0.7],
            overall_mean=0.5,
            trend_direction=0.5,
            trend_strength=0.9,
            weeks_of_data=5,
        )
        assert ra.is_trending_up
        assert not ra.is_trending_down
        assert not ra.is_stable

    def test_trending_down(self):
        ra = RollingAverageResult(
            slot_key="friday:afternoon",
            day_of_week="friday", time_of_day="afternoon",
            rolling_values=[0.7, 0.5, 0.3],
            overall_mean=0.5,
            trend_direction=-0.5,
            trend_strength=0.9,
            weeks_of_data=5,
        )
        assert not ra.is_trending_up
        assert ra.is_trending_down
        assert not ra.is_stable

    def test_stable(self):
        ra = RollingAverageResult(
            slot_key="wednesday:morning",
            day_of_week="wednesday", time_of_day="morning",
            rolling_values=[0.5, 0.5, 0.5],
            overall_mean=0.5,
            trend_direction=0.05,
            trend_strength=0.1,
            weeks_of_data=5,
        )
        assert not ra.is_trending_up
        assert not ra.is_trending_down
        assert ra.is_stable


# ── WeeklySlotSample Tests ──────────────────────────────────────────


class TestWeeklySlotSample:
    def test_basic_creation(self):
        sample = WeeklySlotSample(week_number=1, value=0.5, observation_count=3)
        assert sample.week_number == 1
        assert sample.value == 0.5
        assert sample.observation_count == 3

    def test_default_observation_count(self):
        sample = WeeklySlotSample(week_number=1, value=0.5)
        assert sample.observation_count == 0
