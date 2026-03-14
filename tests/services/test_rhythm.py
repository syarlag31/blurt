"""Tests for rhythm detection service.

Validates that the algorithm correctly identifies recurring temporal patterns
(energy crashes, creativity peaks, productivity windows, mood cycles)
from episodic memory data across weekly cycles.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from blurt.memory.episodic import (
    BehavioralSignal,
    EmotionSnapshot,
    Episode,
    EpisodeContext,
    InMemoryEpisodicStore,
    InputModality,
)
from blurt.models.entities import LearnedPattern, PatternType
from blurt.services.rhythm import (
    DetectedRhythm,
    RhythmAnalysisResult,
    RhythmDetectionService,
    RhythmType,
    RhythmBucket,
    _bucket_key,
    _confidence_from_count,
    _generate_recommendations,
    _std,
    _z_score,
    aggregate_episodes,
    analyze_rhythms,
    detect_creativity_peaks,
    detect_energy_rhythms,
    detect_mood_cycles,
    detect_productivity_patterns,
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
    import uuid
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


def _make_episodes_for_pattern(
    day: str,
    time: str,
    count: int,
    valence: float,
    arousal: float,
    intent: str = "task",
    signal: BehavioralSignal = BehavioralSignal.NONE,
) -> list[Episode]:
    """Generate multiple episodes for a specific slot with consistent values."""
    return [
        _make_episode(
            day=day,
            time=time,
            valence=valence,
            arousal=arousal,
            intent=intent,
            signal=signal,
        )
        for _ in range(count)
    ]


# ── Unit tests: helpers ──────────────────────────────────────────────

class TestHelpers:
    def test_std_empty(self):
        assert _std(0, 0, 0) == 0.0

    def test_std_single(self):
        assert _std(5.0, 25.0, 1) == 0.0

    def test_std_basic(self):
        # Values: [2, 4, 6] -> mean=4, var=8/3, std≈1.633
        s = _std(12.0, 56.0, 3)
        assert abs(s - 1.633) < 0.01

    def test_z_score_zero_std(self):
        assert _z_score(5.0, 3.0, 0.0) == 0.0

    def test_z_score_basic(self):
        z = _z_score(7.0, 5.0, 1.0)
        assert z == 2.0

    def test_z_score_negative(self):
        z = _z_score(3.0, 5.0, 1.0)
        assert z == -2.0

    def test_bucket_key(self):
        assert _bucket_key("monday", "morning") == "monday:morning"

    def test_confidence_scaling(self):
        assert _confidence_from_count(0) == 0.0
        assert _confidence_from_count(25) == 0.5
        assert _confidence_from_count(50) == 1.0
        assert _confidence_from_count(100) == 1.0


# ── Unit tests: RhythmBucket ───────────────────────────────────────

class TestRhythmBucket:
    def test_empty_bucket(self):
        b = RhythmBucket(day_of_week="monday", time_of_day="morning")
        assert b.mean_valence == 0.0
        assert b.mean_arousal == 0.0
        assert b.energy_score == 0.0
        assert b.completion_rate == 0.0
        assert b.creativity_ratio == 0.0

    def test_add_episode(self):
        b = RhythmBucket(day_of_week="monday", time_of_day="morning")
        ep = _make_episode(valence=0.5, arousal=0.8)
        b.add_episode(ep)

        assert b.observation_count == 1
        assert b.mean_valence == 0.5
        assert b.mean_arousal == 0.8
        assert ep.id in b.episode_ids

    def test_behavioral_counting(self):
        b = RhythmBucket(day_of_week="monday", time_of_day="morning")
        b.add_episode(_make_episode(signal=BehavioralSignal.COMPLETED))
        b.add_episode(_make_episode(signal=BehavioralSignal.COMPLETED))
        b.add_episode(_make_episode(signal=BehavioralSignal.SKIPPED))

        assert b.completions == 2
        assert b.skips == 1
        assert abs(b.completion_rate - 2 / 3) < 0.01
        assert abs(b.skip_rate - 1 / 3) < 0.01

    def test_creative_counting(self):
        b = RhythmBucket(day_of_week="monday", time_of_day="morning")
        b.add_episode(_make_episode(intent="idea"))
        b.add_episode(_make_episode(intent="journal"))
        b.add_episode(_make_episode(intent="task"))

        assert b.creative_count == 2
        assert abs(b.creativity_ratio - 2 / 3) < 0.01

    def test_energy_score_positive(self):
        """High arousal + positive valence = high energy."""
        b = RhythmBucket(day_of_week="monday", time_of_day="morning")
        b.add_episode(_make_episode(valence=0.8, arousal=0.9))
        # energy = 0.9 * (0.5 + 0.5 * 0.8) = 0.9 * 0.9 = 0.81
        assert abs(b.energy_score - 0.81) < 0.01

    def test_energy_score_crash(self):
        """Low arousal + negative valence = near-zero energy."""
        b = RhythmBucket(day_of_week="friday", time_of_day="afternoon")
        b.add_episode(_make_episode(valence=-0.8, arousal=0.2))
        # energy = 0.2 * (0.5 + 0.5 * (-0.8)) = 0.2 * 0.1 = 0.02
        assert abs(b.energy_score - 0.02) < 0.01


# ── Unit tests: aggregation ──────────────────────────────────────────

class TestAggregation:
    def test_empty_episodes(self):
        buckets = aggregate_episodes([])
        assert len(buckets) == 28  # 7 days × 4 periods
        for b in buckets.values():
            assert b.observation_count == 0

    def test_basic_aggregation(self):
        episodes = [
            _make_episode(day="monday", time="morning"),
            _make_episode(day="monday", time="morning"),
            _make_episode(day="tuesday", time="evening"),
        ]
        buckets = aggregate_episodes(episodes)
        assert buckets["monday:morning"].observation_count == 2
        assert buckets["tuesday:evening"].observation_count == 1
        assert buckets["wednesday:afternoon"].observation_count == 0

    def test_invalid_day_skipped(self):
        ep = _make_episode(day="invalid_day", time="morning")
        buckets = aggregate_episodes([ep])
        for b in buckets.values():
            assert b.observation_count == 0

    def test_invalid_time_skipped(self):
        ep = _make_episode(day="monday", time="midnight")
        buckets = aggregate_episodes([ep])
        for b in buckets.values():
            assert b.observation_count == 0

    def test_case_insensitive(self):
        ep = _make_episode(day="Monday", time="Morning")
        buckets = aggregate_episodes([ep])
        assert buckets["monday:morning"].observation_count == 1


# ── Unit tests: energy rhythm detection ──────────────────────────────

class TestEnergyRhythms:
    def test_no_data_returns_empty(self):
        buckets = aggregate_episodes([])
        assert detect_energy_rhythms(buckets) == []

    def test_detects_energy_crash(self):
        """Friday afternoon crash vs positive energy elsewhere."""
        episodes: list[Episode] = []

        # Normal energy across most slots
        for day in ["monday", "tuesday", "wednesday", "thursday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 5, valence=0.5, arousal=0.7,
            ))

        # Energy crash on Friday afternoon
        episodes.extend(_make_episodes_for_pattern(
            "friday", "afternoon", 5, valence=-0.7, arousal=0.2,
        ))

        buckets = aggregate_episodes(episodes)
        rhythms = detect_energy_rhythms(buckets)

        crash_rhythms = [r for r in rhythms if r.rhythm_type == RhythmType.ENERGY_CRASH]
        assert len(crash_rhythms) >= 1
        assert any(r.day_of_week == "friday" and r.time_of_day == "afternoon"
                    for r in crash_rhythms)

    def test_detects_energy_peak(self):
        """Monday morning peak vs neutral energy elsewhere."""
        episodes: list[Episode] = []

        # Neutral energy baseline
        for day in ["tuesday", "wednesday", "thursday", "friday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 5, valence=0.0, arousal=0.4,
            ))

        # Energy peak Monday morning
        episodes.extend(_make_episodes_for_pattern(
            "monday", "morning", 5, valence=0.9, arousal=0.9,
        ))

        buckets = aggregate_episodes(episodes)
        rhythms = detect_energy_rhythms(buckets)

        peak_rhythms = [r for r in rhythms if r.rhythm_type == RhythmType.ENERGY_PEAK]
        assert len(peak_rhythms) >= 1
        assert any(r.day_of_week == "monday" for r in peak_rhythms)

    def test_insufficient_data_returns_empty(self):
        """Too few observations per bucket → no patterns detected."""
        episodes = [
            _make_episode(day="monday", time="morning", valence=-1.0, arousal=0.1),
            _make_episode(day="monday", time="morning", valence=-1.0, arousal=0.1),
        ]
        buckets = aggregate_episodes(episodes)
        assert detect_energy_rhythms(buckets) == []


# ── Unit tests: creativity peak detection ────────────────────────────

class TestCreativityPeaks:
    def test_detects_creative_peak(self):
        episodes: list[Episode] = []

        # Normal non-creative across most slots
        for day in ["tuesday", "wednesday", "thursday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 5, valence=0.3, arousal=0.5, intent="task",
            ))

        # Creative burst Sunday evening
        episodes.extend(_make_episodes_for_pattern(
            "sunday", "evening", 5, valence=0.5, arousal=0.7, intent="idea",
        ))

        buckets = aggregate_episodes(episodes)
        rhythms = detect_creativity_peaks(buckets)

        assert len(rhythms) >= 1
        assert any(r.day_of_week == "sunday" and r.time_of_day == "evening"
                    for r in rhythms)

    def test_no_creative_episodes_returns_empty(self):
        episodes = _make_episodes_for_pattern(
            "monday", "morning", 10, valence=0.3, arousal=0.5, intent="task",
        )
        buckets = aggregate_episodes(episodes)
        # All in one bucket → no variance → no peaks
        assert detect_creativity_peaks(buckets) == []


# ── Unit tests: productivity patterns ────────────────────────────────

class TestProductivityPatterns:
    def test_detects_productivity_window(self):
        episodes: list[Episode] = []

        # Mixed completion/skip elsewhere
        for day in ["tuesday", "wednesday", "thursday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 3,
                valence=0.0, arousal=0.5,
                signal=BehavioralSignal.COMPLETED,
            ))
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 3,
                valence=0.0, arousal=0.5,
                signal=BehavioralSignal.SKIPPED,
            ))

        # High completion Monday morning
        episodes.extend(_make_episodes_for_pattern(
            "monday", "morning", 6,
            valence=0.5, arousal=0.7,
            signal=BehavioralSignal.COMPLETED,
        ))

        buckets = aggregate_episodes(episodes)
        rhythms = detect_productivity_patterns(buckets)

        prod_windows = [r for r in rhythms
                        if r.rhythm_type == RhythmType.PRODUCTIVITY_WINDOW]
        assert len(prod_windows) >= 1
        assert any(r.day_of_week == "monday" for r in prod_windows)

    def test_detects_productivity_dip(self):
        episodes: list[Episode] = []

        # Good completion elsewhere
        for day in ["monday", "tuesday", "wednesday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 5,
                valence=0.5, arousal=0.7,
                signal=BehavioralSignal.COMPLETED,
            ))

        # All skips on Friday afternoon
        episodes.extend(_make_episodes_for_pattern(
            "friday", "afternoon", 5,
            valence=-0.3, arousal=0.3,
            signal=BehavioralSignal.SKIPPED,
        ))

        buckets = aggregate_episodes(episodes)
        rhythms = detect_productivity_patterns(buckets)

        dips = [r for r in rhythms if r.rhythm_type == RhythmType.PRODUCTIVITY_DIP]
        assert len(dips) >= 1
        assert any(r.day_of_week == "friday" for r in dips)


# ── Unit tests: mood cycle detection ─────────────────────────────────

class TestMoodCycles:
    def test_detects_mood_low(self):
        episodes: list[Episode] = []

        # Positive mood most of the week
        for day in ["monday", "tuesday", "wednesday", "thursday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 5, valence=0.6, arousal=0.5,
            ))

        # Mood dip Sunday night
        episodes.extend(_make_episodes_for_pattern(
            "sunday", "night", 5, valence=-0.7, arousal=0.3,
        ))

        buckets = aggregate_episodes(episodes)
        rhythms = detect_mood_cycles(buckets)

        lows = [r for r in rhythms if r.rhythm_type == RhythmType.MOOD_LOW]
        assert len(lows) >= 1
        assert any(r.day_of_week == "sunday" for r in lows)

    def test_detects_mood_high(self):
        episodes: list[Episode] = []

        # Neutral mood baseline
        for day in ["tuesday", "wednesday", "thursday", "friday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 5, valence=0.0, arousal=0.5,
            ))

        # Great mood Saturday morning
        episodes.extend(_make_episodes_for_pattern(
            "saturday", "morning", 5, valence=0.9, arousal=0.8,
        ))

        buckets = aggregate_episodes(episodes)
        rhythms = detect_mood_cycles(buckets)

        highs = [r for r in rhythms if r.rhythm_type == RhythmType.MOOD_HIGH]
        assert len(highs) >= 1
        assert any(r.day_of_week == "saturday" for r in highs)


# ── Integration: full pipeline ───────────────────────────────────────

class TestAnalyzeRhythms:
    def test_empty_episodes(self):
        result = analyze_rhythms("user-1", [])
        assert result.total_episodes_analyzed == 0
        assert result.rhythms == []

    def test_below_minimum_observations(self):
        episodes = [_make_episode() for _ in range(5)]
        result = analyze_rhythms("user-1", episodes)
        assert result.total_episodes_analyzed == 5
        assert result.rhythms == []  # Below MIN_TOTAL_OBSERVATIONS

    def test_full_pipeline_detects_patterns(self):
        """End-to-end test with a realistic weekly pattern."""
        episodes: list[Episode] = []

        # Monday morning: high energy, productive
        episodes.extend(_make_episodes_for_pattern(
            "monday", "morning", 8,
            valence=0.7, arousal=0.8, intent="task",
            signal=BehavioralSignal.COMPLETED,
        ))

        # Tuesday-Thursday: neutral baseline
        for day in ["tuesday", "wednesday", "thursday"]:
            for time in ["morning", "afternoon"]:
                episodes.extend(_make_episodes_for_pattern(
                    day, time, 4,
                    valence=0.1, arousal=0.5, intent="task",
                    signal=BehavioralSignal.COMPLETED,
                ))

        # Friday afternoon: energy crash + skipping tasks
        episodes.extend(_make_episodes_for_pattern(
            "friday", "afternoon", 8,
            valence=-0.6, arousal=0.2, intent="task",
            signal=BehavioralSignal.SKIPPED,
        ))

        # Sunday evening: creative burst
        episodes.extend(_make_episodes_for_pattern(
            "sunday", "evening", 6,
            valence=0.5, arousal=0.6, intent="idea",
        ))

        result = analyze_rhythms("user-1", episodes)

        assert result.total_episodes_analyzed == len(episodes)
        assert result.user_id == "user-1"
        assert len(result.rhythms) > 0

        # Check we detected the expected patterns
        rhythm_types = {r.rhythm_type for r in result.rhythms}

        # Should detect at least energy patterns
        assert any(rt in rhythm_types for rt in [
            RhythmType.ENERGY_CRASH, RhythmType.ENERGY_PEAK,
            RhythmType.MOOD_LOW, RhythmType.MOOD_HIGH,
        ]), f"Expected energy/mood patterns, got {rhythm_types}"

    def test_learned_patterns_generated(self):
        """Verify rhythms convert to LearnedPattern instances."""
        episodes: list[Episode] = []

        for day in ["monday", "tuesday", "wednesday", "thursday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 5, valence=0.5, arousal=0.6,
            ))

        episodes.extend(_make_episodes_for_pattern(
            "friday", "afternoon", 5, valence=-0.8, arousal=0.1,
        ))

        result = analyze_rhythms("user-1", episodes)
        patterns = result.to_learned_patterns()

        for pattern in patterns:
            assert isinstance(pattern, LearnedPattern)
            assert pattern.user_id == "user-1"
            assert pattern.pattern_type in list(PatternType)
            assert 0.0 <= pattern.confidence <= 1.0
            assert pattern.description  # Non-empty
            assert "rhythm_type" in pattern.parameters

    def test_rhythms_sorted_by_significance(self):
        """Rhythms should be sorted by confidence × z_score descending."""
        episodes: list[Episode] = []

        for day in ["monday", "tuesday", "wednesday", "thursday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 5, valence=0.3, arousal=0.5,
            ))

        # Strong pattern
        episodes.extend(_make_episodes_for_pattern(
            "friday", "afternoon", 10, valence=-0.9, arousal=0.1,
        ))

        result = analyze_rhythms("user-1", episodes)

        if len(result.rhythms) >= 2:
            for i in range(len(result.rhythms) - 1):
                a = result.rhythms[i]
                b = result.rhythms[i + 1]
                assert a.confidence * a.z_score >= b.confidence * b.z_score

    def test_to_dict_serialization(self):
        episodes: list[Episode] = []
        for day in ["monday", "tuesday", "wednesday", "thursday"]:
            episodes.extend(_make_episodes_for_pattern(
                day, "morning", 6, valence=0.5, arousal=0.7,
            ))
        episodes.extend(_make_episodes_for_pattern(
            "friday", "afternoon", 6, valence=-0.7, arousal=0.2,
        ))

        result = analyze_rhythms("user-1", episodes)
        data = result.to_dict()

        assert "user_id" in data
        assert "rhythms" in data
        assert "summary" in data
        assert isinstance(data["summary"]["energy_crashes"], int)
        assert isinstance(data["summary"]["creativity_peaks"], int)

    def test_bucket_stats_populated(self):
        episodes = _make_episodes_for_pattern(
            "monday", "morning", 5, valence=0.5, arousal=0.7,
        )
        result = analyze_rhythms("user-1", episodes)

        assert "monday:morning" in result.bucket_stats
        stats = result.bucket_stats["monday:morning"]
        assert stats["observation_count"] == 5
        assert "mean_valence" in stats
        assert "energy_score" in stats


# ── DetectedRhythm unit tests ───────────────────────────────────────

class TestDetectedRhythm:
    def test_description_generation(self):
        rhythm = DetectedRhythm(
            rhythm_type=RhythmType.ENERGY_CRASH,
            day_of_week="friday",
            time_of_day="afternoon",
            z_score=2.1,
            metric_value=0.05,
            metric_mean=0.5,
            metric_std=0.2,
            observation_count=10,
            confidence=0.8,
        )
        desc = rhythm.description
        assert "Friday" in desc
        assert "afternoon" in desc

    def test_to_learned_pattern(self):
        rhythm = DetectedRhythm(
            rhythm_type=RhythmType.CREATIVITY_PEAK,
            day_of_week="sunday",
            time_of_day="evening",
            z_score=1.8,
            metric_value=0.6,
            metric_mean=0.2,
            metric_std=0.15,
            observation_count=12,
            confidence=0.24,
            evidence_episode_ids=["ep1", "ep2"],
        )
        pattern = rhythm.to_learned_pattern("user-1")

        assert pattern.user_id == "user-1"
        assert pattern.pattern_type == PatternType.TIME_OF_DAY
        assert pattern.confidence == 0.24
        assert "sunday" in pattern.parameters["day_of_week"]
        assert "evening" in pattern.parameters["time_of_day"]
        assert "ep1" in pattern.supporting_evidence


# ── Service integration tests ────────────────────────────────────────

class TestRhythmDetectionService:
    @pytest.mark.asyncio
    async def test_analyze_user_rhythms(self):
        store = InMemoryEpisodicStore()

        # Populate store with episodes
        for day in ["monday", "tuesday", "wednesday", "thursday"]:
            for ep in _make_episodes_for_pattern(
                day, "morning", 6, valence=0.5, arousal=0.6,
            ):
                await store.append(ep)

        for ep in _make_episodes_for_pattern(
            "friday", "afternoon", 6, valence=-0.7, arousal=0.15,
        ):
            await store.append(ep)

        service = RhythmDetectionService(store)
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30)

        result = await service.analyze_user_rhythms("user-1", start, now)

        assert result.total_episodes_analyzed == 30
        assert result.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_get_current_rhythm_context(self):
        store = InMemoryEpisodicStore()

        # Create enough data for pattern detection
        for day in ["monday", "tuesday", "wednesday", "thursday"]:
            for ep in _make_episodes_for_pattern(
                day, "morning", 5, valence=0.4, arousal=0.6,
            ):
                await store.append(ep)

        for ep in _make_episodes_for_pattern(
            "friday", "afternoon", 5, valence=-0.8, arousal=0.1,
        ):
            await store.append(ep)

        service = RhythmDetectionService(store)
        context = await service.get_current_rhythm_context(
            "user-1", "friday", "afternoon",
        )

        assert context["day_of_week"] == "friday"
        assert context["time_of_day"] == "afternoon"
        assert isinstance(context["active_rhythms"], list)
        assert isinstance(context["recommendations"], list)

    @pytest.mark.asyncio
    async def test_empty_store(self):
        store = InMemoryEpisodicStore()
        service = RhythmDetectionService(store)
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=7)

        result = await service.analyze_user_rhythms("user-1", start, now)
        assert result.total_episodes_analyzed == 0
        assert result.rhythms == []


# ── Recommendations ──────────────────────────────────────────────────

class TestRecommendations:
    def test_energy_crash_recommendation(self):
        rhythms = [DetectedRhythm(
            rhythm_type=RhythmType.ENERGY_CRASH,
            day_of_week="friday", time_of_day="afternoon",
            z_score=2.0, metric_value=0.05, metric_mean=0.5,
            metric_std=0.2, observation_count=10, confidence=0.8,
        )]
        recs = _generate_recommendations(rhythms)
        assert len(recs) == 1
        assert "lower-energy" in recs[0]
        # Anti-shame: no guilt language
        assert "overdue" not in recs[0].lower()
        assert "should" not in recs[0].lower()
        assert "must" not in recs[0].lower()

    def test_productivity_dip_recommendation_is_shame_free(self):
        rhythms = [DetectedRhythm(
            rhythm_type=RhythmType.PRODUCTIVITY_DIP,
            day_of_week="friday", time_of_day="afternoon",
            z_score=1.8, metric_value=0.1, metric_mean=0.6,
            metric_std=0.2, observation_count=8, confidence=0.5,
        )]
        recs = _generate_recommendations(rhythms)
        assert len(recs) == 1
        assert "okay" in recs[0].lower()
        # No shame language
        assert "fail" not in recs[0].lower()
        assert "behind" not in recs[0].lower()

    def test_mood_low_recommendation_is_gentle(self):
        rhythms = [DetectedRhythm(
            rhythm_type=RhythmType.MOOD_LOW,
            day_of_week="sunday", time_of_day="night",
            z_score=2.0, metric_value=-0.6, metric_mean=0.1,
            metric_std=0.3, observation_count=7, confidence=0.5,
        )]
        recs = _generate_recommendations(rhythms)
        assert len(recs) == 1
        assert "gentle" in recs[0].lower()

    def test_empty_rhythms_no_recommendations(self):
        assert _generate_recommendations([]) == []

    def test_multiple_rhythms_multiple_recommendations(self):
        rhythms = [
            DetectedRhythm(
                rhythm_type=RhythmType.ENERGY_PEAK,
                day_of_week="monday", time_of_day="morning",
                z_score=2.0, metric_value=0.8, metric_mean=0.4,
                metric_std=0.2, observation_count=10, confidence=0.8,
            ),
            DetectedRhythm(
                rhythm_type=RhythmType.CREATIVITY_PEAK,
                day_of_week="monday", time_of_day="morning",
                z_score=1.7, metric_value=0.5, metric_mean=0.2,
                metric_std=0.15, observation_count=8, confidence=0.6,
            ),
        ]
        recs = _generate_recommendations(rhythms)
        assert len(recs) == 2


# ── Result property helpers ──────────────────────────────────────────

class TestResultProperties:
    def test_energy_crashes_filter(self):
        result = RhythmAnalysisResult(
            user_id="u1",
            analysis_period_start=datetime.now(timezone.utc),
            analysis_period_end=datetime.now(timezone.utc),
            total_episodes_analyzed=50,
            rhythms=[
                DetectedRhythm(
                    rhythm_type=RhythmType.ENERGY_CRASH,
                    day_of_week="friday", time_of_day="afternoon",
                    z_score=2.0, metric_value=0.1, metric_mean=0.5,
                    metric_std=0.2, observation_count=10, confidence=0.8,
                ),
                DetectedRhythm(
                    rhythm_type=RhythmType.MOOD_HIGH,
                    day_of_week="saturday", time_of_day="morning",
                    z_score=1.8, metric_value=0.8, metric_mean=0.3,
                    metric_std=0.25, observation_count=8, confidence=0.6,
                ),
            ],
        )

        assert len(result.energy_crashes) == 1
        assert len(result.mood_patterns) == 1
        assert len(result.creativity_peaks) == 0
        assert len(result.productivity_windows) == 0
