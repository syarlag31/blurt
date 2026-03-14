"""Tests for temporal activity aggregation service.

Validates that user interaction data (energy levels, productivity metrics,
emotion signals) is correctly collected, bucketed by day-of-week and
time-of-day, and aggregated into temporal profiles.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from blurt.memory.episodic import (
    BehavioralSignal,
    EmotionSnapshot,
    Episode,
    EpisodeContext,
    InputModality,
)
from blurt.services.temporal_activity import (
    DAYS_OF_WEEK,
    HourlyBucket,
    InteractionRecord,
    TemporalActivityService,
    TemporalActivityStore,
    TemporalBucket,
    TemporalProfile,
    TemporalSlot,
    TimeOfDay,
    episode_to_interaction,
    hour_to_time_of_day,
    weekday_to_name,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_episode(
    user_id: str = "user-1",
    raw_text: str = "test blurt",
    intent: str = "journal",
    day_of_week: str = "monday",
    time_of_day: str = "morning",
    arousal: float = 0.5,
    valence: float = 0.0,
    primary_emotion: str = "trust",
    emotion_intensity: float = 0.3,
    behavioral_signal: BehavioralSignal = BehavioralSignal.NONE,
    modality: InputModality = InputModality.VOICE,
    timestamp: datetime | None = None,
) -> Episode:
    """Create an Episode for testing."""
    ts = timestamp or datetime(2026, 3, 9, 10, 0, 0, tzinfo=timezone.utc)  # Monday morning
    return Episode(
        user_id=user_id,
        raw_text=raw_text,
        intent=intent,
        timestamp=ts,
        modality=modality,
        emotion=EmotionSnapshot(
            primary=primary_emotion,
            intensity=emotion_intensity,
            valence=valence,
            arousal=arousal,
        ),
        behavioral_signal=behavioral_signal,
        context=EpisodeContext(
            time_of_day=time_of_day,
            day_of_week=day_of_week,
        ),
    )


# ---------------------------------------------------------------------------
# Unit tests: time bucketing helpers
# ---------------------------------------------------------------------------


class TestTimeBucketing:
    def test_hour_to_morning(self):
        for h in [6, 7, 8, 9, 10, 11]:
            assert hour_to_time_of_day(h) == TimeOfDay.MORNING

    def test_hour_to_afternoon(self):
        for h in [12, 13, 14, 15, 16]:
            assert hour_to_time_of_day(h) == TimeOfDay.AFTERNOON

    def test_hour_to_evening(self):
        for h in [17, 18, 19, 20]:
            assert hour_to_time_of_day(h) == TimeOfDay.EVENING

    def test_hour_to_night(self):
        for h in [21, 22, 23, 0, 1, 2, 3, 4, 5]:
            assert hour_to_time_of_day(h) == TimeOfDay.NIGHT

    def test_weekday_to_name(self):
        assert weekday_to_name(0) == "monday"
        assert weekday_to_name(6) == "sunday"
        assert weekday_to_name(4) == "friday"

    def test_days_of_week_tuple(self):
        assert len(DAYS_OF_WEEK) == 7
        assert DAYS_OF_WEEK[0] == "monday"
        assert DAYS_OF_WEEK[6] == "sunday"


# ---------------------------------------------------------------------------
# Unit tests: TemporalBucket
# ---------------------------------------------------------------------------


class TestTemporalBucket:
    def test_empty_bucket_defaults(self):
        bucket = TemporalBucket()
        assert bucket.avg_energy == 0.5  # default when no samples
        assert bucket.avg_valence == 0.0
        assert bucket.avg_emotion_intensity == 0.0
        assert bucket.avg_word_count == 0.0
        assert bucket.completion_rate == 0.0
        assert bucket.dominant_emotion is None
        assert bucket.dominant_intent is None
        assert bucket.productivity_score == 0.2  # 0.5 * 0.4 = 0.2 (default energy only)

    def test_record_single_interaction(self):
        bucket = TemporalBucket(day_of_week="monday", time_of_day="morning")
        record = InteractionRecord(
            energy_level=0.8,
            valence=0.5,
            task_completed=True,
            intent="task",
            primary_emotion="joy",
            emotion_intensity=0.7,
            word_count=20,
            modality="voice",
        )
        bucket.record(record)

        assert bucket.interaction_count == 1
        assert bucket.avg_energy == 0.8
        assert bucket.avg_valence == 0.5
        assert bucket.tasks_completed == 1
        assert bucket.dominant_emotion == "joy"
        assert bucket.dominant_intent == "task"
        assert bucket.voice_count == 1
        assert bucket.text_count == 0

    def test_record_multiple_interactions(self):
        bucket = TemporalBucket()
        for energy in [0.3, 0.7, 0.5]:
            bucket.record(InteractionRecord(
                energy_level=energy,
                valence=0.0,
                primary_emotion="trust",
                word_count=10,
            ))
        assert bucket.interaction_count == 3
        assert abs(bucket.avg_energy - 0.5) < 0.001

    def test_completion_rate(self):
        bucket = TemporalBucket()
        bucket.record(InteractionRecord(task_completed=True))
        bucket.record(InteractionRecord(task_completed=True))
        bucket.record(InteractionRecord(task_skipped=True))
        assert abs(bucket.completion_rate - 2/3) < 0.001

    def test_to_dict(self):
        bucket = TemporalBucket(day_of_week="friday", time_of_day="evening")
        d = bucket.to_dict()
        assert d["day_of_week"] == "friday"
        assert d["time_of_day"] == "evening"
        assert "avg_energy" in d
        assert "productivity_score" in d

    def test_text_modality_counted(self):
        bucket = TemporalBucket()
        bucket.record(InteractionRecord(modality="text"))
        assert bucket.text_count == 1
        assert bucket.voice_count == 0


# ---------------------------------------------------------------------------
# Unit tests: episode_to_interaction converter
# ---------------------------------------------------------------------------


class TestEpisodeToInteraction:
    def test_basic_conversion(self):
        ep = make_episode(
            raw_text="I need to finish the report",
            intent="task",
            arousal=0.7,
            valence=0.3,
            primary_emotion="anticipation",
            emotion_intensity=0.6,
        )
        record = episode_to_interaction(ep)

        assert record.user_id == "user-1"
        assert record.energy_level == 0.7  # from arousal
        assert record.valence == 0.3
        assert record.intent == "task"
        assert record.task_created  # task intent = task created
        assert record.primary_emotion == "anticipation"
        assert record.emotion_intensity == 0.6
        assert record.word_count == 6
        assert record.episode_id is not None

    def test_timestamp_overrides_context(self):
        # Wednesday at 14:30 UTC
        ts = datetime(2026, 3, 11, 14, 30, 0, tzinfo=timezone.utc)
        ep = make_episode(
            timestamp=ts,
            day_of_week="monday",  # context says monday
            time_of_day="morning",  # context says morning
        )
        record = episode_to_interaction(ep)

        # Timestamp should override context
        assert record.day_of_week == "wednesday"
        assert record.time_of_day == "afternoon"
        assert record.hour == 14

    def test_behavioral_signal_completed(self):
        ep = make_episode(behavioral_signal=BehavioralSignal.COMPLETED)
        record = episode_to_interaction(ep)
        assert record.task_completed is True
        assert record.task_skipped is False

    def test_behavioral_signal_skipped(self):
        ep = make_episode(behavioral_signal=BehavioralSignal.SKIPPED)
        record = episode_to_interaction(ep)
        assert record.task_skipped is True
        assert record.task_completed is False

    def test_behavioral_signal_dismissed(self):
        ep = make_episode(behavioral_signal=BehavioralSignal.DISMISSED)
        record = episode_to_interaction(ep)
        assert record.task_dismissed is True

    def test_task_creating_intents(self):
        for intent in ["task", "reminder", "event"]:
            ep = make_episode(intent=intent)
            record = episode_to_interaction(ep)
            assert record.task_created, f"{intent} should set task_created"

    def test_non_task_intents(self):
        for intent in ["journal", "idea", "question", "update"]:
            ep = make_episode(intent=intent)
            record = episode_to_interaction(ep)
            assert not record.task_created, f"{intent} should not set task_created"

    def test_energy_clamped(self):
        ep = make_episode(arousal=1.5)  # Over max
        record = episode_to_interaction(ep)
        assert record.energy_level == 1.0

    def test_voice_modality(self):
        ep = make_episode(modality=InputModality.VOICE)
        record = episode_to_interaction(ep)
        assert record.modality == "voice"

    def test_text_modality(self):
        ep = make_episode(modality=InputModality.TEXT)
        record = episode_to_interaction(ep)
        assert record.modality == "text"


# ---------------------------------------------------------------------------
# Unit tests: InteractionRecord
# ---------------------------------------------------------------------------


class TestInteractionRecord:
    def test_to_dict(self):
        record = InteractionRecord(
            user_id="u1",
            energy_level=0.8,
            intent="task",
        )
        d = record.to_dict()
        assert d["user_id"] == "u1"
        assert d["energy_level"] == 0.8
        assert d["intent"] == "task"
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# Unit tests: TemporalProfile
# ---------------------------------------------------------------------------


class TestTemporalProfile:
    def test_empty_profile(self):
        profile = TemporalProfile(user_id="u1")
        assert profile.total_interactions == 0
        assert profile.all_buckets() == []
        assert profile.best_slots_for_focus() == []

    def test_get_bucket_creates_if_missing(self):
        profile = TemporalProfile(user_id="u1")
        bucket = profile.get_bucket("monday", "morning")
        assert bucket.day_of_week == "monday"
        assert bucket.time_of_day == "morning"
        assert bucket.interaction_count == 0

    def test_best_slots_requires_min_data(self):
        profile = TemporalProfile(user_id="u1")
        bucket = profile.get_bucket("monday", "morning")
        # Only 1 interaction - below threshold of 2
        bucket.record(InteractionRecord(energy_level=0.9))
        assert profile.best_slots_for_focus() == []

    def test_best_slots_for_focus_ranking(self):
        profile = TemporalProfile(user_id="u1")
        # Monday morning: high energy, high completion
        monday_am = profile.get_bucket("monday", "morning")
        for _ in range(3):
            monday_am.record(InteractionRecord(
                energy_level=0.9, task_completed=True, word_count=40,
            ))
        # Friday evening: low energy, no completion
        friday_eve = profile.get_bucket("friday", "evening")
        for _ in range(3):
            friday_eve.record(InteractionRecord(
                energy_level=0.2, word_count=5,
            ))

        slots = profile.best_slots_for_focus(top_n=2)
        assert len(slots) == 2
        assert slots[0].day_of_week == "monday"
        assert slots[0].time_of_day == "morning"
        assert slots[0].score > slots[1].score

    def test_best_slots_for_energy(self):
        profile = TemporalProfile(user_id="u1")
        high = profile.get_bucket("tuesday", "morning")
        low = profile.get_bucket("wednesday", "night")
        for _ in range(3):
            high.record(InteractionRecord(energy_level=0.9))
            low.record(InteractionRecord(energy_level=0.2))

        best = profile.best_slots_for_energy(top_n=1)
        assert best[0].day_of_week == "tuesday"
        assert best[0].score > 0.8

    def test_lowest_energy_slots(self):
        profile = TemporalProfile(user_id="u1")
        high = profile.get_bucket("tuesday", "morning")
        low = profile.get_bucket("wednesday", "night")
        for _ in range(3):
            high.record(InteractionRecord(energy_level=0.9))
            low.record(InteractionRecord(energy_level=0.2))

        lowest = profile.lowest_energy_slots(top_n=1)
        assert lowest[0].day_of_week == "wednesday"
        assert lowest[0].score < 0.3

    def test_mood_for_slot(self):
        profile = TemporalProfile(user_id="u1")
        bucket = profile.get_bucket("monday", "morning")
        bucket.record(InteractionRecord(
            primary_emotion="joy", emotion_intensity=0.8,
            valence=0.7, energy_level=0.9,
        ))
        mood = profile.mood_for_slot("monday", "morning")
        assert mood["dominant_emotion"] == "joy"
        assert mood["avg_valence"] == 0.7
        assert mood["interaction_count"] == 1

    def test_day_summary(self):
        profile = TemporalProfile(user_id="u1")
        for tod in ["morning", "afternoon"]:
            bucket = profile.get_bucket("monday", tod)
            for _ in range(3):
                bucket.record(InteractionRecord(energy_level=0.6, valence=0.2))

        summary = profile.day_summary("monday")
        assert summary["day"] == "monday"
        assert summary["total_interactions"] == 6
        assert summary["avg_energy"] > 0

    def test_day_summary_empty(self):
        profile = TemporalProfile(user_id="u1")
        summary = profile.day_summary("saturday")
        assert summary["total_interactions"] == 0
        assert summary["most_active_time"] is None

    def test_weekly_heatmap(self):
        profile = TemporalProfile(user_id="u1")
        heatmap = profile.weekly_heatmap()
        assert len(heatmap) == 28  # 7 days x 4 time slots
        assert all("day_of_week" in cell for cell in heatmap)
        assert all("time_of_day" in cell for cell in heatmap)

    def test_to_dict(self):
        profile = TemporalProfile(user_id="u1", total_interactions=5)
        d = profile.to_dict()
        assert d["user_id"] == "u1"
        assert d["total_interactions"] == 5


# ---------------------------------------------------------------------------
# Integration tests: TemporalActivityStore
# ---------------------------------------------------------------------------


class TestTemporalActivityStore:
    @pytest.fixture
    def store(self):
        return TemporalActivityStore()

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, store):
        record = InteractionRecord(user_id="u1", energy_level=0.8)
        stored = await store.store_interaction(record)
        assert stored.id == record.id

        records = await store.get_interactions("u1")
        assert len(records) == 1
        assert records[0].energy_level == 0.8

    @pytest.mark.asyncio
    async def test_filter_by_day(self, store):
        await store.store_interaction(InteractionRecord(user_id="u1", day_of_week="monday"))
        await store.store_interaction(InteractionRecord(user_id="u1", day_of_week="friday"))

        monday = await store.get_interactions("u1", day_of_week="monday")
        assert len(monday) == 1

    @pytest.mark.asyncio
    async def test_filter_by_time_of_day(self, store):
        await store.store_interaction(InteractionRecord(user_id="u1", time_of_day="morning"))
        await store.store_interaction(InteractionRecord(user_id="u1", time_of_day="evening"))

        morning = await store.get_interactions("u1", time_of_day="morning")
        assert len(morning) == 1

    @pytest.mark.asyncio
    async def test_profile_built_from_interactions(self, store):
        for _ in range(5):
            await store.store_interaction(InteractionRecord(
                user_id="u1", day_of_week="monday", time_of_day="morning",
                energy_level=0.8,
            ))
        profile = await store.get_profile("u1")
        assert profile.total_interactions == 5
        bucket = profile.get_bucket("monday", "morning")
        assert bucket.interaction_count == 5
        assert abs(bucket.avg_energy - 0.8) < 0.001

    @pytest.mark.asyncio
    async def test_profile_incremental_update(self, store):
        # Get profile first (builds cache)
        await store.store_interaction(InteractionRecord(
            user_id="u1", day_of_week="monday", time_of_day="morning",
            energy_level=0.6,
        ))
        profile1 = await store.get_profile("u1")
        assert profile1.total_interactions == 1

        # Add more data - profile should update incrementally
        await store.store_interaction(InteractionRecord(
            user_id="u1", day_of_week="monday", time_of_day="morning",
            energy_level=0.8,
        ))
        profile2 = await store.get_profile("u1")
        assert profile2.total_interactions == 2

    @pytest.mark.asyncio
    async def test_interaction_count(self, store):
        assert await store.interaction_count("u1") == 0
        await store.store_interaction(InteractionRecord(user_id="u1"))
        assert await store.interaction_count("u1") == 1

    @pytest.mark.asyncio
    async def test_clear_user(self, store):
        await store.store_interaction(InteractionRecord(user_id="u1"))
        await store.clear_user("u1")
        assert await store.interaction_count("u1") == 0

    @pytest.mark.asyncio
    async def test_user_isolation(self, store):
        await store.store_interaction(InteractionRecord(user_id="u1"))
        await store.store_interaction(InteractionRecord(user_id="u2"))

        u1_records = await store.get_interactions("u1")
        u2_records = await store.get_interactions("u2")
        assert len(u1_records) == 1
        assert len(u2_records) == 1

    @pytest.mark.asyncio
    async def test_filter_since(self, store):
        old = InteractionRecord(
            user_id="u1",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        new = InteractionRecord(
            user_id="u1",
            timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        await store.store_interaction(old)
        await store.store_interaction(new)

        recent = await store.get_interactions(
            "u1", since=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
        assert len(recent) == 1

    @pytest.mark.asyncio
    async def test_interactions_sorted_newest_first(self, store):
        for i in range(3):
            await store.store_interaction(InteractionRecord(
                user_id="u1",
                timestamp=datetime(2026, 3, i + 1, tzinfo=timezone.utc),
            ))
        records = await store.get_interactions("u1")
        assert records[0].timestamp > records[1].timestamp


# ---------------------------------------------------------------------------
# Integration tests: TemporalActivityService
# ---------------------------------------------------------------------------


class TestTemporalActivityService:
    @pytest.fixture
    def service(self):
        return TemporalActivityService()

    @pytest.mark.asyncio
    async def test_record_from_episode(self, service):
        ep = make_episode(
            raw_text="I should call mom tomorrow",
            intent="reminder",
            arousal=0.4,
            valence=0.3,
            primary_emotion="trust",
        )
        record = await service.record_from_episode(ep)

        assert record.user_id == "user-1"
        assert record.intent == "reminder"
        assert record.task_created  # reminder is task-creating
        assert record.energy_level == 0.4
        assert record.word_count == 5

    @pytest.mark.asyncio
    async def test_temporal_profile_from_episodes(self, service):
        # Simulate a week of interactions
        episodes = [
            make_episode(
                timestamp=datetime(2026, 3, 9, 9, 0, tzinfo=timezone.utc),  # Monday morning
                arousal=0.8, intent="task",
            ),
            make_episode(
                timestamp=datetime(2026, 3, 9, 10, 0, tzinfo=timezone.utc),  # Monday morning
                arousal=0.7, intent="task",
            ),
            make_episode(
                timestamp=datetime(2026, 3, 9, 14, 0, tzinfo=timezone.utc),  # Monday afternoon
                arousal=0.4, intent="journal",
            ),
            make_episode(
                timestamp=datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc),  # Tuesday morning
                arousal=0.9, intent="task",
            ),
        ]
        for ep in episodes:
            await service.record_from_episode(ep)

        profile = await service.get_temporal_profile("user-1")
        assert profile.total_interactions == 4

        # Monday morning should have 2 interactions
        mon_am = profile.get_bucket("monday", "morning")
        assert mon_am.interaction_count == 2

    @pytest.mark.asyncio
    async def test_energy_pattern(self, service):
        for i in range(3):
            await service.record_from_episode(make_episode(
                timestamp=datetime(2026, 3, 9, 9 + i, tzinfo=timezone.utc),
                arousal=0.8,
            ))
            await service.record_from_episode(make_episode(
                timestamp=datetime(2026, 3, 9, 21 + i, tzinfo=timezone.utc),
                arousal=0.2,
            ))

        pattern = await service.get_energy_pattern("user-1")
        assert pattern["total_interactions"] == 6
        assert len(pattern["weekly_heatmap"]) == 28

        # Best energy should be morning
        best = pattern["best_energy_slots"]
        if best:
            assert best[0]["energy"] > 0.5

    @pytest.mark.asyncio
    async def test_mood_pattern(self, service):
        await service.record_from_episode(make_episode(
            timestamp=datetime(2026, 3, 9, 9, tzinfo=timezone.utc),
            primary_emotion="joy", valence=0.8,
        ))
        pattern = await service.get_mood_pattern("user-1")
        assert pattern["total_interactions"] == 1
        assert len(pattern["day_summaries"]) == 7

    @pytest.mark.asyncio
    async def test_interaction_count(self, service):
        assert await service.interaction_count("user-1") == 0
        await service.record_from_episode(make_episode())
        assert await service.interaction_count("user-1") == 1

    @pytest.mark.asyncio
    async def test_get_interactions_with_filters(self, service):
        await service.record_from_episode(make_episode(
            timestamp=datetime(2026, 3, 9, 9, tzinfo=timezone.utc),  # Monday morning
        ))
        await service.record_from_episode(make_episode(
            timestamp=datetime(2026, 3, 13, 19, tzinfo=timezone.utc),  # Friday evening
        ))

        monday = await service.get_interactions("user-1", day_of_week="monday")
        assert len(monday) == 1

        friday = await service.get_interactions("user-1", day_of_week="friday")
        assert len(friday) == 1

    @pytest.mark.asyncio
    async def test_multiple_users_isolated(self, service):
        await service.record_from_episode(make_episode(user_id="u1"))
        await service.record_from_episode(make_episode(user_id="u2"))

        assert await service.interaction_count("u1") == 1
        assert await service.interaction_count("u2") == 1

        p1 = await service.get_temporal_profile("u1")
        p2 = await service.get_temporal_profile("u2")
        assert p1.total_interactions == 1
        assert p2.total_interactions == 1

    @pytest.mark.asyncio
    async def test_productivity_score_calculation(self, service):
        """Verify that productivity score accounts for energy, completion, and engagement."""
        # High productivity: high energy, completing tasks, engaged
        for _ in range(5):
            ep = make_episode(
                timestamp=datetime(2026, 3, 9, 9, tzinfo=timezone.utc),
                arousal=0.9,
                intent="task",
                behavioral_signal=BehavioralSignal.COMPLETED,
                raw_text="finished the quarterly report and sent it to the team for review",
            )
            await service.record_from_episode(ep)

        profile = await service.get_temporal_profile("user-1")
        bucket = profile.get_bucket("monday", "morning")
        assert bucket.productivity_score > 0.5  # Should be high

    @pytest.mark.asyncio
    async def test_casual_remarks_recorded(self, service):
        """Anti-shame: casual remarks should be recorded without special treatment."""
        casual_episodes = [
            make_episode(raw_text="huh", intent="journal"),
            make_episode(raw_text="nice weather today", intent="journal"),
            make_episode(raw_text="oh well", intent="journal"),
        ]
        for ep in casual_episodes:
            await service.record_from_episode(ep)

        assert await service.interaction_count("user-1") == 3
        profile = await service.get_temporal_profile("user-1")
        assert profile.total_interactions == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestHourlyBucket:
    def test_empty_bucket_defaults(self):
        bucket = HourlyBucket()
        assert bucket.avg_energy == 0.5
        assert bucket.avg_valence == 0.0
        assert bucket.avg_emotion_intensity == 0.0
        assert bucket.dominant_emotion is None
        assert bucket.interaction_count == 0

    def test_record_single_interaction(self):
        bucket = HourlyBucket(day_of_week="monday", hour=9)
        bucket.record(InteractionRecord(
            energy_level=0.8, valence=0.5,
            primary_emotion="joy", emotion_intensity=0.7,
            task_created=True,
        ))
        assert bucket.interaction_count == 1
        assert bucket.avg_energy == 0.8
        assert bucket.avg_valence == 0.5
        assert bucket.dominant_emotion == "joy"
        assert bucket.tasks_created == 1

    def test_record_multiple(self):
        bucket = HourlyBucket(day_of_week="tuesday", hour=14)
        for energy in [0.3, 0.7, 0.5]:
            bucket.record(InteractionRecord(energy_level=energy))
        assert bucket.interaction_count == 3
        assert abs(bucket.avg_energy - 0.5) < 0.001

    def test_to_dict(self):
        bucket = HourlyBucket(day_of_week="friday", hour=17)
        bucket.record(InteractionRecord(energy_level=0.9, primary_emotion="anticipation"))
        d = bucket.to_dict()
        assert d["day_of_week"] == "friday"
        assert d["hour"] == 17
        assert d["avg_energy"] == 0.9
        assert d["dominant_emotion"] == "anticipation"

    def test_task_completed_tracked(self):
        bucket = HourlyBucket()
        bucket.record(InteractionRecord(task_completed=True))
        bucket.record(InteractionRecord(task_completed=True))
        bucket.record(InteractionRecord(task_completed=False))
        assert bucket.tasks_completed == 2


class TestHourlyProfile:
    def test_get_hourly_bucket_creates_if_missing(self):
        profile = TemporalProfile(user_id="u1")
        bucket = profile.get_hourly_bucket("monday", 9)
        assert bucket.day_of_week == "monday"
        assert bucket.hour == 9
        assert bucket.interaction_count == 0

    def test_hourly_heatmap_size(self):
        profile = TemporalProfile(user_id="u1")
        heatmap = profile.hourly_heatmap()
        assert len(heatmap) == 168  # 7 * 24

    def test_hourly_heatmap_contains_data(self):
        profile = TemporalProfile(user_id="u1")
        bucket = profile.get_hourly_bucket("wednesday", 15)
        bucket.record(InteractionRecord(energy_level=0.9))

        heatmap = profile.hourly_heatmap()
        wed_15 = [c for c in heatmap if c["day_of_week"] == "wednesday" and c["hour"] == 15]
        assert len(wed_15) == 1
        assert wed_15[0]["interaction_count"] == 1
        assert wed_15[0]["avg_energy"] == 0.9

    def test_peak_hours_ranking(self):
        profile = TemporalProfile(user_id="u1")
        # High energy at 9am
        h9 = profile.get_hourly_bucket("monday", 9)
        for _ in range(3):
            h9.record(InteractionRecord(energy_level=0.9))
        # Low energy at 3pm
        h15 = profile.get_hourly_bucket("monday", 15)
        for _ in range(3):
            h15.record(InteractionRecord(energy_level=0.2))

        peaks = profile.peak_hours(top_n=2)
        assert len(peaks) == 2
        assert peaks[0].hour == 9
        assert peaks[0].score > peaks[1].score

    def test_peak_hours_filtered_by_day(self):
        profile = TemporalProfile(user_id="u1")
        for _ in range(3):
            profile.get_hourly_bucket("monday", 9).record(
                InteractionRecord(energy_level=0.9)
            )
            profile.get_hourly_bucket("friday", 14).record(
                InteractionRecord(energy_level=0.8)
            )

        mon_peaks = profile.peak_hours(day="monday")
        assert all(s.day_of_week == "monday" for s in mon_peaks)

    def test_peak_hours_min_threshold(self):
        profile = TemporalProfile(user_id="u1")
        # Only 1 interaction — below threshold
        profile.get_hourly_bucket("monday", 9).record(
            InteractionRecord(energy_level=0.9)
        )
        assert profile.peak_hours() == []

    def test_energy_by_hour(self):
        profile = TemporalProfile(user_id="u1")
        for _ in range(3):
            profile.get_hourly_bucket("monday", 9).record(
                InteractionRecord(energy_level=0.8)
            )
            profile.get_hourly_bucket("monday", 14).record(
                InteractionRecord(energy_level=0.4)
            )

        energy = profile.energy_by_hour(day="monday")
        assert len(energy) == 2
        h9 = [e for e in energy if e["hour"] == 9][0]
        h14 = [e for e in energy if e["hour"] == 14][0]
        assert h9["avg_energy"] == 0.8
        assert h14["avg_energy"] == 0.4

    def test_energy_by_hour_all_days(self):
        profile = TemporalProfile(user_id="u1")
        profile.get_hourly_bucket("monday", 9).record(
            InteractionRecord(energy_level=0.8)
        )
        profile.get_hourly_bucket("tuesday", 9).record(
            InteractionRecord(energy_level=0.6)
        )
        # No day filter -> aggregates both
        energy = profile.energy_by_hour()
        h9 = [e for e in energy if e["hour"] == 9][0]
        assert abs(h9["avg_energy"] - 0.7) < 0.001
        assert h9["interaction_count"] == 2

    def test_to_dict_includes_hourly(self):
        profile = TemporalProfile(user_id="u1")
        profile.get_hourly_bucket("monday", 10).record(
            InteractionRecord(energy_level=0.7)
        )
        d = profile.to_dict()
        assert "hourly_buckets" in d
        assert len(d["hourly_buckets"]) == 1


class TestHourlyStoreIntegration:
    @pytest.fixture
    def store(self):
        return TemporalActivityStore()

    @pytest.mark.asyncio
    async def test_store_populates_hourly_buckets(self, store):
        await store.store_interaction(InteractionRecord(
            user_id="u1", day_of_week="monday", time_of_day="morning",
            hour=9, energy_level=0.8,
        ))
        profile = await store.get_profile("u1")
        hb = profile.get_hourly_bucket("monday", 9)
        assert hb.interaction_count == 1
        assert hb.avg_energy == 0.8

    @pytest.mark.asyncio
    async def test_incremental_hourly_update(self, store):
        await store.store_interaction(InteractionRecord(
            user_id="u1", day_of_week="monday", time_of_day="morning",
            hour=9, energy_level=0.6,
        ))
        # Build profile cache
        profile = await store.get_profile("u1")
        assert profile.get_hourly_bucket("monday", 9).interaction_count == 1

        # Add another — should incrementally update
        await store.store_interaction(InteractionRecord(
            user_id="u1", day_of_week="monday", time_of_day="morning",
            hour=9, energy_level=0.8,
        ))
        assert profile.get_hourly_bucket("monday", 9).interaction_count == 2

    @pytest.mark.asyncio
    async def test_filter_by_hour(self, store):
        await store.store_interaction(InteractionRecord(
            user_id="u1", hour=9,
        ))
        await store.store_interaction(InteractionRecord(
            user_id="u1", hour=14,
        ))
        h9 = await store.get_interactions("u1", hour=9)
        assert len(h9) == 1
        h14 = await store.get_interactions("u1", hour=14)
        assert len(h14) == 1


class TestHourlyServiceIntegration:
    @pytest.fixture
    def service(self):
        return TemporalActivityService()

    @pytest.mark.asyncio
    async def test_get_hourly_pattern(self, service):
        for _ in range(3):
            await service.record_from_episode(make_episode(
                timestamp=datetime(2026, 3, 9, 9, 0, tzinfo=timezone.utc),
                arousal=0.8,
            ))
            await service.record_from_episode(make_episode(
                timestamp=datetime(2026, 3, 9, 21, 0, tzinfo=timezone.utc),
                arousal=0.2,
            ))

        pattern = await service.get_hourly_pattern("user-1")
        assert pattern["total_interactions"] == 6
        assert len(pattern["hourly_heatmap"]) == 168

        # Peak hours should include hour 9
        peaks = pattern["peak_hours"]
        if peaks:
            assert peaks[0]["energy"] > 0.5

    @pytest.mark.asyncio
    async def test_hourly_buckets_match_episodes(self, service):
        """Verify episode timestamps correctly populate hourly buckets."""
        ts = datetime(2026, 3, 11, 14, 30, 0, tzinfo=timezone.utc)  # Wed 14:30
        await service.record_from_episode(make_episode(timestamp=ts))

        profile = await service.get_temporal_profile("user-1")
        hb = profile.get_hourly_bucket("wednesday", 14)
        assert hb.interaction_count == 1

    @pytest.mark.asyncio
    async def test_get_interactions_filter_by_hour(self, service):
        await service.record_from_episode(make_episode(
            timestamp=datetime(2026, 3, 9, 9, 0, tzinfo=timezone.utc),
        ))
        await service.record_from_episode(make_episode(
            timestamp=datetime(2026, 3, 9, 14, 0, tzinfo=timezone.utc),
        ))
        h9 = await service.get_interactions("user-1", hour=9)
        assert len(h9) == 1
        h14 = await service.get_interactions("user-1", hour=14)
        assert len(h14) == 1


class TestEdgeCases:
    def test_temporal_slot_repr(self):
        slot = TemporalSlot("monday", "morning", 0.85)
        assert "monday" in repr(slot)
        assert "morning" in repr(slot)

    def test_temporal_slot_repr_with_hour(self):
        slot = TemporalSlot("monday", "morning", 0.85, hour=9)
        assert "h09" in repr(slot)

    @pytest.mark.asyncio
    async def test_empty_user_profile(self):
        service = TemporalActivityService()
        profile = await service.get_temporal_profile("nonexistent")
        assert profile.total_interactions == 0
        assert profile.best_slots_for_focus() == []

    @pytest.mark.asyncio
    async def test_limit_on_interactions(self):
        store = TemporalActivityStore()
        for i in range(10):
            await store.store_interaction(InteractionRecord(user_id="u1"))
        records = await store.get_interactions("u1", limit=3)
        assert len(records) == 3

    def test_interaction_record_to_dict_complete(self):
        record = InteractionRecord(
            user_id="u1",
            energy_level=0.7,
            valence=-0.3,
            task_completed=True,
            intent="task",
            primary_emotion="anger",
            emotion_intensity=0.9,
            word_count=15,
            modality="voice",
            episode_id="ep-123",
        )
        d = record.to_dict()
        assert d["user_id"] == "u1"
        assert d["energy_level"] == 0.7
        assert d["valence"] == -0.3
        assert d["task_completed"] is True
        assert d["episode_id"] == "ep-123"
