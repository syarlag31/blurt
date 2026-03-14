"""Temporal activity aggregation service — collects and buckets user interaction data.

Aggregates timestamped user interaction data (energy levels, productivity metrics,
emotion signals) bucketed by day-of-week and time-of-day. This powers Blurt's
behavioral learning: understanding when users are most productive, what their
emotional patterns look like across the week, and how energy levels fluctuate.

Design principles:
- Append-only aggregation: new data always adds to existing buckets, never replaces
- Two-axis bucketing: day-of-week (7 buckets) x time-of-day (4 buckets) = 28 cells
- Incrementally compounding: each interaction enriches the temporal profile
- Anti-shame: data is used to surface tasks at optimal times, never to guilt users
- Privacy-safe: all data stays per-user, no cross-user aggregation

Time-of-day buckets:
    morning:   06:00–11:59
    afternoon: 12:00–16:59
    evening:   17:00–20:59
    night:     21:00–05:59

Usage::

    service = TemporalActivityService()
    await service.record_interaction(interaction)
    profile = await service.get_temporal_profile("user-1")
    best_slot = profile.best_slot_for_focus()
"""

from __future__ import annotations

import logging
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums / Constants
# ---------------------------------------------------------------------------

DAYS_OF_WEEK = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

class TimeOfDay(str, Enum):
    """Time-of-day buckets for temporal aggregation."""
    MORNING = "morning"      # 06:00–11:59
    AFTERNOON = "afternoon"  # 12:00–16:59
    EVENING = "evening"      # 17:00–20:59
    NIGHT = "night"          # 21:00–05:59


def hour_to_time_of_day(hour: int) -> TimeOfDay:
    """Map an hour (0-23) to a TimeOfDay bucket."""
    if 6 <= hour < 12:
        return TimeOfDay.MORNING
    elif 12 <= hour < 17:
        return TimeOfDay.AFTERNOON
    elif 17 <= hour < 21:
        return TimeOfDay.EVENING
    else:
        return TimeOfDay.NIGHT


def weekday_to_name(weekday: int) -> str:
    """Map Python weekday (0=Monday) to day name."""
    return DAYS_OF_WEEK[weekday]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class InteractionRecord:
    """A single timestamped user interaction with activity signals.

    Captures the raw signals from one episode that feed into temporal
    aggregation. Created from Episode data after capture pipeline runs.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Temporal bucket keys (derived from timestamp)
    day_of_week: str = "monday"
    time_of_day: str = "morning"
    hour: int = 9  # 0-23 for finer-grained analysis

    # Energy / arousal signals (from emotion detection)
    energy_level: float = 0.5       # 0.0 (depleted) to 1.0 (energized), derived from arousal
    valence: float = 0.0            # -1.0 to 1.0, emotional valence

    # Productivity signals
    task_created: bool = False
    task_completed: bool = False
    task_skipped: bool = False
    task_dismissed: bool = False
    intent: str = "journal"         # classified intent of the interaction

    # Emotion signals
    primary_emotion: str = "trust"  # Plutchik primary emotion
    emotion_intensity: float = 0.0  # 0.0 to 1.0

    # Engagement signals
    word_count: int = 0             # proxy for engagement depth
    modality: str = "voice"         # voice or text

    # Source tracking
    episode_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "day_of_week": self.day_of_week,
            "time_of_day": self.time_of_day,
            "hour": self.hour,
            "energy_level": self.energy_level,
            "valence": self.valence,
            "task_created": self.task_created,
            "task_completed": self.task_completed,
            "task_skipped": self.task_skipped,
            "task_dismissed": self.task_dismissed,
            "intent": self.intent,
            "primary_emotion": self.primary_emotion,
            "emotion_intensity": self.emotion_intensity,
            "word_count": self.word_count,
            "modality": self.modality,
            "episode_id": self.episode_id,
        }


@dataclass
class TemporalBucket:
    """Aggregated metrics for a single day-of-week x time-of-day cell.

    Stores running statistics so patterns can be detected without
    re-scanning all historical episodes.
    """

    day_of_week: str = "monday"
    time_of_day: str = "morning"
    interaction_count: int = 0

    # Energy aggregates
    energy_samples: list[float] = field(default_factory=list)

    # Valence aggregates
    valence_samples: list[float] = field(default_factory=list)

    # Productivity counts
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_skipped: int = 0
    tasks_dismissed: int = 0

    # Intent distribution within this bucket
    intent_counts: dict[str, int] = field(default_factory=dict)

    # Emotion distribution within this bucket
    emotion_counts: dict[str, int] = field(default_factory=dict)
    emotion_intensity_samples: list[float] = field(default_factory=list)

    # Engagement
    word_count_samples: list[int] = field(default_factory=list)
    voice_count: int = 0
    text_count: int = 0

    @property
    def avg_energy(self) -> float:
        """Average energy level in this bucket."""
        if not self.energy_samples:
            return 0.5
        return statistics.mean(self.energy_samples)

    @property
    def avg_valence(self) -> float:
        """Average emotional valence in this bucket."""
        if not self.valence_samples:
            return 0.0
        return statistics.mean(self.valence_samples)

    @property
    def avg_emotion_intensity(self) -> float:
        """Average emotion intensity in this bucket."""
        if not self.emotion_intensity_samples:
            return 0.0
        return statistics.mean(self.emotion_intensity_samples)

    @property
    def avg_word_count(self) -> float:
        """Average word count per interaction in this bucket."""
        if not self.word_count_samples:
            return 0.0
        return statistics.mean(self.word_count_samples)

    @property
    def completion_rate(self) -> float:
        """Task completion rate in this bucket."""
        total_actions = self.tasks_completed + self.tasks_skipped + self.tasks_dismissed
        if total_actions == 0:
            return 0.0
        return self.tasks_completed / total_actions

    @property
    def dominant_emotion(self) -> str | None:
        """Most common emotion in this bucket."""
        if not self.emotion_counts:
            return None
        return max(self.emotion_counts, key=self.emotion_counts.get)  # type: ignore[arg-type]

    @property
    def dominant_intent(self) -> str | None:
        """Most common intent in this bucket."""
        if not self.intent_counts:
            return None
        return max(self.intent_counts, key=self.intent_counts.get)  # type: ignore[arg-type]

    @property
    def productivity_score(self) -> float:
        """Composite productivity score (0.0–1.0) combining energy, completion, engagement.

        Weighted combination:
        - 40% energy level
        - 30% task completion rate
        - 30% engagement (normalized word count)
        """
        energy_component = self.avg_energy * 0.4
        completion_component = self.completion_rate * 0.3
        # Normalize word count: assume 50 words is "high engagement"
        engagement_norm = min(self.avg_word_count / 50.0, 1.0) if self.avg_word_count > 0 else 0.0
        engagement_component = engagement_norm * 0.3
        return energy_component + completion_component + engagement_component

    def record(self, interaction: InteractionRecord) -> None:
        """Add an interaction to this bucket's aggregates."""
        self.interaction_count += 1
        self.energy_samples.append(interaction.energy_level)
        self.valence_samples.append(interaction.valence)

        if interaction.task_created:
            self.tasks_created += 1
        if interaction.task_completed:
            self.tasks_completed += 1
        if interaction.task_skipped:
            self.tasks_skipped += 1
        if interaction.task_dismissed:
            self.tasks_dismissed += 1

        # Intent tracking
        self.intent_counts[interaction.intent] = (
            self.intent_counts.get(interaction.intent, 0) + 1
        )

        # Emotion tracking
        self.emotion_counts[interaction.primary_emotion] = (
            self.emotion_counts.get(interaction.primary_emotion, 0) + 1
        )
        self.emotion_intensity_samples.append(interaction.emotion_intensity)

        # Engagement tracking
        self.word_count_samples.append(interaction.word_count)
        if interaction.modality == "voice":
            self.voice_count += 1
        else:
            self.text_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_of_week": self.day_of_week,
            "time_of_day": self.time_of_day,
            "interaction_count": self.interaction_count,
            "avg_energy": round(self.avg_energy, 3),
            "avg_valence": round(self.avg_valence, 3),
            "avg_emotion_intensity": round(self.avg_emotion_intensity, 3),
            "avg_word_count": round(self.avg_word_count, 1),
            "completion_rate": round(self.completion_rate, 3),
            "productivity_score": round(self.productivity_score, 3),
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "tasks_skipped": self.tasks_skipped,
            "tasks_dismissed": self.tasks_dismissed,
            "dominant_emotion": self.dominant_emotion,
            "dominant_intent": self.dominant_intent,
            "intent_counts": self.intent_counts,
            "emotion_counts": self.emotion_counts,
            "voice_count": self.voice_count,
            "text_count": self.text_count,
        }


@dataclass
class HourlyBucket:
    """Aggregated metrics for a single day-of-week x hour-of-day cell.

    Provides finer-grained (24-hour) bucketing compared to TemporalBucket's
    4-slot time-of-day bucketing. Used when hour-level precision is needed
    for pattern detection and surfacing decisions.
    """

    day_of_week: str = "monday"
    hour: int = 9  # 0-23
    interaction_count: int = 0

    # Core signal samples
    energy_samples: list[float] = field(default_factory=list)
    valence_samples: list[float] = field(default_factory=list)
    emotion_intensity_samples: list[float] = field(default_factory=list)

    # Productivity counts
    tasks_created: int = 0
    tasks_completed: int = 0

    # Emotion tracking
    emotion_counts: dict[str, int] = field(default_factory=dict)

    @property
    def avg_energy(self) -> float:
        if not self.energy_samples:
            return 0.5
        return statistics.mean(self.energy_samples)

    @property
    def avg_valence(self) -> float:
        if not self.valence_samples:
            return 0.0
        return statistics.mean(self.valence_samples)

    @property
    def avg_emotion_intensity(self) -> float:
        if not self.emotion_intensity_samples:
            return 0.0
        return statistics.mean(self.emotion_intensity_samples)

    @property
    def dominant_emotion(self) -> str | None:
        if not self.emotion_counts:
            return None
        return max(self.emotion_counts, key=self.emotion_counts.get)  # type: ignore[arg-type]

    def record(self, interaction: InteractionRecord) -> None:
        """Add an interaction to this hourly bucket."""
        self.interaction_count += 1
        self.energy_samples.append(interaction.energy_level)
        self.valence_samples.append(interaction.valence)
        self.emotion_intensity_samples.append(interaction.emotion_intensity)

        if interaction.task_created:
            self.tasks_created += 1
        if interaction.task_completed:
            self.tasks_completed += 1

        self.emotion_counts[interaction.primary_emotion] = (
            self.emotion_counts.get(interaction.primary_emotion, 0) + 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "day_of_week": self.day_of_week,
            "hour": self.hour,
            "interaction_count": self.interaction_count,
            "avg_energy": round(self.avg_energy, 3),
            "avg_valence": round(self.avg_valence, 3),
            "avg_emotion_intensity": round(self.avg_emotion_intensity, 3),
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "dominant_emotion": self.dominant_emotion,
        }


@dataclass
class TemporalSlot:
    """A (day_of_week, time_of_day) pair with a score for ranking."""
    day_of_week: str
    time_of_day: str
    score: float
    hour: int | None = None  # Optional hour for hourly-granularity slots

    def __repr__(self) -> str:
        if self.hour is not None:
            return (
                f"TemporalSlot({self.day_of_week} h{self.hour:02d}: {self.score:.3f})"
            )
        return f"TemporalSlot({self.day_of_week} {self.time_of_day}: {self.score:.3f})"


@dataclass
class TemporalProfile:
    """Complete temporal activity profile for a user.

    Contains all 28 buckets (7 days x 4 time slots) with aggregated
    activity data. Provides query methods for surfacing insights.
    """

    user_id: str = ""
    buckets: dict[str, TemporalBucket] = field(default_factory=dict)
    hourly_buckets: dict[str, HourlyBucket] = field(default_factory=dict)
    total_interactions: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def _bucket_key(day: str, tod: str) -> str:
        """Create a consistent key for a (day, time_of_day) pair."""
        return f"{day}:{tod}"

    @staticmethod
    def _hourly_key(day: str, hour: int) -> str:
        """Create a consistent key for a (day, hour) pair."""
        return f"{day}:h{hour:02d}"

    def get_bucket(self, day: str, tod: str) -> TemporalBucket:
        """Get or create a bucket for the given day/time slot."""
        key = self._bucket_key(day, tod)
        if key not in self.buckets:
            self.buckets[key] = TemporalBucket(day_of_week=day, time_of_day=tod)
        return self.buckets[key]

    def get_hourly_bucket(self, day: str, hour: int) -> HourlyBucket:
        """Get or create a bucket for the given day/hour slot."""
        key = self._hourly_key(day, hour)
        if key not in self.hourly_buckets:
            self.hourly_buckets[key] = HourlyBucket(day_of_week=day, hour=hour)
        return self.hourly_buckets[key]

    def all_buckets(self) -> list[TemporalBucket]:
        """Return all non-empty buckets."""
        return [b for b in self.buckets.values() if b.interaction_count > 0]

    def best_slots_for_focus(self, top_n: int = 3) -> list[TemporalSlot]:
        """Find the best time slots for focused work, ranked by productivity score.

        Uses a minimum interaction threshold to avoid suggesting slots
        with too little data.
        """
        scored: list[TemporalSlot] = []
        for bucket in self.all_buckets():
            if bucket.interaction_count < 2:
                continue  # Need enough data points
            scored.append(TemporalSlot(
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                score=bucket.productivity_score,
            ))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_n]

    def best_slots_for_energy(self, top_n: int = 3) -> list[TemporalSlot]:
        """Find time slots with highest average energy levels."""
        scored: list[TemporalSlot] = []
        for bucket in self.all_buckets():
            if bucket.interaction_count < 2:
                continue
            scored.append(TemporalSlot(
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                score=bucket.avg_energy,
            ))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_n]

    def lowest_energy_slots(self, top_n: int = 3) -> list[TemporalSlot]:
        """Find time slots with lowest average energy (for lighter tasks)."""
        scored: list[TemporalSlot] = []
        for bucket in self.all_buckets():
            if bucket.interaction_count < 2:
                continue
            scored.append(TemporalSlot(
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                score=bucket.avg_energy,
            ))
        scored.sort(key=lambda s: s.score)  # ascending = lowest first
        return scored[:top_n]

    def mood_for_slot(self, day: str, tod: str) -> dict[str, Any]:
        """Get mood/emotion summary for a specific time slot."""
        bucket = self.get_bucket(day, tod)
        return {
            "dominant_emotion": bucket.dominant_emotion,
            "avg_valence": round(bucket.avg_valence, 3),
            "avg_energy": round(bucket.avg_energy, 3),
            "avg_emotion_intensity": round(bucket.avg_emotion_intensity, 3),
            "emotion_distribution": dict(bucket.emotion_counts),
            "interaction_count": bucket.interaction_count,
        }

    def day_summary(self, day: str) -> dict[str, Any]:
        """Get aggregated summary for an entire day of the week."""
        day_buckets = [
            self.get_bucket(day, tod.value)
            for tod in TimeOfDay
        ]
        active_buckets = [b for b in day_buckets if b.interaction_count > 0]
        if not active_buckets:
            return {
                "day": day,
                "total_interactions": 0,
                "avg_energy": 0.5,
                "avg_valence": 0.0,
                "most_active_time": None,
            }

        total = sum(b.interaction_count for b in active_buckets)
        all_energy = []
        all_valence = []
        for b in active_buckets:
            all_energy.extend(b.energy_samples)
            all_valence.extend(b.valence_samples)

        most_active = max(active_buckets, key=lambda b: b.interaction_count)

        return {
            "day": day,
            "total_interactions": total,
            "avg_energy": round(statistics.mean(all_energy), 3) if all_energy else 0.5,
            "avg_valence": round(statistics.mean(all_valence), 3) if all_valence else 0.0,
            "most_active_time": most_active.time_of_day,
        }

    def weekly_heatmap(self) -> list[dict[str, Any]]:
        """Generate a heatmap of activity across the week.

        Returns a list of {day, time_of_day, interaction_count, energy, valence}
        for all 28 slots.
        """
        heatmap = []
        for day in DAYS_OF_WEEK:
            for tod in TimeOfDay:
                bucket = self.get_bucket(day, tod.value)
                heatmap.append({
                    "day_of_week": day,
                    "time_of_day": tod.value,
                    "interaction_count": bucket.interaction_count,
                    "avg_energy": round(bucket.avg_energy, 3),
                    "avg_valence": round(bucket.avg_valence, 3),
                    "productivity_score": round(bucket.productivity_score, 3),
                })
        return heatmap

    def all_hourly_buckets(self) -> list[HourlyBucket]:
        """Return all non-empty hourly buckets."""
        return [b for b in self.hourly_buckets.values() if b.interaction_count > 0]

    def hourly_heatmap(self) -> list[dict[str, Any]]:
        """Generate a fine-grained heatmap across the week by hour.

        Returns a list of {day, hour, interaction_count, energy, valence, ...}
        for all 168 slots (7 days x 24 hours).
        """
        heatmap = []
        for day in DAYS_OF_WEEK:
            for hour in range(24):
                bucket = self.get_hourly_bucket(day, hour)
                heatmap.append({
                    "day_of_week": day,
                    "hour": hour,
                    "interaction_count": bucket.interaction_count,
                    "avg_energy": round(bucket.avg_energy, 3),
                    "avg_valence": round(bucket.avg_valence, 3),
                    "avg_emotion_intensity": round(bucket.avg_emotion_intensity, 3),
                    "dominant_emotion": bucket.dominant_emotion,
                })
        return heatmap

    def peak_hours(self, day: str | None = None, top_n: int = 5) -> list[TemporalSlot]:
        """Find the hours with highest energy, optionally filtered by day.

        Returns TemporalSlot with the hour field set for granular identification.
        """
        scored: list[TemporalSlot] = []
        for hb in self.all_hourly_buckets():
            if day and hb.day_of_week != day:
                continue
            if hb.interaction_count < 2:
                continue
            scored.append(TemporalSlot(
                day_of_week=hb.day_of_week,
                time_of_day=hour_to_time_of_day(hb.hour).value,
                score=hb.avg_energy,
                hour=hb.hour,
            ))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_n]

    def energy_by_hour(self, day: str | None = None) -> list[dict[str, Any]]:
        """Get average energy for each hour (0-23), optionally filtered by day.

        Returns sorted list of {hour, avg_energy, interaction_count} entries
        that have data, useful for charting energy curves.
        """
        # Aggregate across all days if no day specified
        hour_data: dict[int, list[float]] = defaultdict(list)
        hour_counts: dict[int, int] = defaultdict(int)
        for hb in self.all_hourly_buckets():
            if day and hb.day_of_week != day:
                continue
            hour_data[hb.hour].extend(hb.energy_samples)
            hour_counts[hb.hour] += hb.interaction_count

        result = []
        for h in range(24):
            if hour_counts.get(h, 0) > 0:
                result.append({
                    "hour": h,
                    "avg_energy": round(statistics.mean(hour_data[h]), 3),
                    "interaction_count": hour_counts[h],
                })
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "total_interactions": self.total_interactions,
            "last_updated": self.last_updated.isoformat(),
            "buckets": {k: v.to_dict() for k, v in self.buckets.items() if v.interaction_count > 0},
            "hourly_buckets": {
                k: v.to_dict()
                for k, v in self.hourly_buckets.items()
                if v.interaction_count > 0
            },
        }


# ---------------------------------------------------------------------------
# Episode-to-InteractionRecord converter
# ---------------------------------------------------------------------------


def episode_to_interaction(episode: Any) -> InteractionRecord:
    """Convert an Episode to an InteractionRecord for temporal aggregation.

    Extracts energy, productivity, and emotion signals from episode data.
    Uses duck typing to avoid tight coupling to Episode class.
    """
    # Determine temporal bucket from episode
    day = getattr(getattr(episode, "context", None), "day_of_week", "monday")
    tod = getattr(getattr(episode, "context", None), "time_of_day", "morning")

    # Map time_of_day string to hour estimate for finer analysis
    tod_hour_map = {"morning": 9, "afternoon": 14, "evening": 19, "night": 22}
    hour = tod_hour_map.get(tod, 9)

    # If timestamp available, use actual hour
    ts = getattr(episode, "timestamp", None)
    if ts and isinstance(ts, datetime):
        hour = ts.hour
        # Also derive more accurate day/tod from timestamp
        day = weekday_to_name(ts.weekday())
        tod = hour_to_time_of_day(hour).value

    # Energy from arousal (emotion detection)
    emotion = getattr(episode, "emotion", None)
    energy = getattr(emotion, "arousal", 0.5) if emotion else 0.5
    valence = getattr(emotion, "valence", 0.0) if emotion else 0.0
    primary_emotion = getattr(emotion, "primary", "trust") if emotion else "trust"
    emotion_intensity = getattr(emotion, "intensity", 0.0) if emotion else 0.0

    # Productivity from behavioral signals
    behavioral = getattr(episode, "behavioral_signal", None)
    behavioral_val = behavioral.value if behavioral else "none"

    # Word count as engagement proxy
    raw_text = getattr(episode, "raw_text", "")
    word_count = len(raw_text.split()) if raw_text else 0

    # Modality
    modality_obj = getattr(episode, "modality", None)
    modality = modality_obj.value if modality_obj else "voice"

    # Intent - check if it's a task-creating intent
    intent = getattr(episode, "intent", "journal")
    task_creating_intents = {"task", "reminder", "event"}

    return InteractionRecord(
        user_id=getattr(episode, "user_id", ""),
        timestamp=ts or datetime.now(timezone.utc),
        day_of_week=day,
        time_of_day=tod,
        hour=hour,
        energy_level=max(0.0, min(1.0, energy)),
        valence=max(-1.0, min(1.0, valence)),
        task_created=intent in task_creating_intents,
        task_completed=behavioral_val == "completed",
        task_skipped=behavioral_val == "skipped",
        task_dismissed=behavioral_val == "dismissed",
        intent=intent,
        primary_emotion=primary_emotion,
        emotion_intensity=max(0.0, min(1.0, emotion_intensity)),
        word_count=word_count,
        modality=modality,
        episode_id=getattr(episode, "id", None),
    )


# ---------------------------------------------------------------------------
# Storage interface + in-memory implementation
# ---------------------------------------------------------------------------


class TemporalActivityStore:
    """In-memory store for temporal activity data.

    Production would use SQLite/Postgres with indexed columns on
    (user_id, day_of_week, time_of_day).
    """

    def __init__(self) -> None:
        # user_id -> list of interaction records
        self._interactions: dict[str, list[InteractionRecord]] = defaultdict(list)
        # user_id -> TemporalProfile (cached aggregation)
        self._profiles: dict[str, TemporalProfile] = {}

    async def store_interaction(self, record: InteractionRecord) -> InteractionRecord:
        """Store a single interaction record."""
        self._interactions[record.user_id].append(record)
        # Incrementally update cached profile if present
        if record.user_id in self._profiles:
            profile = self._profiles[record.user_id]
            bucket = profile.get_bucket(record.day_of_week, record.time_of_day)
            bucket.record(record)
            hourly = profile.get_hourly_bucket(record.day_of_week, record.hour)
            hourly.record(record)
            profile.total_interactions += 1
            profile.last_updated = datetime.now(timezone.utc)
        return record

    async def get_interactions(
        self,
        user_id: str,
        *,
        day_of_week: str | None = None,
        time_of_day: str | None = None,
        hour: int | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[InteractionRecord]:
        """Retrieve interaction records with optional filters."""
        records = self._interactions.get(user_id, [])
        if day_of_week:
            records = [r for r in records if r.day_of_week == day_of_week]
        if time_of_day:
            records = [r for r in records if r.time_of_day == time_of_day]
        if hour is not None:
            records = [r for r in records if r.hour == hour]
        if since:
            records = [r for r in records if r.timestamp >= since]
        # Sort newest first
        records = sorted(records, key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    async def get_profile(self, user_id: str) -> TemporalProfile:
        """Get or build the temporal profile for a user."""
        if user_id not in self._profiles:
            self._profiles[user_id] = await self._build_profile(user_id)
        return self._profiles[user_id]

    async def _build_profile(self, user_id: str) -> TemporalProfile:
        """Build a temporal profile by aggregating all stored interactions."""
        profile = TemporalProfile(user_id=user_id)
        for record in self._interactions.get(user_id, []):
            bucket = profile.get_bucket(record.day_of_week, record.time_of_day)
            bucket.record(record)
            hourly = profile.get_hourly_bucket(record.day_of_week, record.hour)
            hourly.record(record)
            profile.total_interactions += 1
        profile.last_updated = datetime.now(timezone.utc)
        return profile

    async def interaction_count(self, user_id: str) -> int:
        """Total interaction count for a user."""
        return len(self._interactions.get(user_id, []))

    async def clear_user(self, user_id: str) -> None:
        """Clear all data for a user (for testing / account deletion)."""
        self._interactions.pop(user_id, None)
        self._profiles.pop(user_id, None)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class TemporalActivityService:
    """Service that collects, stores, and queries temporal activity data.

    Integrates with the capture pipeline to automatically record interaction
    data from every episode, bucketed by day-of-week and time-of-day.

    Usage::

        service = TemporalActivityService()

        # Record from an episode
        record = await service.record_from_episode(episode)

        # Get temporal profile
        profile = await service.get_temporal_profile("user-1")

        # Find best focus times
        best_slots = profile.best_slots_for_focus()
    """

    def __init__(self, store: TemporalActivityStore | None = None) -> None:
        self._store = store or TemporalActivityStore()

    @property
    def store(self) -> TemporalActivityStore:
        return self._store

    async def record_interaction(self, record: InteractionRecord) -> InteractionRecord:
        """Record a pre-built interaction."""
        stored = await self._store.store_interaction(record)
        logger.debug(
            "Recorded interaction %s for user %s: %s %s energy=%.2f",
            stored.id, stored.user_id, stored.day_of_week,
            stored.time_of_day, stored.energy_level,
        )
        return stored

    async def record_from_episode(self, episode: Any) -> InteractionRecord:
        """Extract interaction signals from an Episode and record them.

        This is the primary entry point, called after each capture pipeline
        completes. Converts episode data to an InteractionRecord and stores it.
        """
        record = episode_to_interaction(episode)
        return await self.record_interaction(record)

    async def get_temporal_profile(self, user_id: str) -> TemporalProfile:
        """Get the complete temporal activity profile for a user."""
        return await self._store.get_profile(user_id)

    async def get_interactions(
        self,
        user_id: str,
        *,
        day_of_week: str | None = None,
        time_of_day: str | None = None,
        hour: int | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[InteractionRecord]:
        """Query interaction records with filters."""
        return await self._store.get_interactions(
            user_id,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
            hour=hour,
            since=since,
            limit=limit,
        )

    async def get_hourly_pattern(self, user_id: str) -> dict[str, Any]:
        """Get hour-of-day activity patterns across the week.

        Returns a fine-grained (168-cell) heatmap of energy, valence, and
        emotion intensity bucketed by day-of-week and hour-of-day.
        """
        profile = await self.get_temporal_profile(user_id)
        return {
            "user_id": user_id,
            "hourly_heatmap": profile.hourly_heatmap(),
            "peak_hours": [
                {
                    "day": s.day_of_week,
                    "hour": s.hour,
                    "time_of_day": s.time_of_day,
                    "energy": round(s.score, 3),
                }
                for s in profile.peak_hours()
            ],
            "energy_by_hour": profile.energy_by_hour(),
            "total_interactions": profile.total_interactions,
        }

    async def get_energy_pattern(self, user_id: str) -> dict[str, Any]:
        """Get energy level patterns across the week.

        Returns a summary of average energy by day and time slot,
        useful for scheduling recommendations.
        """
        profile = await self.get_temporal_profile(user_id)
        return {
            "user_id": user_id,
            "weekly_heatmap": profile.weekly_heatmap(),
            "best_energy_slots": [
                {"day": s.day_of_week, "time": s.time_of_day, "energy": round(s.score, 3)}
                for s in profile.best_slots_for_energy()
            ],
            "lowest_energy_slots": [
                {"day": s.day_of_week, "time": s.time_of_day, "energy": round(s.score, 3)}
                for s in profile.lowest_energy_slots()
            ],
            "total_interactions": profile.total_interactions,
        }

    async def get_mood_pattern(self, user_id: str) -> dict[str, Any]:
        """Get emotion/mood patterns across the week."""
        profile = await self.get_temporal_profile(user_id)
        day_summaries = [profile.day_summary(day) for day in DAYS_OF_WEEK]
        return {
            "user_id": user_id,
            "day_summaries": day_summaries,
            "total_interactions": profile.total_interactions,
        }

    async def interaction_count(self, user_id: str) -> int:
        """Total recorded interactions for a user."""
        return await self._store.interaction_count(user_id)
