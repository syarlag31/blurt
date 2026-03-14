"""Data models for the task surfacing engine.

Defines TaskItem (what to surface), SurfacingContext (user state), and
supporting models for behavioral learning and calendar awareness.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from enum import Enum
from typing import Any


class TimePreference(str, Enum):
    """Learned time-of-day preference buckets."""

    EARLY_MORNING = "early_morning"   # 05:00–08:00
    MORNING = "morning"               # 08:00–12:00
    AFTERNOON = "afternoon"           # 12:00–17:00
    EVENING = "evening"               # 17:00–21:00
    NIGHT = "night"                   # 21:00–05:00


def time_to_preference(t: time) -> TimePreference:
    """Map a wall-clock time to its TimePreference bucket."""
    hour = t.hour
    if 5 <= hour < 8:
        return TimePreference.EARLY_MORNING
    elif 8 <= hour < 12:
        return TimePreference.MORNING
    elif 12 <= hour < 17:
        return TimePreference.AFTERNOON
    elif 17 <= hour < 21:
        return TimePreference.EVENING
    else:
        return TimePreference.NIGHT


@dataclass
class CalendarSlot:
    """A time block from the user's calendar (busy or free)."""

    start: datetime
    end: datetime
    is_busy: bool = True
    title: str = ""  # redacted/encrypted — only used for entity matching


@dataclass
class TaskItem:
    """A surfaceable task/reminder/event from the knowledge graph.

    Represents something that *could* be shown to the user.  The surfacing
    engine scores it against the current context to decide *whether* and
    *when* to show it.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    intent: str = "task"  # task | event | reminder

    # Entities linked to this task (people, projects, places, etc.)
    entity_ids: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)

    # Temporal hints
    due_at: datetime | None = None
    preferred_time: TimePreference | None = None
    estimated_minutes: int | None = None

    # Difficulty / cognitive load (0.0 = trivial, 1.0 = deep work)
    cognitive_load: float = 0.5

    # Category tags inferred by the classifier
    tags: list[str] = field(default_factory=list)

    # How many times this task has been surfaced without action
    surface_count: int = 0

    # Completion state
    completed: bool = False

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehavioralProfile:
    """Learned behavioral patterns for the user.

    Built from historical interaction data — which times they're productive,
    what kind of tasks they complete in various moods, etc.
    """

    # Completion rate by time-of-day bucket (0-1)
    completion_by_time: dict[str, float] = field(default_factory=dict)

    # Completion rate by cognitive load bucket (low/med/high → 0-1)
    completion_by_load: dict[str, float] = field(default_factory=dict)

    # Completion rate by tag (project names, categories → 0-1)
    completion_by_tag: dict[str, float] = field(default_factory=dict)

    # Average tasks completed per day (for load-balancing)
    avg_daily_completions: float = 3.0

    # Tasks completed today so far
    tasks_completed_today: int = 0

    # Historical preferred working hours (hour → relative frequency 0-1)
    hourly_activity: dict[int, float] = field(default_factory=dict)

    # Recent interaction pattern — hours since last blurt
    hours_since_last_interaction: float = 0.0


@dataclass
class SurfacingContext:
    """Current user context snapshot used for scoring.

    Captures the user's mood, energy, time, calendar, and active entities
    so each scoring function can evaluate tasks against real-time state.
    """

    # Current time in user's local timezone
    current_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Mood: valence from emotion detection (-1.0 negative to 1.0 positive)
    mood_valence: float = 0.0

    # Arousal / energy from emotion detection (0.0 calm to 1.0 activated)
    energy_level: float = 0.5

    # Currently active / recently mentioned entity IDs
    active_entity_ids: list[str] = field(default_factory=list)

    # Calendar slots for the next few hours
    calendar_slots: list[CalendarSlot] = field(default_factory=list)

    # Learned behavioral profile
    behavioral: BehavioralProfile = field(default_factory=BehavioralProfile)

    # User's timezone offset from UTC in hours (for time-of-day scoring)
    timezone_offset_hours: float = 0.0


@dataclass
class ScoredTask:
    """A task with individual dimension scores and a composite score."""

    task: TaskItem
    mood_score: float = 0.0
    energy_score: float = 0.0
    time_of_day_score: float = 0.0
    calendar_score: float = 0.0
    entity_relevance_score: float = 0.0
    behavioral_score: float = 0.0
    composite_score: float = 0.0
