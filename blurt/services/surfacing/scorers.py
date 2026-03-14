"""Individual scoring functions for task surfacing.

Each function takes a TaskItem and SurfacingContext, returning a normalized
0.0–1.0 score for one signal dimension.  These are pure functions with no
side-effects, making them easy to test and compose.

Design principles:
- Anti-shame: scores never penalize the user; a low score just means "not
  the best moment" rather than "you failed".
- No-tasks-pending is valid: these scorers never inflate scores to force
  surfacing.
- All scores are soft signals, not hard gates.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from blurt.services.surfacing.models import (
    CalendarSlot,
    SurfacingContext,
    TaskItem,
    TimePreference,
    time_to_preference,
)

# ── 1. Mood Score ────────────────────────────────────────────────────


def score_mood(task: TaskItem, ctx: SurfacingContext) -> float:
    """Score task suitability for the user's current mood.

    Strategy:
    - Positive mood (high valence) → good for any task, slight boost
      for creative/complex work.
    - Neutral mood → baseline, no bias.
    - Negative mood (low valence) → favor low-cognitive-load tasks
      (quick wins) to avoid overwhelm.  Never zero — we don't block,
      we gently deprioritize.

    Returns:
        Float in [0.0, 1.0].
    """
    valence = ctx.mood_valence  # -1.0 to 1.0
    load = task.cognitive_load   # 0.0 to 1.0

    if valence >= 0:
        # Positive mood: slight preference for challenging work
        # High valence + high load → higher score
        # High valence + low load → still good, but less optimal
        base = 0.5 + 0.3 * valence  # 0.5–0.8
        load_bonus = 0.2 * valence * load  # 0.0–0.2
        return _clamp(base + load_bonus)
    else:
        # Negative mood: prefer easier tasks (quick wins)
        # More negative → stronger preference for low-load tasks
        abs_neg = abs(valence)
        # Base decreases with negativity
        base = 0.5 - 0.2 * abs_neg  # 0.3–0.5
        # Penalize high cognitive load when mood is low
        load_penalty = 0.25 * abs_neg * load  # 0.0–0.25
        # Anti-shame floor: never drop below 0.05
        return _clamp(max(0.05, base - load_penalty))


# ── 2. Energy Score ──────────────────────────────────────────────────


def score_energy(task: TaskItem, ctx: SurfacingContext) -> float:
    """Score task suitability for the user's current energy level.

    Strategy:
    - High energy → favor deep work / high cognitive load tasks.
    - Low energy → favor lightweight, routine tasks.
    - Match principle: score is highest when task load ≈ energy level.

    Returns:
        Float in [0.0, 1.0].
    """
    energy = ctx.energy_level  # 0.0 (calm/tired) to 1.0 (activated)
    load = task.cognitive_load  # 0.0 (trivial) to 1.0 (deep work)

    # Gaussian-style match: highest when |energy - load| is small
    # sigma controls how forgiving the mismatch is
    diff = abs(energy - load)
    sigma = 0.5
    match_score = math.exp(-(diff ** 2) / (2 * sigma ** 2))

    # Slight baseline so nothing scores zero
    return _clamp(0.1 + 0.9 * match_score)


# ── 3. Time-of-Day Score ────────────────────────────────────────────


def score_time_of_day(task: TaskItem, ctx: SurfacingContext) -> float:
    """Score task fitness for the current time of day.

    Strategy:
    - If task has a preferred_time, score based on match with current
      time bucket.
    - If task has a due_at within the next few hours, boost urgency
      (gentle, not guilt-inducing).
    - Otherwise use behavioral profile's hourly activity patterns.

    Returns:
        Float in [0.0, 1.0].
    """
    # Compute user-local time
    offset = timedelta(hours=ctx.timezone_offset_hours)
    local_time = (ctx.current_time + offset).time()
    current_bucket = time_to_preference(local_time)

    score = 0.5  # neutral baseline

    # 1) Preferred time match
    if task.preferred_time is not None:
        if task.preferred_time == current_bucket:
            score = 0.9
        elif _adjacent_buckets(task.preferred_time, current_bucket):
            score = 0.65
        else:
            score = 0.3

    # 2) Due date proximity boost (gentle urgency, NOT guilt)
    if task.due_at is not None:
        hours_until_due = (
            task.due_at - ctx.current_time
        ).total_seconds() / 3600.0

        if hours_until_due < 0:
            # Already past due — slight boost to surface but NO penalty language
            # Anti-shame: we just make it a bit more visible
            score = max(score, 0.7)
        elif hours_until_due <= 2:
            score = max(score, 0.85)
        elif hours_until_due <= 6:
            score = max(score, 0.7)
        elif hours_until_due <= 24:
            score = max(score, 0.55)

    # 3) Behavioral hourly activity match
    hourly = ctx.behavioral.hourly_activity
    if hourly:
        current_hour = local_time.hour
        activity = hourly.get(current_hour, 0.5)
        # Blend behavioral signal (30% weight)
        score = 0.7 * score + 0.3 * activity

    return _clamp(score)


# ── 4. Calendar Availability Score ──────────────────────────────────


def score_calendar_availability(
    task: TaskItem, ctx: SurfacingContext
) -> float:
    """Score based on whether the user has free time for this task.

    Strategy:
    - No calendar data → neutral (0.5), don't penalize.
    - Currently in a meeting → low score (not zero — they might
      glance at tasks between meetings).
    - Free block coming up that fits estimated duration → high score.
    - Back-to-back busy → lower score.

    Returns:
        Float in [0.0, 1.0].
    """
    slots = ctx.calendar_slots
    now = ctx.current_time

    if not slots:
        return 0.5  # No calendar data — neutral

    # Check if currently in a busy slot
    in_meeting = any(
        slot.is_busy and slot.start <= now < slot.end for slot in slots
    )
    if in_meeting:
        return 0.15  # Low but not zero

    # Find the next free window
    estimated = task.estimated_minutes or 30  # default 30 min
    free_minutes = _find_free_minutes(now, slots)

    if free_minutes is None:
        # No busy slots ahead — assume free
        return 0.8

    if free_minutes >= estimated:
        # Enough free time for this task
        return 0.9
    elif free_minutes >= estimated * 0.5:
        # Might squeeze it in
        return 0.6
    else:
        # Not enough time before next commitment
        return 0.3


# ── 5. Entity Relevance Score ───────────────────────────────────────


def score_entity_relevance(
    task: TaskItem, ctx: SurfacingContext
) -> float:
    """Score based on overlap between task entities and active context.

    Strategy:
    - Tasks mentioning entities the user is currently engaged with
      get a boost (contextual relevance).
    - No entity overlap → neutral baseline.
    - More overlapping entities → higher score (diminishing returns).

    Returns:
        Float in [0.0, 1.0].
    """
    if not task.entity_ids or not ctx.active_entity_ids:
        return 0.4  # No entity data — slightly below neutral

    active_set = set(ctx.active_entity_ids)
    task_set = set(task.entity_ids)

    overlap = len(active_set & task_set)
    total_task_entities = len(task_set)

    if overlap == 0:
        return 0.3  # No relevance — low but present

    # Overlap ratio with diminishing returns
    ratio = overlap / total_task_entities
    # Log scale for diminishing returns on multiple matches
    relevance = min(1.0, 0.5 + 0.5 * math.log1p(overlap) / math.log1p(3))

    # Blend ratio and absolute overlap
    return _clamp(0.4 * ratio + 0.6 * relevance)


# ── 6. Behavioral Score ─────────────────────────────────────────────


def score_behavioral(task: TaskItem, ctx: SurfacingContext) -> float:
    """Score based on learned behavioral patterns.

    Strategy:
    - Completion history by time-of-day: if user completes similar tasks
      at this time, boost.
    - Completion by cognitive load: if user handles this load level well
      historically, boost.
    - Tag affinity: if user has high completion on similar tagged tasks,
      boost.
    - Daily load balancing: if already completed many tasks today,
      gently reduce score (anti-overwhelm, NOT guilt).
    - Surface count: if task has been shown many times without action,
      gently reduce (the user chose not to act — respect that).

    Returns:
        Float in [0.0, 1.0].
    """
    profile = ctx.behavioral

    signals: list[float] = []
    weights: list[float] = []

    # 1) Time-of-day completion history
    offset = timedelta(hours=ctx.timezone_offset_hours)
    local_time = (ctx.current_time + offset).time()
    current_bucket = time_to_preference(local_time).value

    if profile.completion_by_time:
        time_rate = profile.completion_by_time.get(current_bucket, 0.5)
        signals.append(time_rate)
        weights.append(0.25)

    # 2) Cognitive load match
    if profile.completion_by_load:
        load_bucket = _load_bucket(task.cognitive_load)
        load_rate = profile.completion_by_load.get(load_bucket, 0.5)
        signals.append(load_rate)
        weights.append(0.25)

    # 3) Tag affinity
    if profile.completion_by_tag and task.tags:
        tag_rates = [
            profile.completion_by_tag.get(tag, 0.5)
            for tag in task.tags
        ]
        avg_tag_rate = sum(tag_rates) / len(tag_rates)
        signals.append(avg_tag_rate)
        weights.append(0.2)

    # 4) Daily load balancing (anti-overwhelm)
    if profile.avg_daily_completions > 0:
        load_ratio = profile.tasks_completed_today / profile.avg_daily_completions
        # If at/above average → gently decrease; if below → slight boost
        # Sigmoid-ish: flatten at extremes
        load_score = 1.0 / (1.0 + math.exp(2.0 * (load_ratio - 1.0)))
        signals.append(load_score)
        weights.append(0.15)

    # 5) Surface count fatigue (respect user's non-action)
    if task.surface_count > 0:
        # Gentle decay: after 5 surfaces, score contribution is ~0.37
        fatigue = math.exp(-task.surface_count / 5.0)
        signals.append(fatigue)
        weights.append(0.15)
    else:
        signals.append(0.7)  # fresh task baseline
        weights.append(0.15)

    if not signals:
        return 0.5  # No behavioral data — neutral

    total_weight = sum(weights)
    return _clamp(sum(s * w for s, w in zip(signals, weights)) / total_weight)


# ── Helpers ──────────────────────────────────────────────────────────


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _adjacent_buckets(a: TimePreference, b: TimePreference) -> bool:
    """Check if two time preference buckets are adjacent."""
    order = [
        TimePreference.EARLY_MORNING,
        TimePreference.MORNING,
        TimePreference.AFTERNOON,
        TimePreference.EVENING,
        TimePreference.NIGHT,
    ]
    try:
        ia = order.index(a)
        ib = order.index(b)
    except ValueError:
        return False
    return abs(ia - ib) == 1 or {ia, ib} == {0, len(order) - 1}


def _find_free_minutes(
    now: datetime, slots: list[CalendarSlot]
) -> float | None:
    """Find minutes of free time before the next busy slot.

    Returns None if there are no upcoming busy slots (assume free).
    """
    upcoming_busy = sorted(
        (s for s in slots if s.is_busy and s.start > now),
        key=lambda s: s.start,
    )
    if not upcoming_busy:
        return None  # No upcoming busy — assume free
    next_busy = upcoming_busy[0]
    return (next_busy.start - now).total_seconds() / 60.0


def _load_bucket(cognitive_load: float) -> str:
    """Map cognitive load (0-1) to a bucket label."""
    if cognitive_load < 0.33:
        return "low"
    elif cognitive_load < 0.66:
        return "medium"
    else:
        return "high"
