"""Rhythm detection — identifies recurring temporal patterns from episodic memory.

Analyzes aggregated temporal data to find recurring behavioral rhythms such as:
- Energy crashes (periods of low arousal + negative valence)
- Creativity peaks (high arousal + positive valence + idea/journal intents)
- Productivity windows (high completion rates + positive momentum)
- Mood cycles (recurring emotional shifts across the week)

The algorithm:
1. Aggregates episodes into temporal buckets (hour-of-day × day-of-week)
2. Computes per-bucket statistics (energy, valence, arousal, completions, skips)
3. Identifies statistically significant deviations (z-score ≥ 1.5)
4. Applies rolling averages over weekly windows to smooth noise
5. Uses autocorrelation-based periodicity detection to confirm weekly recurrence
6. Clusters adjacent significant buckets into named rhythm windows
7. Produces LearnedPattern instances for the semantic memory tier

Statistical methods:
- Rolling averages: Sliding window over week-numbered data to detect trends
- Periodicity detection: Autocorrelation at lag=7 (weekly) over daily time series
- Confidence boosting: Periodic patterns get higher confidence than one-off deviations

Designed for weekly cycle analysis with at least 2 weeks of data for reliable
detection. Confidence scales with observation count per bucket and periodicity
strength.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from blurt.memory.episodic import (
    BehavioralSignal,
    Episode,
    EpisodicMemoryStore,
)
from blurt.models.entities import LearnedPattern, PatternType


# ── Constants ────────────────────────────────────────────────────────

DAYS_OF_WEEK = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]
DAY_INDEX = {d: i for i, d in enumerate(DAYS_OF_WEEK)}

TIME_PERIODS = ["morning", "afternoon", "evening", "night"]
TIME_PERIOD_INDEX = {p: i for i, p in enumerate(TIME_PERIODS)}

# Minimum observations per bucket to be considered for pattern detection
MIN_BUCKET_OBSERVATIONS = 3

# Z-score threshold for statistically significant deviation
Z_SCORE_THRESHOLD = 1.5

# Minimum overall observations to attempt rhythm detection
MIN_TOTAL_OBSERVATIONS = 20

# Confidence scaling: confidence = min(1.0, obs_count / this_value)
CONFIDENCE_SATURATION_COUNT = 50

# Rolling average window size (number of weeks)
ROLLING_WINDOW_WEEKS = 3

# Minimum weeks of data required for periodicity detection
MIN_WEEKS_FOR_PERIODICITY = 2

# Autocorrelation threshold for confirming weekly periodicity
PERIODICITY_THRESHOLD = 0.3

# Confidence boost when periodicity is confirmed (multiplicative)
PERIODICITY_CONFIDENCE_BOOST = 1.3

# Number of slots in the weekly cycle (7 days × 4 time periods)
WEEKLY_SLOTS = 28


# ── Rhythm types ─────────────────────────────────────────────────────

class RhythmType(str, Enum):
    """Types of detected temporal rhythms."""
    ENERGY_CRASH = "energy_crash"
    ENERGY_PEAK = "energy_peak"
    CREATIVITY_PEAK = "creativity_peak"
    PRODUCTIVITY_WINDOW = "productivity_window"
    PRODUCTIVITY_DIP = "productivity_dip"
    MOOD_LOW = "mood_low"
    MOOD_HIGH = "mood_high"
    FOCUS_WINDOW = "focus_window"


# Map rhythm types to the PatternType they produce
RHYTHM_TO_PATTERN_TYPE: dict[RhythmType, PatternType] = {
    RhythmType.ENERGY_CRASH: PatternType.ENERGY_RHYTHM,
    RhythmType.ENERGY_PEAK: PatternType.ENERGY_RHYTHM,
    RhythmType.CREATIVITY_PEAK: PatternType.TIME_OF_DAY,
    RhythmType.PRODUCTIVITY_WINDOW: PatternType.TIME_OF_DAY,
    RhythmType.PRODUCTIVITY_DIP: PatternType.SKIP_SIGNAL,
    RhythmType.MOOD_LOW: PatternType.MOOD_CYCLE,
    RhythmType.MOOD_HIGH: PatternType.MOOD_CYCLE,
    RhythmType.FOCUS_WINDOW: PatternType.TIME_OF_DAY,
}

# Intent types associated with creative output
CREATIVE_INTENTS = {"idea", "journal"}

# Intent types associated with productive work
PRODUCTIVE_INTENTS = {"task", "update"}


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class RhythmBucket:
    """Aggregated statistics for a specific day-of-week × time-of-day slot."""

    day_of_week: str
    time_of_day: str
    observation_count: int = 0

    # Emotional aggregates
    valence_sum: float = 0.0
    valence_sq_sum: float = 0.0
    arousal_sum: float = 0.0
    arousal_sq_sum: float = 0.0

    # Behavioral counts
    completions: int = 0
    skips: int = 0
    dismissals: int = 0

    # Intent counts
    creative_count: int = 0
    productive_count: int = 0

    # Episode IDs for evidence
    episode_ids: list[str] = field(default_factory=list)

    @property
    def mean_valence(self) -> float:
        if self.observation_count == 0:
            return 0.0
        return self.valence_sum / self.observation_count

    @property
    def std_valence(self) -> float:
        return _std(self.valence_sum, self.valence_sq_sum, self.observation_count)

    @property
    def mean_arousal(self) -> float:
        if self.observation_count == 0:
            return 0.0
        return self.arousal_sum / self.observation_count

    @property
    def std_arousal(self) -> float:
        return _std(self.arousal_sum, self.arousal_sq_sum, self.observation_count)

    @property
    def energy_score(self) -> float:
        """Combined energy metric: arousal weighted by valence direction.

        High arousal + positive valence = high energy
        High arousal + negative valence = stressed (negative energy)
        Low arousal + negative valence = crashed
        """
        return self.mean_arousal * (0.5 + 0.5 * self.mean_valence)

    @property
    def completion_rate(self) -> float:
        total_actions = self.completions + self.skips + self.dismissals
        if total_actions == 0:
            return 0.0
        return self.completions / total_actions

    @property
    def skip_rate(self) -> float:
        total_actions = self.completions + self.skips + self.dismissals
        if total_actions == 0:
            return 0.0
        return (self.skips + self.dismissals) / total_actions

    @property
    def creativity_ratio(self) -> float:
        if self.observation_count == 0:
            return 0.0
        return self.creative_count / self.observation_count

    @property
    def productivity_ratio(self) -> float:
        if self.observation_count == 0:
            return 0.0
        return self.productive_count / self.observation_count

    def add_episode(self, episode: Episode) -> None:
        """Ingest a single episode into this bucket's aggregates."""
        self.observation_count += 1
        self.valence_sum += episode.emotion.valence
        self.valence_sq_sum += episode.emotion.valence ** 2
        self.arousal_sum += episode.emotion.arousal
        self.arousal_sq_sum += episode.emotion.arousal ** 2
        self.episode_ids.append(episode.id)

        # Behavioral signals
        if episode.behavioral_signal == BehavioralSignal.COMPLETED:
            self.completions += 1
        elif episode.behavioral_signal == BehavioralSignal.SKIPPED:
            self.skips += 1
        elif episode.behavioral_signal == BehavioralSignal.DISMISSED:
            self.dismissals += 1

        # Intent classification
        if episode.intent in CREATIVE_INTENTS:
            self.creative_count += 1
        if episode.intent in PRODUCTIVE_INTENTS:
            self.productive_count += 1


@dataclass
class WeeklySlotSample:
    """A single week's observation for a specific day×time slot.

    Used for rolling average computation and periodicity detection.
    Each sample represents one week's aggregated value for a slot.
    """

    week_number: int  # ISO week number or sequential week index
    value: float  # The metric value for this slot in this week
    observation_count: int = 0


@dataclass
class RollingAverageResult:
    """Result of rolling average computation for a single slot."""

    slot_key: str
    day_of_week: str
    time_of_day: str
    rolling_values: list[float]  # Rolling averages per window
    overall_mean: float = 0.0
    trend_direction: float = 0.0  # Positive = increasing, negative = decreasing
    trend_strength: float = 0.0  # 0.0–1.0 how strong the trend is
    weeks_of_data: int = 0

    @property
    def is_trending_up(self) -> bool:
        return self.trend_direction > 0.1

    @property
    def is_trending_down(self) -> bool:
        return self.trend_direction < -0.1

    @property
    def is_stable(self) -> bool:
        return abs(self.trend_direction) <= 0.1


@dataclass
class PeriodicityResult:
    """Result of periodicity detection for a metric time series."""

    slot_key: str
    day_of_week: str
    time_of_day: str
    autocorrelation_lag7: float  # Autocorrelation at weekly lag
    is_periodic: bool  # Whether the pattern recurs weekly
    period_strength: float  # 0.0–1.0 strength of the periodicity
    sample_count: int = 0

    @property
    def confidence_multiplier(self) -> float:
        """Confidence boost for confirmed periodic patterns."""
        if self.is_periodic:
            return min(PERIODICITY_CONFIDENCE_BOOST,
                       1.0 + self.period_strength * 0.5)
        return 1.0


@dataclass(frozen=True)
class DetectedRhythm:
    """A single detected temporal rhythm."""

    rhythm_type: RhythmType
    day_of_week: str
    time_of_day: str
    z_score: float
    metric_value: float
    metric_mean: float
    metric_std: float
    observation_count: int
    confidence: float
    evidence_episode_ids: list[str] = field(default_factory=list)
    # Rolling average and periodicity metadata
    is_periodic: bool = False
    periodicity_strength: float = 0.0
    trend_direction: float = 0.0  # Positive = getting stronger, negative = weakening
    weeks_observed: int = 0

    @property
    def description(self) -> str:
        """Human-readable description of this rhythm."""
        templates = {
            RhythmType.ENERGY_CRASH: "Energy tends to drop on {day} {time}",
            RhythmType.ENERGY_PEAK: "Energy peaks on {day} {time}",
            RhythmType.CREATIVITY_PEAK: "Creative ideas flow most on {day} {time}",
            RhythmType.PRODUCTIVITY_WINDOW: "Most productive on {day} {time}",
            RhythmType.PRODUCTIVITY_DIP: "Tasks often skipped on {day} {time}",
            RhythmType.MOOD_LOW: "Mood tends to dip on {day} {time}",
            RhythmType.MOOD_HIGH: "Mood is usually highest on {day} {time}",
            RhythmType.FOCUS_WINDOW: "Deep focus happens on {day} {time}",
        }
        template = templates.get(self.rhythm_type, "{day} {time}: {type}")
        return template.format(
            day=self.day_of_week.capitalize(),
            time=self.time_of_day,
            type=self.rhythm_type.value,
        )

    def to_learned_pattern(self, user_id: str) -> LearnedPattern:
        """Convert to a LearnedPattern for semantic memory storage."""
        return LearnedPattern(
            user_id=user_id,
            pattern_type=RHYTHM_TO_PATTERN_TYPE[self.rhythm_type],
            description=self.description,
            parameters={
                "rhythm_type": self.rhythm_type.value,
                "day_of_week": self.day_of_week,
                "time_of_day": self.time_of_day,
                "z_score": round(self.z_score, 3),
                "metric_value": round(self.metric_value, 3),
                "metric_mean": round(self.metric_mean, 3),
                "metric_std": round(self.metric_std, 3),
                "is_periodic": self.is_periodic,
                "periodicity_strength": round(self.periodicity_strength, 3),
                "trend_direction": round(self.trend_direction, 3),
                "weeks_observed": self.weeks_observed,
            },
            confidence=self.confidence,
            observation_count=self.observation_count,
            supporting_evidence=self.evidence_episode_ids[:20],  # Cap evidence list
        )


@dataclass
class RhythmAnalysisResult:
    """Complete result of rhythm detection analysis."""

    user_id: str
    analysis_period_start: datetime
    analysis_period_end: datetime
    total_episodes_analyzed: int
    rhythms: list[DetectedRhythm] = field(default_factory=list)
    bucket_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    rolling_averages: dict[str, RollingAverageResult] = field(default_factory=dict)
    periodicity_results: dict[str, PeriodicityResult] = field(default_factory=dict)

    @property
    def periodic_rhythms(self) -> list[DetectedRhythm]:
        """Rhythms confirmed as recurring weekly patterns."""
        return [r for r in self.rhythms if r.is_periodic]

    @property
    def energy_crashes(self) -> list[DetectedRhythm]:
        return [r for r in self.rhythms if r.rhythm_type == RhythmType.ENERGY_CRASH]

    @property
    def creativity_peaks(self) -> list[DetectedRhythm]:
        return [r for r in self.rhythms if r.rhythm_type == RhythmType.CREATIVITY_PEAK]

    @property
    def productivity_windows(self) -> list[DetectedRhythm]:
        return [r for r in self.rhythms if r.rhythm_type == RhythmType.PRODUCTIVITY_WINDOW]

    @property
    def mood_patterns(self) -> list[DetectedRhythm]:
        return [r for r in self.rhythms if r.rhythm_type in (
            RhythmType.MOOD_LOW, RhythmType.MOOD_HIGH,
        )]

    def to_learned_patterns(self) -> list[LearnedPattern]:
        """Convert all detected rhythms to LearnedPattern instances."""
        return [r.to_learned_pattern(self.user_id) for r in self.rhythms]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses."""
        return {
            "user_id": self.user_id,
            "analysis_period_start": self.analysis_period_start.isoformat(),
            "analysis_period_end": self.analysis_period_end.isoformat(),
            "total_episodes_analyzed": self.total_episodes_analyzed,
            "rhythms": [
                {
                    "rhythm_type": r.rhythm_type.value,
                    "day_of_week": r.day_of_week,
                    "time_of_day": r.time_of_day,
                    "z_score": round(r.z_score, 3),
                    "metric_value": round(r.metric_value, 3),
                    "confidence": round(r.confidence, 3),
                    "observation_count": r.observation_count,
                    "description": r.description,
                }
                for r in self.rhythms
            ],
            "summary": {
                "energy_crashes": len(self.energy_crashes),
                "creativity_peaks": len(self.creativity_peaks),
                "productivity_windows": len(self.productivity_windows),
                "mood_patterns": len(self.mood_patterns),
                "periodic_patterns": len(self.periodic_rhythms),
            },
            "periodicity": {
                key: {
                    "autocorrelation_lag7": round(pr.autocorrelation_lag7, 3),
                    "is_periodic": pr.is_periodic,
                    "period_strength": round(pr.period_strength, 3),
                }
                for key, pr in self.periodicity_results.items()
            },
        }


# ── Helpers ──────────────────────────────────────────────────────────

def _std(sum_x: float, sum_x2: float, n: int) -> float:
    """Compute standard deviation from running sums."""
    if n < 2:
        return 0.0
    variance = (sum_x2 / n) - (sum_x / n) ** 2
    return math.sqrt(max(0.0, variance))


def _z_score(value: float, mean: float, std: float) -> float:
    """Compute z-score. Returns 0 if std is effectively zero."""
    if std < 1e-9:
        return 0.0
    return (value - mean) / std


def _bucket_key(day: str, time: str) -> str:
    """Create a canonical bucket key."""
    return f"{day}:{time}"


def _confidence_from_count(count: int) -> float:
    """Scale confidence by observation count. Saturates at CONFIDENCE_SATURATION_COUNT."""
    return min(1.0, count / CONFIDENCE_SATURATION_COUNT)


# ── Rolling averages ──────────────────────────────────────────────────

def _episodes_to_weekly_slot_samples(
    episodes: list[Episode],
    metric_fn: Callable[..., Any],
) -> dict[str, list[WeeklySlotSample]]:
    """Group episodes by (day×time) slot and week, computing a metric per week per slot.

    Args:
        episodes: Episodes with timestamps.
        metric_fn: Function(Episode) -> float extracting the metric value.

    Returns:
        Dict keyed by slot key (e.g. "monday:morning") -> list of WeeklySlotSamples.
    """
    # Group episodes by (slot_key, iso_week_index)
    slot_week_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    slot_week_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for episode in episodes:
        day = episode.context.day_of_week.lower()
        time = episode.context.time_of_day.lower()
        if day not in DAY_INDEX or time not in TIME_PERIOD_INDEX:
            continue

        key = _bucket_key(day, time)
        # Use ISO calendar week as the grouping key
        iso_year, iso_week, _ = episode.timestamp.isocalendar()
        week_idx = iso_year * 100 + iso_week  # Unique week identifier

        val = metric_fn(episode)
        slot_week_values[key][week_idx].append(val)
        slot_week_counts[key][week_idx] += 1

    # Convert to WeeklySlotSamples (average metric per week per slot)
    result: dict[str, list[WeeklySlotSample]] = {}
    for slot_key, weeks in slot_week_values.items():
        samples: list[WeeklySlotSample] = []
        for week_idx in sorted(weeks.keys()):
            values = weeks[week_idx]
            avg = sum(values) / len(values) if values else 0.0
            samples.append(WeeklySlotSample(
                week_number=week_idx,
                value=avg,
                observation_count=slot_week_counts[slot_key][week_idx],
            ))
        result[slot_key] = samples

    return result


def compute_rolling_average(
    samples: list[WeeklySlotSample],
    window_size: int = ROLLING_WINDOW_WEEKS,
) -> list[float]:
    """Compute rolling averages over weekly samples.

    Uses a simple moving average with the given window size.
    If fewer samples than window_size exist, uses all available.

    Args:
        samples: Weekly metric samples, sorted by week.
        window_size: Number of weeks in the rolling window.

    Returns:
        List of rolling average values, one per valid window position.
    """
    if not samples:
        return []

    values = [s.value for s in samples]

    if len(values) < window_size:
        # Not enough data for a full window — return a single average
        return [sum(values) / len(values)]

    rolling: list[float] = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        rolling.append(sum(window) / len(window))

    return rolling


def compute_trend(rolling_values: list[float]) -> tuple[float, float]:
    """Compute trend direction and strength from rolling averages.

    Uses simple linear regression (least squares) over the rolling values
    to determine if a metric is trending up, down, or stable.

    Args:
        rolling_values: Rolling average values over time.

    Returns:
        (trend_direction, trend_strength) where direction is the slope
        normalized by the range, and strength is R² (0.0–1.0).
    """
    n = len(rolling_values)
    if n < 2:
        return 0.0, 0.0

    # Simple linear regression: y = mx + b
    x_mean = (n - 1) / 2.0
    y_mean = sum(rolling_values) / n

    numerator = 0.0
    denominator = 0.0
    ss_total = 0.0

    for i, y in enumerate(rolling_values):
        x_diff = i - x_mean
        y_diff = y - y_mean
        numerator += x_diff * y_diff
        denominator += x_diff * x_diff
        ss_total += y_diff * y_diff

    if denominator < 1e-9 or ss_total < 1e-9:
        return 0.0, 0.0

    slope = numerator / denominator

    # R² for trend strength
    ss_residual = 0.0
    for i, y in enumerate(rolling_values):
        predicted = y_mean + slope * (i - x_mean)
        ss_residual += (y - predicted) ** 2

    r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0
    r_squared = max(0.0, r_squared)

    # Normalize slope by the mean to get relative direction
    if abs(y_mean) > 1e-9:
        direction = slope / abs(y_mean)
    else:
        direction = slope

    return direction, r_squared


def compute_rolling_averages_for_episodes(
    episodes: list[Episode],
    metric_name: str = "energy",
) -> dict[str, RollingAverageResult]:
    """Compute rolling averages for all slots using episode data.

    Args:
        episodes: Episodes to analyze.
        metric_name: Which metric to compute ("energy", "valence", "creativity").

    Returns:
        Dict of slot_key -> RollingAverageResult.
    """
    metric_fns = {
        "energy": lambda ep: ep.emotion.arousal * (0.5 + 0.5 * ep.emotion.valence),
        "valence": lambda ep: ep.emotion.valence,
        "arousal": lambda ep: ep.emotion.arousal,
        "creativity": lambda ep: 1.0 if ep.intent in CREATIVE_INTENTS else 0.0,
        "productivity": lambda ep: 1.0 if ep.intent in PRODUCTIVE_INTENTS else 0.0,
    }

    fn = metric_fns.get(metric_name, metric_fns["energy"])
    weekly_samples = _episodes_to_weekly_slot_samples(episodes, fn)

    results: dict[str, RollingAverageResult] = {}

    for slot_key, samples in weekly_samples.items():
        if not samples:
            continue

        day, time = slot_key.split(":")
        rolling = compute_rolling_average(samples)
        all_values = [s.value for s in samples]
        overall_mean = sum(all_values) / len(all_values) if all_values else 0.0
        trend_dir, trend_str = compute_trend(rolling)

        results[slot_key] = RollingAverageResult(
            slot_key=slot_key,
            day_of_week=day,
            time_of_day=time,
            rolling_values=rolling,
            overall_mean=overall_mean,
            trend_direction=trend_dir,
            trend_strength=trend_str,
            weeks_of_data=len(samples),
        )

    return results


# ── Periodicity detection ────────────────────────────────────────────

def _episodes_to_daily_series(
    episodes: list[Episode],
    metric_fn: Callable[..., Any],
) -> list[float]:
    """Convert episodes to a daily time series for periodicity analysis.

    Creates a series where each position is one day, with the average
    metric value for that day. Missing days get the overall mean.

    Args:
        episodes: Episodes with timestamps, sorted by time.
        metric_fn: Function(Episode) -> float.

    Returns:
        Daily time series of metric values.
    """
    if not episodes:
        return []

    sorted_eps = sorted(episodes, key=lambda e: e.timestamp)
    start_date = sorted_eps[0].timestamp.date()
    end_date = sorted_eps[-1].timestamp.date()
    total_days = (end_date - start_date).days + 1

    if total_days < 1:
        return []

    # Collect values per day
    day_values: dict[int, list[float]] = defaultdict(list)
    for ep in sorted_eps:
        day_offset = (ep.timestamp.date() - start_date).days
        day_values[day_offset].append(metric_fn(ep))

    # Compute overall mean for filling gaps
    all_vals = [v for vals in day_values.values() for v in vals]
    overall_mean = sum(all_vals) / len(all_vals) if all_vals else 0.0

    # Build series
    series: list[float] = []
    for d in range(total_days):
        if d in day_values:
            series.append(sum(day_values[d]) / len(day_values[d]))
        else:
            series.append(overall_mean)

    return series


def compute_autocorrelation(series: list[float], lag: int) -> float:
    """Compute autocorrelation of a time series at a given lag.

    Uses the standard Pearson correlation between the series and
    its lagged version.

    Args:
        series: Time series values.
        lag: Number of positions to shift.

    Returns:
        Autocorrelation coefficient (-1.0 to 1.0), or 0.0 if insufficient data.
    """
    n = len(series)
    if n <= lag or n < 3:
        return 0.0

    # Compute mean
    mean = sum(series) / n

    # Compute autocovariance at the given lag
    numerator = 0.0
    for i in range(n - lag):
        numerator += (series[i] - mean) * (series[i + lag] - mean)

    # Compute variance
    variance = sum((x - mean) ** 2 for x in series)

    if variance < 1e-9:
        return 0.0

    return numerator / variance


def detect_weekly_periodicity(
    episodes: list[Episode],
    metric_fn: Callable[..., Any],
) -> float:
    """Detect if a metric shows weekly periodicity using autocorrelation at lag=7.

    Args:
        episodes: Episodes to analyze.
        metric_fn: Function(Episode) -> float.

    Returns:
        Autocorrelation at lag=7 (weekly). Values > PERIODICITY_THRESHOLD
        indicate weekly recurrence.
    """
    series = _episodes_to_daily_series(episodes, metric_fn)

    if len(series) < 14:  # Need at least 2 weeks
        return 0.0

    return compute_autocorrelation(series, lag=7)


def compute_periodicity_for_slots(
    episodes: list[Episode],
    metric_name: str = "energy",
) -> dict[str, PeriodicityResult]:
    """Compute periodicity detection for each day×time slot.

    For each slot, extracts the daily series of the metric and tests
    for weekly periodicity using autocorrelation at lag=7.

    Args:
        episodes: Episodes to analyze.
        metric_name: Which metric to test ("energy", "valence", etc).

    Returns:
        Dict of slot_key -> PeriodicityResult.
    """
    metric_fns = {
        "energy": lambda ep: ep.emotion.arousal * (0.5 + 0.5 * ep.emotion.valence),
        "valence": lambda ep: ep.emotion.valence,
        "arousal": lambda ep: ep.emotion.arousal,
    }

    fn = metric_fns.get(metric_name, metric_fns["energy"])

    # Group episodes by day-of-week for day-level periodicity
    day_episodes: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        day = ep.context.day_of_week.lower()
        if day in DAY_INDEX:
            day_episodes[day].append(ep)

    # Check overall periodicity first
    overall_autocorr = detect_weekly_periodicity(episodes, fn)

    results: dict[str, PeriodicityResult] = {}

    # For each slot, compute periodicity based on week-over-week consistency
    weekly_samples = _episodes_to_weekly_slot_samples(episodes, fn)

    for slot_key, samples in weekly_samples.items():
        day, time = slot_key.split(":")

        if len(samples) < MIN_WEEKS_FOR_PERIODICITY:
            results[slot_key] = PeriodicityResult(
                slot_key=slot_key,
                day_of_week=day,
                time_of_day=time,
                autocorrelation_lag7=0.0,
                is_periodic=False,
                period_strength=0.0,
                sample_count=len(samples),
            )
            continue

        # Compute week-over-week consistency for this slot
        values = [s.value for s in samples]
        slot_autocorr = _compute_slot_consistency(values)

        # Combine slot-level consistency with overall weekly periodicity
        combined = (slot_autocorr * 0.6 + max(0.0, overall_autocorr) * 0.4)
        is_periodic = combined >= PERIODICITY_THRESHOLD and len(samples) >= MIN_WEEKS_FOR_PERIODICITY

        results[slot_key] = PeriodicityResult(
            slot_key=slot_key,
            day_of_week=day,
            time_of_day=time,
            autocorrelation_lag7=overall_autocorr,
            is_periodic=is_periodic,
            period_strength=min(1.0, max(0.0, combined)),
            sample_count=len(samples),
        )

    return results


def _compute_slot_consistency(values: list[float]) -> float:
    """Measure how consistent a metric is across weeks for a single slot.

    Uses coefficient of variation inverted: low CV = high consistency.
    Returns 0.0–1.0 where 1.0 means perfectly consistent.

    Args:
        values: Weekly values for a single slot.

    Returns:
        Consistency score (0.0–1.0).
    """
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    if abs(mean) < 1e-9:
        # All values near zero — check if they're consistently near zero
        max_dev = max(abs(v) for v in values)
        return 1.0 if max_dev < 0.1 else 0.0

    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(max(0.0, variance))
    cv = std / abs(mean)

    # Invert CV: low variance = high consistency
    # CV of 0 → 1.0, CV of 1 → 0.5, CV of 2 → 0.33, etc.
    return 1.0 / (1.0 + cv)


# ── Core aggregation ─────────────────────────────────────────────────

def aggregate_episodes(episodes: list[Episode]) -> dict[str, RhythmBucket]:
    """Aggregate episodes into day-of-week × time-of-day buckets.

    Returns a dict keyed by "day:time_period" with aggregated stats.
    """
    buckets: dict[str, RhythmBucket] = {}

    # Pre-create all 28 buckets (7 days × 4 time periods)
    for day in DAYS_OF_WEEK:
        for period in TIME_PERIODS:
            key = _bucket_key(day, period)
            buckets[key] = RhythmBucket(day_of_week=day, time_of_day=period)

    for episode in episodes:
        day = episode.context.day_of_week.lower()
        time = episode.context.time_of_day.lower()

        # Validate and normalize
        if day not in DAY_INDEX:
            continue
        if time not in TIME_PERIOD_INDEX:
            continue

        key = _bucket_key(day, time)
        buckets[key].add_episode(episode)

    return buckets


# ── Pattern detection functions ──────────────────────────────────────

def detect_energy_rhythms(
    buckets: dict[str, RhythmBucket],
) -> list[DetectedRhythm]:
    """Detect energy crashes and peaks from aggregated temporal data.

    Energy is computed as arousal weighted by valence direction:
    - High arousal + positive valence = peak
    - Low arousal + negative valence = crash
    """
    eligible = {k: b for k, b in buckets.items()
                if b.observation_count >= MIN_BUCKET_OBSERVATIONS}

    if len(eligible) < 3:
        return []

    energy_scores = {k: b.energy_score for k, b in eligible.items()}
    values = list(energy_scores.values())
    mean_energy = sum(values) / len(values)
    std_energy = _std(sum(values), sum(v ** 2 for v in values), len(values))

    rhythms: list[DetectedRhythm] = []

    for key, bucket in eligible.items():
        score = energy_scores[key]
        z = _z_score(score, mean_energy, std_energy)

        if z <= -Z_SCORE_THRESHOLD:
            # Energy crash
            rhythms.append(DetectedRhythm(
                rhythm_type=RhythmType.ENERGY_CRASH,
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                z_score=abs(z),
                metric_value=score,
                metric_mean=mean_energy,
                metric_std=std_energy,
                observation_count=bucket.observation_count,
                confidence=_confidence_from_count(bucket.observation_count),
                evidence_episode_ids=bucket.episode_ids,
            ))
        elif z >= Z_SCORE_THRESHOLD:
            # Energy peak
            rhythms.append(DetectedRhythm(
                rhythm_type=RhythmType.ENERGY_PEAK,
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                z_score=z,
                metric_value=score,
                metric_mean=mean_energy,
                metric_std=std_energy,
                observation_count=bucket.observation_count,
                confidence=_confidence_from_count(bucket.observation_count),
                evidence_episode_ids=bucket.episode_ids,
            ))

    return rhythms


def detect_creativity_peaks(
    buckets: dict[str, RhythmBucket],
) -> list[DetectedRhythm]:
    """Detect times when creative output (ideas, journaling) is concentrated."""
    eligible = {k: b for k, b in buckets.items()
                if b.observation_count >= MIN_BUCKET_OBSERVATIONS}

    if len(eligible) < 3:
        return []

    ratios = {k: b.creativity_ratio for k, b in eligible.items()}
    values = list(ratios.values())
    mean_cr = sum(values) / len(values)
    std_cr = _std(sum(values), sum(v ** 2 for v in values), len(values))

    rhythms: list[DetectedRhythm] = []

    for key, bucket in eligible.items():
        ratio = ratios[key]
        z = _z_score(ratio, mean_cr, std_cr)

        if z >= Z_SCORE_THRESHOLD and bucket.creative_count >= 2:
            rhythms.append(DetectedRhythm(
                rhythm_type=RhythmType.CREATIVITY_PEAK,
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                z_score=z,
                metric_value=ratio,
                metric_mean=mean_cr,
                metric_std=std_cr,
                observation_count=bucket.observation_count,
                confidence=_confidence_from_count(bucket.observation_count),
                evidence_episode_ids=bucket.episode_ids,
            ))

    return rhythms


def detect_productivity_patterns(
    buckets: dict[str, RhythmBucket],
) -> list[DetectedRhythm]:
    """Detect productivity windows (high completion) and dips (high skip/dismiss)."""
    # Only consider buckets with behavioral signals
    eligible = {k: b for k, b in buckets.items()
                if (b.completions + b.skips + b.dismissals) >= MIN_BUCKET_OBSERVATIONS}

    if len(eligible) < 3:
        return []

    comp_rates = {k: b.completion_rate for k, b in eligible.items()}
    values = list(comp_rates.values())
    mean_cr = sum(values) / len(values)
    std_cr = _std(sum(values), sum(v ** 2 for v in values), len(values))

    rhythms: list[DetectedRhythm] = []

    for key, bucket in eligible.items():
        rate = comp_rates[key]
        z = _z_score(rate, mean_cr, std_cr)

        if z >= Z_SCORE_THRESHOLD:
            rhythms.append(DetectedRhythm(
                rhythm_type=RhythmType.PRODUCTIVITY_WINDOW,
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                z_score=z,
                metric_value=rate,
                metric_mean=mean_cr,
                metric_std=std_cr,
                observation_count=bucket.observation_count,
                confidence=_confidence_from_count(bucket.observation_count),
                evidence_episode_ids=bucket.episode_ids,
            ))
        elif z <= -Z_SCORE_THRESHOLD:
            rhythms.append(DetectedRhythm(
                rhythm_type=RhythmType.PRODUCTIVITY_DIP,
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                z_score=abs(z),
                metric_value=rate,
                metric_mean=mean_cr,
                metric_std=std_cr,
                observation_count=bucket.observation_count,
                confidence=_confidence_from_count(bucket.observation_count),
                evidence_episode_ids=bucket.episode_ids,
            ))

    return rhythms


def detect_mood_cycles(
    buckets: dict[str, RhythmBucket],
) -> list[DetectedRhythm]:
    """Detect recurring mood highs and lows across the weekly cycle."""
    eligible = {k: b for k, b in buckets.items()
                if b.observation_count >= MIN_BUCKET_OBSERVATIONS}

    if len(eligible) < 3:
        return []

    valences = {k: b.mean_valence for k, b in eligible.items()}
    values = list(valences.values())
    mean_v = sum(values) / len(values)
    std_v = _std(sum(values), sum(v ** 2 for v in values), len(values))

    rhythms: list[DetectedRhythm] = []

    for key, bucket in eligible.items():
        val = valences[key]
        z = _z_score(val, mean_v, std_v)

        if z <= -Z_SCORE_THRESHOLD:
            rhythms.append(DetectedRhythm(
                rhythm_type=RhythmType.MOOD_LOW,
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                z_score=abs(z),
                metric_value=val,
                metric_mean=mean_v,
                metric_std=std_v,
                observation_count=bucket.observation_count,
                confidence=_confidence_from_count(bucket.observation_count),
                evidence_episode_ids=bucket.episode_ids,
            ))
        elif z >= Z_SCORE_THRESHOLD:
            rhythms.append(DetectedRhythm(
                rhythm_type=RhythmType.MOOD_HIGH,
                day_of_week=bucket.day_of_week,
                time_of_day=bucket.time_of_day,
                z_score=z,
                metric_value=val,
                metric_mean=mean_v,
                metric_std=std_v,
                observation_count=bucket.observation_count,
                confidence=_confidence_from_count(bucket.observation_count),
                evidence_episode_ids=bucket.episode_ids,
            ))

    return rhythms


# ── Orchestrator ─────────────────────────────────────────────────────

def analyze_rhythms(
    user_id: str,
    episodes: list[Episode],
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> RhythmAnalysisResult:
    """Run the full rhythm detection pipeline on a list of episodes.

    This is the main entry point for rhythm detection. It:
    1. Aggregates episodes into temporal buckets
    2. Runs all detection algorithms
    3. Sorts results by confidence × z-score
    4. Returns a complete analysis result

    Args:
        user_id: The user whose rhythms are being analyzed.
        episodes: Episodes to analyze (should span at least 2 weeks).
        period_start: Start of the analysis period (defaults to earliest episode).
        period_end: End of the analysis period (defaults to latest episode).

    Returns:
        RhythmAnalysisResult with all detected patterns.
    """
    if not episodes:
        now = datetime.now(timezone.utc)
        return RhythmAnalysisResult(
            user_id=user_id,
            analysis_period_start=period_start or now,
            analysis_period_end=period_end or now,
            total_episodes_analyzed=0,
        )

    sorted_eps = sorted(episodes, key=lambda e: e.timestamp)
    start = period_start or sorted_eps[0].timestamp
    end = period_end or sorted_eps[-1].timestamp

    # Step 1: Aggregate into temporal buckets
    buckets = aggregate_episodes(episodes)

    # Step 2: Run all detectors
    all_rhythms: list[DetectedRhythm] = []

    if len(episodes) >= MIN_TOTAL_OBSERVATIONS:
        all_rhythms.extend(detect_energy_rhythms(buckets))
        all_rhythms.extend(detect_creativity_peaks(buckets))
        all_rhythms.extend(detect_productivity_patterns(buckets))
        all_rhythms.extend(detect_mood_cycles(buckets))

    # Step 3: Compute rolling averages and periodicity
    rolling_averages = compute_rolling_averages_for_episodes(episodes, "energy")
    periodicity_results = compute_periodicity_for_slots(episodes, "energy")

    # Step 4: Enrich rhythms with periodicity and trend data
    all_rhythms = _enrich_rhythms_with_periodicity(
        all_rhythms, rolling_averages, periodicity_results,
    )

    # Step 5: Sort by significance (confidence × z_score, descending)
    all_rhythms.sort(key=lambda r: r.confidence * r.z_score, reverse=True)

    # Step 6: Build bucket stats summary
    bucket_stats = {}
    for key, bucket in buckets.items():
        if bucket.observation_count > 0:
            bucket_stats[key] = {
                "observation_count": bucket.observation_count,
                "mean_valence": round(bucket.mean_valence, 3),
                "mean_arousal": round(bucket.mean_arousal, 3),
                "energy_score": round(bucket.energy_score, 3),
                "completion_rate": round(bucket.completion_rate, 3),
                "creativity_ratio": round(bucket.creativity_ratio, 3),
            }

    return RhythmAnalysisResult(
        user_id=user_id,
        analysis_period_start=start,
        analysis_period_end=end,
        total_episodes_analyzed=len(episodes),
        rhythms=all_rhythms,
        bucket_stats=bucket_stats,
        rolling_averages=rolling_averages,
        periodicity_results=periodicity_results,
    )


def _enrich_rhythms_with_periodicity(
    rhythms: list[DetectedRhythm],
    rolling_averages: dict[str, RollingAverageResult],
    periodicity_results: dict[str, PeriodicityResult],
) -> list[DetectedRhythm]:
    """Enrich detected rhythms with periodicity and trend data.

    Periodic patterns get a confidence boost. Trend data helps distinguish
    stable patterns from emerging or fading ones.

    Args:
        rhythms: Base detected rhythms.
        rolling_averages: Rolling average results per slot.
        periodicity_results: Periodicity detection results per slot.

    Returns:
        New list of DetectedRhythm with updated fields.
    """
    enriched: list[DetectedRhythm] = []

    for rhythm in rhythms:
        key = _bucket_key(rhythm.day_of_week, rhythm.time_of_day)
        ra = rolling_averages.get(key)
        pr = periodicity_results.get(key)

        is_periodic = pr.is_periodic if pr else False
        period_strength = pr.period_strength if pr else 0.0
        trend_dir = ra.trend_direction if ra else 0.0
        weeks = ra.weeks_of_data if ra else 0

        # Boost confidence for periodic patterns
        confidence = rhythm.confidence
        if is_periodic and pr:
            confidence = min(1.0, confidence * pr.confidence_multiplier)

        # Create enriched rhythm (DetectedRhythm is frozen, so we rebuild)
        enriched.append(DetectedRhythm(
            rhythm_type=rhythm.rhythm_type,
            day_of_week=rhythm.day_of_week,
            time_of_day=rhythm.time_of_day,
            z_score=rhythm.z_score,
            metric_value=rhythm.metric_value,
            metric_mean=rhythm.metric_mean,
            metric_std=rhythm.metric_std,
            observation_count=rhythm.observation_count,
            confidence=confidence,
            evidence_episode_ids=rhythm.evidence_episode_ids,
            is_periodic=is_periodic,
            periodicity_strength=period_strength,
            trend_direction=trend_dir,
            weeks_observed=weeks,
        ))

    return enriched


# ── Service class ────────────────────────────────────────────────────

class RhythmDetectionService:
    """Service for detecting behavioral rhythms from episodic memory.

    Wraps the rhythm detection pipeline and integrates with the
    EpisodicMemoryStore for data retrieval.
    """

    def __init__(
        self,
        episodic_store: EpisodicMemoryStore,
        *,
        min_observations: int = MIN_TOTAL_OBSERVATIONS,
        z_threshold: float = Z_SCORE_THRESHOLD,
    ) -> None:
        self._store = episodic_store
        self._min_observations = min_observations
        self._z_threshold = z_threshold

    async def analyze_user_rhythms(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> RhythmAnalysisResult:
        """Analyze rhythms for a user over a time range.

        Args:
            user_id: The user to analyze.
            start: Beginning of the analysis window.
            end: End of the analysis window.

        Returns:
            Complete rhythm analysis result.
        """
        episodes = await self._store.get_emotion_timeline(user_id, start, end)
        return analyze_rhythms(user_id, episodes, start, end)

    async def get_current_rhythm_context(
        self,
        user_id: str,
        current_day: str,
        current_time: str,
        lookback_weeks: int = 4,
    ) -> dict[str, Any]:
        """Get the rhythm context for the current moment.

        Useful for task surfacing — tells the caller what kind of window
        the user is typically in at this day/time.

        Args:
            user_id: The user to analyze.
            current_day: Current day of week (e.g., "monday").
            current_time: Current time period (e.g., "morning").
            lookback_weeks: How many weeks of history to analyze.

        Returns:
            Dict with rhythm context for the current time slot.
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(weeks=lookback_weeks)

        result = await self.analyze_user_rhythms(user_id, start, now)

        # Find rhythms matching the current slot
        current_rhythms = [
            r for r in result.rhythms
            if r.day_of_week == current_day.lower()
            and r.time_of_day == current_time.lower()
        ]

        # Get bucket stats for this slot
        key = _bucket_key(current_day.lower(), current_time.lower())
        bucket = result.bucket_stats.get(key, {})

        # Get rolling average and periodicity for this slot
        ra = result.rolling_averages.get(key)
        pr = result.periodicity_results.get(key)

        return {
            "day_of_week": current_day,
            "time_of_day": current_time,
            "active_rhythms": [
                {
                    "type": r.rhythm_type.value,
                    "confidence": round(r.confidence, 3),
                    "description": r.description,
                    "is_periodic": r.is_periodic,
                    "trend": "up" if r.trend_direction > 0.1
                             else "down" if r.trend_direction < -0.1
                             else "stable",
                }
                for r in current_rhythms
            ],
            "bucket_stats": bucket,
            "rolling_average": {
                "overall_mean": round(ra.overall_mean, 3),
                "trend_direction": round(ra.trend_direction, 3),
                "trend_strength": round(ra.trend_strength, 3),
                "weeks_of_data": ra.weeks_of_data,
            } if ra else None,
            "periodicity": {
                "is_periodic": pr.is_periodic,
                "strength": round(pr.period_strength, 3),
                "autocorrelation": round(pr.autocorrelation_lag7, 3),
            } if pr else None,
            "recommendations": _generate_recommendations(current_rhythms),
        }


def _generate_recommendations(rhythms: list[DetectedRhythm]) -> list[str]:
    """Generate shame-free recommendations based on detected rhythms.

    Anti-shame: suggestions are positive, never guilt-inducing.
    """
    recommendations: list[str] = []

    for rhythm in rhythms:
        if rhythm.rhythm_type == RhythmType.ENERGY_CRASH:
            recommendations.append(
                "This is typically a lower-energy time — "
                "lighter tasks or a break might feel right"
            )
        elif rhythm.rhythm_type == RhythmType.ENERGY_PEAK:
            recommendations.append(
                "You usually have great energy now — "
                "good time for challenging work if you're up for it"
            )
        elif rhythm.rhythm_type == RhythmType.CREATIVITY_PEAK:
            recommendations.append(
                "Creative ideas tend to flow well at this time"
            )
        elif rhythm.rhythm_type == RhythmType.PRODUCTIVITY_WINDOW:
            recommendations.append(
                "You've been getting things done at this time — "
                "a good window for tasks if you'd like"
            )
        elif rhythm.rhythm_type == RhythmType.PRODUCTIVITY_DIP:
            recommendations.append(
                "Tasks sometimes feel harder right now — "
                "totally okay to focus on lighter things"
            )
        elif rhythm.rhythm_type == RhythmType.MOOD_LOW:
            recommendations.append(
                "Mood sometimes dips around this time — "
                "be gentle with yourself"
            )
        elif rhythm.rhythm_type == RhythmType.MOOD_HIGH:
            recommendations.append(
                "You're usually in good spirits now — enjoy it"
            )

    return recommendations
