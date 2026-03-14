"""Task surfacing engine — scores and ranks tasks for contextual presentation.

AC 9: Surface the right task based on mood/energy/context.
Anti-shame design: no-tasks-pending is a valid state, never force-surface tasks.
"""

from blurt.services.surfacing.engine import (
    CompositeResult,
    CompositeScoringEngine,
    DimensionScore,
    RankingResult,
    SignalDimension,
    SignalWeights,
)
from blurt.services.surfacing.models import (
    BehavioralProfile,
    CalendarSlot,
    ScoredTask,
    SurfacingContext,
    TaskItem,
    TimePreference,
)
from blurt.services.surfacing.scorers import (
    score_behavioral,
    score_calendar_availability,
    score_energy,
    score_entity_relevance,
    score_mood,
    score_time_of_day,
)
from blurt.services.surfacing.thompson import (
    ArmState,
    ThompsonSampler,
)

__all__ = [
    # Composite Scoring Engine (AC 9 Sub-AC 2)
    "CompositeResult",
    "CompositeScoringEngine",
    "DimensionScore",
    "RankingResult",
    "SignalDimension",
    "SignalWeights",
    # Models
    "BehavioralProfile",
    "CalendarSlot",
    "ScoredTask",
    "SurfacingContext",
    "TaskItem",
    "TimePreference",
    # Scoring functions
    "score_behavioral",
    "score_calendar_availability",
    "score_energy",
    "score_entity_relevance",
    "score_mood",
    "score_time_of_day",
    # Thompson Sampling (AC 11)
    "ArmState",
    "ThompsonSampler",
]
