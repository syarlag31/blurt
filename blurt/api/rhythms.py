"""API routes for user rhythms — detected temporal behavioral patterns.

Exposes rhythm detection results: energy crashes, creativity peaks,
productivity windows, and mood cycles. Each detected rhythm includes
confidence scores and supporting episode evidence.

Endpoints:
- GET  /api/v1/users/{user_id}/rhythms           — All detected rhythms
- GET  /api/v1/users/{user_id}/rhythms/current    — Rhythms for current moment
- GET  /api/v1/users/{user_id}/rhythms/heatmap    — Weekly energy/mood heatmap
- POST /api/v1/users/{user_id}/rhythms/analyze    — Trigger fresh rhythm analysis
- POST /api/v1/users/{user_id}/rhythms/sync-graph — Sync rhythms into knowledge graph

Anti-shame design: Rhythms are neutral observations. No guilt language,
no streaks, no overdue counters. "Energy tends to dip" not "you fail to".
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from blurt.memory.episodic import EpisodicMemoryStore
from blurt.memory.graph_store import EntityGraphStore
from blurt.models.entities import (
    EntityType,
    FactType,
    LearnedPattern,
    PatternType,
)
from blurt.services.patterns import PatternService
from blurt.services.rhythm import (
    DetectedRhythm,
    RhythmAnalysisResult,
    RhythmDetectionService,
    RhythmType,
    analyze_rhythms,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/users", tags=["rhythms"])

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_rhythm_service: RhythmDetectionService | None = None
_pattern_service: PatternService | None = None
_graph_store: EntityGraphStore | None = None


def get_rhythm_service() -> RhythmDetectionService:
    """DI for the rhythm detection service."""
    global _rhythm_service
    if _rhythm_service is None:
        raise HTTPException(
            status_code=503,
            detail="Rhythm detection service not initialized",
        )
    return _rhythm_service


def set_rhythm_service(service: RhythmDetectionService | None) -> None:
    """Override the rhythm service (for testing)."""
    global _rhythm_service
    _rhythm_service = service


def get_pattern_service() -> PatternService:
    """DI for the pattern service."""
    global _pattern_service
    if _pattern_service is None:
        _pattern_service = PatternService()
    return _pattern_service


def set_pattern_service(service: PatternService | None) -> None:
    """Override the pattern service (for testing)."""
    global _pattern_service
    _pattern_service = service


def get_graph_store() -> EntityGraphStore | None:
    """DI for the knowledge graph store (optional)."""
    return _graph_store


def set_graph_store(store: EntityGraphStore | None) -> None:
    """Override the graph store (for testing)."""
    global _graph_store
    _graph_store = store


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RhythmEvidenceItem(BaseModel):
    """A piece of evidence supporting a rhythm detection."""

    episode_id: str
    description: str = ""


class DetectedRhythmResponse(BaseModel):
    """A single detected behavioral rhythm with confidence and evidence."""

    rhythm_type: str = Field(description="Type of rhythm: energy_crash, energy_peak, creativity_peak, etc.")
    day_of_week: str = Field(description="Day when this rhythm occurs")
    time_of_day: str = Field(description="Time period: morning, afternoon, evening, night")
    description: str = Field(description="Human-readable, shame-free description")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    z_score: float = Field(description="Statistical significance (z-score)")
    metric_value: float = Field(description="The observed metric value for this slot")
    metric_mean: float = Field(description="The population mean of this metric across all slots")
    observation_count: int = Field(description="Number of observations supporting this rhythm")
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Episode IDs that support this rhythm detection",
    )
    pattern_type: str = Field(description="Mapped pattern type for knowledge graph storage")
    is_periodic: bool = Field(default=False, description="Whether this rhythm recurs weekly")
    periodicity_strength: float = Field(default=0.0, description="Strength of weekly recurrence 0-1")
    trend: str = Field(default="stable", description="Trend direction: up, down, or stable")
    weeks_observed: int = Field(default=0, description="Number of weeks of data for this rhythm")


class RhythmSummary(BaseModel):
    """Summary counts of detected rhythms by category."""

    energy_crashes: int = 0
    energy_peaks: int = 0
    creativity_peaks: int = 0
    productivity_windows: int = 0
    productivity_dips: int = 0
    mood_lows: int = 0
    mood_highs: int = 0


class UserRhythmsResponse(BaseModel):
    """Complete rhythm analysis result for a user."""

    user_id: str
    analysis_period_start: str = Field(description="ISO 8601 start of analysis window")
    analysis_period_end: str = Field(description="ISO 8601 end of analysis window")
    total_episodes_analyzed: int
    rhythms: list[DetectedRhythmResponse]
    summary: RhythmSummary
    recommendations: list[str] = Field(
        default_factory=list,
        description="Shame-free, supportive recommendations based on detected rhythms",
    )


class CurrentRhythmResponse(BaseModel):
    """Rhythm context for the current moment."""

    day_of_week: str
    time_of_day: str
    active_rhythms: list[DetectedRhythmResponse]
    bucket_stats: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    """Request to trigger rhythm analysis."""

    lookback_weeks: int = Field(
        default=4, ge=1, le=52,
        description="Number of weeks of history to analyze",
    )


class GraphSyncRequest(BaseModel):
    """Request to sync detected rhythms into the knowledge graph."""

    lookback_weeks: int = Field(
        default=4, ge=1, le=52,
        description="Number of weeks of history to analyze for syncing",
    )
    min_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum confidence threshold for syncing to graph",
    )


class GraphSyncResponse(BaseModel):
    """Result of syncing rhythms to the knowledge graph."""

    patterns_synced: int
    temporal_nodes_created: int
    facts_created: int
    details: list[dict[str, Any]] = Field(default_factory=list)


class HeatmapCell(BaseModel):
    """A single cell in the weekly heatmap."""

    day_of_week: str
    time_of_day: str
    observation_count: int = 0
    mean_valence: float = 0.0
    mean_arousal: float = 0.0
    energy_score: float = 0.0
    completion_rate: float = 0.0
    creativity_ratio: float = 0.0
    active_rhythms: list[str] = Field(default_factory=list)


class WeeklyHeatmapResponse(BaseModel):
    """Weekly heatmap of user rhythms across 28 slots."""

    user_id: str
    cells: list[HeatmapCell]
    total_observations: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from blurt.services.rhythm import RHYTHM_TO_PATTERN_TYPE, _generate_recommendations


def _rhythm_to_response(r: DetectedRhythm) -> DetectedRhythmResponse:
    """Convert a DetectedRhythm to its API response model."""
    # Determine trend direction from the enriched rhythm data
    trend_dir = getattr(r, "trend_direction", 0.0)
    if trend_dir > 0.1:
        trend = "up"
    elif trend_dir < -0.1:
        trend = "down"
    else:
        trend = "stable"

    return DetectedRhythmResponse(
        rhythm_type=r.rhythm_type.value,
        day_of_week=r.day_of_week,
        time_of_day=r.time_of_day,
        description=r.description,
        confidence=round(r.confidence, 4),
        z_score=round(r.z_score, 3),
        metric_value=round(r.metric_value, 4),
        metric_mean=round(r.metric_mean, 4),
        observation_count=r.observation_count,
        supporting_evidence=r.evidence_episode_ids[:20],
        pattern_type=RHYTHM_TO_PATTERN_TYPE[r.rhythm_type].value,
        is_periodic=getattr(r, "is_periodic", False),
        periodicity_strength=round(getattr(r, "periodicity_strength", 0.0), 4),
        trend=trend,
        weeks_observed=getattr(r, "weeks_observed", 0),
    )


def _build_summary(rhythms: list[DetectedRhythm]) -> RhythmSummary:
    """Build a summary of rhythm counts by category."""
    summary = RhythmSummary()
    for r in rhythms:
        if r.rhythm_type == RhythmType.ENERGY_CRASH:
            summary.energy_crashes += 1
        elif r.rhythm_type == RhythmType.ENERGY_PEAK:
            summary.energy_peaks += 1
        elif r.rhythm_type == RhythmType.CREATIVITY_PEAK:
            summary.creativity_peaks += 1
        elif r.rhythm_type == RhythmType.PRODUCTIVITY_WINDOW:
            summary.productivity_windows += 1
        elif r.rhythm_type == RhythmType.PRODUCTIVITY_DIP:
            summary.productivity_dips += 1
        elif r.rhythm_type == RhythmType.MOOD_LOW:
            summary.mood_lows += 1
        elif r.rhythm_type == RhythmType.MOOD_HIGH:
            summary.mood_highs += 1
    return summary


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/{user_id}/rhythms", response_model=UserRhythmsResponse)
async def get_user_rhythms(
    user_id: str,
    lookback_weeks: int = Query(4, ge=1, le=52, description="Weeks of history to analyze"),
    rhythm_type: str | None = Query(None, description="Filter by rhythm type"),
    day: str | None = Query(None, description="Filter by day of week"),
    time: str | None = Query(None, description="Filter by time of day"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    service: RhythmDetectionService = Depends(get_rhythm_service),
) -> UserRhythmsResponse:
    """Get all detected rhythms for a user.

    Returns temporal behavioral patterns (energy crashes, creativity peaks,
    productivity windows, mood cycles) with confidence scores and supporting
    evidence from episodic memory.

    Patterns are neutral observations — never judgmental or guilt-inducing.

    Examples:
    - GET /api/v1/users/{id}/rhythms?rhythm_type=energy_crash
    - GET /api/v1/users/{id}/rhythms?day=thursday&time=afternoon
    - GET /api/v1/users/{id}/rhythms?min_confidence=0.5
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(weeks=lookback_weeks)

    result = await service.analyze_user_rhythms(user_id, start, now)

    # Apply filters
    filtered_rhythms = result.rhythms
    if rhythm_type:
        filtered_rhythms = [r for r in filtered_rhythms if r.rhythm_type.value == rhythm_type]
    if day:
        filtered_rhythms = [r for r in filtered_rhythms if r.day_of_week == day.lower()]
    if time:
        filtered_rhythms = [r for r in filtered_rhythms if r.time_of_day == time.lower()]
    if min_confidence > 0:
        filtered_rhythms = [r for r in filtered_rhythms if r.confidence >= min_confidence]

    recommendations = _generate_recommendations(filtered_rhythms)

    return UserRhythmsResponse(
        user_id=user_id,
        analysis_period_start=result.analysis_period_start.isoformat(),
        analysis_period_end=result.analysis_period_end.isoformat(),
        total_episodes_analyzed=result.total_episodes_analyzed,
        rhythms=[_rhythm_to_response(r) for r in filtered_rhythms],
        summary=_build_summary(filtered_rhythms),
        recommendations=recommendations,
    )


@router.get("/{user_id}/rhythms/current", response_model=CurrentRhythmResponse)
async def get_current_rhythms(
    user_id: str,
    lookback_weeks: int = Query(4, ge=1, le=52),
    service: RhythmDetectionService = Depends(get_rhythm_service),
) -> CurrentRhythmResponse:
    """Get rhythm context for the current moment.

    Returns what rhythms are active right now based on the current day
    and time of day. Used by task surfacing to adjust recommendations.

    Example: On Thursday afternoon, might return:
    - "Energy tends to drop on Thursday afternoon" (confidence: 0.72)
    - Recommendation: "lighter tasks or a break might feel right"
    """
    now = datetime.now(timezone.utc)
    day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    current_day = day_names[now.weekday()]

    hour = now.hour
    if 6 <= hour < 12:
        current_time = "morning"
    elif 12 <= hour < 17:
        current_time = "afternoon"
    elif 17 <= hour < 21:
        current_time = "evening"
    else:
        current_time = "night"

    ctx = await service.get_current_rhythm_context(
        user_id, current_day, current_time, lookback_weeks,
    )

    active_rhythm_responses = []
    for rhythm_data in ctx.get("active_rhythms", []):
        active_rhythm_responses.append(DetectedRhythmResponse(
            rhythm_type=rhythm_data["type"],
            day_of_week=current_day,
            time_of_day=current_time,
            description=rhythm_data["description"],
            confidence=rhythm_data["confidence"],
            z_score=0.0,  # Not available from context summary
            metric_value=0.0,
            metric_mean=0.0,
            observation_count=0,
            supporting_evidence=[],
            pattern_type="",
            is_periodic=rhythm_data.get("is_periodic", False),
            trend=rhythm_data.get("trend", "stable"),
        ))

    return CurrentRhythmResponse(
        day_of_week=current_day,
        time_of_day=current_time,
        active_rhythms=active_rhythm_responses,
        bucket_stats=ctx.get("bucket_stats", {}),
        recommendations=ctx.get("recommendations", []),
    )


@router.post("/{user_id}/rhythms/analyze", response_model=UserRhythmsResponse)
async def analyze_user_rhythms(
    user_id: str,
    request: AnalyzeRequest,
    service: RhythmDetectionService = Depends(get_rhythm_service),
) -> UserRhythmsResponse:
    """Trigger a fresh rhythm analysis for the user.

    Re-analyzes episodic memory over the specified lookback window
    and returns all detected rhythms. Useful for forcing re-detection
    after a period of new data accumulation.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(weeks=request.lookback_weeks)

    result = await service.analyze_user_rhythms(user_id, start, now)
    recommendations = _generate_recommendations(result.rhythms)

    return UserRhythmsResponse(
        user_id=user_id,
        analysis_period_start=result.analysis_period_start.isoformat(),
        analysis_period_end=result.analysis_period_end.isoformat(),
        total_episodes_analyzed=result.total_episodes_analyzed,
        rhythms=[_rhythm_to_response(r) for r in result.rhythms],
        summary=_build_summary(result.rhythms),
        recommendations=recommendations,
    )


@router.post("/{user_id}/rhythms/sync-graph", response_model=GraphSyncResponse)
async def sync_rhythms_to_graph(
    user_id: str,
    request: GraphSyncRequest,
    service: RhythmDetectionService = Depends(get_rhythm_service),
    pattern_service: PatternService = Depends(get_pattern_service),
    graph_store: EntityGraphStore | None = Depends(get_graph_store),
) -> GraphSyncResponse:
    """Sync detected rhythms into the knowledge graph as temporal context nodes.

    For each detected rhythm:
    1. Creates/updates a LearnedPattern in the pattern store
    2. If a graph store is available, creates temporal context entities
       and facts linking rhythms to the knowledge graph

    Temporal context nodes allow the knowledge graph to represent
    time-based patterns (e.g., "Thursday afternoon" is associated with
    "energy crash") alongside entity relationships.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(weeks=request.lookback_weeks)

    result = await service.analyze_user_rhythms(user_id, start, now)

    # Filter by confidence
    qualified_rhythms = [
        r for r in result.rhythms
        if r.confidence >= request.min_confidence
    ]

    patterns_synced = 0
    temporal_nodes_created = 0
    facts_created = 0
    details: list[dict[str, Any]] = []

    for rhythm in qualified_rhythms:
        # 1. Create/update pattern in pattern store
        learned_pattern = rhythm.to_learned_pattern(user_id)
        stored_pattern = await pattern_service.create_pattern(
            user_id=user_id,
            pattern_type=learned_pattern.pattern_type,
            description=learned_pattern.description,
            parameters=learned_pattern.parameters,
            confidence=learned_pattern.confidence,
            observation_count=learned_pattern.observation_count,
            supporting_evidence=learned_pattern.supporting_evidence,
        )
        patterns_synced += 1

        detail: dict[str, Any] = {
            "rhythm_type": rhythm.rhythm_type.value,
            "pattern_id": stored_pattern.id,
            "confidence": round(rhythm.confidence, 4),
        }

        # 2. Sync to knowledge graph if available
        if graph_store is not None:
            node_result = await _sync_rhythm_to_graph(
                graph_store, user_id, rhythm, stored_pattern.id,
            )
            temporal_nodes_created += node_result["nodes_created"]
            facts_created += node_result["facts_created"]
            detail["graph_entity_id"] = node_result.get("entity_id")
            detail["facts"] = node_result.get("fact_ids", [])

        details.append(detail)

    logger.info(
        "Synced %d rhythms to graph for user %s: %d nodes, %d facts",
        patterns_synced, user_id, temporal_nodes_created, facts_created,
    )

    return GraphSyncResponse(
        patterns_synced=patterns_synced,
        temporal_nodes_created=temporal_nodes_created,
        facts_created=facts_created,
        details=details,
    )


@router.get("/{user_id}/rhythms/heatmap", response_model=WeeklyHeatmapResponse)
async def get_rhythm_heatmap(
    user_id: str,
    lookback_weeks: int = Query(4, ge=1, le=52),
    service: RhythmDetectionService = Depends(get_rhythm_service),
) -> WeeklyHeatmapResponse:
    """Get a weekly heatmap of rhythm data across all 28 time slots.

    Each cell (day × time period) contains aggregated metrics and
    any active rhythms detected for that slot. Useful for
    visualization and weekly planning.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(weeks=lookback_weeks)

    result = await service.analyze_user_rhythms(user_id, start, now)

    # Build rhythm index by slot
    rhythm_index: dict[str, list[str]] = {}
    for r in result.rhythms:
        key = f"{r.day_of_week}:{r.time_of_day}"
        rhythm_index.setdefault(key, []).append(r.rhythm_type.value)

    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    times = ["morning", "afternoon", "evening", "night"]

    cells: list[HeatmapCell] = []
    total_obs = 0

    for day in days:
        for time in times:
            key = f"{day}:{time}"
            stats = result.bucket_stats.get(key, {})
            obs = stats.get("observation_count", 0)
            total_obs += obs

            cells.append(HeatmapCell(
                day_of_week=day,
                time_of_day=time,
                observation_count=obs,
                mean_valence=stats.get("mean_valence", 0.0),
                mean_arousal=stats.get("mean_arousal", 0.0),
                energy_score=stats.get("energy_score", 0.0),
                completion_rate=stats.get("completion_rate", 0.0),
                creativity_ratio=stats.get("creativity_ratio", 0.0),
                active_rhythms=rhythm_index.get(key, []),
            ))

    return WeeklyHeatmapResponse(
        user_id=user_id,
        cells=cells,
        total_observations=total_obs,
    )


# ---------------------------------------------------------------------------
# Knowledge graph integration
# ---------------------------------------------------------------------------

# Temporal context entity name format
_TEMPORAL_ENTITY_NAME = "temporal:{day}:{time}"


async def _sync_rhythm_to_graph(
    graph_store: EntityGraphStore,
    user_id: str,
    rhythm: DetectedRhythm,
    pattern_id: str,
) -> dict[str, Any]:
    """Sync a single detected rhythm into the knowledge graph.

    Creates a temporal context entity (e.g., "temporal:thursday:afternoon")
    and adds facts describing the detected rhythm. This allows the
    knowledge graph to link temporal patterns to entities, projects,
    and other graph nodes.

    Temporal context nodes use EntityType.TOPIC to represent time-based
    concepts in the graph without requiring schema changes.
    """
    nodes_created = 0
    facts_created_count = 0
    fact_ids: list[str] = []

    entity_name = _TEMPORAL_ENTITY_NAME.format(
        day=rhythm.day_of_week, time=rhythm.time_of_day,
    )

    # Try to find existing temporal entity, create if missing
    entity = await graph_store.find_entity_by_name(entity_name)
    if entity is None:
        entity = await graph_store.add_entity(
            name=entity_name,
            entity_type=EntityType.TOPIC,
            aliases=[
                f"{rhythm.day_of_week.capitalize()} {rhythm.time_of_day}",
                f"{rhythm.day_of_week} {rhythm.time_of_day}",
            ],
            attributes={
                "temporal_context": True,
                "day_of_week": rhythm.day_of_week,
                "time_of_day": rhythm.time_of_day,
                "node_category": "temporal_rhythm",
            },
        )
        nodes_created += 1

    # Add a fact describing this rhythm observation
    fact_content = (
        f"{rhythm.description} "
        f"(confidence: {rhythm.confidence:.2f}, "
        f"z-score: {rhythm.z_score:.2f}, "
        f"observations: {rhythm.observation_count}, "
        f"pattern_id: {pattern_id})"
    )
    fact = await graph_store.add_fact(
        content=fact_content,
        fact_type=FactType.ATTRIBUTE,
        subject_entity_id=entity.id,
        source_blurt_id=pattern_id,
        confidence=rhythm.confidence,
    )
    facts_created_count += 1
    fact_ids.append(fact.id)

    # Also store the pattern in the graph's pattern system
    await graph_store.add_pattern(
        pattern_type=RHYTHM_TO_PATTERN_TYPE[rhythm.rhythm_type],
        description=rhythm.description,
        parameters={
            "rhythm_type": rhythm.rhythm_type.value,
            "day_of_week": rhythm.day_of_week,
            "time_of_day": rhythm.time_of_day,
            "z_score": round(rhythm.z_score, 3),
            "metric_value": round(rhythm.metric_value, 3),
            "entity_id": entity.id,
        },
        confidence=rhythm.confidence,
        observation_count=rhythm.observation_count,
        supporting_evidence=rhythm.evidence_episode_ids[:20],
    )

    return {
        "entity_id": entity.id,
        "nodes_created": nodes_created,
        "facts_created": facts_created_count,
        "fact_ids": fact_ids,
    }
