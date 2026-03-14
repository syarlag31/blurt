"""API routes for temporal activity aggregation — view interaction patterns by time.

Endpoints:
- GET  /api/v1/users/{user_id}/temporal/profile       — Full temporal profile
- GET  /api/v1/users/{user_id}/temporal/heatmap        — Weekly heatmap (4-slot)
- GET  /api/v1/users/{user_id}/temporal/hourly         — Hour-of-day heatmap (24-hour)
- GET  /api/v1/users/{user_id}/temporal/energy          — Energy patterns
- GET  /api/v1/users/{user_id}/temporal/mood            — Mood/emotion patterns
- POST /api/v1/users/{user_id}/temporal/record          — Record an interaction

Anti-shame design: temporal data is used to surface tasks at optimal times,
never to guilt users about inactivity or missed periods.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from blurt.services.temporal_activity import (
    InteractionRecord,
    TemporalActivityService,
)

router = APIRouter(prefix="/api/v1/users", tags=["temporal-activity"])

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_service: TemporalActivityService | None = None


def get_temporal_service() -> TemporalActivityService:
    """DI for the temporal activity service."""
    global _service
    if _service is None:
        _service = TemporalActivityService()
    return _service


def set_temporal_service(svc: TemporalActivityService | None) -> None:
    """Override service for testing."""
    global _service
    _service = svc


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class RecordInteractionRequest(BaseModel):
    """Request to record a user interaction."""

    energy_level: float = Field(0.5, ge=0.0, le=1.0)
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    primary_emotion: str = "trust"
    emotion_intensity: float = Field(0.0, ge=0.0, le=1.0)
    intent: str = "journal"
    word_count: int = Field(0, ge=0)
    modality: str = "voice"
    task_created: bool = False
    task_completed: bool = False
    task_skipped: bool = False
    task_dismissed: bool = False
    episode_id: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/{user_id}/temporal/profile")
async def get_temporal_profile(
    user_id: str,
    svc: TemporalActivityService = Depends(get_temporal_service),
) -> dict[str, Any]:
    """Get the full temporal activity profile for a user."""
    profile = await svc.get_temporal_profile(user_id)
    return profile.to_dict()


@router.get("/{user_id}/temporal/heatmap")
async def get_weekly_heatmap(
    user_id: str,
    svc: TemporalActivityService = Depends(get_temporal_service),
) -> dict[str, Any]:
    """Get the 28-cell weekly heatmap (7 days x 4 time slots)."""
    profile = await svc.get_temporal_profile(user_id)
    return {
        "user_id": user_id,
        "heatmap": profile.weekly_heatmap(),
        "total_interactions": profile.total_interactions,
    }


@router.get("/{user_id}/temporal/hourly")
async def get_hourly_pattern(
    user_id: str,
    day: str | None = Query(None, description="Filter by day of week"),
    svc: TemporalActivityService = Depends(get_temporal_service),
) -> dict[str, Any]:
    """Get hour-of-day activity patterns (168-cell heatmap or filtered by day)."""
    pattern = await svc.get_hourly_pattern(user_id)
    if day:
        # Filter heatmap to a specific day
        pattern["hourly_heatmap"] = [
            cell for cell in pattern["hourly_heatmap"]
            if cell["day_of_week"] == day
        ]
        profile = await svc.get_temporal_profile(user_id)
        pattern["energy_by_hour"] = profile.energy_by_hour(day=day)
        pattern["peak_hours"] = [
            p for p in pattern["peak_hours"] if p["day"] == day
        ]
    return pattern


@router.get("/{user_id}/temporal/energy")
async def get_energy_pattern(
    user_id: str,
    svc: TemporalActivityService = Depends(get_temporal_service),
) -> dict[str, Any]:
    """Get energy level patterns across the week."""
    return await svc.get_energy_pattern(user_id)


@router.get("/{user_id}/temporal/mood")
async def get_mood_pattern(
    user_id: str,
    svc: TemporalActivityService = Depends(get_temporal_service),
) -> dict[str, Any]:
    """Get mood/emotion patterns across the week."""
    return await svc.get_mood_pattern(user_id)


@router.post("/{user_id}/temporal/record")
async def record_interaction(
    user_id: str,
    body: RecordInteractionRequest,
    svc: TemporalActivityService = Depends(get_temporal_service),
) -> dict[str, Any]:
    """Record a user interaction for temporal aggregation."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    from blurt.services.temporal_activity import hour_to_time_of_day, weekday_to_name

    record = InteractionRecord(
        user_id=user_id,
        timestamp=now,
        day_of_week=weekday_to_name(now.weekday()),
        time_of_day=hour_to_time_of_day(now.hour).value,
        hour=now.hour,
        energy_level=body.energy_level,
        valence=body.valence,
        primary_emotion=body.primary_emotion,
        emotion_intensity=body.emotion_intensity,
        intent=body.intent,
        word_count=body.word_count,
        modality=body.modality,
        task_created=body.task_created,
        task_completed=body.task_completed,
        task_skipped=body.task_skipped,
        task_dismissed=body.task_dismissed,
        episode_id=body.episode_id,
    )
    stored = await svc.record_interaction(record)
    return stored.to_dict()
