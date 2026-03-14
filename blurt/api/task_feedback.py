"""API endpoints for task feedback — recording user interactions with surfaced tasks.

Endpoints:
- POST /api/v1/tasks/{task_id}/feedback   — Record feedback (accept/dismiss/snooze/complete)
- GET  /api/v1/tasks/{task_id}/feedback    — Get feedback summary for a task
- GET  /api/v1/feedback/recent             — Get recent feedback events for a user

Anti-shame design:
- All actions are shame-free. Dismiss and snooze are respected, not penalized.
- No guilt language in responses ("you haven't completed" etc.).
- "No tasks pending" is valid — never force engagement.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from blurt.services.feedback import (
    FeedbackAction,
    InMemoryFeedbackStore,
    TaskFeedbackService,
)

router = APIRouter(prefix="/api/v1", tags=["task-feedback"])

# ---------------------------------------------------------------------------
# Service DI
# ---------------------------------------------------------------------------

_service: TaskFeedbackService | None = None


def get_feedback_service() -> TaskFeedbackService:
    """DI for the feedback service."""
    global _service
    if _service is None:
        _service = TaskFeedbackService(store=InMemoryFeedbackStore())
    return _service


def set_feedback_service(service: TaskFeedbackService) -> None:
    """Override the feedback service (for testing)."""
    global _service
    _service = service


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class TaskFeedbackRequest(BaseModel):
    """Request to record feedback on a surfaced task.

    All actions are valid and shame-free:
    - accept: User starts working on the task
    - dismiss: User doesn't want this task now — respected, no guilt
    - snooze: User wants it later — will resurface after snooze_minutes
    - complete: User marks the task as done
    """

    user_id: str = Field(description="The user recording the feedback")
    action: FeedbackAction = Field(description="What the user did with the surfaced task")

    # Current user context — used for contextual Thompson Sampling
    mood_valence: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Current mood valence (-1 negative to 1 positive)",
    )
    energy_level: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Current energy level (0 calm to 1 activated)",
    )
    time_of_day: str = Field(
        default="",
        description="Time bucket: morning, afternoon, evening, night",
    )

    # Snooze-specific
    snooze_minutes: int | None = Field(
        default=None,
        ge=1,
        description="Minutes to snooze (only for snooze action)",
    )

    # Task context
    intent: str = Field(
        default="task",
        description="The task's classified intent type",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata",
    )


class TaskFeedbackResponse(BaseModel):
    """Response after recording feedback. Shame-free acknowledgment."""

    event_id: str
    task_id: str
    action: FeedbackAction
    recorded_at: datetime
    thompson_update_applied: bool = True
    context_key: str = ""
    message: str = ""  # Shame-free acknowledgment


class TaskFeedbackSummaryResponse(BaseModel):
    """Aggregated feedback summary for a task."""

    task_id: str
    total_events: int
    accept_count: int
    dismiss_count: int
    snooze_count: int
    complete_count: int
    acceptance_rate: float
    thompson_mean: float
    last_feedback_at: datetime | None


class RecentFeedbackResponse(BaseModel):
    """A single feedback event in the recent list."""

    event_id: str
    task_id: str
    action: FeedbackAction
    context_key: str
    timestamp: datetime
    mood_valence: float
    energy_level: float
    time_of_day: str
    snooze_minutes: int | None = None


# ---------------------------------------------------------------------------
# Shame-free acknowledgment messages
# ---------------------------------------------------------------------------

_ACKNOWLEDGMENTS: dict[FeedbackAction, list[str]] = {
    FeedbackAction.ACCEPT: [
        "Got it, starting on that.",
        "On it.",
        "Noted, good to go.",
    ],
    FeedbackAction.DISMISS: [
        "No problem, moved aside.",
        "Understood, cleared.",
        "Got it.",
    ],
    FeedbackAction.SNOOZE: [
        "Will bring it back soon.",
        "Snoozed, coming back when you're ready.",
        "Deferred for now.",
    ],
    FeedbackAction.COMPLETE: [
        "Done, nice.",
        "Marked complete.",
        "Wrapped up.",
    ],
}


def _pick_acknowledgment(action: FeedbackAction) -> str:
    """Pick a shame-free acknowledgment for the action."""
    import random

    messages = _ACKNOWLEDGMENTS[action]
    return random.choice(messages)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/tasks/{task_id}/feedback",
    response_model=TaskFeedbackResponse,
    status_code=201,
)
async def record_task_feedback(
    task_id: str,
    request: TaskFeedbackRequest,
    service: TaskFeedbackService = Depends(get_feedback_service),
) -> TaskFeedbackResponse:
    """Record user feedback on a surfaced task.

    This triggers Thompson Sampling parameter updates at three levels:
    1. Global task level — overall quality signal for this task
    2. Task + context — how well this task fits the current context
    3. Intent + context — generalized learning for this intent type

    All actions are shame-free:
    - Accept: positive signal (alpha +1)
    - Complete: strong positive signal (alpha +2)
    - Dismiss: negative signal (beta +1) — respected, not punished
    - Snooze: weak negative (beta +0.3) — they want it, just not now
    """
    # Validate snooze_minutes for snooze actions
    if request.action == FeedbackAction.SNOOZE and request.snooze_minutes is None:
        request.snooze_minutes = 30  # Default snooze: 30 minutes

    event = service.record_feedback(
        task_id=task_id,
        user_id=request.user_id,
        action=request.action,
        mood_valence=request.mood_valence,
        energy_level=request.energy_level,
        time_of_day=request.time_of_day,
        snooze_minutes=request.snooze_minutes,
        intent=request.intent,
        metadata=request.metadata,
    )

    return TaskFeedbackResponse(
        event_id=event.id,
        task_id=task_id,
        action=event.action,
        recorded_at=event.timestamp,
        thompson_update_applied=True,
        context_key=event.context_key,
        message=_pick_acknowledgment(event.action),
    )


@router.get(
    "/tasks/{task_id}/feedback",
    response_model=TaskFeedbackSummaryResponse,
)
async def get_task_feedback_summary(
    task_id: str,
    service: TaskFeedbackService = Depends(get_feedback_service),
) -> TaskFeedbackSummaryResponse:
    """Get aggregated feedback summary for a task.

    Shows total interactions, acceptance rate, and Thompson Sampling
    mean (expected value of the Beta distribution).
    """
    summary = service.get_task_summary(task_id)

    return TaskFeedbackSummaryResponse(
        task_id=summary.task_id,
        total_events=summary.total_events,
        accept_count=summary.accept_count,
        dismiss_count=summary.dismiss_count,
        snooze_count=summary.snooze_count,
        complete_count=summary.complete_count,
        acceptance_rate=summary.acceptance_rate,
        thompson_mean=summary.thompson_mean,
        last_feedback_at=summary.last_feedback_at,
    )


@router.get(
    "/feedback/recent",
    response_model=list[RecentFeedbackResponse],
)
async def get_recent_feedback(
    user_id: str,
    limit: int = 20,
    service: TaskFeedbackService = Depends(get_feedback_service),
) -> list[RecentFeedbackResponse]:
    """Get recent feedback events for a user.

    Returns the most recent feedback actions across all tasks,
    ordered by timestamp descending.
    """
    if limit < 1:
        limit = 1
    elif limit > 100:
        limit = 100

    events = service.get_recent_feedback(user_id=user_id, limit=limit)

    return [
        RecentFeedbackResponse(
            event_id=e.id,
            task_id=e.task_id,
            action=e.action,
            context_key=e.context_key,
            timestamp=e.timestamp,
            mood_valence=e.mood_valence,
            energy_level=e.energy_level,
            time_of_day=e.time_of_day,
            snooze_minutes=e.snooze_minutes,
        )
        for e in events
    ]
