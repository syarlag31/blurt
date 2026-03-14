"""API routes for clear-state checking.

When the user asks "what do I need to do?" or the system proactively
checks for pending tasks, these endpoints determine whether to surface
tasks or return a positive "you're all clear" message.

Anti-shame design: "no tasks pending" is always valid and affirmed.

Endpoints:
- POST /api/v1/status/check    — Check if tasks need attention
- GET  /api/v1/status/clear    — Quick clear-state check
"""

from __future__ import annotations


from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from blurt.services.clear_state import (
    ClearStateService,
)
from blurt.services.task_surfacing import (
    EnergyLevel,
    TaskScoringEngine,
    UserContext,
)

router = APIRouter(prefix="/api/v1/status", tags=["status"])

# ---------------------------------------------------------------------------
# Service singletons (DI-managed)
# ---------------------------------------------------------------------------

_clear_service: ClearStateService | None = None
_scoring_engine: TaskScoringEngine | None = None


def get_clear_service() -> ClearStateService:
    """DI for the clear-state service."""
    global _clear_service
    if _clear_service is None:
        _clear_service = ClearStateService()
    return _clear_service


def get_scoring_engine() -> TaskScoringEngine:
    """DI for the task scoring engine."""
    global _scoring_engine
    if _scoring_engine is None:
        _scoring_engine = TaskScoringEngine()
    return _scoring_engine


def set_clear_service(service: ClearStateService) -> None:
    """Override the clear-state service (for testing)."""
    global _clear_service
    _clear_service = service


def set_scoring_engine(engine: TaskScoringEngine) -> None:
    """Override the scoring engine (for testing)."""
    global _scoring_engine
    _scoring_engine = engine


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class StatusCheckRequest(BaseModel):
    """Request to check task status."""

    user_id: str
    energy: str = Field(default="medium", description="low/medium/high")
    current_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    current_arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    time_of_day: str | None = Field(default=None, description="morning/afternoon/evening/night")


class ClearMessageResponse(BaseModel):
    """Response when no tasks need attention."""

    is_clear: bool = True
    message: str
    tone: str
    total_tasks_checked: int = 0
    time_of_day: str | None = None


class StatusCheckResponse(BaseModel):
    """Response from a status check — either clear or has tasks."""

    is_clear: bool
    clear_message: ClearMessageResponse | None = None
    tasks_to_surface: int = 0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/check", response_model=StatusCheckResponse)
async def check_status(
    request: StatusCheckRequest,
    clear_service: ClearStateService = Depends(get_clear_service),
    scoring_engine: TaskScoringEngine = Depends(get_scoring_engine),
) -> StatusCheckResponse:
    """Check whether any tasks need the user's attention.

    If no tasks are pending, returns a positive "you-are-clear" message.
    This is an affirmation, not an absence — being clear is celebrated.

    Anti-shame: never returns guilt language, overdue counts, or pressure.
    """
    # Build user context from request
    energy_map = {
        "low": EnergyLevel.LOW,
        "medium": EnergyLevel.MEDIUM,
        "high": EnergyLevel.HIGH,
    }
    context = UserContext(
        energy=energy_map.get(request.energy, EnergyLevel.MEDIUM),
        current_valence=request.current_valence,
        current_arousal=request.current_arousal,
    )

    # Score with empty task list (real implementation would fetch from store)
    surfacing_result = scoring_engine.score_and_rank(tasks=[], context=context)

    clear_msg = clear_service.check_and_respond(
        surfacing_result=surfacing_result,
        time_of_day=request.time_of_day,
    )

    if clear_msg is not None:
        return StatusCheckResponse(
            is_clear=True,
            clear_message=ClearMessageResponse(
                is_clear=True,
                message=clear_msg.text,
                tone=clear_msg.tone.value,
                total_tasks_checked=clear_msg.total_tasks_checked,
                time_of_day=clear_msg.time_of_day,
            ),
            tasks_to_surface=0,
        )

    # Tasks exist — return count (actual task details handled by surfacing endpoint)
    return StatusCheckResponse(
        is_clear=False,
        tasks_to_surface=len(surfacing_result.tasks),
    )


@router.get("/clear", response_model=ClearMessageResponse)
async def get_clear_message(
    clear_service: ClearStateService = Depends(get_clear_service),
) -> ClearMessageResponse:
    """Get a 'you-are-clear' affirmation.

    Always returns a positive message. Useful for the UI to show
    when the user's task list is empty.

    No-tasks-pending is a valid state. This endpoint celebrates it.
    """
    msg = clear_service.generate()
    return ClearMessageResponse(
        is_clear=True,
        message=msg.text,
        tone=msg.tone.value,
        total_tasks_checked=msg.total_tasks_checked,
        time_of_day=msg.time_of_day,
    )
