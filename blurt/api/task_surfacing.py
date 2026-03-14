"""API endpoints for task surfacing — query interface for ranked task lists.

Endpoints:
- POST /api/v1/tasks/surface     — Query for surfaced tasks with context
- GET  /api/v1/tasks/surface     — Quick surface with minimal context (query params)
- POST /api/v1/tasks             — Add a task to the surfacing store
- GET  /api/v1/tasks/{task_id}   — Get a specific task

Anti-shame design:
- Empty results are valid. "No tasks pending" is a healthy state.
- No overdue counters, no guilt language, no forced engagement.
- Score breakdowns are transparent — users can see why tasks were surfaced.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from blurt.services.task_surfacing import (
    EnergyLevel,
    SurfaceableTask,
    SurfacingWeights,
)
from blurt.services.task_surfacing_query import (
    SurfacingQuery,
    TaskSurfacingQueryService,
)

router = APIRouter(prefix="/api/v1", tags=["task-surfacing"])

# ---------------------------------------------------------------------------
# Service DI
# ---------------------------------------------------------------------------

_service: TaskSurfacingQueryService | None = None


def get_surfacing_service() -> TaskSurfacingQueryService:
    """DI for the surfacing query service."""
    global _service
    if _service is None:
        _service = TaskSurfacingQueryService()
    return _service


def set_surfacing_service(service: TaskSurfacingQueryService) -> None:
    """Override the surfacing service (for testing)."""
    global _service
    _service = service


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class SurfaceTasksRequest(BaseModel):
    """Request to query for surfaced tasks.

    Provides user context so the scoring engine can rank tasks
    appropriately for the current moment.
    """

    user_id: str = Field(default="", description="User ID for multi-user isolation")

    # User context
    energy: EnergyLevel = Field(
        default=EnergyLevel.MEDIUM,
        description="Current energy level",
    )
    mood_valence: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Current mood (-1 negative to 1 positive)",
    )
    mood_arousal: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Current arousal (0 calm to 1 activated)",
    )
    active_entity_ids: list[str] = Field(
        default_factory=list,
        description="Entity IDs currently in the user's context",
    )
    active_entity_names: list[str] = Field(
        default_factory=list,
        description="Entity names currently in the user's context",
    )
    active_project: str | None = Field(
        default=None,
        description="Currently active project name",
    )
    recent_task_ids: list[str] = Field(
        default_factory=list,
        description="IDs of recently interacted tasks",
    )

    # Filtering
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of tasks to return",
    )
    min_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum composite score threshold",
    )
    include_intents: list[str] | None = Field(
        default=None,
        description="Only include tasks with these intents (None = all)",
    )
    exclude_entity_ids: list[str] | None = Field(
        default=None,
        description="Exclude tasks linked to these entity IDs",
    )
    tags_filter: list[str] | None = Field(
        default=None,
        description="Only include tasks with any of these tags",
    )
    max_energy: EnergyLevel | None = Field(
        default=None,
        description="Cap on task energy requirement",
    )

    # Custom weights (optional override)
    weights: dict[str, float] | None = Field(
        default=None,
        description="Custom signal weights override {signal_name: weight}",
    )


class SignalScoreResponse(BaseModel):
    """A single signal score in the breakdown."""

    signal: str
    value: float
    reason: str


class ScoredTaskResponse(BaseModel):
    """A task with its composite score and signal breakdown."""

    task_id: str
    content: str
    intent: str
    status: str
    composite_score: float
    signal_scores: list[SignalScoreResponse]
    surfacing_reason: str

    # Task metadata
    due_at: datetime | None = None
    estimated_energy: str
    entity_names: list[str] = []
    project: str | None = None
    times_surfaced: int = 0
    created_at: datetime


class SurfaceTasksResponse(BaseModel):
    """Response from a task surfacing query.

    Contains ranked tasks with full score transparency.
    Anti-shame: empty results get a positive message.
    """

    query_id: str
    tasks: list[ScoredTaskResponse]
    returned_count: int
    total_in_store: int
    total_eligible: int
    total_after_filters: int
    total_above_threshold: int
    weights_used: dict[str, float]
    message: str = ""  # Anti-shame message for empty results


class AddTaskRequest(BaseModel):
    """Request to add a task to the surfacing store."""

    content: str = Field(description="Task content/description")
    user_id: str = Field(default="", description="Owner user ID")
    intent: str = Field(default="task", description="Classified intent")
    due_at: datetime | None = Field(default=None, description="Soft due date")
    estimated_energy: EnergyLevel = Field(
        default=EnergyLevel.MEDIUM,
        description="Energy level required",
    )
    estimated_duration_minutes: int | None = Field(
        default=None,
        description="Estimated duration in minutes",
    )
    entity_ids: list[str] = Field(default_factory=list)
    entity_names: list[str] = Field(default_factory=list)
    project: str | None = Field(default=None)
    capture_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    capture_arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddTaskResponse(BaseModel):
    """Response after adding a task."""

    task_id: str
    content: str
    status: str
    message: str = "Task added."


class TaskDetailResponse(BaseModel):
    """Detailed task response."""

    task_id: str
    content: str
    intent: str
    status: str
    due_at: datetime | None = None
    estimated_energy: str
    estimated_duration_minutes: int | None = None
    entity_ids: list[str] = []
    entity_names: list[str] = []
    project: str | None = None
    capture_valence: float = 0.0
    capture_arousal: float = 0.5
    times_surfaced: int = 0
    times_deferred: int = 0
    created_at: datetime
    tags: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scored_task_to_response(scored: Any) -> ScoredTaskResponse:
    """Convert a ScoredTask dataclass to the API response model."""
    task = scored.task
    return ScoredTaskResponse(
        task_id=task.id,
        content=task.content,
        intent=task.intent,
        status=task.status.value,
        composite_score=round(scored.composite_score, 4),
        signal_scores=[
            SignalScoreResponse(
                signal=s.signal.value,
                value=round(s.value, 4),
                reason=s.reason,
            )
            for s in scored.signal_scores
        ],
        surfacing_reason=scored.surfacing_reason,
        due_at=task.due_at,
        estimated_energy=task.estimated_energy.value,
        entity_names=task.entity_names,
        project=task.project,
        times_surfaced=task.times_surfaced,
        created_at=task.created_at,
    )


def _build_weights(weights_dict: dict[str, float] | None) -> SurfacingWeights | None:
    """Build SurfacingWeights from a dict, or None."""
    if not weights_dict:
        return None
    return SurfacingWeights(
        time_relevance=weights_dict.get("time_relevance", 0.25),
        energy_match=weights_dict.get("energy_match", 0.20),
        context_relevance=weights_dict.get("context_relevance", 0.20),
        emotional_alignment=weights_dict.get("emotional_alignment", 0.15),
        momentum=weights_dict.get("momentum", 0.10),
        freshness=weights_dict.get("freshness", 0.10),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/tasks/surface",
    response_model=SurfaceTasksResponse,
)
async def surface_tasks(
    request: SurfaceTasksRequest,
    service: TaskSurfacingQueryService = Depends(get_surfacing_service),
) -> SurfaceTasksResponse:
    """Query for surfaced tasks ranked by composite score.

    Accepts the user's current context (energy, mood, active entities)
    and returns the most relevant tasks with full score breakdowns.

    Anti-shame design:
    - Empty results are valid and expected
    - No overdue language or guilt
    - Score breakdowns are fully transparent
    """
    query = SurfacingQuery(
        user_id=request.user_id,
        energy=request.energy,
        mood_valence=request.mood_valence,
        mood_arousal=request.mood_arousal,
        active_entity_ids=request.active_entity_ids,
        active_entity_names=request.active_entity_names,
        active_project=request.active_project,
        recent_task_ids=request.recent_task_ids,
        max_results=request.max_results,
        min_score=request.min_score,
        include_intents=request.include_intents,
        exclude_entity_ids=request.exclude_entity_ids,
        tags_filter=request.tags_filter,
        max_energy=request.max_energy,
        weights=_build_weights(request.weights),
    )

    result = service.query(query)

    # Strip internal marker from weights
    weights_used = {
        k: v
        for k, v in (result.weights_used or {}).items()
        if k != "_thompson_modulated"
    }

    return SurfaceTasksResponse(
        query_id=result.query_id,
        tasks=[_scored_task_to_response(t) for t in result.tasks],
        returned_count=result.returned_count,
        total_in_store=result.total_in_store,
        total_eligible=result.total_eligible,
        total_after_filters=result.total_after_filters,
        total_above_threshold=result.total_above_threshold,
        weights_used=weights_used,
        message=result.message,
    )


@router.get(
    "/tasks/surface",
    response_model=SurfaceTasksResponse,
)
async def surface_tasks_quick(
    user_id: str = Query(default="", description="User ID"),
    energy: EnergyLevel = Query(default=EnergyLevel.MEDIUM),
    mood_valence: float = Query(default=0.0, ge=-1.0, le=1.0),
    max_results: int = Query(default=5, ge=1, le=20),
    min_score: float = Query(default=0.15, ge=0.0, le=1.0),
    service: TaskSurfacingQueryService = Depends(get_surfacing_service),
) -> SurfaceTasksResponse:
    """Quick surface with minimal context via query parameters.

    Simpler alternative to the POST endpoint for basic queries
    when you just need to surface tasks with minimal context.
    """
    query = SurfacingQuery(
        user_id=user_id,
        energy=energy,
        mood_valence=mood_valence,
        max_results=max_results,
        min_score=min_score,
    )

    result = service.query(query)

    weights_used = {
        k: v
        for k, v in (result.weights_used or {}).items()
        if k != "_thompson_modulated"
    }

    return SurfaceTasksResponse(
        query_id=result.query_id,
        tasks=[_scored_task_to_response(t) for t in result.tasks],
        returned_count=result.returned_count,
        total_in_store=result.total_in_store,
        total_eligible=result.total_eligible,
        total_after_filters=result.total_after_filters,
        total_above_threshold=result.total_above_threshold,
        weights_used=weights_used,
        message=result.message,
    )


@router.post(
    "/tasks",
    response_model=AddTaskResponse,
    status_code=201,
)
async def add_task(
    request: AddTaskRequest,
    service: TaskSurfacingQueryService = Depends(get_surfacing_service),
) -> AddTaskResponse:
    """Add a task to the surfacing store.

    Tasks start as ACTIVE and will be surfaced when contextually appropriate.
    """
    task = SurfaceableTask(
        content=request.content,
        intent=request.intent,
        due_at=request.due_at,
        estimated_energy=request.estimated_energy,
        estimated_duration_minutes=request.estimated_duration_minutes,
        entity_ids=request.entity_ids,
        entity_names=request.entity_names,
        project=request.project,
        capture_valence=request.capture_valence,
        capture_arousal=request.capture_arousal,
        metadata=request.metadata,
    )
    # Attach tags via metadata since SurfaceableTask doesn't have a tags field
    # Tags are stored in metadata for filter matching
    if request.tags:
        task.metadata["tags"] = request.tags

    service.add_task(task, user_id=request.user_id or None)

    return AddTaskResponse(
        task_id=task.id,
        content=task.content,
        status=task.status.value,
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskDetailResponse,
)
async def get_task(
    task_id: str,
    user_id: str = Query(default="", description="User ID"),
    service: TaskSurfacingQueryService = Depends(get_surfacing_service),
) -> TaskDetailResponse:
    """Get details for a specific task."""
    task = service.get_task(task_id, user_id=user_id or None)
    if task is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")

    return TaskDetailResponse(
        task_id=task.id,
        content=task.content,
        intent=task.intent,
        status=task.status.value,
        due_at=task.due_at,
        estimated_energy=task.estimated_energy.value,
        estimated_duration_minutes=task.estimated_duration_minutes,
        entity_ids=task.entity_ids,
        entity_names=task.entity_names,
        project=task.project,
        capture_valence=task.capture_valence,
        capture_arousal=task.capture_arousal,
        times_surfaced=task.times_surfaced,
        times_deferred=task.times_deferred,
        created_at=task.created_at,
        tags=task.metadata.get("tags", []),
    )
