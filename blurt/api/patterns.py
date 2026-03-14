"""API routes for pattern storage — persist and query learned user rhythms.

Endpoints:
- GET  /api/v1/users/{user_id}/patterns          — Query patterns with filters
- GET  /api/v1/users/{user_id}/patterns/summary   — Pattern summary grouped by type
- GET  /api/v1/users/{user_id}/patterns/{id}       — Get a specific pattern
- POST /api/v1/users/{user_id}/patterns            — Create a new pattern
- PUT  /api/v1/users/{user_id}/patterns/{id}       — Update/reinforce a pattern
- DELETE /api/v1/users/{user_id}/patterns/{id}     — Deactivate a pattern

Anti-shame design: Patterns are neutral observations about user rhythms.
No language implies failure, guilt, or obligation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from blurt.models.entities import LearnedPattern, PatternType
from blurt.services.patterns import PatternService, resolve_pattern_type

router = APIRouter(prefix="/api/v1/users", tags=["patterns"])

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_service: PatternService | None = None


def get_pattern_service() -> PatternService:
    """DI for the pattern service."""
    global _service
    if _service is None:
        _service = PatternService()
    return _service


def set_pattern_service(service: PatternService | None) -> None:
    """Override the service (for testing)."""
    global _service
    _service = service


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CreatePatternRequest(BaseModel):
    """Request to create a new learned pattern."""

    pattern_type: str = Field(
        description=(
            "Type of pattern. Accepts friendly aliases: "
            "'energy', 'mood', 'time', 'day', 'completion', 'skip', 'entity'"
        )
    )
    description: str = Field(
        description="Human-readable description of the pattern (neutral, non-judgmental)"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Pattern-specific parameters. Use 'day_of_week' or 'days' for day filters, "
            "'time_of_day' or 'times' for time filters."
        ),
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Initial confidence level",
    )
    observation_count: int = Field(
        default=1, ge=0,
        description="Number of observations supporting this pattern",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Blurt IDs or text snippets supporting this pattern",
    )


class ReinforcePatternRequest(BaseModel):
    """Request to reinforce a pattern with new evidence."""

    evidence: str | None = Field(
        default=None,
        description="New evidence supporting this pattern",
    )
    confidence_boost: float = Field(
        default=0.05, ge=0.0, le=0.5,
        description="How much to increase confidence",
    )


class WeakenPatternRequest(BaseModel):
    """Request to weaken a pattern due to contradicting evidence."""

    confidence_penalty: float = Field(
        default=0.1, ge=0.0, le=0.5,
        description="How much to decrease confidence",
    )


class PatternResponse(BaseModel):
    """Response representing a single learned pattern."""

    id: str
    user_id: str
    pattern_type: str
    description: str
    parameters: dict[str, Any]
    confidence: float
    observation_count: int
    supporting_evidence: list[str]
    is_active: bool
    first_detected: str
    last_confirmed: str
    created_at: str
    updated_at: str


class PatternListResponse(BaseModel):
    """Response for pattern list queries."""

    patterns: list[PatternResponse]
    total_count: int
    limit: int
    offset: int


class PatternSummaryResponse(BaseModel):
    """Summary of all patterns for a user, grouped by type."""

    total_active: int
    by_type: dict[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def _pattern_to_response(p: LearnedPattern) -> PatternResponse:
    return PatternResponse(
        id=p.id,
        user_id=p.user_id,
        pattern_type=p.pattern_type.value,
        description=p.description,
        parameters=p.parameters,
        confidence=round(p.confidence, 4),
        observation_count=p.observation_count,
        supporting_evidence=p.supporting_evidence,
        is_active=p.is_active,
        first_detected=p.first_detected.isoformat(),
        last_confirmed=p.last_confirmed.isoformat(),
        created_at=p.created_at.isoformat(),
        updated_at=p.updated_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/{user_id}/patterns", response_model=PatternListResponse)
async def list_patterns(
    user_id: str,
    type: str | None = Query(
        None,
        description="Pattern type filter (e.g., 'energy', 'mood', 'time', 'day', 'completion', 'skip', 'entity')",
    ),
    day: str | None = Query(
        None,
        description="Day of week filter (e.g., 'monday', 'thursday')",
    ),
    time: str | None = Query(
        None,
        description="Time of day filter (e.g., 'morning', 'afternoon', 'evening')",
    ),
    min_confidence: float = Query(
        0.0, ge=0.0, le=1.0,
        description="Minimum confidence threshold",
    ),
    active: bool = Query(True, description="Filter by active status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: PatternService = Depends(get_pattern_service),
) -> PatternListResponse:
    """Query learned user patterns with optional filters.

    Examples:
    - GET /api/v1/users/{id}/patterns?type=energy&day=thursday
    - GET /api/v1/users/{id}/patterns?type=mood&min_confidence=0.7
    - GET /api/v1/users/{id}/patterns?day=monday&time=morning

    Patterns are neutral observations about user rhythms.
    They never imply obligation or failure.
    """
    pattern_type: PatternType | None = None
    if type is not None:
        pattern_type = resolve_pattern_type(type)
        if pattern_type is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown pattern type: '{type}'. Valid types: energy, mood, time, day, completion, skip, entity",
            )

    patterns = await service.query_patterns(
        user_id,
        pattern_type=pattern_type,
        day_of_week=day,
        time_of_day=time,
        min_confidence=min_confidence,
        is_active=active,
        limit=limit,
        offset=offset,
    )
    total = await service.count_patterns(user_id, is_active=active)

    return PatternListResponse(
        patterns=[_pattern_to_response(p) for p in patterns],
        total_count=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{user_id}/patterns/summary", response_model=PatternSummaryResponse)
async def pattern_summary(
    user_id: str,
    service: PatternService = Depends(get_pattern_service),
) -> PatternSummaryResponse:
    """Get a summary of all active patterns for a user, grouped by type.

    Returns count and average confidence per pattern type.
    """
    summary = await service.get_pattern_summary(user_id)
    return PatternSummaryResponse(**summary)


@router.get("/{user_id}/patterns/{pattern_id}", response_model=PatternResponse)
async def get_pattern(
    user_id: str,
    pattern_id: str,
    service: PatternService = Depends(get_pattern_service),
) -> PatternResponse:
    """Retrieve a specific pattern by ID."""
    pattern = await service.get_pattern(pattern_id)
    if pattern is None or pattern.user_id != user_id:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return _pattern_to_response(pattern)


@router.post(
    "/{user_id}/patterns",
    response_model=PatternResponse,
    status_code=201,
)
async def create_pattern(
    user_id: str,
    request: CreatePatternRequest,
    service: PatternService = Depends(get_pattern_service),
) -> PatternResponse:
    """Create a new learned pattern.

    Patterns are typically created by the behavioral detection pipeline,
    but can also be created manually for testing or seeding.
    """
    pattern_type = resolve_pattern_type(request.pattern_type)
    if pattern_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pattern type: '{request.pattern_type}'",
        )

    pattern = await service.create_pattern(
        user_id=user_id,
        pattern_type=pattern_type,
        description=request.description,
        parameters=request.parameters,
        confidence=request.confidence,
        observation_count=request.observation_count,
        supporting_evidence=request.supporting_evidence,
    )
    return _pattern_to_response(pattern)


@router.put("/{user_id}/patterns/{pattern_id}/reinforce", response_model=PatternResponse)
async def reinforce_pattern(
    user_id: str,
    pattern_id: str,
    request: ReinforcePatternRequest,
    service: PatternService = Depends(get_pattern_service),
) -> PatternResponse:
    """Reinforce a pattern with new confirming evidence.

    Increases confidence and records the new evidence. This is
    how patterns compound over time — each confirming observation
    strengthens them.
    """
    pattern = await service.get_pattern(pattern_id)
    if pattern is None or pattern.user_id != user_id:
        raise HTTPException(status_code=404, detail="Pattern not found")

    updated = await service.reinforce_pattern(
        pattern_id,
        evidence=request.evidence,
        confidence_boost=request.confidence_boost,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return _pattern_to_response(updated)


@router.put("/{user_id}/patterns/{pattern_id}/weaken", response_model=PatternResponse)
async def weaken_pattern(
    user_id: str,
    pattern_id: str,
    request: WeakenPatternRequest,
    service: PatternService = Depends(get_pattern_service),
) -> PatternResponse:
    """Weaken a pattern due to contradicting evidence.

    Decreases confidence. Patterns below 0.1 confidence are
    auto-deactivated.
    """
    pattern = await service.get_pattern(pattern_id)
    if pattern is None or pattern.user_id != user_id:
        raise HTTPException(status_code=404, detail="Pattern not found")

    updated = await service.weaken_pattern(
        pattern_id,
        confidence_penalty=request.confidence_penalty,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return _pattern_to_response(updated)


@router.delete("/{user_id}/patterns/{pattern_id}", response_model=PatternResponse)
async def deactivate_pattern(
    user_id: str,
    pattern_id: str,
    service: PatternService = Depends(get_pattern_service),
) -> PatternResponse:
    """Deactivate a pattern (soft delete).

    The pattern remains in storage for historical analysis but
    won't be returned in active queries.
    """
    pattern = await service.get_pattern(pattern_id)
    if pattern is None or pattern.user_id != user_id:
        raise HTTPException(status_code=404, detail="Pattern not found")

    deactivated = await service.deactivate_pattern(pattern_id)
    if deactivated is None:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return _pattern_to_response(deactivated)
