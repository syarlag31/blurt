"""API routes for QUESTION intent — tier-gated recall and retrieval.

Endpoints:
- POST /api/v1/question          — Ask a question with tier-aware recall
- GET  /api/v1/question/types    — List available query types for a tier

Free tier gets structured queries (entity lookup, fact lookup, counts).
Premium tier gets full semantic recall, graph traversal, and pattern access.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from blurt.services.access_control import (
    FREE_QUERY_TYPES,
    PREMIUM_QUERY_TYPES,
    QuestionQueryType,
    QuestionRequest,
    QuestionResult,
    UserTier,
)
from blurt.services.question import QuestionService

router = APIRouter(prefix="/api/v1/question", tags=["question"])

# ---------------------------------------------------------------------------
# Service singleton (DI-managed)
# ---------------------------------------------------------------------------

_question_service: QuestionService | None = None


def get_question_service() -> QuestionService:
    """DI for the question service."""
    global _question_service
    if _question_service is None:
        _question_service = QuestionService()
    return _question_service


def set_question_service(service: QuestionService) -> None:
    """Override the question service (for testing)."""
    global _question_service
    _question_service = service


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class QuestionAPIRequest(BaseModel):
    """API request for asking a question."""

    user_id: str
    query: str = Field(description="Natural language question")
    tier: UserTier = Field(
        default=UserTier.FREE,
        description="User subscription tier",
    )
    query_type: QuestionQueryType | None = Field(
        default=None,
        description="Explicit query type (auto-detected if not provided)",
    )
    entity_name: str | None = Field(
        default=None,
        description="Target entity name for lookup queries",
    )
    entity_type: str | None = Field(
        default=None,
        description="Filter by entity type",
    )
    max_results: int | None = Field(
        default=None, ge=1, le=200,
        description="Max results (capped by tier)",
    )
    time_range_days: int | None = Field(
        default=None, ge=1,
        description="Look back N days (capped by tier)",
    )


class QuestionAPIResponse(BaseModel):
    """API response from a question query."""

    query: str
    query_type: str
    tier: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    result_count: int = 0
    total_available: int = 0
    truncated: bool = False
    answer_summary: str = ""

    # Premium-only fields
    source_episodes: list[str] = Field(default_factory=list)
    confidence_scores: list[float] = Field(default_factory=list)
    relationship_context: list[dict[str, Any]] = Field(default_factory=list)

    # Upgrade hint (anti-shame)
    upgrade_hint: str | None = None


class QueryTypesResponse(BaseModel):
    """List of available query types for a tier."""

    tier: str
    available_types: list[dict[str, str]]
    total_types: int


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def _result_to_response(result: QuestionResult) -> QuestionAPIResponse:
    return QuestionAPIResponse(
        query=result.query,
        query_type=result.query_type.value,
        tier=result.tier.value,
        results=result.results,
        result_count=result.result_count,
        total_available=result.total_available,
        truncated=result.truncated,
        answer_summary=result.answer_summary,
        source_episodes=result.source_episodes,
        confidence_scores=result.confidence_scores,
        relationship_context=result.relationship_context,
        upgrade_hint=result.upgrade_hint,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=QuestionAPIResponse, status_code=200)
async def ask_question(
    request: QuestionAPIRequest,
    service: QuestionService = Depends(get_question_service),
) -> QuestionAPIResponse:
    """Ask a question against your personal knowledge graph.

    Free tier: Structured queries (entity/fact lookup, counts).
    Premium tier: Full semantic recall, graph traversal, patterns.

    The response always contains useful results — free-tier users
    never see empty "upgrade" walls.
    """
    question_request = QuestionRequest(
        user_id=request.user_id,
        query=request.query,
        query_type=request.query_type,
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        max_results=request.max_results,
        time_range_days=request.time_range_days,
    )

    result = await service.answer(
        request=question_request,
        tier=request.tier,
    )

    return _result_to_response(result)


@router.get("/types", response_model=QueryTypesResponse)
async def get_query_types(
    tier: UserTier = Query(default=UserTier.FREE, description="User tier"),
) -> QueryTypesResponse:
    """List available query types for the given tier."""
    type_descriptions = {
        QuestionQueryType.ENTITY_LOOKUP: "Look up entities by name or type",
        QuestionQueryType.FACT_LOOKUP: "Look up facts about entities",
        QuestionQueryType.RECENT_FACTS: "Get recently learned facts",
        QuestionQueryType.COUNT_QUERY: "Count entities or facts",
        QuestionQueryType.SEMANTIC_RECALL: "Full semantic memory recall",
        QuestionQueryType.GRAPH_QUERY: "Explore knowledge graph connections",
        QuestionQueryType.TEMPORAL_RECALL: "Time-scoped memory recall",
        QuestionQueryType.PATTERN_QUERY: "Behavioral pattern insights",
        QuestionQueryType.NEIGHBORHOOD: "Knowledge graph neighborhood search",
    }

    if tier in (UserTier.PREMIUM, UserTier.TEAM):
        available = PREMIUM_QUERY_TYPES
    else:
        available = FREE_QUERY_TYPES

    types_list = [
        {"type": qt.value, "description": type_descriptions.get(qt, "")}
        for qt in available
    ]

    return QueryTypesResponse(
        tier=tier.value,
        available_types=types_list,
        total_types=len(types_list),
    )
