"""API routes for personal history recall — QUESTION intent handler.

Provides semantic search across all captured conversations, entities,
and knowledge graph nodes. Premium-tier feature that enables full
recall of personal history.

Endpoints:
- POST /api/v1/recall          — Execute a recall query
- GET  /api/v1/recall/stats    — Recall engine statistics
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from blurt.services.recall import (
    PersonalHistoryRecallEngine,
    RecallResponse,
    RecallResult,
    RecallSourceType,
    SourceContext,
)

router = APIRouter(prefix="/api/v1/recall", tags=["recall"])

# ---------------------------------------------------------------------------
# Engine singleton (DI-managed)
# ---------------------------------------------------------------------------

_engine: PersonalHistoryRecallEngine | None = None


def get_engine() -> PersonalHistoryRecallEngine:
    """DI for the recall engine."""
    global _engine
    if _engine is None:
        _engine = PersonalHistoryRecallEngine()
    return _engine


def set_engine(engine: PersonalHistoryRecallEngine) -> None:
    """Override the engine (for testing)."""
    global _engine
    _engine = engine


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class RecallRequest(BaseModel):
    """A recall query — natural language question about personal history.

    Users ask things like:
    - "what did I say about the project last week?"
    - "when is Sarah's birthday?"
    - "what was that restaurant I liked?"

    The engine searches across all captured data to find relevant answers.
    """

    user_id: str
    query: str = Field(
        description="Natural language question to recall from personal history"
    )
    max_results: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    source_filter: list[str] | None = Field(
        default=None,
        description="Filter by source types: episode, entity, fact, pattern, summary",
    )
    time_start: datetime | None = Field(
        default=None,
        description="Only return results after this time (ISO 8601)",
    )
    time_end: datetime | None = Field(
        default=None,
        description="Only return results before this time (ISO 8601)",
    )


class SourceContextResponse(BaseModel):
    """Surrounding context from the source of a recall result."""

    preceding_text: str | None = None
    following_text: str | None = None
    session_id: str | None = None
    session_episode_count: int = 0
    surrounding_entities: list[str] = Field(default_factory=list)
    surrounding_intents: list[str] = Field(default_factory=list)


class QueryUnderstandingResponse(BaseModel):
    """Parsed understanding of the natural language query."""

    original_query: str
    temporal_hint: str | None = None
    temporal_start: str | None = None
    temporal_end: str | None = None
    entity_references: list[str] = Field(default_factory=list)
    is_relationship_query: bool = False
    is_count_query: bool = False
    is_emotion_query: bool = False
    search_text: str = ""


class RecallResultResponse(BaseModel):
    """A single recall result in the API response."""

    source_type: str
    source_id: str
    content: str
    relevance_score: float
    raw_similarity: float
    timestamp: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_context: SourceContextResponse | None = None


class RecallApiResponse(BaseModel):
    """Full response from a recall query."""

    query: str
    results: list[RecallResultResponse] = Field(default_factory=list)
    total_results: int = 0
    sources_searched: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    has_results: bool = False
    entity_context_used: list[str] = Field(default_factory=list)
    query_understanding: QueryUnderstandingResponse | None = None


class RecallStatsResponse(BaseModel):
    """Recall engine statistics."""

    total_queries: int = 0
    total_results_returned: int = 0
    avg_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def _source_context_to_response(ctx: SourceContext) -> SourceContextResponse:
    return SourceContextResponse(
        preceding_text=ctx.preceding_text,
        following_text=ctx.following_text,
        session_id=ctx.session_id,
        session_episode_count=ctx.session_episode_count,
        surrounding_entities=ctx.surrounding_entities,
        surrounding_intents=ctx.surrounding_intents,
    )


def _result_to_response(result: RecallResult) -> RecallResultResponse:
    ctx_resp = None
    if result.source_context is not None:
        ctx_resp = _source_context_to_response(result.source_context)
    return RecallResultResponse(
        source_type=result.source_type.value,
        source_id=result.source_id,
        content=result.content,
        relevance_score=round(result.relevance_score, 4),
        raw_similarity=round(result.raw_similarity, 4),
        timestamp=result.timestamp.isoformat() if result.timestamp else None,
        metadata=result.metadata,
        source_context=ctx_resp,
    )


def _recall_to_response(recall: RecallResponse) -> RecallApiResponse:
    qu_resp = None
    if recall.query_understanding is not None:
        qu = recall.query_understanding
        qu_resp = QueryUnderstandingResponse(
            original_query=qu.original_query,
            temporal_hint=qu.temporal_hint,
            temporal_start=qu.temporal_start.isoformat() if qu.temporal_start else None,
            temporal_end=qu.temporal_end.isoformat() if qu.temporal_end else None,
            entity_references=qu.entity_references,
            is_relationship_query=qu.is_relationship_query,
            is_count_query=qu.is_count_query,
            is_emotion_query=qu.is_emotion_query,
            search_text=qu.search_text,
        )
    return RecallApiResponse(
        query=recall.query,
        results=[_result_to_response(r) for r in recall.results],
        total_results=recall.total_results,
        sources_searched=recall.sources_searched,
        latency_ms=round(recall.latency_ms, 2),
        has_results=recall.has_results,
        entity_context_used=recall.entity_context_used,
        query_understanding=qu_resp,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=RecallApiResponse)
async def recall_history(
    request: RecallRequest,
    engine: PersonalHistoryRecallEngine = Depends(get_engine),
) -> RecallApiResponse:
    """Execute a personal history recall query.

    Searches across all captured conversations, entities, facts, and
    patterns to answer questions about the user's personal history.

    Examples:
    - "what did I say about the project last week?"
    - "when is Sarah's birthday?"
    - "how many times did I mention the gym?"
    - "what was that restaurant recommendation?"

    Returns ranked results from all sources with relevance scoring.
    """
    # Parse source filter
    source_filter: list[RecallSourceType] | None = None
    if request.source_filter:
        source_filter = []
        for sf in request.source_filter:
            try:
                source_filter.append(RecallSourceType(sf))
            except ValueError:
                pass  # Ignore invalid source types

    response = await engine.recall(
        user_id=request.user_id,
        query=request.query,
        max_results=request.max_results,
        source_filter=source_filter or None,
        time_start=request.time_start,
        time_end=request.time_end,
    )

    return _recall_to_response(response)


@router.get("/stats", response_model=RecallStatsResponse)
async def get_recall_stats(
    engine: PersonalHistoryRecallEngine = Depends(get_engine),
) -> RecallStatsResponse:
    """Get recall engine statistics."""
    stats = engine.stats
    return RecallStatsResponse(
        total_queries=stats["total_queries"],
        total_results_returned=stats["total_results_returned"],
        avg_latency_ms=stats["avg_latency_ms"],
    )
