"""API routes for silent classification pipeline.

Every blurt is automatically classified as it flows through the system.
These endpoints expose classification capabilities and pipeline statistics
for monitoring and debugging — the user never interacts with classification directly.

Endpoints:
- POST /api/v1/classify          — Classify text silently (returns classification result)
- POST /api/v1/classify/batch    — Classify multiple texts in one request
- GET  /api/v1/classify/stats    — Classification pipeline statistics
- GET  /api/v1/classify/health   — Classification pipeline health check
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from blurt.classification.models import ClassificationResult
from blurt.classification.pipeline import ClassificationPipeline
from blurt.clients.gemini import GeminiClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/classify", tags=["classification"])

# ---------------------------------------------------------------------------
# Pipeline singleton (DI-managed)
# ---------------------------------------------------------------------------

_pipeline: ClassificationPipeline | None = None


def get_classification_pipeline() -> ClassificationPipeline:
    """DI for the classification pipeline."""
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Classification pipeline not initialized",
        )
    return _pipeline


def set_classification_pipeline(pipeline: ClassificationPipeline) -> None:
    """Override the classification pipeline (for testing or setup)."""
    global _pipeline
    _pipeline = pipeline


def init_classification_pipeline(client: GeminiClient) -> ClassificationPipeline:
    """Initialize the classification pipeline with a Gemini client.

    Called during application startup to wire the pipeline.

    Args:
        client: Gemini API client for model access.

    Returns:
        Configured ClassificationPipeline.
    """
    global _pipeline
    _pipeline = ClassificationPipeline(client)
    return _pipeline


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    """Request to classify a single input text.

    The classification is silent — no user-facing labels or categories.
    This endpoint exists for pipeline integration and debugging.
    """

    text: str = Field(description="Text to classify")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata to attach to the classification",
    )


class BatchClassifyRequest(BaseModel):
    """Request to classify multiple texts in a single batch."""

    texts: list[str] = Field(
        description="List of texts to classify",
        min_length=1,
        max_length=50,
    )


class ClassifyResponse(BaseModel):
    """Response from silent classification. Internal use — never shown to users."""

    id: str
    input_text: str
    primary_intent: str
    confidence: float
    status: str
    all_scores: dict[str, float]
    secondary_intent: str | None = None
    was_ambiguous: bool = False
    is_multi_intent: bool = False
    model_used: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchClassifyResponse(BaseModel):
    """Response from batch classification."""

    results: list[ClassifyResponse]
    total: int
    avg_latency_ms: float


class ClassificationStatsResponse(BaseModel):
    """Classification pipeline statistics for monitoring."""

    total_classified: int
    confident_count: int
    ambiguous_count: int
    low_confidence_count: int
    escalated_count: int
    multi_intent_count: int
    error_count: int
    fallback_to_journal_count: int
    avg_latency_ms: float
    confident_rate: float


class ClassificationHealthResponse(BaseModel):
    """Health status of the classification pipeline."""

    healthy: bool
    pipeline_initialized: bool
    total_classified: int
    error_rate: float
    avg_latency_ms: float


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def _result_to_response(result: ClassificationResult) -> ClassifyResponse:
    """Convert a ClassificationResult to API response."""
    scores_dict = {
        score.intent.value: round(score.confidence, 4)
        for score in result.all_scores
    }

    return ClassifyResponse(
        id=result.id,
        input_text=result.input_text,
        primary_intent=result.primary_intent.value,
        confidence=round(result.confidence, 4),
        status=result.status.value,
        all_scores=scores_dict,
        secondary_intent=(
            result.secondary_intent.value if result.secondary_intent else None
        ),
        was_ambiguous=result.was_ambiguous,
        is_multi_intent=result.is_multi_intent,
        model_used=result.model_used,
        latency_ms=round(result.latency_ms, 2),
        metadata=result.metadata,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=ClassifyResponse)
async def classify_text(
    request: ClassifyRequest,
    pipeline: ClassificationPipeline = Depends(get_classification_pipeline),
) -> ClassifyResponse:
    """Classify a single text input silently.

    This runs the full classification pipeline:
    1. Flash-Lite classifies across all 7 intent types
    2. If confidence < 85%, escalates to Flash for resolution
    3. If still ambiguous, defaults to journal (safe, shame-free fallback)

    The user never sees this classification — it's internal pipeline routing.
    """
    result = await pipeline.classify(request.text, **request.metadata)
    return _result_to_response(result)


@router.post("/batch", response_model=BatchClassifyResponse)
async def batch_classify(
    request: BatchClassifyRequest,
    pipeline: ClassificationPipeline = Depends(get_classification_pipeline),
) -> BatchClassifyResponse:
    """Classify multiple texts in a single request.

    Each text goes through the full classification pipeline independently.
    Useful for bulk processing and testing classification accuracy.
    """
    results: list[ClassifyResponse] = []
    total_latency = 0.0

    for text in request.texts:
        result = await pipeline.classify(text)
        response = _result_to_response(result)
        results.append(response)
        total_latency += response.latency_ms

    avg_latency = total_latency / len(results) if results else 0.0

    return BatchClassifyResponse(
        results=results,
        total=len(results),
        avg_latency_ms=round(avg_latency, 2),
    )


@router.get("/stats", response_model=ClassificationStatsResponse)
async def get_classification_stats(
    pipeline: ClassificationPipeline = Depends(get_classification_pipeline),
) -> ClassificationStatsResponse:
    """Get classification pipeline statistics.

    Monitor classification quality, escalation rates, and latency.
    A high confident_rate (>0.85) indicates good model performance.
    """
    stats = pipeline.stats
    return ClassificationStatsResponse(
        total_classified=stats.total_classified,
        confident_count=stats.confident_count,
        ambiguous_count=stats.ambiguous_count,
        low_confidence_count=stats.low_confidence_count,
        escalated_count=stats.escalated_count,
        multi_intent_count=stats.multi_intent_count,
        error_count=stats.error_count,
        fallback_to_journal_count=stats.fallback_to_journal_count,
        avg_latency_ms=round(stats.avg_latency_ms, 2),
        confident_rate=round(stats.confident_rate, 4),
    )


@router.get("/health", response_model=ClassificationHealthResponse)
async def get_classification_health(
) -> ClassificationHealthResponse:
    """Health check for the classification pipeline.

    Returns whether the pipeline is initialized and operational.
    """
    pipeline = _pipeline
    if pipeline is None:
        return ClassificationHealthResponse(
            healthy=False,
            pipeline_initialized=False,
            total_classified=0,
            error_rate=0.0,
            avg_latency_ms=0.0,
        )

    stats = pipeline.stats
    total = stats.total_classified
    error_rate = stats.error_count / total if total > 0 else 0.0

    return ClassificationHealthResponse(
        healthy=error_rate < 0.1,  # Healthy if error rate < 10%
        pipeline_initialized=True,
        total_classified=total,
        error_rate=round(error_rate, 4),
        avg_latency_ms=round(stats.avg_latency_ms, 2),
    )
