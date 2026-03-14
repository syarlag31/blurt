"""API routes for episodic memory — store and retrieve conversation episodes.

Provides REST endpoints for:
- Creating episodes (append-only)
- Retrieving episodes by ID, user, session
- Querying with filters (time, entity, emotion, intent)
- Entity and emotion timelines
- Semantic similarity search via embeddings
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from blurt.memory.episodic import (
    BehavioralSignal,
    EmotionFilter,
    EmotionSnapshot,
    EntityFilter,
    EntityRef,
    Episode,
    EpisodeContext,
    EpisodeSummary,
    EpisodicMemoryStore,
    InMemoryEpisodicStore,
    InputModality,
    IntentFilter,
    TimeRangeFilter,
    compress_episodes,
)

router = APIRouter(prefix="/api/v1/episodes", tags=["episodic-memory"])

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_store: EpisodicMemoryStore | None = None


def get_store() -> EpisodicMemoryStore:
    """DI for the episodic memory store."""
    global _store
    if _store is None:
        _store = InMemoryEpisodicStore()
    return _store


def set_store(store: EpisodicMemoryStore) -> None:
    """Override the store (for testing or production config)."""
    global _store
    _store = store


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic for validation + serialization)
# ---------------------------------------------------------------------------


class EmotionPayload(BaseModel):
    primary: str = "trust"
    intensity: float = Field(default=0.0, ge=0.0, le=3.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    secondary: str | None = None


class EntityPayload(BaseModel):
    name: str
    entity_type: str
    entity_id: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ContextPayload(BaseModel):
    time_of_day: str = "morning"
    day_of_week: str = "monday"
    session_id: str = ""
    preceding_episode_id: str | None = None
    active_task_id: str | None = None


class CreateEpisodeRequest(BaseModel):
    user_id: str
    raw_text: str
    modality: str = "voice"
    intent: str = "task"
    intent_confidence: float = Field(ge=0.0, le=1.0)
    emotion: EmotionPayload
    entities: list[EntityPayload] = Field(default_factory=list)
    behavioral_signal: str = "none"
    surfaced_task_id: str | None = None
    context: ContextPayload
    embedding: list[float] | None = None


class EpisodeResponse(BaseModel):
    id: str
    user_id: str
    timestamp: str
    raw_text: str
    modality: str
    intent: str
    intent_confidence: float
    emotion: EmotionPayload
    entities: list[EntityPayload]
    behavioral_signal: str
    surfaced_task_id: str | None
    context: ContextPayload
    is_compressed: bool


class EpisodeListResponse(BaseModel):
    episodes: list[EpisodeResponse]
    total_count: int
    limit: int
    offset: int


class EntityTimelineResponse(BaseModel):
    entity_name: str
    episodes: list[EpisodeResponse]
    count: int


class SemanticSearchRequest(BaseModel):
    user_id: str
    query_embedding: list[float]
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0)


class SemanticSearchResult(BaseModel):
    episode: EpisodeResponse
    similarity: float


class SemanticSearchResponse(BaseModel):
    results: list[SemanticSearchResult]
    count: int


class ObservationRequest(BaseModel):
    """Store a raw observation (blurt) — high-level endpoint that creates an episode
    from natural spoken/typed input along with pipeline-extracted metadata."""

    user_id: str
    raw_text: str
    modality: str = "voice"
    intent: str = "task"
    intent_confidence: float = Field(ge=0.0, le=1.0)
    emotion: EmotionPayload
    entities: list[EntityPayload] = Field(default_factory=list)
    behavioral_signal: str = "none"
    surfaced_task_id: str | None = None
    context: ContextPayload
    embedding: list[float] | None = None
    source_working_id: str | None = None


class ObservationResponse(BaseModel):
    """Response from storing an observation."""

    episode: EpisodeResponse
    observation_stored: bool = True
    entity_count: int = 0


class CompressRequest(BaseModel):
    """Request to compress a set of episodes into a summary."""

    user_id: str
    episode_ids: list[str]
    summary_text: str
    embedding: list[float] | None = None


class SummaryResponse(BaseModel):
    """Serialized episode summary."""

    id: str
    user_id: str
    created_at: str
    period_start: str
    period_end: str
    source_episode_ids: list[str]
    episode_count: int
    summary_text: str
    dominant_emotions: list[EmotionPayload]
    entity_mentions: dict[str, int]
    intent_distribution: dict[str, int]
    behavioral_signals: dict[str, int | str]


class SummaryListResponse(BaseModel):
    summaries: list[SummaryResponse]
    count: int


class RecallEntry(BaseModel):
    """A single recall entry — either a raw episode or a compressed summary."""

    entry_type: str  # "episode" or "summary"
    episode: EpisodeResponse | None = None
    summary: SummaryResponse | None = None
    timestamp: str  # for unified chronological sorting


class RecallResponse(BaseModel):
    """Full recall response combining raw episodes and compressed summaries."""

    entries: list[RecallEntry]
    raw_count: int
    summary_count: int
    total_count: int


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def _episode_to_response(ep: Episode) -> EpisodeResponse:
    return EpisodeResponse(
        id=ep.id,
        user_id=ep.user_id,
        timestamp=ep.timestamp.isoformat(),
        raw_text=ep.raw_text,
        modality=ep.modality.value,
        intent=ep.intent,
        intent_confidence=ep.intent_confidence,
        emotion=EmotionPayload(
            primary=ep.emotion.primary,
            intensity=ep.emotion.intensity,
            valence=ep.emotion.valence,
            arousal=ep.emotion.arousal,
            secondary=ep.emotion.secondary,
        ),
        entities=[
            EntityPayload(
                name=e.name,
                entity_type=e.entity_type,
                entity_id=e.entity_id,
                confidence=e.confidence,
            )
            for e in ep.entities
        ],
        behavioral_signal=ep.behavioral_signal.value,
        surfaced_task_id=ep.surfaced_task_id,
        context=ContextPayload(
            time_of_day=ep.context.time_of_day,
            day_of_week=ep.context.day_of_week,
            session_id=ep.context.session_id,
            preceding_episode_id=ep.context.preceding_episode_id,
            active_task_id=ep.context.active_task_id,
        ),
        is_compressed=ep.is_compressed,
    )


def _summary_to_response(s: EpisodeSummary) -> SummaryResponse:
    return SummaryResponse(
        id=s.id,
        user_id=s.user_id,
        created_at=s.created_at.isoformat(),
        period_start=s.period_start.isoformat(),
        period_end=s.period_end.isoformat(),
        source_episode_ids=s.source_episode_ids,
        episode_count=s.episode_count,
        summary_text=s.summary_text,
        dominant_emotions=[
            EmotionPayload(
                primary=e.primary,
                intensity=e.intensity,
                valence=e.valence,
                arousal=e.arousal,
                secondary=getattr(e, "secondary", None),
            )
            for e in s.dominant_emotions
        ],
        entity_mentions=s.entity_mentions,
        intent_distribution=s.intent_distribution,
        behavioral_signals=s.behavioral_signals,
    )


def _request_to_episode(req: CreateEpisodeRequest) -> Episode:
    return Episode(
        user_id=req.user_id,
        raw_text=req.raw_text,
        modality=InputModality(req.modality),
        intent=req.intent,
        intent_confidence=req.intent_confidence,
        emotion=EmotionSnapshot(
            primary=req.emotion.primary,
            intensity=req.emotion.intensity,
            valence=req.emotion.valence,
            arousal=req.emotion.arousal,
            secondary=req.emotion.secondary,
        ),
        entities=[
            EntityRef(
                name=e.name,
                entity_type=e.entity_type,
                entity_id=e.entity_id,
                confidence=e.confidence,
            )
            for e in req.entities
        ],
        behavioral_signal=BehavioralSignal(req.behavioral_signal),
        surfaced_task_id=req.surfaced_task_id,
        context=EpisodeContext(
            time_of_day=req.context.time_of_day,
            day_of_week=req.context.day_of_week,
            session_id=req.context.session_id,
            preceding_episode_id=req.context.preceding_episode_id,
            active_task_id=req.context.active_task_id,
        ),
        embedding=req.embedding,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=EpisodeResponse, status_code=201)
async def create_episode(
    request: CreateEpisodeRequest,
    store: EpisodicMemoryStore = Depends(get_store),
) -> EpisodeResponse:
    """Store a new episode in episodic memory. Append-only."""
    episode = _request_to_episode(request)
    stored = await store.append(episode)
    return _episode_to_response(stored)


@router.get("/{episode_id}", response_model=EpisodeResponse)
async def get_episode(
    episode_id: str,
    store: EpisodicMemoryStore = Depends(get_store),
) -> EpisodeResponse:
    """Retrieve a single episode by ID."""
    episode = await store.get(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return _episode_to_response(episode)


@router.get("/user/{user_id}", response_model=EpisodeListResponse)
async def list_user_episodes(
    user_id: str,
    start: datetime | None = Query(None, description="Start of time range (ISO 8601)"),
    end: datetime | None = Query(None, description="End of time range (ISO 8601)"),
    entity: str | None = Query(None, description="Filter by entity name"),
    emotion: str | None = Query(None, description="Filter by primary emotion"),
    min_intensity: float = Query(0.0, ge=0.0, le=3.0),
    intent: str | None = Query(None, description="Filter by intent type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_compressed: bool = Query(False),
    store: EpisodicMemoryStore = Depends(get_store),
) -> EpisodeListResponse:
    """Query episodes with optional filters."""
    time_filter = TimeRangeFilter(start, end) if start or end else None
    entity_f = EntityFilter(entity_name=entity) if entity else None
    emotion_f = (
        EmotionFilter(primary=emotion, min_intensity=min_intensity)
        if emotion
        else None
    )
    intent_f = IntentFilter(intent) if intent else None

    episodes = await store.query(
        user_id,
        time_range=time_filter,
        entity_filter=entity_f,
        emotion_filter=emotion_f,
        intent_filter=intent_f,
        limit=limit,
        offset=offset,
        include_compressed=include_compressed,
    )
    total = await store.count(user_id)

    return EpisodeListResponse(
        episodes=[_episode_to_response(ep) for ep in episodes],
        total_count=total,
        limit=limit,
        offset=offset,
    )


@router.get("/session/{session_id}", response_model=list[EpisodeResponse])
async def get_session_episodes(
    session_id: str,
    store: EpisodicMemoryStore = Depends(get_store),
) -> list[EpisodeResponse]:
    """Get all episodes in a session, in chronological order."""
    episodes = await store.get_session_episodes(session_id)
    return [_episode_to_response(ep) for ep in episodes]


@router.get("/entity/{user_id}/{entity_name}", response_model=EntityTimelineResponse)
async def get_entity_timeline(
    user_id: str,
    entity_name: str,
    limit: int = Query(20, ge=1, le=100),
    store: EpisodicMemoryStore = Depends(get_store),
) -> EntityTimelineResponse:
    """Get timeline of episodes mentioning a specific entity."""
    episodes = await store.get_entity_timeline(user_id, entity_name, limit=limit)
    return EntityTimelineResponse(
        entity_name=entity_name,
        episodes=[_episode_to_response(ep) for ep in episodes],
        count=len(episodes),
    )


@router.get("/emotions/{user_id}", response_model=list[EpisodeResponse])
async def get_emotion_timeline(
    user_id: str,
    start: datetime = Query(..., description="Start of time range"),
    end: datetime = Query(..., description="End of time range"),
    store: EpisodicMemoryStore = Depends(get_store),
) -> list[EpisodeResponse]:
    """Get episodes in a time range for emotion pattern analysis."""
    episodes = await store.get_emotion_timeline(user_id, start, end)
    return [_episode_to_response(ep) for ep in episodes]


@router.post("/search/semantic", response_model=SemanticSearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    store: EpisodicMemoryStore = Depends(get_store),
) -> SemanticSearchResponse:
    """Search episodes by semantic similarity using embeddings."""
    results = await store.semantic_search(
        request.user_id,
        request.query_embedding,
        limit=request.limit,
        min_similarity=request.min_similarity,
    )
    return SemanticSearchResponse(
        results=[
            SemanticSearchResult(
                episode=_episode_to_response(ep),
                similarity=score,
            )
            for ep, score in results
        ],
        count=len(results),
    )


# ---------------------------------------------------------------------------
# Observation endpoint — high-level blurt intake
# ---------------------------------------------------------------------------


@router.post("/observations", response_model=ObservationResponse, status_code=201)
async def store_observation(
    request: ObservationRequest,
    store: EpisodicMemoryStore = Depends(get_store),
) -> ObservationResponse:
    """Store a raw observation (blurt) as an episodic memory entry.

    This is the primary intake endpoint for the pipeline. After classification,
    entity extraction, and emotion detection, the result is stored here as an
    append-only episode with full metadata.
    """
    episode = Episode(
        user_id=request.user_id,
        raw_text=request.raw_text,
        modality=InputModality(request.modality),
        intent=request.intent,
        intent_confidence=request.intent_confidence,
        emotion=EmotionSnapshot(
            primary=request.emotion.primary,
            intensity=request.emotion.intensity,
            valence=request.emotion.valence,
            arousal=request.emotion.arousal,
            secondary=request.emotion.secondary,
        ),
        entities=[
            EntityRef(
                name=e.name,
                entity_type=e.entity_type,
                entity_id=e.entity_id,
                confidence=e.confidence,
            )
            for e in request.entities
        ],
        behavioral_signal=BehavioralSignal(request.behavioral_signal),
        surfaced_task_id=request.surfaced_task_id,
        context=EpisodeContext(
            time_of_day=request.context.time_of_day,
            day_of_week=request.context.day_of_week,
            session_id=request.context.session_id,
            preceding_episode_id=request.context.preceding_episode_id,
            active_task_id=request.context.active_task_id,
        ),
        embedding=request.embedding,
        source_working_id=request.source_working_id,
    )

    stored = await store.append(episode)
    return ObservationResponse(
        episode=_episode_to_response(stored),
        observation_stored=True,
        entity_count=len(stored.entities),
    )


# ---------------------------------------------------------------------------
# Compression endpoints — compress episodes into summaries
# ---------------------------------------------------------------------------


@router.post("/compress", response_model=SummaryResponse, status_code=201)
async def compress_episodes_endpoint(
    request: CompressRequest,
    store: EpisodicMemoryStore = Depends(get_store),
) -> SummaryResponse:
    """Compress a set of episodes into a summary.

    The raw episodes are preserved but marked as compressed. The summary
    captures aggregated patterns for efficient long-term recall.
    """
    # Resolve episodes by ID
    episodes: list[Episode] = []
    for eid in request.episode_ids:
        ep = await store.get(eid)
        if ep is None:
            raise HTTPException(
                status_code=404,
                detail=f"Episode {eid} not found",
            )
        if ep.user_id != request.user_id:
            raise HTTPException(
                status_code=403,
                detail=f"Episode {eid} does not belong to user {request.user_id}",
            )
        episodes.append(ep)

    if not episodes:
        raise HTTPException(status_code=400, detail="No episodes to compress")

    summary = await compress_episodes(
        store,
        request.user_id,
        episodes,
        request.summary_text,
        request.embedding,
    )
    return _summary_to_response(summary)


@router.get("/summaries/{user_id}", response_model=SummaryListResponse)
async def list_summaries(
    user_id: str,
    start: datetime | None = Query(None, description="Start of time range (ISO 8601)"),
    end: datetime | None = Query(None, description="End of time range (ISO 8601)"),
    store: EpisodicMemoryStore = Depends(get_store),
) -> SummaryListResponse:
    """Retrieve episode summaries for a user, optionally filtered by time range."""
    summaries = await store.get_summaries(user_id, start=start, end=end)
    return SummaryListResponse(
        summaries=[_summary_to_response(s) for s in summaries],
        count=len(summaries),
    )


# ---------------------------------------------------------------------------
# Full recall — raw episodes + compressed summaries
# ---------------------------------------------------------------------------


@router.get("/recall/{user_id}", response_model=RecallResponse)
async def recall(
    user_id: str,
    start: datetime | None = Query(None, description="Start of time range (ISO 8601)"),
    end: datetime | None = Query(None, description="End of time range (ISO 8601)"),
    entity: str | None = Query(None, description="Filter by entity name"),
    intent: str | None = Query(None, description="Filter by intent type"),
    include_compressed: bool = Query(True, description="Include compressed episodes"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    store: EpisodicMemoryStore = Depends(get_store),
) -> RecallResponse:
    """Full recall of personal history — returns both raw episodes and
    compressed summaries in unified chronological order.

    This is the primary retrieval endpoint for enabling full recall of
    personal history. It merges raw (uncompressed) episodes and summaries
    into a single timeline.
    """
    time_filter = TimeRangeFilter(start, end) if start or end else None
    entity_f = EntityFilter(entity_name=entity) if entity else None
    intent_f = IntentFilter(intent) if intent else None

    # Get raw (uncompressed) episodes
    raw_episodes = await store.query(
        user_id,
        time_range=time_filter,
        entity_filter=entity_f,
        intent_filter=intent_f,
        include_compressed=include_compressed,
        limit=limit,
        offset=offset,
    )

    # Get summaries
    summaries = await store.get_summaries(user_id, start=start, end=end)

    # Build unified timeline
    entries: list[RecallEntry] = []

    for ep in raw_episodes:
        entries.append(
            RecallEntry(
                entry_type="episode",
                episode=_episode_to_response(ep),
                timestamp=ep.timestamp.isoformat(),
            )
        )

    for s in summaries:
        entries.append(
            RecallEntry(
                entry_type="summary",
                summary=_summary_to_response(s),
                timestamp=s.period_end.isoformat(),
            )
        )

    # Sort by timestamp descending (newest first)
    entries.sort(key=lambda e: e.timestamp, reverse=True)

    return RecallResponse(
        entries=entries,
        raw_count=len(raw_episodes),
        summary_count=len(summaries),
        total_count=len(entries),
    )
