"""Blurt services — business logic layer."""

from blurt.services.acknowledgment import (
    Acknowledgment,
    AcknowledgmentService,
    AcknowledgmentTone,
    generate_acknowledgment,
    generate_acknowledgment_for_error,
)
from blurt.services.compression import (
    CompressionConfig,
    CompressionResult,
    CompressionTier,
    EpisodicCompressionService,
    FullCompressionResult,
    LocalSummaryGenerator,
    SummaryGenerator,
)
from blurt.services.embedding import (
    EmbeddingService,
    SimilarityMatch,
)
from blurt.services.emotion import (
    EmotionDetectionService,
    parse_emotion_response,
)
from blurt.services.capture import (
    BlurtCapturePipeline,
    CaptureResult,
    CaptureStage,
    CaptureStats,
)
from blurt.services.provider import (
    ServiceProvider,
    detect_environment,
    get_provider,
    reset_provider,
)
from blurt.services.query import (
    DateRangeFilter,
    DateReference,
    QueryExecutor,
    QueryParser,
    QueryResult,
    QueryType,
    ResponseFormatter,
    StructuredQuery,
)
from blurt.services.patterns import (
    InMemoryPatternStore,
    PATTERN_TYPE_ALIASES,
    PatternService,
    PatternStore,
    resolve_pattern_type,
)
from blurt.services.recall import (
    PersonalHistoryRecallEngine,
    QueryUnderstanding,
    RecallConfig,
    RecallResponse,
    RecallResult,
    RecallSourceType,
    SourceContext,
    parse_query,
)
from blurt.services.relationships import (
    RelationshipConfig,
    RelationshipDetectionMode,
    RelationshipDetectionResult,
    RelationshipScore,
    RelationshipTrackingService,
    infer_relationship_type,
)
from blurt.services.rhythm import (
    DetectedRhythm,
    PeriodicityResult,
    RhythmAnalysisResult,
    RhythmBucket,
    RhythmDetectionService,
    RhythmType,
    RollingAverageResult,
    WeeklySlotSample,
    analyze_rhythms,
    compute_autocorrelation,
    compute_rolling_average,
    compute_trend,
)
from blurt.services.clear_state import (
    ClearStateMessage,
    ClearStateService,
    ClearTone,
    generate_clear_message,
    select_clear_tone,
)
from blurt.services.access_control import (
    FREE_QUERY_TYPES,
    PREMIUM_QUERY_TYPES,
    QuestionQueryType,
    QuestionRequest,
    QuestionResult,
    TierCapabilities,
    UserTier,
    classify_question_type,
    format_free_tier_response,
    format_premium_tier_response,
    gate_query_for_tier,
    get_capabilities,
    is_query_allowed,
)
from blurt.services.question import (
    QuestionService,
)
from blurt.services.task_surfacing import (
    EnergyLevel,
    ScoredTask,
    SignalScore,
    SignalType,
    SurfaceableTask,
    SurfacingResult,
    SurfacingWeights,
    TaskScoringEngine,
    TaskStatus,
    UserContext,
)
from blurt.services.task_surfacing_query import (
    InMemoryTaskStore,
    SurfacingQuery,
    SurfacingQueryResult,
    TaskSurfacingQueryService,
)
from blurt.services.surfacing import (
    ArmState,
    BehavioralProfile,
    CalendarSlot,
    CompositeResult,
    CompositeScoringEngine,
    DimensionScore,
    RankingResult,
    SignalDimension,
    SignalWeights,
    SurfacingContext,
    TaskItem,
    ThompsonSampler,
    TimePreference,
    score_behavioral,
    score_calendar_availability,
    score_energy,
    score_entity_relevance,
    score_mood,
    score_time_of_day,
)
from blurt.services.feedback import (
    FeedbackAction,
    FeedbackEvent,
    FeedbackSummary,
    InMemoryFeedbackStore,
    TaskFeedbackService,
    ThompsonParams,
)
from blurt.services.preference_store import (
    InMemoryPreferenceBackend,
    UserPreferenceSnapshot,
    UserPreferenceStore,
)
from blurt.services.thompson_sampling import (
    BetaParams,
    DecayConfig,
    DEFAULT_CATEGORIES,
    FeedbackType,
    FeedbackWeights,
    SamplingResult,
    ThompsonSamplingEngine,
)
from blurt.services.thompson_ranking import (
    RankedTask,
    ThompsonRankingPipeline,
    ThompsonRankingResult,
    ThompsonScoreBreakdown,
)
from blurt.services.classification import (
    ClassificationResponse,
    ClassificationServiceConfig,
    IntentClassificationService,
)
from blurt.services.entity_extraction import (
    EntityExtractionService,
    EntityExtractionStats,
)
from blurt.services.behavioral_signals import (
    BehavioralSignal,
    BehavioralSignalCollector,
    BehavioralSignalStore,
    InMemorySignalStore,
    InteractionStats,
    RewardConfig,
    SignalBatch,
    SignalContext,
    SignalKind,
)
from blurt.services.temporal_activity import (
    HourlyBucket,
    InteractionRecord,
    TemporalActivityService,
    TemporalActivityStore,
    TemporalBucket,
    TemporalProfile,
    TemporalSlot,
    TimeOfDay,
    episode_to_interaction,
    hour_to_time_of_day,
    weekday_to_name,
)

__all__ = [
    # Acknowledgment (AC 13)
    "Acknowledgment",
    "AcknowledgmentService",
    "AcknowledgmentTone",
    "generate_acknowledgment",
    "generate_acknowledgment_for_error",
    # Compression / Episodic Memory (AC 7)
    "CompressionConfig",
    "CompressionResult",
    "CompressionTier",
    "EpisodicCompressionService",
    "FullCompressionResult",
    "LocalSummaryGenerator",
    "SummaryGenerator",
    # Embedding Service (AC 6)
    "EmbeddingService",
    "SimilarityMatch",
    # Emotion Detection (AC 4)
    "EmotionDetectionService",
    "parse_emotion_response",
    # Capture Pipeline (AC 8)
    "BlurtCapturePipeline",
    "CaptureResult",
    "CaptureStage",
    "CaptureStats",
    # Provider / Local-only (AC 17)
    "ServiceProvider",
    "detect_environment",
    "get_provider",
    "reset_provider",
    # Pattern Storage (AC 10)
    "InMemoryPatternStore",
    "PATTERN_TYPE_ALIASES",
    "PatternService",
    "PatternStore",
    "resolve_pattern_type",
    # Query Parser/Executor (AC 14)
    "DateRangeFilter",
    "DateReference",
    "QueryExecutor",
    "QueryParser",
    "QueryResult",
    "QueryType",
    "ResponseFormatter",
    "StructuredQuery",
    # Relationship Tracking (AC 6)
    "RelationshipConfig",
    "RelationshipDetectionMode",
    "RelationshipDetectionResult",
    "RelationshipScore",
    "RelationshipTrackingService",
    "infer_relationship_type",
    # Clear State (AC 15)
    "ClearStateMessage",
    "ClearStateService",
    "ClearTone",
    "generate_clear_message",
    "select_clear_tone",
    # Access Control & Question Service (AC 14)
    "FREE_QUERY_TYPES",
    "PREMIUM_QUERY_TYPES",
    "QuestionQueryType",
    "QuestionRequest",
    "QuestionResult",
    "QuestionService",
    "TierCapabilities",
    "UserTier",
    "classify_question_type",
    "format_free_tier_response",
    "format_premium_tier_response",
    "gate_query_for_tier",
    "get_capabilities",
    "is_query_allowed",
    # Task Surfacing (AC 9)
    "EnergyLevel",
    "ScoredTask",
    "SignalScore",
    "SignalType",
    "SurfaceableTask",
    "SurfacingResult",
    "SurfacingWeights",
    "TaskScoringEngine",
    "TaskStatus",
    "UserContext",
    # Task Surfacing Query (AC 9 Sub-AC 3)
    "InMemoryTaskStore",
    "SurfacingQuery",
    "SurfacingQueryResult",
    "TaskSurfacingQueryService",
    # Personal History Recall (AC 14 Sub-AC 2)
    "PersonalHistoryRecallEngine",
    "QueryUnderstanding",
    "RecallConfig",
    "RecallResponse",
    "RecallResult",
    "RecallSourceType",
    "SourceContext",
    "parse_query",
    # Thompson Sampling (AC 11)
    "ArmState",
    "ThompsonSampler",
    # Composite Scoring Engine (AC 9 Sub-AC 2)
    "CompositeResult",
    "CompositeScoringEngine",
    "DimensionScore",
    "RankingResult",
    "SignalDimension",
    "SignalWeights",
    # Task Surfacing Scorers (AC 9 Sub-AC 1)
    "BehavioralProfile",
    "CalendarSlot",
    "SurfacingContext",
    "TaskItem",
    "TimePreference",
    "score_behavioral",
    "score_calendar_availability",
    "score_energy",
    "score_entity_relevance",
    "score_mood",
    "score_time_of_day",
    # Rhythm Detection (AC 10 Sub-AC 2)
    "DetectedRhythm",
    "PeriodicityResult",
    "RhythmAnalysisResult",
    "RhythmBucket",
    "RhythmDetectionService",
    "RhythmType",
    "RollingAverageResult",
    "WeeklySlotSample",
    "analyze_rhythms",
    "compute_autocorrelation",
    "compute_rolling_average",
    "compute_trend",
    # Task Feedback & Thompson Sampling (AC 11 Sub-AC 3)
    "FeedbackAction",
    "FeedbackEvent",
    "FeedbackSummary",
    "InMemoryFeedbackStore",
    "TaskFeedbackService",
    "ThompsonParams",
    # Thompson Sampling Engine (AC 11 Sub-AC 1)
    "BetaParams",
    "DecayConfig",
    "DEFAULT_CATEGORIES",
    "FeedbackType",
    "FeedbackWeights",
    "SamplingResult",
    "ThompsonSamplingEngine",
    # Thompson Ranking Pipeline (AC 11 Sub-AC 3)
    "RankedTask",
    "ThompsonRankingPipeline",
    "ThompsonRankingResult",
    "ThompsonScoreBreakdown",
    # Intent Classification Service (AC 2)
    "ClassificationResponse",
    "ClassificationServiceConfig",
    "IntentClassificationService",
    # Entity Extraction Service (AC 3)
    "EntityExtractionService",
    "EntityExtractionStats",
    # Temporal Activity Aggregation (AC 10)
    "HourlyBucket",
    "InteractionRecord",
    "TemporalActivityService",
    "TemporalActivityStore",
    "TemporalBucket",
    "TemporalProfile",
    "TemporalSlot",
    "TimeOfDay",
    "episode_to_interaction",
    "hour_to_time_of_day",
    "weekday_to_name",
    # Per-User Preference Persistence (AC 11 Sub-AC 4)
    "InMemoryPreferenceBackend",
    "UserPreferenceSnapshot",
    "UserPreferenceStore",
    # Behavioral Signal Collector (AC 11 Sub-AC 2)
    "BehavioralSignal",
    "BehavioralSignalCollector",
    "BehavioralSignalStore",
    "InMemorySignalStore",
    "InteractionStats",
    "RewardConfig",
    "SignalBatch",
    "SignalContext",
    "SignalKind",
]
