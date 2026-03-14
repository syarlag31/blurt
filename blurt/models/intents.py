"""Intent classification types and result models for Blurt.

Defines the 7-intent taxonomy, the schema for classification results
returned by the intent classification service, and intent-specific
metadata models that capture structured data extracted for each intent.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class BlurtIntent(str, Enum):
    """The 7 intent types that Blurt classifies every input into.

    Each intent maps to a distinct processing path:
    - TASK: Actionable items with implicit/explicit deadlines → sync to Notion/task tools
    - EVENT: Calendar-bound occurrences with time/place → sync to Google Calendar
    - REMINDER: Time-triggered nudges without full task weight → lightweight alerts
    - IDEA: Creative thoughts, hypotheses, brainstorms → knowledge graph storage
    - JOURNAL: Reflections, feelings, personal narrative → episodic memory
    - UPDATE: Status changes on existing items → update linked records
    - QUESTION: Queries against personal knowledge or external info → retrieval + answer
    """

    TASK = "task"
    EVENT = "event"
    REMINDER = "reminder"
    IDEA = "idea"
    JOURNAL = "journal"
    UPDATE = "update"
    QUESTION = "question"

    @property
    def description(self) -> str:
        """Human-readable description for prompt engineering."""
        return _INTENT_DESCRIPTIONS[self]

    @property
    def examples(self) -> list[str]:
        """Example utterances for this intent (used in few-shot prompts)."""
        return _INTENT_EXAMPLES[self]


# Detailed descriptions used in the classification prompt
_INTENT_DESCRIPTIONS: dict[BlurtIntent, str] = {
    BlurtIntent.TASK: (
        "An actionable item the user wants to do or needs to complete. "
        "May have an implicit or explicit deadline, assignee, or priority. "
        "Examples: 'I need to buy groceries', 'finish the report by Friday', "
        "'call the dentist'."
    ),
    BlurtIntent.EVENT: (
        "A calendar-bound occurrence with a specific time, date, or place. "
        "Something that happens at a particular moment. "
        "Examples: 'dinner with Sarah at 7pm', 'team standup tomorrow morning', "
        "'flight to NYC on March 20th'."
    ),
    BlurtIntent.REMINDER: (
        "A time-triggered nudge — lighter than a task, just needs a ping. "
        "Often starts with 'remind me' or implies a future notification. "
        "Examples: 'remind me to take my meds at 9pm', "
        "'don't forget to water the plants tomorrow', 'ping me about this later'."
    ),
    BlurtIntent.IDEA: (
        "A creative thought, hypothesis, brainstorm, or conceptual note. "
        "Not immediately actionable — something to capture and connect later. "
        "Examples: 'what if we combined X with Y', 'I think the market is shifting toward...', "
        "'random thought — maybe we should try...'."
    ),
    BlurtIntent.JOURNAL: (
        "A personal reflection, emotional expression, or narrative about the user's "
        "life, feelings, or experiences. Introspective and subjective. "
        "Examples: 'today was really tough', 'I'm feeling grateful for...', "
        "'had an amazing conversation with...', 'I've been thinking about my career'."
    ),
    BlurtIntent.UPDATE: (
        "A status change or progress report on something already tracked. "
        "Modifies existing knowledge rather than creating new items. "
        "Examples: 'actually the meeting moved to 3pm', 'I finished that report', "
        "'the project deadline got extended', 'cancel the dentist appointment'."
    ),
    BlurtIntent.QUESTION: (
        "A query seeking information — either from the user's personal knowledge "
        "graph or general information. Expects an answer or retrieval. "
        "Examples: 'what did I say about that project last week?', "
        "'when is Sarah's birthday?', 'how many tasks do I have this week?'."
    ),
}

# Few-shot examples for each intent (used in the classification prompt)
_INTENT_EXAMPLES: dict[BlurtIntent, list[str]] = {
    BlurtIntent.TASK: [
        "I need to buy groceries this weekend",
        "Finish the quarterly report by Friday",
        "Call the dentist to schedule an appointment",
        "Submit the expense report",
        "Pick up dry cleaning after work",
    ],
    BlurtIntent.EVENT: [
        "Dinner with Sarah at 7pm on Saturday",
        "Team standup tomorrow at 9am",
        "Flight to NYC on March 20th at 6am",
        "Doctor's appointment next Tuesday at 2pm",
        "Conference call with the London team at 3pm GMT",
    ],
    BlurtIntent.REMINDER: [
        "Remind me to take my meds at 9pm",
        "Don't forget to water the plants tomorrow",
        "Ping me about the proposal in two hours",
        "Remind me to follow up with Jake next week",
        "I should remember to check on the deployment tonight",
    ],
    BlurtIntent.IDEA: [
        "What if we combined the recommendation engine with user mood data?",
        "I think the market is shifting toward subscription models",
        "Random thought — maybe we should try a podcast format",
        "It would be cool to build a tool that automatically...",
        "I wonder if we could use embeddings for personal memory",
    ],
    BlurtIntent.JOURNAL: [
        "Today was really tough, the presentation didn't go well",
        "I'm feeling grateful for the support from my team",
        "Had an amazing conversation with my mentor today",
        "I've been thinking about my career direction lately",
        "Feeling energized after that workout this morning",
    ],
    BlurtIntent.UPDATE: [
        "Actually the meeting moved to 3pm",
        "I finished that quarterly report",
        "The project deadline got extended to next month",
        "Cancel the dentist appointment",
        "The grocery list should also include eggs",
    ],
    BlurtIntent.QUESTION: [
        "What did I say about that project last week?",
        "When is Sarah's birthday?",
        "How many tasks do I have this week?",
        "Did I ever finish that book I was reading?",
        "What was the name of that restaurant I liked?",
    ],
}


class IntentClassificationResult(BaseModel):
    """Result of classifying a blurt's intent.

    Attributes:
        intent: The primary classified intent.
        confidence: Confidence score for the primary intent (0.0-1.0).
        secondary_intent: Optional secondary intent if the blurt is ambiguous.
        secondary_confidence: Confidence for the secondary intent.
        raw_text: The original text that was classified.
        reasoning: Brief explanation of why this intent was chosen.
    """

    intent: BlurtIntent = Field(description="Primary classified intent")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for primary intent (0.0-1.0)",
    )
    secondary_intent: BlurtIntent | None = Field(
        default=None,
        description="Optional secondary intent for ambiguous inputs",
    )
    secondary_confidence: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Confidence for secondary intent",
    )
    raw_text: str = Field(description="Original text that was classified")
    reasoning: str = Field(
        default="",
        description="Brief explanation of classification rationale",
    )

    @property
    def is_high_confidence(self) -> bool:
        """Whether the classification confidence exceeds the threshold (0.85)."""
        return self.confidence >= 0.85

    @property
    def is_ambiguous(self) -> bool:
        """Whether the classification is ambiguous (secondary intent is close)."""
        if self.secondary_intent is None:
            return False
        return (self.confidence - self.secondary_confidence) < 0.2


# Intents that trigger external sync operations
SYNCABLE_INTENTS = frozenset({
    BlurtIntent.TASK,
    BlurtIntent.EVENT,
    BlurtIntent.REMINDER,
    BlurtIntent.UPDATE,
})


# Map of intent to its downstream processing actions
INTENT_ACTIONS: dict[BlurtIntent, list[str]] = {
    BlurtIntent.TASK: ["extract_entities", "store_memory", "sync_task_tool"],
    BlurtIntent.EVENT: ["extract_entities", "store_memory", "sync_calendar"],
    BlurtIntent.REMINDER: ["extract_entities", "store_memory", "schedule_reminder"],
    BlurtIntent.IDEA: ["extract_entities", "store_memory", "link_knowledge_graph"],
    BlurtIntent.JOURNAL: ["detect_emotion", "store_episodic", "update_patterns"],
    BlurtIntent.UPDATE: ["resolve_reference", "update_existing", "sync_if_needed"],
    BlurtIntent.QUESTION: ["search_memory", "retrieve_context", "generate_answer"],
}


# ── Intent-Specific Metadata Models ─────────────────────────────────
#
# Each intent type captures different structured data from the user's
# natural speech. These models define what gets extracted per intent
# and flow downstream through the pipeline.
#
# All fields are optional — extraction is incremental and best-effort.
# Missing fields are never an error (anti-shame: partial capture is fine).


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskMetadata(BaseModel):
    """Structured metadata extracted from a TASK intent.

    Captures actionable item details. Priority is always inferred (never
    forced on the user). No overdue counters or guilt language — a task
    with no deadline is perfectly valid.
    """

    intent_type: Literal["task"] = "task"
    action_summary: str = Field(
        default="",
        description="Concise summary of what needs to be done",
    )
    deadline: datetime | None = Field(
        default=None,
        description="Extracted or inferred deadline (never forced)",
    )
    deadline_flexibility: str = Field(
        default="flexible",
        description="How firm the deadline is: 'hard', 'soft', 'flexible'",
    )
    assignee: str | None = Field(
        default=None,
        description="Person responsible (often the user themselves)",
    )
    priority_hint: str | None = Field(
        default=None,
        description="Inferred priority: 'high', 'medium', 'low' — never user-set",
    )
    project_ref: str | None = Field(
        default=None,
        description="Referenced project name or entity ID",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Other tasks/items this depends on",
    )
    sync_targets: list[str] = Field(
        default_factory=list,
        description="External tools to sync to: 'notion', 'todoist', etc.",
    )
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class EventMetadata(BaseModel):
    """Structured metadata extracted from an EVENT intent.

    Captures calendar-bound occurrence details for sync to Google Calendar.
    """

    intent_type: Literal["event"] = "event"
    title: str = Field(default="", description="Event title/summary")
    start_time: datetime | None = Field(
        default=None, description="When the event starts",
    )
    end_time: datetime | None = Field(
        default=None, description="When the event ends",
    )
    duration_minutes: int | None = Field(
        default=None, description="Duration in minutes if end_time not explicit",
    )
    all_day: bool = Field(default=False, description="Whether this is an all-day event")
    location: str | None = Field(default=None, description="Event location")
    attendees: list[str] = Field(
        default_factory=list,
        description="Names of people involved",
    )
    recurrence: str | None = Field(
        default=None,
        description="Recurrence pattern: 'daily', 'weekly', 'monthly', etc.",
    )
    calendar_id: str | None = Field(
        default=None,
        description="Target calendar (if user has multiple)",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class ReminderMetadata(BaseModel):
    """Structured metadata extracted from a REMINDER intent.

    Lighter than a task — just a time-triggered nudge. No recurring
    reminders unless explicitly requested.
    """

    intent_type: Literal["reminder"] = "reminder"
    content: str = Field(default="", description="What to be reminded about")
    trigger_time: datetime | None = Field(
        default=None, description="When to fire the reminder",
    )
    trigger_relative: str | None = Field(
        default=None,
        description="Relative time expression: 'in 2 hours', 'tomorrow morning'",
    )
    recurring: bool = Field(
        default=False, description="Whether this repeats",
    )
    recurrence_pattern: str | None = Field(
        default=None, description="Recurrence if recurring: 'daily at 9pm', etc.",
    )
    linked_entity: str | None = Field(
        default=None, description="Entity this reminder relates to",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class IdeaMetadata(BaseModel):
    """Structured metadata extracted from an IDEA intent.

    Captures creative thoughts for the knowledge graph. Ideas are never
    actionable by default — they're seeds for future connections.
    """

    intent_type: Literal["idea"] = "idea"
    summary: str = Field(default="", description="Core idea in one sentence")
    domain: str | None = Field(
        default=None,
        description="Domain/area: 'product', 'personal', 'technical', etc.",
    )
    related_ideas: list[str] = Field(
        default_factory=list,
        description="References to other ideas or concepts mentioned",
    )
    inspiration_source: str | None = Field(
        default=None,
        description="What inspired this idea (conversation, article, etc.)",
    )
    potential_connections: list[str] = Field(
        default_factory=list,
        description="Entities/projects this could connect to",
    )
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class JournalMetadata(BaseModel):
    """Structured metadata extracted from a JOURNAL intent.

    Captures personal reflections and emotional context. This is the
    safe default intent — capturing a thought is always valuable.
    No judgments, no productivity metrics.
    """

    intent_type: Literal["journal"] = "journal"
    summary: str = Field(default="", description="Brief summary of the entry")
    mood_keywords: list[str] = Field(
        default_factory=list,
        description="Detected mood/emotion keywords from speech",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Topics or themes mentioned",
    )
    people_mentioned: list[str] = Field(
        default_factory=list,
        description="People referenced in the reflection",
    )
    time_context: str | None = Field(
        default=None,
        description="When this happened: 'today', 'this morning', 'last week'",
    )
    energy_level: str | None = Field(
        default=None,
        description="Inferred energy: 'high', 'medium', 'low', 'depleted'",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class UpdateMetadata(BaseModel):
    """Structured metadata extracted from an UPDATE intent.

    Captures status changes on existing items. Updates modify the
    knowledge graph rather than creating new entries.
    """

    intent_type: Literal["update"] = "update"
    update_type: str = Field(
        default="modify",
        description="Type of update: 'modify', 'complete', 'cancel', 'reschedule', 'extend'",
    )
    target_description: str = Field(
        default="",
        description="What is being updated (natural language reference)",
    )
    target_entity_id: str | None = Field(
        default=None,
        description="Resolved entity/item ID if found in knowledge graph",
    )
    changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs of what changed (e.g., {'time': '3pm'})",
    )
    previous_value: str | None = Field(
        default=None,
        description="Previous value if mentioned ('moved FROM 2pm')",
    )
    new_value: str | None = Field(
        default=None,
        description="New value ('moved TO 3pm')",
    )
    sync_required: bool = Field(
        default=False,
        description="Whether this update needs to sync to external tools",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class QuestionMetadata(BaseModel):
    """Structured metadata extracted from a QUESTION intent.

    Captures queries against personal knowledge or external info.
    """

    intent_type: Literal["question"] = "question"
    query_text: str = Field(
        default="",
        description="The core question being asked",
    )
    query_type: str = Field(
        default="recall",
        description="Type: 'recall' (personal memory), 'factual', 'aggregate', 'temporal'",
    )
    time_scope: str | None = Field(
        default=None,
        description="Time scope of the query: 'last week', 'ever', 'this month'",
    )
    entity_references: list[str] = Field(
        default_factory=list,
        description="Entities referenced in the question",
    )
    expected_answer_type: str | None = Field(
        default=None,
        description="Expected answer format: 'date', 'name', 'count', 'narrative'",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


# Union of all intent-specific metadata types
IntentMetadata = Union[
    TaskMetadata,
    EventMetadata,
    ReminderMetadata,
    IdeaMetadata,
    JournalMetadata,
    UpdateMetadata,
    QuestionMetadata,
]


# Map intent enum to its metadata model class
INTENT_METADATA_MODELS: dict[BlurtIntent, type[IntentMetadata]] = {
    BlurtIntent.TASK: TaskMetadata,
    BlurtIntent.EVENT: EventMetadata,
    BlurtIntent.REMINDER: ReminderMetadata,
    BlurtIntent.IDEA: IdeaMetadata,
    BlurtIntent.JOURNAL: JournalMetadata,
    BlurtIntent.UPDATE: UpdateMetadata,
    BlurtIntent.QUESTION: QuestionMetadata,
}


def create_intent_metadata(
    intent: BlurtIntent,
    raw_data: dict[str, Any] | None = None,
) -> IntentMetadata:
    """Create the appropriate intent metadata model for a given intent.

    Args:
        intent: The classified intent type.
        raw_data: Optional raw extracted data to populate the model.
            Unknown keys are captured in the ``extra`` field.

    Returns:
        An instance of the intent-specific metadata model.
    """
    model_cls = INTENT_METADATA_MODELS[intent]
    if raw_data is None:
        return model_cls()

    # Separate known fields from unknown ones
    known_fields = set(model_cls.model_fields.keys()) - {"extra", "intent_type"}
    known: dict[str, Any] = {}
    overflow: dict[str, Any] = {}

    for k, v in raw_data.items():
        if k in known_fields:
            known[k] = v
        elif k not in ("intent_type",):
            overflow[k] = v

    known["extra"] = overflow
    return model_cls.model_validate(known)


# ── Unified Classification Output ───────────────────────────────────


class BlurtClassificationOutput(BaseModel):
    """Unified output of the full classification + extraction pipeline.

    This is the single data structure that flows through the entire
    pipeline: classify → extract → detect emotion → store → surface.
    Every blurt produces exactly one of these — no data loss.

    Combines:
    - Intent classification result (which intent, how confident)
    - Intent-specific metadata (structured data for this intent type)
    - Entity references (people, places, projects mentioned)
    - Emotion detection result reference
    - Pipeline tracking (timing, model used, blurt ID)
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this classification output",
    )
    blurt_id: str = Field(
        default="",
        description="ID of the originating blurt/capture",
    )
    user_id: str = Field(default="", description="User who produced this blurt")

    # Classification result
    primary_intent: BlurtIntent = Field(
        default=BlurtIntent.JOURNAL,
        description="Primary classified intent",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Confidence score for primary intent",
    )
    secondary_intent: BlurtIntent | None = Field(
        default=None,
        description="Secondary intent if ambiguous",
    )
    secondary_confidence: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Confidence for secondary intent",
    )
    all_intent_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for all 7 intents",
    )
    classification_reasoning: str = Field(
        default="",
        description="Brief explanation of classification rationale",
    )

    # Intent-specific metadata (discriminated union)
    intent_metadata: IntentMetadata | None = Field(
        default=None,
        description="Structured metadata specific to the classified intent",
    )

    # Raw input
    raw_text: str = Field(default="", description="Original transcribed text")
    input_source: str = Field(
        default="voice",
        description="Input source: 'voice', 'text', 'correction'",
    )

    # Entity references extracted from this blurt
    entity_ids: list[str] = Field(
        default_factory=list,
        description="IDs of entities mentioned/extracted",
    )
    entity_names: list[str] = Field(
        default_factory=list,
        description="Names of entities (before resolution to IDs)",
    )

    # Emotion detection reference
    emotion_valence: float = Field(
        default=0.0,
        description="Detected emotional valence (-1.0 to 1.0)",
    )
    emotion_arousal: float = Field(
        default=0.5,
        description="Detected emotional arousal (0.0 to 1.0)",
    )
    emotion_label: str = Field(
        default="neutral",
        description="Primary emotion label",
    )

    # Pipeline tracking
    model_used: str = Field(
        default="",
        description="Model that performed classification: 'flash-lite', 'flash'",
    )
    was_escalated: bool = Field(
        default=False,
        description="Whether classification was escalated to the smart model",
    )
    is_multi_intent: bool = Field(
        default=False,
        description="Whether multiple intents were detected",
    )
    sub_intents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sub-intent segments if multi-intent",
    )
    pipeline_latency_ms: float = Field(
        default=0.0,
        description="Total pipeline processing time in milliseconds",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When this output was created",
    )

    @property
    def is_high_confidence(self) -> bool:
        """Whether classification confidence meets the 85% threshold."""
        return self.confidence >= 0.85

    @property
    def is_syncable(self) -> bool:
        """Whether this intent triggers external sync operations."""
        return self.primary_intent in SYNCABLE_INTENTS

    @property
    def downstream_actions(self) -> list[str]:
        """Get the downstream processing actions for this intent."""
        return INTENT_ACTIONS.get(self.primary_intent, [])
