"""Structured query parser and executor for QUESTION intent.

Parses natural language questions into structured queries against the
knowledge graph, supporting filters by entity type, date range, and topic.
Executes queries against both the semantic memory (knowledge graph) and
episodic memory to assemble comprehensive answers.

This is the free-tier query path: no LLM reasoning, just structured
retrieval from the user's personal knowledge. The query parser uses
pattern matching and keyword extraction to decompose questions into
filters that map directly to store operations.

Design decisions:
- No LLM required for parsing (free tier)
- Composable filters: entity type, date range, topic, relationship
- Falls back to semantic search when structured parsing fails
- Results ranked by relevance (recency + mention count + similarity)
- Shame-free: no "you forgot" or "overdue" language in results
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from blurt.memory.episodic import (
    EntityFilter,
    Episode,
    EpisodicMemoryStore,
    IntentFilter,
    TimeRangeFilter,
)
from blurt.memory.graph_store import EntityGraphStore
from blurt.models.entities import (
    EntityNode,
    EntityType,
    Fact,
    RelationshipEdge,
    SemanticSearchResult,
)

logger = logging.getLogger(__name__)


# ── Query Models ──────────────────────────────────────────────────────


class QueryType(str, Enum):
    """Types of structured queries that can be parsed from questions."""

    ENTITY_LOOKUP = "entity_lookup"  # "who is Sarah?", "what is project X?"
    ENTITY_LIST = "entity_list"  # "who do I work with?", "what projects do I have?"
    FACT_RECALL = "fact_recall"  # "when is Sarah's birthday?", "what did I say about X?"
    TIMELINE = "timeline"  # "what happened last week?", "what did I do yesterday?"
    RELATIONSHIP = "relationship"  # "how is Sarah connected to project X?"
    COUNT = "count"  # "how many tasks do I have?", "how many people..."
    SEMANTIC = "semantic"  # fallback: free-text semantic search


class DateReference(str, Enum):
    """Relative date references parsed from natural language."""

    TODAY = "today"
    TOMORROW = "tomorrow"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    CUSTOM = "custom"


@dataclass
class DateRangeFilter:
    """Parsed date range for temporal queries."""

    reference: DateReference = DateReference.CUSTOM
    start: datetime | None = None
    end: datetime | None = None

    @staticmethod
    def from_reference(ref: DateReference, now: datetime | None = None) -> DateRangeFilter:
        """Create a DateRangeFilter from a relative reference."""
        now = now or datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if ref == DateReference.TODAY:
            return DateRangeFilter(
                reference=ref,
                start=today_start,
                end=now,
            )
        elif ref == DateReference.TOMORROW:
            tomorrow_start = today_start + timedelta(days=1)
            tomorrow_end = tomorrow_start + timedelta(days=1)
            return DateRangeFilter(
                reference=ref,
                start=tomorrow_start,
                end=tomorrow_end,
            )
        elif ref == DateReference.YESTERDAY:
            yesterday = today_start - timedelta(days=1)
            return DateRangeFilter(
                reference=ref,
                start=yesterday,
                end=today_start,
            )
        elif ref == DateReference.THIS_WEEK:
            # Monday of current week
            days_since_monday = now.weekday()
            week_start = today_start - timedelta(days=days_since_monday)
            return DateRangeFilter(
                reference=ref,
                start=week_start,
                end=now,
            )
        elif ref == DateReference.LAST_WEEK:
            days_since_monday = now.weekday()
            this_week_start = today_start - timedelta(days=days_since_monday)
            last_week_start = this_week_start - timedelta(days=7)
            return DateRangeFilter(
                reference=ref,
                start=last_week_start,
                end=this_week_start,
            )
        elif ref == DateReference.THIS_MONTH:
            month_start = today_start.replace(day=1)
            return DateRangeFilter(
                reference=ref,
                start=month_start,
                end=now,
            )
        elif ref == DateReference.LAST_MONTH:
            this_month_start = today_start.replace(day=1)
            if this_month_start.month == 1:
                last_month_start = this_month_start.replace(
                    year=this_month_start.year - 1, month=12
                )
            else:
                last_month_start = this_month_start.replace(
                    month=this_month_start.month - 1
                )
            return DateRangeFilter(
                reference=ref,
                start=last_month_start,
                end=this_month_start,
            )
        else:
            return DateRangeFilter(reference=ref)


@dataclass
class StructuredQuery:
    """A parsed structured query from a natural language question.

    Contains the decomposed filters and query type that map directly
    to store operations. The parser populates this from free text,
    and the executor runs it against the knowledge graph + episodic memory.
    """

    query_type: QueryType = QueryType.SEMANTIC
    raw_text: str = ""

    # Entity filters
    entity_names: list[str] = field(default_factory=list)
    entity_type_filter: EntityType | None = None

    # Date filters
    date_range: DateRangeFilter | None = None

    # Topic / keyword filters
    topics: list[str] = field(default_factory=list)

    # Intent filter (for episodic queries)
    intent_filter: str | None = None

    # Limit
    limit: int = 10

    @property
    def has_entity_filter(self) -> bool:
        return bool(self.entity_names) or self.entity_type_filter is not None

    @property
    def has_date_filter(self) -> bool:
        return self.date_range is not None

    @property
    def has_topic_filter(self) -> bool:
        return bool(self.topics)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/debugging."""
        result: dict[str, Any] = {
            "query_type": self.query_type.value,
            "raw_text": self.raw_text,
        }
        if self.entity_names:
            result["entity_names"] = self.entity_names
        if self.entity_type_filter:
            result["entity_type_filter"] = self.entity_type_filter.value
        if self.date_range:
            result["date_range"] = {
                "reference": self.date_range.reference.value,
                "start": self.date_range.start.isoformat() if self.date_range.start else None,
                "end": self.date_range.end.isoformat() if self.date_range.end else None,
            }
        if self.topics:
            result["topics"] = self.topics
        if self.intent_filter:
            result["intent_filter"] = self.intent_filter
        result["limit"] = self.limit
        return result


# ── Query Results ─────────────────────────────────────────────────────


@dataclass
class QueryResult:
    """Result from executing a structured query.

    Aggregates results from knowledge graph and episodic memory,
    ranked by relevance. Provides both raw data and a natural language
    summary for the response pipeline.
    """

    query: StructuredQuery
    entities: list[EntityNode] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    relationships: list[RelationshipEdge] = field(default_factory=list)
    semantic_results: list[SemanticSearchResult] = field(default_factory=list)
    total_results: int = 0
    execution_time_ms: float = 0.0

    @property
    def has_results(self) -> bool:
        return self.total_results > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "query": self.query.to_dict(),
            "entities": [
                e.model_dump(exclude={"embedding"}) for e in self.entities
            ],
            "facts": [f.model_dump(exclude={"embedding"}) for f in self.facts],
            "episodes": [ep.to_dict() for ep in self.episodes],
            "relationships": [r.model_dump() for r in self.relationships],
            "semantic_results": [
                s.model_dump() for s in self.semantic_results
            ],
            "total_results": self.total_results,
            "execution_time_ms": self.execution_time_ms,
        }


# ── Patterns for Parsing ─────────────────────────────────────────────

# Entity type keywords
_ENTITY_TYPE_KEYWORDS: dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "people": EntityType.PERSON,
    "who": EntityType.PERSON,
    "place": EntityType.PLACE,
    "places": EntityType.PLACE,
    "where": EntityType.PLACE,
    "project": EntityType.PROJECT,
    "projects": EntityType.PROJECT,
    "organization": EntityType.ORGANIZATION,
    "organizations": EntityType.ORGANIZATION,
    "company": EntityType.ORGANIZATION,
    "companies": EntityType.ORGANIZATION,
    "org": EntityType.ORGANIZATION,
    "orgs": EntityType.ORGANIZATION,
    "topic": EntityType.TOPIC,
    "topics": EntityType.TOPIC,
    "tool": EntityType.TOOL,
    "tools": EntityType.TOOL,
}

# Date reference patterns
_DATE_PATTERNS: list[tuple[re.Pattern[str], DateReference]] = [
    (re.compile(r"\btoday\b", re.IGNORECASE), DateReference.TODAY),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), DateReference.TOMORROW),
    (re.compile(r"\byesterday\b", re.IGNORECASE), DateReference.YESTERDAY),
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), DateReference.THIS_WEEK),
    (re.compile(r"\blast\s+week\b", re.IGNORECASE), DateReference.LAST_WEEK),
    (re.compile(r"\bthis\s+month\b", re.IGNORECASE), DateReference.THIS_MONTH),
    (re.compile(r"\blast\s+month\b", re.IGNORECASE), DateReference.LAST_MONTH),
]

# Intent keywords for filtering episodic memory
_INTENT_KEYWORDS: dict[str, str] = {
    "task": "task",
    "tasks": "task",
    "event": "event",
    "events": "event",
    "meeting": "event",
    "meetings": "event",
    "reminder": "reminder",
    "reminders": "reminder",
    "idea": "idea",
    "ideas": "idea",
    "journal": "journal",
    "reflection": "journal",
    "reflections": "journal",
    "update": "update",
    "updates": "update",
}

# Count query patterns
_COUNT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bhow\s+many\b", re.IGNORECASE),
    re.compile(r"\bcount\b", re.IGNORECASE),
    re.compile(r"\bnumber\s+of\b", re.IGNORECASE),
]

# Relationship query patterns
_RELATIONSHIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bhow\s+(?:is|are)\s+(.+?)\s+(?:connected|related|linked)\s+to\s+(.+)", re.IGNORECASE),
    re.compile(r"\brelationship\s+between\s+(.+?)\s+and\s+(.+)", re.IGNORECASE),
    re.compile(r"\bconnection\s+between\s+(.+?)\s+and\s+(.+)", re.IGNORECASE),
]

# Entity lookup patterns (who/what is X)
_LOOKUP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwho\s+is\s+(.+?)[\?\.]?\s*$", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+(.+?)[\?\.]?\s*$", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\s+(.+?)[\?\.]?\s*$", re.IGNORECASE),
]

# Timeline patterns (these should NOT match "what did I say about X" — that's fact recall)
_TIMELINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhat\s+happened\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+did\s+I\s+do\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+was\s+(?:going\s+on|happening)\b", re.IGNORECASE),
    re.compile(r"\bshow\s+me\s+(?:everything|what)\b", re.IGNORECASE),
]

# Fact recall patterns
_FACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhen\s+is\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:did\s+I\s+(?:say|mention|note)\s+about)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+I\s+(?:know|have)\b", re.IGNORECASE),
    re.compile(r"\bdid\s+I\s+(?:ever|already)\b", re.IGNORECASE),
]

# "Who did I talk to about X?" patterns — entity + topic intersection
_ENTITY_TOPIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwho\s+did\s+I\s+(?:talk|speak|chat|discuss|meet)\s+(?:to|with)\s+about\s+(.+?)[\?\.]?\s*$", re.IGNORECASE),
    re.compile(r"\bwho\s+(?:was\s+I|have\s+I\s+been)\s+(?:talking|speaking|meeting)\s+(?:to|with)\s+about\s+(.+?)[\?\.]?\s*$", re.IGNORECASE),
    re.compile(r"\bwho\s+(?:knows|mentioned|brought\s+up)\s+(.+?)[\?\.]?\s*$", re.IGNORECASE),
]

# "What [intent] do I have [date]?" patterns — intent-scoped temporal queries
_INTENT_TEMPORAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhat\s+(meetings?|events?|tasks?|reminders?|ideas?)\s+(?:do|did|will)\s+I\s+have\b", re.IGNORECASE),
    re.compile(r"\b(?:do|did|will)\s+I\s+have\s+(?:any\s+)?(meetings?|events?|tasks?|reminders?)\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?(?:my\s+)?(meetings?|events?|tasks?|reminders?|ideas?)\b", re.IGNORECASE),
    re.compile(r"\blist\s+(?:my\s+)?(meetings?|events?|tasks?|reminders?|ideas?)\b", re.IGNORECASE),
]

# Stop words to filter from topic extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "it", "they", "them",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "when", "where", "why", "how", "all", "any", "both", "each",
    "about", "with", "from", "for", "of", "on", "in", "to", "at",
    "by", "and", "or", "but", "not", "if", "so", "just", "also",
    "than", "too", "very", "many", "much", "some", "no", "there",
    "ever", "already", "tell", "show", "give", "get", "know", "say",
    "said", "mentioned", "happened",
})


# ── Query Parser ──────────────────────────────────────────────────────


class QueryParser:
    """Parses natural language questions into structured queries.

    Uses pattern matching and keyword extraction — no LLM required.
    This is the free-tier parsing path. Premium tier uses Flash for
    more nuanced question understanding.

    The parser applies rules in priority order:
    1. Relationship queries (explicit relationship patterns)
    2. Count queries ("how many...")
    3. Entity lookup queries ("who is...", "what is...")
    4. Timeline queries ("what happened...")
    5. Fact recall queries ("when is...", "what did I say about...")
    6. Entity list queries (entity type keywords without specific names)
    7. Semantic fallback (free-text search)
    """

    def __init__(self, now: datetime | None = None):
        """Initialize the parser.

        Args:
            now: Override for current time (useful for testing).
        """
        self._now = now

    @property
    def now(self) -> datetime:
        return self._now or datetime.now(timezone.utc)

    def parse(self, question: str) -> StructuredQuery:
        """Parse a natural language question into a StructuredQuery.

        Args:
            question: The raw question text from the user.

        Returns:
            A StructuredQuery with decomposed filters and query type.
        """
        question = question.strip()
        if not question:
            return StructuredQuery(raw_text=question)

        query = StructuredQuery(raw_text=question)

        # Extract date range first (applies to all query types)
        query.date_range = self._extract_date_range(question)

        # Extract intent filter
        query.intent_filter = self._extract_intent_filter(question)

        # Try each query type in priority order
        if self._try_parse_relationship(question, query):
            pass
        elif self._try_parse_count(question, query):
            pass
        elif self._try_parse_entity_topic(question, query):
            pass
        elif self._try_parse_intent_temporal(question, query):
            pass
        elif self._try_parse_lookup(question, query):
            pass
        elif self._try_parse_entity_list(question, query):
            pass
        elif self._try_parse_fact_recall(question, query):
            pass
        elif self._try_parse_timeline(question, query):
            pass
        else:
            # Semantic fallback
            query.query_type = QueryType.SEMANTIC
            query.topics = self._extract_topics(question)

        # Extract entity type filter if not already set
        if query.entity_type_filter is None:
            query.entity_type_filter = self._extract_entity_type(question)

        return query

    def _extract_date_range(self, text: str) -> DateRangeFilter | None:
        """Extract date range from text using pattern matching."""
        for pattern, ref in _DATE_PATTERNS:
            if pattern.search(text):
                return DateRangeFilter.from_reference(ref, self.now)
        return None

    def _extract_intent_filter(self, text: str) -> str | None:
        """Extract intent filter from text keywords."""
        words = text.lower().split()
        for word in words:
            if word in _INTENT_KEYWORDS:
                return _INTENT_KEYWORDS[word]
        return None

    def _extract_entity_type(self, text: str) -> EntityType | None:
        """Extract entity type filter from text keywords."""
        words = text.lower().split()
        for word in words:
            if word in _ENTITY_TYPE_KEYWORDS:
                return _ENTITY_TYPE_KEYWORDS[word]
        return None

    def _extract_topics(self, text: str) -> list[str]:
        """Extract meaningful topic words from text."""
        # Remove punctuation and split
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        words = cleaned.split()
        topics = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        return topics

    def _extract_entity_names(self, text: str) -> list[str]:
        """Extract potential entity names (capitalized words/phrases).

        Finds sequences of capitalized words that likely represent
        proper nouns (entity names).
        """
        # Find capitalized word sequences (2+ chars)
        names: list[str] = []
        # Match sequences of capitalized words
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            name = match.group(1)
            # Skip common sentence starters
            if name.lower() not in _STOP_WORDS and len(name) > 1:
                names.append(name)
        return names

    def _try_parse_relationship(self, text: str, query: StructuredQuery) -> bool:
        """Try to parse as a relationship query."""
        for pattern in _RELATIONSHIP_PATTERNS:
            match = pattern.search(text)
            if match:
                query.query_type = QueryType.RELATIONSHIP
                entity_a = match.group(1).strip().rstrip("?.")
                entity_b = match.group(2).strip().rstrip("?.")
                query.entity_names = [entity_a, entity_b]
                return True
        return False

    def _try_parse_count(self, text: str, query: StructuredQuery) -> bool:
        """Try to parse as a count query."""
        for pattern in _COUNT_PATTERNS:
            if pattern.search(text):
                query.query_type = QueryType.COUNT
                query.topics = self._extract_topics(text)
                # Extract entity type for "how many people/projects/etc"
                query.entity_type_filter = self._extract_entity_type(text)
                return True
        return False

    def _try_parse_lookup(self, text: str, query: StructuredQuery) -> bool:
        """Try to parse as an entity lookup query."""
        for pattern in _LOOKUP_PATTERNS:
            match = pattern.search(text)
            if match:
                query.query_type = QueryType.ENTITY_LOOKUP
                entity_name = match.group(1).strip().rstrip("?.")
                query.entity_names = [entity_name]
                return True
        return False

    def _try_parse_timeline(self, text: str, query: StructuredQuery) -> bool:
        """Try to parse as a timeline query."""
        for pattern in _TIMELINE_PATTERNS:
            if pattern.search(text):
                query.query_type = QueryType.TIMELINE
                query.topics = self._extract_topics(text)
                # If no date range was extracted, default to this week
                if query.date_range is None:
                    query.date_range = DateRangeFilter.from_reference(
                        DateReference.THIS_WEEK, self.now
                    )
                # Extract entity names mentioned in the question
                query.entity_names = self._extract_entity_names(text)
                return True
        return False

    def _try_parse_fact_recall(self, text: str, query: StructuredQuery) -> bool:
        """Try to parse as a fact recall query."""
        for pattern in _FACT_PATTERNS:
            if pattern.search(text):
                query.query_type = QueryType.FACT_RECALL
                query.entity_names = self._extract_entity_names(text)
                query.topics = self._extract_topics(text)
                return True
        return False

    def _try_parse_entity_topic(self, text: str, query: StructuredQuery) -> bool:
        """Try to parse 'Who did I talk to about X?' style queries.

        These combine an entity type filter (person) with a topic filter,
        searching episodic memory for episodes that mention both.
        """
        for pattern in _ENTITY_TOPIC_PATTERNS:
            match = pattern.search(text)
            if match:
                topic_text = match.group(1).strip().rstrip("?.")
                query.query_type = QueryType.ENTITY_LIST
                query.entity_type_filter = EntityType.PERSON
                query.topics = self._extract_topics(topic_text) or [topic_text]
                return True
        return False

    def _try_parse_intent_temporal(self, text: str, query: StructuredQuery) -> bool:
        """Try to parse 'What meetings do I have tomorrow?' style queries.

        These combine an intent filter with a date range, querying
        episodic memory for episodes matching both.
        """
        for pattern in _INTENT_TEMPORAL_PATTERNS:
            match = pattern.search(text)
            if match:
                intent_word = match.group(1).lower().rstrip("s")
                # Map to canonical intent
                intent_map = {
                    "meeting": "event",
                    "event": "event",
                    "task": "task",
                    "reminder": "reminder",
                    "idea": "idea",
                }
                query.intent_filter = intent_map.get(intent_word, intent_word)
                query.query_type = QueryType.TIMELINE
                # Date range already extracted; default to today if none
                if query.date_range is None:
                    query.date_range = DateRangeFilter.from_reference(
                        DateReference.TODAY, self.now
                    )
                query.entity_names = self._extract_entity_names(text)
                return True
        return False

    def _try_parse_entity_list(self, text: str, query: StructuredQuery) -> bool:
        """Try to parse as an entity list query."""
        entity_type = self._extract_entity_type(text)
        if entity_type is not None:
            # Check if there's a list-like pattern
            lower = text.lower()
            list_indicators = [
                "what", "which", "list", "show", "all",
                "do i have", "do i know", "am i",
            ]
            if any(ind in lower for ind in list_indicators):
                query.query_type = QueryType.ENTITY_LIST
                query.entity_type_filter = entity_type
                return True
        return False


# ── Query Executor ────────────────────────────────────────────────────


class QueryExecutor:
    """Executes structured queries against the knowledge graph and episodic memory.

    Routes each query type to the appropriate store operations,
    aggregates results, and returns a unified QueryResult.
    """

    def __init__(
        self,
        graph_store: EntityGraphStore,
        episodic_store: EpisodicMemoryStore,
        user_id: str,
    ):
        self._graph = graph_store
        self._episodic = episodic_store
        self._user_id = user_id

    async def execute(self, query: StructuredQuery) -> QueryResult:
        """Execute a structured query and return results.

        Routes to the appropriate handler based on query type.
        """
        import time

        start = time.monotonic()

        handlers = {
            QueryType.ENTITY_LOOKUP: self._execute_entity_lookup,
            QueryType.ENTITY_LIST: self._execute_entity_list,
            QueryType.FACT_RECALL: self._execute_fact_recall,
            QueryType.TIMELINE: self._execute_timeline,
            QueryType.RELATIONSHIP: self._execute_relationship,
            QueryType.COUNT: self._execute_count,
            QueryType.SEMANTIC: self._execute_semantic,
        }

        handler = handlers.get(query.query_type, self._execute_semantic)
        result = await handler(query)

        elapsed_ms = (time.monotonic() - start) * 1000
        result.execution_time_ms = elapsed_ms
        result.total_results = (
            len(result.entities)
            + len(result.facts)
            + len(result.episodes)
            + len(result.relationships)
            + len(result.semantic_results)
        )

        logger.info(
            "Query executed: type=%s, results=%d, time=%.1fms",
            query.query_type.value,
            result.total_results,
            elapsed_ms,
        )

        return result

    async def _execute_entity_lookup(self, query: StructuredQuery) -> QueryResult:
        """Look up a specific entity by name."""
        result = QueryResult(query=query)

        for name in query.entity_names:
            entity = await self._graph.find_entity_by_name(name)
            if entity:
                result.entities.append(entity)
                # Get facts about this entity
                facts = await self._graph.get_entity_facts(entity.id)
                result.facts.extend(facts)
                # Get relationships
                rels = await self._graph.get_entity_relationships(entity.id)
                result.relationships.extend(rels)

        # Also search episodic memory for mentions
        if query.entity_names:
            episodes = await self._episodic.get_entity_timeline(
                self._user_id,
                query.entity_names[0],
                limit=query.limit,
            )
            result.episodes = episodes

        return result

    async def _execute_entity_list(self, query: StructuredQuery) -> QueryResult:
        """List all entities of a given type."""
        result = QueryResult(query=query)
        entities = await self._graph.get_all_entities(query.entity_type_filter)

        # Apply date filter if present
        if query.date_range and query.date_range.start:
            entities = [
                e for e in entities
                if e.last_seen >= query.date_range.start
            ]
        if query.date_range and query.date_range.end:
            entities = [
                e for e in entities
                if e.first_seen <= query.date_range.end
            ]

        # Sort by mention count (most mentioned first)
        entities.sort(key=lambda e: e.mention_count, reverse=True)
        result.entities = entities[:query.limit]

        return result

    async def _execute_fact_recall(self, query: StructuredQuery) -> QueryResult:
        """Recall facts about entities or topics."""
        result = QueryResult(query=query)

        # Look up entities mentioned in the query
        for name in query.entity_names:
            entity = await self._graph.find_entity_by_name(name)
            if entity:
                result.entities.append(entity)
                facts = await self._graph.get_entity_facts(entity.id)
                result.facts.extend(facts)

        # If no entity names, search all facts semantically
        if not query.entity_names and query.topics:
            search_text = " ".join(query.topics)
            semantic = await self._graph.search(
                search_text,
                top_k=query.limit,
                item_types=["fact"],
            )
            result.semantic_results = semantic

        # Also search episodic memory
        if query.entity_names:
            episodes = await self._episodic.get_entity_timeline(
                self._user_id,
                query.entity_names[0],
                limit=query.limit,
            )
            # Apply date filter
            if query.date_range:
                time_filter = TimeRangeFilter(
                    start=query.date_range.start,
                    end=query.date_range.end,
                )
                episodes = [ep for ep in episodes if time_filter.matches(ep)]
            result.episodes = episodes

        return result

    async def _execute_timeline(self, query: StructuredQuery) -> QueryResult:
        """Query episodic memory for timeline of events."""
        result = QueryResult(query=query)

        # Build episodic filters
        time_filter = None
        if query.date_range:
            time_filter = TimeRangeFilter(
                start=query.date_range.start,
                end=query.date_range.end,
            )

        entity_filter = None
        if query.entity_names:
            entity_filter = EntityFilter(entity_name=query.entity_names[0])

        intent_filter = None
        if query.intent_filter:
            intent_filter = IntentFilter(intent=query.intent_filter)

        episodes = await self._episodic.query(
            self._user_id,
            time_range=time_filter,
            entity_filter=entity_filter,
            intent_filter=intent_filter,
            limit=query.limit,
        )
        result.episodes = episodes

        return result

    async def _execute_relationship(self, query: StructuredQuery) -> QueryResult:
        """Find relationships between two entities."""
        result = QueryResult(query=query)

        if len(query.entity_names) < 2:
            return result

        entity_a = await self._graph.find_entity_by_name(query.entity_names[0])
        entity_b = await self._graph.find_entity_by_name(query.entity_names[1])

        if entity_a:
            result.entities.append(entity_a)
        if entity_b:
            result.entities.append(entity_b)

        if entity_a and entity_b:
            # Get all relationships for entity_a and filter for entity_b
            rels_a = await self._graph.get_entity_relationships(entity_a.id)
            for rel in rels_a:
                if (
                    rel.target_entity_id == entity_b.id
                    or rel.source_entity_id == entity_b.id
                ):
                    result.relationships.append(rel)

            # Also check entity_b's relationships
            if not result.relationships:
                rels_b = await self._graph.get_entity_relationships(entity_b.id)
                for rel in rels_b:
                    if (
                        rel.target_entity_id == entity_a.id
                        or rel.source_entity_id == entity_a.id
                    ):
                        result.relationships.append(rel)

        return result

    async def _execute_count(self, query: StructuredQuery) -> QueryResult:
        """Count entities or episodes matching filters."""
        result = QueryResult(query=query)

        if query.entity_type_filter:
            entities = await self._graph.get_all_entities(query.entity_type_filter)

            # Apply date filter
            if query.date_range and query.date_range.start:
                entities = [
                    e for e in entities
                    if e.last_seen >= query.date_range.start
                ]

            result.entities = entities
        elif query.intent_filter:
            # Count episodes with this intent
            time_filter = None
            if query.date_range:
                time_filter = TimeRangeFilter(
                    start=query.date_range.start,
                    end=query.date_range.end,
                )

            episodes = await self._episodic.query(
                self._user_id,
                time_range=time_filter,
                intent_filter=IntentFilter(intent=query.intent_filter),
                limit=1000,  # Get all for counting
            )
            result.episodes = episodes

        return result

    async def _execute_semantic(self, query: StructuredQuery) -> QueryResult:
        """Fallback: semantic search across the knowledge graph."""
        result = QueryResult(query=query)

        search_text = query.raw_text
        if query.topics:
            search_text = " ".join(query.topics) or query.raw_text

        # Search knowledge graph
        semantic = await self._graph.search(
            search_text,
            top_k=query.limit,
        )
        result.semantic_results = semantic

        # Also search episodic memory if date-scoped
        if query.date_range:
            time_filter = TimeRangeFilter(
                start=query.date_range.start,
                end=query.date_range.end,
            )
            episodes = await self._episodic.query(
                self._user_id,
                time_range=time_filter,
                limit=query.limit,
            )
            result.episodes = episodes

        return result


# ── Response Formatter ─────────────────────────────────────────────────


class ResponseFormatter:
    """Formats QueryResult into structured, human-readable API responses.

    Converts raw query results (entities, facts, episodes, relationships)
    into a consistent response format with:
    - A natural language answer summary
    - Structured result items with uniform keys
    - Metadata (query type, execution time, result count)

    Design principles:
    - Anti-shame: no "overdue", "missed", or guilt language
    - Conversational: summaries read like a friend answering
    - Informative: always returns something useful, never empty walls
    """

    @staticmethod
    def format(result: QueryResult) -> dict[str, Any]:
        """Format a QueryResult into a structured API response dict.

        Args:
            result: The raw query result from the executor.

        Returns:
            Dict with 'answer_summary', 'items', 'query_info', and 'metadata'.
        """
        items = ResponseFormatter._build_items(result)
        summary = ResponseFormatter._build_summary(result, items)

        return {
            "answer_summary": summary,
            "items": items,
            "query_info": {
                "query_type": result.query.query_type.value,
                "raw_text": result.query.raw_text,
                "parsed": result.query.to_dict(),
            },
            "metadata": {
                "total_results": result.total_results,
                "execution_time_ms": round(result.execution_time_ms, 1),
                "has_results": result.has_results,
            },
        }

    @staticmethod
    def _build_items(result: QueryResult) -> list[dict[str, Any]]:
        """Build a flat list of result items from heterogeneous result types."""
        items: list[dict[str, Any]] = []

        for entity in result.entities:
            items.append({
                "type": "entity",
                "name": entity.name,
                "entity_type": entity.entity_type.value,
                "mention_count": entity.mention_count,
                "last_seen": entity.last_seen.isoformat(),
                "first_seen": entity.first_seen.isoformat(),
                "attributes": entity.attributes,
                "content": f"{entity.entity_type.value}: {entity.name}",
            })

        for fact in result.facts:
            items.append({
                "type": "fact",
                "content": fact.content,
                "fact_type": fact.fact_type.value,
                "confidence": fact.confidence,
                "confirmed_count": fact.confirmation_count,
                "last_confirmed": fact.last_confirmed.isoformat(),
            })

        for episode in result.episodes:
            items.append({
                "type": "episode",
                "content": episode.raw_text,
                "intent": episode.intent,
                "timestamp": episode.timestamp.isoformat(),
                "entities": [
                    {"name": e.name, "entity_type": e.entity_type}
                    for e in episode.entities
                ],
                "emotion": {
                    "primary": episode.emotion.primary,
                    "valence": episode.emotion.valence,
                },
            })

        for rel in result.relationships:
            items.append({
                "type": "relationship",
                "relationship_type": rel.relationship_type.value,
                "source_entity_id": rel.source_entity_id,
                "target_entity_id": rel.target_entity_id,
                "strength": rel.strength,
                "content": f"{rel.relationship_type.value} (strength: {rel.strength})",
            })

        for sr in result.semantic_results:
            items.append({
                "type": "semantic_match",
                "content": sr.content,
                "item_type": sr.item_type,
                "similarity": sr.similarity_score,
                "metadata": sr.metadata,
            })

        return items

    @staticmethod
    def _build_summary(result: QueryResult, items: list[dict[str, Any]]) -> str:
        """Generate a natural language summary of the results.

        Summaries are conversational and shame-free. They describe
        what was found without judgment about what's missing.
        """
        query = result.query

        if not result.has_results:
            return ResponseFormatter._no_results_summary(query)

        query_type = query.query_type

        if query_type == QueryType.ENTITY_LOOKUP:
            return ResponseFormatter._entity_lookup_summary(result)
        elif query_type == QueryType.ENTITY_LIST:
            return ResponseFormatter._entity_list_summary(result)
        elif query_type == QueryType.FACT_RECALL:
            return ResponseFormatter._fact_recall_summary(result)
        elif query_type == QueryType.TIMELINE:
            return ResponseFormatter._timeline_summary(result)
        elif query_type == QueryType.RELATIONSHIP:
            return ResponseFormatter._relationship_summary(result)
        elif query_type == QueryType.COUNT:
            return ResponseFormatter._count_summary(result)
        elif query_type == QueryType.SEMANTIC:
            return ResponseFormatter._semantic_summary(result)

        return f"Found {result.total_results} results."

    @staticmethod
    def _no_results_summary(query: StructuredQuery) -> str:
        """Shame-free message when no results are found."""
        if query.entity_names:
            return f"I don't have anything about {query.entity_names[0]} yet."
        if query.intent_filter:
            date_desc = ""
            if query.date_range:
                date_desc = f" for {query.date_range.reference.value.replace('_', ' ')}"
            return f"No {query.intent_filter}s found{date_desc}."
        if query.date_range:
            date_desc = query.date_range.reference.value.replace("_", " ")
            return f"Nothing recorded for {date_desc}."
        return "I don't have information about that yet."

    @staticmethod
    def _entity_lookup_summary(result: QueryResult) -> str:
        if not result.entities:
            return "I don't have information about that yet."
        entity = result.entities[0]
        parts = [f"{entity.name} is a {entity.entity_type.value}"]
        if result.facts:
            fact_texts = [f.content for f in result.facts[:3]]
            parts.append(". ".join(fact_texts))
        if result.relationships:
            parts.append(
                f"{len(result.relationships)} connection"
                f"{'s' if len(result.relationships) != 1 else ''} found"
            )
        return ". ".join(parts) + "."

    @staticmethod
    def _entity_list_summary(result: QueryResult) -> str:
        if not result.entities:
            return "No matching entities found."
        names = [e.name for e in result.entities[:5]]
        entity_type = result.entities[0].entity_type.value
        summary = f"Found {len(result.entities)} {entity_type}"
        if len(result.entities) != 1:
            summary += "s"
        summary += f": {', '.join(names)}"
        if len(result.entities) > 5:
            summary += f" and {len(result.entities) - 5} more"
        return summary + "."

    @staticmethod
    def _fact_recall_summary(result: QueryResult) -> str:
        if result.facts:
            fact_texts = [f.content for f in result.facts[:3]]
            return ". ".join(fact_texts) + "."
        if result.semantic_results:
            return f"Found {len(result.semantic_results)} related memories."
        return "I don't have specific facts about that yet."

    @staticmethod
    def _timeline_summary(result: QueryResult) -> str:
        if not result.episodes:
            date_desc = ""
            if result.query.date_range:
                date_desc = f" for {result.query.date_range.reference.value.replace('_', ' ')}"
            return f"Nothing recorded{date_desc}."

        count = len(result.episodes)
        intents: dict[str, int] = {}
        for ep in result.episodes:
            intents[ep.intent] = intents.get(ep.intent, 0) + 1

        parts = [f"Found {count} item{'s' if count != 1 else ''}"]

        if result.query.date_range:
            parts[0] += f" from {result.query.date_range.reference.value.replace('_', ' ')}"

        if intents:
            intent_parts = []
            for intent, cnt in sorted(intents.items(), key=lambda x: -x[1]):
                intent_parts.append(f"{cnt} {intent}{'s' if cnt != 1 else ''}")
            parts.append(", ".join(intent_parts))

        return ": ".join(parts) + "."

    @staticmethod
    def _relationship_summary(result: QueryResult) -> str:
        if not result.relationships:
            names = result.query.entity_names
            if len(names) >= 2:
                return f"No direct connection found between {names[0]} and {names[1]}."
            return "No relationships found."
        names = result.query.entity_names
        rel_types = [r.relationship_type.value for r in result.relationships]
        if len(names) >= 2:
            return (
                f"{names[0]} and {names[1]} are connected: "
                f"{', '.join(rel_types)}."
            )
        return f"Found {len(result.relationships)} relationship{'s' if len(result.relationships) != 1 else ''}."

    @staticmethod
    def _count_summary(result: QueryResult) -> str:
        entity_count = len(result.entities)
        episode_count = len(result.episodes)

        if result.query.entity_type_filter:
            type_name = result.query.entity_type_filter.value
            return f"You have {entity_count} {type_name}{'s' if entity_count != 1 else ''}."
        if result.query.intent_filter:
            intent = result.query.intent_filter
            return f"You have {episode_count} {intent}{'s' if episode_count != 1 else ''}."
        return f"Found {entity_count + episode_count} items."

    @staticmethod
    def _semantic_summary(result: QueryResult) -> str:
        total = len(result.semantic_results) + len(result.episodes)
        if total == 0:
            return "I don't have information about that yet."
        return f"Found {total} related item{'s' if total != 1 else ''}."
