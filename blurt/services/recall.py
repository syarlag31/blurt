"""Personal history recall engine — full semantic search for QUESTION intent.

Premium-tier recall engine that searches across ALL captured conversations,
entities, and knowledge graph nodes to answer questions about the user's
personal history.

When a blurt is classified as QUESTION intent, the recall engine:
1. Parses the natural language query to extract temporal hints and entity refs
2. Embeds the query using Gemini 2 (RETRIEVAL_QUERY task type)
3. Searches episodic memory (conversations) for semantic matches
4. Searches knowledge graph entities for relevant nodes
5. Searches facts/patterns for supporting knowledge
6. Searches entity relationships for connection context
7. Merges and ranks results by relevance across all sources
8. Enriches top results with surrounding source context
9. Returns a unified, ranked set of recall results with source context

The engine supports:
- Cross-source recall: episodes, entities, facts, patterns, relationships
- Natural language query understanding: temporal refs, entity extraction
- Temporal context: recent items get a recency boost
- Entity-aware ranking: results mentioning query-relevant entities rank higher
- Source context enrichment: surrounding episodes for richer answers
- Configurable thresholds and limits per source
- Works in both cloud and local-only modes (feature parity)

Design principles:
- Zero friction: users ask natural questions, recall handles the rest
- Full history: nothing is excluded from search
- Privacy-first: all search happens within the user's encrypted context
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from blurt.clients.embeddings import EmbeddingProvider, cosine_similarity
from blurt.memory.episodic import (
    EpisodicMemoryStore,
    InMemoryEpisodicStore,
)
from blurt.memory.semantic import SemanticMemoryStore
from blurt.models.entities import (
    Fact,
    LearnedPattern,
)

logger = logging.getLogger(__name__)


# ── Natural Language Query Understanding ──────────────────────────────


@dataclass
class QueryUnderstanding:
    """Parsed understanding of a natural language question.

    Extracts temporal hints, entity references, and query intent
    from open-ended natural language questions to improve search precision.
    """

    original_query: str
    # Temporal hints extracted from the query
    temporal_start: datetime | None = None
    temporal_end: datetime | None = None
    temporal_hint: str | None = None  # e.g. "last week", "yesterday"
    # Entity references detected in the query
    entity_references: list[str] = field(default_factory=list)
    # Whether the query is asking about a specific entity relationship
    is_relationship_query: bool = False
    # Whether the query is a count/aggregation question
    is_count_query: bool = False
    # Whether the query asks about emotions or feelings
    is_emotion_query: bool = False
    # Core search text after removing temporal/structural words
    search_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "temporal_hint": self.temporal_hint,
            "temporal_start": self.temporal_start.isoformat() if self.temporal_start else None,
            "temporal_end": self.temporal_end.isoformat() if self.temporal_end else None,
            "entity_references": self.entity_references,
            "is_relationship_query": self.is_relationship_query,
            "is_count_query": self.is_count_query,
            "is_emotion_query": self.is_emotion_query,
            "search_text": self.search_text,
        }


# Temporal patterns for NL parsing
_TEMPORAL_PATTERNS: list[tuple[str, int]] = [
    # (regex pattern, days_back)
    (r"\byesterday\b", 1),
    (r"\blast\s+week\b", 7),
    (r"\bthis\s+week\b", 7),
    (r"\blast\s+month\b", 30),
    (r"\bthis\s+month\b", 30),
    (r"\blast\s+year\b", 365),
    (r"\bthis\s+year\b", 365),
    (r"\b(\d+)\s+days?\s+ago\b", -1),  # Special: uses captured group
    (r"\b(\d+)\s+weeks?\s+ago\b", -7),  # Special: uses captured group * 7
    (r"\b(\d+)\s+months?\s+ago\b", -30),  # Special: uses captured group * 30
    (r"\btoday\b", 0),
    (r"\brecently\b", 14),
    (r"\ba\s+while\s+ago\b", 90),
    (r"\blast\s+few\s+days\b", 5),
]

# Relationship indicators
_RELATIONSHIP_PATTERNS = [
    r"\bconnect(?:ed|ion)?\s+(?:to|with|between)\b",
    r"\brelat(?:ed|ionship)\s+(?:to|with|between)\b",
    r"\bknow(?:s)?\s+(?:about|of)\b",
    r"\bwork(?:s|ing)?\s+with\b",
    r"\btogether\s+with\b",
    r"\bbetween\s+\w+\s+and\s+\w+\b",
]

# Count/aggregation indicators
_COUNT_PATTERNS = [
    r"\bhow\s+many\b",
    r"\bhow\s+often\b",
    r"\bhow\s+much\b",
    r"\bcount\b",
    r"\bnumber\s+of\b",
    r"\btotal\b",
]

# Emotion query indicators
_EMOTION_PATTERNS = [
    r"\bhow\s+(?:did|do|was|were)\s+I\s+feel\b",
    r"\bfeel(?:ing|s)?\s+(?:about|when)\b",
    r"\bmood\b",
    r"\bhappy|sad|angry|anxious|stressed|excited|frustrated\b",
    r"\bemotion(?:al|s)?\b",
]

# Structural words to strip from search text
_STRUCTURAL_WORDS = re.compile(
    r"\b(what|when|where|who|why|how|did|do|does|was|were|is|are|am|"
    r"the|a|an|i|my|me|about|that|this|of|in|on|at|to|for|with|"
    r"tell|remind|show|find|search|recall|remember|said|say|mention|"
    r"talked|talk|spoke|speak)\b",
    re.IGNORECASE,
)


def parse_query(query: str, now: datetime | None = None) -> QueryUnderstanding:
    """Parse a natural language question into structured understanding.

    Extracts temporal hints, entity references, and query characteristics
    without requiring an LLM call — pure pattern matching for speed.

    Args:
        query: The natural language question.
        now: Current time (defaults to UTC now).

    Returns:
        QueryUnderstanding with extracted signals.
    """
    now = now or datetime.now(timezone.utc)
    understanding = QueryUnderstanding(original_query=query)
    q_lower = query.lower().strip()

    # Extract temporal hints
    for pattern, days_back in _TEMPORAL_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            if days_back == -1:
                # Dynamic: N days ago
                n = int(match.group(1))
                understanding.temporal_start = now - timedelta(days=n + 1)
                understanding.temporal_end = now - timedelta(days=max(0, n - 1))
                understanding.temporal_hint = match.group(0)
            elif days_back == -7:
                # Dynamic: N weeks ago
                n = int(match.group(1))
                days = n * 7
                understanding.temporal_start = now - timedelta(days=days + 7)
                understanding.temporal_end = now - timedelta(days=max(0, days - 7))
                understanding.temporal_hint = match.group(0)
            elif days_back == -30:
                # Dynamic: N months ago
                n = int(match.group(1))
                days = n * 30
                understanding.temporal_start = now - timedelta(days=days + 30)
                understanding.temporal_end = now - timedelta(days=max(0, days - 30))
                understanding.temporal_hint = match.group(0)
            elif days_back == 0:
                # Today
                understanding.temporal_start = now.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                understanding.temporal_end = now
                understanding.temporal_hint = "today"
            else:
                understanding.temporal_start = now - timedelta(days=days_back)
                understanding.temporal_end = now
                understanding.temporal_hint = match.group(0)
            break  # Use first match only

    # Detect relationship queries
    for pattern in _RELATIONSHIP_PATTERNS:
        if re.search(pattern, q_lower):
            understanding.is_relationship_query = True
            break

    # Detect count queries
    for pattern in _COUNT_PATTERNS:
        if re.search(pattern, q_lower):
            understanding.is_count_query = True
            break

    # Detect emotion queries
    for pattern in _EMOTION_PATTERNS:
        if re.search(pattern, q_lower):
            understanding.is_emotion_query = True
            break

    # Extract potential entity references (capitalized words/phrases)
    # Heuristic: words starting with uppercase that aren't sentence starters
    words = query.split()
    for i, word in enumerate(words):
        cleaned = re.sub(r"[^\w]", "", word)
        if not cleaned:
            continue
        if cleaned[0].isupper() and i > 0 and len(cleaned) > 1:
            # Check if it's a multi-word entity (consecutive capitalized)
            entity_parts = [cleaned]
            for j in range(i + 1, min(i + 4, len(words))):
                next_word = re.sub(r"[^\w]", "", words[j])
                if next_word and next_word[0].isupper() and len(next_word) > 1:
                    entity_parts.append(next_word)
                else:
                    break
            entity = " ".join(entity_parts)
            if entity not in understanding.entity_references:
                understanding.entity_references.append(entity)

    # Build search text by stripping structural words
    search_text = _STRUCTURAL_WORDS.sub("", q_lower)
    search_text = re.sub(r"\s+", " ", search_text).strip()
    understanding.search_text = search_text or q_lower

    return understanding


# ── Result Types ─────────────────────────────────────────────────────


class RecallSourceType(str, Enum):
    """Source of a recall result."""

    EPISODE = "episode"         # From episodic memory (conversations)
    ENTITY = "entity"           # From knowledge graph entities
    FACT = "fact"               # From learned facts/preferences
    PATTERN = "pattern"         # From behavioral patterns
    RELATIONSHIP = "relationship"  # From entity relationships
    SUMMARY = "summary"         # From episode summaries


@dataclass
class SourceContext:
    """Surrounding context from the source of a recall result.

    Provides temporal neighbors and session context so the user
    can understand what was happening around the recalled item.
    """

    preceding_text: str | None = None
    following_text: str | None = None
    session_id: str | None = None
    session_episode_count: int = 0
    surrounding_entities: list[str] = field(default_factory=list)
    surrounding_intents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preceding_text": self.preceding_text,
            "following_text": self.following_text,
            "session_id": self.session_id,
            "session_episode_count": self.session_episode_count,
            "surrounding_entities": self.surrounding_entities,
            "surrounding_intents": self.surrounding_intents,
        }


@dataclass
class RecallResult:
    """A single result from the recall engine.

    Each result carries its source type, relevance score, and the
    content needed to construct an answer. Results from different
    sources are comparable via their relevance_score.
    """

    source_type: RecallSourceType
    source_id: str
    content: str
    relevance_score: float  # 0.0 to 1.0, combines similarity + boosts
    timestamp: datetime | None = None  # When the source was created/observed
    metadata: dict[str, Any] = field(default_factory=dict)

    # Original raw similarity before boosts
    raw_similarity: float = 0.0

    # Source context — populated for top results during enrichment
    source_context: SourceContext | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "content": self.content,
            "relevance_score": round(self.relevance_score, 4),
            "raw_similarity": round(self.raw_similarity, 4),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }
        if self.source_context is not None:
            d["source_context"] = self.source_context.to_dict()
        return d


@dataclass
class RecallResponse:
    """Full response from a recall query.

    Contains merged, ranked results from all sources along with
    diagnostics about the search and the query understanding.
    """

    query: str
    results: list[RecallResult] = field(default_factory=list)
    total_results: int = 0
    sources_searched: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    query_embedding_generated: bool = False
    entity_context_used: list[str] = field(default_factory=list)
    query_understanding: QueryUnderstanding | None = None

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    @property
    def top_result(self) -> RecallResult | None:
        return self.results[0] if self.results else None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "sources_searched": self.sources_searched,
            "latency_ms": round(self.latency_ms, 2),
            "query_embedding_generated": self.query_embedding_generated,
            "entity_context_used": self.entity_context_used,
            "has_results": self.has_results,
        }
        if self.query_understanding is not None:
            d["query_understanding"] = self.query_understanding.to_dict()
        return d


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class RecallConfig:
    """Configuration for the recall engine.

    All thresholds and limits are tunable per deployment.
    """

    # Per-source result limits
    max_episode_results: int = 20
    max_entity_results: int = 10
    max_fact_results: int = 10
    max_pattern_results: int = 5
    max_relationship_results: int = 5
    max_summary_results: int = 5

    # Final merged result limit
    max_total_results: int = 25

    # Minimum similarity thresholds per source
    min_episode_similarity: float = 0.3
    min_entity_similarity: float = 0.3
    min_fact_similarity: float = 0.3
    min_pattern_similarity: float = 0.3
    min_relationship_similarity: float = 0.3

    # Recency boost: how much to boost recent results (decays with age)
    recency_boost_weight: float = 0.15
    recency_half_life_days: float = 14.0

    # Entity mention boost: boost results that mention query-relevant entities
    entity_mention_boost: float = 0.1

    # Source-type weights (how important each source is in final ranking)
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "episode": 1.0,
        "entity": 0.9,
        "fact": 0.95,
        "pattern": 0.7,
        "relationship": 0.6,
        "summary": 0.8,
    })


# ── Recall Engine ────────────────────────────────────────────────────


class PersonalHistoryRecallEngine:
    """Full personal history recall engine for QUESTION intent.

    Searches across all captured conversations, entities, and knowledge
    graph nodes with relevance ranking. This is the premium-tier recall
    that enables "what did I say about X?" and "when was Y?" queries.

    Usage::

        engine = PersonalHistoryRecallEngine(
            episodic_store=episodic_store,
            semantic_store=semantic_store,
            embedding_provider=embedding_provider,
        )
        response = await engine.recall("user-1", "what did I say about the project?")
        for result in response.results:
            print(f"[{result.source_type}] {result.content} ({result.relevance_score})")
    """

    def __init__(
        self,
        episodic_store: EpisodicMemoryStore | None = None,
        semantic_store: SemanticMemoryStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: RecallConfig | None = None,
    ) -> None:
        self._episodic = episodic_store or InMemoryEpisodicStore()
        self._semantic = semantic_store
        self._embeddings = embedding_provider
        self._config = config or RecallConfig()

        # Stats
        self._total_queries = 0
        self._total_results_returned = 0
        self._total_latency_ms = 0.0

    @property
    def config(self) -> RecallConfig:
        return self._config

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_queries": self._total_queries,
            "total_results_returned": self._total_results_returned,
            "avg_latency_ms": (
                round(self._total_latency_ms / self._total_queries, 2)
                if self._total_queries > 0
                else 0.0
            ),
        }

    async def recall(
        self,
        user_id: str,
        query: str,
        *,
        max_results: int | None = None,
        source_filter: list[RecallSourceType] | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        enrich_context: bool = True,
        context_window: int = 3,
    ) -> RecallResponse:
        """Execute a full personal history recall query.

        Searches across all memory sources, merges results, applies
        relevance ranking with recency and entity boosts. Supports
        open-ended natural language questions by parsing temporal hints
        and entity references from the query text.

        Args:
            user_id: The user whose history to search.
            query: Natural language query (e.g. "what did I say about X?").
            max_results: Override max total results.
            source_filter: Only search specific source types.
            time_start: Only consider results after this time.
            time_end: Only consider results before this time.
            enrich_context: Whether to enrich top results with source context.
            context_window: Number of top results to enrich with context.

        Returns:
            RecallResponse with ranked results from all sources.
        """
        start_time = time.monotonic()
        effective_max = max_results or self._config.max_total_results
        sources_searched: list[str] = []
        entity_context: list[str] = []
        all_results: list[RecallResult] = []

        # Step 0: Parse the natural language query
        understanding = parse_query(query)

        # Use parsed temporal hints if caller didn't specify explicit times
        effective_time_start = time_start or understanding.temporal_start
        effective_time_end = time_end or understanding.temporal_end

        # Add parsed entity references to context
        for entity_ref in understanding.entity_references:
            if entity_ref not in entity_context:
                entity_context.append(entity_ref)

        # Step 1: Generate query embedding
        query_embedding: list[float] | None = None
        embedding_generated = False
        if self._embeddings is not None:
            try:
                query_embedding = await self._embeddings.embed(query)
                embedding_generated = True
            except Exception as e:
                logger.warning("Failed to generate query embedding: %s", e)

        # Step 2: Find query-relevant entities for context-aware boosting
        relevant_entity_names: set[str] = set()
        # Include entities from NL parsing
        for er in understanding.entity_references:
            relevant_entity_names.add(er.lower())

        if self._semantic is not None and query_embedding is not None:
            try:
                entity_results = await self._search_entities(
                    query, query_embedding, source_filter
                )
                for r in entity_results:
                    entity_name = r.metadata.get("entity_name", "")
                    if entity_name:
                        relevant_entity_names.add(entity_name.lower())
                        if entity_name not in entity_context:
                            entity_context.append(entity_name)
            except Exception as e:
                logger.warning("Entity search failed: %s", e)

        # Step 3: Search episodic memory (conversations)
        if self._should_search(RecallSourceType.EPISODE, source_filter):
            try:
                episode_results = await self._search_episodes(
                    user_id, query, query_embedding,
                    time_start=effective_time_start,
                    time_end=effective_time_end,
                )
                all_results.extend(episode_results)
                sources_searched.append("episode")
            except Exception as e:
                logger.warning("Episode search failed: %s", e)

        # Step 4: Search knowledge graph entities
        if self._should_search(RecallSourceType.ENTITY, source_filter):
            if self._semantic is not None and query_embedding is not None:
                try:
                    entity_results = await self._search_entities(
                        query, query_embedding, source_filter
                    )
                    all_results.extend(entity_results)
                    sources_searched.append("entity")
                except Exception as e:
                    logger.warning("Entity search failed: %s", e)

        # Step 5: Search facts
        if self._should_search(RecallSourceType.FACT, source_filter):
            if self._semantic is not None and query_embedding is not None:
                try:
                    fact_results = await self._search_facts(
                        query_embedding
                    )
                    all_results.extend(fact_results)
                    sources_searched.append("fact")
                except Exception as e:
                    logger.warning("Fact search failed: %s", e)

        # Step 6: Search patterns
        if self._should_search(RecallSourceType.PATTERN, source_filter):
            if self._semantic is not None and query_embedding is not None:
                try:
                    pattern_results = await self._search_patterns(
                        query_embedding
                    )
                    all_results.extend(pattern_results)
                    sources_searched.append("pattern")
                except Exception as e:
                    logger.warning("Pattern search failed: %s", e)

        # Step 6.5: Search relationships (new)
        if self._should_search(RecallSourceType.RELATIONSHIP, source_filter):
            if self._semantic is not None and query_embedding is not None:
                try:
                    relationship_results = await self._search_relationships(
                        query, query_embedding, relevant_entity_names,
                    )
                    all_results.extend(relationship_results)
                    sources_searched.append("relationship")
                except Exception as e:
                    logger.warning("Relationship search failed: %s", e)

        # Step 7: Search episode summaries
        if self._should_search(RecallSourceType.SUMMARY, source_filter):
            try:
                summary_results = await self._search_summaries(
                    user_id, query_embedding,
                    time_start=effective_time_start,
                    time_end=effective_time_end,
                )
                all_results.extend(summary_results)
                sources_searched.append("summary")
            except Exception as e:
                logger.warning("Summary search failed: %s", e)

        # Step 8: Apply boosts and re-rank
        now = datetime.now(timezone.utc)
        for result in all_results:
            result.relevance_score = self._compute_final_score(
                result, now, relevant_entity_names
            )

        # Step 9: Deduplicate (same content from different paths)
        all_results = self._deduplicate(all_results)

        # Step 10: Sort by final relevance score
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Step 11: Trim to max results
        final_results = all_results[:effective_max]

        # Step 12: Enrich top results with source context
        if enrich_context and final_results:
            await self._enrich_source_context(
                user_id, final_results[:context_window]
            )

        latency = (time.monotonic() - start_time) * 1000

        # Update stats
        self._total_queries += 1
        self._total_results_returned += len(final_results)
        self._total_latency_ms += latency

        response = RecallResponse(
            query=query,
            results=final_results,
            total_results=len(final_results),
            sources_searched=sources_searched,
            latency_ms=latency,
            query_embedding_generated=embedding_generated,
            entity_context_used=entity_context,
            query_understanding=understanding,
        )

        logger.info(
            "Recall query '%s': %d results from %d sources in %.0fms "
            "(temporal_hint=%s, entities=%s)",
            query[:50],
            len(final_results),
            len(sources_searched),
            latency,
            understanding.temporal_hint,
            understanding.entity_references[:3],
        )

        return response

    # ── Source-Specific Search Methods ────────────────────────────

    async def _search_episodes(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float] | None,
        *,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[RecallResult]:
        """Search episodic memory for conversations matching the query."""
        results: list[RecallResult] = []

        if query_embedding is not None:
            # Semantic search using embeddings
            episode_matches = await self._episodic.semantic_search(
                user_id=user_id,
                query_embedding=query_embedding,
                limit=self._config.max_episode_results,
                min_similarity=self._config.min_episode_similarity,
            )

            for episode, similarity in episode_matches:
                # Apply time filter if specified
                if time_start and episode.timestamp < time_start:
                    continue
                if time_end and episode.timestamp > time_end:
                    continue

                entity_names = [e.name for e in episode.entities]
                results.append(RecallResult(
                    source_type=RecallSourceType.EPISODE,
                    source_id=episode.id,
                    content=episode.raw_text,
                    relevance_score=similarity,  # Will be adjusted in boost step
                    raw_similarity=similarity,
                    timestamp=episode.timestamp,
                    metadata={
                        "intent": episode.intent,
                        "emotion": episode.emotion.primary,
                        "emotion_valence": episode.emotion.valence,
                        "entities": entity_names,
                        "session_id": episode.context.session_id,
                        "modality": episode.modality.value,
                    },
                ))

        return results

    async def _search_entities(
        self,
        query: str,
        query_embedding: list[float],
        source_filter: list[RecallSourceType] | None = None,
    ) -> list[RecallResult]:
        """Search knowledge graph entities matching the query."""
        if self._semantic is None:
            return []

        results: list[RecallResult] = []
        search_results = await self._semantic.search(
            query=query,
            top_k=self._config.max_entity_results,
            item_types=["entity"],
            min_similarity=self._config.min_entity_similarity,
        )

        for sr in search_results:
            results.append(RecallResult(
                source_type=RecallSourceType.ENTITY,
                source_id=sr.item_id,
                content=sr.content,
                relevance_score=sr.similarity_score,
                raw_similarity=sr.similarity_score,
                metadata=sr.metadata,
            ))

        return results

    async def _search_facts(
        self,
        query_embedding: list[float],
    ) -> list[RecallResult]:
        """Search learned facts matching the query."""
        if self._semantic is None:
            return []

        results: list[RecallResult] = []

        # Get all active facts and compare embeddings
        all_facts = await self._semantic.get_all_facts(active_only=True)
        scored: list[tuple[Fact, float]] = []

        for fact in all_facts:
            if fact.embedding is None:
                continue
            try:
                sim = cosine_similarity(query_embedding, fact.embedding)
            except ValueError:
                continue
            if sim >= self._config.min_fact_similarity:
                scored.append((fact, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        for fact, similarity in scored[: self._config.max_fact_results]:
            results.append(RecallResult(
                source_type=RecallSourceType.FACT,
                source_id=fact.id,
                content=fact.content,
                relevance_score=similarity,
                raw_similarity=similarity,
                timestamp=fact.first_learned,
                metadata={
                    "fact_type": fact.fact_type.value,
                    "confidence": fact.confidence,
                    "confirmation_count": fact.confirmation_count,
                    "subject_entity_id": fact.subject_entity_id,
                },
            ))

        return results

    async def _search_patterns(
        self,
        query_embedding: list[float],
    ) -> list[RecallResult]:
        """Search behavioral patterns matching the query."""
        if self._semantic is None:
            return []

        results: list[RecallResult] = []
        active_patterns = await self._semantic.get_active_patterns()
        scored: list[tuple[LearnedPattern, float]] = []

        for pattern in active_patterns:
            if pattern.embedding is None:
                continue
            try:
                sim = cosine_similarity(query_embedding, pattern.embedding)
            except ValueError:
                continue
            if sim >= self._config.min_pattern_similarity:
                scored.append((pattern, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        for pattern, similarity in scored[: self._config.max_pattern_results]:
            results.append(RecallResult(
                source_type=RecallSourceType.PATTERN,
                source_id=pattern.id,
                content=pattern.description,
                relevance_score=similarity,
                raw_similarity=similarity,
                timestamp=pattern.first_detected,
                metadata={
                    "pattern_type": pattern.pattern_type.value,
                    "confidence": pattern.confidence,
                    "observation_count": pattern.observation_count,
                    "parameters": pattern.parameters,
                },
            ))

        return results

    async def _search_summaries(
        self,
        user_id: str,
        query_embedding: list[float] | None,
        *,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[RecallResult]:
        """Search episode summaries matching the query."""
        if query_embedding is None:
            return []

        results: list[RecallResult] = []
        summaries = await self._episodic.get_summaries(
            user_id=user_id,
            start=time_start,
            end=time_end,
        )

        scored: list[tuple[Any, float]] = []
        for summary in summaries:
            if summary.embedding is None:
                continue
            try:
                sim = cosine_similarity(query_embedding, summary.embedding)
            except ValueError:
                continue
            if sim >= self._config.min_episode_similarity:
                scored.append((summary, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        for summary, similarity in scored[: self._config.max_summary_results]:
            results.append(RecallResult(
                source_type=RecallSourceType.SUMMARY,
                source_id=summary.id,
                content=summary.summary_text,
                relevance_score=similarity,
                raw_similarity=similarity,
                timestamp=summary.period_end,
                metadata={
                    "period_start": summary.period_start.isoformat(),
                    "period_end": summary.period_end.isoformat(),
                    "episode_count": summary.episode_count,
                    "entity_mentions": summary.entity_mentions,
                    "intent_distribution": summary.intent_distribution,
                },
            ))

        return results

    async def _search_relationships(
        self,
        query: str,
        query_embedding: list[float],
        relevant_entity_names: set[str],
    ) -> list[RecallResult]:
        """Search entity relationships for connection context.

        Finds relationships between entities that are relevant to the query,
        particularly useful for "how are X and Y connected?" type questions.
        """
        if self._semantic is None:
            return []

        results: list[RecallResult] = []

        # Strategy 1: Find relationships involving query-relevant entities
        for entity_name in relevant_entity_names:
            entity = await self._semantic.find_entity_by_name(entity_name)
            if entity is None:
                continue

            connected = await self._semantic.get_connected_entities(entity.id)
            for other_entity, rel in connected:
                if rel.strength < self._config.min_relationship_similarity:
                    continue

                content = (
                    f"{entity.name} → {rel.relationship_type.value} → "
                    f"{other_entity.name} (strength: {rel.strength:.2f})"
                )
                context_snippets = rel.context_snippets[:3] if rel.context_snippets else []

                results.append(RecallResult(
                    source_type=RecallSourceType.RELATIONSHIP,
                    source_id=rel.id if hasattr(rel, "id") else f"{entity.id}-{other_entity.id}",
                    content=content,
                    relevance_score=rel.strength,
                    raw_similarity=rel.strength,
                    timestamp=rel.last_seen if hasattr(rel, "last_seen") else None,
                    metadata={
                        "source_entity": entity.name,
                        "target_entity": other_entity.name,
                        "relationship_type": rel.relationship_type.value,
                        "strength": rel.strength,
                        "co_mention_count": rel.co_mention_count,
                        "context_snippets": context_snippets,
                        "entities": [entity.name, other_entity.name],
                    },
                ))

        # Strategy 2: If no entity-based results, use semantic search on entity names
        if not results:
            entity_search = await self._semantic.search(
                query=query,
                top_k=self._config.max_relationship_results,
                item_types=["entity"],
                min_similarity=self._config.min_relationship_similarity,
            )
            for sr in entity_search:
                entity_id = sr.item_id
                connected = await self._semantic.get_connected_entities(entity_id)
                for other_entity, rel in connected[:2]:  # Top 2 per entity
                    content = (
                        f"{sr.content} → {rel.relationship_type.value} → "
                        f"{other_entity.name}"
                    )
                    results.append(RecallResult(
                        source_type=RecallSourceType.RELATIONSHIP,
                        source_id=f"{entity_id}-{other_entity.id}",
                        content=content,
                        relevance_score=sr.similarity_score * rel.strength,
                        raw_similarity=sr.similarity_score * rel.strength,
                        metadata={
                            "source_entity": sr.content,
                            "target_entity": other_entity.name,
                            "relationship_type": rel.relationship_type.value,
                            "strength": rel.strength,
                            "entities": [sr.content, other_entity.name],
                        },
                    ))

        # Limit and sort
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:self._config.max_relationship_results]

    async def _enrich_source_context(
        self,
        user_id: str,
        results: list[RecallResult],
    ) -> None:
        """Enrich top results with surrounding source context.

        For episode results, fetches neighboring episodes from the same
        session to provide conversational context around the match.
        """
        for result in results:
            if result.source_type != RecallSourceType.EPISODE:
                continue

            session_id = result.metadata.get("session_id")
            if not session_id:
                continue

            try:
                session_episodes = await self._episodic.get_session_episodes(
                    session_id
                )
                if not session_episodes:
                    continue

                # Find the index of this episode in the session
                idx = None
                for i, ep in enumerate(session_episodes):
                    if ep.id == result.source_id:
                        idx = i
                        break

                if idx is None:
                    continue

                # Get preceding and following episode text
                preceding = session_episodes[idx - 1] if idx > 0 else None
                following = (
                    session_episodes[idx + 1]
                    if idx < len(session_episodes) - 1
                    else None
                )

                # Collect surrounding entities and intents
                surrounding_entities: list[str] = []
                surrounding_intents: list[str] = []
                for ep in session_episodes:
                    for ent in ep.entities:
                        if ent.name not in surrounding_entities:
                            surrounding_entities.append(ent.name)
                    if ep.intent not in surrounding_intents:
                        surrounding_intents.append(ep.intent)

                result.source_context = SourceContext(
                    preceding_text=preceding.raw_text if preceding else None,
                    following_text=following.raw_text if following else None,
                    session_id=session_id,
                    session_episode_count=len(session_episodes),
                    surrounding_entities=surrounding_entities[:10],
                    surrounding_intents=surrounding_intents,
                )
            except Exception as e:
                logger.debug("Failed to enrich context for %s: %s", result.source_id, e)

    # ── Scoring & Ranking ────────────────────────────────────────

    def _compute_final_score(
        self,
        result: RecallResult,
        now: datetime,
        relevant_entity_names: set[str],
    ) -> float:
        """Compute final relevance score with boosts.

        Final score = raw_similarity * source_weight
                    + recency_boost
                    + entity_mention_boost

        All components are bounded to keep the final score in [0, 1].
        """
        cfg = self._config
        base_score = result.raw_similarity

        # Apply source weight
        source_weight = cfg.source_weights.get(
            result.source_type.value, 1.0
        )
        weighted_score = base_score * source_weight

        # Recency boost
        recency_boost = 0.0
        if result.timestamp is not None and cfg.recency_boost_weight > 0:
            age_days = max(0, (now - result.timestamp).total_seconds() / 86400)
            decay = math.exp(
                -0.693 * age_days / cfg.recency_half_life_days
            )
            recency_boost = cfg.recency_boost_weight * decay

        # Entity mention boost
        entity_boost = 0.0
        if relevant_entity_names and cfg.entity_mention_boost > 0:
            result_entities = result.metadata.get("entities", [])
            if isinstance(result_entities, list):
                matching = sum(
                    1 for e in result_entities
                    if isinstance(e, str) and e.lower() in relevant_entity_names
                )
                if matching > 0:
                    entity_boost = min(
                        cfg.entity_mention_boost * matching,
                        cfg.entity_mention_boost * 3,  # Cap at 3x
                    )

        final = min(1.0, weighted_score + recency_boost + entity_boost)
        return max(0.0, final)

    def _deduplicate(self, results: list[RecallResult]) -> list[RecallResult]:
        """Remove duplicate results (same source_id from different search paths)."""
        seen: set[tuple[str, str]] = set()
        deduplicated: list[RecallResult] = []

        for result in results:
            key = (result.source_type.value, result.source_id)
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)
            else:
                # Keep the higher-scored version
                for i, existing in enumerate(deduplicated):
                    existing_key = (existing.source_type.value, existing.source_id)
                    if existing_key == key and result.relevance_score > existing.relevance_score:
                        deduplicated[i] = result
                        break

        return deduplicated

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _should_search(
        source_type: RecallSourceType,
        source_filter: list[RecallSourceType] | None,
    ) -> bool:
        """Check if a source type should be searched."""
        if source_filter is None:
            return True
        return source_type in source_filter
