"""Tier-based access control for Blurt services.

Implements a freemium tier model that gates advanced features like full
semantic recall behind premium tiers, while keeping structured queries
available for free-tier users.

Tier structure:
- FREE: Structured queries (entity lookup, recent facts, simple filters).
  Limited to 10 results per query, no graph traversal, no semantic search.
- PREMIUM: Full recall — semantic search, graph traversal, neighborhood
  search, unlimited results, pattern access, full episodic history.
- TEAM: Everything in Premium + shared knowledge graphs, team patterns.

Design principles:
- Anti-shame: tier limitations are never communicated as punishment.
  Free tier is described as "focused" not "limited".
- No dark patterns: upgrades are offered contextually, never forced.
- Graceful degradation: free-tier always returns SOMETHING useful,
  never an empty "upgrade to see results" response.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserTier(str, Enum):
    """Subscription tiers for Blurt users."""

    FREE = "free"
    PREMIUM = "premium"
    TEAM = "team"


class TierCapabilities(BaseModel):
    """What each tier can do for QUESTION intent processing."""

    # Query types
    semantic_search: bool = False
    graph_traversal: bool = False
    neighborhood_search: bool = False
    episodic_recall: bool = False
    pattern_access: bool = False

    # Limits
    max_results: int = 10
    max_graph_hops: int = 0
    max_history_days: int = 30

    # Response formatting
    include_source_episodes: bool = False
    include_confidence_scores: bool = False
    include_relationship_context: bool = False


# Capability definitions per tier
TIER_CAPABILITIES: dict[UserTier, TierCapabilities] = {
    UserTier.FREE: TierCapabilities(
        semantic_search=False,
        graph_traversal=False,
        neighborhood_search=False,
        episodic_recall=False,
        pattern_access=False,
        max_results=10,
        max_graph_hops=0,
        max_history_days=30,
        include_source_episodes=False,
        include_confidence_scores=False,
        include_relationship_context=False,
    ),
    UserTier.PREMIUM: TierCapabilities(
        semantic_search=True,
        graph_traversal=True,
        neighborhood_search=True,
        episodic_recall=True,
        pattern_access=True,
        max_results=100,
        max_graph_hops=3,
        max_history_days=365 * 10,  # Effectively unlimited
        include_source_episodes=True,
        include_confidence_scores=True,
        include_relationship_context=True,
    ),
    UserTier.TEAM: TierCapabilities(
        semantic_search=True,
        graph_traversal=True,
        neighborhood_search=True,
        episodic_recall=True,
        pattern_access=True,
        max_results=200,
        max_graph_hops=4,
        max_history_days=365 * 10,
        include_source_episodes=True,
        include_confidence_scores=True,
        include_relationship_context=True,
    ),
}


def get_capabilities(tier: UserTier) -> TierCapabilities:
    """Get the capabilities for a given user tier."""
    return TIER_CAPABILITIES[tier]


class QuestionQueryType(str, Enum):
    """Types of questions users can ask.

    Structured queries are available on free tier.
    Semantic/recall queries require premium.
    """

    # Free tier — structured lookups
    ENTITY_LOOKUP = "entity_lookup"       # "Who is Sarah?"
    FACT_LOOKUP = "fact_lookup"            # "What is Sarah's role?"
    RECENT_FACTS = "recent_facts"          # "What did I learn recently?"
    COUNT_QUERY = "count_query"            # "How many tasks do I have?"

    # Premium tier — full recall
    SEMANTIC_RECALL = "semantic_recall"    # "What did I say about X last week?"
    GRAPH_QUERY = "graph_query"            # "How are X and Y connected?"
    TEMPORAL_RECALL = "temporal_recall"    # "What was I thinking about in January?"
    PATTERN_QUERY = "pattern_query"        # "When am I most productive?"
    NEIGHBORHOOD = "neighborhood"          # "Everything related to project X"


# Which query types are available in each tier
FREE_QUERY_TYPES = frozenset({
    QuestionQueryType.ENTITY_LOOKUP,
    QuestionQueryType.FACT_LOOKUP,
    QuestionQueryType.RECENT_FACTS,
    QuestionQueryType.COUNT_QUERY,
})

PREMIUM_QUERY_TYPES = FREE_QUERY_TYPES | frozenset({
    QuestionQueryType.SEMANTIC_RECALL,
    QuestionQueryType.GRAPH_QUERY,
    QuestionQueryType.TEMPORAL_RECALL,
    QuestionQueryType.PATTERN_QUERY,
    QuestionQueryType.NEIGHBORHOOD,
})


def is_query_allowed(query_type: QuestionQueryType, tier: UserTier) -> bool:
    """Check if a query type is allowed for the given tier."""
    if tier in (UserTier.PREMIUM, UserTier.TEAM):
        return query_type in PREMIUM_QUERY_TYPES
    return query_type in FREE_QUERY_TYPES


class QuestionRequest(BaseModel):
    """A QUESTION intent query with tier-aware processing."""

    user_id: str
    query: str = Field(description="The natural language question")
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
        description="Filter by entity type (person, place, project, etc.)",
    )
    max_results: int | None = Field(
        default=None, ge=1, le=200,
        description="Max results (capped by tier)",
    )
    time_range_days: int | None = Field(
        default=None, ge=1,
        description="Look back N days (capped by tier)",
    )


class QuestionResult(BaseModel):
    """Result from processing a QUESTION intent."""

    query: str
    query_type: QuestionQueryType
    tier: UserTier
    results: list[dict[str, Any]] = Field(default_factory=list)
    result_count: int = 0
    total_available: int = 0  # How many results exist (may exceed tier limit)
    truncated: bool = False
    answer_summary: str = ""  # Natural language summary of results

    # Premium-only fields (empty for free tier)
    source_episodes: list[str] = Field(default_factory=list)
    confidence_scores: list[float] = Field(default_factory=list)
    relationship_context: list[dict[str, Any]] = Field(default_factory=list)

    # Upgrade hint (anti-shame: informational, never guilt-inducing)
    upgrade_hint: str | None = None


def classify_question_type(query: str) -> QuestionQueryType:
    """Auto-detect the question type from natural language.

    Uses keyword heuristics for fast classification. In production,
    this would be backed by the Flash-Lite model.
    """
    q = query.lower().strip()

    # Count queries (check early since "this week" could match temporal)
    count_keywords = ["how many", "how much", "count", "number of", "total"]
    if any(kw in q for kw in count_keywords):
        return QuestionQueryType.COUNT_QUERY

    # Pattern queries
    pattern_keywords = [
        "when am i", "when do i", "pattern", "usually", "tend to",
        "most productive", "energy", "rhythm", "habit",
    ]
    if any(kw in q for kw in pattern_keywords):
        return QuestionQueryType.PATTERN_QUERY

    # Semantic recall ("what did I say about...") — check before temporal
    # so "what did I say last week" is recall, not temporal
    recall_keywords = [
        "what did i say", "what did i mention", "what did i think",
        "did i ever", "have i ever", "do i remember", "recall",
    ]
    if any(kw in q for kw in recall_keywords):
        return QuestionQueryType.SEMANTIC_RECALL

    # Temporal recall
    temporal_keywords = [
        "last week", "last month", "yesterday", "in january", "in february",
        "in march", "in april", "in may", "in june", "in july", "in august",
        "in september", "in october", "in november", "in december",
        "this morning", "today", "this week", "last year",
    ]
    if any(kw in q for kw in temporal_keywords):
        return QuestionQueryType.TEMPORAL_RECALL

    # Graph queries ("how are X and Y connected")
    graph_keywords = [
        "connected", "related to", "relationship between",
        "link between", "connection",
    ]
    if any(kw in q for kw in graph_keywords):
        return QuestionQueryType.GRAPH_QUERY

    # Neighborhood queries ("everything about X")
    neighborhood_keywords = [
        "everything about", "everything related", "all about",
        "tell me about", "what do i know about",
    ]
    if any(kw in q for kw in neighborhood_keywords):
        return QuestionQueryType.NEIGHBORHOOD

    # Fact lookup ("what is X's role", "what is X")
    fact_keywords = ["what is", "what's", "what are", "who is", "who's", "where is"]
    if any(kw in q for kw in fact_keywords):
        # If it's a simple "who is X" style, it's entity lookup
        if any(q.startswith(kw) for kw in ["who is", "who's"]):
            return QuestionQueryType.ENTITY_LOOKUP
        return QuestionQueryType.FACT_LOOKUP

    # Default to entity lookup for simple questions, semantic recall for complex ones
    if len(q.split()) <= 5:
        return QuestionQueryType.ENTITY_LOOKUP
    return QuestionQueryType.SEMANTIC_RECALL


def format_free_tier_response(
    results: list[dict[str, Any]],
    query: str,
    query_type: QuestionQueryType,
    total_available: int,
    caps: TierCapabilities,
) -> QuestionResult:
    """Format a response appropriate for free-tier users.

    Free tier gets structured results without source episodes,
    confidence scores, or relationship context. Response is still
    useful and complete within its scope — never an empty "upgrade" wall.
    """
    truncated = total_available > caps.max_results
    visible = results[:caps.max_results]

    # Generate a natural language summary
    if not visible:
        summary = "I don't have information about that yet."
    elif query_type == QuestionQueryType.ENTITY_LOOKUP:
        names = [r.get("name", r.get("content", "")) for r in visible]
        summary = f"Found {len(visible)} match{'es' if len(visible) != 1 else ''}: {', '.join(names[:3])}"
        if len(names) > 3:
            summary += f" and {len(names) - 3} more"
    elif query_type == QuestionQueryType.COUNT_QUERY:
        summary = f"Found {total_available} matching items."
    elif query_type == QuestionQueryType.FACT_LOOKUP:
        facts = [r.get("content", "") for r in visible[:3]]
        summary = ". ".join(facts) if facts else "No matching facts found."
    else:
        summary = f"Found {len(visible)} result{'s' if len(visible) != 1 else ''}."

    upgrade_hint = None
    if truncated:
        upgrade_hint = (
            f"Showing {caps.max_results} of {total_available} results. "
            "Upgrade to Premium for full access to your knowledge graph."
        )

    return QuestionResult(
        query=query,
        query_type=query_type,
        tier=UserTier.FREE,
        results=visible,
        result_count=len(visible),
        total_available=total_available,
        truncated=truncated,
        answer_summary=summary,
        upgrade_hint=upgrade_hint,
    )


def format_premium_tier_response(
    results: list[dict[str, Any]],
    query: str,
    query_type: QuestionQueryType,
    total_available: int,
    caps: TierCapabilities,
    source_episodes: list[str] | None = None,
    confidence_scores: list[float] | None = None,
    relationship_context: list[dict[str, Any]] | None = None,
    tier: UserTier = UserTier.PREMIUM,
) -> QuestionResult:
    """Format a response with full premium-tier detail.

    Premium users get source episodes, confidence scores, and
    relationship context alongside the results.
    """
    visible = results[:caps.max_results]

    # Richer summary for premium
    if not visible:
        summary = "I don't have information about that yet. Try asking differently or adding more context."
    elif query_type == QuestionQueryType.SEMANTIC_RECALL:
        summary = f"Found {len(visible)} relevant memories."
        if confidence_scores:
            avg_conf = sum(confidence_scores[:len(visible)]) / len(confidence_scores[:len(visible)])
            summary += f" Average relevance: {avg_conf:.0%}."
    elif query_type == QuestionQueryType.GRAPH_QUERY:
        summary = f"Found {len(visible)} connections in your knowledge graph."
    elif query_type == QuestionQueryType.PATTERN_QUERY:
        summary = f"Found {len(visible)} behavioral pattern{'s' if len(visible) != 1 else ''}."
    elif query_type == QuestionQueryType.NEIGHBORHOOD:
        summary = f"Found {len(visible)} items related to your query."
    elif query_type == QuestionQueryType.TEMPORAL_RECALL:
        summary = f"Found {len(visible)} memories from that time period."
    else:
        summary = f"Found {len(visible)} result{'s' if len(visible) != 1 else ''}."

    return QuestionResult(
        query=query,
        query_type=query_type,
        tier=tier,
        results=visible,
        result_count=len(visible),
        total_available=total_available,
        truncated=total_available > caps.max_results,
        answer_summary=summary,
        source_episodes=source_episodes or [],
        confidence_scores=confidence_scores or [],
        relationship_context=relationship_context or [],
    )


def gate_query_for_tier(
    query_type: QuestionQueryType,
    tier: UserTier,
) -> tuple[bool, QuestionQueryType | None, str | None]:
    """Check if a query type is allowed and suggest alternatives if not.

    Returns:
        (allowed, fallback_query_type, upgrade_message)
        - allowed: True if the query type is permitted for this tier
        - fallback_query_type: Alternative query type for free tier (None if allowed)
        - upgrade_message: Anti-shame message about upgrade (None if allowed)
    """
    if is_query_allowed(query_type, tier):
        return (True, None, None)

    # Map premium-only queries to their best free-tier alternatives
    fallback_map: dict[QuestionQueryType, QuestionQueryType] = {
        QuestionQueryType.SEMANTIC_RECALL: QuestionQueryType.FACT_LOOKUP,
        QuestionQueryType.GRAPH_QUERY: QuestionQueryType.ENTITY_LOOKUP,
        QuestionQueryType.TEMPORAL_RECALL: QuestionQueryType.RECENT_FACTS,
        QuestionQueryType.PATTERN_QUERY: QuestionQueryType.RECENT_FACTS,
        QuestionQueryType.NEIGHBORHOOD: QuestionQueryType.ENTITY_LOOKUP,
    }

    fallback = fallback_map.get(query_type, QuestionQueryType.ENTITY_LOOKUP)

    # Anti-shame upgrade messages — informational, not guilt-inducing
    upgrade_messages: dict[QuestionQueryType, str] = {
        QuestionQueryType.SEMANTIC_RECALL: (
            "Full memory recall is available with Premium. "
            "Here's what I found with a focused search."
        ),
        QuestionQueryType.GRAPH_QUERY: (
            "Knowledge graph exploration is available with Premium. "
            "Here are the entities I found."
        ),
        QuestionQueryType.TEMPORAL_RECALL: (
            "Extended history search is available with Premium. "
            "Here's what I found from the last 30 days."
        ),
        QuestionQueryType.PATTERN_QUERY: (
            "Behavioral pattern insights are available with Premium. "
            "Here are some recent observations."
        ),
        QuestionQueryType.NEIGHBORHOOD: (
            "Full knowledge graph exploration is available with Premium. "
            "Here's what I found directly."
        ),
    }

    message = upgrade_messages.get(query_type, (
        "This feature is available with Premium. "
        "Here's what I found with a focused search."
    ))

    return (False, fallback, message)
