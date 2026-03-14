"""Question intent handler — tier-aware recall and retrieval service.

Processes QUESTION intent blurts by searching the user's knowledge graph,
episodic memory, and semantic memory. Access is gated by user tier:

- Free tier: Structured queries (entity lookup, fact lookup, counts).
  Returns focused results without source episodes or confidence data.
- Premium tier: Full recall with semantic search, graph traversal,
  neighborhood exploration, pattern access, and complete episodic history.

The service always returns something useful — free-tier users never see
empty "upgrade to continue" walls. Premium features are offered
contextually with anti-shame messaging.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from blurt.memory.episodic import EpisodicMemoryStore
from blurt.memory.semantic import SemanticMemoryStore
from blurt.models.entities import EntityType
from blurt.services.access_control import (
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
)
from blurt.services.recall import PersonalHistoryRecallEngine

logger = logging.getLogger(__name__)


class QuestionService:
    """Tier-aware question answering service.

    Handles QUESTION intent by querying the appropriate memory tier
    based on the user's subscription level. Ensures free-tier users
    get useful structured results while premium users get full recall.

    Usage::

        service = QuestionService(
            semantic_store=semantic_store,
            episodic_store=episodic_store,
        )

        # Free tier — structured query
        result = await service.answer(
            request=QuestionRequest(user_id="u1", query="Who is Sarah?"),
            tier=UserTier.FREE,
        )
        assert result.query_type == QuestionQueryType.ENTITY_LOOKUP
        assert len(result.source_episodes) == 0  # Not available on free tier

        # Premium tier — full recall
        result = await service.answer(
            request=QuestionRequest(user_id="u1", query="What did I say about the project?"),
            tier=UserTier.PREMIUM,
        )
        assert result.query_type == QuestionQueryType.SEMANTIC_RECALL
        assert len(result.source_episodes) > 0  # Available on premium
    """

    def __init__(
        self,
        semantic_store: SemanticMemoryStore | None = None,
        episodic_store: EpisodicMemoryStore | None = None,
        recall_engine: PersonalHistoryRecallEngine | None = None,
    ) -> None:
        self._semantic = semantic_store
        self._episodic = episodic_store
        self._recall_engine = recall_engine

    async def answer(
        self,
        request: QuestionRequest,
        tier: UserTier = UserTier.FREE,
    ) -> QuestionResult:
        """Answer a QUESTION intent with tier-appropriate recall.

        Auto-detects query type if not specified, gates access by tier,
        and falls back to free-tier alternatives for premium-only queries.

        Args:
            request: The question request with query and optional filters.
            tier: The user's subscription tier.

        Returns:
            QuestionResult with tier-appropriate results and formatting.
        """
        caps = get_capabilities(tier)

        # Auto-detect query type if not provided
        query_type = request.query_type or classify_question_type(request.query)

        # Gate access by tier
        allowed, fallback, upgrade_msg = gate_query_for_tier(query_type, tier)

        if not allowed:
            # Use fallback query type for free tier
            actual_query_type = fallback or QuestionQueryType.ENTITY_LOOKUP
            logger.info(
                "Query type %s gated for tier %s, falling back to %s",
                query_type.value,
                tier.value,
                actual_query_type.value,
            )
            result = await self._execute_query(
                request, actual_query_type, caps, tier
            )
            # Add the upgrade hint (anti-shame)
            result.upgrade_hint = upgrade_msg
            # Record the original query type that was requested
            result.query_type = query_type
            return result

        return await self._execute_query(request, query_type, caps, tier)

    async def _execute_query(
        self,
        request: QuestionRequest,
        query_type: QuestionQueryType,
        caps: TierCapabilities,
        tier: UserTier,
    ) -> QuestionResult:
        """Execute the appropriate query handler based on query type."""
        # Apply tier limits
        max_results = min(
            request.max_results or caps.max_results,
            caps.max_results,
        )
        time_range_days = min(
            request.time_range_days or caps.max_history_days,
            caps.max_history_days,
        )

        handlers = {
            QuestionQueryType.ENTITY_LOOKUP: self._entity_lookup,
            QuestionQueryType.FACT_LOOKUP: self._fact_lookup,
            QuestionQueryType.RECENT_FACTS: self._recent_facts,
            QuestionQueryType.COUNT_QUERY: self._count_query,
            QuestionQueryType.SEMANTIC_RECALL: self._semantic_recall,
            QuestionQueryType.GRAPH_QUERY: self._graph_query,
            QuestionQueryType.TEMPORAL_RECALL: self._temporal_recall,
            QuestionQueryType.PATTERN_QUERY: self._pattern_query,
            QuestionQueryType.NEIGHBORHOOD: self._neighborhood_query,
        }

        handler = handlers.get(query_type, self._entity_lookup)
        results, total, extras = await handler(
            request, max_results, time_range_days
        )

        # Format response based on tier
        if tier in (UserTier.PREMIUM, UserTier.TEAM):
            return format_premium_tier_response(
                results=results,
                query=request.query,
                query_type=query_type,
                total_available=total,
                caps=caps,
                source_episodes=extras.get("source_episodes"),
                confidence_scores=extras.get("confidence_scores"),
                relationship_context=extras.get("relationship_context"),
                tier=tier,
            )
        else:
            return format_free_tier_response(
                results=results,
                query=request.query,
                query_type=query_type,
                total_available=total,
                caps=caps,
            )

    # ── Free-tier query handlers ─────────────────────────────────────

    async def _entity_lookup(
        self,
        request: QuestionRequest,
        max_results: int,
        time_range_days: int,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Look up entities by name or type — available on all tiers."""
        if self._semantic is None:
            return [], 0, {}

        results: list[dict[str, Any]] = []

        if request.entity_name:
            entity = await self._semantic.find_entity_by_name(request.entity_name)
            if entity:
                results.append({
                    "name": entity.name,
                    "type": entity.entity_type.value,
                    "mention_count": entity.mention_count,
                    "last_seen": entity.last_seen.isoformat(),
                    "attributes": entity.attributes,
                    "content": f"{entity.entity_type.value}: {entity.name}",
                })
        else:
            # Search by query text
            entity_type = None
            if request.entity_type:
                try:
                    entity_type = EntityType(request.entity_type)
                except ValueError:
                    pass

            search_results = await self._semantic.search_entities_by_query(
                query=request.query,
                top_k=max_results,
                entity_type=entity_type,
            )
            for sr in search_results:
                results.append({
                    "name": sr.content,
                    "type": sr.metadata.get("entity_type", ""),
                    "mention_count": sr.metadata.get("mention_count", 0),
                    "last_seen": sr.metadata.get("last_seen", ""),
                    "content": sr.content,
                    "similarity": sr.similarity_score,
                })

        total = len(results)
        return results[:max_results], total, {}

    async def _fact_lookup(
        self,
        request: QuestionRequest,
        max_results: int,
        time_range_days: int,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Look up facts — available on all tiers."""
        if self._semantic is None:
            return [], 0, {}

        results: list[dict[str, Any]] = []

        # If entity specified, get facts about that entity
        if request.entity_name:
            entity = await self._semantic.find_entity_by_name(request.entity_name)
            if entity:
                facts = await self._semantic.get_entity_facts(entity.id)
                for fact in facts:
                    results.append({
                        "content": fact.content,
                        "fact_type": fact.fact_type.value,
                        "confidence": fact.confidence,
                        "confirmed_count": fact.confirmation_count,
                    })
        else:
            # Search facts by query
            search_results = await self._semantic.search(
                query=request.query,
                top_k=max_results,
                item_types=["fact"],
            )
            for sr in search_results:
                results.append({
                    "content": sr.content,
                    "fact_type": sr.metadata.get("fact_type", ""),
                    "confidence": sr.metadata.get("confidence", 0),
                    "similarity": sr.similarity_score,
                })

        total = len(results)
        return results[:max_results], total, {}

    async def _recent_facts(
        self,
        request: QuestionRequest,
        max_results: int,
        time_range_days: int,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Get recently learned facts — available on all tiers."""
        if self._semantic is None:
            return [], 0, {}

        all_facts = await self._semantic.get_all_facts()

        # Filter by time range
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (time_range_days * 86400)
        recent = [
            f for f in all_facts
            if f.last_confirmed.timestamp() >= cutoff
        ]

        # Sort by recency
        recent.sort(key=lambda f: f.last_confirmed, reverse=True)

        results = []
        for fact in recent:
            results.append({
                "content": fact.content,
                "fact_type": fact.fact_type.value,
                "confidence": fact.confidence,
                "last_confirmed": fact.last_confirmed.isoformat(),
            })

        total = len(results)
        return results[:max_results], total, {}

    async def _count_query(
        self,
        request: QuestionRequest,
        max_results: int,
        time_range_days: int,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Count entities or facts — available on all tiers."""
        if self._semantic is None:
            return [], 0, {}

        entity_type = None
        if request.entity_type:
            try:
                entity_type = EntityType(request.entity_type)
            except ValueError:
                pass

        entities = await self._semantic.get_all_entities(entity_type)
        facts = await self._semantic.get_all_facts()

        results = [{
            "content": f"Entities: {len(entities)}, Facts: {len(facts)}",
            "entity_count": len(entities),
            "fact_count": len(facts),
        }]

        if entity_type:
            results[0]["content"] = (
                f"{entity_type.value.title()} entities: {len(entities)}, "
                f"Total facts: {len(facts)}"
            )

        return results, 1, {}

    # ── Premium-tier query handlers ──────────────────────────────────

    async def _semantic_recall(
        self,
        request: QuestionRequest,
        max_results: int,
        time_range_days: int,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Full semantic recall — premium only.

        Delegates to PersonalHistoryRecallEngine when available for
        cross-source search with NL understanding, relationship search,
        and source context enrichment. Falls back to direct semantic
        store search otherwise.
        """
        # Use the full recall engine if available
        if self._recall_engine is not None:
            now = datetime.now(timezone.utc)
            time_start = now - timedelta(days=time_range_days)

            recall_response = await self._recall_engine.recall(
                user_id=request.user_id,
                query=request.query,
                max_results=max_results,
                time_start=time_start,
                enrich_context=True,
            )

            results = []
            source_episodes: list[str] = []
            confidence_scores: list[float] = []

            for rr in recall_response.results:
                result_item: dict[str, Any] = {
                    "content": rr.content,
                    "type": rr.source_type.value,
                    "similarity": rr.raw_similarity,
                    "relevance_score": rr.relevance_score,
                    "metadata": rr.metadata,
                }
                if rr.source_context is not None:
                    result_item["source_context"] = rr.source_context.to_dict()
                if rr.timestamp is not None:
                    result_item["timestamp"] = rr.timestamp.isoformat()

                results.append(result_item)
                confidence_scores.append(rr.relevance_score)

                # Collect source episode IDs
                if rr.source_type.value == "episode":
                    source_episodes.append(rr.source_id)

            total = len(results)
            extras: dict[str, Any] = {
                "source_episodes": source_episodes,
                "confidence_scores": confidence_scores,
            }
            if recall_response.query_understanding:
                extras["query_understanding"] = (
                    recall_response.query_understanding.to_dict()
                )
            return results, total, extras

        # Fallback: direct semantic store search
        if self._semantic is None:
            return [], 0, {}

        search_results = await self._semantic.search(
            query=request.query,
            top_k=max_results,
        )

        results = []
        source_episodes_fb: list[str] = []
        confidence_scores_fb: list[float] = []

        for sr in search_results:
            result_item_fb: dict[str, Any] = {
                "content": sr.content,
                "type": sr.item_type,
                "similarity": sr.similarity_score,
                "metadata": sr.metadata,
            }
            results.append(result_item_fb)
            confidence_scores_fb.append(sr.similarity_score)

            # Collect source episode IDs from facts
            if sr.item_type == "fact":
                fact_id = sr.item_id
                for fact in (await self._semantic.get_all_facts()):
                    if fact.id == fact_id and fact.source_blurt_ids:
                        source_episodes_fb.extend(fact.source_blurt_ids)

        total = len(results)
        extras_fb: dict[str, Any] = {
            "source_episodes": source_episodes_fb,
            "confidence_scores": confidence_scores_fb,
        }
        return results[:max_results], total, extras_fb

    async def _graph_query(
        self,
        request: QuestionRequest,
        max_results: int,
        time_range_days: int,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Graph traversal query — premium only."""
        if self._semantic is None:
            return [], 0, {}

        # Find the entity mentioned in the query
        entity = None
        if request.entity_name:
            entity = await self._semantic.find_entity_by_name(request.entity_name)

        results: list[dict[str, Any]] = []
        relationship_context: list[dict[str, Any]] = []

        if entity:
            connected = await self._semantic.get_connected_entities(entity.id)
            for other_entity, rel in connected:
                results.append({
                    "content": f"{entity.name} → {rel.relationship_type.value} → {other_entity.name}",
                    "source": entity.name,
                    "target": other_entity.name,
                    "relationship": rel.relationship_type.value,
                    "strength": rel.strength,
                })
                relationship_context.append({
                    "relationship_type": rel.relationship_type.value,
                    "strength": rel.strength,
                    "co_mentions": rel.co_mention_count,
                    "context_snippets": rel.context_snippets[:3],
                })
        else:
            # Fall back to semantic search
            search_results = await self._semantic.search(
                query=request.query,
                top_k=max_results,
            )
            for sr in search_results:
                results.append({
                    "content": sr.content,
                    "type": sr.item_type,
                    "similarity": sr.similarity_score,
                })

        total = len(results)
        extras: dict[str, Any] = {
            "relationship_context": relationship_context,
        }
        return results[:max_results], total, extras

    async def _temporal_recall(
        self,
        request: QuestionRequest,
        max_results: int,
        time_range_days: int,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Time-scoped memory recall — premium only."""
        if self._semantic is None:
            return [], 0, {}

        # Use semantic search with time filtering
        search_results = await self._semantic.search(
            query=request.query,
            top_k=max_results,
        )

        results = []
        confidence_scores: list[float] = []
        source_episodes: list[str] = []

        for sr in search_results:
            results.append({
                "content": sr.content,
                "type": sr.item_type,
                "similarity": sr.similarity_score,
                "metadata": sr.metadata,
            })
            confidence_scores.append(sr.similarity_score)

        total = len(results)
        extras: dict[str, Any] = {
            "source_episodes": source_episodes,
            "confidence_scores": confidence_scores,
        }
        return results[:max_results], total, extras

    async def _pattern_query(
        self,
        request: QuestionRequest,
        max_results: int,
        time_range_days: int,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Behavioral pattern query — premium only."""
        if self._semantic is None:
            return [], 0, {}

        patterns = await self._semantic.get_active_patterns()

        results = []
        confidence_scores: list[float] = []

        for pattern in patterns:
            results.append({
                "content": pattern.description,
                "pattern_type": pattern.pattern_type.value,
                "confidence": pattern.confidence,
                "observation_count": pattern.observation_count,
                "parameters": pattern.parameters,
            })
            confidence_scores.append(pattern.confidence)

        total = len(results)
        extras: dict[str, Any] = {
            "confidence_scores": confidence_scores,
        }
        return results[:max_results], total, extras

    async def _neighborhood_query(
        self,
        request: QuestionRequest,
        max_results: int,
        time_range_days: int,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        """Graph neighborhood exploration — premium only."""
        if self._semantic is None:
            return [], 0, {}

        caps = get_capabilities(UserTier.PREMIUM)

        search_results = await self._semantic.search_neighborhood(
            query=request.query,
            top_k=max_results,
            max_hops=caps.max_graph_hops,
        )

        results = []
        confidence_scores: list[float] = []

        for sr in search_results:
            results.append({
                "content": sr.content,
                "type": sr.item_type,
                "similarity": sr.similarity_score,
                "metadata": sr.metadata,
            })
            confidence_scores.append(sr.similarity_score)

        total = len(results)
        extras: dict[str, Any] = {
            "confidence_scores": confidence_scores,
        }
        return results[:max_results], total, extras
