"""Semantic memory tier — long-term knowledge graph with vector embeddings.

This is Tier 3 of Blurt's 3-tier memory architecture. It stores:
- Entity nodes: people, places, projects, organizations, topics, tools
- Relationship edges: connections between entities with co-mention strength
- Facts: learned attributes, preferences, habits, associations, aliases
- Patterns: behavioral patterns detected from accumulated observations

Every item gets a vector embedding (via Gemini 2 or local model) for
semantic search. Relationships strengthen with co-mentions and decay
with absence. Facts can be superseded by newer information.

The knowledge graph is append-oriented and grows with every interaction,
creating the compounding moat described in the product vision.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from blurt.clients.embeddings import EmbeddingProvider, cosine_similarity
from blurt.models.entities import (
    EntityNode,
    EntityType,
    Fact,
    FactType,
    LearnedPattern,
    PatternType,
    RelationshipEdge,
    RelationshipType,
    SemanticSearchResult,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Relationship strength decay: halves every 30 days without co-mention
STRENGTH_DECAY_HALF_LIFE_DAYS = 30.0

# Minimum strength before a relationship is considered dormant
MIN_RELATIONSHIP_STRENGTH = 0.01

# Maximum context snippets to keep per relationship
MAX_CONTEXT_SNIPPETS = 10

# Confidence threshold for promoting a pattern to active
PATTERN_PROMOTION_THRESHOLD = 0.7

# Minimum observations needed before a pattern is considered
MIN_PATTERN_OBSERVATIONS = 5


class SemanticMemoryStore:
    """Long-term knowledge graph storage with vector embeddings.

    Manages entities, relationships, facts, and patterns for a single user.
    Provides semantic search, relationship strengthening/decay, fact
    supersession, and pattern detection support.

    Storage is pluggable — uses in-memory dicts by default, can be backed
    by SQLite, PostgreSQL, or any persistent store.
    """

    def __init__(
        self,
        user_id: str,
        embedding_provider: EmbeddingProvider,
    ):
        self.user_id = user_id
        self._embeddings = embedding_provider

        # In-memory storage (swap for persistent backend)
        self._entities: dict[str, EntityNode] = {}
        self._relationships: dict[str, RelationshipEdge] = {}
        self._facts: dict[str, Fact] = {}
        self._patterns: dict[str, LearnedPattern] = {}

        # Indexes for fast lookup
        self._entity_by_name: dict[str, str] = {}  # normalized_name -> id
        self._entity_by_alias: dict[str, str] = {}  # alias -> entity_id
        self._relationships_by_entity: dict[str, list[str]] = {}  # entity_id -> [rel_ids]
        self._facts_by_entity: dict[str, list[str]] = {}  # entity_id -> [fact_ids]

    # ── Entity Operations ─────────────────────────────────────────────

    async def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> EntityNode:
        """Add a new entity to the knowledge graph, or return existing one.

        If an entity with the same normalized name already exists, increments
        its mention count and updates last_seen instead of creating a duplicate.
        """
        normalized = name.lower().strip()

        # Check for existing entity by name or alias
        existing_id = self._entity_by_name.get(normalized)
        if existing_id is None:
            existing_id = self._entity_by_alias.get(normalized)

        if existing_id and existing_id in self._entities:
            entity = self._entities[existing_id]
            entity.mention_count += 1
            entity.last_seen = _utcnow()
            entity.updated_at = _utcnow()
            if attributes:
                entity.attributes.update(attributes)
            if aliases:
                for alias in aliases:
                    a_norm = alias.lower().strip()
                    if a_norm not in entity.aliases:
                        entity.aliases.append(a_norm)
                        self._entity_by_alias[a_norm] = entity.id
            return entity

        # Create new entity
        embedding = await self._embeddings.embed(f"{entity_type.value}: {name}")

        entity = EntityNode(
            user_id=self.user_id,
            name=name,
            normalized_name=normalized,
            entity_type=entity_type,
            aliases=[a.lower().strip() for a in (aliases or [])],
            attributes=attributes or {},
            mention_count=1,
            embedding=embedding,
        )

        self._entities[entity.id] = entity
        self._entity_by_name[normalized] = entity.id
        for alias in entity.aliases:
            self._entity_by_alias[alias] = entity.id

        return entity

    async def get_entity(self, entity_id: str) -> EntityNode | None:
        """Retrieve an entity by ID."""
        return self._entities.get(entity_id)

    async def find_entity_by_name(self, name: str) -> EntityNode | None:
        """Find an entity by name or alias."""
        normalized = name.lower().strip()
        entity_id = self._entity_by_name.get(normalized) or self._entity_by_alias.get(
            normalized
        )
        if entity_id:
            return self._entities.get(entity_id)
        return None

    async def get_all_entities(
        self, entity_type: EntityType | None = None
    ) -> list[EntityNode]:
        """Get all entities, optionally filtered by type."""
        entities = list(self._entities.values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        return entities

    async def update_entity_embedding(self, entity_id: str) -> EntityNode | None:
        """Regenerate the embedding for an entity based on its current state."""
        entity = self._entities.get(entity_id)
        if not entity:
            return None

        # Build a rich text representation for embedding
        parts = [f"{entity.entity_type.value}: {entity.name}"]
        if entity.aliases:
            parts.append(f"also known as: {', '.join(entity.aliases)}")
        if entity.attributes:
            for k, v in entity.attributes.items():
                parts.append(f"{k}: {v}")

        # Include related facts
        fact_ids = self._facts_by_entity.get(entity_id, [])
        for fid in fact_ids[:5]:
            fact = self._facts.get(fid)
            if fact and fact.is_active:
                parts.append(fact.content)

        text = ". ".join(parts)
        entity.embedding = await self._embeddings.embed(text)
        entity.updated_at = _utcnow()
        return entity

    # ── Relationship Operations ───────────────────────────────────────

    async def add_or_strengthen_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: RelationshipType,
        context: str | None = None,
    ) -> RelationshipEdge:
        """Add a new relationship or strengthen an existing one.

        If the relationship already exists, increments co-mention count
        and boosts strength. Strength follows a log curve so early
        mentions have more impact than later ones.
        """
        existing = await self._find_relationship(
            source_entity_id, target_entity_id, relationship_type
        )

        if existing:
            existing.co_mention_count += 1
            # Log-based strength growth: diminishing returns
            existing.strength = min(
                100.0,
                math.log2(existing.co_mention_count + 1) * 10,
            )
            existing.last_seen = _utcnow()
            existing.updated_at = _utcnow()
            if context:
                existing.context_snippets.append(context)
                if len(existing.context_snippets) > MAX_CONTEXT_SNIPPETS:
                    existing.context_snippets = existing.context_snippets[
                        -MAX_CONTEXT_SNIPPETS:
                    ]
            return existing

        # Create new relationship
        edge = RelationshipEdge(
            user_id=self.user_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            strength=1.0,
            context_snippets=[context] if context else [],
        )

        self._relationships[edge.id] = edge
        self._relationships_by_entity.setdefault(source_entity_id, []).append(edge.id)
        self._relationships_by_entity.setdefault(target_entity_id, []).append(edge.id)

        return edge

    async def _find_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType,
    ) -> RelationshipEdge | None:
        """Find an existing relationship between two entities."""
        rel_ids = self._relationships_by_entity.get(source_id, [])
        for rid in rel_ids:
            rel = self._relationships.get(rid)
            if rel and rel.relationship_type == rel_type:
                if (
                    rel.source_entity_id == source_id
                    and rel.target_entity_id == target_id
                ) or (
                    rel.source_entity_id == target_id
                    and rel.target_entity_id == source_id
                ):
                    return rel
        return None

    async def get_entity_relationships(
        self, entity_id: str
    ) -> list[RelationshipEdge]:
        """Get all relationships for an entity."""
        rel_ids = self._relationships_by_entity.get(entity_id, [])
        return [
            self._relationships[rid] for rid in rel_ids if rid in self._relationships
        ]

    async def get_connected_entities(
        self, entity_id: str
    ) -> list[tuple[EntityNode, RelationshipEdge]]:
        """Get all entities connected to the given entity with their relationships."""
        relationships = await self.get_entity_relationships(entity_id)
        results = []
        for rel in relationships:
            other_id = (
                rel.target_entity_id
                if rel.source_entity_id == entity_id
                else rel.source_entity_id
            )
            other_entity = self._entities.get(other_id)
            if other_entity:
                results.append((other_entity, rel))
        # Sort by relationship strength descending
        results.sort(key=lambda x: x[1].strength, reverse=True)
        return results

    async def decay_relationships(self, as_of: datetime | None = None) -> int:
        """Apply time-based decay to all relationships.

        Relationships weaken when entities aren't co-mentioned.
        Returns the number of relationships that decayed below threshold.
        """
        now = as_of or _utcnow()
        dormant_count = 0

        for rel in list(self._relationships.values()):
            days_since = (now - rel.last_seen).total_seconds() / 86400
            if days_since <= 0:
                continue

            # Exponential decay with half-life
            decay_factor = math.pow(0.5, days_since / STRENGTH_DECAY_HALF_LIFE_DAYS)
            rel.strength = max(MIN_RELATIONSHIP_STRENGTH, rel.strength * decay_factor)
            rel.updated_at = now

            if rel.strength <= MIN_RELATIONSHIP_STRENGTH:
                dormant_count += 1

        return dormant_count

    # ── Fact Operations ───────────────────────────────────────────────

    async def add_fact(
        self,
        content: str,
        fact_type: FactType,
        subject_entity_id: str | None = None,
        source_blurt_id: str | None = None,
        confidence: float = 1.0,
    ) -> Fact:
        """Store a new fact learned from user interaction.

        If a similar fact already exists (high embedding similarity),
        confirms it instead of creating a duplicate. If a contradicting
        fact exists, supersedes the old one.
        """
        embedding = await self._embeddings.embed(content)

        # Check for existing similar facts
        existing = await self._find_similar_fact(
            embedding, subject_entity_id, threshold=0.92
        )

        if existing:
            existing.confirmation_count += 1
            existing.last_confirmed = _utcnow()
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.updated_at = _utcnow()
            if source_blurt_id:
                existing.source_blurt_ids.append(source_blurt_id)
            return existing

        fact = Fact(
            user_id=self.user_id,
            fact_type=fact_type,
            subject_entity_id=subject_entity_id,
            content=content,
            confidence=confidence,
            source_blurt_ids=[source_blurt_id] if source_blurt_id else [],
            embedding=embedding,
        )

        self._facts[fact.id] = fact
        if subject_entity_id:
            self._facts_by_entity.setdefault(subject_entity_id, []).append(fact.id)

        return fact

    async def _find_similar_fact(
        self,
        embedding: list[float],
        subject_entity_id: str | None,
        threshold: float = 0.92,
    ) -> Fact | None:
        """Find an existing fact with high semantic similarity."""
        best_match: Fact | None = None
        best_score = 0.0

        candidates: list[Fact] | dict[str, Fact] = list(self._facts.values())
        if subject_entity_id:
            fact_ids = self._facts_by_entity.get(subject_entity_id, [])
            candidates = [self._facts[fid] for fid in fact_ids if fid in self._facts]

        for fact in candidates:
            if not fact.is_active or fact.embedding is None:
                continue
            score = cosine_similarity(embedding, fact.embedding)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = fact

        return best_match

    async def supersede_fact(
        self,
        old_fact_id: str,
        new_content: str,
        source_blurt_id: str | None = None,
    ) -> Fact | None:
        """Replace an old fact with a new one.

        The old fact is marked inactive and linked to the new one.
        This preserves history while keeping the graph current.
        """
        old_fact = self._facts.get(old_fact_id)
        if not old_fact:
            return None

        new_fact = await self.add_fact(
            content=new_content,
            fact_type=old_fact.fact_type,
            subject_entity_id=old_fact.subject_entity_id,
            source_blurt_id=source_blurt_id,
        )

        old_fact.is_active = False
        old_fact.superseded_by = new_fact.id
        old_fact.updated_at = _utcnow()

        return new_fact

    async def get_entity_facts(
        self,
        entity_id: str,
        active_only: bool = True,
    ) -> list[Fact]:
        """Get all facts about a specific entity."""
        fact_ids = self._facts_by_entity.get(entity_id, [])
        facts = [self._facts[fid] for fid in fact_ids if fid in self._facts]
        if active_only:
            facts = [f for f in facts if f.is_active]
        return facts

    async def get_all_facts(
        self,
        fact_type: FactType | None = None,
        active_only: bool = True,
    ) -> list[Fact]:
        """Get all facts, optionally filtered by type."""
        facts = list(self._facts.values())
        if active_only:
            facts = [f for f in facts if f.is_active]
        if fact_type:
            facts = [f for f in facts if f.fact_type == fact_type]
        return facts

    # ── Pattern Operations ────────────────────────────────────────────

    async def add_pattern(
        self,
        pattern_type: PatternType,
        description: str,
        parameters: dict[str, Any] | None = None,
        confidence: float = 0.5,
        observation_count: int = 0,
        supporting_evidence: list[str] | None = None,
    ) -> LearnedPattern:
        """Store a newly detected behavioral pattern."""
        embedding = await self._embeddings.embed(description)

        pattern = LearnedPattern(
            user_id=self.user_id,
            pattern_type=pattern_type,
            description=description,
            parameters=parameters or {},
            confidence=confidence,
            observation_count=observation_count,
            supporting_evidence=supporting_evidence or [],
            embedding=embedding,
            is_active=confidence >= PATTERN_PROMOTION_THRESHOLD,
        )

        self._patterns[pattern.id] = pattern
        return pattern

    async def confirm_pattern(
        self,
        pattern_id: str,
        observation_id: str | None = None,
    ) -> LearnedPattern | None:
        """Confirm a pattern with new supporting evidence.

        Increases confidence and may promote the pattern to active.
        """
        pattern = self._patterns.get(pattern_id)
        if not pattern:
            return None

        pattern.observation_count += 1
        if observation_id:
            pattern.supporting_evidence.append(observation_id)

        # Confidence grows toward 1.0 with more observations
        # Using a steeper sigmoid so patterns promote after ~5-7 observations
        pattern.confidence = min(
            1.0,
            1.0 - (1.0 / (1.0 + pattern.observation_count * 0.5)),
        )

        # Promote to active if threshold met
        if (
            pattern.confidence >= PATTERN_PROMOTION_THRESHOLD
            and pattern.observation_count >= MIN_PATTERN_OBSERVATIONS
        ):
            pattern.is_active = True

        pattern.last_confirmed = _utcnow()
        pattern.updated_at = _utcnow()
        return pattern

    async def get_active_patterns(
        self,
        pattern_type: PatternType | None = None,
    ) -> list[LearnedPattern]:
        """Get all active patterns, optionally filtered by type."""
        patterns = [p for p in self._patterns.values() if p.is_active]
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        return patterns

    async def get_all_patterns(self) -> list[LearnedPattern]:
        """Get all patterns, including inactive ones."""
        return list(self._patterns.values())

    # ── Semantic Search ───────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 10,
        item_types: list[str] | None = None,
        min_similarity: float = 0.3,
    ) -> list[SemanticSearchResult]:
        """Search across the entire knowledge graph using semantic similarity.

        Searches entities, facts, and patterns using vector embeddings.
        Returns results ranked by cosine similarity.
        """
        query_embedding = await self._embeddings.embed(query)

        results: list[SemanticSearchResult] = []
        search_types = item_types or ["entity", "fact", "pattern"]

        if "entity" in search_types:
            for entity in self._entities.values():
                if entity.embedding is None:
                    continue
                raw_score = cosine_similarity(query_embedding, entity.embedding)
                if raw_score >= min_similarity:
                    results.append(
                        SemanticSearchResult(
                            item_type="entity",
                            item_id=entity.id,
                            content=f"{entity.entity_type.value}: {entity.name}",
                            similarity_score=max(0.0, raw_score),
                            metadata={
                                "entity_type": entity.entity_type.value,
                                "mention_count": entity.mention_count,
                                "last_seen": entity.last_seen.isoformat(),
                            },
                        )
                    )

        if "fact" in search_types:
            for fact in self._facts.values():
                if not fact.is_active or fact.embedding is None:
                    continue
                raw_score = cosine_similarity(query_embedding, fact.embedding)
                if raw_score >= min_similarity:
                    results.append(
                        SemanticSearchResult(
                            item_type="fact",
                            item_id=fact.id,
                            content=fact.content,
                            similarity_score=max(0.0, raw_score),
                            metadata={
                                "fact_type": fact.fact_type.value,
                                "confidence": fact.confidence,
                                "confirmation_count": fact.confirmation_count,
                            },
                        )
                    )

        if "pattern" in search_types:
            for pattern in self._patterns.values():
                if not pattern.is_active or pattern.embedding is None:
                    continue
                raw_score = cosine_similarity(query_embedding, pattern.embedding)
                if raw_score >= min_similarity:
                    results.append(
                        SemanticSearchResult(
                            item_type="pattern",
                            item_id=pattern.id,
                            content=pattern.description,
                            similarity_score=max(0.0, raw_score),
                            metadata={
                                "pattern_type": pattern.pattern_type.value,
                                "confidence": pattern.confidence,
                                "observation_count": pattern.observation_count,
                            },
                        )
                    )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    # ── Semantic Similarity Search & Retrieval ─────────────────────────

    async def search_similar_entities(
        self,
        entity_id: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        entity_type: EntityType | None = None,
    ) -> list[SemanticSearchResult]:
        """Find entities semantically similar to a given entity.

        Uses the entity's embedding to find other entities with similar
        vector representations. Useful for discovering implicit connections
        that haven't been explicitly mentioned together.

        Args:
            entity_id: The seed entity to find similar entities for.
            top_k: Maximum number of results to return.
            min_similarity: Minimum cosine similarity threshold (0.0–1.0).
            entity_type: Optionally filter results to a specific entity type.

        Returns:
            List of SemanticSearchResult sorted by similarity score descending.
        """
        seed = self._entities.get(entity_id)
        if not seed or seed.embedding is None:
            return []

        results: list[SemanticSearchResult] = []
        for entity in self._entities.values():
            if entity.id == entity_id or entity.embedding is None:
                continue
            if entity_type and entity.entity_type != entity_type:
                continue

            raw_score = cosine_similarity(seed.embedding, entity.embedding)
            score = max(0.0, raw_score)
            if score >= min_similarity:
                results.append(
                    SemanticSearchResult(
                        item_type="entity",
                        item_id=entity.id,
                        content=f"{entity.entity_type.value}: {entity.name}",
                        similarity_score=score,
                        metadata={
                            "entity_type": entity.entity_type.value,
                            "mention_count": entity.mention_count,
                            "last_seen": entity.last_seen.isoformat(),
                        },
                    )
                )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    async def search_entities_by_query(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        entity_type: EntityType | None = None,
        min_mentions: int = 0,
    ) -> list[SemanticSearchResult]:
        """Search entities by natural language query with optional filters.

        Combines embedding-based similarity with attribute filters for
        precise entity retrieval. Supports filtering by entity type and
        minimum mention count to prioritize well-known entities.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results to return.
            min_similarity: Minimum cosine similarity threshold.
            entity_type: Filter results to a specific entity type.
            min_mentions: Minimum mention count filter.

        Returns:
            List of SemanticSearchResult sorted by similarity score descending.
        """
        query_embedding = await self._embeddings.embed(query)

        results: list[SemanticSearchResult] = []
        for entity in self._entities.values():
            if entity.embedding is None:
                continue
            if entity_type and entity.entity_type != entity_type:
                continue
            if entity.mention_count < min_mentions:
                continue

            raw_score = cosine_similarity(query_embedding, entity.embedding)
            score = max(0.0, raw_score)
            if score >= min_similarity:
                results.append(
                    SemanticSearchResult(
                        item_type="entity",
                        item_id=entity.id,
                        content=f"{entity.entity_type.value}: {entity.name}",
                        similarity_score=score,
                        metadata={
                            "entity_type": entity.entity_type.value,
                            "mention_count": entity.mention_count,
                            "last_seen": entity.last_seen.isoformat(),
                            "attributes": entity.attributes,
                        },
                    )
                )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    async def search_neighborhood(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        max_hops: int = 2,
        relationship_types: list[RelationshipType] | None = None,
    ) -> list[SemanticSearchResult]:
        """Graph-aware semantic search combining vector similarity with graph traversal.

        Starts with a semantic search to find seed entities, then expands
        through the relationship graph up to ``max_hops`` hops, scoring
        connected entities by a blend of embedding similarity and
        relationship strength. This surfaces contextually relevant entities
        that may not match the query directly but are strongly connected
        to matching entities.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results to return.
            min_similarity: Minimum cosine similarity for seed entities.
            max_hops: Maximum graph traversal depth from seed entities.
            relationship_types: Filter traversal to specific relationship types.

        Returns:
            List of SemanticSearchResult sorted by blended score descending.
        """
        query_embedding = await self._embeddings.embed(query)

        # Phase 1: Find seed entities by embedding similarity
        seed_scores: dict[str, float] = {}
        for entity in self._entities.values():
            if entity.embedding is None:
                continue
            score = cosine_similarity(query_embedding, entity.embedding)
            if score >= min_similarity:
                seed_scores[entity.id] = score

        # Phase 2: Expand through graph, accumulating blended scores
        # visited tracks the best score seen for each entity
        visited: dict[str, float] = dict(seed_scores)
        frontier = list(seed_scores.keys())

        for hop in range(max_hops):
            next_frontier: list[str] = []
            for entity_id in frontier:
                parent_score = visited[entity_id]
                rel_ids = self._relationships_by_entity.get(entity_id, [])

                for rid in rel_ids:
                    rel = self._relationships.get(rid)
                    if rel is None:
                        continue
                    if relationship_types and rel.relationship_type not in relationship_types:
                        continue

                    other_id = (
                        rel.target_entity_id
                        if rel.source_entity_id == entity_id
                        else rel.source_entity_id
                    )

                    # Blend: parent score decays with hops, boosted by relationship strength
                    # Strength is 0–100, normalize to 0–1 for blending
                    hop_decay = 0.5 ** (hop + 1)
                    strength_factor = min(rel.strength / 50.0, 1.0)
                    neighbor_score = parent_score * hop_decay * strength_factor

                    # Also incorporate direct embedding similarity if available
                    other = self._entities.get(other_id)
                    if other and other.embedding is not None:
                        direct_sim = cosine_similarity(query_embedding, other.embedding)
                        # Weighted average: 60% graph-based, 40% direct similarity
                        neighbor_score = 0.6 * neighbor_score + 0.4 * direct_sim

                    if other_id not in visited or neighbor_score > visited[other_id]:
                        visited[other_id] = neighbor_score
                        next_frontier.append(other_id)

            frontier = next_frontier

        # Build results
        results: list[SemanticSearchResult] = []
        for eid, score in visited.items():
            entity = self._entities.get(eid)
            if not entity:
                continue
            clamped_score = max(0.0, min(score, 1.0))
            is_seed = eid in seed_scores
            results.append(
                SemanticSearchResult(
                    item_type="entity",
                    item_id=entity.id,
                    content=f"{entity.entity_type.value}: {entity.name}",
                    similarity_score=clamped_score,
                    metadata={
                        "entity_type": entity.entity_type.value,
                        "mention_count": entity.mention_count,
                        "is_seed": is_seed,
                        "last_seen": entity.last_seen.isoformat(),
                    },
                )
            )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    async def recall(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.2,
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        """Full personal history recall — the primary retrieval interface.

        Searches across entities, facts, patterns, and relationships,
        then enriches top results with their graph context (connected
        entities, related facts). This powers the "full recall of personal
        history" capability described in the product vision.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results to return.
            min_similarity: Minimum similarity threshold.
            include_context: Whether to enrich entity results with full graph context.

        Returns:
            List of dicts, each containing:
              - item_type, item_id, content, similarity_score
              - context (if include_context and item is an entity): connected entities, facts
              - related_entities (for facts with a subject entity)
        """
        query_embedding = await self._embeddings.embed(query)

        raw_results: list[dict[str, Any]] = []

        # Search entities
        for entity in self._entities.values():
            if entity.embedding is None:
                continue
            score = cosine_similarity(query_embedding, entity.embedding)
            if score >= min_similarity:
                raw_results.append({
                    "item_type": "entity",
                    "item_id": entity.id,
                    "content": f"{entity.entity_type.value}: {entity.name}",
                    "similarity_score": score,
                    "entity_type": entity.entity_type.value,
                    "mention_count": entity.mention_count,
                })

        # Search facts
        for fact in self._facts.values():
            if not fact.is_active or fact.embedding is None:
                continue
            score = cosine_similarity(query_embedding, fact.embedding)
            if score >= min_similarity:
                raw_results.append({
                    "item_type": "fact",
                    "item_id": fact.id,
                    "content": fact.content,
                    "similarity_score": score,
                    "fact_type": fact.fact_type.value,
                    "confidence": fact.confidence,
                    "subject_entity_id": fact.subject_entity_id,
                })

        # Search patterns
        for pattern in self._patterns.values():
            if not pattern.is_active or pattern.embedding is None:
                continue
            score = cosine_similarity(query_embedding, pattern.embedding)
            if score >= min_similarity:
                raw_results.append({
                    "item_type": "pattern",
                    "item_id": pattern.id,
                    "content": pattern.description,
                    "similarity_score": score,
                    "pattern_type": pattern.pattern_type.value,
                    "confidence": pattern.confidence,
                    "observation_count": pattern.observation_count,
                })

        # Sort and trim
        raw_results.sort(key=lambda r: r["similarity_score"], reverse=True)
        raw_results = raw_results[:top_k]

        # Enrich with context
        if include_context:
            for result in raw_results:
                if result["item_type"] == "entity":
                    entity_id = result["item_id"]
                    # Add connected entities (names + relationship types)
                    connections = await self.get_connected_entities(entity_id)
                    result["connections"] = [
                        {
                            "name": e.name,
                            "entity_type": e.entity_type.value,
                            "relationship": r.relationship_type.value,
                            "strength": r.strength,
                        }
                        for e, r in connections[:5]  # Top 5 connections
                    ]
                    # Add related facts
                    facts = await self.get_entity_facts(entity_id)
                    result["facts"] = [f.content for f in facts[:5]]

                elif result["item_type"] == "fact":
                    # Enrich fact with subject entity info
                    subject_id = result.get("subject_entity_id")
                    if subject_id:
                        subject = self._entities.get(subject_id)
                        if subject:
                            result["subject_entity"] = {
                                "name": subject.name,
                                "entity_type": subject.entity_type.value,
                            }

        return raw_results

    async def find_related_by_embedding(
        self,
        entity_id: str,
        item_types: list[str] | None = None,
        top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> list[SemanticSearchResult]:
        """Find items semantically related to a specific entity's embedding.

        Unlike search_similar_entities which only finds other entities,
        this method searches across entities, facts, and patterns using
        the given entity's embedding as the query vector. Useful for
        discovering all knowledge related to a specific entity.

        Args:
            entity_id: The entity whose embedding is used as the query.
            item_types: Types to search ("entity", "fact", "pattern").
            top_k: Maximum number of results to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of SemanticSearchResult sorted by similarity score descending.
        """
        seed = self._entities.get(entity_id)
        if not seed or seed.embedding is None:
            return []

        search_types = item_types or ["entity", "fact", "pattern"]
        results: list[SemanticSearchResult] = []

        if "entity" in search_types:
            for entity in self._entities.values():
                if entity.id == entity_id or entity.embedding is None:
                    continue
                score = max(0.0, cosine_similarity(seed.embedding, entity.embedding))
                if score >= min_similarity:
                    results.append(
                        SemanticSearchResult(
                            item_type="entity",
                            item_id=entity.id,
                            content=f"{entity.entity_type.value}: {entity.name}",
                            similarity_score=score,
                            metadata={
                                "entity_type": entity.entity_type.value,
                                "mention_count": entity.mention_count,
                            },
                        )
                    )

        if "fact" in search_types:
            for fact in self._facts.values():
                if not fact.is_active or fact.embedding is None:
                    continue
                score = max(0.0, cosine_similarity(seed.embedding, fact.embedding))
                if score >= min_similarity:
                    results.append(
                        SemanticSearchResult(
                            item_type="fact",
                            item_id=fact.id,
                            content=fact.content,
                            similarity_score=score,
                            metadata={
                                "fact_type": fact.fact_type.value,
                                "confidence": fact.confidence,
                                "subject_entity_id": fact.subject_entity_id,
                            },
                        )
                    )

        if "pattern" in search_types:
            for pattern in self._patterns.values():
                if not pattern.is_active or pattern.embedding is None:
                    continue
                score = max(0.0, cosine_similarity(seed.embedding, pattern.embedding))
                if score >= min_similarity:
                    results.append(
                        SemanticSearchResult(
                            item_type="pattern",
                            item_id=pattern.id,
                            content=pattern.description,
                            similarity_score=score,
                            metadata={
                                "pattern_type": pattern.pattern_type.value,
                                "confidence": pattern.confidence,
                            },
                        )
                    )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    # ── Graph Queries ─────────────────────────────────────────────────

    async def get_entity_context(self, entity_id: str) -> dict[str, Any]:
        """Get full context for an entity: its facts, relationships, and connected entities.

        This is the primary method for building rich context about a known entity,
        used for task surfacing, query answering, and pattern matching.
        """
        entity = self._entities.get(entity_id)
        if not entity:
            return {}

        facts = await self.get_entity_facts(entity_id)
        connections = await self.get_connected_entities(entity_id)

        return {
            "entity": entity.model_dump(exclude={"embedding"}),
            "facts": [f.model_dump(exclude={"embedding"}) for f in facts],
            "connections": [
                {
                    "entity": e.model_dump(exclude={"embedding"}),
                    "relationship": r.model_dump(),
                }
                for e, r in connections
            ],
        }

    async def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph."""
        active_facts = [f for f in self._facts.values() if f.is_active]
        active_patterns = [p for p in self._patterns.values() if p.is_active]

        entity_type_counts: dict[str, int] = {}
        for entity in self._entities.values():
            t = entity.entity_type.value
            entity_type_counts[t] = entity_type_counts.get(t, 0) + 1

        return {
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
            "total_facts": len(self._facts),
            "active_facts": len(active_facts),
            "total_patterns": len(self._patterns),
            "active_patterns": len(active_patterns),
            "entity_type_counts": entity_type_counts,
        }

    # ── Batch Operations for Pipeline Integration ─────────────────────

    async def process_extracted_entities(
        self,
        entities: list[dict[str, Any]],
        blurt_id: str,
        raw_text: str,
    ) -> list[EntityNode]:
        """Process entities extracted from a blurt through the pipeline.

        This is the main integration point called by the blurt processing
        pipeline after entity extraction. It:
        1. Adds/updates all extracted entities
        2. Creates MENTIONED_WITH relationships between co-mentioned entities
        3. Returns the processed entity nodes
        """
        processed: list[EntityNode] = []

        for entity_data in entities:
            name = entity_data.get("name", "")
            entity_type_str = entity_data.get("type", "topic")
            aliases = entity_data.get("aliases", [])
            attributes = entity_data.get("attributes", {})

            try:
                entity_type = EntityType(entity_type_str.lower())
            except ValueError:
                entity_type = EntityType.TOPIC

            node = await self.add_entity(
                name=name,
                entity_type=entity_type,
                aliases=aliases,
                attributes=attributes,
            )
            processed.append(node)

        # Create MENTIONED_WITH relationships for all co-mentioned entities
        for i, e1 in enumerate(processed):
            for e2 in processed[i + 1 :]:
                if e1.id != e2.id:
                    await self.add_or_strengthen_relationship(
                        source_entity_id=e1.id,
                        target_entity_id=e2.id,
                        relationship_type=RelationshipType.MENTIONED_WITH,
                        context=raw_text[:200],
                    )

        return processed

    async def process_extracted_facts(
        self,
        facts: list[dict[str, Any]],
        blurt_id: str,
    ) -> list[Fact]:
        """Process facts extracted from a blurt through the pipeline."""
        processed: list[Fact] = []

        for fact_data in facts:
            content = fact_data.get("content", "")
            fact_type_str = fact_data.get("type", "attribute")
            subject_name = fact_data.get("subject_entity")

            try:
                fact_type = FactType(fact_type_str.lower())
            except ValueError:
                fact_type = FactType.ATTRIBUTE

            # Resolve subject entity
            subject_id = None
            if subject_name:
                entity = await self.find_entity_by_name(subject_name)
                if entity:
                    subject_id = entity.id

            fact = await self.add_fact(
                content=content,
                fact_type=fact_type,
                subject_entity_id=subject_id,
                source_blurt_id=blurt_id,
            )
            processed.append(fact)

        return processed
