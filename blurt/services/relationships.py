"""Relationship tracking service for the knowledge graph.

Uses Gemini 2 embeddings to detect, score, and store semantic relationships
between entities. Goes beyond explicit extraction by discovering implicit
relationships through embedding proximity — entities that are semantically
close but never explicitly linked get RELATED_TO edges.

This service is the bridge between the entity extraction pipeline and the
semantic memory store. It:
1. Processes extracted relationships and stores them with strength scores.
2. Detects implicit relationships via embedding cosine similarity.
3. Infers relationship types from entity types and embedding context.
4. Manages relationship lifecycle: creation, strengthening, decay, pruning.
5. Provides graph traversal for surfacing context and connections.

Two-model strategy: Uses Flash-Lite (via extraction) for initial relationship
detection, then embeddings for semantic relationship scoring.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from blurt.clients.embeddings import EmbeddingProvider, cosine_similarity
from blurt.models.entities import (
    EntityNode,
    EntityType,
    RelationshipEdge,
    RelationshipType,
    is_valid_relationship,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Configuration ────────────────────────────────────────────────────


class RelationshipDetectionMode(str, Enum):
    """How aggressively to detect implicit relationships."""

    CONSERVATIVE = "conservative"  # Only explicit + very high similarity
    BALANCED = "balanced"          # Default: explicit + moderate similarity
    AGGRESSIVE = "aggressive"      # Detect even weak semantic connections


@dataclass
class RelationshipConfig:
    """Configuration for relationship tracking behavior.

    All thresholds are tuned for Gemini text-embedding-004 (768d).
    Lower-dimensional models (e.g., local 384d) may need adjusted thresholds.
    """

    # Embedding similarity threshold for implicit relationship detection
    implicit_similarity_threshold: float = 0.65

    # Minimum similarity to consider entities semantically related at all
    min_semantic_similarity: float = 0.3

    # Strength boost per co-mention (additive, before log scaling)
    co_mention_strength_boost: float = 1.0

    # Weight of embedding similarity in overall relationship score (0-1)
    embedding_weight: float = 0.4

    # Weight of co-mention frequency in overall relationship score (0-1)
    co_mention_weight: float = 0.4

    # Weight of recency in overall relationship score (0-1)
    recency_weight: float = 0.2

    # Maximum number of implicit relationships to create per entity batch
    max_implicit_per_batch: int = 10

    # Minimum entity mentions before it's eligible for implicit relationships
    min_mentions_for_implicit: int = 2

    # Decay half-life for relationship strength (days)
    decay_half_life_days: float = 30.0

    # Threshold below which relationships are considered dormant
    dormant_threshold: float = 0.01

    # Maximum context snippets to retain per relationship
    max_context_snippets: int = 10

    # Detection mode
    mode: RelationshipDetectionMode = RelationshipDetectionMode.BALANCED

    def __post_init__(self) -> None:
        if self.mode == RelationshipDetectionMode.CONSERVATIVE:
            self.implicit_similarity_threshold = 0.80
            self.max_implicit_per_batch = 5
            self.min_mentions_for_implicit = 3
        elif self.mode == RelationshipDetectionMode.AGGRESSIVE:
            self.implicit_similarity_threshold = 0.50
            self.max_implicit_per_batch = 20
            self.min_mentions_for_implicit = 1


# ── Relationship scoring ────────────────────────────────────────────


@dataclass
class RelationshipScore:
    """Composite score for a relationship between two entities.

    Combines embedding similarity, co-mention frequency, and recency
    into a single score for ranking and surfacing.
    """

    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    embedding_similarity: float = 0.0
    co_mention_score: float = 0.0
    recency_score: float = 0.0
    composite_score: float = 0.0
    is_explicit: bool = False  # From extraction vs inferred
    confidence: float = 0.0

    def compute_composite(self, config: RelationshipConfig) -> float:
        """Compute weighted composite score."""
        self.composite_score = (
            config.embedding_weight * self.embedding_similarity
            + config.co_mention_weight * self.co_mention_score
            + config.recency_weight * self.recency_score
        )
        return self.composite_score


@dataclass
class RelationshipDetectionResult:
    """Result from processing a batch of entities for relationships."""

    explicit_relationships: list[RelationshipEdge] = field(default_factory=list)
    implicit_relationships: list[RelationshipEdge] = field(default_factory=list)
    strengthened_relationships: list[RelationshipEdge] = field(default_factory=list)
    scores: list[RelationshipScore] = field(default_factory=list)
    entities_processed: int = 0
    latency_ms: float = 0.0

    @property
    def total_relationships(self) -> int:
        return (
            len(self.explicit_relationships)
            + len(self.implicit_relationships)
            + len(self.strengthened_relationships)
        )


# ── Relationship type inference ──────────────────────────────────────


# Context keywords that suggest specific relationship types
_RELATIONSHIP_CONTEXT_HINTS: dict[str, list[RelationshipType]] = {
    "work": [RelationshipType.WORKS_WITH, RelationshipType.COLLABORATES_ON],
    "manage": [RelationshipType.MANAGES, RelationshipType.MANAGED_BY],
    "report": [RelationshipType.REPORTS_TO],
    "team": [RelationshipType.WORKS_WITH, RelationshipType.MEMBER_OF],
    "friend": [RelationshipType.FRIEND_OF],
    "family": [RelationshipType.FAMILY_OF],
    "know": [RelationshipType.KNOWS],
    "project": [RelationshipType.COLLABORATES_ON, RelationshipType.CONTRIBUTES_TO],
    "office": [RelationshipType.LOCATED_AT, RelationshipType.BASED_IN],
    "company": [RelationshipType.EMPLOYED_BY, RelationshipType.MEMBER_OF],
    "found": [RelationshipType.FOUNDED],
    "own": [RelationshipType.OWNS],
    "block": [RelationshipType.BLOCKED_BY],
    "depend": [RelationshipType.DEPENDS_ON],
    "part": [RelationshipType.PART_OF],
    "contain": [RelationshipType.CONTAINS],
    "locat": [RelationshipType.LOCATED_AT],
    "base": [RelationshipType.BASED_IN],
}


def infer_relationship_type(
    source_type: EntityType,
    target_type: EntityType,
    context: str | None = None,
) -> RelationshipType:
    """Infer the most likely relationship type between two entity types.

    Uses the entity type pair and optional context text to determine the
    best relationship type. Falls back to RELATED_TO for unknown combos.

    Args:
        source_type: Type of the source entity.
        target_type: Type of the target entity.
        context: Optional text context where entities were co-mentioned.

    Returns:
        The inferred RelationshipType.
    """
    # If context is provided, look for keyword hints
    if context:
        context_lower = context.lower()
        for keyword, rel_types in _RELATIONSHIP_CONTEXT_HINTS.items():
            if keyword in context_lower:
                for rel_type in rel_types:
                    if is_valid_relationship(source_type, target_type, rel_type):
                        return rel_type

    # Default type inference based on entity type pairs
    pair = (source_type, target_type)

    pair_defaults: dict[tuple[EntityType, EntityType], RelationshipType] = {
        (EntityType.PERSON, EntityType.PERSON): RelationshipType.KNOWS,
        (EntityType.PERSON, EntityType.ORGANIZATION): RelationshipType.MEMBER_OF,
        (EntityType.PERSON, EntityType.PROJECT): RelationshipType.COLLABORATES_ON,
        (EntityType.PERSON, EntityType.PLACE): RelationshipType.LOCATED_AT,
        (EntityType.PERSON, EntityType.TOOL): RelationshipType.RELATED_TO,
        (EntityType.PERSON, EntityType.TOPIC): RelationshipType.RELATED_TO,
        (EntityType.ORGANIZATION, EntityType.PLACE): RelationshipType.BASED_IN,
        (EntityType.ORGANIZATION, EntityType.PROJECT): RelationshipType.OWNS,
        (EntityType.ORGANIZATION, EntityType.PERSON): RelationshipType.CONTAINS,
        (EntityType.ORGANIZATION, EntityType.ORGANIZATION): RelationshipType.PART_OF,
        (EntityType.PROJECT, EntityType.PROJECT): RelationshipType.DEPENDS_ON,
        (EntityType.PROJECT, EntityType.PLACE): RelationshipType.LOCATED_AT,
        (EntityType.PROJECT, EntityType.PERSON): RelationshipType.MANAGED_BY,
        (EntityType.PLACE, EntityType.PLACE): RelationshipType.PART_OF,
    }

    return pair_defaults.get(pair, RelationshipType.RELATED_TO)


# ── Relationship tracking service ────────────────────────────────────


class RelationshipTrackingService:
    """Service for detecting, scoring, and managing entity relationships.

    Uses Gemini 2 embeddings to discover semantic connections between
    entities in the knowledge graph. Handles both:
    - **Explicit relationships**: Directly extracted from speech by the
      entity extraction pipeline (e.g., "Sarah manages the Q2 project").
    - **Implicit relationships**: Detected through embedding similarity
      between entities that were never explicitly linked but are
      semantically close (e.g., two people mentioned in similar contexts).

    The service scores relationships using a composite of:
    - Embedding cosine similarity (semantic closeness)
    - Co-mention frequency (how often entities appear together)
    - Recency (how recently the relationship was observed)

    Usage::

        service = RelationshipTrackingService(
            embedding_provider=gemini_embeddings,
            config=RelationshipConfig(),
        )

        # Process extracted entities and their relationships
        result = await service.process_extraction(
            entities=extracted_entities,
            explicit_relationships=extracted_rels,
            context="Sarah and I discussed the Q2 project at the office",
            user_id="user-123",
        )

        # Find semantically similar entities
        related = await service.find_related_entities(
            entity=sarah_node,
            all_entities=all_nodes,
            top_k=5,
        )
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: RelationshipConfig | None = None,
    ) -> None:
        self._embeddings = embedding_provider
        self._config = config or RelationshipConfig()

        # In-memory relationship store (same pattern as SemanticMemoryStore)
        self._relationships: dict[str, RelationshipEdge] = {}
        self._by_entity: dict[str, list[str]] = {}  # entity_id -> [rel_ids]
        self._by_pair: dict[tuple[str, str, str], str] = {}  # (src, tgt, type) -> rel_id

    @property
    def config(self) -> RelationshipConfig:
        return self._config

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)

    # ── Core Operations ──────────────────────────────────────────────

    async def process_extraction(
        self,
        entities: list[EntityNode],
        explicit_relationships: list[dict[str, Any]] | None = None,
        context: str | None = None,
        user_id: str = "",
    ) -> RelationshipDetectionResult:
        """Process extracted entities and detect all relationships.

        This is the main entry point called after entity extraction.
        It handles both explicit relationships from the extraction pipeline
        and implicit relationships detected via embedding similarity.

        Args:
            entities: Entity nodes extracted from the blurt.
            explicit_relationships: Relationships explicitly extracted
                (source_name, target_name, type, confidence).
            context: The original blurt text for context-aware inference.
            user_id: The user's ID for relationship ownership.

        Returns:
            RelationshipDetectionResult with all detected relationships.
        """
        import time

        start = time.monotonic()
        result = RelationshipDetectionResult(entities_processed=len(entities))

        if len(entities) < 2 and not explicit_relationships:
            result.latency_ms = (time.monotonic() - start) * 1000
            return result

        # Build name-to-entity lookup
        entity_by_name: dict[str, EntityNode] = {}
        for e in entities:
            entity_by_name[e.normalized_name] = e
            for alias in e.aliases:
                entity_by_name[alias.lower().strip()] = e

        # 1. Process explicit relationships from extraction
        if explicit_relationships:
            for rel_data in explicit_relationships:
                edge = await self._process_explicit_relationship(
                    rel_data, entity_by_name, context, user_id
                )
                if edge:
                    if rel_data.get("_strengthened"):
                        result.strengthened_relationships.append(edge)
                    else:
                        result.explicit_relationships.append(edge)

        # 2. Detect implicit relationships via embedding similarity
        implicit = await self._detect_implicit_relationships(
            entities, context, user_id
        )
        result.implicit_relationships.extend(implicit)

        # 3. Score all relationships
        all_edges = (
            result.explicit_relationships
            + result.implicit_relationships
            + result.strengthened_relationships
        )
        for edge in all_edges:
            score = await self._score_relationship(edge, entities)
            result.scores.append(score)

        result.latency_ms = (time.monotonic() - start) * 1000
        return result

    async def _process_explicit_relationship(
        self,
        rel_data: dict[str, Any],
        entity_by_name: dict[str, EntityNode],
        context: str | None,
        user_id: str,
    ) -> RelationshipEdge | None:
        """Process a single explicit relationship from extraction."""
        source_name = str(rel_data.get("source_name", rel_data.get("source", ""))).lower().strip()
        target_name = str(rel_data.get("target_name", rel_data.get("target", ""))).lower().strip()

        # Skip self-referential or 'speaker' relationships for now
        if source_name == "speaker" or not source_name or not target_name:
            return None

        source = entity_by_name.get(source_name)
        target = entity_by_name.get(target_name)

        if not source or not target or source.id == target.id:
            return None

        # Resolve relationship type
        rel_type_str = str(rel_data.get("relationship_type", rel_data.get("type", "related_to")))
        try:
            rel_type = RelationshipType(rel_type_str.lower().strip())
        except ValueError:
            rel_type = RelationshipType.RELATED_TO

        # Validate the relationship is valid for these entity types
        if not is_valid_relationship(source.entity_type, target.entity_type, rel_type):
            rel_type = infer_relationship_type(
                source.entity_type, target.entity_type, context
            )

        return await self._store_relationship(
            source_id=source.id,
            target_id=target.id,
            rel_type=rel_type,
            context=context,
            user_id=user_id,
            is_explicit=True,
            confidence=float(rel_data.get("confidence", 0.8)),
        )

    async def _detect_implicit_relationships(
        self,
        entities: list[EntityNode],
        context: str | None,
        user_id: str,
    ) -> list[RelationshipEdge]:
        """Detect implicit relationships through embedding similarity.

        Compares entity embeddings pairwise. If similarity exceeds the
        threshold and no explicit relationship exists, creates a RELATED_TO
        edge with strength proportional to similarity.
        """
        implicit: list[RelationshipEdge] = []

        if len(entities) < 2:
            return implicit

        # Filter entities eligible for implicit detection
        eligible = [
            e for e in entities
            if e.embedding is not None
            and e.mention_count >= self._config.min_mentions_for_implicit
        ]

        if len(eligible) < 2:
            return implicit

        # Compute pairwise similarities
        scored_pairs: list[tuple[float, EntityNode, EntityNode]] = []

        for i, e1 in enumerate(eligible):
            for e2 in eligible[i + 1:]:
                if e1.id == e2.id:
                    continue

                assert e1.embedding is not None and e2.embedding is not None
                similarity = cosine_similarity(e1.embedding, e2.embedding)

                if similarity >= self._config.implicit_similarity_threshold:
                    # Check if an explicit relationship already exists
                    existing = self._find_any_relationship(e1.id, e2.id)
                    if existing is None:
                        scored_pairs.append((similarity, e1, e2))

        # Sort by similarity descending, take top N
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        scored_pairs = scored_pairs[: self._config.max_implicit_per_batch]

        for similarity, e1, e2 in scored_pairs:
            # Infer relationship type from entity types and context
            rel_type = infer_relationship_type(
                e1.entity_type, e2.entity_type, context
            )

            edge = await self._store_relationship(
                source_id=e1.id,
                target_id=e2.id,
                rel_type=rel_type,
                context=context,
                user_id=user_id,
                is_explicit=False,
                confidence=similarity,
            )
            if edge:
                implicit.append(edge)

        return implicit

    async def _store_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType,
        context: str | None,
        user_id: str,
        is_explicit: bool = True,
        confidence: float = 1.0,
    ) -> RelationshipEdge:
        """Store or strengthen a relationship edge.

        If the relationship already exists, strengthens it. Otherwise
        creates a new edge. Returns the edge.
        """
        pair_key = (source_id, target_id, rel_type.value)
        reverse_key = (target_id, source_id, rel_type.value)

        existing_id = self._by_pair.get(pair_key) or self._by_pair.get(reverse_key)

        if existing_id and existing_id in self._relationships:
            edge = self._relationships[existing_id]
            edge.co_mention_count += 1
            # Log-based strength growth: diminishing returns
            edge.strength = min(
                100.0,
                math.log2(edge.co_mention_count + 1) * 10 * confidence,
            )
            edge.last_seen = _utcnow()
            edge.updated_at = _utcnow()
            if context:
                edge.context_snippets.append(context[:200])
                if len(edge.context_snippets) > self._config.max_context_snippets:
                    edge.context_snippets = edge.context_snippets[
                        -self._config.max_context_snippets:
                    ]
            return edge

        # Create new edge
        initial_strength = confidence if is_explicit else confidence * 0.5
        edge = RelationshipEdge(
            user_id=user_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relationship_type=rel_type,
            strength=min(100.0, initial_strength * 10),
            co_mention_count=1,
            context_snippets=[context[:200]] if context else [],
        )

        self._relationships[edge.id] = edge
        self._by_entity.setdefault(source_id, []).append(edge.id)
        self._by_entity.setdefault(target_id, []).append(edge.id)
        self._by_pair[pair_key] = edge.id

        return edge

    def _find_any_relationship(
        self, entity_id_1: str, entity_id_2: str
    ) -> RelationshipEdge | None:
        """Find any existing relationship between two entities, regardless of type."""
        rel_ids = self._by_entity.get(entity_id_1, [])
        for rid in rel_ids:
            rel = self._relationships.get(rid)
            if rel and (
                (rel.source_entity_id == entity_id_1 and rel.target_entity_id == entity_id_2)
                or (rel.source_entity_id == entity_id_2 and rel.target_entity_id == entity_id_1)
            ):
                return rel
        return None

    # ── Scoring ──────────────────────────────────────────────────────

    async def _score_relationship(
        self,
        edge: RelationshipEdge,
        entities: list[EntityNode],
    ) -> RelationshipScore:
        """Compute a composite score for a relationship.

        Combines embedding similarity, co-mention frequency, and recency.
        """
        # Find the entity objects
        source = next((e for e in entities if e.id == edge.source_entity_id), None)
        target = next((e for e in entities if e.id == edge.target_entity_id), None)

        score = RelationshipScore(
            source_entity_id=edge.source_entity_id,
            target_entity_id=edge.target_entity_id,
            relationship_type=edge.relationship_type,
        )

        # Embedding similarity
        if source and target and source.embedding and target.embedding:
            score.embedding_similarity = max(
                0.0, cosine_similarity(source.embedding, target.embedding)
            )

        # Co-mention score (log-scaled, normalized to 0-1)
        score.co_mention_score = min(1.0, math.log2(edge.co_mention_count + 1) / 5)

        # Recency score (exponential decay from last_seen)
        days_since = (_utcnow() - edge.last_seen).total_seconds() / 86400
        score.recency_score = math.exp(-days_since / self._config.decay_half_life_days)

        # Overall confidence
        score.confidence = edge.strength / 100.0

        score.compute_composite(self._config)
        return score

    async def score_entity_pair(
        self,
        entity_a: EntityNode,
        entity_b: EntityNode,
    ) -> RelationshipScore:
        """Score the relationship strength between two specific entities.

        Useful for querying whether two entities are related even if
        no explicit relationship exists yet.
        """
        score = RelationshipScore(
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type=RelationshipType.RELATED_TO,
        )

        # Embedding similarity
        if entity_a.embedding and entity_b.embedding:
            score.embedding_similarity = max(
                0.0, cosine_similarity(entity_a.embedding, entity_b.embedding)
            )

        # Check existing relationship
        existing = self._find_any_relationship(entity_a.id, entity_b.id)
        if existing:
            score.relationship_type = existing.relationship_type
            score.co_mention_score = min(1.0, math.log2(existing.co_mention_count + 1) / 5)
            days_since = (_utcnow() - existing.last_seen).total_seconds() / 86400
            score.recency_score = math.exp(-days_since / self._config.decay_half_life_days)
            score.is_explicit = True
            score.confidence = existing.strength / 100.0

        score.compute_composite(self._config)
        return score

    # ── Queries ──────────────────────────────────────────────────────

    async def find_related_entities(
        self,
        entity: EntityNode,
        all_entities: list[EntityNode],
        top_k: int = 10,
        min_similarity: float | None = None,
    ) -> list[tuple[EntityNode, float]]:
        """Find entities most semantically related to the given entity.

        Uses embedding cosine similarity to rank all entities by how
        closely they relate to the target entity.

        Args:
            entity: The reference entity.
            all_entities: Pool of entities to search.
            top_k: Maximum results to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (entity, similarity_score) tuples, sorted by score desc.
        """
        if entity.embedding is None:
            return []

        threshold = min_similarity or self._config.min_semantic_similarity
        scored: list[tuple[EntityNode, float]] = []

        for other in all_entities:
            if other.id == entity.id or other.embedding is None:
                continue

            similarity = cosine_similarity(entity.embedding, other.embedding)
            if similarity >= threshold:
                scored.append((other, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def get_entity_relationships(
        self, entity_id: str
    ) -> list[RelationshipEdge]:
        """Get all relationships for an entity."""
        rel_ids = self._by_entity.get(entity_id, [])
        return [
            self._relationships[rid]
            for rid in rel_ids
            if rid in self._relationships
        ]

    async def get_relationship_between(
        self,
        entity_a_id: str,
        entity_b_id: str,
        rel_type: RelationshipType | None = None,
    ) -> RelationshipEdge | None:
        """Get a specific relationship between two entities.

        If rel_type is None, returns the strongest relationship of any type.
        """
        rel_ids = self._by_entity.get(entity_a_id, [])
        matches: list[RelationshipEdge] = []

        for rid in rel_ids:
            rel = self._relationships.get(rid)
            if not rel:
                continue

            is_match = (
                (rel.source_entity_id == entity_a_id and rel.target_entity_id == entity_b_id)
                or (rel.source_entity_id == entity_b_id and rel.target_entity_id == entity_a_id)
            )

            if is_match:
                if rel_type is None or rel.relationship_type == rel_type:
                    matches.append(rel)

        if not matches:
            return None

        # Return strongest
        return max(matches, key=lambda r: r.strength)

    async def get_strongest_connections(
        self,
        entity_id: str,
        top_k: int = 10,
        min_strength: float = 0.0,
    ) -> list[tuple[str, RelationshipEdge]]:
        """Get the strongest connections for an entity.

        Returns (connected_entity_id, relationship_edge) pairs sorted
        by relationship strength descending.
        """
        rels = await self.get_entity_relationships(entity_id)
        connections: list[tuple[str, RelationshipEdge]] = []

        for rel in rels:
            if rel.strength < min_strength:
                continue
            other_id = (
                rel.target_entity_id
                if rel.source_entity_id == entity_id
                else rel.source_entity_id
            )
            connections.append((other_id, rel))

        connections.sort(key=lambda x: x[1].strength, reverse=True)
        return connections[:top_k]

    async def find_entity_clusters(
        self,
        entities: list[EntityNode],
        similarity_threshold: float | None = None,
    ) -> list[list[EntityNode]]:
        """Find clusters of semantically related entities.

        Uses single-linkage clustering based on embedding similarity.
        Entities within a cluster are all transitively connected above
        the similarity threshold.

        Args:
            entities: Entities to cluster.
            similarity_threshold: Minimum similarity for cluster membership.

        Returns:
            List of entity clusters (each cluster is a list of EntityNode).
        """
        threshold = similarity_threshold or self._config.implicit_similarity_threshold

        # Union-Find for clustering
        parent: dict[str, str] = {e.id: e.id for e in entities}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Compare all pairs
        eligible = [e for e in entities if e.embedding is not None]
        for i, e1 in enumerate(eligible):
            for e2 in eligible[i + 1:]:
                assert e1.embedding is not None and e2.embedding is not None
                sim = cosine_similarity(e1.embedding, e2.embedding)
                if sim >= threshold:
                    union(e1.id, e2.id)

        # Group by root
        clusters: dict[str, list[EntityNode]] = {}
        entity_map = {e.id: e for e in entities}
        for e in entities:
            root = find(e.id)
            clusters.setdefault(root, []).append(entity_map[e.id])

        # Return clusters with 2+ members, sorted by size desc
        result = [c for c in clusters.values() if len(c) >= 2]
        result.sort(key=len, reverse=True)
        return result

    # ── Lifecycle ────────────────────────────────────────────────────

    async def decay_relationships(
        self, as_of: datetime | None = None
    ) -> int:
        """Apply time-based decay to all tracked relationships.

        Relationships weaken when entities aren't co-mentioned.
        Returns the count of relationships that decayed below dormant threshold.
        """
        now = as_of or _utcnow()
        dormant_count = 0

        for rel in list(self._relationships.values()):
            days_since = (now - rel.last_seen).total_seconds() / 86400
            if days_since <= 0:
                continue

            decay_factor = math.pow(
                0.5, days_since / self._config.decay_half_life_days
            )
            rel.strength = max(
                self._config.dormant_threshold, rel.strength * decay_factor
            )
            rel.updated_at = now

            if rel.strength <= self._config.dormant_threshold:
                dormant_count += 1

        return dormant_count

    async def prune_dormant_relationships(
        self, min_strength: float | None = None,
    ) -> int:
        """Remove relationships that have decayed below threshold.

        Returns the number of relationships pruned.
        """
        threshold = min_strength or self._config.dormant_threshold
        to_remove: list[str] = []

        for rid, rel in self._relationships.items():
            if rel.strength <= threshold:
                to_remove.append(rid)

        for rid in to_remove:
            rel = self._relationships.pop(rid, None)
            if rel:
                # Clean up indexes
                src_rels = self._by_entity.get(rel.source_entity_id, [])
                if rid in src_rels:
                    src_rels.remove(rid)
                tgt_rels = self._by_entity.get(rel.target_entity_id, [])
                if rid in tgt_rels:
                    tgt_rels.remove(rid)

                pair_key = (rel.source_entity_id, rel.target_entity_id, rel.relationship_type.value)
                self._by_pair.pop(pair_key, None)

        return len(to_remove)

    # ── Statistics ───────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """Get relationship tracking statistics."""
        active = [r for r in self._relationships.values() if r.strength > self._config.dormant_threshold]
        type_counts: dict[str, int] = {}
        total_strength = 0.0

        for rel in self._relationships.values():
            t = rel.relationship_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
            total_strength += rel.strength

        avg_strength = total_strength / len(self._relationships) if self._relationships else 0.0

        return {
            "total_relationships": len(self._relationships),
            "active_relationships": len(active),
            "dormant_relationships": len(self._relationships) - len(active),
            "relationship_type_counts": type_counts,
            "average_strength": round(avg_strength, 2),
            "unique_entities": len(self._by_entity),
        }
