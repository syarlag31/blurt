"""Entity graph storage layer — abstract interface and enhanced graph operations.

Defines the ``EntityGraphStore`` protocol that both in-memory and persistent
backends implement. Adds graph traversal, entity merging, subgraph extraction,
and batch embedding operations on top of the base semantic memory store.

Every node and edge carries vector embeddings for semantic similarity search.
The graph compounds with every interaction: entities link, relationships
strengthen, patterns emerge.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

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

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Storage Protocol ────────────────────────────────────────────────────


@runtime_checkable
class EntityGraphStore(Protocol):
    """Protocol defining the entity graph storage interface.

    Both the in-memory ``SemanticMemoryStore`` and the SQLite-backed
    ``LocalKnowledgeGraphStore`` implement this interface, ensuring
    feature parity between cloud and local-only modes.
    """

    user_id: str

    async def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> EntityNode: ...

    async def get_entity(self, entity_id: str) -> EntityNode | None: ...

    async def find_entity_by_name(self, name: str) -> EntityNode | None: ...

    async def get_all_entities(
        self, entity_type: EntityType | None = None
    ) -> list[EntityNode]: ...

    async def update_entity_embedding(self, entity_id: str) -> EntityNode | None: ...

    async def add_or_strengthen_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: RelationshipType,
        context: str | None = None,
    ) -> RelationshipEdge: ...

    async def get_entity_relationships(
        self, entity_id: str
    ) -> list[RelationshipEdge]: ...

    async def get_connected_entities(
        self, entity_id: str
    ) -> list[tuple[EntityNode, RelationshipEdge]]: ...

    async def decay_relationships(self, as_of: datetime | None = None) -> int: ...

    async def add_fact(
        self,
        content: str,
        fact_type: FactType,
        subject_entity_id: str | None = None,
        source_blurt_id: str | None = None,
        confidence: float = 1.0,
    ) -> Fact: ...

    async def get_entity_facts(
        self, entity_id: str, active_only: bool = True
    ) -> list[Fact]: ...

    async def get_all_facts(
        self, fact_type: FactType | None = None, active_only: bool = True
    ) -> list[Fact]: ...

    async def supersede_fact(
        self,
        old_fact_id: str,
        new_content: str,
        source_blurt_id: str | None = None,
    ) -> Fact | None: ...

    async def add_pattern(
        self,
        pattern_type: PatternType,
        description: str,
        parameters: dict[str, Any] | None = None,
        confidence: float = 0.5,
        observation_count: int = 0,
        supporting_evidence: list[str] | None = None,
    ) -> LearnedPattern: ...

    async def confirm_pattern(
        self, pattern_id: str, observation_id: str | None = None
    ) -> LearnedPattern | None: ...

    async def get_active_patterns(
        self, pattern_type: PatternType | None = None
    ) -> list[LearnedPattern]: ...

    async def search(
        self,
        query: str,
        top_k: int = 10,
        item_types: list[str] | None = None,
        min_similarity: float = 0.3,
    ) -> list[SemanticSearchResult]: ...

    async def get_entity_context(self, entity_id: str) -> dict[str, Any]: ...

    async def get_graph_stats(self) -> dict[str, Any]: ...

    async def process_extracted_entities(
        self,
        entities: list[dict[str, Any]],
        blurt_id: str,
        raw_text: str,
    ) -> list[EntityNode]: ...

    async def process_extracted_facts(
        self,
        facts: list[dict[str, Any]],
        blurt_id: str,
    ) -> list[Fact]: ...


# ── Traversal Result Models ─────────────────────────────────────────────


class TraversalNode:
    """A node encountered during graph traversal with distance metadata."""

    __slots__ = ("entity", "depth", "path", "relationship")

    def __init__(
        self,
        entity: EntityNode,
        depth: int,
        path: list[str],
        relationship: RelationshipEdge | None = None,
    ):
        self.entity = entity
        self.depth = depth
        self.path = path  # entity IDs from start to this node
        self.relationship = relationship  # edge that led here

    def __repr__(self) -> str:
        return (
            f"TraversalNode(entity={self.entity.name!r}, "
            f"depth={self.depth}, path_len={len(self.path)})"
        )


class Subgraph:
    """A subgraph extracted from the knowledge graph.

    Contains a subset of entities and their connecting relationships,
    useful for context assembly and visualization.
    """

    def __init__(
        self,
        entities: list[EntityNode],
        relationships: list[RelationshipEdge],
        center_entity_id: str | None = None,
    ):
        self.entities = entities
        self.relationships = relationships
        self.center_entity_id = center_entity_id
        self._entity_index: dict[str, EntityNode] = {e.id: e for e in entities}

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    def get_entity(self, entity_id: str) -> EntityNode | None:
        return self._entity_index.get(entity_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize subgraph to a dict (excludes embeddings for readability)."""
        return {
            "center_entity_id": self.center_entity_id,
            "entities": [
                e.model_dump(exclude={"embedding"}) for e in self.entities
            ],
            "relationships": [r.model_dump() for r in self.relationships],
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
        }


class EntityMergeResult:
    """Result of merging two entities in the knowledge graph."""

    def __init__(
        self,
        merged_entity: EntityNode,
        removed_entity_id: str,
        relationships_transferred: int,
        facts_transferred: int,
    ):
        self.merged_entity = merged_entity
        self.removed_entity_id = removed_entity_id
        self.relationships_transferred = relationships_transferred
        self.facts_transferred = facts_transferred


# ── Enhanced Graph Operations ───────────────────────────────────────────


class EntityGraphOperations:
    """Enhanced graph operations built on top of any EntityGraphStore.

    Provides graph traversal, entity merging, subgraph extraction,
    similarity-based entity discovery, and batch operations. Works
    with both in-memory and persistent backends.
    """

    def __init__(
        self,
        store: EntityGraphStore,
        embedding_provider: EmbeddingProvider,
    ):
        self._store = store
        self._embeddings = embedding_provider

    @property
    def store(self) -> EntityGraphStore:
        """Access the underlying storage backend."""
        return self._store

    # ── Graph Traversal ──────────────────────────────────────────

    async def traverse_bfs(
        self,
        start_entity_id: str,
        max_depth: int = 3,
        max_nodes: int = 50,
        min_relationship_strength: float = 0.1,
        entity_type_filter: EntityType | None = None,
        relationship_type_filter: RelationshipType | None = None,
    ) -> list[TraversalNode]:
        """Breadth-first traversal from a starting entity.

        Explores the knowledge graph outward from a starting entity,
        following relationships that meet the strength threshold.
        Returns nodes in BFS order with depth and path metadata.

        Args:
            start_entity_id: Entity to start traversal from.
            max_depth: Maximum hops from the starting entity.
            max_nodes: Maximum number of nodes to visit.
            min_relationship_strength: Only follow edges above this strength.
            entity_type_filter: Only include entities of this type.
            relationship_type_filter: Only follow edges of this type.

        Returns:
            List of TraversalNode objects in BFS order.
        """
        start = await self._store.get_entity(start_entity_id)
        if not start:
            return []

        visited: set[str] = {start_entity_id}
        queue: deque[TraversalNode] = deque()
        results: list[TraversalNode] = []

        start_node = TraversalNode(
            entity=start, depth=0, path=[start_entity_id]
        )
        results.append(start_node)
        queue.append(start_node)

        while queue and len(results) < max_nodes:
            current = queue.popleft()
            if current.depth >= max_depth:
                continue

            connections = await self._store.get_connected_entities(current.entity.id)
            for connected_entity, relationship in connections:
                if connected_entity.id in visited:
                    continue
                if relationship.strength < min_relationship_strength:
                    continue
                if (
                    relationship_type_filter
                    and relationship.relationship_type != relationship_type_filter
                ):
                    continue
                if (
                    entity_type_filter
                    and connected_entity.entity_type != entity_type_filter
                ):
                    continue

                visited.add(connected_entity.id)
                new_path = current.path + [connected_entity.id]
                node = TraversalNode(
                    entity=connected_entity,
                    depth=current.depth + 1,
                    path=new_path,
                    relationship=relationship,
                )
                results.append(node)
                queue.append(node)

                if len(results) >= max_nodes:
                    break

        return results

    async def find_path(
        self,
        source_entity_id: str,
        target_entity_id: str,
        max_depth: int = 5,
    ) -> list[TraversalNode] | None:
        """Find the shortest path between two entities using BFS.

        Returns the path as a list of TraversalNode objects from source
        to target, or None if no path exists within max_depth.
        """
        if source_entity_id == target_entity_id:
            entity = await self._store.get_entity(source_entity_id)
            if entity:
                return [TraversalNode(entity=entity, depth=0, path=[entity.id])]
            return None

        start = await self._store.get_entity(source_entity_id)
        if not start:
            return None

        visited: set[str] = {source_entity_id}
        # Store parent mapping for path reconstruction
        parent_map: dict[str, TraversalNode] = {}
        queue: deque[TraversalNode] = deque()

        start_node = TraversalNode(
            entity=start, depth=0, path=[source_entity_id]
        )
        queue.append(start_node)

        while queue:
            current = queue.popleft()
            if current.depth >= max_depth:
                continue

            connections = await self._store.get_connected_entities(current.entity.id)
            for connected_entity, relationship in connections:
                if connected_entity.id in visited:
                    continue

                visited.add(connected_entity.id)
                new_path = current.path + [connected_entity.id]
                node = TraversalNode(
                    entity=connected_entity,
                    depth=current.depth + 1,
                    path=new_path,
                    relationship=relationship,
                )

                if connected_entity.id == target_entity_id:
                    # Reconstruct path
                    return self._reconstruct_path(start_node, node, parent_map)

                parent_map[connected_entity.id] = node
                queue.append(node)

        return None

    def _reconstruct_path(
        self,
        start: TraversalNode,
        end: TraversalNode,
        parent_map: dict[str, TraversalNode],
    ) -> list[TraversalNode]:
        """Reconstruct the path from BFS parent map."""
        # The path is already stored in end.path
        path_nodes: list[TraversalNode] = [start]

        for entity_id in end.path[1:]:
            if entity_id == end.entity.id:
                path_nodes.append(end)
            elif entity_id in parent_map:
                path_nodes.append(parent_map[entity_id])

        return path_nodes

    # ── Subgraph Extraction ──────────────────────────────────────

    async def extract_neighborhood(
        self,
        entity_id: str,
        depth: int = 2,
        min_strength: float = 0.1,
    ) -> Subgraph:
        """Extract a subgraph centered on an entity.

        Returns all entities and relationships within `depth` hops,
        filtered by minimum relationship strength. Useful for building
        context windows and visualization.
        """
        traversal = await self.traverse_bfs(
            start_entity_id=entity_id,
            max_depth=depth,
            min_relationship_strength=min_strength,
        )

        entities = [node.entity for node in traversal]
        entity_ids = {e.id for e in entities}

        # Collect all relationships between entities in the subgraph
        relationships: list[RelationshipEdge] = []
        seen_rel_ids: set[str] = set()

        for node in traversal:
            rels = await self._store.get_entity_relationships(node.entity.id)
            for rel in rels:
                if rel.id in seen_rel_ids:
                    continue
                if (
                    rel.source_entity_id in entity_ids
                    and rel.target_entity_id in entity_ids
                ):
                    relationships.append(rel)
                    seen_rel_ids.add(rel.id)

        return Subgraph(
            entities=entities,
            relationships=relationships,
            center_entity_id=entity_id,
        )

    # ── Entity Merging ───────────────────────────────────────────

    async def merge_entities(
        self,
        keep_entity_id: str,
        remove_entity_id: str,
    ) -> EntityMergeResult | None:
        """Merge two entities, keeping one and transferring all data from the other.

        This handles the case where two entities are discovered to be the same
        (e.g., "Sarah Chen" and "Sarah"). The kept entity absorbs all aliases,
        attributes, relationships, and facts from the removed entity.

        Args:
            keep_entity_id: ID of the entity to keep.
            remove_entity_id: ID of the entity to absorb and remove.

        Returns:
            EntityMergeResult or None if either entity doesn't exist.
        """
        keep = await self._store.get_entity(keep_entity_id)
        remove = await self._store.get_entity(remove_entity_id)
        if not keep or not remove:
            return None

        # Transfer aliases
        for alias in remove.aliases:
            if alias not in keep.aliases:
                keep.aliases.append(alias)
        # Add removed entity's name and normalized name as aliases
        if remove.normalized_name not in keep.aliases:
            keep.aliases.append(remove.normalized_name)

        # Merge attributes (keep's values take precedence)
        merged_attrs = {**remove.attributes, **keep.attributes}
        keep.attributes = merged_attrs

        # Accumulate mention counts
        keep.mention_count += remove.mention_count

        # Take the earliest first_seen
        if remove.first_seen < keep.first_seen:
            keep.first_seen = remove.first_seen

        keep.last_seen = max(keep.last_seen, remove.last_seen)
        keep.updated_at = _utcnow()

        # Transfer relationships
        remove_rels = await self._store.get_entity_relationships(remove_entity_id)
        rels_transferred = 0
        for rel in remove_rels:
            # Determine the other entity in the relationship
            if rel.source_entity_id == remove_entity_id:
                other_id = rel.target_entity_id
                if other_id == keep_entity_id:
                    continue  # Skip self-referencing after merge
                await self._store.add_or_strengthen_relationship(
                    keep_entity_id, other_id, rel.relationship_type,
                    context=f"Merged from {remove.name}",
                )
            else:
                other_id = rel.source_entity_id
                if other_id == keep_entity_id:
                    continue
                await self._store.add_or_strengthen_relationship(
                    other_id, keep_entity_id, rel.relationship_type,
                    context=f"Merged from {remove.name}",
                )
            rels_transferred += 1

        # Transfer facts
        remove_facts = await self._store.get_entity_facts(remove_entity_id, active_only=False)
        facts_transferred = 0
        for fact in remove_facts:
            fact.subject_entity_id = keep_entity_id
            fact.updated_at = _utcnow()
            facts_transferred += 1

        # Regenerate embedding for merged entity
        await self._store.update_entity_embedding(keep_entity_id)

        logger.info(
            "Merged entity '%s' into '%s': %d relationships, %d facts transferred",
            remove.name, keep.name, rels_transferred, facts_transferred,
        )

        return EntityMergeResult(
            merged_entity=keep,
            removed_entity_id=remove_entity_id,
            relationships_transferred=rels_transferred,
            facts_transferred=facts_transferred,
        )

    # ── Semantic Entity Discovery ────────────────────────────────

    async def find_similar_entities(
        self,
        query: str,
        top_k: int = 5,
        entity_type: EntityType | None = None,
        min_similarity: float = 0.5,
    ) -> list[tuple[EntityNode, float]]:
        """Find entities semantically similar to a query string.

        Uses vector embeddings to find entities whose names, descriptions,
        or accumulated context are semantically close to the query.

        Returns:
            List of (entity, similarity_score) tuples sorted by score descending.
        """
        query_embedding = await self._embeddings.embed(query)

        all_entities = await self._store.get_all_entities(entity_type)
        scored: list[tuple[EntityNode, float]] = []

        for entity in all_entities:
            if entity.embedding is None:
                continue
            score = cosine_similarity(query_embedding, entity.embedding)
            if score >= min_similarity:
                scored.append((entity, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def find_potential_duplicates(
        self,
        similarity_threshold: float = 0.85,
    ) -> list[tuple[EntityNode, EntityNode, float]]:
        """Find entity pairs that may be duplicates based on embedding similarity.

        Compares all entity embeddings pairwise within the same entity type
        to detect potential duplicates (e.g., "Sarah" and "Sarah Chen").

        Returns:
            List of (entity_a, entity_b, similarity) tuples above threshold.
        """
        all_entities = await self._store.get_all_entities()
        # Group by type for more efficient comparison
        by_type: dict[EntityType, list[EntityNode]] = {}
        for e in all_entities:
            by_type.setdefault(e.entity_type, []).append(e)

        duplicates: list[tuple[EntityNode, EntityNode, float]] = []

        for entity_type, entities in by_type.items():
            for i, e1 in enumerate(entities):
                if e1.embedding is None:
                    continue
                for e2 in entities[i + 1:]:
                    if e2.embedding is None:
                        continue
                    score = cosine_similarity(e1.embedding, e2.embedding)
                    if score >= similarity_threshold:
                        duplicates.append((e1, e2, score))

        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates

    # ── Batch Embedding Operations ───────────────────────────────

    async def reindex_all_embeddings(self) -> int:
        """Regenerate embeddings for all entities.

        Useful after changing the embedding model or adding new context
        (facts, relationships) that should be reflected in entity vectors.

        Returns:
            Number of entities re-embedded.
        """
        all_entities = await self._store.get_all_entities()
        count = 0

        for entity in all_entities:
            result = await self._store.update_entity_embedding(entity.id)
            if result is not None:
                count += 1

        logger.info("Re-indexed embeddings for %d entities", count)
        return count

    # ── Rich Context Assembly ────────────────────────────────────

    async def build_entity_profile(
        self,
        entity_id: str,
        include_connected: bool = True,
        max_connections: int = 10,
    ) -> dict[str, Any]:
        """Build a rich profile for an entity including all graph context.

        Assembles a complete view of an entity: its attributes, facts,
        relationships, connected entities, and any patterns that mention it.
        This is used for context injection during AI reasoning.

        Returns:
            Dict with entity profile, or empty dict if entity not found.
        """
        entity = await self._store.get_entity(entity_id)
        if not entity:
            return {}

        # Core context
        context = await self._store.get_entity_context(entity_id)

        # Add related patterns
        all_patterns = await self._store.get_active_patterns()
        related_patterns = []
        if entity.embedding:
            for pattern in all_patterns:
                if pattern.embedding:
                    score = cosine_similarity(entity.embedding, pattern.embedding)
                    if score > 0.4:
                        related_patterns.append({
                            "pattern": pattern.model_dump(exclude={"embedding"}),
                            "relevance_score": score,
                        })
            related_patterns.sort(key=lambda x: float(x["relevance_score"]), reverse=True)

        context["patterns"] = related_patterns[:5]

        # Add semantic neighbors (similar but not directly connected entities)
        if include_connected and entity.embedding:
            similar = await self.find_similar_entities(
                entity.name,
                top_k=max_connections,
                min_similarity=0.3,
            )
            # Exclude already-connected entities and self
            connected_ids = {
                c["entity"]["id"] for c in context.get("connections", [])
            }
            connected_ids.add(entity_id)
            semantic_neighbors = [
                {
                    "entity": e.model_dump(exclude={"embedding"}),
                    "similarity": score,
                }
                for e, score in similar
                if e.id not in connected_ids
            ]
            context["semantic_neighbors"] = semantic_neighbors[:5]

        return context

    # ── Graph Export / Import ────────────────────────────────────

    async def export_graph(
        self,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        """Export the entire knowledge graph as a serializable dict.

        Useful for backup, migration between storage backends, and debugging.

        Args:
            include_embeddings: Whether to include embedding vectors (large).

        Returns:
            Dict with all entities, relationships, facts, and patterns.
        """
        exclude_fields = set() if include_embeddings else {"embedding"}

        entities = await self._store.get_all_entities()
        all_facts = await self._store.get_all_facts(active_only=False)
        all_patterns = await self._store.get_active_patterns()
        stats = await self._store.get_graph_stats()

        # Collect all relationships
        all_rels: dict[str, RelationshipEdge] = {}
        for entity in entities:
            rels = await self._store.get_entity_relationships(entity.id)
            for rel in rels:
                all_rels[rel.id] = rel

        return {
            "version": "1.0",
            "user_id": self._store.user_id,
            "exported_at": _utcnow().isoformat(),
            "stats": stats,
            "entities": [
                e.model_dump(exclude=exclude_fields) for e in entities
            ],
            "relationships": [r.model_dump() for r in all_rels.values()],
            "facts": [
                f.model_dump(exclude=exclude_fields) for f in all_facts
            ],
            "patterns": [
                p.model_dump(exclude=exclude_fields) for p in all_patterns
            ],
        }

    async def import_entities(
        self,
        data: dict[str, Any],
        regenerate_embeddings: bool = True,
    ) -> dict[str, int]:
        """Import entities and relationships from exported graph data.

        Args:
            data: Exported graph dict from export_graph().
            regenerate_embeddings: Whether to regenerate embeddings on import.

        Returns:
            Dict with counts of imported items.
        """
        counts = {
            "entities": 0,
            "relationships": 0,
            "facts": 0,
        }

        # Import entities
        id_mapping: dict[str, str] = {}  # old_id -> new_id
        for entity_data in data.get("entities", []):
            entity = await self._store.add_entity(
                name=entity_data["name"],
                entity_type=EntityType(entity_data["entity_type"]),
                aliases=entity_data.get("aliases", []),
                attributes=entity_data.get("attributes", {}),
            )
            id_mapping[entity_data["id"]] = entity.id
            counts["entities"] += 1

        # Import relationships
        for rel_data in data.get("relationships", []):
            source_id = id_mapping.get(rel_data["source_entity_id"])
            target_id = id_mapping.get(rel_data["target_entity_id"])
            if source_id and target_id:
                await self._store.add_or_strengthen_relationship(
                    source_id,
                    target_id,
                    RelationshipType(rel_data["relationship_type"]),
                    context=rel_data.get("context_snippets", [""])[0] if rel_data.get("context_snippets") else None,
                )
                counts["relationships"] += 1

        # Import facts
        for fact_data in data.get("facts", []):
            subject_id = None
            if fact_data.get("subject_entity_id"):
                subject_id = id_mapping.get(fact_data["subject_entity_id"])
            await self._store.add_fact(
                content=fact_data["content"],
                fact_type=FactType(fact_data["fact_type"]),
                subject_entity_id=subject_id,
            )
            counts["facts"] += 1

        if regenerate_embeddings:
            await self.reindex_all_embeddings()

        logger.info(
            "Imported graph: %d entities, %d relationships, %d facts",
            counts["entities"], counts["relationships"], counts["facts"],
        )
        return counts

    # ── Relationship-Level Semantic Search ────────────────────────

    async def search_relationships(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search relationships by semantic similarity of their context.

        Embeds the query and compares against relationship context snippets
        to find relevant connections in the graph.

        Returns:
            List of dicts with relationship, source entity, target entity,
            and similarity score.
        """
        query_embedding = await self._embeddings.embed(query)

        all_entities = await self._store.get_all_entities()
        all_rels: dict[str, RelationshipEdge] = {}
        entity_index: dict[str, EntityNode] = {e.id: e for e in all_entities}

        for entity in all_entities:
            rels = await self._store.get_entity_relationships(entity.id)
            for rel in rels:
                all_rels[rel.id] = rel

        results: list[dict[str, Any]] = []
        for rel in all_rels.values():
            if not rel.context_snippets:
                continue

            # Embed the combined context of the relationship
            context_text = " ".join(rel.context_snippets[:5])
            context_embedding = await self._embeddings.embed(context_text)
            score = cosine_similarity(query_embedding, context_embedding)

            if score >= min_similarity:
                source = entity_index.get(rel.source_entity_id)
                target = entity_index.get(rel.target_entity_id)
                results.append({
                    "relationship": rel.model_dump(),
                    "source_entity": source.model_dump(exclude={"embedding"}) if source else None,
                    "target_entity": target.model_dump(exclude={"embedding"}) if target else None,
                    "similarity_score": score,
                })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    # ── Graph Analytics ──────────────────────────────────────────

    async def get_entity_importance(
        self,
        entity_id: str,
    ) -> float:
        """Calculate an importance score for an entity based on graph position.

        Considers: mention count, number of connections, connection strength,
        and number of associated facts.

        Returns:
            Importance score in [0, 1].
        """
        entity = await self._store.get_entity(entity_id)
        if not entity:
            return 0.0

        connections = await self._store.get_connected_entities(entity_id)
        facts = await self._store.get_entity_facts(entity_id)

        # Factors (all normalized to roughly 0-1 range)
        mention_score = min(1.0, math.log2(entity.mention_count + 1) / 10.0)
        connection_score = min(1.0, len(connections) / 10.0)
        avg_strength = (
            sum(r.strength for _, r in connections) / len(connections)
            if connections
            else 0.0
        )
        strength_score = min(1.0, avg_strength / 20.0)
        fact_score = min(1.0, len(facts) / 5.0)

        # Weighted combination
        importance = (
            mention_score * 0.25
            + connection_score * 0.30
            + strength_score * 0.25
            + fact_score * 0.20
        )

        return min(1.0, importance)

    async def get_most_connected_entities(
        self,
        top_k: int = 10,
        entity_type: EntityType | None = None,
    ) -> list[tuple[EntityNode, int]]:
        """Get entities ranked by number of connections.

        Returns:
            List of (entity, connection_count) tuples sorted descending.
        """
        all_entities = await self._store.get_all_entities(entity_type)
        scored: list[tuple[EntityNode, int]] = []

        for entity in all_entities:
            connections = await self._store.get_connected_entities(entity.id)
            scored.append((entity, len(connections)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
