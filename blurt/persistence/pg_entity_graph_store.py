"""Neon Postgres implementation of the EntityGraphStore protocol.

Stores entity nodes, relationship edges, facts, and learned patterns
in Postgres with pgvector for semantic search. Implements the same
interface as the in-memory SemanticMemoryStore so both satisfy
the EntityGraphStore protocol.

All queries use parameterized SQL — no string interpolation.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any
from collections.abc import Mapping

import asyncpg

from blurt.clients.embeddings import EmbeddingProvider
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


# Relationship strength decay: halves every 30 days without co-mention
STRENGTH_DECAY_HALF_LIFE_DAYS = 30.0
MIN_RELATIONSHIP_STRENGTH = 0.01
MAX_CONTEXT_SNIPPETS = 10
PATTERN_PROMOTION_THRESHOLD = 0.7


def _row_to_entity(row: Mapping[str, Any]) -> EntityNode:
    """Convert a database row to an EntityNode."""
    embedding = None
    if row["embedding"] is not None:
        embedding = list(row["embedding"])

    return EntityNode(
        id=row["id"],
        user_id=row["user_id"],
        name=row["name"],
        normalized_name=row["normalized_name"],
        entity_type=EntityType(row["entity_type"]),
        aliases=list(row["aliases"] or []),
        attributes=row["attributes"] if isinstance(row["attributes"], dict) else json.loads(row["attributes"] or "{}"),
        mention_count=row["mention_count"],
        first_seen=row["first_seen"],
        last_seen=row["last_seen"],
        embedding=embedding,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_relationship(row: Mapping[str, Any]) -> RelationshipEdge:
    """Convert a database row to a RelationshipEdge."""
    return RelationshipEdge(
        id=row["id"],
        user_id=row["user_id"],
        source_entity_id=row["source_entity_id"],
        target_entity_id=row["target_entity_id"],
        relationship_type=RelationshipType(row["relationship_type"]),
        strength=row["strength"],
        co_mention_count=row["co_mention_count"],
        context_snippets=list(row["context_snippets"] or []),
        first_seen=row["first_seen"],
        last_seen=row["last_seen"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_fact(row: Mapping[str, Any]) -> Fact:
    """Convert a database row to a Fact."""
    embedding = None
    if row["embedding"] is not None:
        embedding = list(row["embedding"])

    return Fact(
        id=row["id"],
        user_id=row["user_id"],
        fact_type=FactType(row["fact_type"]),
        subject_entity_id=row["subject_entity_id"],
        content=row["content"],
        confidence=row["confidence"],
        source_blurt_ids=list(row["source_blurt_ids"] or []),
        embedding=embedding,
        is_active=row["is_active"],
        superseded_by=row["superseded_by"],
        first_learned=row["first_learned"],
        last_confirmed=row["last_confirmed"],
        confirmation_count=row["confirmation_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_pattern(row: Mapping[str, Any]) -> LearnedPattern:
    """Convert a database row to a LearnedPattern."""
    embedding = None
    if row["embedding"] is not None:
        embedding = list(row["embedding"])

    parameters = row["parameters"]
    if isinstance(parameters, str):
        parameters = json.loads(parameters)

    return LearnedPattern(
        id=row["id"],
        user_id=row["user_id"],
        pattern_type=PatternType(row["pattern_type"]),
        description=row["description"],
        parameters=parameters or {},
        confidence=row["confidence"],
        observation_count=row["observation_count"],
        supporting_evidence=list(row["supporting_evidence"] or []),
        embedding=embedding,
        is_active=row["is_active"],
        first_detected=row["first_detected"],
        last_confirmed=row["last_confirmed"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PgEntityGraphStore:
    """Postgres-backed entity graph store implementing the EntityGraphStore protocol.

    Manages entities, relationships, facts, and patterns in Neon Postgres
    with pgvector for semantic similarity search.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        user_id: str,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._pool = pool
        self.user_id = user_id
        self._embeddings = embedding_provider

    # ── Entity Operations ─────────────────────────────────────────────

    async def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> EntityNode:
        """Add or upsert an entity. If normalized name exists, update mention count."""
        normalized = name.lower().strip()
        now = _utcnow()

        async with self._pool.acquire() as conn:
            # Check if entity already exists
            existing = await conn.fetchrow(
                "SELECT * FROM entity_nodes WHERE user_id = $1 AND normalized_name = $2",
                self.user_id, normalized,
            )

            if existing:
                # Update existing entity
                entity = _row_to_entity(existing)
                entity.mention_count += 1
                entity.last_seen = now
                entity.updated_at = now

                # Merge aliases
                if aliases:
                    new_aliases = list(set(entity.aliases) | set(aliases))
                    entity.aliases = new_aliases

                # Merge attributes
                if attributes:
                    merged = {**entity.attributes, **attributes}
                    entity.attributes = merged

                await conn.execute(
                    """
                    UPDATE entity_nodes
                    SET mention_count = $1, last_seen = $2, updated_at = $3,
                        aliases = $4, attributes = $5::jsonb
                    WHERE id = $6
                    """,
                    entity.mention_count, entity.last_seen, entity.updated_at,
                    entity.aliases, json.dumps(entity.attributes),
                    entity.id,
                )
                return entity

            # Create new entity
            entity_id = str(uuid.uuid4())
            entity_aliases = aliases or []
            entity_attrs = attributes or {}

            # Generate embedding for the entity name
            try:
                embedding = await self._embeddings.embed(name)
            except Exception:
                logger.warning("Failed to generate embedding for entity '%s'", name)
                embedding = None

            await conn.execute(
                """
                INSERT INTO entity_nodes (
                    id, user_id, name, normalized_name, entity_type,
                    aliases, attributes, mention_count,
                    first_seen, last_seen, embedding,
                    created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7::jsonb, $8,
                    $9, $10, $11,
                    $12, $13
                )
                """,
                entity_id, self.user_id, name, normalized, entity_type.value,
                entity_aliases, json.dumps(entity_attrs), 1,
                now, now, embedding,
                now, now,
            )

            return EntityNode(
                id=entity_id,
                user_id=self.user_id,
                name=name,
                normalized_name=normalized,
                entity_type=entity_type,
                aliases=entity_aliases,
                attributes=entity_attrs,
                mention_count=1,
                first_seen=now,
                last_seen=now,
                embedding=embedding,
                created_at=now,
                updated_at=now,
            )

    async def get_entity(self, entity_id: str) -> EntityNode | None:
        """Retrieve an entity by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entity_nodes WHERE id = $1", entity_id
            )
            return _row_to_entity(row) if row else None

    async def find_entity_by_name(self, name: str) -> EntityNode | None:
        """Find an entity by normalized name or alias."""
        normalized = name.lower().strip()
        async with self._pool.acquire() as conn:
            # Try normalized name first
            row = await conn.fetchrow(
                "SELECT * FROM entity_nodes WHERE user_id = $1 AND normalized_name = $2",
                self.user_id, normalized,
            )
            if row:
                return _row_to_entity(row)

            # Try aliases
            row = await conn.fetchrow(
                "SELECT * FROM entity_nodes WHERE user_id = $1 AND $2 = ANY(aliases)",
                self.user_id, normalized,
            )
            return _row_to_entity(row) if row else None

    async def get_all_entities(
        self, entity_type: EntityType | None = None
    ) -> list[EntityNode]:
        """Get all entities, optionally filtered by type."""
        async with self._pool.acquire() as conn:
            if entity_type:
                rows = await conn.fetch(
                    "SELECT * FROM entity_nodes WHERE user_id = $1 AND entity_type = $2 ORDER BY mention_count DESC",
                    self.user_id, entity_type.value,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM entity_nodes WHERE user_id = $1 ORDER BY mention_count DESC",
                    self.user_id,
                )
            return [_row_to_entity(row) for row in rows]

    async def update_entity_embedding(self, entity_id: str) -> EntityNode | None:
        """Regenerate and store the embedding for an entity."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entity_nodes WHERE id = $1", entity_id
            )
            if not row:
                return None

            entity = _row_to_entity(row)

            # Build embedding text from name + attributes + facts
            parts = [entity.name]
            if entity.attributes:
                for k, v in entity.attributes.items():
                    parts.append(f"{k}: {v}")

            # Include facts about this entity
            fact_rows = await conn.fetch(
                "SELECT content FROM facts WHERE subject_entity_id = $1 AND is_active = TRUE LIMIT 10",
                entity_id,
            )
            for fr in fact_rows:
                parts.append(fr["content"])

            embed_text = ". ".join(parts)
            try:
                embedding = await self._embeddings.embed(embed_text)
            except Exception:
                logger.warning("Failed to generate embedding for entity '%s'", entity.name)
                return entity

            now = _utcnow()
            await conn.execute(
                "UPDATE entity_nodes SET embedding = $1, updated_at = $2 WHERE id = $3",
                embedding, now, entity_id,
            )
            entity.embedding = embedding
            entity.updated_at = now
            return entity

    # ── Relationship Operations ───────────────────────────────────────

    async def add_or_strengthen_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: RelationshipType,
        context: str | None = None,
    ) -> RelationshipEdge:
        """Add a new relationship or strengthen an existing one."""
        now = _utcnow()

        async with self._pool.acquire() as conn:
            existing = await conn.fetchrow(
                """
                SELECT * FROM relationship_edges
                WHERE source_entity_id = $1 AND target_entity_id = $2
                  AND relationship_type = $3
                """,
                source_entity_id, target_entity_id, relationship_type.value,
            )

            if existing:
                edge = _row_to_relationship(existing)
                edge.co_mention_count += 1
                edge.strength = min(100.0, edge.strength + 1.0)
                edge.last_seen = now
                edge.updated_at = now

                # Append context snippet
                snippets = edge.context_snippets
                if context and context not in snippets:
                    snippets = (snippets + [context])[-MAX_CONTEXT_SNIPPETS:]
                    edge.context_snippets = snippets

                await conn.execute(
                    """
                    UPDATE relationship_edges
                    SET co_mention_count = $1, strength = $2,
                        last_seen = $3, updated_at = $4, context_snippets = $5
                    WHERE id = $6
                    """,
                    edge.co_mention_count, edge.strength,
                    edge.last_seen, edge.updated_at, edge.context_snippets,
                    edge.id,
                )
                return edge

            # Create new relationship
            edge_id = str(uuid.uuid4())
            snippets = [context] if context else []

            # Determine user_id from source entity
            source_row = await conn.fetchrow(
                "SELECT user_id FROM entity_nodes WHERE id = $1", source_entity_id
            )
            user_id = source_row["user_id"] if source_row else self.user_id

            await conn.execute(
                """
                INSERT INTO relationship_edges (
                    id, user_id, source_entity_id, target_entity_id,
                    relationship_type, strength, co_mention_count,
                    context_snippets, first_seen, last_seen,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                edge_id, user_id, source_entity_id, target_entity_id,
                relationship_type.value, 1.0, 1,
                snippets, now, now,
                now, now,
            )

            return RelationshipEdge(
                id=edge_id,
                user_id=user_id,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                relationship_type=relationship_type,
                strength=1.0,
                co_mention_count=1,
                context_snippets=snippets,
                first_seen=now,
                last_seen=now,
                created_at=now,
                updated_at=now,
            )

    async def get_entity_relationships(
        self, entity_id: str
    ) -> list[RelationshipEdge]:
        """Get all relationships involving an entity (source or target)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM relationship_edges
                WHERE source_entity_id = $1 OR target_entity_id = $1
                ORDER BY strength DESC
                """,
                entity_id,
            )
            return [_row_to_relationship(row) for row in rows]

    async def get_connected_entities(
        self, entity_id: str
    ) -> list[tuple[EntityNode, RelationshipEdge]]:
        """Get all entities connected to the given entity with their relationships."""
        rels = await self.get_entity_relationships(entity_id)
        results: list[tuple[EntityNode, RelationshipEdge]] = []

        async with self._pool.acquire() as conn:
            for rel in rels:
                other_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == entity_id
                    else rel.source_entity_id
                )
                row = await conn.fetchrow(
                    "SELECT * FROM entity_nodes WHERE id = $1", other_id
                )
                if row:
                    results.append((_row_to_entity(row), rel))

        return results

    async def decay_relationships(self, as_of: datetime | None = None) -> int:
        """Apply exponential decay to relationship strength based on time since last seen."""
        as_of = as_of or _utcnow()
        count = 0

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM relationship_edges WHERE user_id = $1",
                self.user_id,
            )

            for row in rows:
                edge = _row_to_relationship(row)
                days_since = (as_of - edge.last_seen).total_seconds() / 86400.0
                if days_since <= 0:
                    continue

                decay_factor = math.pow(0.5, days_since / STRENGTH_DECAY_HALF_LIFE_DAYS)
                new_strength = edge.strength * decay_factor

                if new_strength < MIN_RELATIONSHIP_STRENGTH:
                    new_strength = 0.0

                if new_strength != edge.strength:
                    await conn.execute(
                        "UPDATE relationship_edges SET strength = $1, updated_at = $2 WHERE id = $3",
                        new_strength, as_of, edge.id,
                    )
                    count += 1

        return count

    # ── Fact Operations ───────────────────────────────────────────────

    async def add_fact(
        self,
        content: str,
        fact_type: FactType,
        subject_entity_id: str | None = None,
        source_blurt_id: str | None = None,
        confidence: float = 1.0,
    ) -> Fact:
        """Add a new fact to the knowledge graph."""
        fact_id = str(uuid.uuid4())
        now = _utcnow()
        source_ids = [source_blurt_id] if source_blurt_id else []

        try:
            embedding = await self._embeddings.embed(content)
        except Exception:
            logger.warning("Failed to generate embedding for fact")
            embedding = None

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO facts (
                    id, user_id, fact_type, subject_entity_id, content,
                    confidence, source_blurt_ids, embedding, is_active,
                    superseded_by, first_learned, last_confirmed,
                    confirmation_count, created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9,
                    $10, $11, $12,
                    $13, $14, $15
                )
                """,
                fact_id, self.user_id, fact_type.value, subject_entity_id, content,
                confidence, source_ids, embedding, True,
                None, now, now,
                1, now, now,
            )

        return Fact(
            id=fact_id,
            user_id=self.user_id,
            fact_type=fact_type,
            subject_entity_id=subject_entity_id,
            content=content,
            confidence=confidence,
            source_blurt_ids=source_ids,
            embedding=embedding,
            is_active=True,
            superseded_by=None,
            first_learned=now,
            last_confirmed=now,
            confirmation_count=1,
            created_at=now,
            updated_at=now,
        )

    async def supersede_fact(
        self,
        old_fact_id: str,
        new_content: str,
        source_blurt_id: str | None = None,
    ) -> Fact | None:
        """Supersede an existing fact with new content."""
        async with self._pool.acquire() as conn:
            old_row = await conn.fetchrow(
                "SELECT * FROM facts WHERE id = $1", old_fact_id
            )
            if not old_row:
                return None

            old_fact = _row_to_fact(old_row)
            new_fact = await self.add_fact(
                content=new_content,
                fact_type=old_fact.fact_type,
                subject_entity_id=old_fact.subject_entity_id,
                source_blurt_id=source_blurt_id,
                confidence=old_fact.confidence,
            )

            now = _utcnow()
            await conn.execute(
                """
                UPDATE facts
                SET is_active = FALSE, superseded_by = $1, updated_at = $2
                WHERE id = $3
                """,
                new_fact.id, now, old_fact_id,
            )

            return new_fact

    async def get_entity_facts(
        self,
        entity_id: str,
        active_only: bool = True,
    ) -> list[Fact]:
        """Get facts about a specific entity."""
        async with self._pool.acquire() as conn:
            if active_only:
                rows = await conn.fetch(
                    "SELECT * FROM facts WHERE subject_entity_id = $1 AND is_active = TRUE ORDER BY last_confirmed DESC",
                    entity_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM facts WHERE subject_entity_id = $1 ORDER BY last_confirmed DESC",
                    entity_id,
                )
            return [_row_to_fact(row) for row in rows]

    async def get_all_facts(
        self,
        fact_type: FactType | None = None,
        active_only: bool = True,
    ) -> list[Fact]:
        """Get all facts, optionally filtered by type and active status."""
        conditions = ["user_id = $1"]
        params: list[Any] = [self.user_id]
        idx = 2

        if active_only:
            conditions.append("is_active = TRUE")

        if fact_type:
            conditions.append(f"fact_type = ${idx}")
            params.append(fact_type.value)
            idx += 1

        where = " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM facts WHERE {where} ORDER BY last_confirmed DESC",
                *params,
            )
            return [_row_to_fact(row) for row in rows]

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
        """Add a new behavioral pattern."""
        pattern_id = str(uuid.uuid4())
        now = _utcnow()
        params = parameters or {}
        evidence = supporting_evidence or []

        try:
            embedding = await self._embeddings.embed(description)
        except Exception:
            logger.warning("Failed to generate embedding for pattern")
            embedding = None

        is_active = confidence >= PATTERN_PROMOTION_THRESHOLD

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO learned_patterns (
                    id, user_id, pattern_type, description, parameters,
                    confidence, observation_count, supporting_evidence,
                    embedding, is_active, first_detected, last_confirmed,
                    created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5::jsonb,
                    $6, $7, $8,
                    $9, $10, $11, $12,
                    $13, $14
                )
                """,
                pattern_id, self.user_id, pattern_type.value, description,
                json.dumps(params),
                confidence, observation_count, evidence,
                embedding, is_active, now, now,
                now, now,
            )

        return LearnedPattern(
            id=pattern_id,
            user_id=self.user_id,
            pattern_type=pattern_type,
            description=description,
            parameters=params,
            confidence=confidence,
            observation_count=observation_count,
            supporting_evidence=evidence,
            embedding=embedding,
            is_active=is_active,
            first_detected=now,
            last_confirmed=now,
            created_at=now,
            updated_at=now,
        )

    async def confirm_pattern(
        self, pattern_id: str, observation_id: str | None = None
    ) -> LearnedPattern | None:
        """Confirm a pattern observation, increasing confidence."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM learned_patterns WHERE id = $1", pattern_id
            )
            if not row:
                return None

            pattern = _row_to_pattern(row)
            pattern.observation_count += 1
            pattern.last_confirmed = _utcnow()
            pattern.updated_at = pattern.last_confirmed

            if observation_id and observation_id not in pattern.supporting_evidence:
                pattern.supporting_evidence.append(observation_id)

            # Increase confidence based on observations
            pattern.confidence = min(
                1.0, pattern.confidence + 0.05
            )
            pattern.is_active = pattern.confidence >= PATTERN_PROMOTION_THRESHOLD

            await conn.execute(
                """
                UPDATE learned_patterns
                SET observation_count = $1, last_confirmed = $2, updated_at = $3,
                    supporting_evidence = $4, confidence = $5, is_active = $6
                WHERE id = $7
                """,
                pattern.observation_count, pattern.last_confirmed, pattern.updated_at,
                pattern.supporting_evidence, pattern.confidence, pattern.is_active,
                pattern.id,
            )
            return pattern

    async def get_active_patterns(
        self, pattern_type: PatternType | None = None
    ) -> list[LearnedPattern]:
        """Get active patterns, optionally filtered by type."""
        async with self._pool.acquire() as conn:
            if pattern_type:
                rows = await conn.fetch(
                    "SELECT * FROM learned_patterns WHERE user_id = $1 AND is_active = TRUE AND pattern_type = $2 ORDER BY confidence DESC",
                    self.user_id, pattern_type.value,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM learned_patterns WHERE user_id = $1 AND is_active = TRUE ORDER BY confidence DESC",
                    self.user_id,
                )
            return [_row_to_pattern(row) for row in rows]

    async def get_all_patterns(self) -> list[LearnedPattern]:
        """Get all patterns regardless of active status."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM learned_patterns WHERE user_id = $1 ORDER BY confidence DESC",
                self.user_id,
            )
            return [_row_to_pattern(row) for row in rows]

    # ── Semantic Search ───────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 10,
        item_types: list[str] | None = None,
        min_similarity: float = 0.3,
    ) -> list[SemanticSearchResult]:
        """Search across entities, facts, and patterns using vector similarity."""
        try:
            query_embedding = await self._embeddings.embed(query)
        except Exception:
            logger.warning("Failed to generate query embedding")
            return []

        max_distance = 1.0 - min_similarity
        results: list[SemanticSearchResult] = []
        search_types = item_types or ["entity", "fact", "pattern"]

        async with self._pool.acquire() as conn:
            if "entity" in search_types:
                entity_rows = await conn.fetch(
                    """
                    SELECT *, (embedding <=> $1::vector) AS distance
                    FROM entity_nodes
                    WHERE user_id = $2
                      AND embedding IS NOT NULL
                      AND (embedding <=> $1::vector) <= $3
                    ORDER BY distance ASC
                    LIMIT $4
                    """,
                    str(query_embedding), self.user_id, max_distance, top_k,
                )
                for row in entity_rows:
                    sim = 1.0 - row["distance"]
                    results.append(SemanticSearchResult(
                        item_type="entity",
                        item_id=row["id"],
                        content=f"{row['name']} ({row['entity_type']})",
                        similarity_score=sim,
                        metadata={
                            "name": row["name"],
                            "entity_type": row["entity_type"],
                            "mention_count": row["mention_count"],
                        },
                    ))

            if "fact" in search_types:
                fact_rows = await conn.fetch(
                    """
                    SELECT *, (embedding <=> $1::vector) AS distance
                    FROM facts
                    WHERE user_id = $2
                      AND is_active = TRUE
                      AND embedding IS NOT NULL
                      AND (embedding <=> $1::vector) <= $3
                    ORDER BY distance ASC
                    LIMIT $4
                    """,
                    str(query_embedding), self.user_id, max_distance, top_k,
                )
                for row in fact_rows:
                    sim = 1.0 - row["distance"]
                    results.append(SemanticSearchResult(
                        item_type="fact",
                        item_id=row["id"],
                        content=row["content"],
                        similarity_score=sim,
                        metadata={
                            "fact_type": row["fact_type"],
                            "confidence": row["confidence"],
                        },
                    ))

            if "pattern" in search_types:
                pattern_rows = await conn.fetch(
                    """
                    SELECT *, (embedding <=> $1::vector) AS distance
                    FROM learned_patterns
                    WHERE user_id = $2
                      AND is_active = TRUE
                      AND embedding IS NOT NULL
                      AND (embedding <=> $1::vector) <= $3
                    ORDER BY distance ASC
                    LIMIT $4
                    """,
                    str(query_embedding), self.user_id, max_distance, top_k,
                )
                for row in pattern_rows:
                    sim = 1.0 - row["distance"]
                    results.append(SemanticSearchResult(
                        item_type="pattern",
                        item_id=row["id"],
                        content=row["description"],
                        similarity_score=sim,
                        metadata={
                            "pattern_type": row["pattern_type"],
                            "confidence": row["confidence"],
                        },
                    ))

        # Sort combined results by similarity
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    async def search_similar_entities(
        self,
        entity_id: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        entity_type: EntityType | None = None,
    ) -> list[SemanticSearchResult]:
        """Find entities similar to a given entity by embedding."""
        entity = await self.get_entity(entity_id)
        if not entity or not entity.embedding:
            return []

        max_distance = 1.0 - min_similarity

        async with self._pool.acquire() as conn:
            if entity_type:
                rows = await conn.fetch(
                    """
                    SELECT *, (embedding <=> $1::vector) AS distance
                    FROM entity_nodes
                    WHERE user_id = $2 AND id != $3
                      AND entity_type = $4
                      AND embedding IS NOT NULL
                      AND (embedding <=> $1::vector) <= $5
                    ORDER BY distance ASC
                    LIMIT $6
                    """,
                    str(entity.embedding), self.user_id, entity_id,
                    entity_type.value, max_distance, top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT *, (embedding <=> $1::vector) AS distance
                    FROM entity_nodes
                    WHERE user_id = $2 AND id != $3
                      AND embedding IS NOT NULL
                      AND (embedding <=> $1::vector) <= $4
                    ORDER BY distance ASC
                    LIMIT $5
                    """,
                    str(entity.embedding), self.user_id, entity_id,
                    max_distance, top_k,
                )

            results = []
            for row in rows:
                sim = 1.0 - row["distance"]
                results.append(SemanticSearchResult(
                    item_type="entity",
                    item_id=row["id"],
                    content=f"{row['name']} ({row['entity_type']})",
                    similarity_score=sim,
                    metadata={
                        "name": row["name"],
                        "entity_type": row["entity_type"],
                        "mention_count": row["mention_count"],
                    },
                ))
            return results

    async def search_entities_by_query(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        entity_type: EntityType | None = None,
        min_mentions: int = 0,
    ) -> list[SemanticSearchResult]:
        """Search entities by a text query with optional filters."""
        try:
            query_embedding = await self._embeddings.embed(query)
        except Exception:
            return []

        max_distance = 1.0 - min_similarity

        conditions = [
            "user_id = $2",
            "embedding IS NOT NULL",
            "(embedding <=> $1::vector) <= $3",
        ]
        params: list[Any] = [str(query_embedding), self.user_id, max_distance]
        idx = 4

        if entity_type:
            conditions.append(f"entity_type = ${idx}")
            params.append(entity_type.value)
            idx += 1

        if min_mentions > 0:
            conditions.append(f"mention_count >= ${idx}")
            params.append(min_mentions)
            idx += 1

        params.append(top_k)
        where = " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT *, (embedding <=> $1::vector) AS distance
                FROM entity_nodes
                WHERE {where}
                ORDER BY distance ASC
                LIMIT ${idx}
                """,
                *params,
            )

            return [
                SemanticSearchResult(
                    item_type="entity",
                    item_id=row["id"],
                    content=f"{row['name']} ({row['entity_type']})",
                    similarity_score=1.0 - row["distance"],
                    metadata={
                        "name": row["name"],
                        "entity_type": row["entity_type"],
                        "mention_count": row["mention_count"],
                    },
                )
                for row in rows
            ]

    async def search_neighborhood(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        max_hops: int = 2,
        relationship_types: list[RelationshipType] | None = None,
    ) -> list[SemanticSearchResult]:
        """Search entities and expand to neighborhood via relationships."""
        # Start with entity search
        seed_results = await self.search(
            query, top_k=top_k, item_types=["entity"], min_similarity=min_similarity
        )

        if not seed_results:
            return seed_results

        # Expand to connected entities
        seen_ids = {r.item_id for r in seed_results}
        expanded: list[SemanticSearchResult] = list(seed_results)

        for result in seed_results[:3]:  # Expand top 3 seeds
            connections = await self.get_connected_entities(result.item_id)
            for entity, rel in connections:
                if entity.id in seen_ids:
                    continue
                if relationship_types and rel.relationship_type not in relationship_types:
                    continue
                seen_ids.add(entity.id)
                # Blend relationship strength with seed similarity
                blended_score = result.similarity_score * 0.5 + (rel.strength / 100.0) * 0.5
                expanded.append(SemanticSearchResult(
                    item_type="entity",
                    item_id=entity.id,
                    content=f"{entity.name} ({entity.entity_type.value})",
                    similarity_score=min(1.0, blended_score),
                    metadata={
                        "name": entity.name,
                        "entity_type": entity.entity_type.value,
                        "via_relationship": rel.relationship_type.value,
                        "seed_entity": result.item_id,
                    },
                ))

        expanded.sort(key=lambda r: r.similarity_score, reverse=True)
        return expanded[:top_k]

    async def recall(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.2,
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        """Rich recall — search everything and include graph context."""
        results = await self.search(query, top_k=top_k, min_similarity=min_similarity)

        enriched: list[dict[str, Any]] = []
        for result in results:
            item: dict[str, Any] = {
                "type": result.item_type,
                "id": result.item_id,
                "content": result.content,
                "similarity": result.similarity_score,
                "metadata": result.metadata,
            }

            if include_context and result.item_type == "entity":
                ctx = await self.get_entity_context(result.item_id)
                item["context"] = ctx

            enriched.append(item)

        return enriched

    async def find_related_by_embedding(
        self,
        entity_id: str,
        item_types: list[str] | None = None,
        top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> list[SemanticSearchResult]:
        """Find items related to an entity by embedding similarity."""
        entity = await self.get_entity(entity_id)
        if not entity or not entity.embedding:
            return []

        # Use the entity's embedding as the query
        return await self.search(
            entity.name,  # fallback to name-based search
            top_k=top_k,
            item_types=item_types,
            min_similarity=min_similarity,
        )

    # ── Context & Stats ───────────────────────────────────────────────

    async def get_entity_context(self, entity_id: str) -> dict[str, Any]:
        """Get full context for an entity: attributes, facts, connections."""
        entity = await self.get_entity(entity_id)
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
        async with self._pool.acquire() as conn:
            entity_count = await conn.fetchval(
                "SELECT COUNT(*) FROM entity_nodes WHERE user_id = $1",
                self.user_id,
            )
            rel_count = await conn.fetchval(
                "SELECT COUNT(*) FROM relationship_edges WHERE user_id = $1",
                self.user_id,
            )
            fact_count = await conn.fetchval(
                "SELECT COUNT(*) FROM facts WHERE user_id = $1 AND is_active = TRUE",
                self.user_id,
            )
            pattern_count = await conn.fetchval(
                "SELECT COUNT(*) FROM learned_patterns WHERE user_id = $1 AND is_active = TRUE",
                self.user_id,
            )

            # Entity type distribution
            type_rows = await conn.fetch(
                "SELECT entity_type, COUNT(*) as cnt FROM entity_nodes WHERE user_id = $1 GROUP BY entity_type",
                self.user_id,
            )
            type_dist = {row["entity_type"]: row["cnt"] for row in type_rows}

        return {
            "total_entities": entity_count or 0,
            "total_relationships": rel_count or 0,
            "total_facts": fact_count or 0,
            "total_patterns": pattern_count or 0,
            "entity_type_distribution": type_dist,
        }

    # ── Batch Operations ──────────────────────────────────────────────

    async def process_extracted_entities(
        self,
        entities: list[dict[str, Any]],
        blurt_id: str,
        raw_text: str,
    ) -> list[EntityNode]:
        """Process entities extracted from a blurt and add to the graph."""
        added_entities: list[EntityNode] = []
        entity_ids: list[str] = []

        for entity_data in entities:
            name = entity_data.get("name", "")
            if not name:
                continue

            entity_type_str = entity_data.get("type", entity_data.get("entity_type", "topic"))
            try:
                entity_type = EntityType(entity_type_str.lower())
            except ValueError:
                entity_type = EntityType.TOPIC

            attrs = entity_data.get("attributes", {})
            aliases = entity_data.get("aliases", [])

            entity = await self.add_entity(
                name=name,
                entity_type=entity_type,
                aliases=aliases,
                attributes=attrs,
            )
            added_entities.append(entity)
            entity_ids.append(entity.id)

        # Create MENTIONED_WITH relationships for co-occurring entities
        for i, eid1 in enumerate(entity_ids):
            for eid2 in entity_ids[i + 1:]:
                if eid1 != eid2:
                    await self.add_or_strengthen_relationship(
                        eid1, eid2,
                        RelationshipType.MENTIONED_WITH,
                        context=raw_text[:200],
                    )

        return added_entities

    async def process_extracted_facts(
        self,
        facts: list[dict[str, Any]],
        blurt_id: str,
    ) -> list[Fact]:
        """Process facts extracted from a blurt."""
        added_facts: list[Fact] = []

        for fact_data in facts:
            content = fact_data.get("content", "")
            if not content:
                continue

            fact_type_str = fact_data.get("type", fact_data.get("fact_type", "attribute"))
            try:
                fact_type = FactType(fact_type_str.lower())
            except ValueError:
                fact_type = FactType.ATTRIBUTE

            subject_entity_id = fact_data.get("subject_entity_id")
            confidence = fact_data.get("confidence", 1.0)

            fact = await self.add_fact(
                content=content,
                fact_type=fact_type,
                subject_entity_id=subject_entity_id,
                source_blurt_id=blurt_id,
                confidence=confidence,
            )
            added_facts.append(fact)

        return added_facts
