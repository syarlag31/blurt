"""Local SQLite-backed knowledge graph storage.

Provides persistent, offline storage for the knowledge graph — entities,
relationships, facts, and patterns — using SQLite. No external API calls.
Full feature parity with the cloud-backed SemanticMemoryStore.

Data is stored in ``~/.blurt/knowledge.db`` by default (configurable via
BlurtConfig.data_dir). All data stays on-device.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
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

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Relationship strength decay: halves every 30 days without co-mention
STRENGTH_DECAY_HALF_LIFE_DAYS = 30.0
MIN_RELATIONSHIP_STRENGTH = 0.01
MAX_CONTEXT_SNIPPETS = 10
PATTERN_PROMOTION_THRESHOLD = 0.7
MIN_PATTERN_OBSERVATIONS = 5

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    attributes TEXT NOT NULL DEFAULT '{}',
    mention_count INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    embedding TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 1.0,
    co_mention_count INTEGER NOT NULL DEFAULT 1,
    context_snippets TEXT NOT NULL DEFAULT '[]',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (source_entity_id) REFERENCES entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relationship_type);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    subject_entity_id TEXT,
    content TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source_blurt_ids TEXT NOT NULL DEFAULT '[]',
    embedding TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    superseded_by TEXT,
    first_learned TEXT NOT NULL,
    last_confirmed TEXT NOT NULL,
    confirmation_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (subject_entity_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(is_active);

CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    parameters TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.5,
    observation_count INTEGER NOT NULL DEFAULT 0,
    supporting_evidence TEXT NOT NULL DEFAULT '[]',
    embedding TEXT,
    is_active INTEGER NOT NULL DEFAULT 0,
    first_detected TEXT NOT NULL,
    last_confirmed TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_patterns_user ON patterns(user_id);
CREATE INDEX IF NOT EXISTS idx_patterns_active ON patterns(is_active);

CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (alias, entity_id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias);
"""


class LocalKnowledgeGraphStore:
    """SQLite-backed knowledge graph for fully offline operation.

    Drop-in replacement for SemanticMemoryStore that persists all data
    to a local SQLite database. No network calls, no data leakage.

    Usage::

        from blurt.clients.embeddings import MockEmbeddingProvider
        store = LocalKnowledgeGraphStore(
            user_id="user-1",
            embedding_provider=MockEmbeddingProvider(),
            db_path=Path("~/.blurt/knowledge.db"),
        )
        await store.initialize()
        entity = await store.add_entity("Sarah", EntityType.PERSON)
    """

    def __init__(
        self,
        user_id: str,
        embedding_provider: EmbeddingProvider,
        db_path: Path | None = None,
    ) -> None:
        self.user_id = user_id
        self._embeddings = embedding_provider
        self._db_path = db_path or (Path.home() / ".blurt" / "knowledge.db")
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Create the database and tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        return self._conn

    # ── Serialization helpers ────────────────────────────────────────

    @staticmethod
    def _to_json(obj: Any) -> str:
        return json.dumps(obj, default=str)

    @staticmethod
    def _from_json(s: str | None) -> Any:
        if not s:
            return None
        return json.loads(s)

    @staticmethod
    def _parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s)

    @staticmethod
    def _dt_str(dt: datetime) -> str:
        return dt.isoformat()

    # ── Entity Operations ────────────────────────────────────────────

    def _row_to_entity(self, row: sqlite3.Row) -> EntityNode:
        return EntityNode(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            normalized_name=row["normalized_name"],
            entity_type=EntityType(row["entity_type"]),
            aliases=self._from_json(row["aliases"]) or [],
            attributes=self._from_json(row["attributes"]) or {},
            mention_count=row["mention_count"],
            first_seen=self._parse_dt(row["first_seen"]),
            last_seen=self._parse_dt(row["last_seen"]),
            embedding=self._from_json(row["embedding"]),
            created_at=self._parse_dt(row["created_at"]),
            updated_at=self._parse_dt(row["updated_at"]),
        )

    async def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> EntityNode:
        """Add or update an entity in the knowledge graph."""
        normalized = name.lower().strip()

        # Check for existing entity by name or alias
        existing = await self._find_entity_by_normalized(normalized)
        if existing is None and aliases:
            for alias in aliases:
                existing = await self._find_entity_by_alias(alias.lower().strip())
                if existing:
                    break

        if existing:
            now = _utcnow()
            new_attrs = dict(existing.attributes)
            if attributes:
                new_attrs.update(attributes)
            new_aliases = list(existing.aliases)
            if aliases:
                for a in aliases:
                    a_norm = a.lower().strip()
                    if a_norm not in new_aliases:
                        new_aliases.append(a_norm)
                        self._db.execute(
                            "INSERT OR IGNORE INTO aliases (alias, entity_id) VALUES (?, ?)",
                            (a_norm, existing.id),
                        )

            self._db.execute(
                """UPDATE entities SET
                    mention_count = mention_count + 1,
                    last_seen = ?,
                    updated_at = ?,
                    attributes = ?,
                    aliases = ?
                WHERE id = ?""",
                (
                    self._dt_str(now),
                    self._dt_str(now),
                    self._to_json(new_attrs),
                    self._to_json(new_aliases),
                    existing.id,
                ),
            )
            self._db.commit()

            existing.mention_count += 1
            existing.last_seen = now
            existing.updated_at = now
            existing.attributes = new_attrs
            existing.aliases = new_aliases
            return existing

        # Create new entity
        embedding = await self._embeddings.embed(f"{entity_type.value}: {name}")
        now = _utcnow()
        alias_list = [a.lower().strip() for a in (aliases or [])]

        entity = EntityNode(
            user_id=self.user_id,
            name=name,
            normalized_name=normalized,
            entity_type=entity_type,
            aliases=alias_list,
            attributes=attributes or {},
            mention_count=1,
            embedding=embedding,
            first_seen=now,
            last_seen=now,
            created_at=now,
            updated_at=now,
        )

        self._db.execute(
            """INSERT INTO entities
            (id, user_id, name, normalized_name, entity_type, aliases,
             attributes, mention_count, first_seen, last_seen, embedding,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entity.id,
                entity.user_id,
                entity.name,
                entity.normalized_name,
                entity.entity_type.value,
                self._to_json(entity.aliases),
                self._to_json(entity.attributes),
                entity.mention_count,
                self._dt_str(entity.first_seen),
                self._dt_str(entity.last_seen),
                self._to_json(entity.embedding),
                self._dt_str(entity.created_at),
                self._dt_str(entity.updated_at),
            ),
        )

        # Insert alias index entries
        for alias in alias_list:
            self._db.execute(
                "INSERT OR IGNORE INTO aliases (alias, entity_id) VALUES (?, ?)",
                (alias, entity.id),
            )

        self._db.commit()
        return entity

    async def _find_entity_by_normalized(self, normalized: str) -> EntityNode | None:
        row = self._db.execute(
            "SELECT * FROM entities WHERE normalized_name = ? AND user_id = ?",
            (normalized, self.user_id),
        ).fetchone()
        return self._row_to_entity(row) if row else None

    async def _find_entity_by_alias(self, alias: str) -> EntityNode | None:
        row = self._db.execute(
            """SELECT e.* FROM entities e
            JOIN aliases a ON e.id = a.entity_id
            WHERE a.alias = ? AND e.user_id = ?""",
            (alias, self.user_id),
        ).fetchone()
        return self._row_to_entity(row) if row else None

    async def get_entity(self, entity_id: str) -> EntityNode | None:
        """Retrieve an entity by ID."""
        row = self._db.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    async def find_entity_by_name(self, name: str) -> EntityNode | None:
        """Find an entity by name or alias."""
        normalized = name.lower().strip()
        entity = await self._find_entity_by_normalized(normalized)
        if entity:
            return entity
        return await self._find_entity_by_alias(normalized)

    async def get_all_entities(
        self, entity_type: EntityType | None = None
    ) -> list[EntityNode]:
        """Get all entities, optionally filtered by type."""
        if entity_type:
            rows = self._db.execute(
                "SELECT * FROM entities WHERE user_id = ? AND entity_type = ?",
                (self.user_id, entity_type.value),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM entities WHERE user_id = ?", (self.user_id,)
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    async def update_entity_embedding(self, entity_id: str) -> EntityNode | None:
        """Regenerate the embedding for an entity."""
        entity = await self.get_entity(entity_id)
        if not entity:
            return None

        parts = [f"{entity.entity_type.value}: {entity.name}"]
        if entity.aliases:
            parts.append(f"also known as: {', '.join(entity.aliases)}")
        if entity.attributes:
            for k, v in entity.attributes.items():
                parts.append(f"{k}: {v}")

        facts = await self.get_entity_facts(entity_id)
        for fact in facts[:5]:
            parts.append(fact.content)

        text = ". ".join(parts)
        embedding = await self._embeddings.embed(text)
        now = _utcnow()

        self._db.execute(
            "UPDATE entities SET embedding = ?, updated_at = ? WHERE id = ?",
            (self._to_json(embedding), self._dt_str(now), entity_id),
        )
        self._db.commit()

        entity.embedding = embedding
        entity.updated_at = now
        return entity

    # ── Relationship Operations ──────────────────────────────────────

    def _row_to_relationship(self, row: sqlite3.Row) -> RelationshipEdge:
        return RelationshipEdge(
            id=row["id"],
            user_id=row["user_id"],
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            relationship_type=RelationshipType(row["relationship_type"]),
            strength=row["strength"],
            co_mention_count=row["co_mention_count"],
            context_snippets=self._from_json(row["context_snippets"]) or [],
            first_seen=self._parse_dt(row["first_seen"]),
            last_seen=self._parse_dt(row["last_seen"]),
            created_at=self._parse_dt(row["created_at"]),
            updated_at=self._parse_dt(row["updated_at"]),
        )

    async def add_or_strengthen_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: RelationshipType,
        context: str | None = None,
    ) -> RelationshipEdge:
        """Add a new relationship or strengthen an existing one."""
        existing = await self._find_relationship(
            source_entity_id, target_entity_id, relationship_type
        )

        if existing:
            existing.co_mention_count += 1
            existing.strength = min(
                100.0, math.log2(existing.co_mention_count + 1) * 10
            )
            existing.last_seen = _utcnow()
            existing.updated_at = _utcnow()
            if context:
                existing.context_snippets.append(context)
                if len(existing.context_snippets) > MAX_CONTEXT_SNIPPETS:
                    existing.context_snippets = existing.context_snippets[
                        -MAX_CONTEXT_SNIPPETS:
                    ]

            self._db.execute(
                """UPDATE relationships SET
                    co_mention_count = ?, strength = ?, last_seen = ?,
                    updated_at = ?, context_snippets = ?
                WHERE id = ?""",
                (
                    existing.co_mention_count,
                    existing.strength,
                    self._dt_str(existing.last_seen),
                    self._dt_str(existing.updated_at),
                    self._to_json(existing.context_snippets),
                    existing.id,
                ),
            )
            self._db.commit()
            return existing

        now = _utcnow()
        edge = RelationshipEdge(
            user_id=self.user_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            strength=1.0,
            context_snippets=[context] if context else [],
            first_seen=now,
            last_seen=now,
            created_at=now,
            updated_at=now,
        )

        self._db.execute(
            """INSERT INTO relationships
            (id, user_id, source_entity_id, target_entity_id, relationship_type,
             strength, co_mention_count, context_snippets, first_seen, last_seen,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge.id,
                edge.user_id,
                edge.source_entity_id,
                edge.target_entity_id,
                edge.relationship_type.value,
                edge.strength,
                edge.co_mention_count,
                self._to_json(edge.context_snippets),
                self._dt_str(edge.first_seen),
                self._dt_str(edge.last_seen),
                self._dt_str(edge.created_at),
                self._dt_str(edge.updated_at),
            ),
        )
        self._db.commit()
        return edge

    async def _find_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType,
    ) -> RelationshipEdge | None:
        """Find existing relationship between two entities."""
        row = self._db.execute(
            """SELECT * FROM relationships
            WHERE relationship_type = ?
            AND ((source_entity_id = ? AND target_entity_id = ?)
                 OR (source_entity_id = ? AND target_entity_id = ?))""",
            (rel_type.value, source_id, target_id, target_id, source_id),
        ).fetchone()
        return self._row_to_relationship(row) if row else None

    async def get_entity_relationships(
        self, entity_id: str
    ) -> list[RelationshipEdge]:
        """Get all relationships for an entity."""
        rows = self._db.execute(
            """SELECT * FROM relationships
            WHERE source_entity_id = ? OR target_entity_id = ?""",
            (entity_id, entity_id),
        ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    async def get_connected_entities(
        self, entity_id: str
    ) -> list[tuple[EntityNode, RelationshipEdge]]:
        """Get all entities connected to the given entity."""
        relationships = await self.get_entity_relationships(entity_id)
        results = []
        for rel in relationships:
            other_id = (
                rel.target_entity_id
                if rel.source_entity_id == entity_id
                else rel.source_entity_id
            )
            other_entity = await self.get_entity(other_id)
            if other_entity:
                results.append((other_entity, rel))
        results.sort(key=lambda x: x[1].strength, reverse=True)
        return results

    async def decay_relationships(self, as_of: datetime | None = None) -> int:
        """Apply time-based decay to all relationships."""
        now = as_of or _utcnow()
        rows = self._db.execute(
            "SELECT * FROM relationships WHERE user_id = ?", (self.user_id,)
        ).fetchall()

        dormant_count = 0
        for row in rows:
            rel = self._row_to_relationship(row)
            days_since = (now - rel.last_seen).total_seconds() / 86400
            if days_since <= 0:
                continue

            decay_factor = math.pow(0.5, days_since / STRENGTH_DECAY_HALF_LIFE_DAYS)
            new_strength = max(MIN_RELATIONSHIP_STRENGTH, rel.strength * decay_factor)

            self._db.execute(
                "UPDATE relationships SET strength = ?, updated_at = ? WHERE id = ?",
                (new_strength, self._dt_str(now), rel.id),
            )

            if new_strength <= MIN_RELATIONSHIP_STRENGTH:
                dormant_count += 1

        self._db.commit()
        return dormant_count

    # ── Fact Operations ──────────────────────────────────────────────

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            user_id=row["user_id"],
            fact_type=FactType(row["fact_type"]),
            subject_entity_id=row["subject_entity_id"],
            content=row["content"],
            confidence=row["confidence"],
            source_blurt_ids=self._from_json(row["source_blurt_ids"]) or [],
            embedding=self._from_json(row["embedding"]),
            is_active=bool(row["is_active"]),
            superseded_by=row["superseded_by"],
            first_learned=self._parse_dt(row["first_learned"]),
            last_confirmed=self._parse_dt(row["last_confirmed"]),
            confirmation_count=row["confirmation_count"],
            created_at=self._parse_dt(row["created_at"]),
            updated_at=self._parse_dt(row["updated_at"]),
        )

    async def add_fact(
        self,
        content: str,
        fact_type: FactType,
        subject_entity_id: str | None = None,
        source_blurt_id: str | None = None,
        confidence: float = 1.0,
    ) -> Fact:
        """Store a new fact, or confirm an existing similar one."""
        embedding = await self._embeddings.embed(content)

        existing = await self._find_similar_fact(
            embedding, subject_entity_id, threshold=0.92
        )

        if existing:
            now = _utcnow()
            existing.confirmation_count += 1
            existing.last_confirmed = now
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.updated_at = now
            if source_blurt_id:
                existing.source_blurt_ids.append(source_blurt_id)

            self._db.execute(
                """UPDATE facts SET
                    confirmation_count = ?, last_confirmed = ?,
                    confidence = ?, updated_at = ?, source_blurt_ids = ?
                WHERE id = ?""",
                (
                    existing.confirmation_count,
                    self._dt_str(existing.last_confirmed),
                    existing.confidence,
                    self._dt_str(existing.updated_at),
                    self._to_json(existing.source_blurt_ids),
                    existing.id,
                ),
            )
            self._db.commit()
            return existing

        now = _utcnow()
        fact = Fact(
            user_id=self.user_id,
            fact_type=fact_type,
            subject_entity_id=subject_entity_id,
            content=content,
            confidence=confidence,
            source_blurt_ids=[source_blurt_id] if source_blurt_id else [],
            embedding=embedding,
            first_learned=now,
            last_confirmed=now,
            created_at=now,
            updated_at=now,
        )

        self._db.execute(
            """INSERT INTO facts
            (id, user_id, fact_type, subject_entity_id, content, confidence,
             source_blurt_ids, embedding, is_active, superseded_by,
             first_learned, last_confirmed, confirmation_count,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.id,
                fact.user_id,
                fact.fact_type.value,
                fact.subject_entity_id,
                fact.content,
                fact.confidence,
                self._to_json(fact.source_blurt_ids),
                self._to_json(fact.embedding),
                1,
                None,
                self._dt_str(fact.first_learned),
                self._dt_str(fact.last_confirmed),
                fact.confirmation_count,
                self._dt_str(fact.created_at),
                self._dt_str(fact.updated_at),
            ),
        )
        self._db.commit()
        return fact

    async def _find_similar_fact(
        self,
        embedding: list[float],
        subject_entity_id: str | None,
        threshold: float = 0.92,
    ) -> Fact | None:
        """Find an existing fact with high semantic similarity."""
        if subject_entity_id:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE subject_entity_id = ? AND is_active = 1",
                (subject_entity_id,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE user_id = ? AND is_active = 1",
                (self.user_id,),
            ).fetchall()

        best_match: Fact | None = None
        best_score = 0.0

        for row in rows:
            fact = self._row_to_fact(row)
            if fact.embedding is None:
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
        """Replace an old fact with a new one."""
        row = self._db.execute(
            "SELECT * FROM facts WHERE id = ?", (old_fact_id,)
        ).fetchone()
        if not row:
            return None

        old_fact = self._row_to_fact(row)
        new_fact = await self.add_fact(
            content=new_content,
            fact_type=old_fact.fact_type,
            subject_entity_id=old_fact.subject_entity_id,
            source_blurt_id=source_blurt_id,
        )

        now = _utcnow()
        self._db.execute(
            """UPDATE facts SET
                is_active = 0, superseded_by = ?, updated_at = ?
            WHERE id = ?""",
            (new_fact.id, self._dt_str(now), old_fact_id),
        )
        self._db.commit()
        return new_fact

    async def get_entity_facts(
        self, entity_id: str, active_only: bool = True
    ) -> list[Fact]:
        """Get all facts about a specific entity."""
        if active_only:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE subject_entity_id = ? AND is_active = 1",
                (entity_id,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE subject_entity_id = ?", (entity_id,)
            ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    async def get_all_facts(
        self, fact_type: FactType | None = None, active_only: bool = True
    ) -> list[Fact]:
        """Get all facts, optionally filtered by type."""
        query = "SELECT * FROM facts WHERE user_id = ?"
        params: list[Any] = [self.user_id]
        if active_only:
            query += " AND is_active = 1"
        if fact_type:
            query += " AND fact_type = ?"
            params.append(fact_type.value)
        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_fact(r) for r in rows]

    # ── Pattern Operations ───────────────────────────────────────────

    def _row_to_pattern(self, row: sqlite3.Row) -> LearnedPattern:
        return LearnedPattern(
            id=row["id"],
            user_id=row["user_id"],
            pattern_type=PatternType(row["pattern_type"]),
            description=row["description"],
            parameters=self._from_json(row["parameters"]) or {},
            confidence=row["confidence"],
            observation_count=row["observation_count"],
            supporting_evidence=self._from_json(row["supporting_evidence"]) or [],
            embedding=self._from_json(row["embedding"]),
            is_active=bool(row["is_active"]),
            first_detected=self._parse_dt(row["first_detected"]),
            last_confirmed=self._parse_dt(row["last_confirmed"]),
            created_at=self._parse_dt(row["created_at"]),
            updated_at=self._parse_dt(row["updated_at"]),
        )

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
        now = _utcnow()

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
            first_detected=now,
            last_confirmed=now,
            created_at=now,
            updated_at=now,
        )

        self._db.execute(
            """INSERT INTO patterns
            (id, user_id, pattern_type, description, parameters, confidence,
             observation_count, supporting_evidence, embedding, is_active,
             first_detected, last_confirmed, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pattern.id,
                pattern.user_id,
                pattern.pattern_type.value,
                pattern.description,
                self._to_json(pattern.parameters),
                pattern.confidence,
                pattern.observation_count,
                self._to_json(pattern.supporting_evidence),
                self._to_json(pattern.embedding),
                int(pattern.is_active),
                self._dt_str(pattern.first_detected),
                self._dt_str(pattern.last_confirmed),
                self._dt_str(pattern.created_at),
                self._dt_str(pattern.updated_at),
            ),
        )
        self._db.commit()
        return pattern

    async def confirm_pattern(
        self, pattern_id: str, observation_id: str | None = None
    ) -> LearnedPattern | None:
        """Confirm a pattern with new supporting evidence."""
        row = self._db.execute(
            "SELECT * FROM patterns WHERE id = ?", (pattern_id,)
        ).fetchone()
        if not row:
            return None

        pattern = self._row_to_pattern(row)
        pattern.observation_count += 1
        if observation_id:
            pattern.supporting_evidence.append(observation_id)

        pattern.confidence = min(
            1.0, 1.0 - (1.0 / (1.0 + pattern.observation_count * 0.5))
        )

        if (
            pattern.confidence >= PATTERN_PROMOTION_THRESHOLD
            and pattern.observation_count >= MIN_PATTERN_OBSERVATIONS
        ):
            pattern.is_active = True

        pattern.last_confirmed = _utcnow()
        pattern.updated_at = _utcnow()

        self._db.execute(
            """UPDATE patterns SET
                observation_count = ?, confidence = ?, is_active = ?,
                supporting_evidence = ?, last_confirmed = ?, updated_at = ?
            WHERE id = ?""",
            (
                pattern.observation_count,
                pattern.confidence,
                int(pattern.is_active),
                self._to_json(pattern.supporting_evidence),
                self._dt_str(pattern.last_confirmed),
                self._dt_str(pattern.updated_at),
                pattern.id,
            ),
        )
        self._db.commit()
        return pattern

    async def get_active_patterns(
        self, pattern_type: PatternType | None = None
    ) -> list[LearnedPattern]:
        """Get all active patterns, optionally filtered by type."""
        if pattern_type:
            rows = self._db.execute(
                "SELECT * FROM patterns WHERE user_id = ? AND is_active = 1 AND pattern_type = ?",
                (self.user_id, pattern_type.value),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM patterns WHERE user_id = ? AND is_active = 1",
                (self.user_id,),
            ).fetchall()
        return [self._row_to_pattern(r) for r in rows]

    async def get_all_patterns(self) -> list[LearnedPattern]:
        """Get all patterns, including inactive ones."""
        rows = self._db.execute(
            "SELECT * FROM patterns WHERE user_id = ?", (self.user_id,)
        ).fetchall()
        return [self._row_to_pattern(r) for r in rows]

    # ── Semantic Search ──────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 10,
        item_types: list[str] | None = None,
        min_similarity: float = 0.3,
    ) -> list[SemanticSearchResult]:
        """Search across the knowledge graph using semantic similarity."""
        # Clamp min_similarity to valid range for SemanticSearchResult
        min_similarity = max(0.0, min_similarity)
        query_embedding = await self._embeddings.embed(query)
        results: list[SemanticSearchResult] = []
        search_types = item_types or ["entity", "fact", "pattern"]

        if "entity" in search_types:
            rows = self._db.execute(
                "SELECT * FROM entities WHERE user_id = ? AND embedding IS NOT NULL",
                (self.user_id,),
            ).fetchall()
            for row in rows:
                entity = self._row_to_entity(row)
                if entity.embedding is None:
                    continue
                score = cosine_similarity(query_embedding, entity.embedding)
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

        if "fact" in search_types:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE user_id = ? AND is_active = 1 AND embedding IS NOT NULL",
                (self.user_id,),
            ).fetchall()
            for row in rows:
                fact = self._row_to_fact(row)
                if fact.embedding is None:
                    continue
                score = cosine_similarity(query_embedding, fact.embedding)
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
                                "confirmation_count": fact.confirmation_count,
                            },
                        )
                    )

        if "pattern" in search_types:
            rows = self._db.execute(
                "SELECT * FROM patterns WHERE user_id = ? AND is_active = 1 AND embedding IS NOT NULL",
                (self.user_id,),
            ).fetchall()
            for row in rows:
                pattern = self._row_to_pattern(row)
                if pattern.embedding is None:
                    continue
                score = cosine_similarity(query_embedding, pattern.embedding)
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
                                "observation_count": pattern.observation_count,
                            },
                        )
                    )

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    # ── Graph Queries ────────────────────────────────────────────────

    async def get_entity_context(self, entity_id: str) -> dict[str, Any]:
        """Get full context for an entity."""
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
        entity_count = self._db.execute(
            "SELECT COUNT(*) FROM entities WHERE user_id = ?", (self.user_id,)
        ).fetchone()[0]
        rel_count = self._db.execute(
            "SELECT COUNT(*) FROM relationships WHERE user_id = ?", (self.user_id,)
        ).fetchone()[0]
        fact_count = self._db.execute(
            "SELECT COUNT(*) FROM facts WHERE user_id = ?", (self.user_id,)
        ).fetchone()[0]
        active_fact_count = self._db.execute(
            "SELECT COUNT(*) FROM facts WHERE user_id = ? AND is_active = 1",
            (self.user_id,),
        ).fetchone()[0]
        pattern_count = self._db.execute(
            "SELECT COUNT(*) FROM patterns WHERE user_id = ?", (self.user_id,)
        ).fetchone()[0]
        active_pattern_count = self._db.execute(
            "SELECT COUNT(*) FROM patterns WHERE user_id = ? AND is_active = 1",
            (self.user_id,),
        ).fetchone()[0]

        type_rows = self._db.execute(
            "SELECT entity_type, COUNT(*) as cnt FROM entities WHERE user_id = ? GROUP BY entity_type",
            (self.user_id,),
        ).fetchall()
        entity_type_counts = {row["entity_type"]: row["cnt"] for row in type_rows}

        return {
            "total_entities": entity_count,
            "total_relationships": rel_count,
            "total_facts": fact_count,
            "active_facts": active_fact_count,
            "total_patterns": pattern_count,
            "active_patterns": active_pattern_count,
            "entity_type_counts": entity_type_counts,
        }

    # ── Batch Operations for Pipeline Integration ────────────────────

    async def process_extracted_entities(
        self,
        entities: list[dict[str, Any]],
        blurt_id: str,
        raw_text: str,
    ) -> list[EntityNode]:
        """Process entities extracted from a blurt through the pipeline."""
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

        for i, e1 in enumerate(processed):
            for e2 in processed[i + 1:]:
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
