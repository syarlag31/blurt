"""Integration tests for PgEntityGraphStore.

Verifies entity graph persistence, co-mention strength updates on
repeated mentions, and correct retrieval of relationship data.

Uses a mock asyncpg pool to verify SQL correctness and data mapping
without requiring a live Neon Postgres connection.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.models.entities import (
    EntityNode,
    EntityType,
    Fact,
    FactType,
    LearnedPattern,
    PatternType,
    RelationshipEdge,
    RelationshipType,
)
from blurt.persistence.pg_entity_graph_store import (
    MAX_CONTEXT_SNIPPETS,
    STRENGTH_DECAY_HALF_LIFE_DAYS,
    PgEntityGraphStore,
    _row_to_entity,
    _row_to_fact,
    _row_to_pattern,
    _row_to_relationship,
)


# ── Helpers ──────────────────────────────────────────────────────────

USER_ID = "__test_user__"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_mock_conn() -> AsyncMock:
    """Create a mock asyncpg connection with async context manager."""
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    return conn


def _make_pool(conn: AsyncMock | None = None) -> MagicMock:
    """Create a mock asyncpg pool with acquire() returning async CM."""
    pool = MagicMock()
    if conn is None:
        conn = _make_mock_conn()
    pool._conn = conn  # store for test assertions
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool


def _mock_embedding_provider() -> MagicMock:
    """Create a mock embedding provider that returns deterministic vectors."""
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[0.1] * 768)
    return provider


def _make_entity_row(**overrides: Any) -> dict:
    """Create a dict simulating an asyncpg Record for entity_nodes."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": USER_ID,
        "name": "Alice",
        "normalized_name": "alice",
        "entity_type": "person",
        "aliases": [],
        "attributes": {},
        "mention_count": 1,
        "first_seen": now,
        "last_seen": now,
        "embedding": None,
        "created_at": now,
        "updated_at": now,
    }
    row.update(overrides)
    return row


def _make_relationship_row(**overrides: Any) -> dict:
    """Create a dict simulating an asyncpg Record for relationship_edges."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": USER_ID,
        "source_entity_id": str(uuid.uuid4()),
        "target_entity_id": str(uuid.uuid4()),
        "relationship_type": "works_with",
        "strength": 1.0,
        "co_mention_count": 1,
        "context_snippets": [],
        "first_seen": now,
        "last_seen": now,
        "created_at": now,
        "updated_at": now,
    }
    row.update(overrides)
    return row


def _make_fact_row(**overrides: Any) -> dict:
    """Create a dict simulating an asyncpg Record for facts."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": USER_ID,
        "fact_type": "attribute",
        "subject_entity_id": None,
        "content": "Alice is an engineer",
        "confidence": 1.0,
        "source_blurt_ids": [],
        "embedding": None,
        "is_active": True,
        "superseded_by": None,
        "first_learned": now,
        "last_confirmed": now,
        "confirmation_count": 1,
        "created_at": now,
        "updated_at": now,
    }
    row.update(overrides)
    return row


def _make_pattern_row(**overrides: Any) -> dict:
    """Create a dict simulating an asyncpg Record for learned_patterns."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": USER_ID,
        "pattern_type": "energy_rhythm",
        "description": "High energy in mornings",
        "parameters": {},
        "confidence": 0.75,
        "observation_count": 5,
        "supporting_evidence": ["ev1"],
        "embedding": None,
        "is_active": True,
        "first_detected": now,
        "last_confirmed": now,
        "created_at": now,
        "updated_at": now,
    }
    row.update(overrides)
    return row


# ── Row conversion tests ─────────────────────────────────────────────


class TestRowConversions:
    """Test data mapping between database rows and entity graph models."""

    def test_row_to_entity_basic(self):
        row = _make_entity_row()
        entity = _row_to_entity(row)
        assert isinstance(entity, EntityNode)
        assert entity.name == "Alice"
        assert entity.entity_type == EntityType.PERSON
        assert entity.mention_count == 1

    def test_row_to_entity_with_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        row = _make_entity_row(embedding=embedding)
        entity = _row_to_entity(row)
        assert entity.embedding == [0.1, 0.2, 0.3]

    def test_row_to_entity_with_aliases(self):
        row = _make_entity_row(aliases=["ally", "al"])
        entity = _row_to_entity(row)
        assert entity.aliases == ["ally", "al"]

    def test_row_to_entity_with_json_attributes(self):
        row = _make_entity_row(attributes=json.dumps({"role": "engineer"}))
        entity = _row_to_entity(row)
        assert entity.attributes == {"role": "engineer"}

    def test_row_to_entity_with_dict_attributes(self):
        row = _make_entity_row(attributes={"role": "engineer"})
        entity = _row_to_entity(row)
        assert entity.attributes == {"role": "engineer"}

    def test_row_to_relationship_basic(self):
        row = _make_relationship_row()
        rel = _row_to_relationship(row)
        assert isinstance(rel, RelationshipEdge)
        assert rel.relationship_type == RelationshipType.WORKS_WITH
        assert rel.strength == 1.0
        assert rel.co_mention_count == 1

    def test_row_to_relationship_with_context(self):
        row = _make_relationship_row(context_snippets=["met at standup", "lunch together"])
        rel = _row_to_relationship(row)
        assert len(rel.context_snippets) == 2
        assert "met at standup" in rel.context_snippets

    def test_row_to_fact_basic(self):
        row = _make_fact_row()
        fact = _row_to_fact(row)
        assert isinstance(fact, Fact)
        assert fact.fact_type == FactType.ATTRIBUTE
        assert fact.content == "Alice is an engineer"
        assert fact.is_active is True

    def test_row_to_fact_with_embedding(self):
        row = _make_fact_row(embedding=[0.5, 0.6, 0.7])
        fact = _row_to_fact(row)
        assert fact.embedding == [0.5, 0.6, 0.7]

    def test_row_to_pattern_basic(self):
        row = _make_pattern_row()
        pattern = _row_to_pattern(row)
        assert isinstance(pattern, LearnedPattern)
        assert pattern.pattern_type == PatternType.ENERGY_RHYTHM
        assert pattern.confidence == 0.75

    def test_row_to_pattern_with_string_parameters(self):
        row = _make_pattern_row(parameters='{"time_of_day": "morning"}')
        pattern = _row_to_pattern(row)
        assert pattern.parameters == {"time_of_day": "morning"}


# ── Entity persistence tests ─────────────────────────────────────────


class TestEntityPersistence:
    """Test entity CRUD operations in PgEntityGraphStore."""

    @pytest.fixture
    def conn(self):
        return _make_mock_conn()

    @pytest.fixture
    def pool(self, conn):
        return _make_pool(conn)

    @pytest.fixture
    def store(self, pool):
        return PgEntityGraphStore(pool, USER_ID, _mock_embedding_provider())

    @pytest.mark.asyncio
    async def test_add_entity_new(self, store, conn):
        """Adding a new entity inserts into entity_nodes table."""
        conn.fetchrow = AsyncMock(return_value=None)  # no existing entity
        entity = await store.add_entity("Alice", EntityType.PERSON)
        assert entity.name == "Alice"
        assert entity.normalized_name == "alice"
        assert entity.entity_type == EntityType.PERSON
        assert entity.mention_count == 1
        # Verify INSERT was called
        conn.execute.assert_awaited_once()
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO entity_nodes" in sql

    @pytest.mark.asyncio
    async def test_add_entity_upsert_increments_mention_count(self, store, conn):
        """Adding an entity with existing name increments mention_count."""
        existing = _make_entity_row(mention_count=3)
        conn.fetchrow = AsyncMock(return_value=existing)
        entity = await store.add_entity("Alice", EntityType.PERSON)
        assert entity.mention_count == 4
        # Verify UPDATE was called
        conn.execute.assert_awaited_once()
        sql = conn.execute.call_args[0][0]
        assert "UPDATE entity_nodes" in sql

    @pytest.mark.asyncio
    async def test_add_entity_merges_aliases(self, store, conn):
        """Re-adding with new aliases merges them into existing aliases."""
        existing = _make_entity_row(aliases=["ally"])
        conn.fetchrow = AsyncMock(return_value=existing)
        entity = await store.add_entity("Alice", EntityType.PERSON, aliases=["al", "ally"])
        assert "ally" in entity.aliases
        assert "al" in entity.aliases

    @pytest.mark.asyncio
    async def test_add_entity_merges_attributes(self, store, conn):
        """Re-adding with new attributes merges into existing attributes."""
        existing = _make_entity_row(attributes={"role": "engineer"})
        conn.fetchrow = AsyncMock(return_value=existing)
        entity = await store.add_entity(
            "Alice", EntityType.PERSON,
            attributes={"team": "backend"},
        )
        assert entity.attributes["role"] == "engineer"
        assert entity.attributes["team"] == "backend"

    @pytest.mark.asyncio
    async def test_get_entity_found(self, store, conn):
        """get_entity returns EntityNode when found."""
        row = _make_entity_row(id="ent-1", name="Bob")
        conn.fetchrow = AsyncMock(return_value=row)
        entity = await store.get_entity("ent-1")
        assert entity is not None
        assert entity.name == "Bob"
        sql = conn.fetchrow.call_args[0][0]
        assert "SELECT * FROM entity_nodes WHERE id = $1" in sql

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, store, conn):
        """get_entity returns None when not found."""
        conn.fetchrow = AsyncMock(return_value=None)
        entity = await store.get_entity("nonexistent")
        assert entity is None

    @pytest.mark.asyncio
    async def test_find_entity_by_name(self, store, conn):
        """find_entity_by_name searches by normalized name first."""
        row = _make_entity_row(name="Alice")
        conn.fetchrow = AsyncMock(return_value=row)
        entity = await store.find_entity_by_name("ALICE")
        assert entity is not None
        assert entity.name == "Alice"
        # Should search with lowercase
        args = conn.fetchrow.call_args[0]
        assert args[2] == "alice"  # normalized

    @pytest.mark.asyncio
    async def test_find_entity_by_alias(self, store, conn):
        """find_entity_by_name falls back to alias search."""
        # First call (normalized_name) returns None, second (alias) returns row
        row = _make_entity_row(name="Alice", aliases=["ally"])
        conn.fetchrow = AsyncMock(side_effect=[None, row])
        entity = await store.find_entity_by_name("Ally")
        assert entity is not None
        assert entity.name == "Alice"
        assert conn.fetchrow.call_count == 2
        # Second query should check aliases
        sql = conn.fetchrow.call_args_list[1][0][0]
        assert "ANY(aliases)" in sql

    @pytest.mark.asyncio
    async def test_get_all_entities_unfiltered(self, store, conn):
        """get_all_entities returns all entities for user."""
        rows = [_make_entity_row(name="Alice"), _make_entity_row(name="Bob")]
        conn.fetch = AsyncMock(return_value=rows)
        entities = await store.get_all_entities()
        assert len(entities) == 2
        sql = conn.fetch.call_args[0][0]
        assert "user_id = $1" in sql
        assert "entity_type" not in sql.split("WHERE")[1].split("ORDER")[0]

    @pytest.mark.asyncio
    async def test_get_all_entities_filtered_by_type(self, store, conn):
        """get_all_entities filters by entity_type when provided."""
        conn.fetch = AsyncMock(return_value=[])
        await store.get_all_entities(entity_type=EntityType.PERSON)
        sql = conn.fetch.call_args[0][0]
        assert "entity_type = $2" in sql

    @pytest.mark.asyncio
    async def test_add_entity_uses_parameterized_sql(self, store, conn):
        """Verify no SQL injection possible via entity name."""
        conn.fetchrow = AsyncMock(return_value=None)
        await store.add_entity("'; DROP TABLE entity_nodes; --", EntityType.PERSON)
        sql = conn.execute.call_args[0][0]
        assert "DROP TABLE" not in sql
        assert "$1" in sql


# ── Relationship persistence & co-mention tests ──────────────────────


class TestRelationshipPersistence:
    """Test relationship CRUD and co-mention strength updates."""

    @pytest.fixture
    def conn(self):
        return _make_mock_conn()

    @pytest.fixture
    def pool(self, conn):
        return _make_pool(conn)

    @pytest.fixture
    def store(self, pool):
        return PgEntityGraphStore(pool, USER_ID, _mock_embedding_provider())

    @pytest.mark.asyncio
    async def test_add_new_relationship(self, store, conn):
        """Creating a new relationship inserts with strength 1.0 and co_mention_count 1."""
        # First fetchrow: no existing relationship, second: source entity lookup
        source_row = _make_entity_row(id="ent-1", user_id=USER_ID)
        conn.fetchrow = AsyncMock(side_effect=[None, source_row])

        edge = await store.add_or_strengthen_relationship(
            "ent-1", "ent-2", RelationshipType.WORKS_WITH, context="on project X",
        )
        assert edge.strength == 1.0
        assert edge.co_mention_count == 1
        assert edge.context_snippets == ["on project X"]
        assert edge.source_entity_id == "ent-1"
        assert edge.target_entity_id == "ent-2"
        # Verify INSERT
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO relationship_edges" in sql

    @pytest.mark.asyncio
    async def test_strengthen_existing_relationship(self, store, conn):
        """Repeated co-mention increases strength and co_mention_count."""
        existing_row = _make_relationship_row(
            source_entity_id="ent-1",
            target_entity_id="ent-2",
            relationship_type="works_with",
            strength=3.0,
            co_mention_count=3,
            context_snippets=["first meeting"],
        )
        conn.fetchrow = AsyncMock(return_value=existing_row)

        edge = await store.add_or_strengthen_relationship(
            "ent-1", "ent-2", RelationshipType.WORKS_WITH,
            context="second meeting",
        )
        assert edge.co_mention_count == 4
        assert edge.strength == 4.0  # 3.0 + 1.0
        assert "second meeting" in edge.context_snippets
        assert "first meeting" in edge.context_snippets
        # Verify UPDATE
        sql = conn.execute.call_args[0][0]
        assert "UPDATE relationship_edges" in sql

    @pytest.mark.asyncio
    async def test_strengthen_caps_at_100(self, store, conn):
        """Strength is capped at 100.0."""
        existing_row = _make_relationship_row(strength=99.5, co_mention_count=99)
        conn.fetchrow = AsyncMock(return_value=existing_row)

        edge = await store.add_or_strengthen_relationship(
            existing_row["source_entity_id"],
            existing_row["target_entity_id"],
            RelationshipType.WORKS_WITH,
        )
        assert edge.strength == 100.0  # min(100.0, 99.5 + 1.0)

    @pytest.mark.asyncio
    async def test_duplicate_context_not_added(self, store, conn):
        """Duplicate context snippets are not added again."""
        existing_row = _make_relationship_row(
            context_snippets=["same context"],
            co_mention_count=2,
            strength=2.0,
        )
        conn.fetchrow = AsyncMock(return_value=existing_row)

        edge = await store.add_or_strengthen_relationship(
            existing_row["source_entity_id"],
            existing_row["target_entity_id"],
            RelationshipType.WORKS_WITH,
            context="same context",
        )
        assert edge.context_snippets.count("same context") == 1

    @pytest.mark.asyncio
    async def test_context_snippets_capped(self, store, conn):
        """Context snippets are limited to MAX_CONTEXT_SNIPPETS."""
        snippets = [f"context-{i}" for i in range(MAX_CONTEXT_SNIPPETS)]
        existing_row = _make_relationship_row(
            context_snippets=snippets,
            co_mention_count=MAX_CONTEXT_SNIPPETS,
            strength=float(MAX_CONTEXT_SNIPPETS),
        )
        conn.fetchrow = AsyncMock(return_value=existing_row)

        edge = await store.add_or_strengthen_relationship(
            existing_row["source_entity_id"],
            existing_row["target_entity_id"],
            RelationshipType.WORKS_WITH,
            context="overflow context",
        )
        assert len(edge.context_snippets) == MAX_CONTEXT_SNIPPETS
        # Newest should be kept, oldest dropped
        assert "overflow context" in edge.context_snippets

    @pytest.mark.asyncio
    async def test_get_entity_relationships(self, store, conn):
        """get_entity_relationships returns all relationships for an entity."""
        rows = [
            _make_relationship_row(source_entity_id="ent-1", target_entity_id="ent-2"),
            _make_relationship_row(source_entity_id="ent-3", target_entity_id="ent-1"),
        ]
        conn.fetch = AsyncMock(return_value=rows)

        rels = await store.get_entity_relationships("ent-1")
        assert len(rels) == 2
        sql = conn.fetch.call_args[0][0]
        # Should match both source and target
        assert "source_entity_id = $1 OR target_entity_id = $1" in sql

    @pytest.mark.asyncio
    async def test_get_connected_entities(self, store, conn):
        """get_connected_entities returns (EntityNode, RelationshipEdge) tuples."""
        rel_rows = [
            _make_relationship_row(
                source_entity_id="ent-1", target_entity_id="ent-2",
                relationship_type="works_with",
            ),
        ]
        entity_row = _make_entity_row(id="ent-2", name="Bob")

        # First acquire: get_entity_relationships -> fetch
        # Second acquire: get each connected entity -> fetchrow
        conn.fetch = AsyncMock(return_value=rel_rows)
        conn.fetchrow = AsyncMock(return_value=entity_row)

        connected = await store.get_connected_entities("ent-1")
        assert len(connected) == 1
        entity, edge = connected[0]
        assert entity.name == "Bob"
        assert edge.relationship_type == RelationshipType.WORKS_WITH

    @pytest.mark.asyncio
    async def test_relationship_uses_parameterized_sql(self, store, conn):
        """Verify no SQL injection in relationship queries."""
        source_row = _make_entity_row(id="ent-1")
        conn.fetchrow = AsyncMock(side_effect=[None, source_row])

        await store.add_or_strengthen_relationship(
            "ent-1", "ent-2", RelationshipType.WORKS_WITH,
            context="'; DROP TABLE relationship_edges; --",
        )
        sql = conn.execute.call_args[0][0]
        assert "DROP TABLE" not in sql
        assert "$1" in sql


# ── Relationship decay tests ─────────────────────────────────────────


class TestRelationshipDecay:
    """Test exponential decay of relationship strength over time."""

    @pytest.fixture
    def conn(self):
        return _make_mock_conn()

    @pytest.fixture
    def pool(self, conn):
        return _make_pool(conn)

    @pytest.fixture
    def store(self, pool):
        return PgEntityGraphStore(pool, USER_ID, _mock_embedding_provider())

    @pytest.mark.asyncio
    async def test_decay_halves_after_half_life(self, store, conn):
        """Strength halves after STRENGTH_DECAY_HALF_LIFE_DAYS days."""
        now = _utcnow()
        old_time = now - timedelta(days=STRENGTH_DECAY_HALF_LIFE_DAYS)
        row = _make_relationship_row(
            strength=10.0,
            last_seen=old_time,
        )
        conn.fetch = AsyncMock(return_value=[row])

        count = await store.decay_relationships(as_of=now)
        assert count == 1
        # The UPDATE call should have approximately half the original strength
        update_args = conn.execute.call_args[0]
        new_strength = update_args[1]
        assert abs(new_strength - 5.0) < 0.1  # ~5.0 after 30 days

    @pytest.mark.asyncio
    async def test_decay_removes_below_threshold(self, store, conn):
        """Very old relationships decay to 0.0 when below threshold."""
        now = _utcnow()
        very_old = now - timedelta(days=365)  # ~12 half-lives
        row = _make_relationship_row(
            strength=1.0,
            last_seen=very_old,
        )
        conn.fetch = AsyncMock(return_value=[row])

        count = await store.decay_relationships(as_of=now)
        assert count == 1
        new_strength = conn.execute.call_args[0][1]
        assert new_strength == 0.0  # Below MIN_RELATIONSHIP_STRENGTH

    @pytest.mark.asyncio
    async def test_no_decay_for_fresh_relationships(self, store, conn):
        """Relationships last seen right now are not decayed."""
        now = _utcnow()
        row = _make_relationship_row(strength=5.0, last_seen=now)
        conn.fetch = AsyncMock(return_value=[row])

        count = await store.decay_relationships(as_of=now)
        assert count == 0
        conn.execute.assert_not_awaited()


# ── Fact persistence tests ────────────────────────────────────────────


class TestFactPersistence:
    """Test fact CRUD operations."""

    @pytest.fixture
    def conn(self):
        return _make_mock_conn()

    @pytest.fixture
    def pool(self, conn):
        return _make_pool(conn)

    @pytest.fixture
    def store(self, pool):
        return PgEntityGraphStore(pool, USER_ID, _mock_embedding_provider())

    @pytest.mark.asyncio
    async def test_add_fact(self, store, conn):
        """add_fact inserts into facts table with correct parameters."""
        fact = await store.add_fact(
            "Alice is an engineer",
            FactType.ATTRIBUTE,
            subject_entity_id="ent-1",
            source_blurt_id="blurt-1",
        )
        assert fact.content == "Alice is an engineer"
        assert fact.fact_type == FactType.ATTRIBUTE
        assert fact.subject_entity_id == "ent-1"
        assert "blurt-1" in fact.source_blurt_ids
        assert fact.is_active is True
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO facts" in sql

    @pytest.mark.asyncio
    async def test_supersede_fact(self, store, conn):
        """supersede_fact marks old fact inactive and creates new one."""
        old_row = _make_fact_row(
            id="fact-old",
            content="Alice is a junior engineer",
            fact_type="attribute",
            subject_entity_id="ent-1",
        )
        conn.fetchrow = AsyncMock(return_value=old_row)

        new_fact = await store.supersede_fact(
            "fact-old", "Alice is a senior engineer",
        )
        assert new_fact is not None
        assert new_fact.content == "Alice is a senior engineer"
        # Verify the old fact was marked inactive
        update_calls = [
            c for c in conn.execute.call_args_list
            if "UPDATE facts" in c[0][0]
        ]
        assert len(update_calls) >= 1
        update_sql = update_calls[0][0][0]
        assert "is_active = FALSE" in update_sql
        assert "superseded_by = $1" in update_sql

    @pytest.mark.asyncio
    async def test_get_entity_facts_active_only(self, store, conn):
        """get_entity_facts returns only active facts by default."""
        rows = [_make_fact_row(is_active=True)]
        conn.fetch = AsyncMock(return_value=rows)
        facts = await store.get_entity_facts("ent-1")
        assert len(facts) == 1
        sql = conn.fetch.call_args[0][0]
        assert "is_active = TRUE" in sql

    @pytest.mark.asyncio
    async def test_get_entity_facts_include_inactive(self, store, conn):
        """get_entity_facts includes inactive facts when requested."""
        rows = [
            _make_fact_row(is_active=True),
            _make_fact_row(is_active=False),
        ]
        conn.fetch = AsyncMock(return_value=rows)
        facts = await store.get_entity_facts("ent-1", active_only=False)
        assert len(facts) == 2
        sql = conn.fetch.call_args[0][0]
        assert "is_active = TRUE" not in sql

    @pytest.mark.asyncio
    async def test_get_all_facts_by_type(self, store, conn):
        """get_all_facts filters by fact_type."""
        conn.fetch = AsyncMock(return_value=[])
        await store.get_all_facts(fact_type=FactType.PREFERENCE)
        sql = conn.fetch.call_args[0][0]
        assert "fact_type = $2" in sql


# ── Pattern persistence tests ─────────────────────────────────────────


class TestPatternPersistence:
    """Test pattern CRUD operations."""

    @pytest.fixture
    def conn(self):
        return _make_mock_conn()

    @pytest.fixture
    def pool(self, conn):
        return _make_pool(conn)

    @pytest.fixture
    def store(self, pool):
        return PgEntityGraphStore(pool, USER_ID, _mock_embedding_provider())

    @pytest.mark.asyncio
    async def test_add_pattern(self, store, conn):
        """add_pattern inserts into learned_patterns table."""
        pattern = await store.add_pattern(
            PatternType.ENERGY_RHYTHM,
            "High energy in mornings",
            parameters={"time": "morning"},
            confidence=0.8,
        )
        assert pattern.description == "High energy in mornings"
        assert pattern.pattern_type == PatternType.ENERGY_RHYTHM
        assert pattern.is_active is True  # confidence 0.8 >= 0.7 threshold
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO learned_patterns" in sql

    @pytest.mark.asyncio
    async def test_add_pattern_below_threshold_inactive(self, store, conn):
        """Patterns with low confidence are marked inactive."""
        pattern = await store.add_pattern(
            PatternType.MOOD_CYCLE,
            "Mood dips on Mondays",
            confidence=0.3,
        )
        assert pattern.is_active is False


# ── Co-mention integration scenario ──────────────────────────────────


class TestCoMentionIntegration:
    """End-to-end test of co-mention strength updates across repeated mentions."""

    @pytest.fixture
    def conn(self):
        return _make_mock_conn()

    @pytest.fixture
    def pool(self, conn):
        return _make_pool(conn)

    @pytest.fixture
    def store(self, pool):
        return PgEntityGraphStore(pool, USER_ID, _mock_embedding_provider())

    @pytest.mark.asyncio
    async def test_repeated_mentions_accumulate_strength(self, store, conn):
        """Three co-mentions build strength from 1.0 → 2.0 → 3.0."""
        _ent_a = _make_entity_row(id="ent-a", name="Alice")
        _ent_b = _make_entity_row(id="ent-b", name="Bob")
        source_row = _make_entity_row(id="ent-a", user_id=USER_ID)

        # First mention: no existing relationship
        conn.fetchrow = AsyncMock(side_effect=[None, source_row])
        edge1 = await store.add_or_strengthen_relationship(
            "ent-a", "ent-b", RelationshipType.WORKS_WITH,
            context="standup meeting",
        )
        assert edge1.strength == 1.0
        assert edge1.co_mention_count == 1

        # Second mention: existing with strength 1.0
        existing_after_first = _make_relationship_row(
            id=edge1.id,
            source_entity_id="ent-a",
            target_entity_id="ent-b",
            relationship_type="works_with",
            strength=1.0,
            co_mention_count=1,
            context_snippets=["standup meeting"],
        )
        conn.fetchrow = AsyncMock(return_value=existing_after_first)
        edge2 = await store.add_or_strengthen_relationship(
            "ent-a", "ent-b", RelationshipType.WORKS_WITH,
            context="code review",
        )
        assert edge2.strength == 2.0
        assert edge2.co_mention_count == 2
        assert "code review" in edge2.context_snippets

        # Third mention: existing with strength 2.0
        existing_after_second = _make_relationship_row(
            id=edge1.id,
            source_entity_id="ent-a",
            target_entity_id="ent-b",
            relationship_type="works_with",
            strength=2.0,
            co_mention_count=2,
            context_snippets=["standup meeting", "code review"],
        )
        conn.fetchrow = AsyncMock(return_value=existing_after_second)
        edge3 = await store.add_or_strengthen_relationship(
            "ent-a", "ent-b", RelationshipType.WORKS_WITH,
            context="pair programming",
        )
        assert edge3.strength == 3.0
        assert edge3.co_mention_count == 3
        assert len(edge3.context_snippets) == 3

    @pytest.mark.asyncio
    async def test_entity_mention_count_grows_on_repeated_add(self, store, conn):
        """Entity mention_count increments on each add_entity call."""
        # First add: no existing
        conn.fetchrow = AsyncMock(return_value=None)
        e1 = await store.add_entity("Alice", EntityType.PERSON)
        assert e1.mention_count == 1

        # Second add: existing with mention_count=1
        existing = _make_entity_row(
            id=e1.id, name="Alice", mention_count=1,
        )
        conn.fetchrow = AsyncMock(return_value=existing)
        e2 = await store.add_entity("Alice", EntityType.PERSON)
        assert e2.mention_count == 2

        # Third add: existing with mention_count=2
        existing2 = _make_entity_row(
            id=e1.id, name="Alice", mention_count=2,
        )
        conn.fetchrow = AsyncMock(return_value=existing2)
        e3 = await store.add_entity("Alice", EntityType.PERSON)
        assert e3.mention_count == 3


# ── SQL safety tests ─────────────────────────────────────────────────


class TestSQLSafety:
    """Verify all SQL uses parameterized queries, zero string interpolation."""

    @pytest.fixture
    def conn(self):
        return _make_mock_conn()

    @pytest.fixture
    def pool(self, conn):
        return _make_pool(conn)

    @pytest.fixture
    def store(self, pool):
        return PgEntityGraphStore(pool, USER_ID, _mock_embedding_provider())

    @pytest.mark.asyncio
    async def test_entity_name_injection(self, store, conn):
        conn.fetchrow = AsyncMock(return_value=None)
        await store.add_entity("'; DROP TABLE entity_nodes; --", EntityType.PERSON)
        sql = conn.execute.call_args[0][0]
        assert "DROP TABLE" not in sql

    @pytest.mark.asyncio
    async def test_fact_content_injection(self, store, conn):
        await store.add_fact(
            "'; DELETE FROM facts; --",
            FactType.ATTRIBUTE,
        )
        sql = conn.execute.call_args[0][0]
        assert "DELETE FROM" not in sql

    @pytest.mark.asyncio
    async def test_relationship_context_injection(self, store, conn):
        source_row = _make_entity_row(id="ent-1")
        conn.fetchrow = AsyncMock(side_effect=[None, source_row])
        await store.add_or_strengthen_relationship(
            "ent-1", "ent-2", RelationshipType.WORKS_WITH,
            context="'; DROP TABLE relationship_edges; --",
        )
        sql = conn.execute.call_args[0][0]
        assert "DROP TABLE" not in sql

    @pytest.mark.asyncio
    async def test_find_entity_injection(self, store, conn):
        conn.fetchrow = AsyncMock(return_value=None)
        await store.find_entity_by_name("'; DROP TABLE entity_nodes; --")
        for call in conn.fetchrow.call_args_list:
            sql = call[0][0]
            assert "DROP TABLE" not in sql
