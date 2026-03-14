"""Tests for pgvector cosine distance semantic similarity search.

Verifies that:
- The <=> cosine distance operator is used in all search SQL
- Cosine distance is correctly converted to similarity (1 - distance)
- Results are sorted by similarity descending
- min_similarity threshold filtering works via max_distance
- All SQL uses parameterized queries (no string interpolation)
- Entity, fact, and pattern search all use pgvector
- Episode embedding search uses pgvector
- search_neighborhood expands via relationships after vector seed
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.models.entities import (
    EntityType,
)
from blurt.persistence.pg_entity_graph_store import PgEntityGraphStore
from blurt.persistence.pg_episodic_store import PgEpisodicStore


# ── Helpers ──────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_pool_with_conn() -> tuple[MagicMock, AsyncMock]:
    """Create a mock asyncpg pool with acquire() context manager."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=None)
    conn.execute = AsyncMock()

    # Make pool.acquire() return an async context manager yielding conn
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx

    return pool, conn


def _mock_embedding_provider(dimension: int = 768) -> MagicMock:
    """Create a mock EmbeddingProvider that returns deterministic embeddings."""
    provider = MagicMock()
    provider.dimension = dimension
    # Return a simple normalized vector based on call count
    call_count = [0]

    async def mock_embed(text: str) -> list[float]:
        call_count[0] += 1
        # Deterministic embedding: set one dimension based on hash
        vec = [0.0] * dimension
        idx = hash(text) % dimension
        vec[idx] = 1.0
        return vec

    provider.embed = AsyncMock(side_effect=mock_embed)
    return provider


_EMBEDDING_768 = [0.1] * 768  # reusable embedding list


def _make_entity_row(**overrides) -> dict:
    """Create a dict simulating an asyncpg Record for entity_nodes."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": "__default__",
        "name": "TestEntity",
        "normalized_name": "testentity",
        "entity_type": "person",
        "aliases": [],
        "attributes": {},
        "mention_count": 1,
        "first_seen": now,
        "last_seen": now,
        "embedding": list(_EMBEDDING_768),
        "created_at": now,
        "updated_at": now,
        "distance": 0.2,  # cosine distance from query
    }
    row.update(overrides)
    return row


def _make_fact_row(**overrides) -> dict:
    """Create a dict simulating an asyncpg Record for facts."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": "__default__",
        "fact_type": "preference",
        "subject_entity_id": None,
        "content": "Likes coffee",
        "confidence": 0.9,
        "source_blurt_ids": [],
        "embedding": list(_EMBEDDING_768),
        "is_active": True,
        "superseded_by": None,
        "first_learned": now,
        "last_confirmed": now,
        "confirmation_count": 1,
        "created_at": now,
        "updated_at": now,
        "distance": 0.3,
    }
    row.update(overrides)
    return row


def _make_pattern_row(**overrides) -> dict:
    """Create a dict simulating an asyncpg Record for learned_patterns."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": "__default__",
        "pattern_type": "energy_rhythm",
        "description": "High energy in mornings",
        "parameters": '{"time_of_day": "morning"}',
        "confidence": 0.75,
        "observation_count": 5,
        "supporting_evidence": ["ev1"],
        "embedding": list(_EMBEDDING_768),
        "is_active": True,
        "first_detected": now,
        "last_confirmed": now,
        "created_at": now,
        "updated_at": now,
        "distance": 0.4,
    }
    row.update(overrides)
    return row


def _make_episode_row(**overrides) -> dict:
    """Create a dict simulating an asyncpg Record for episodes."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": "__default__",
        "timestamp": now,
        "raw_text": "Test episode",
        "modality": "voice",
        "embedding": list(_EMBEDDING_768),
        "intent": "task",
        "intent_confidence": 0.9,
        "emotion_primary": "trust",
        "emotion_intensity": 0.5,
        "emotion_valence": 0.0,
        "emotion_arousal": 0.2,
        "emotion_secondary": None,
        "behavioral_signal": "none",
        "surfaced_task_id": None,
        "context_time_of_day": "morning",
        "context_day_of_week": "monday",
        "context_session_id": None,
        "context_preceding_episode_id": None,
        "context_active_task_id": None,
        "is_compressed": False,
        "compressed_into_id": None,
        "source_working_id": None,
        "created_at": now,
        "distance": 0.15,
    }
    row.update(overrides)
    return row


# ── Entity Graph Store: search() ────────────────────────────────────


class TestPgEntityGraphStoreSearch:
    """Test PgEntityGraphStore.search() uses pgvector cosine distance."""

    @pytest.fixture
    def setup(self):
        pool, conn = _make_pool_with_conn()
        embeddings = _mock_embedding_provider()
        store = PgEntityGraphStore(pool, "__default__", embeddings)
        return store, conn, embeddings

    @pytest.mark.asyncio
    async def test_search_uses_cosine_distance_operator(self, setup):
        """Verify search SQL uses <=> pgvector cosine distance operator."""
        store, conn, _ = setup
        conn.fetch = AsyncMock(return_value=[])

        await store.search("test query", top_k=5, min_similarity=0.3)

        # Should have called fetch for entities, facts, and patterns
        assert conn.fetch.await_count == 3

        for call in conn.fetch.call_args_list:
            sql = call[0][0]
            # Each query MUST use the <=> cosine distance operator
            assert "<=>" in sql, f"Missing <=> operator in SQL: {sql}"
            # Must use parameterized $1::vector cast
            assert "$1::vector" in sql
            # Must order by distance ascending (closest = most similar)
            assert "ORDER BY distance ASC" in sql

    @pytest.mark.asyncio
    async def test_search_converts_distance_to_similarity(self, setup):
        """Verify similarity = 1 - cosine_distance."""
        store, conn, _ = setup
        entity_row = _make_entity_row(distance=0.25)
        # Return entities only, empty for facts and patterns
        conn.fetch = AsyncMock(side_effect=[[entity_row], [], []])

        results = await store.search("test", min_similarity=0.0)

        assert len(results) == 1
        # similarity = 1.0 - 0.25 = 0.75
        assert results[0].similarity_score == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_search_min_similarity_converted_to_max_distance(self, setup):
        """min_similarity=0.7 should become max_distance=0.3 in SQL."""
        store, conn, _ = setup
        conn.fetch = AsyncMock(return_value=[])

        await store.search("test", min_similarity=0.7)

        for call in conn.fetch.call_args_list:
            args = call[0]
            # $3 is the max_distance parameter
            max_distance = args[3]
            assert max_distance == pytest.approx(0.3), (
                f"Expected max_distance=0.3, got {max_distance}"
            )

    @pytest.mark.asyncio
    async def test_search_returns_all_item_types(self, setup):
        """search() queries entities, facts, and patterns."""
        store, conn, _ = setup

        entity = _make_entity_row(distance=0.1, name="Alice")
        fact = _make_fact_row(distance=0.2, content="Likes Python")
        pattern = _make_pattern_row(distance=0.3, description="Morning person")

        conn.fetch = AsyncMock(side_effect=[[entity], [fact], [pattern]])

        results = await store.search("test", min_similarity=0.0)

        assert len(results) == 3
        types = {r.item_type for r in results}
        assert types == {"entity", "fact", "pattern"}

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_similarity_descending(self, setup):
        """Results from all tables are merged and sorted by similarity desc."""
        store, conn, _ = setup

        entity = _make_entity_row(distance=0.5, name="Far entity")
        fact = _make_fact_row(distance=0.1, content="Close fact")
        pattern = _make_pattern_row(distance=0.3, description="Mid pattern")

        conn.fetch = AsyncMock(side_effect=[[entity], [fact], [pattern]])

        results = await store.search("test", min_similarity=0.0)

        similarities = [r.similarity_score for r in results]
        assert similarities == sorted(similarities, reverse=True)
        # Fact (0.9) > Pattern (0.7) > Entity (0.5)
        assert results[0].item_type == "fact"
        assert results[0].similarity_score == pytest.approx(0.9)
        assert results[1].item_type == "pattern"
        assert results[2].item_type == "entity"

    @pytest.mark.asyncio
    async def test_search_respects_item_types_filter(self, setup):
        """Only query specified item_types tables."""
        store, conn, _ = setup
        conn.fetch = AsyncMock(return_value=[])

        await store.search("test", item_types=["entity"])

        # Should only call fetch once (for entities only)
        assert conn.fetch.await_count == 1
        sql = conn.fetch.call_args[0][0]
        assert "entity_nodes" in sql

    @pytest.mark.asyncio
    async def test_search_top_k_limits_per_table_and_total(self, setup):
        """top_k is passed as LIMIT per table; total truncated to top_k."""
        store, conn, _ = setup
        conn.fetch = AsyncMock(return_value=[])

        await store.search("test", top_k=3, min_similarity=0.0)

        for call in conn.fetch.call_args_list:
            args = call[0]
            # Last arg is the limit ($4 = top_k)
            limit = args[-1]
            assert limit == 3

    @pytest.mark.asyncio
    async def test_search_embedding_failure_returns_empty(self, setup):
        """If embedding generation fails, search returns empty list."""
        store, conn, embeddings = setup
        embeddings.embed = AsyncMock(side_effect=RuntimeError("API error"))

        results = await store.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_no_string_interpolation_in_sql(self, setup):
        """Malicious query text should not appear in SQL."""
        store, conn, _ = setup
        conn.fetch = AsyncMock(return_value=[])

        await store.search("'; DROP TABLE entity_nodes; --")

        for call in conn.fetch.call_args_list:
            sql = call[0][0]
            assert "DROP TABLE" not in sql
            # All parameters use $N placeholders
            assert "$1" in sql
            assert "$2" in sql


# ── Entity Graph Store: search_similar_entities() ────────────────────


class TestSearchSimilarEntities:
    """Test entity-to-entity similarity search via pgvector."""

    @pytest.fixture
    def setup(self):
        pool, conn = _make_pool_with_conn()
        embeddings = _mock_embedding_provider()
        store = PgEntityGraphStore(pool, "__default__", embeddings)
        return store, conn, embeddings

    @pytest.mark.asyncio
    async def test_search_similar_entities_uses_cosine_operator(self, setup):
        """search_similar_entities uses <=> for entity-to-entity similarity."""
        store, conn, _ = setup

        source_entity = _make_entity_row(id="ent-1")
        # get_entity uses fetchrow, search uses fetch
        conn.fetchrow = AsyncMock(return_value=source_entity)
        conn.fetch = AsyncMock(return_value=[])

        await store.search_similar_entities("ent-1", top_k=5)

        # The fetch call should use <=> operator
        sql = conn.fetch.call_args[0][0]
        assert "<=>" in sql
        assert "$1::vector" in sql
        assert "ORDER BY distance ASC" in sql
        assert "entity_nodes" in sql

    @pytest.mark.asyncio
    async def test_search_similar_entities_excludes_source(self, setup):
        """Should not return the source entity itself."""
        store, conn, _ = setup

        source_entity = _make_entity_row(id="ent-1")
        conn.fetchrow = AsyncMock(return_value=source_entity)
        conn.fetch = AsyncMock(return_value=[])

        await store.search_similar_entities("ent-1")

        sql = conn.fetch.call_args[0][0]
        assert "id != $3" in sql

    @pytest.mark.asyncio
    async def test_search_similar_entities_type_filter(self, setup):
        """Entity type filter is passed as parameter."""
        store, conn, _ = setup

        source_entity = _make_entity_row(id="ent-1")
        conn.fetchrow = AsyncMock(return_value=source_entity)
        conn.fetch = AsyncMock(return_value=[])

        await store.search_similar_entities(
            "ent-1", entity_type=EntityType.PERSON
        )

        sql = conn.fetch.call_args[0][0]
        assert "entity_type = $4" in sql
        args = conn.fetch.call_args[0]
        assert "person" in args  # EntityType.PERSON.value

    @pytest.mark.asyncio
    async def test_search_similar_entities_no_embedding_returns_empty(self, setup):
        """If source entity has no embedding, return empty."""
        store, conn, _ = setup

        source_entity = _make_entity_row(id="ent-1", embedding=None)
        conn.fetchrow = AsyncMock(return_value=source_entity)

        results = await store.search_similar_entities("ent-1")
        assert results == []


# ── Entity Graph Store: search_entities_by_query() ───────────────────


class TestSearchEntitiesByQuery:
    """Test text query → entity search via pgvector."""

    @pytest.fixture
    def setup(self):
        pool, conn = _make_pool_with_conn()
        embeddings = _mock_embedding_provider()
        store = PgEntityGraphStore(pool, "__default__", embeddings)
        return store, conn, embeddings

    @pytest.mark.asyncio
    async def test_search_by_query_uses_cosine_operator(self, setup):
        """search_entities_by_query uses <=> for text-to-entity similarity."""
        store, conn, _ = setup
        conn.fetch = AsyncMock(return_value=[])

        await store.search_entities_by_query("Python developer")

        sql = conn.fetch.call_args[0][0]
        assert "<=>" in sql
        assert "$1::vector" in sql
        assert "entity_nodes" in sql

    @pytest.mark.asyncio
    async def test_search_by_query_with_entity_type_filter(self, setup):
        """Entity type filter adds parameterized condition."""
        store, conn, _ = setup
        conn.fetch = AsyncMock(return_value=[])

        await store.search_entities_by_query(
            "Python", entity_type=EntityType.PROJECT
        )

        sql = conn.fetch.call_args[0][0]
        assert "entity_type = $4" in sql
        args = conn.fetch.call_args[0]
        assert "project" in args

    @pytest.mark.asyncio
    async def test_search_by_query_with_min_mentions(self, setup):
        """min_mentions filter adds parameterized condition."""
        store, conn, _ = setup
        conn.fetch = AsyncMock(return_value=[])

        await store.search_entities_by_query("Python", min_mentions=3)

        sql = conn.fetch.call_args[0][0]
        assert "mention_count >= $4" in sql

    @pytest.mark.asyncio
    async def test_search_by_query_returns_similarity(self, setup):
        """Results include similarity computed as 1 - distance."""
        store, conn, _ = setup
        row = _make_entity_row(distance=0.15)
        conn.fetch = AsyncMock(return_value=[row])

        results = await store.search_entities_by_query("test", min_similarity=0.0)

        assert len(results) == 1
        assert results[0].similarity_score == pytest.approx(0.85)
        assert results[0].item_type == "entity"


# ── Entity Graph Store: search_neighborhood() ───────────────────────


class TestSearchNeighborhood:
    """Test graph-aware neighborhood search seeded by pgvector similarity."""

    @pytest.fixture
    def setup(self):
        pool, conn = _make_pool_with_conn()
        embeddings = _mock_embedding_provider()
        store = PgEntityGraphStore(pool, "__default__", embeddings)
        return store, conn, embeddings

    @pytest.mark.asyncio
    async def test_neighborhood_starts_with_vector_search(self, setup):
        """search_neighborhood seeds from pgvector entity search."""
        store, conn, _ = setup
        # Entity search returns empty → no expansion
        conn.fetch = AsyncMock(return_value=[])

        _results = await store.search_neighborhood("test query")

        # Should have searched entities (at minimum)
        assert conn.fetch.await_count >= 1
        first_sql = conn.fetch.call_args_list[0][0][0]
        assert "<=>" in first_sql

    @pytest.mark.asyncio
    async def test_neighborhood_no_seeds_returns_empty(self, setup):
        """If vector search finds no seeds, neighborhood returns empty."""
        store, conn, _ = setup
        # All table queries return empty
        conn.fetch = AsyncMock(return_value=[])

        results = await store.search_neighborhood("nonexistent")
        assert results == []


# ── Episodic Store: semantic_search() ────────────────────────────


class TestPgEpisodicStoreSearch:
    """Test PgEpisodicStore.semantic_search() uses pgvector cosine distance."""

    @pytest.fixture
    def setup(self):
        pool, conn = _make_pool_with_conn()
        store = PgEpisodicStore(pool)
        return store, conn

    @pytest.mark.asyncio
    async def test_episode_search_uses_cosine_distance_operator(self, setup):
        """Verify episode search SQL uses <=> pgvector operator."""
        store, conn = setup
        conn.fetch = AsyncMock(return_value=[])

        query_vec = [0.1] * 768
        await store.semantic_search("user-1", query_vec, limit=5)

        sql = conn.fetch.call_args[0][0]
        assert "<=>" in sql
        assert "$1::vector" in sql
        assert "ORDER BY distance ASC" in sql
        assert "episodes" in sql

    @pytest.mark.asyncio
    async def test_episode_search_min_similarity_to_max_distance(self, setup):
        """min_similarity=0.7 → max_distance=0.3."""
        store, conn = setup
        conn.fetch = AsyncMock(return_value=[])

        query_vec = [0.1] * 768
        await store.semantic_search("user-1", query_vec, min_similarity=0.7)

        args = conn.fetch.call_args[0]
        max_distance = args[3]  # $3 = max_distance
        assert max_distance == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_episode_search_returns_similarity_score(self, setup):
        """Episode search converts cosine distance to similarity."""
        store, conn = setup

        episode_row = _make_episode_row(distance=0.2)
        # First fetch: episode rows, second fetch: entity rows per episode
        conn.fetch = AsyncMock(side_effect=[[episode_row], []])

        query_vec = [0.1] * 768
        results = await store.semantic_search(
            "user-1", query_vec, min_similarity=0.0
        )

        assert len(results) == 1
        episode, similarity = results[0]
        assert similarity == pytest.approx(0.8)  # 1.0 - 0.2

    @pytest.mark.asyncio
    async def test_episode_search_filters_compressed_episodes(self, setup):
        """Only non-compressed episodes are searched."""
        store, conn = setup
        conn.fetch = AsyncMock(return_value=[])

        query_vec = [0.1] * 768
        await store.semantic_search("user-1", query_vec)

        sql = conn.fetch.call_args[0][0]
        assert "is_compressed = FALSE" in sql

    @pytest.mark.asyncio
    async def test_episode_search_requires_non_null_embedding(self, setup):
        """Only episodes with embeddings are searched."""
        store, conn = setup
        conn.fetch = AsyncMock(return_value=[])

        query_vec = [0.1] * 768
        await store.semantic_search("user-1", query_vec)

        sql = conn.fetch.call_args[0][0]
        assert "embedding IS NOT NULL" in sql

    @pytest.mark.asyncio
    async def test_episode_search_no_string_interpolation(self, setup):
        """Malicious embedding values don't appear in SQL."""
        store, conn = setup
        conn.fetch = AsyncMock(return_value=[])

        query_vec = [0.1] * 768
        await store.semantic_search(
            "'; DROP TABLE episodes; --", query_vec
        )

        sql = conn.fetch.call_args[0][0]
        assert "DROP TABLE" not in sql
        assert "$1" in sql
        assert "$2" in sql


# ── Cross-cutting: SQL Parameterization ─────────────────────────────


class TestCosineSearchParameterization:
    """Verify all cosine search queries use parameterized SQL."""

    @pytest.mark.asyncio
    async def test_entity_graph_search_all_params(self):
        """All search methods pass vectors and thresholds as parameters."""
        pool, conn = _make_pool_with_conn()
        embeddings = _mock_embedding_provider()
        store = PgEntityGraphStore(pool, "__default__", embeddings)
        conn.fetch = AsyncMock(return_value=[])

        await store.search("test", top_k=5, min_similarity=0.5)

        for call in conn.fetch.call_args_list:
            sql = call[0][0]
            args = call[0][1:]
            # SQL should have $N placeholders for all variable data
            assert "$1" in sql  # embedding vector
            assert "$2" in sql  # user_id
            assert "$3" in sql  # max_distance
            assert "$4" in sql  # limit
            # Actual values are in args, not in SQL
            assert len(args) >= 4

    @pytest.mark.asyncio
    async def test_episodic_search_all_params(self):
        """Episode search passes all values as parameters."""
        pool, conn = _make_pool_with_conn()
        store = PgEpisodicStore(pool)
        conn.fetch = AsyncMock(return_value=[])

        await store.semantic_search("user-1", [0.1] * 768, limit=10, min_similarity=0.5)

        sql = conn.fetch.call_args[0][0]
        args = conn.fetch.call_args[0][1:]
        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql
        assert "$4" in sql
        assert len(args) >= 4


# ── HNSW Index Verification ─────────────────────────────────────────


class TestHNSWIndexDDL:
    """Verify schema migrations create HNSW indexes for pgvector."""

    @pytest.mark.asyncio
    async def test_schema_creates_hnsw_indexes(self):
        """run_schema_migrations creates HNSW indexes with vector_cosine_ops."""
        from blurt.persistence.database import run_schema_migrations

        pool = MagicMock()
        conn = AsyncMock()
        conn.execute = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        await run_schema_migrations(pool)

        executed_sqls = [call[0][0] for call in conn.execute.call_args_list]
        all_sql = "\n".join(executed_sqls)

        # Verify HNSW indexes exist for all embedding columns
        assert "vector_cosine_ops" in all_sql

        # Check specific table indexes
        assert "idx_entity_nodes_embedding" in all_sql
        assert "idx_facts_embedding" in all_sql
        assert "idx_patterns_embedding" in all_sql
        assert "idx_episodes_embedding" in all_sql

        # All should use HNSW
        for line in executed_sqls:
            if "vector_cosine_ops" in line:
                assert "hnsw" in line.lower(), (
                    f"Expected HNSW index, got: {line}"
                )
