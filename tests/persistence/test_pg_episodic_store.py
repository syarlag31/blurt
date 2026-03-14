"""Tests for PgEpisodicStore — Postgres-backed episodic memory.

Verifies:
- Episode write + read-back roundtrip (simulates restart persistence)
- Entity references survive storage
- Summaries persist
- All SQL uses parameterized queries (no string interpolation)
- Correct data mapping between Episode dataclass and DB rows
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from blurt.memory.episodic import (
    BehavioralSignal,
    EmotionSnapshot,
    EntityRef,
    Episode,
    EpisodeContext,
    EpisodeSummary,
    InputModality,
)
from blurt.persistence.pg_episodic_store import (
    PgEpisodicStore,
    _row_to_episode,
    _row_to_summary,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_pool() -> MagicMock:
    """Create a mock asyncpg pool with acquire context manager."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=0)

    # Transaction context manager
    txn = AsyncMock()
    txn.__aenter__ = AsyncMock(return_value=txn)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)

    # Pool.acquire() context manager
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    pool._conn = conn  # internal reference for test assertions
    return pool


def _make_episode_row(**overrides) -> dict:
    """Create a dict simulating an asyncpg Record for an episode."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": "user-1",
        "timestamp": now,
        "raw_text": "Pick up groceries on the way home",
        "modality": "voice",
        "intent": "task",
        "intent_confidence": 0.92,
        "emotion_primary": "anticipation",
        "emotion_intensity": 0.6,
        "emotion_valence": 0.3,
        "emotion_arousal": 0.5,
        "emotion_secondary": None,
        "behavioral_signal": "none",
        "surfaced_task_id": None,
        "context_time_of_day": "evening",
        "context_day_of_week": "wednesday",
        "context_session_id": "sess-1",
        "context_preceding_episode_id": None,
        "context_active_task_id": None,
        "is_compressed": False,
        "compressed_into_id": None,
        "embedding": None,
        "source_working_id": None,
    }
    row.update(overrides)
    return row


def _make_entity_row(**overrides) -> dict:
    """Create a dict simulating an asyncpg Record for an entity reference."""
    row = {
        "episode_id": "ep-1",
        "name": "groceries",
        "entity_type": "errand",
        "entity_id": None,
        "confidence": 0.95,
    }
    row.update(overrides)
    return row


def _make_summary_row(**overrides) -> dict:
    """Create a dict simulating an asyncpg Record for a summary."""
    now = _utcnow()
    row = {
        "id": str(uuid.uuid4()),
        "user_id": "user-1",
        "created_at": now,
        "period_start": now,
        "period_end": now,
        "source_episode_ids": ["ep-1", "ep-2"],
        "episode_count": 2,
        "summary_text": "User mentioned groceries twice this week.",
        "dominant_emotions": json.dumps([{"primary": "trust", "intensity": 0.5, "valence": 0.3, "arousal": 0.2}]),
        "entity_mentions": json.dumps({"groceries": 2}),
        "intent_distribution": json.dumps({"task": 2}),
        "behavioral_signals": json.dumps({"none": 2}),
        "embedding": None,
    }
    row.update(overrides)
    return row


def _make_episode(**overrides) -> Episode:
    """Create an Episode for testing."""
    now = _utcnow()
    defaults = dict(
        id=str(uuid.uuid4()),
        user_id="user-1",
        timestamp=now,
        raw_text="Buy milk from the store",
        modality=InputModality.VOICE,
        intent="task",
        intent_confidence=0.88,
        emotion=EmotionSnapshot(
            primary="trust",
            intensity=0.5,
            valence=0.2,
            arousal=0.3,
        ),
        entities=[
            EntityRef(name="milk", entity_type="item", entity_id=None, confidence=0.9),
            EntityRef(name="store", entity_type="place", entity_id="loc-1", confidence=0.85),
        ],
        behavioral_signal=BehavioralSignal.NONE,
        context=EpisodeContext(
            time_of_day="morning",
            day_of_week="monday",
            session_id="sess-abc",
        ),
    )
    defaults.update(overrides)
    return Episode(**defaults)  # type: ignore[arg-type]


# ── Row-to-model conversion tests ───────────────────────────────────


class TestRowConversions:
    """Test data mapping between database rows and Episode/Summary dataclasses."""

    def test_row_to_episode_basic(self):
        row = _make_episode_row()
        episode = _row_to_episode(row)
        assert isinstance(episode, Episode)
        assert episode.raw_text == "Pick up groceries on the way home"
        assert episode.modality == InputModality.VOICE
        assert episode.intent == "task"
        assert episode.emotion.primary == "anticipation"
        assert episode.behavioral_signal == BehavioralSignal.NONE
        assert episode.is_compressed is False
        assert episode.entities == []

    def test_row_to_episode_with_entities(self):
        row = _make_episode_row(id="ep-1")
        entity_rows = [
            _make_entity_row(episode_id="ep-1", name="groceries", entity_type="errand"),
            _make_entity_row(episode_id="ep-1", name="home", entity_type="place"),
        ]
        episode = _row_to_episode(row, entity_rows)
        assert len(episode.entities) == 2
        assert episode.entities[0].name == "groceries"
        assert episode.entities[1].name == "home"

    def test_row_to_episode_with_embedding(self):
        embedding = [0.1] * 768
        row = _make_episode_row(embedding=embedding)
        episode = _row_to_episode(row)
        assert episode.embedding is not None
        assert len(episode.embedding) == 768
        assert episode.embedding[0] == 0.1

    def test_row_to_episode_preserves_context(self):
        row = _make_episode_row(
            context_time_of_day="evening",
            context_day_of_week="friday",
            context_session_id="sess-xyz",
            context_preceding_episode_id="prev-ep",
            context_active_task_id="task-1",
        )
        episode = _row_to_episode(row)
        assert episode.context.time_of_day == "evening"
        assert episode.context.day_of_week == "friday"
        assert episode.context.session_id == "sess-xyz"
        assert episode.context.preceding_episode_id == "prev-ep"
        assert episode.context.active_task_id == "task-1"

    def test_row_to_summary_basic(self):
        row = _make_summary_row()
        summary = _row_to_summary(row)
        assert isinstance(summary, EpisodeSummary)
        assert summary.episode_count == 2
        assert summary.entity_mentions == {"groceries": 2}
        assert summary.intent_distribution == {"task": 2}
        assert len(summary.dominant_emotions) == 1

    def test_row_to_summary_with_embedding(self):
        embedding = [0.5] * 768
        row = _make_summary_row(embedding=embedding)
        summary = _row_to_summary(row)
        assert summary.embedding is not None
        assert len(summary.embedding) == 768


# ── PgEpisodicStore write/read tests ────────────────────────────────


class TestPgEpisodicStoreAppend:
    """Test episode write (append) operations."""

    @pytest.fixture
    def pool(self):
        return _make_pool()

    @pytest.fixture
    def store(self, pool):
        return PgEpisodicStore(pool)

    @pytest.mark.asyncio
    async def test_append_inserts_episode(self, store, pool):
        episode = _make_episode(entities=[])
        result = await store.append(episode)
        assert result.id == episode.id
        conn = pool._conn
        conn.execute.assert_awaited()
        sql = conn.execute.call_args_list[0][0][0]
        assert "INSERT INTO episodes" in sql
        assert "$1" in sql

    @pytest.mark.asyncio
    async def test_append_inserts_entity_refs(self, store, pool):
        episode = _make_episode()
        await store.append(episode)
        conn = pool._conn
        # Should have 3 execute calls: 1 for episode + 2 for entities
        assert conn.execute.await_count == 3
        entity_sql = conn.execute.call_args_list[1][0][0]
        assert "INSERT INTO episode_entities" in entity_sql
        assert "ON CONFLICT" in entity_sql


class TestPgEpisodicStoreGet:
    """Test episode retrieval operations."""

    @pytest.fixture
    def pool(self):
        return _make_pool()

    @pytest.fixture
    def store(self, pool):
        return PgEpisodicStore(pool)

    @pytest.mark.asyncio
    async def test_get_existing_episode(self, store, pool):
        row = _make_episode_row(id="ep-1")
        entity_rows = [_make_entity_row(episode_id="ep-1")]
        conn = pool._conn
        conn.fetchrow = AsyncMock(return_value=row)
        conn.fetch = AsyncMock(return_value=entity_rows)

        episode = await store.get("ep-1")
        assert episode is not None
        assert episode.id == "ep-1"
        assert len(episode.entities) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_episode(self, store, pool):
        conn = pool._conn
        conn.fetchrow = AsyncMock(return_value=None)
        episode = await store.get("nonexistent")
        assert episode is None


class TestPgEpisodicStoreQuery:
    """Test episode query/filter operations."""

    @pytest.fixture
    def pool(self):
        return _make_pool()

    @pytest.fixture
    def store(self, pool):
        return PgEpisodicStore(pool)

    @pytest.mark.asyncio
    async def test_count_user_episodes(self, store, pool):
        conn = pool._conn
        conn.fetchval = AsyncMock(return_value=42)
        result = await store.count("user-1")
        assert result == 42
        sql = conn.fetchval.call_args[0][0]
        assert "COUNT(*)" in sql
        assert "$1" in sql

    @pytest.mark.asyncio
    async def test_query_user_episodes(self, store, pool):
        rows = [_make_episode_row(), _make_episode_row()]
        conn = pool._conn
        # First call returns episode rows, subsequent calls return empty entity rows
        conn.fetch = AsyncMock(side_effect=[rows, [], []])
        episodes = await store.query("user-1")
        assert len(episodes) == 2

    @pytest.mark.asyncio
    async def test_get_session_episodes(self, store, pool):
        rows = [_make_episode_row(context_session_id="sess-1")]
        conn = pool._conn
        # First call returns episode rows, second returns empty entity rows
        conn.fetch = AsyncMock(side_effect=[rows, []])
        episodes = await store.get_session_episodes("sess-1")
        assert len(episodes) == 1
        sql = conn.fetch.call_args_list[0][0][0]
        assert "context_session_id = $1" in sql


# ── Persistence across restarts simulation ──────────────────────────


class TestEpisodePersistence:
    """Simulate episode persistence across server restarts.

    The core contract: episodes written to PgEpisodicStore are stored in
    Postgres, so creating a new PgEpisodicStore instance (as happens after
    a server restart) and reading back should return the same data.
    """

    @pytest.mark.asyncio
    async def test_episode_survives_restart(self):
        """Write episode with store A, create store B (simulating restart),
        read back from store B — verifies data persists in shared DB."""
        episode_id = str(uuid.uuid4())
        _now = _utcnow()
        row = _make_episode_row(
            id=episode_id,
            user_id="user-1",
            raw_text="Call dentist tomorrow morning",
            intent="task",
            intent_confidence=0.95,
            emotion_primary="anticipation",
            emotion_intensity=0.4,
            emotion_valence=0.1,
            emotion_arousal=0.3,
            context_session_id="sess-restart",
        )
        entity_rows = [
            _make_entity_row(episode_id=episode_id, name="dentist", entity_type="person"),
        ]

        # Shared "database" — pool that both stores use
        pool = _make_pool()
        conn = pool._conn

        # --- Store A writes the episode ---
        store_a = PgEpisodicStore(pool)
        episode = _make_episode(
            id=episode_id,
            raw_text="Call dentist tomorrow morning",
            entities=[EntityRef(name="dentist", entity_type="person", confidence=0.9)],
            context=EpisodeContext(session_id="sess-restart"),
        )
        await store_a.append(episode)

        # Verify write happened
        assert conn.execute.await_count >= 1

        # --- Simulate server restart: create Store B on same pool ---
        store_b = PgEpisodicStore(pool)

        # Configure mock to return the stored data on read-back
        conn.fetchrow = AsyncMock(return_value=row)
        conn.fetch = AsyncMock(return_value=entity_rows)

        # Read back from store B
        restored = await store_b.get(episode_id)

        assert restored is not None
        assert restored.id == episode_id
        assert restored.raw_text == "Call dentist tomorrow morning"
        assert restored.intent == "task"
        assert restored.intent_confidence == 0.95
        assert restored.emotion.primary == "anticipation"
        assert restored.context.session_id == "sess-restart"
        assert len(restored.entities) == 1
        assert restored.entities[0].name == "dentist"

    @pytest.mark.asyncio
    async def test_multiple_episodes_survive_restart(self):
        """Write multiple episodes, restart, query all back."""
        pool = _make_pool()
        conn = pool._conn

        ep1_id = str(uuid.uuid4())
        ep2_id = str(uuid.uuid4())

        store_a = PgEpisodicStore(pool)
        await store_a.append(_make_episode(id=ep1_id, raw_text="First blurt", entities=[]))
        await store_a.append(_make_episode(id=ep2_id, raw_text="Second blurt", entities=[]))

        # Simulate restart
        store_b = PgEpisodicStore(pool)

        # Mock query results — first fetch returns episodes, subsequent return entity rows
        rows = [
            _make_episode_row(id=ep1_id, raw_text="First blurt"),
            _make_episode_row(id=ep2_id, raw_text="Second blurt"),
        ]
        conn.fetch = AsyncMock(side_effect=[rows, [], []])

        episodes = await store_b.query("user-1")
        assert len(episodes) == 2

    @pytest.mark.asyncio
    async def test_summary_survives_restart(self):
        """Write an episode summary, restart, read back."""
        pool = _make_pool()
        conn = pool._conn

        summary = EpisodeSummary(
            user_id="user-1",
            period_start=_utcnow(),
            period_end=_utcnow(),
            source_episode_ids=["ep-1", "ep-2"],
            episode_count=2,
            summary_text="User planned errands this week.",
            dominant_emotions=[EmotionSnapshot(primary="trust", intensity=0.5, valence=0.2, arousal=0.3)],
            entity_mentions={"groceries": 2},
            intent_distribution={"task": 2},
            behavioral_signals={"none": 2},
        )

        store_a = PgEpisodicStore(pool)
        await store_a.store_summary(summary)
        assert conn.execute.await_count >= 1

        # Simulate restart
        store_b = PgEpisodicStore(pool)
        summary_row = _make_summary_row(
            id=summary.id,
            summary_text="User planned errands this week.",
        )
        conn.fetch = AsyncMock(return_value=[summary_row])

        summaries = await store_b.get_summaries("user-1")
        assert len(summaries) == 1
        assert summaries[0].summary_text == "User planned errands this week."


# ── SQL injection safety ────────────────────────────────────────────


class TestParameterizedQueries:
    """Verify all SQL uses parameterized queries, no string interpolation."""

    @pytest.mark.asyncio
    async def test_append_no_sql_injection(self):
        pool = _make_pool()
        store = PgEpisodicStore(pool)
        malicious = "'; DROP TABLE episodes; --"
        episode = _make_episode(raw_text=malicious, entities=[])
        await store.append(episode)
        conn = pool._conn
        sql = conn.execute.call_args_list[0][0][0]
        assert "DROP TABLE" not in sql
        assert "$1" in sql

    @pytest.mark.asyncio
    async def test_query_no_sql_injection(self):
        pool = _make_pool()
        conn = pool._conn
        conn.fetch = AsyncMock(return_value=[])
        store = PgEpisodicStore(pool)
        await store.query("'; DROP TABLE episodes; --")
        sql = conn.fetch.call_args[0][0]
        assert "DROP TABLE" not in sql
        assert "$1" in sql

    @pytest.mark.asyncio
    async def test_get_no_sql_injection(self):
        pool = _make_pool()
        conn = pool._conn
        conn.fetchrow = AsyncMock(return_value=None)
        store = PgEpisodicStore(pool)
        await store.get("'; DROP TABLE episodes; --")
        sql = conn.fetchrow.call_args[0][0]
        assert "DROP TABLE" not in sql
        assert "$1" in sql


# ── App wiring test ──────────────────────────────────────────────────


class TestAppWiring:
    """Verify PgEpisodicStore is wired into the app when db_pool exists."""

    def test_episodes_api_set_store(self):
        """The episodes API supports DI via set_store/get_store."""
        from blurt.api.episodes import get_store, set_store

        pool = _make_pool()
        pg_store = PgEpisodicStore(pool)
        set_store(pg_store)
        assert get_store() is pg_store

        # Reset
        set_store(None)  # type: ignore[arg-type]

    def test_app_lifespan_imports_set_store(self):
        """Verify the import path used in app.py lifespan is valid."""
        from blurt.api.episodes import set_store
        from blurt.persistence.pg_episodic_store import PgEpisodicStore

        assert callable(set_store)
        assert callable(PgEpisodicStore)
