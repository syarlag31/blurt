"""asyncpg connection pool management for Neon Postgres.

Provides pool creation/teardown and a helper to get the pool from FastAPI app state.
The pool is created during FastAPI lifespan startup and closed on shutdown.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Default pool settings tuned for Neon serverless Postgres
_DEFAULT_MIN_POOL_SIZE = 2
_DEFAULT_MAX_POOL_SIZE = 10
_DEFAULT_COMMAND_TIMEOUT = 30.0
_DEFAULT_STATEMENT_CACHE_SIZE = 100


async def create_pool(
    database_url: str,
    *,
    min_size: int = _DEFAULT_MIN_POOL_SIZE,
    max_size: int = _DEFAULT_MAX_POOL_SIZE,
    command_timeout: float = _DEFAULT_COMMAND_TIMEOUT,
    statement_cache_size: int = _DEFAULT_STATEMENT_CACHE_SIZE,
) -> asyncpg.Pool:
    """Create and return an asyncpg connection pool.

    Args:
        database_url: Neon Postgres connection string (from DATABASE_URL env var).
        min_size: Minimum number of connections in the pool.
        max_size: Maximum number of connections in the pool.
        command_timeout: Default timeout for commands in seconds.
        statement_cache_size: Number of prepared statements to cache per connection.

    Returns:
        An initialized asyncpg.Pool ready for queries.

    Raises:
        asyncpg.PostgresError: If connection to the database fails.
    """
    logger.info("Creating asyncpg connection pool (min=%d, max=%d)", min_size, max_size)

    pool = await asyncpg.create_pool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        command_timeout=command_timeout,
        statement_cache_size=statement_cache_size,
    )

    if pool is None:
        raise RuntimeError("asyncpg.create_pool returned None — check DATABASE_URL")

    logger.info("asyncpg connection pool created successfully")
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    """Gracefully close the asyncpg connection pool.

    Args:
        pool: The pool to close.
    """
    logger.info("Closing asyncpg connection pool")
    await pool.close()
    logger.info("asyncpg connection pool closed")


async def run_schema_migrations(pool: asyncpg.Pool) -> None:
    """Run CREATE TABLE IF NOT EXISTS statements for all Blurt tables.

    Uses raw SQL — no Alembic. Safe to call on every startup.
    Enables pgvector extension if not already enabled.

    Args:
        pool: Active connection pool.
    """
    logger.info("Running schema migrations (CREATE IF NOT EXISTS)")

    async with pool.acquire() as conn:
        # Enable pgvector extension for embedding storage
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # ── episodes table ───────────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id                      TEXT PRIMARY KEY,
                user_id                 TEXT NOT NULL,
                timestamp               TIMESTAMPTZ NOT NULL DEFAULT now(),
                raw_text                TEXT NOT NULL DEFAULT '',
                modality                TEXT NOT NULL DEFAULT 'voice',
                intent                  TEXT NOT NULL DEFAULT 'task',
                intent_confidence       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                emotion_primary         TEXT NOT NULL DEFAULT 'trust',
                emotion_intensity       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                emotion_valence         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                emotion_arousal         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                emotion_secondary       TEXT,
                behavioral_signal       TEXT NOT NULL DEFAULT 'none',
                surfaced_task_id        TEXT,
                context_time_of_day     TEXT NOT NULL DEFAULT 'morning',
                context_day_of_week     TEXT NOT NULL DEFAULT 'monday',
                context_session_id      TEXT NOT NULL DEFAULT '',
                context_preceding_episode_id TEXT,
                context_active_task_id  TEXT,
                is_compressed           BOOLEAN NOT NULL DEFAULT FALSE,
                compressed_into_id      TEXT,
                embedding               vector(768),
                source_working_id       TEXT
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_user_id
            ON episodes (user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_user_timestamp
            ON episodes (user_id, timestamp DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_session
            ON episodes (context_session_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_embedding
            ON episodes USING hnsw (embedding vector_cosine_ops)
        """)

        # ── episode_entities junction table ──────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS episode_entities (
                episode_id      TEXT NOT NULL REFERENCES episodes(id),
                name            TEXT NOT NULL,
                entity_type     TEXT NOT NULL,
                entity_id       TEXT,
                confidence      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                PRIMARY KEY (episode_id, name)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episode_entities_name
            ON episode_entities (lower(name))
        """)

        # ── episode_summaries table ──────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS episode_summaries (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                period_start        TIMESTAMPTZ NOT NULL,
                period_end          TIMESTAMPTZ NOT NULL,
                source_episode_ids  TEXT[] NOT NULL DEFAULT '{}',
                episode_count       INTEGER NOT NULL DEFAULT 0,
                summary_text        TEXT NOT NULL DEFAULT '',
                dominant_emotions   JSONB NOT NULL DEFAULT '[]'::jsonb,
                entity_mentions     JSONB NOT NULL DEFAULT '{}'::jsonb,
                intent_distribution JSONB NOT NULL DEFAULT '{}'::jsonb,
                behavioral_signals  JSONB NOT NULL DEFAULT '{}'::jsonb,
                embedding           vector(768)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episode_summaries_user
            ON episode_summaries (user_id, period_start DESC)
        """)

        # ── entity_nodes table ───────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_nodes (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                name            TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                entity_type     TEXT NOT NULL,
                aliases         TEXT[] NOT NULL DEFAULT '{}',
                attributes      JSONB NOT NULL DEFAULT '{}'::jsonb,
                mention_count   INTEGER NOT NULL DEFAULT 0,
                first_seen      TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_seen       TIMESTAMPTZ NOT NULL DEFAULT now(),
                embedding       vector(768),
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_nodes_user
            ON entity_nodes (user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_nodes_name
            ON entity_nodes (user_id, normalized_name)
        """)
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_nodes_user_norm_name
            ON entity_nodes (user_id, normalized_name)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_nodes_embedding
            ON entity_nodes USING hnsw (embedding vector_cosine_ops)
        """)

        # ── relationship_edges table ─────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS relationship_edges (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                source_entity_id    TEXT NOT NULL REFERENCES entity_nodes(id),
                target_entity_id    TEXT NOT NULL REFERENCES entity_nodes(id),
                relationship_type   TEXT NOT NULL,
                strength            DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                co_mention_count    INTEGER NOT NULL DEFAULT 1,
                context_snippets    TEXT[] NOT NULL DEFAULT '{}',
                first_seen          TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_seen           TIMESTAMPTZ NOT NULL DEFAULT now(),
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_edges_source
            ON relationship_edges (source_entity_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rel_edges_target
            ON relationship_edges (target_entity_id)
        """)
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_edges_pair
            ON relationship_edges (source_entity_id, target_entity_id, relationship_type)
        """)

        # ── facts table ──────────────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                fact_type           TEXT NOT NULL,
                subject_entity_id   TEXT,
                content             TEXT NOT NULL,
                confidence          DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                source_blurt_ids    TEXT[] NOT NULL DEFAULT '{}',
                embedding           vector(768),
                is_active           BOOLEAN NOT NULL DEFAULT TRUE,
                superseded_by       TEXT,
                first_learned       TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_confirmed      TIMESTAMPTZ NOT NULL DEFAULT now(),
                confirmation_count  INTEGER NOT NULL DEFAULT 1,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_user
            ON facts (user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_entity
            ON facts (subject_entity_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_embedding
            ON facts USING hnsw (embedding vector_cosine_ops)
        """)

        # ── learned_patterns table ───────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                pattern_type        TEXT NOT NULL,
                description         TEXT NOT NULL,
                parameters          JSONB NOT NULL DEFAULT '{}'::jsonb,
                confidence          DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                observation_count   INTEGER NOT NULL DEFAULT 0,
                supporting_evidence TEXT[] NOT NULL DEFAULT '{}',
                embedding           vector(768),
                is_active           BOOLEAN NOT NULL DEFAULT TRUE,
                first_detected      TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_confirmed      TIMESTAMPTZ NOT NULL DEFAULT now(),
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_user
            ON learned_patterns (user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_embedding
            ON learned_patterns USING hnsw (embedding vector_cosine_ops)
        """)

        # ── tasks table ──────────────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id                          TEXT PRIMARY KEY,
                user_id                     TEXT NOT NULL,
                content                     TEXT NOT NULL DEFAULT '',
                status                      TEXT NOT NULL DEFAULT 'active',
                intent                      TEXT NOT NULL DEFAULT 'task',
                created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
                due_at                      TIMESTAMPTZ,
                last_mentioned_at           TIMESTAMPTZ,
                estimated_energy            TEXT NOT NULL DEFAULT 'medium',
                estimated_duration_minutes  INTEGER,
                entity_ids                  TEXT[] NOT NULL DEFAULT '{}',
                entity_names                TEXT[] NOT NULL DEFAULT '{}',
                project                     TEXT,
                capture_valence             DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                capture_arousal             DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                times_surfaced              INTEGER NOT NULL DEFAULT 0,
                times_deferred              INTEGER NOT NULL DEFAULT 0,
                metadata                    JSONB NOT NULL DEFAULT '{}'::jsonb
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_user_id
            ON tasks (user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_user_status
            ON tasks (user_id, status)
        """)

        # ── feedback_events table ────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_events (
                id              TEXT PRIMARY KEY,
                task_id         TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                action          TEXT NOT NULL,
                timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
                context_key     TEXT NOT NULL DEFAULT '',
                mood_valence    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                energy_level    DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                time_of_day     TEXT NOT NULL DEFAULT '',
                snooze_minutes  INTEGER,
                metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_events_task
            ON feedback_events (task_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_events_user
            ON feedback_events (user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_events_timestamp
            ON feedback_events (timestamp DESC)
        """)

        # ── thompson_params table ────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS thompson_params (
                key                 TEXT PRIMARY KEY,
                alpha               DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                beta                DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                total_observations  INTEGER NOT NULL DEFAULT 0,
                last_updated        TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        # ── user_preferences table ────────────────────────────────────
        from blurt.persistence.preference_store import (
            CREATE_TABLE_SQL as PREF_TABLE_SQL,
            CREATE_INDEX_SQL as PREF_INDEX_SQL,
        )
        await conn.execute(PREF_TABLE_SQL)
        await conn.execute(PREF_INDEX_SQL)

        # ── behavioral_signals table ──────────────────────────────────
        from blurt.persistence.behavioral_signal_store import (
            CREATE_TABLE_SQL as SIGNAL_TABLE_SQL,
            CREATE_INDEX_USER_SQL,
            CREATE_INDEX_TIMESTAMP_SQL,
            CREATE_INDEX_INTERACTION_SQL,
        )
        await conn.execute(SIGNAL_TABLE_SQL)
        await conn.execute(CREATE_INDEX_USER_SQL)
        await conn.execute(CREATE_INDEX_TIMESTAMP_SQL)
        await conn.execute(CREATE_INDEX_INTERACTION_SQL)

        # ── sync tables (records, operations, conflicts) ──────────────
        from blurt.persistence.sync_state_store import (
            CREATE_SYNC_RECORDS_SQL,
            CREATE_SYNC_RECORDS_INDEX_SQL,
            CREATE_SYNC_RECORDS_EXTERNAL_INDEX_SQL,
            CREATE_SYNC_RECORDS_STATUS_INDEX_SQL,
            CREATE_SYNC_OPERATIONS_SQL,
            CREATE_SYNC_OPERATIONS_INDEX_SQL,
            CREATE_SYNC_CONFLICTS_SQL,
            CREATE_SYNC_CONFLICTS_INDEX_SQL,
        )
        await conn.execute(CREATE_SYNC_RECORDS_SQL)
        await conn.execute(CREATE_SYNC_RECORDS_INDEX_SQL)
        await conn.execute(CREATE_SYNC_RECORDS_EXTERNAL_INDEX_SQL)
        await conn.execute(CREATE_SYNC_RECORDS_STATUS_INDEX_SQL)
        await conn.execute(CREATE_SYNC_OPERATIONS_SQL)
        await conn.execute(CREATE_SYNC_OPERATIONS_INDEX_SQL)
        await conn.execute(CREATE_SYNC_CONFLICTS_SQL)
        await conn.execute(CREATE_SYNC_CONFLICTS_INDEX_SQL)

        logger.info("Schema migrations complete — all tables created")


def get_pool(app: FastAPI) -> asyncpg.Pool:
    """Retrieve the asyncpg pool from FastAPI app state.

    Args:
        app: The FastAPI application instance.

    Returns:
        The shared asyncpg connection pool.

    Raises:
        RuntimeError: If the pool hasn't been initialized yet.
    """
    pool: asyncpg.Pool | None = getattr(app.state, "db_pool", None)
    if pool is None:
        raise RuntimeError(
            "Database pool not initialized. "
            "Ensure the FastAPI lifespan has started and DATABASE_URL is configured."
        )
    return pool
