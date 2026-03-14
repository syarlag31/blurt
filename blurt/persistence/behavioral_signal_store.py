"""Neon Postgres backend for BehavioralSignalStore.

Implements BehavioralSignalStore Protocol using asyncpg with parameterized
queries. Stores behavioral signals with full context for the Thompson Sampler
learning loop.

Table: behavioral_signals
- id (TEXT PK)
- user_id (TEXT, indexed)
- task_id (TEXT)
- kind (TEXT) — SignalKind enum value
- magnitude (DOUBLE PRECISION)
- reward_value (DOUBLE PRECISION)
- signal_contributions (JSONB)
- context (JSONB) — SignalContext serialized
- source_action (TEXT) — FeedbackAction enum value
- source_event_id (TEXT)
- timestamp (TIMESTAMPTZ, indexed)
- metadata (JSONB)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from collections.abc import Mapping

import asyncpg

from blurt.services.behavioral_signals import (
    BehavioralSignal,
    SignalContext,
    SignalKind,
)
from blurt.services.feedback import FeedbackAction

logger = logging.getLogger(__name__)


# ── Schema DDL ───────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS behavioral_signals (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    task_id             TEXT NOT NULL DEFAULT '',
    kind                TEXT NOT NULL DEFAULT 'neutral',
    magnitude           DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    reward_value        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    signal_contributions JSONB NOT NULL DEFAULT '{}'::jsonb,
    context             JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_action       TEXT NOT NULL DEFAULT 'dismiss',
    source_event_id     TEXT NOT NULL DEFAULT '',
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""

CREATE_INDEX_USER_SQL = """
CREATE INDEX IF NOT EXISTS idx_behavioral_signals_user_id
ON behavioral_signals (user_id)
"""

CREATE_INDEX_TIMESTAMP_SQL = """
CREATE INDEX IF NOT EXISTS idx_behavioral_signals_user_timestamp
ON behavioral_signals (user_id, timestamp DESC)
"""

CREATE_INDEX_INTERACTION_SQL = """
CREATE INDEX IF NOT EXISTS idx_behavioral_signals_interaction
ON behavioral_signals (user_id, task_id, source_action)
"""


class PgBehavioralSignalStore:
    """Postgres implementation of BehavioralSignalStore Protocol.

    Provides both async (preferred) and sync (Protocol compliance) interfaces.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── Async API (preferred) ────────────────────────────────────────────

    async def store_signal_async(self, signal: BehavioralSignal) -> None:
        """Persist a behavioral signal to Postgres."""
        context_json = json.dumps({
            "time_of_day": signal.context.time_of_day,
            "energy_level": signal.context.energy_level,
            "mood_valence": signal.context.mood_valence,
            "intent": signal.context.intent,
            "cognitive_load": signal.context.cognitive_load,
            "tags": signal.context.tags,
            "entity_ids": signal.context.entity_ids,
        })
        contributions_json = json.dumps(signal.signal_contributions)
        metadata_json = json.dumps(signal.metadata)

        await self._pool.execute(
            """
            INSERT INTO behavioral_signals
                (id, user_id, task_id, kind, magnitude, reward_value,
                 signal_contributions, context, source_action, source_event_id,
                 timestamp, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10, $11, $12::jsonb)
            ON CONFLICT (id) DO NOTHING
            """,
            signal.id,
            signal.user_id,
            signal.task_id,
            signal.kind.value,
            signal.magnitude,
            signal.reward_value,
            contributions_json,
            context_json,
            signal.source_action.value,
            signal.source_event_id,
            signal.timestamp,
            metadata_json,
        )

    async def get_signals_async(
        self,
        user_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[BehavioralSignal]:
        """Retrieve behavioral signals for a user, optionally filtered by time."""
        if since is not None:
            rows = await self._pool.fetch(
                """
                SELECT * FROM behavioral_signals
                WHERE user_id = $1 AND timestamp >= $2
                ORDER BY timestamp DESC
                LIMIT $3
                """,
                user_id,
                since,
                limit,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT * FROM behavioral_signals
                WHERE user_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
        return [_row_to_signal(row) for row in rows]

    async def get_interaction_count_async(
        self,
        user_id: str,
        task_id: str,
        action: FeedbackAction | None = None,
    ) -> int:
        """Count interactions for a user-task pair, optionally filtered by action."""
        if action is not None:
            row = await self._pool.fetchrow(
                """
                SELECT COUNT(*) AS cnt FROM behavioral_signals
                WHERE user_id = $1 AND task_id = $2 AND source_action = $3
                """,
                user_id,
                task_id,
                action.value,
            )
        else:
            row = await self._pool.fetchrow(
                """
                SELECT COUNT(*) AS cnt FROM behavioral_signals
                WHERE user_id = $1 AND task_id = $2
                """,
                user_id,
                task_id,
            )
        return row["cnt"] if row else 0

    # ── Sync API (Protocol compliance) ───────────────────────────────────

    def store_signal(self, signal: BehavioralSignal) -> None:
        """Sync wrapper — raises if called outside async context."""
        raise NotImplementedError(
            "Use store_signal_async in async context. "
            "PgBehavioralSignalStore requires an event loop."
        )

    def get_signals(
        self,
        user_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[BehavioralSignal]:
        """Sync wrapper — raises if called outside async context."""
        raise NotImplementedError(
            "Use get_signals_async in async context. "
            "PgBehavioralSignalStore requires an event loop."
        )

    def get_interaction_count(
        self,
        user_id: str,
        task_id: str,
        action: FeedbackAction | None = None,
    ) -> int:
        """Sync wrapper — raises if called outside async context."""
        raise NotImplementedError(
            "Use get_interaction_count_async in async context. "
            "PgBehavioralSignalStore requires an event loop."
        )


def _row_to_signal(row: Mapping[str, Any]) -> BehavioralSignal:
    """Convert an asyncpg Record to a BehavioralSignal."""
    # Parse context JSONB
    context_raw: dict[str, Any] = (
        json.loads(row["context"]) if isinstance(row["context"], str) else row["context"]
    )
    context = SignalContext(
        time_of_day=context_raw.get("time_of_day", ""),
        energy_level=float(context_raw.get("energy_level", 0.5)),
        mood_valence=float(context_raw.get("mood_valence", 0.0)),
        intent=context_raw.get("intent", "task"),
        cognitive_load=float(context_raw.get("cognitive_load", 0.5)),
        tags=context_raw.get("tags", []),
        entity_ids=context_raw.get("entity_ids", []),
    )

    # Parse contributions JSONB
    contributions_raw = row["signal_contributions"]
    contributions: dict[str, float] = (
        json.loads(contributions_raw)
        if isinstance(contributions_raw, str)
        else contributions_raw
    ) or {}

    # Parse metadata JSONB
    metadata_raw = row["metadata"]
    metadata: dict[str, Any] = (
        json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    ) or {}

    timestamp = row["timestamp"]
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return BehavioralSignal(
        id=row["id"],
        user_id=row["user_id"],
        task_id=row["task_id"],
        kind=SignalKind(row["kind"]),
        magnitude=float(row["magnitude"]),
        reward_value=float(row["reward_value"]),
        signal_contributions=contributions,
        context=context,
        source_action=FeedbackAction(row["source_action"]),
        source_event_id=row["source_event_id"],
        timestamp=timestamp,
        metadata=metadata,
    )
