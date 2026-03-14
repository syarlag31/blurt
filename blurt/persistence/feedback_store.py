"""Neon Postgres implementation of FeedbackStore.

Replaces InMemoryFeedbackStore with asyncpg-backed persistence.
All SQL uses parameterized queries — zero string interpolation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from collections.abc import Mapping

import asyncpg

from blurt.services.feedback import (
    FeedbackAction,
    FeedbackEvent,
    FeedbackSummary,
    ThompsonParams,
)

logger = logging.getLogger(__name__)


def _event_to_params(event: FeedbackEvent) -> tuple:
    """Convert a FeedbackEvent to a tuple of positional parameters."""
    return (
        event.id,
        event.task_id,
        event.user_id,
        event.action.value,
        event.timestamp,
        event.context_key,
        event.mood_valence,
        event.energy_level,
        event.time_of_day,
        event.snooze_minutes,
        json.dumps(event.metadata),
    )


def _row_to_event(row: Mapping[str, Any]) -> FeedbackEvent:
    """Convert a database row to a FeedbackEvent."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return FeedbackEvent(
        id=row["id"],
        task_id=row["task_id"],
        user_id=row["user_id"],
        action=FeedbackAction(row["action"]),
        timestamp=row["timestamp"],
        context_key=row["context_key"],
        mood_valence=row["mood_valence"],
        energy_level=row["energy_level"],
        time_of_day=row["time_of_day"],
        snooze_minutes=row["snooze_minutes"],
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _row_to_thompson(row: Mapping[str, Any]) -> ThompsonParams:
    """Convert a database row to ThompsonParams."""
    return ThompsonParams(
        alpha=row["alpha"],
        beta=row["beta"],
        total_observations=row["total_observations"],
        last_updated=row["last_updated"],
    )


class PgFeedbackStore:
    """Postgres-backed feedback store implementing the FeedbackStore protocol.

    The FeedbackStore Protocol defines synchronous methods. This implementation
    provides both sync-named methods (that raise NotImplementedError) and
    async variants for actual database operations.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── Sync protocol stubs (Protocol is sync) ──────────────────────

    def store_event(self, event: FeedbackEvent) -> None:
        raise NotImplementedError("Use store_event_async for the Postgres store")

    def get_events(
        self,
        task_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackEvent]:
        raise NotImplementedError("Use get_events_async for the Postgres store")

    def get_params(self, key: str) -> ThompsonParams:
        raise NotImplementedError("Use get_params_async for the Postgres store")

    def set_params(self, key: str, params: ThompsonParams) -> None:
        raise NotImplementedError("Use set_params_async for the Postgres store")

    def get_task_summary(self, task_id: str) -> FeedbackSummary:
        raise NotImplementedError(
            "Use get_task_summary_async for the Postgres store"
        )

    # ── Async implementations ────────────────────────────────────────

    async def store_event_async(self, event: FeedbackEvent) -> None:
        """Insert a feedback event into the database."""
        params = _event_to_params(event)
        await self._pool.execute(
            """
            INSERT INTO feedback_events (
                id, task_id, user_id, action, timestamp,
                context_key, mood_valence, energy_level,
                time_of_day, snooze_minutes, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
            ON CONFLICT (id) DO NOTHING
            """,
            *params,
        )

    async def get_events_async(
        self,
        task_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[FeedbackEvent]:
        """Retrieve feedback events with optional filters, most recent first."""
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if task_id is not None:
            conditions.append(f"task_id = ${idx}")
            params.append(task_id)
            idx += 1

        if user_id is not None:
            conditions.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        # The WHERE clause is built from static strings + parameter placeholders
        # No user input is interpolated into the SQL string
        query = f"""
            SELECT * FROM feedback_events
            {where}
            ORDER BY timestamp DESC
            LIMIT ${idx}
        """
        params.append(limit)

        rows = await self._pool.fetch(query, *params)
        return [_row_to_event(r) for r in rows]

    async def get_params_async(self, key: str) -> ThompsonParams:
        """Get Thompson params for a key, creating with priors if missing."""
        row = await self._pool.fetchrow(
            "SELECT * FROM thompson_params WHERE key = $1", key
        )
        if row is not None:
            return _row_to_thompson(row)

        # Create default (uniform prior) params
        now = datetime.now(timezone.utc)
        default = ThompsonParams(
            alpha=1.0, beta=1.0, total_observations=0, last_updated=now
        )
        await self._pool.execute(
            """
            INSERT INTO thompson_params (key, alpha, beta, total_observations, last_updated)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (key) DO NOTHING
            """,
            key,
            default.alpha,
            default.beta,
            default.total_observations,
            default.last_updated,
        )
        return default

    async def set_params_async(self, key: str, params: ThompsonParams) -> None:
        """Persist updated Thompson params (upsert)."""
        await self._pool.execute(
            """
            INSERT INTO thompson_params (key, alpha, beta, total_observations, last_updated)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (key) DO UPDATE SET
                alpha = EXCLUDED.alpha,
                beta = EXCLUDED.beta,
                total_observations = EXCLUDED.total_observations,
                last_updated = EXCLUDED.last_updated
            """,
            key,
            params.alpha,
            params.beta,
            params.total_observations,
            params.last_updated,
        )

    async def get_task_summary_async(self, task_id: str) -> FeedbackSummary:
        """Get aggregated feedback summary for a task via SQL aggregation."""
        row = await self._pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_events,
                COUNT(*) FILTER (WHERE action = 'accept') AS accept_count,
                COUNT(*) FILTER (WHERE action = 'dismiss') AS dismiss_count,
                COUNT(*) FILTER (WHERE action = 'snooze') AS snooze_count,
                COUNT(*) FILTER (WHERE action = 'complete') AS complete_count,
                MAX(timestamp) AS last_feedback_at
            FROM feedback_events
            WHERE task_id = $1
            """,
            task_id,
        )

        total = row["total_events"] if row else 0
        if total == 0:
            return FeedbackSummary(task_id=task_id)

        accept_count = row["accept_count"]
        complete_count = row["complete_count"]
        positive = accept_count + complete_count
        acceptance_rate = positive / total if total > 0 else 0.0

        # Get global Thompson params for this task
        global_params = await self.get_params_async(f"task:{task_id}")

        return FeedbackSummary(
            task_id=task_id,
            total_events=total,
            accept_count=accept_count,
            dismiss_count=row["dismiss_count"],
            snooze_count=row["snooze_count"],
            complete_count=complete_count,
            acceptance_rate=round(acceptance_rate, 4),
            thompson_mean=round(global_params.mean, 4),
            last_feedback_at=row["last_feedback_at"],
        )
