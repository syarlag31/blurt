"""Neon Postgres implementation of TaskStore for surfaceable tasks.

Replaces InMemoryTaskStore with asyncpg-backed persistence.
All SQL uses parameterized queries — zero string interpolation.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from collections.abc import Mapping

import asyncpg

from blurt.services.task_surfacing import (
    EnergyLevel,
    SurfaceableTask,
    TaskStatus,
)

logger = logging.getLogger(__name__)


def _task_to_row(task: SurfaceableTask, user_id: str) -> dict[str, Any]:
    """Convert a SurfaceableTask dataclass to a dict of column values."""
    return {
        "id": task.id,
        "user_id": user_id,
        "content": task.content,
        "status": task.status.value,
        "intent": task.intent,
        "created_at": task.created_at,
        "due_at": task.due_at,
        "last_mentioned_at": task.last_mentioned_at,
        "estimated_energy": task.estimated_energy.value,
        "estimated_duration_minutes": task.estimated_duration_minutes,
        "entity_ids": task.entity_ids,
        "entity_names": task.entity_names,
        "project": task.project,
        "capture_valence": task.capture_valence,
        "capture_arousal": task.capture_arousal,
        "times_surfaced": task.times_surfaced,
        "times_deferred": task.times_deferred,
        "metadata": json.dumps(task.metadata),
    }


def _row_to_task(row: Mapping[str, Any]) -> SurfaceableTask:
    """Convert a database row to a SurfaceableTask dataclass."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return SurfaceableTask(
        id=row["id"],
        content=row["content"],
        status=TaskStatus(row["status"]),
        intent=row["intent"],
        created_at=row["created_at"],
        due_at=row["due_at"],
        last_mentioned_at=row["last_mentioned_at"],
        estimated_energy=EnergyLevel(row["estimated_energy"]),
        estimated_duration_minutes=row["estimated_duration_minutes"],
        entity_ids=list(row["entity_ids"]) if row["entity_ids"] else [],
        entity_names=list(row["entity_names"]) if row["entity_names"] else [],
        project=row["project"],
        capture_valence=row["capture_valence"],
        capture_arousal=row["capture_arousal"],
        times_surfaced=row["times_surfaced"],
        times_deferred=row["times_deferred"],
        metadata=metadata if isinstance(metadata, dict) else {},
    )


class PgTaskStore:
    """Postgres-backed task store implementing the TaskStore protocol.

    All methods accept an optional user_id parameter matching the
    InMemoryTaskStore interface signature.
    """

    DEFAULT_USER = "__default__"

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    def add_task(
        self, task: SurfaceableTask, user_id: str | None = None
    ) -> None:
        """Synchronous facade — delegates to async insert.

        The Protocol defines sync methods, so callers must use
        the async variants directly or run in an event loop.
        """
        raise NotImplementedError(
            "Use add_task_async for the Postgres store"
        )

    async def add_task_async(
        self, task: SurfaceableTask, user_id: str | None = None
    ) -> None:
        """Insert a new task into the database."""
        uid = user_id or self.DEFAULT_USER
        row = _task_to_row(task, uid)

        await self._pool.execute(
            """
            INSERT INTO tasks (
                id, user_id, content, status, intent,
                created_at, due_at, last_mentioned_at,
                estimated_energy, estimated_duration_minutes,
                entity_ids, entity_names, project,
                capture_valence, capture_arousal,
                times_surfaced, times_deferred, metadata
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8,
                $9, $10,
                $11, $12, $13,
                $14, $15,
                $16, $17, $18::jsonb
            )
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                status = EXCLUDED.status,
                intent = EXCLUDED.intent,
                due_at = EXCLUDED.due_at,
                last_mentioned_at = EXCLUDED.last_mentioned_at,
                estimated_energy = EXCLUDED.estimated_energy,
                estimated_duration_minutes = EXCLUDED.estimated_duration_minutes,
                entity_ids = EXCLUDED.entity_ids,
                entity_names = EXCLUDED.entity_names,
                project = EXCLUDED.project,
                capture_valence = EXCLUDED.capture_valence,
                capture_arousal = EXCLUDED.capture_arousal,
                times_surfaced = EXCLUDED.times_surfaced,
                times_deferred = EXCLUDED.times_deferred,
                metadata = EXCLUDED.metadata
            """,
            row["id"],
            row["user_id"],
            row["content"],
            row["status"],
            row["intent"],
            row["created_at"],
            row["due_at"],
            row["last_mentioned_at"],
            row["estimated_energy"],
            row["estimated_duration_minutes"],
            row["entity_ids"],
            row["entity_names"],
            row["project"],
            row["capture_valence"],
            row["capture_arousal"],
            row["times_surfaced"],
            row["times_deferred"],
            row["metadata"],
        )

    def get_task(
        self, task_id: str, user_id: str | None = None
    ) -> SurfaceableTask | None:
        raise NotImplementedError("Use get_task_async for the Postgres store")

    async def get_task_async(
        self, task_id: str, user_id: str | None = None
    ) -> SurfaceableTask | None:
        """Retrieve a single task by ID."""
        uid = user_id or self.DEFAULT_USER
        row = await self._pool.fetchrow(
            "SELECT * FROM tasks WHERE id = $1 AND user_id = $2",
            task_id,
            uid,
        )
        if row is None:
            return None
        return _row_to_task(row)

    def get_all_tasks(
        self, user_id: str | None = None
    ) -> list[SurfaceableTask]:
        raise NotImplementedError(
            "Use get_all_tasks_async for the Postgres store"
        )

    async def get_all_tasks_async(
        self, user_id: str | None = None
    ) -> list[SurfaceableTask]:
        """Retrieve all tasks for a user."""
        uid = user_id or self.DEFAULT_USER
        rows = await self._pool.fetch(
            "SELECT * FROM tasks WHERE user_id = $1 ORDER BY created_at DESC",
            uid,
        )
        return [_row_to_task(r) for r in rows]

    def update_task(
        self, task: SurfaceableTask, user_id: str | None = None
    ) -> None:
        raise NotImplementedError(
            "Use update_task_async for the Postgres store"
        )

    async def update_task_async(
        self, task: SurfaceableTask, user_id: str | None = None
    ) -> None:
        """Update an existing task. Uses upsert to be safe."""
        await self.add_task_async(task, user_id=user_id)

    def remove_task(
        self, task_id: str, user_id: str | None = None
    ) -> bool:
        raise NotImplementedError(
            "Use remove_task_async for the Postgres store"
        )

    async def remove_task_async(
        self, task_id: str, user_id: str | None = None
    ) -> bool:
        """Delete a task by ID. Returns True if a row was deleted."""
        uid = user_id or self.DEFAULT_USER
        result = await self._pool.execute(
            "DELETE FROM tasks WHERE id = $1 AND user_id = $2",
            task_id,
            uid,
        )
        # asyncpg returns "DELETE N" where N is the number of rows deleted
        return result == "DELETE 1"

    async def clear_async(self, user_id: str | None = None) -> None:
        """Clear all tasks for a user (or all users if None)."""
        if user_id:
            await self._pool.execute(
                "DELETE FROM tasks WHERE user_id = $1", user_id
            )
        else:
            await self._pool.execute("DELETE FROM tasks")
