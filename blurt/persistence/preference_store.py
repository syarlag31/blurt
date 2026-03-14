"""Neon Postgres backend for UserPreferenceStore.

Implements PreferenceStoreBackend Protocol using asyncpg with parameterized
queries. Stores user preference snapshots (Beta distribution parameters per
category) as JSONB for flexible schema evolution.

Table: user_preferences
- user_id (TEXT PK)
- params (JSONB) — per-category Beta distribution parameters
- created_at (TIMESTAMPTZ)
- last_interaction (TIMESTAMPTZ)
- total_feedback_count (INTEGER)
- version (INTEGER)
"""

from __future__ import annotations

import json
import logging
from datetime import timezone
from typing import Any
from collections.abc import Mapping

import asyncpg

from blurt.services.preference_store import UserPreferenceSnapshot
from blurt.services.thompson_sampling import BetaParams

logger = logging.getLogger(__name__)


# ── Schema DDL ───────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id             TEXT PRIMARY KEY,
    params              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_interaction    TIMESTAMPTZ NOT NULL DEFAULT now(),
    total_feedback_count INTEGER NOT NULL DEFAULT 0,
    version             INTEGER NOT NULL DEFAULT 1
)
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_user_preferences_last_interaction
ON user_preferences (last_interaction)
"""


class PgPreferenceBackend:
    """Postgres implementation of PreferenceStoreBackend Protocol.

    All methods are synchronous wrappers around asyncpg — they use
    asyncio.get_event_loop().run_until_complete internally. However, the
    PreferenceStoreBackend Protocol is synchronous, so we provide both
    sync and async variants. The async variants are preferred when called
    from async context.

    NOTE: The existing PreferenceStoreBackend Protocol is synchronous.
    We implement the async versions and also provide sync-compatible
    interface by storing snapshots via the async methods. The
    UserPreferenceStore calls load_snapshot/save_snapshot synchronously,
    so we need to bridge this gap.

    Strategy: We provide async methods and a thin sync wrapper that
    the caller (UserPreferenceStore) can use. Since UserPreferenceStore
    is used in an async FastAPI context, we'll modify the store to use
    async methods directly.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── Async API (preferred) ────────────────────────────────────────────

    async def load_snapshot_async(self, user_id: str) -> UserPreferenceSnapshot | None:
        """Load a user's preference snapshot from Postgres."""
        row = await self._pool.fetchrow(
            "SELECT user_id, params, created_at, last_interaction, "
            "total_feedback_count, version FROM user_preferences WHERE user_id = $1",
            user_id,
        )
        if row is None:
            return None
        return _row_to_snapshot(row)

    async def save_snapshot_async(self, snapshot: UserPreferenceSnapshot) -> None:
        """Persist a user's preference snapshot to Postgres (upsert)."""
        params_json = json.dumps(
            {cat: p.to_dict() for cat, p in snapshot.params.items()}
        )
        await self._pool.execute(
            """
            INSERT INTO user_preferences (user_id, params, created_at, last_interaction,
                                          total_feedback_count, version)
            VALUES ($1, $2::jsonb, $3, $4, $5, $6)
            ON CONFLICT (user_id) DO UPDATE SET
                params = $2::jsonb,
                last_interaction = $4,
                total_feedback_count = $5,
                version = $6
            """,
            snapshot.user_id,
            params_json,
            snapshot.created_at,
            snapshot.last_interaction,
            snapshot.total_feedback_count,
            snapshot.version,
        )

    async def delete_snapshot_async(self, user_id: str) -> bool:
        """Delete a user's preferences. Returns True if found and deleted."""
        result = await self._pool.execute(
            "DELETE FROM user_preferences WHERE user_id = $1",
            user_id,
        )
        # asyncpg returns "DELETE N" where N is the count
        return result == "DELETE 1"

    async def list_users_async(self) -> list[str]:
        """List all user IDs with stored preferences."""
        rows = await self._pool.fetch(
            "SELECT user_id FROM user_preferences ORDER BY user_id"
        )
        return [row["user_id"] for row in rows]

    # ── Sync API (Protocol compliance) ───────────────────────────────────
    # These are here for duck-typing compatibility with the Protocol.
    # In practice, callers should use the async variants.

    def load_snapshot(self, user_id: str) -> UserPreferenceSnapshot | None:
        """Sync wrapper — raises if called outside async context."""
        raise NotImplementedError(
            "Use load_snapshot_async in async context. "
            "PgPreferenceBackend requires an event loop."
        )

    def save_snapshot(self, snapshot: UserPreferenceSnapshot) -> None:
        """Sync wrapper — raises if called outside async context."""
        raise NotImplementedError(
            "Use save_snapshot_async in async context. "
            "PgPreferenceBackend requires an event loop."
        )

    def delete_snapshot(self, user_id: str) -> bool:
        """Sync wrapper — raises if called outside async context."""
        raise NotImplementedError(
            "Use delete_snapshot_async in async context. "
            "PgPreferenceBackend requires an event loop."
        )

    def list_users(self) -> list[str]:
        """Sync wrapper — raises if called outside async context."""
        raise NotImplementedError(
            "Use list_users_async in async context. "
            "PgPreferenceBackend requires an event loop."
        )


def _row_to_snapshot(row: Mapping[str, Any]) -> UserPreferenceSnapshot:
    """Convert an asyncpg Record to a UserPreferenceSnapshot."""
    params_raw: dict[str, Any] = json.loads(row["params"]) if isinstance(row["params"], str) else row["params"]
    params = {}
    for cat, pdata in params_raw.items():
        params[cat] = BetaParams.from_dict(pdata)

    created_at = row["created_at"]
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    last_interaction = row["last_interaction"]
    if last_interaction.tzinfo is None:
        last_interaction = last_interaction.replace(tzinfo=timezone.utc)

    return UserPreferenceSnapshot(
        user_id=row["user_id"],
        params=params,
        created_at=created_at,
        last_interaction=last_interaction,
        total_feedback_count=row["total_feedback_count"],
        version=row["version"],
    )
