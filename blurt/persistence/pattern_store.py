"""Neon Postgres implementation of PatternStore for learned behavioral patterns.

Replaces InMemoryPatternStore with asyncpg-backed persistence.
All SQL uses parameterized queries — zero string interpolation.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any
from collections.abc import Mapping

import asyncpg

from blurt.models.entities import LearnedPattern, PatternType

logger = logging.getLogger(__name__)


def _pattern_to_params(pattern: LearnedPattern) -> tuple:
    """Convert a LearnedPattern to positional parameters for INSERT."""
    return (
        pattern.id,
        pattern.user_id,
        pattern.pattern_type.value,
        pattern.description,
        json.dumps(pattern.parameters),
        pattern.confidence,
        pattern.observation_count,
        pattern.supporting_evidence,
        pattern.embedding,
        pattern.is_active,
        pattern.first_detected,
        pattern.last_confirmed,
        pattern.created_at,
        pattern.updated_at,
    )


def _row_to_pattern(row: Mapping[str, Any]) -> LearnedPattern:
    """Convert a database row to a LearnedPattern model."""
    parameters = row["parameters"]
    if isinstance(parameters, str):
        parameters = json.loads(parameters)

    embedding = row["embedding"]
    if embedding is not None:
        # pgvector returns a string representation; convert to list[float]
        if isinstance(embedding, str):
            embedding = [float(x) for x in embedding.strip("[]").split(",")]

    return LearnedPattern(
        id=row["id"],
        user_id=row["user_id"],
        pattern_type=PatternType(row["pattern_type"]),
        description=row["description"],
        parameters=parameters if isinstance(parameters, dict) else {},
        confidence=row["confidence"],
        observation_count=row["observation_count"],
        supporting_evidence=list(row["supporting_evidence"])
        if row["supporting_evidence"]
        else [],
        embedding=embedding,
        is_active=row["is_active"],
        first_detected=row["first_detected"],
        last_confirmed=row["last_confirmed"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PgPatternStore:
    """Postgres-backed pattern store implementing the PatternStore protocol.

    All async methods match the PatternStore Protocol signature.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, pattern: LearnedPattern) -> LearnedPattern:
        """Persist a new or updated pattern (upsert)."""
        if not pattern.id:
            pattern.id = str(uuid.uuid4())

        pattern.updated_at = datetime.now(timezone.utc)
        _params = _pattern_to_params(pattern)

        # Handle embedding: convert list[float] to pgvector string
        embedding_val = None
        if pattern.embedding is not None:
            embedding_val = f"[{','.join(str(x) for x in pattern.embedding)}]"

        await self._pool.execute(
            """
            INSERT INTO learned_patterns (
                id, user_id, pattern_type, description, parameters,
                confidence, observation_count, supporting_evidence,
                embedding, is_active,
                first_detected, last_confirmed, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5::jsonb,
                $6, $7, $8,
                $9::vector, $10,
                $11, $12, $13, $14
            )
            ON CONFLICT (id) DO UPDATE SET
                description = EXCLUDED.description,
                parameters = EXCLUDED.parameters,
                confidence = EXCLUDED.confidence,
                observation_count = EXCLUDED.observation_count,
                supporting_evidence = EXCLUDED.supporting_evidence,
                embedding = EXCLUDED.embedding,
                is_active = EXCLUDED.is_active,
                last_confirmed = EXCLUDED.last_confirmed,
                updated_at = EXCLUDED.updated_at
            """,
            pattern.id,
            pattern.user_id,
            pattern.pattern_type.value,
            pattern.description,
            json.dumps(pattern.parameters),
            pattern.confidence,
            pattern.observation_count,
            pattern.supporting_evidence,
            embedding_val,
            pattern.is_active,
            pattern.first_detected,
            pattern.last_confirmed,
            pattern.created_at,
            pattern.updated_at,
        )
        return pattern

    async def get(self, pattern_id: str) -> LearnedPattern | None:
        """Retrieve a pattern by ID."""
        row = await self._pool.fetchrow(
            "SELECT * FROM learned_patterns WHERE id = $1", pattern_id
        )
        if row is None:
            return None
        return _row_to_pattern(row)

    async def query(
        self,
        user_id: str,
        *,
        pattern_type: PatternType | None = None,
        day_of_week: str | None = None,
        time_of_day: str | None = None,
        min_confidence: float = 0.0,
        is_active: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[LearnedPattern]:
        """Query patterns with filters.

        Note: day_of_week and time_of_day filtering is done in
        application code after fetching, since they inspect the JSONB
        parameters field in complex ways matching the in-memory logic.
        """
        # Build parameterized query with static filter clauses
        conditions = ["user_id = $1", "is_active = $2", "confidence >= $3"]
        params: list[Any] = [user_id, is_active, min_confidence]
        idx = 4

        if pattern_type is not None:
            conditions.append(f"pattern_type = ${idx}")
            params.append(pattern_type.value)
            idx += 1

        where = " AND ".join(conditions)
        query = f"""
            SELECT * FROM learned_patterns
            WHERE {where}
            ORDER BY confidence DESC, last_confirmed DESC
        """

        rows = await self._pool.fetch(query, *params)
        learned_patterns = [_row_to_pattern(r) for r in rows]

        # Apply day_of_week and time_of_day filters in Python
        # (matches the in-memory filtering logic for JSONB parameters)
        if day_of_week is not None:
            learned_patterns = self._filter_by_day(learned_patterns, day_of_week)

        if time_of_day is not None:
            learned_patterns = self._filter_by_time(learned_patterns, time_of_day)

        return learned_patterns[offset: offset + limit]

    async def count(self, user_id: str, *, is_active: bool = True) -> int:
        """Count patterns for a user."""
        row = await self._pool.fetchrow(
            """
            SELECT COUNT(*) AS cnt
            FROM learned_patterns
            WHERE user_id = $1 AND is_active = $2
            """,
            user_id,
            is_active,
        )
        return row["cnt"] if row else 0

    async def delete(self, pattern_id: str) -> bool:
        """Hard-delete a pattern. Returns True if found."""
        result = await self._pool.execute(
            "DELETE FROM learned_patterns WHERE id = $1", pattern_id
        )
        return result == "DELETE 1"

    # ── Private helpers for JSONB parameter filtering ────────────────

    @staticmethod
    def _filter_by_day(
        learned_patterns: list[LearnedPattern], day_of_week: str
    ) -> list[LearnedPattern]:
        """Filter patterns by day_of_week in their parameters dict."""
        day_lower = day_of_week.lower()
        result = []
        for p in learned_patterns:
            param_day = p.parameters.get("day_of_week", "")
            param_days = p.parameters.get("days", [])
            if isinstance(param_day, str):
                param_day = param_day.lower()
            if isinstance(param_days, list):
                param_days = [d.lower() for d in param_days]

            if param_day == day_lower or day_lower in param_days:
                result.append(p)
            elif not param_day and not param_days:
                # No day info — include pattern (matches in-memory logic)
                result.append(p)
        return result

    @staticmethod
    def _filter_by_time(
        learned_patterns: list[LearnedPattern], time_of_day: str
    ) -> list[LearnedPattern]:
        """Filter patterns by time_of_day in their parameters dict."""
        time_lower = time_of_day.lower()
        result = []
        for p in learned_patterns:
            param_time = p.parameters.get("time_of_day", "")
            param_times = p.parameters.get("times", [])
            if isinstance(param_time, str):
                param_time = param_time.lower()
            if isinstance(param_times, list):
                param_times = [t.lower() for t in param_times]

            if param_time == time_lower or time_lower in param_times:
                result.append(p)
            elif not param_time and not param_times:
                result.append(p)
        return result
