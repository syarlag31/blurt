"""Pattern storage service — persists and queries learned user rhythms.

Behavioral patterns are detected from accumulated episodic memory observations
and stored here for querying. Patterns include energy rhythms, mood cycles,
time-of-day preferences, day-of-week habits, and completion/skip signals.

The service supports:
- Creating patterns from detection pipeline output
- Querying patterns by type, day, time, and confidence threshold
- Updating pattern confidence as new evidence accumulates
- Deactivating patterns that no longer hold true

Anti-shame design: Patterns are neutral observations, never judgments.
No pattern implies "you should" or "you failed to."
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Protocol

from blurt.models.entities import LearnedPattern, PatternType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Storage protocol
# ---------------------------------------------------------------------------


class PatternStore(Protocol):
    """Abstract protocol for pattern persistence."""

    async def save(self, pattern: LearnedPattern) -> LearnedPattern:
        """Persist a new or updated pattern."""
        ...

    async def get(self, pattern_id: str) -> LearnedPattern | None:
        """Retrieve a pattern by ID."""
        ...

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
        """Query patterns with filters."""
        ...

    async def count(self, user_id: str, *, is_active: bool = True) -> int:
        """Count patterns for a user."""
        ...

    async def delete(self, pattern_id: str) -> bool:
        """Hard-delete a pattern. Returns True if found."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


class InMemoryPatternStore:
    """In-memory pattern store for development and testing.

    Thread-safe for single-process async usage. Patterns are indexed
    by user_id for efficient querying.
    """

    def __init__(self) -> None:
        self._patterns: dict[str, LearnedPattern] = {}  # id -> pattern
        self._user_index: dict[str, list[str]] = defaultdict(list)  # user_id -> [ids]

    async def save(self, pattern: LearnedPattern) -> LearnedPattern:
        """Persist a pattern. If ID exists, updates in place."""
        if not pattern.id:
            pattern.id = str(uuid.uuid4())

        pattern.updated_at = _utcnow()

        existing = pattern.id in self._patterns
        self._patterns[pattern.id] = pattern

        if not existing:
            self._user_index[pattern.user_id].append(pattern.id)

        return pattern

    async def get(self, pattern_id: str) -> LearnedPattern | None:
        return self._patterns.get(pattern_id)

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
        """Query patterns for a user with optional filters.

        Filters:
        - pattern_type: Match exact PatternType enum value
        - day_of_week: Match patterns whose parameters contain this day
        - time_of_day: Match patterns whose parameters contain this time slot
        - min_confidence: Minimum confidence threshold
        - is_active: Only active patterns (default True)
        """
        pattern_ids = self._user_index.get(user_id, [])
        results: list[LearnedPattern] = []

        for pid in pattern_ids:
            p = self._patterns.get(pid)
            if p is None:
                continue

            # Active filter
            if p.is_active != is_active:
                continue

            # Confidence filter
            if p.confidence < min_confidence:
                continue

            # Type filter
            if pattern_type is not None and p.pattern_type != pattern_type:
                continue

            # Day filter — check parameters dict for day_of_week or days key
            if day_of_week is not None:
                day_lower = day_of_week.lower()
                param_day = p.parameters.get("day_of_week", "")
                param_days = p.parameters.get("days", [])
                if isinstance(param_day, str):
                    param_day = param_day.lower()
                if isinstance(param_days, list):
                    param_days = [d.lower() for d in param_days]
                if param_day != day_lower and day_lower not in param_days:
                    # Also check if no day info at all — include pattern
                    if param_day or param_days:
                        continue

            # Time filter — check parameters dict for time_of_day or times key
            if time_of_day is not None:
                time_lower = time_of_day.lower()
                param_time = p.parameters.get("time_of_day", "")
                param_times = p.parameters.get("times", [])
                if isinstance(param_time, str):
                    param_time = param_time.lower()
                if isinstance(param_times, list):
                    param_times = [t.lower() for t in param_times]
                if param_time != time_lower and time_lower not in param_times:
                    if param_time or param_times:
                        continue

            results.append(p)

        # Sort by confidence descending, then by last_confirmed descending
        results.sort(key=lambda p: (-p.confidence, -p.last_confirmed.timestamp()))

        return results[offset : offset + limit]

    async def count(self, user_id: str, *, is_active: bool = True) -> int:
        pattern_ids = self._user_index.get(user_id, [])
        return sum(
            1
            for pid in pattern_ids
            if (p := self._patterns.get(pid)) is not None and p.is_active == is_active
        )

    async def delete(self, pattern_id: str) -> bool:
        pattern = self._patterns.pop(pattern_id, None)
        if pattern is None:
            return False
        user_ids = self._user_index.get(pattern.user_id, [])
        if pattern_id in user_ids:
            user_ids.remove(pattern_id)
        return True


# ---------------------------------------------------------------------------
# Pattern service — business logic layer
# ---------------------------------------------------------------------------


# Map of user-friendly type names to PatternType enum values
PATTERN_TYPE_ALIASES: dict[str, PatternType] = {
    "energy": PatternType.ENERGY_RHYTHM,
    "energy_rhythm": PatternType.ENERGY_RHYTHM,
    "mood": PatternType.MOOD_CYCLE,
    "mood_cycle": PatternType.MOOD_CYCLE,
    "time": PatternType.TIME_OF_DAY,
    "time_of_day": PatternType.TIME_OF_DAY,
    "day": PatternType.DAY_OF_WEEK,
    "day_of_week": PatternType.DAY_OF_WEEK,
    "completion": PatternType.COMPLETION_SIGNAL,
    "completion_signal": PatternType.COMPLETION_SIGNAL,
    "skip": PatternType.SKIP_SIGNAL,
    "skip_signal": PatternType.SKIP_SIGNAL,
    "entity": PatternType.ENTITY_PATTERN,
    "entity_pattern": PatternType.ENTITY_PATTERN,
}


def resolve_pattern_type(type_str: str) -> PatternType | None:
    """Resolve a user-friendly type string to a PatternType enum.

    Supports both exact enum values and friendly aliases:
    - "energy" or "energy_rhythm" -> PatternType.ENERGY_RHYTHM
    - "mood" or "mood_cycle" -> PatternType.MOOD_CYCLE
    """
    normalized = type_str.lower().strip()

    # Check aliases first
    if normalized in PATTERN_TYPE_ALIASES:
        return PATTERN_TYPE_ALIASES[normalized]

    # Try direct enum value
    try:
        return PatternType(normalized)
    except ValueError:
        return None


class PatternService:
    """Service for managing learned behavioral patterns.

    Handles CRUD operations and querying with filtering.
    Patterns are neutral observations — never judgmental language.
    """

    def __init__(self, store: PatternStore | None = None) -> None:
        self._store: PatternStore = store or InMemoryPatternStore()

    @property
    def store(self) -> PatternStore:
        return self._store

    async def create_pattern(
        self,
        user_id: str,
        pattern_type: PatternType,
        description: str,
        parameters: dict[str, Any] | None = None,
        confidence: float = 0.5,
        observation_count: int = 1,
        supporting_evidence: list[str] | None = None,
    ) -> LearnedPattern:
        """Create and persist a new learned pattern."""
        pattern = LearnedPattern(
            user_id=user_id,
            pattern_type=pattern_type,
            description=description,
            parameters=parameters or {},
            confidence=confidence,
            observation_count=observation_count,
            supporting_evidence=supporting_evidence or [],
        )
        return await self._store.save(pattern)

    async def get_pattern(self, pattern_id: str) -> LearnedPattern | None:
        """Retrieve a pattern by ID."""
        return await self._store.get(pattern_id)

    async def query_patterns(
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
        """Query patterns with filters."""
        return await self._store.query(
            user_id,
            pattern_type=pattern_type,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
            min_confidence=min_confidence,
            is_active=is_active,
            limit=limit,
            offset=offset,
        )

    async def reinforce_pattern(
        self,
        pattern_id: str,
        evidence: str | None = None,
        confidence_boost: float = 0.05,
    ) -> LearnedPattern | None:
        """Reinforce a pattern with new evidence, increasing confidence.

        This is the compounding mechanism — patterns strengthen with
        each confirming observation.
        """
        pattern = await self._store.get(pattern_id)
        if pattern is None:
            return None

        pattern.observation_count += 1
        pattern.confidence = min(1.0, pattern.confidence + confidence_boost)
        pattern.last_confirmed = _utcnow()

        if evidence:
            pattern.supporting_evidence.append(evidence)
            # Keep evidence list bounded
            if len(pattern.supporting_evidence) > 50:
                pattern.supporting_evidence = pattern.supporting_evidence[-50:]

        return await self._store.save(pattern)

    async def weaken_pattern(
        self,
        pattern_id: str,
        confidence_penalty: float = 0.1,
    ) -> LearnedPattern | None:
        """Weaken a pattern when contradicting evidence is observed.

        Patterns that drop below 0.1 confidence are auto-deactivated.
        """
        pattern = await self._store.get(pattern_id)
        if pattern is None:
            return None

        pattern.confidence = max(0.0, pattern.confidence - confidence_penalty)
        pattern.updated_at = _utcnow()

        # Auto-deactivate very weak patterns
        if pattern.confidence < 0.1:
            pattern.is_active = False

        return await self._store.save(pattern)

    async def deactivate_pattern(self, pattern_id: str) -> LearnedPattern | None:
        """Deactivate a pattern (soft delete). It remains in storage for history."""
        pattern = await self._store.get(pattern_id)
        if pattern is None:
            return None

        pattern.is_active = False
        pattern.updated_at = _utcnow()
        return await self._store.save(pattern)

    async def count_patterns(self, user_id: str, *, is_active: bool = True) -> int:
        """Count patterns for a user."""
        return await self._store.count(user_id, is_active=is_active)

    async def get_pattern_summary(self, user_id: str) -> dict[str, Any]:
        """Get a summary of all patterns for a user, grouped by type.

        Returns counts and average confidence per pattern type.
        """
        all_patterns = await self._store.query(
            user_id, is_active=True, limit=1000
        )

        summary: dict[str, Any] = {
            "total_active": len(all_patterns),
            "by_type": {},
        }

        type_groups: dict[str, list[LearnedPattern]] = defaultdict(list)
        for p in all_patterns:
            type_groups[p.pattern_type.value].append(p)

        for ptype, patterns in type_groups.items():
            avg_conf = sum(p.confidence for p in patterns) / len(patterns)
            summary["by_type"][ptype] = {
                "count": len(patterns),
                "avg_confidence": round(avg_conf, 3),
                "strongest": max(p.confidence for p in patterns),
            }

        return summary
