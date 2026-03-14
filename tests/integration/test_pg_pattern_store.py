"""Integration tests for PgPatternStore — Neon Postgres persistence of learned user rhythms.

Verifies that learned behavioral patterns (energy rhythms, mood cycles,
day-of-week habits, time-of-day preferences, etc.) are correctly:
  1. Persisted to Postgres via save()
  2. Retrieved via get() and query()
  3. Updated via upsert (save with same ID)
  4. Deleted via delete()
  5. Filtered by pattern_type, day_of_week, time_of_day, confidence, is_active
  6. Counted correctly
  7. Isolated between users

Requires DATABASE_URL env var pointing to a Neon Postgres instance with pgvector.
Tests are skipped if DATABASE_URL is not set.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

import pytest
import pytest_asyncio

# Skip entire module if no DATABASE_URL
pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping Postgres integration tests",
)


@pytest_asyncio.fixture
async def pool():
    """Create an asyncpg connection pool for tests."""
    import asyncpg

    dsn = os.environ["DATABASE_URL"]
    p = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    # Ensure pgvector extension and table exist
    async with p.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
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
    yield p
    # Clean up test data (scoped by user_id prefix) then close
    async with p.acquire() as conn:
        await conn.execute(
            "DELETE FROM learned_patterns WHERE user_id LIKE $1",
            "test-pg-pattern-%",
        )
    await p.close()


@pytest.fixture
def user_id() -> str:
    """Unique user ID per test run to avoid cross-test interference."""
    return f"test-pg-pattern-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def store(pool):
    """PgPatternStore backed by the test pool."""
    from blurt.persistence.pattern_store import PgPatternStore

    return PgPatternStore(pool)


def _make_pattern(user_id: str, **overrides):
    """Helper to build a LearnedPattern with sensible defaults."""
    from blurt.models.entities import LearnedPattern, PatternType

    defaults = dict(
        user_id=user_id,
        pattern_type=PatternType.ENERGY_RHYTHM,
        description="User is more productive in mornings",
        parameters={"time_of_day": "morning"},
        confidence=0.7,
        observation_count=5,
        supporting_evidence=["blurt-1", "blurt-2"],
        is_active=True,
    )
    defaults.update(overrides)
    return LearnedPattern(**defaults)  # type: ignore[arg-type]


# ── CRUD basics ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_and_get(store, user_id):
    """Pattern survives a round-trip through save() → get()."""
    pattern = _make_pattern(user_id)
    saved = await store.save(pattern)
    assert saved.id == pattern.id

    fetched = await store.get(pattern.id)
    assert fetched is not None
    assert fetched.id == pattern.id
    assert fetched.user_id == user_id
    assert fetched.pattern_type.value == "energy_rhythm"
    assert fetched.description == "User is more productive in mornings"
    assert fetched.parameters == {"time_of_day": "morning"}
    assert fetched.confidence == pytest.approx(0.7)
    assert fetched.observation_count == 5
    assert fetched.supporting_evidence == ["blurt-1", "blurt-2"]
    assert fetched.is_active is True


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(store):
    """get() returns None for a missing pattern ID."""
    result = await store.get("nonexistent-id-12345")
    assert result is None


@pytest.mark.asyncio
async def test_save_upsert_updates_existing(store, user_id):
    """Saving a pattern with the same ID updates it (upsert semantics)."""
    pattern = _make_pattern(user_id, confidence=0.5, observation_count=3)
    await store.save(pattern)

    # Update fields
    pattern.confidence = 0.85
    pattern.observation_count = 10
    pattern.description = "Updated description"
    pattern.supporting_evidence = ["blurt-1", "blurt-2", "blurt-3"]
    await store.save(pattern)

    fetched = await store.get(pattern.id)
    assert fetched is not None
    assert fetched.confidence == pytest.approx(0.85)
    assert fetched.observation_count == 10
    assert fetched.description == "Updated description"
    assert len(fetched.supporting_evidence) == 3


@pytest.mark.asyncio
async def test_delete_existing_pattern(store, user_id):
    """delete() removes the pattern and returns True."""
    pattern = _make_pattern(user_id)
    await store.save(pattern)

    deleted = await store.delete(pattern.id)
    assert deleted is True

    fetched = await store.get(pattern.id)
    assert fetched is None


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_false(store):
    """delete() returns False for a pattern that doesn't exist."""
    deleted = await store.delete("nonexistent-id-67890")
    assert deleted is False


# ── Query and filtering ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_by_user_id(store, user_id):
    """query() returns only patterns for the given user."""

    other_user = f"test-pg-pattern-other-{uuid.uuid4().hex[:8]}"

    await store.save(_make_pattern(user_id, description="my pattern"))
    await store.save(_make_pattern(other_user, description="other pattern"))

    results = await store.query(user_id)
    assert all(r.user_id == user_id for r in results)
    assert any(r.description == "my pattern" for r in results)
    assert not any(r.description == "other pattern" for r in results)

    # Clean up other user's data
    other_results = await store.query(other_user)
    for r in other_results:
        await store.delete(r.id)


@pytest.mark.asyncio
async def test_query_by_pattern_type(store, user_id):
    """query() filters by pattern_type."""
    from blurt.models.entities import PatternType

    await store.save(_make_pattern(user_id, pattern_type=PatternType.ENERGY_RHYTHM))
    await store.save(_make_pattern(user_id, pattern_type=PatternType.DAY_OF_WEEK))
    await store.save(_make_pattern(user_id, pattern_type=PatternType.MOOD_CYCLE))

    results = await store.query(user_id, pattern_type=PatternType.DAY_OF_WEEK)
    assert len(results) == 1
    assert results[0].pattern_type == PatternType.DAY_OF_WEEK


@pytest.mark.asyncio
async def test_query_by_min_confidence(store, user_id):
    """query() filters out patterns below min_confidence."""
    await store.save(_make_pattern(user_id, confidence=0.3, description="low"))
    await store.save(_make_pattern(user_id, confidence=0.7, description="medium"))
    await store.save(_make_pattern(user_id, confidence=0.95, description="high"))

    results = await store.query(user_id, min_confidence=0.6)
    descriptions = {r.description for r in results}
    assert "low" not in descriptions
    assert "medium" in descriptions
    assert "high" in descriptions


@pytest.mark.asyncio
async def test_query_active_vs_inactive(store, user_id):
    """query() respects the is_active filter."""
    await store.save(
        _make_pattern(user_id, is_active=True, description="active")
    )
    await store.save(
        _make_pattern(user_id, is_active=False, description="inactive")
    )

    active = await store.query(user_id, is_active=True)
    inactive = await store.query(user_id, is_active=False)

    assert all(r.is_active for r in active)
    assert any(r.description == "active" for r in active)
    assert all(not r.is_active for r in inactive)
    assert any(r.description == "inactive" for r in inactive)


@pytest.mark.asyncio
async def test_query_by_day_of_week(store, user_id):
    """query() filters by day_of_week parameter in JSONB."""
    from blurt.models.entities import PatternType

    await store.save(
        _make_pattern(
            user_id,
            pattern_type=PatternType.DAY_OF_WEEK,
            parameters={"day_of_week": "monday"},
            description="monday pattern",
        )
    )
    await store.save(
        _make_pattern(
            user_id,
            pattern_type=PatternType.DAY_OF_WEEK,
            parameters={"days": ["tuesday", "thursday"]},
            description="tue-thu pattern",
        )
    )

    monday = await store.query(user_id, day_of_week="monday")
    assert any(r.description == "monday pattern" for r in monday)

    tuesday = await store.query(user_id, day_of_week="tuesday")
    assert any(r.description == "tue-thu pattern" for r in tuesday)

    # Saturday should not match either pattern (they have explicit day info)
    saturday = await store.query(user_id, day_of_week="saturday")
    sat_descs = {r.description for r in saturday}
    assert "monday pattern" not in sat_descs
    assert "tue-thu pattern" not in sat_descs


@pytest.mark.asyncio
async def test_query_by_time_of_day(store, user_id):
    """query() filters by time_of_day parameter in JSONB."""
    from blurt.models.entities import PatternType

    await store.save(
        _make_pattern(
            user_id,
            pattern_type=PatternType.TIME_OF_DAY,
            parameters={"time_of_day": "morning"},
            description="morning pattern",
        )
    )
    await store.save(
        _make_pattern(
            user_id,
            pattern_type=PatternType.TIME_OF_DAY,
            parameters={"times": ["afternoon", "evening"]},
            description="pm pattern",
        )
    )

    morning = await store.query(user_id, time_of_day="morning")
    assert any(r.description == "morning pattern" for r in morning)

    evening = await store.query(user_id, time_of_day="evening")
    assert any(r.description == "pm pattern" for r in evening)


@pytest.mark.asyncio
async def test_query_pagination(store, user_id):
    """query() supports limit and offset for pagination."""
    for i in range(5):
        await store.save(
            _make_pattern(user_id, confidence=0.5 + i * 0.1, description=f"p{i}")
        )

    page1 = await store.query(user_id, limit=2, offset=0)
    page2 = await store.query(user_id, limit=2, offset=2)
    page3 = await store.query(user_id, limit=2, offset=4)

    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1

    # No overlap between pages
    ids1 = {r.id for r in page1}
    ids2 = {r.id for r in page2}
    ids3 = {r.id for r in page3}
    assert ids1.isdisjoint(ids2)
    assert ids2.isdisjoint(ids3)


# ── Count ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_count_active(store, user_id):
    """count() returns correct count of active patterns."""
    await store.save(_make_pattern(user_id, is_active=True))
    await store.save(_make_pattern(user_id, is_active=True))
    await store.save(_make_pattern(user_id, is_active=False))

    active_count = await store.count(user_id, is_active=True)
    inactive_count = await store.count(user_id, is_active=False)

    assert active_count == 2
    assert inactive_count == 1


# ── JSONB parameters roundtrip ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_jsonb_parameters_roundtrip(store, user_id):
    """Complex JSONB parameters survive persistence."""
    params = {
        "time_of_day": "morning",
        "days": ["monday", "wednesday", "friday"],
        "threshold": 0.75,
        "nested": {"key": "value", "list": [1, 2, 3]},
    }
    pattern = _make_pattern(user_id, parameters=params)
    await store.save(pattern)

    fetched = await store.get(pattern.id)
    assert fetched is not None
    assert fetched.parameters == params
    assert fetched.parameters["nested"]["list"] == [1, 2, 3]


# ── Supporting evidence array ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supporting_evidence_array(store, user_id):
    """TEXT[] supporting_evidence column round-trips correctly."""
    evidence = ["blurt-aaa", "blurt-bbb", "blurt-ccc"]
    pattern = _make_pattern(user_id, supporting_evidence=evidence)
    await store.save(pattern)

    fetched = await store.get(pattern.id)
    assert fetched is not None
    assert fetched.supporting_evidence == evidence


@pytest.mark.asyncio
async def test_empty_supporting_evidence(store, user_id):
    """Empty supporting_evidence persists as empty list."""
    pattern = _make_pattern(user_id, supporting_evidence=[])
    await store.save(pattern)

    fetched = await store.get(pattern.id)
    assert fetched is not None
    assert fetched.supporting_evidence == []


# ── Timestamps ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_timestamps_persisted(store, user_id):
    """Timestamps (first_detected, last_confirmed, created/updated_at) survive."""
    pattern = _make_pattern(user_id)
    await store.save(pattern)

    fetched = await store.get(pattern.id)
    assert fetched is not None
    assert isinstance(fetched.first_detected, datetime)
    assert isinstance(fetched.last_confirmed, datetime)
    assert isinstance(fetched.created_at, datetime)
    assert isinstance(fetched.updated_at, datetime)


@pytest.mark.asyncio
async def test_updated_at_changes_on_upsert(store, user_id):
    """updated_at advances when a pattern is re-saved."""
    pattern = _make_pattern(user_id)
    await store.save(pattern)

    fetched1 = await store.get(pattern.id)
    assert fetched1 is not None
    old_updated = fetched1.updated_at

    # Re-save with a change
    pattern.confidence = 0.99
    await store.save(pattern)

    fetched2 = await store.get(pattern.id)
    assert fetched2 is not None
    assert fetched2.updated_at >= old_updated


# ── Pattern type roundtrip ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_all_pattern_types_persist(store, user_id):
    """Every PatternType enum value round-trips through the store."""
    from blurt.models.entities import PatternType

    for pt in PatternType:
        pattern = _make_pattern(user_id, pattern_type=pt, description=pt.value)
        await store.save(pattern)

        fetched = await store.get(pattern.id)
        assert fetched is not None
        assert fetched.pattern_type == pt


# ── Deactivate then re-activate ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_deactivate_and_reactivate(store, user_id):
    """A pattern can be deactivated via save(is_active=False) then reactivated."""
    pattern = _make_pattern(user_id, is_active=True)
    await store.save(pattern)

    # Deactivate
    pattern.is_active = False
    await store.save(pattern)
    fetched = await store.get(pattern.id)
    assert fetched is not None
    assert fetched.is_active is False

    # Should not appear in active query
    active = await store.query(user_id, is_active=True)
    assert pattern.id not in {r.id for r in active}

    # Reactivate
    pattern.is_active = True
    await store.save(pattern)
    fetched = await store.get(pattern.id)
    assert fetched is not None
    assert fetched.is_active is True


# ── No pattern without day/time info passes through day/time filter ───────


@pytest.mark.asyncio
async def test_pattern_without_day_info_included_in_day_filter(store, user_id):
    """Patterns with no day_of_week or days parameter are included when filtering by day."""
    await store.save(
        _make_pattern(user_id, parameters={}, description="no day info")
    )
    await store.save(
        _make_pattern(
            user_id,
            parameters={"day_of_week": "monday"},
            description="monday only",
        )
    )

    results = await store.query(user_id, day_of_week="wednesday")
    descs = {r.description for r in results}
    # Pattern with no day info should still be included
    assert "no day info" in descs
    # Monday pattern should NOT match wednesday
    assert "monday only" not in descs


# ── Cross-user isolation ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cross_user_isolation(store, user_id):
    """Patterns from one user do not leak into another user's queries."""
    user_a = user_id
    user_b = f"test-pg-pattern-b-{uuid.uuid4().hex[:8]}"

    await store.save(_make_pattern(user_a, description="user A pattern"))
    await store.save(_make_pattern(user_b, description="user B pattern"))

    a_patterns = await store.query(user_a)
    b_patterns = await store.query(user_b)

    assert all(r.user_id == user_a for r in a_patterns)
    assert all(r.user_id == user_b for r in b_patterns)

    # Clean up user_b
    for r in b_patterns:
        await store.delete(r.id)


# ── Query ordering ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_orders_by_confidence_desc(store, user_id):
    """Results are ordered by confidence DESC."""
    await store.save(_make_pattern(user_id, confidence=0.3, description="low"))
    await store.save(_make_pattern(user_id, confidence=0.9, description="high"))
    await store.save(_make_pattern(user_id, confidence=0.6, description="mid"))

    results = await store.query(user_id)
    confidences = [r.confidence for r in results]
    assert confidences == sorted(confidences, reverse=True)
