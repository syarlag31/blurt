"""Tests for pattern storage API and service.

Validates:
- Pattern CRUD operations (create, read, query, deactivate)
- Query filtering by type, day, time, confidence
- Pattern reinforcement (compounding) and weakening
- Anti-shame: no judgmental language in pattern descriptions
- Friendly type alias resolution (e.g., "energy" -> ENERGY_RHYTHM)
- API endpoint integration via FastAPI TestClient
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blurt.api.patterns import (
    router,
    set_pattern_service,
)
from blurt.models.entities import PatternType
from blurt.services.patterns import (
    InMemoryPatternStore,
    PatternService,
    resolve_pattern_type,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryPatternStore:
    return InMemoryPatternStore()


@pytest.fixture
def service(store: InMemoryPatternStore) -> PatternService:
    return PatternService(store=store)


@pytest.fixture
def app(service: PatternService) -> Generator[FastAPI, None, None]:
    test_app = FastAPI()
    test_app.include_router(router)
    set_pattern_service(service)
    yield test_app
    set_pattern_service(None)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


USER_ID = "user-test-123"


# ---------------------------------------------------------------------------
# Unit tests — PatternService
# ---------------------------------------------------------------------------


class TestResolvePatternType:
    def test_friendly_aliases(self):
        assert resolve_pattern_type("energy") == PatternType.ENERGY_RHYTHM
        assert resolve_pattern_type("mood") == PatternType.MOOD_CYCLE
        assert resolve_pattern_type("time") == PatternType.TIME_OF_DAY
        assert resolve_pattern_type("day") == PatternType.DAY_OF_WEEK
        assert resolve_pattern_type("completion") == PatternType.COMPLETION_SIGNAL
        assert resolve_pattern_type("skip") == PatternType.SKIP_SIGNAL
        assert resolve_pattern_type("entity") == PatternType.ENTITY_PATTERN

    def test_exact_enum_values(self):
        assert resolve_pattern_type("energy_rhythm") == PatternType.ENERGY_RHYTHM
        assert resolve_pattern_type("mood_cycle") == PatternType.MOOD_CYCLE

    def test_case_insensitive(self):
        assert resolve_pattern_type("ENERGY") == PatternType.ENERGY_RHYTHM
        assert resolve_pattern_type("Mood") == PatternType.MOOD_CYCLE

    def test_unknown_returns_none(self):
        assert resolve_pattern_type("unknown_type") is None
        assert resolve_pattern_type("") is None


class TestPatternService:
    @pytest.mark.asyncio
    async def test_create_pattern(self, service: PatternService):
        pattern = await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Tends to have higher energy in mornings",
            parameters={"time_of_day": "morning", "energy_level": "high"},
            confidence=0.7,
        )
        assert pattern.id
        assert pattern.user_id == USER_ID
        assert pattern.pattern_type == PatternType.ENERGY_RHYTHM
        assert pattern.confidence == 0.7
        assert pattern.is_active is True

    @pytest.mark.asyncio
    async def test_get_pattern(self, service: PatternService):
        created = await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.MOOD_CYCLE,
            description="More reflective on Sunday evenings",
        )
        retrieved = await service.get_pattern(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, service: PatternService):
        assert await service.get_pattern("nonexistent-id") is None

    @pytest.mark.asyncio
    async def test_query_by_type(self, service: PatternService):
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Morning energy",
        )
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.MOOD_CYCLE,
            description="Weekend mood",
        )

        energy_patterns = await service.query_patterns(
            USER_ID, pattern_type=PatternType.ENERGY_RHYTHM
        )
        assert len(energy_patterns) == 1
        assert energy_patterns[0].pattern_type == PatternType.ENERGY_RHYTHM

    @pytest.mark.asyncio
    async def test_query_by_day(self, service: PatternService):
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Thursday afternoon energy dip",
            parameters={"day_of_week": "thursday", "time_of_day": "afternoon"},
        )
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Monday morning energy spike",
            parameters={"day_of_week": "monday", "time_of_day": "morning"},
        )

        thursday = await service.query_patterns(USER_ID, day_of_week="thursday")
        assert len(thursday) == 1
        assert "Thursday" in thursday[0].description

    @pytest.mark.asyncio
    async def test_query_by_time(self, service: PatternService):
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.TIME_OF_DAY,
            description="Prefers deep work in morning",
            parameters={"time_of_day": "morning"},
        )
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.TIME_OF_DAY,
            description="Creative work in evening",
            parameters={"time_of_day": "evening"},
        )

        morning = await service.query_patterns(USER_ID, time_of_day="morning")
        assert len(morning) == 1
        assert "morning" in morning[0].description

    @pytest.mark.asyncio
    async def test_query_by_min_confidence(self, service: PatternService):
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Weak pattern",
            confidence=0.3,
        )
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Strong pattern",
            confidence=0.9,
        )

        strong = await service.query_patterns(USER_ID, min_confidence=0.7)
        assert len(strong) == 1
        assert strong[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_reinforce_pattern(self, service: PatternService):
        pattern = await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.COMPLETION_SIGNAL,
            description="Tends to complete tasks after coffee",
            confidence=0.5,
        )

        reinforced = await service.reinforce_pattern(
            pattern.id, evidence="blurt-abc-123", confidence_boost=0.1
        )
        assert reinforced is not None
        assert reinforced.confidence == 0.6
        assert reinforced.observation_count == 2
        assert "blurt-abc-123" in reinforced.supporting_evidence

    @pytest.mark.asyncio
    async def test_weaken_pattern(self, service: PatternService):
        pattern = await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.SKIP_SIGNAL,
            description="Sometimes skips afternoon tasks",
            confidence=0.5,
        )

        weakened = await service.weaken_pattern(pattern.id, confidence_penalty=0.1)
        assert weakened is not None
        assert weakened.confidence == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_weaken_auto_deactivates(self, service: PatternService):
        pattern = await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.SKIP_SIGNAL,
            description="Very weak pattern",
            confidence=0.15,
        )

        weakened = await service.weaken_pattern(pattern.id, confidence_penalty=0.1)
        assert weakened is not None
        assert weakened.confidence == pytest.approx(0.05)
        assert weakened.is_active is False

    @pytest.mark.asyncio
    async def test_deactivate_pattern(self, service: PatternService):
        pattern = await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Old pattern",
        )

        deactivated = await service.deactivate_pattern(pattern.id)
        assert deactivated is not None
        assert deactivated.is_active is False

        # Should not appear in active queries
        active = await service.query_patterns(USER_ID)
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_count_patterns(self, service: PatternService):
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Pattern 1",
        )
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.MOOD_CYCLE,
            description="Pattern 2",
        )

        assert await service.count_patterns(USER_ID) == 2

    @pytest.mark.asyncio
    async def test_pattern_summary(self, service: PatternService):
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Energy 1",
            confidence=0.8,
        )
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.ENERGY_RHYTHM,
            description="Energy 2",
            confidence=0.6,
        )
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.MOOD_CYCLE,
            description="Mood 1",
            confidence=0.9,
        )

        summary = await service.get_pattern_summary(USER_ID)
        assert summary["total_active"] == 3
        assert "energy_rhythm" in summary["by_type"]
        assert summary["by_type"]["energy_rhythm"]["count"] == 2
        assert summary["by_type"]["energy_rhythm"]["avg_confidence"] == 0.7
        assert summary["by_type"]["mood_cycle"]["count"] == 1

    @pytest.mark.asyncio
    async def test_query_with_days_list(self, service: PatternService):
        """Patterns can store multiple days in a 'days' list."""
        await service.create_pattern(
            user_id=USER_ID,
            pattern_type=PatternType.DAY_OF_WEEK,
            description="Productive on weekdays",
            parameters={"days": ["monday", "tuesday", "wednesday", "thursday", "friday"]},
        )

        thursday = await service.query_patterns(USER_ID, day_of_week="thursday")
        assert len(thursday) == 1

        saturday = await service.query_patterns(USER_ID, day_of_week="saturday")
        assert len(saturday) == 0


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestPatternAPI:
    def test_create_pattern(self, client: TestClient):
        resp = client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "energy",
                "description": "Higher energy on Thursday mornings",
                "parameters": {
                    "day_of_week": "thursday",
                    "time_of_day": "morning",
                    "energy_level": "high",
                },
                "confidence": 0.75,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["pattern_type"] == "energy_rhythm"
        assert data["confidence"] == 0.75
        assert data["is_active"] is True
        assert data["user_id"] == USER_ID

    def test_create_pattern_invalid_type(self, client: TestClient):
        resp = client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "invalid_type",
                "description": "Should fail",
            },
        )
        assert resp.status_code == 400

    def test_get_pattern(self, client: TestClient):
        # Create
        create_resp = client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "mood",
                "description": "Reflective on Sundays",
                "parameters": {"day_of_week": "sunday"},
            },
        )
        pattern_id = create_resp.json()["id"]

        # Get
        resp = client.get(f"/api/v1/users/{USER_ID}/patterns/{pattern_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == pattern_id

    def test_get_pattern_not_found(self, client: TestClient):
        resp = client.get(f"/api/v1/users/{USER_ID}/patterns/nonexistent")
        assert resp.status_code == 404

    def test_list_patterns_with_type_filter(self, client: TestClient):
        # Create mixed patterns
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={"pattern_type": "energy", "description": "Energy pattern"},
        )
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={"pattern_type": "mood", "description": "Mood pattern"},
        )

        # Filter by energy
        resp = client.get(f"/api/v1/users/{USER_ID}/patterns?type=energy")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["pattern_type"] == "energy_rhythm"

    def test_list_patterns_with_day_filter(self, client: TestClient):
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "energy",
                "description": "Thursday energy",
                "parameters": {"day_of_week": "thursday"},
            },
        )
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "energy",
                "description": "Monday energy",
                "parameters": {"day_of_week": "monday"},
            },
        )

        resp = client.get(f"/api/v1/users/{USER_ID}/patterns?type=energy&day=thursday")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["patterns"]) == 1
        assert "Thursday" in data["patterns"][0]["description"]

    def test_list_patterns_invalid_type(self, client: TestClient):
        resp = client.get(f"/api/v1/users/{USER_ID}/patterns?type=bogus")
        assert resp.status_code == 400

    def test_reinforce_pattern(self, client: TestClient):
        create_resp = client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "completion",
                "description": "Completes tasks after coffee",
                "confidence": 0.5,
            },
        )
        pattern_id = create_resp.json()["id"]

        resp = client.put(
            f"/api/v1/users/{USER_ID}/patterns/{pattern_id}/reinforce",
            json={"evidence": "blurt-xyz", "confidence_boost": 0.1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == pytest.approx(0.6, abs=0.001)
        assert data["observation_count"] == 2
        assert "blurt-xyz" in data["supporting_evidence"]

    def test_weaken_pattern(self, client: TestClient):
        create_resp = client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "skip",
                "description": "Skip signal",
                "confidence": 0.5,
            },
        )
        pattern_id = create_resp.json()["id"]

        resp = client.put(
            f"/api/v1/users/{USER_ID}/patterns/{pattern_id}/weaken",
            json={"confidence_penalty": 0.1},
        )
        assert resp.status_code == 200
        assert resp.json()["confidence"] == pytest.approx(0.4, abs=0.001)

    def test_deactivate_pattern(self, client: TestClient):
        create_resp = client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={"pattern_type": "entity", "description": "Entity pattern"},
        )
        pattern_id = create_resp.json()["id"]

        resp = client.delete(f"/api/v1/users/{USER_ID}/patterns/{pattern_id}")
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

        # Should not appear in active list
        list_resp = client.get(f"/api/v1/users/{USER_ID}/patterns")
        assert len(list_resp.json()["patterns"]) == 0

    def test_pattern_summary(self, client: TestClient):
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={"pattern_type": "energy", "description": "E1", "confidence": 0.8},
        )
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={"pattern_type": "energy", "description": "E2", "confidence": 0.6},
        )
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={"pattern_type": "mood", "description": "M1", "confidence": 0.9},
        )

        resp = client.get(f"/api/v1/users/{USER_ID}/patterns/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_active"] == 3
        assert "energy_rhythm" in data["by_type"]
        assert data["by_type"]["energy_rhythm"]["count"] == 2

    def test_cross_user_isolation(self, client: TestClient):
        """Patterns from one user should not appear for another."""
        client.post(
            "/api/v1/users/user-a/patterns",
            json={"pattern_type": "energy", "description": "User A pattern"},
        )
        client.post(
            "/api/v1/users/user-b/patterns",
            json={"pattern_type": "energy", "description": "User B pattern"},
        )

        resp_a = client.get("/api/v1/users/user-a/patterns")
        resp_b = client.get("/api/v1/users/user-b/patterns")
        assert len(resp_a.json()["patterns"]) == 1
        assert len(resp_b.json()["patterns"]) == 1
        assert resp_a.json()["patterns"][0]["description"] == "User A pattern"
        assert resp_b.json()["patterns"][0]["description"] == "User B pattern"

    def test_pagination(self, client: TestClient):
        """Test limit and offset parameters."""
        for i in range(5):
            client.post(
                f"/api/v1/users/{USER_ID}/patterns",
                json={
                    "pattern_type": "energy",
                    "description": f"Pattern {i}",
                    "confidence": 0.5 + i * 0.1,
                },
            )

        resp = client.get(f"/api/v1/users/{USER_ID}/patterns?limit=2&offset=0")
        data = resp.json()
        assert len(data["patterns"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0

        resp2 = client.get(f"/api/v1/users/{USER_ID}/patterns?limit=2&offset=2")
        data2 = resp2.json()
        assert len(data2["patterns"]) == 2

    def test_combined_filters(self, client: TestClient):
        """Test type + day + time combined filter."""
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "energy",
                "description": "Thursday morning energy high",
                "parameters": {"day_of_week": "thursday", "time_of_day": "morning"},
                "confidence": 0.8,
            },
        )
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "energy",
                "description": "Thursday evening energy low",
                "parameters": {"day_of_week": "thursday", "time_of_day": "evening"},
                "confidence": 0.7,
            },
        )
        client.post(
            f"/api/v1/users/{USER_ID}/patterns",
            json={
                "pattern_type": "mood",
                "description": "Thursday mood shift",
                "parameters": {"day_of_week": "thursday"},
                "confidence": 0.6,
            },
        )

        # type=energy & day=thursday & time=morning
        resp = client.get(
            f"/api/v1/users/{USER_ID}/patterns?type=energy&day=thursday&time=morning"
        )
        data = resp.json()
        assert len(data["patterns"]) == 1
        assert "morning" in data["patterns"][0]["description"]
