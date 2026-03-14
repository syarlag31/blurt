"""Tests for temporal activity API endpoints.

Validates the REST API for temporal activity aggregation,
including profile retrieval, heatmaps, and interaction recording.
"""

from __future__ import annotations

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from blurt.api.temporal_activity import (
    router,
    set_temporal_service,
)
from blurt.services.temporal_activity import (
    InteractionRecord,
    TemporalActivityService,
    TemporalActivityStore,
)


def _seed_store(store: TemporalActivityStore) -> None:
    """Synchronously seed the store with interaction records."""
    for _ in range(5):
        record = InteractionRecord(
            user_id="u1",
            day_of_week="monday",
            time_of_day="morning",
            hour=9,
            energy_level=0.8,
            valence=0.3,
            primary_emotion="anticipation",
            emotion_intensity=0.6,
            intent="task",
            task_created=True,
            word_count=20,
            modality="voice",
        )
        store._interactions[record.user_id].append(record)
    for _ in range(3):
        record = InteractionRecord(
            user_id="u1",
            day_of_week="monday",
            time_of_day="evening",
            hour=19,
            energy_level=0.3,
            valence=-0.1,
            primary_emotion="sadness",
            emotion_intensity=0.4,
            intent="journal",
            word_count=10,
            modality="text",
        )
        store._interactions[record.user_id].append(record)


@pytest.fixture
def service():
    return TemporalActivityService()


@pytest.fixture
def seeded_service():
    store = TemporalActivityStore()
    _seed_store(store)
    return TemporalActivityService(store=store)


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(router)
    set_temporal_service(service)
    yield TestClient(app)
    set_temporal_service(None)


@pytest.fixture
def seeded_client(seeded_service):
    app = FastAPI()
    app.include_router(router)
    set_temporal_service(seeded_service)
    yield TestClient(app)
    set_temporal_service(None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetProfile:
    def test_empty_profile(self, client):
        resp = client.get("/api/v1/users/u1/temporal/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert data["total_interactions"] == 0

    def test_profile_with_data(self, seeded_client):
        resp = seeded_client.get("/api/v1/users/u1/temporal/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_interactions"] == 8
        assert "buckets" in data
        assert "hourly_buckets" in data


class TestWeeklyHeatmap:
    def test_heatmap_structure(self, client):
        resp = client.get("/api/v1/users/u1/temporal/heatmap")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert len(data["heatmap"]) == 28  # 7 days x 4 slots


class TestHourlyPattern:
    def test_hourly_heatmap_structure(self, client):
        resp = client.get("/api/v1/users/u1/temporal/hourly")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["hourly_heatmap"]) == 168  # 7 days x 24 hours

    def test_hourly_filtered_by_day(self, seeded_client):
        resp = seeded_client.get("/api/v1/users/u1/temporal/hourly?day=monday")
        assert resp.status_code == 200
        data = resp.json()
        # Should only have monday entries (24 hours)
        assert len(data["hourly_heatmap"]) == 24
        assert all(c["day_of_week"] == "monday" for c in data["hourly_heatmap"])


class TestEnergyPattern:
    def test_energy_pattern(self, seeded_client):
        resp = seeded_client.get("/api/v1/users/u1/temporal/energy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert data["total_interactions"] == 8
        assert len(data["weekly_heatmap"]) == 28


class TestMoodPattern:
    def test_mood_pattern(self, seeded_client):
        resp = seeded_client.get("/api/v1/users/u1/temporal/mood")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert len(data["day_summaries"]) == 7


class TestRecordInteraction:
    def test_record_basic(self, client):
        resp = client.post("/api/v1/users/u1/temporal/record", json={
            "energy_level": 0.7,
            "valence": 0.3,
            "primary_emotion": "joy",
            "emotion_intensity": 0.8,
            "intent": "task",
            "word_count": 15,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert data["energy_level"] == 0.7
        assert data["primary_emotion"] == "joy"

    def test_record_updates_profile(self, client):
        client.post("/api/v1/users/u1/temporal/record", json={
            "energy_level": 0.9,
        })
        # Verify via profile endpoint
        resp = client.get("/api/v1/users/u1/temporal/profile")
        assert resp.json()["total_interactions"] == 1

    def test_record_with_episode_id(self, client):
        resp = client.post("/api/v1/users/u1/temporal/record", json={
            "energy_level": 0.5,
            "episode_id": "ep-abc-123",
        })
        assert resp.status_code == 200
        assert resp.json()["episode_id"] == "ep-abc-123"

    def test_record_validation_energy_bounds(self, client):
        resp = client.post("/api/v1/users/u1/temporal/record", json={
            "energy_level": 1.5,  # out of bounds
        })
        assert resp.status_code == 422  # validation error

    def test_record_defaults(self, client):
        resp = client.post("/api/v1/users/u1/temporal/record", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["energy_level"] == 0.5
        assert data["intent"] == "journal"
        assert data["modality"] == "voice"


class TestUserIsolation:
    def test_profiles_isolated(self, client):
        # Record for two users via API
        client.post("/api/v1/users/u1/temporal/record", json={"energy_level": 0.8})
        client.post("/api/v1/users/u2/temporal/record", json={"energy_level": 0.3})

        r1 = client.get("/api/v1/users/u1/temporal/profile")
        r2 = client.get("/api/v1/users/u2/temporal/profile")
        assert r1.json()["total_interactions"] == 1
        assert r2.json()["total_interactions"] == 1
