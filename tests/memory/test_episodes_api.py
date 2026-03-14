"""Tests for the episodic memory API endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blurt.api.episodes import router, set_store
from blurt.memory.episodic import InMemoryEpisodicStore


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    application.include_router(router)
    return application


@pytest.fixture
def store() -> InMemoryEpisodicStore:
    s = InMemoryEpisodicStore()
    set_store(s)
    return s


@pytest.fixture
def client(app: FastAPI, store: InMemoryEpisodicStore) -> TestClient:
    return TestClient(app)


def _episode_payload(**overrides) -> dict:
    base = {
        "user_id": "user-1",
        "raw_text": "Pick up groceries from Trader Joe's",
        "modality": "voice",
        "intent": "task",
        "intent_confidence": 0.92,
        "emotion": {
            "primary": "anticipation",
            "intensity": 1.0,
            "valence": 0.3,
            "arousal": 0.4,
        },
        "entities": [
            {
                "name": "Trader Joe's",
                "entity_type": "place",
                "confidence": 0.95,
            }
        ],
        "context": {
            "time_of_day": "morning",
            "day_of_week": "monday",
            "session_id": "sess-1",
        },
    }
    base.update(overrides)
    return base


class TestCreateEpisode:
    def test_create_episode_success(self, client: TestClient):
        resp = client.post("/api/v1/episodes", json=_episode_payload())
        assert resp.status_code == 201
        data = resp.json()
        assert data["raw_text"] == "Pick up groceries from Trader Joe's"
        assert data["intent"] == "task"
        assert data["intent_confidence"] == 0.92
        assert data["emotion"]["primary"] == "anticipation"
        assert len(data["entities"]) == 1
        assert data["entities"][0]["name"] == "Trader Joe's"
        assert data["is_compressed"] is False
        assert "id" in data
        assert "timestamp" in data

    def test_create_episode_minimal(self, client: TestClient):
        payload = _episode_payload(entities=[])
        resp = client.post("/api/v1/episodes", json=payload)
        assert resp.status_code == 201

    def test_create_episode_with_embedding(self, client: TestClient):
        payload = _episode_payload(embedding=[0.1, 0.2, 0.3])
        resp = client.post("/api/v1/episodes", json=payload)
        assert resp.status_code == 201

    def test_create_episode_invalid_confidence(self, client: TestClient):
        payload = _episode_payload(intent_confidence=1.5)
        resp = client.post("/api/v1/episodes", json=payload)
        assert resp.status_code == 422


class TestGetEpisode:
    def test_get_existing_episode(self, client: TestClient):
        create_resp = client.post("/api/v1/episodes", json=_episode_payload())
        episode_id = create_resp.json()["id"]

        resp = client.get(f"/api/v1/episodes/{episode_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == episode_id

    def test_get_nonexistent_episode(self, client: TestClient):
        resp = client.get("/api/v1/episodes/nonexistent")
        assert resp.status_code == 404


class TestListEpisodes:
    def test_list_empty(self, client: TestClient):
        resp = client.get("/api/v1/episodes/user/user-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["episodes"] == []
        assert data["total_count"] == 0

    def test_list_with_episodes(self, client: TestClient):
        client.post("/api/v1/episodes", json=_episode_payload(raw_text="one"))
        client.post("/api/v1/episodes", json=_episode_payload(raw_text="two"))

        resp = client.get("/api/v1/episodes/user/user-1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["episodes"]) == 2
        assert data["total_count"] == 2

    def test_list_filter_by_intent(self, client: TestClient):
        client.post("/api/v1/episodes", json=_episode_payload(intent="task"))
        client.post("/api/v1/episodes", json=_episode_payload(intent="idea"))

        resp = client.get("/api/v1/episodes/user/user-1?intent=idea")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["episodes"]) == 1
        assert data["episodes"][0]["intent"] == "idea"

    def test_list_filter_by_entity(self, client: TestClient):
        client.post(
            "/api/v1/episodes",
            json=_episode_payload(
                entities=[{"name": "Sarah", "entity_type": "person", "confidence": 0.9}]
            ),
        )
        client.post("/api/v1/episodes", json=_episode_payload(entities=[]))

        resp = client.get("/api/v1/episodes/user/user-1?entity=Sarah")
        assert resp.status_code == 200
        assert len(resp.json()["episodes"]) == 1

    def test_list_filter_by_emotion(self, client: TestClient):
        client.post(
            "/api/v1/episodes",
            json=_episode_payload(
                emotion={"primary": "joy", "intensity": 2.0, "valence": 0.8, "arousal": 0.6}
            ),
        )
        client.post(
            "/api/v1/episodes",
            json=_episode_payload(
                emotion={
                    "primary": "sadness",
                    "intensity": 2.0,
                    "valence": -0.7,
                    "arousal": 0.3,
                }
            ),
        )

        resp = client.get("/api/v1/episodes/user/user-1?emotion=joy")
        assert resp.status_code == 200
        assert len(resp.json()["episodes"]) == 1

    def test_list_pagination(self, client: TestClient):
        for i in range(5):
            client.post("/api/v1/episodes", json=_episode_payload(raw_text=f"ep {i}"))

        resp = client.get("/api/v1/episodes/user/user-1?limit=2&offset=0")
        data = resp.json()
        assert len(data["episodes"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0

    def test_list_isolates_users(self, client: TestClient):
        client.post("/api/v1/episodes", json=_episode_payload(user_id="user-1"))
        client.post("/api/v1/episodes", json=_episode_payload(user_id="user-2"))

        resp = client.get("/api/v1/episodes/user/user-1")
        assert len(resp.json()["episodes"]) == 1


class TestSessionEpisodes:
    def test_get_session_episodes(self, client: TestClient):
        ctx = {
            "time_of_day": "morning",
            "day_of_week": "monday",
            "session_id": "sess-A",
        }
        client.post("/api/v1/episodes", json=_episode_payload(context=ctx, raw_text="one"))
        client.post("/api/v1/episodes", json=_episode_payload(context=ctx, raw_text="two"))
        client.post(
            "/api/v1/episodes",
            json=_episode_payload(
                context={**ctx, "session_id": "sess-B"}, raw_text="other"
            ),
        )

        resp = client.get("/api/v1/episodes/session/sess-A")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2


class TestEntityTimeline:
    def test_entity_timeline(self, client: TestClient):
        for i in range(3):
            client.post(
                "/api/v1/episodes",
                json=_episode_payload(
                    raw_text=f"Sarah mention {i}",
                    entities=[
                        {"name": "Sarah", "entity_type": "person", "confidence": 0.9}
                    ],
                ),
            )

        resp = client.get("/api/v1/episodes/entity/user-1/Sarah")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_name"] == "Sarah"
        assert data["count"] == 3


class TestSemanticSearch:
    def test_semantic_search(self, client: TestClient):
        client.post(
            "/api/v1/episodes",
            json=_episode_payload(raw_text="cooking recipes", embedding=[1.0, 0.0, 0.0]),
        )
        client.post(
            "/api/v1/episodes",
            json=_episode_payload(raw_text="coding python", embedding=[0.0, 1.0, 0.0]),
        )

        resp = client.post(
            "/api/v1/episodes/search/semantic",
            json={
                "user_id": "user-1",
                "query_embedding": [1.0, 0.0, 0.0],
                "limit": 5,
                "min_similarity": 0.5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert data["results"][0]["episode"]["raw_text"] == "cooking recipes"
        assert data["results"][0]["similarity"] > 0.9


class TestEmotionTimeline:
    def test_emotion_timeline(self, client: TestClient):
        client.post(
            "/api/v1/episodes",
            json=_episode_payload(
                raw_text="happy",
                emotion={"primary": "joy", "intensity": 2.0, "valence": 0.8, "arousal": 0.6},
            ),
        )

        resp = client.get(
            "/api/v1/episodes/emotions/user-1"
            "?start=2020-01-01T00:00:00Z"
            "&end=2030-01-01T00:00:00Z"
        )
        assert resp.status_code == 200
        assert len(resp.json()) >= 1
