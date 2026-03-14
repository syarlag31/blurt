"""Tests for observation storage, compression, summaries, and recall API endpoints.

Covers:
- Storing observations via POST /api/v1/episodes/observations
- Compressing episodes via POST /api/v1/episodes/compress
- Retrieving summaries via GET /api/v1/episodes/summaries/{user_id}
- Full recall (raw + compressed) via GET /api/v1/episodes/recall/{user_id}
"""

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


def _observation_payload(**overrides) -> dict:
    """Default observation payload for tests."""
    base = {
        "user_id": "user-1",
        "raw_text": "I need to call Sarah about the Q2 presentation",
        "modality": "voice",
        "intent": "task",
        "intent_confidence": 0.93,
        "emotion": {
            "primary": "anticipation",
            "intensity": 1.2,
            "valence": 0.4,
            "arousal": 0.5,
        },
        "entities": [
            {"name": "Sarah", "entity_type": "person", "confidence": 0.95},
            {"name": "Q2 Presentation", "entity_type": "project", "confidence": 0.88},
        ],
        "context": {
            "time_of_day": "morning",
            "day_of_week": "tuesday",
            "session_id": "sess-obs-1",
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Observation storage
# ---------------------------------------------------------------------------


class TestStoreObservation:
    def test_store_observation_success(self, client: TestClient):
        resp = client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["observation_stored"] is True
        assert data["entity_count"] == 2
        assert data["episode"]["raw_text"] == "I need to call Sarah about the Q2 presentation"
        assert data["episode"]["intent"] == "task"
        assert data["episode"]["is_compressed"] is False
        assert "id" in data["episode"]
        assert "timestamp" in data["episode"]

    def test_store_observation_with_embedding(self, client: TestClient):
        payload = _observation_payload(embedding=[0.1, 0.2, 0.3, 0.4])
        resp = client.post("/api/v1/episodes/observations", json=payload)
        assert resp.status_code == 201
        assert resp.json()["observation_stored"] is True

    def test_store_observation_no_entities(self, client: TestClient):
        payload = _observation_payload(entities=[])
        resp = client.post("/api/v1/episodes/observations", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["entity_count"] == 0
        assert data["episode"]["entities"] == []

    def test_store_observation_with_source_working_id(self, client: TestClient):
        payload = _observation_payload(source_working_id="wm-42")
        resp = client.post("/api/v1/episodes/observations", json=payload)
        assert resp.status_code == 201
        # The episode was created from working memory
        assert resp.json()["observation_stored"] is True

    def test_store_observation_text_modality(self, client: TestClient):
        payload = _observation_payload(modality="text")
        resp = client.post("/api/v1/episodes/observations", json=payload)
        assert resp.status_code == 201
        assert resp.json()["episode"]["modality"] == "text"

    def test_store_observation_all_intents(self, client: TestClient):
        """Every blurt intent should be storable as an observation."""
        intents = ["task", "event", "reminder", "idea", "journal", "update", "question"]
        for intent in intents:
            payload = _observation_payload(
                intent=intent,
                raw_text=f"observation with intent {intent}",
            )
            resp = client.post("/api/v1/episodes/observations", json=payload)
            assert resp.status_code == 201, f"Failed for intent: {intent}"
            assert resp.json()["episode"]["intent"] == intent

    def test_store_observation_with_behavioral_signal(self, client: TestClient):
        payload = _observation_payload(behavioral_signal="completed")
        resp = client.post("/api/v1/episodes/observations", json=payload)
        assert resp.status_code == 201
        assert resp.json()["episode"]["behavioral_signal"] == "completed"

    def test_store_observation_invalid_confidence(self, client: TestClient):
        payload = _observation_payload(intent_confidence=1.5)
        resp = client.post("/api/v1/episodes/observations", json=payload)
        assert resp.status_code == 422

    def test_store_observation_emotion_captures_full_snapshot(self, client: TestClient):
        payload = _observation_payload(
            emotion={
                "primary": "joy",
                "intensity": 2.5,
                "valence": 0.9,
                "arousal": 0.8,
                "secondary": "anticipation",
            }
        )
        resp = client.post("/api/v1/episodes/observations", json=payload)
        assert resp.status_code == 201
        emotion = resp.json()["episode"]["emotion"]
        assert emotion["primary"] == "joy"
        assert emotion["intensity"] == 2.5
        assert emotion["valence"] == 0.9
        assert emotion["arousal"] == 0.8
        assert emotion["secondary"] == "anticipation"

    def test_observation_is_retrievable_by_id(self, client: TestClient):
        """Stored observation can be fetched via the standard episode GET."""
        resp = client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(),
        )
        episode_id = resp.json()["episode"]["id"]

        get_resp = client.get(f"/api/v1/episodes/{episode_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == episode_id

    def test_observation_appears_in_user_episodes(self, client: TestClient):
        """Stored observations show up in user episode listing."""
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(raw_text="obs one"),
        )
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(raw_text="obs two"),
        )

        resp = client.get("/api/v1/episodes/user/user-1")
        assert resp.status_code == 200
        assert len(resp.json()["episodes"]) == 2


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


class TestCompressEpisodes:
    def _create_episodes(self, client: TestClient, count: int = 3) -> list[str]:
        """Helper to create episodes and return their IDs."""
        ids = []
        for i in range(count):
            resp = client.post(
                "/api/v1/episodes/observations",
                json=_observation_payload(
                    raw_text=f"Episode for compression {i}",
                    entities=[
                        {"name": "Sarah", "entity_type": "person", "confidence": 0.9}
                    ],
                ),
            )
            ids.append(resp.json()["episode"]["id"])
        return ids

    def test_compress_success(self, client: TestClient):
        ids = self._create_episodes(client, count=3)

        resp = client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ids,
                "summary_text": "Three observations about calling Sarah regarding Q2.",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["episode_count"] == 3
        assert data["summary_text"] == "Three observations about calling Sarah regarding Q2."
        assert len(data["source_episode_ids"]) == 3
        assert "id" in data
        assert data["entity_mentions"]["Sarah"] == 3

    def test_compress_marks_episodes_as_compressed(self, client: TestClient):
        ids = self._create_episodes(client, count=2)

        client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ids,
                "summary_text": "Summary of two episodes.",
            },
        )

        # Compressed episodes should be marked
        for eid in ids:
            resp = client.get(f"/api/v1/episodes/{eid}")
            assert resp.json()["is_compressed"] is True

    def test_compress_compressed_episodes_excluded_by_default(self, client: TestClient):
        ids = self._create_episodes(client, count=2)

        client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ids,
                "summary_text": "Two compressed.",
            },
        )

        # Default query excludes compressed
        resp = client.get("/api/v1/episodes/user/user-1")
        assert len(resp.json()["episodes"]) == 0

        # include_compressed=true shows them
        resp = client.get("/api/v1/episodes/user/user-1?include_compressed=true")
        assert len(resp.json()["episodes"]) == 2

    def test_compress_nonexistent_episode_404(self, client: TestClient):
        resp = client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ["nonexistent-id"],
                "summary_text": "Should fail.",
            },
        )
        assert resp.status_code == 404

    def test_compress_wrong_user_403(self, client: TestClient):
        # Create episode as user-1
        resp = client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(user_id="user-1"),
        )
        eid = resp.json()["episode"]["id"]

        # Try to compress as user-2
        resp = client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-2",
                "episode_ids": [eid],
                "summary_text": "Wrong user.",
            },
        )
        assert resp.status_code == 403

    def test_compress_empty_list_400(self, client: TestClient):
        resp = client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": [],
                "summary_text": "No episodes.",
            },
        )
        assert resp.status_code == 400

    def test_compress_with_embedding(self, client: TestClient):
        ids = self._create_episodes(client, count=2)

        resp = client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ids,
                "summary_text": "Summary with embedding.",
                "embedding": [0.1, 0.2, 0.3],
            },
        )
        assert resp.status_code == 201

    def test_compress_captures_intent_distribution(self, client: TestClient):
        """Compression aggregates intent distribution from source episodes."""
        # Create episodes with different intents
        ids = []
        for intent in ["task", "task", "idea"]:
            resp = client.post(
                "/api/v1/episodes/observations",
                json=_observation_payload(intent=intent, raw_text=f"{intent} obs"),
            )
            ids.append(resp.json()["episode"]["id"])

        resp = client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ids,
                "summary_text": "Mixed intent summary.",
            },
        )
        data = resp.json()
        assert data["intent_distribution"]["task"] == 2
        assert data["intent_distribution"]["idea"] == 1


# ---------------------------------------------------------------------------
# Summary retrieval
# ---------------------------------------------------------------------------


class TestSummaryRetrieval:
    def _create_and_compress(
        self, client: TestClient, count: int = 2, summary_text: str = "A summary."
    ) -> str:
        """Create episodes, compress them, return summary ID."""
        ids = []
        for i in range(count):
            resp = client.post(
                "/api/v1/episodes/observations",
                json=_observation_payload(raw_text=f"ep {i}"),
            )
            ids.append(resp.json()["episode"]["id"])

        resp = client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ids,
                "summary_text": summary_text,
            },
        )
        return resp.json()["id"]

    def test_list_summaries_empty(self, client: TestClient):
        resp = client.get("/api/v1/episodes/summaries/user-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summaries"] == []
        assert data["count"] == 0

    def test_list_summaries_after_compression(self, client: TestClient):
        self._create_and_compress(client, summary_text="Morning summary")
        self._create_and_compress(client, summary_text="Afternoon summary")

        resp = client.get("/api/v1/episodes/summaries/user-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        texts = {s["summary_text"] for s in data["summaries"]}
        assert "Morning summary" in texts
        assert "Afternoon summary" in texts

    def test_list_summaries_user_isolation(self, client: TestClient):
        """Summaries for user-1 should not appear for user-2."""
        self._create_and_compress(client)

        resp = client.get("/api/v1/episodes/summaries/user-2")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_summary_contains_dominant_emotions(self, client: TestClient):
        self._create_and_compress(client)

        resp = client.get("/api/v1/episodes/summaries/user-1")
        data = resp.json()
        assert len(data["summaries"]) > 0
        summary = data["summaries"][0]
        # Dominant emotions come from the source episodes
        assert isinstance(summary["dominant_emotions"], list)

    def test_summary_contains_entity_mentions(self, client: TestClient):
        self._create_and_compress(client)

        resp = client.get("/api/v1/episodes/summaries/user-1")
        summary = resp.json()["summaries"][0]
        assert isinstance(summary["entity_mentions"], dict)


# ---------------------------------------------------------------------------
# Full recall — raw + compressed
# ---------------------------------------------------------------------------


class TestRecall:
    def test_recall_empty(self, client: TestClient):
        resp = client.get("/api/v1/episodes/recall/user-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entries"] == []
        assert data["raw_count"] == 0
        assert data["summary_count"] == 0
        assert data["total_count"] == 0

    def test_recall_with_only_raw_episodes(self, client: TestClient):
        for i in range(3):
            client.post(
                "/api/v1/episodes/observations",
                json=_observation_payload(raw_text=f"raw {i}"),
            )

        resp = client.get("/api/v1/episodes/recall/user-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["raw_count"] == 3
        assert data["summary_count"] == 0
        assert data["total_count"] == 3
        assert all(e["entry_type"] == "episode" for e in data["entries"])

    def test_recall_with_raw_and_compressed(self, client: TestClient):
        """After compression, recall returns both raw episodes and summaries."""
        # Create 4 episodes
        ids = []
        for i in range(4):
            resp = client.post(
                "/api/v1/episodes/observations",
                json=_observation_payload(raw_text=f"ep {i}"),
            )
            ids.append(resp.json()["episode"]["id"])

        # Compress first 2
        client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ids[:2],
                "summary_text": "Summary of first two.",
            },
        )

        resp = client.get("/api/v1/episodes/recall/user-1")
        data = resp.json()

        # 2 raw (uncompressed) + 2 compressed (include_compressed=True default) + 1 summary
        # Actually: include_compressed defaults to True in recall, so all 4 episodes + 1 summary
        assert data["summary_count"] == 1
        assert data["total_count"] == data["raw_count"] + data["summary_count"]

        # Check both types present
        entry_types = {e["entry_type"] for e in data["entries"]}
        assert "episode" in entry_types
        assert "summary" in entry_types

    def test_recall_entries_have_timestamps(self, client: TestClient):
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(),
        )

        resp = client.get("/api/v1/episodes/recall/user-1")
        for entry in resp.json()["entries"]:
            assert "timestamp" in entry
            assert entry["timestamp"] != ""

    def test_recall_entries_sorted_newest_first(self, client: TestClient):
        """Recall entries should be sorted by timestamp descending."""
        for i in range(3):
            client.post(
                "/api/v1/episodes/observations",
                json=_observation_payload(raw_text=f"obs {i}"),
            )

        resp = client.get("/api/v1/episodes/recall/user-1")
        entries = resp.json()["entries"]
        timestamps = [e["timestamp"] for e in entries]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_recall_filter_by_intent(self, client: TestClient):
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(intent="task", raw_text="a task"),
        )
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(intent="idea", raw_text="an idea"),
        )

        resp = client.get("/api/v1/episodes/recall/user-1?intent=idea")
        data = resp.json()
        episodes = [e for e in data["entries"] if e["entry_type"] == "episode"]
        assert len(episodes) == 1
        assert episodes[0]["episode"]["intent"] == "idea"

    def test_recall_filter_by_entity(self, client: TestClient):
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(
                entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
                raw_text="about Alice",
            ),
        )
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(entities=[], raw_text="no entity"),
        )

        resp = client.get("/api/v1/episodes/recall/user-1?entity=Alice")
        data = resp.json()
        episodes = [e for e in data["entries"] if e["entry_type"] == "episode"]
        assert len(episodes) == 1
        assert episodes[0]["episode"]["raw_text"] == "about Alice"

    def test_recall_user_isolation(self, client: TestClient):
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(user_id="user-1"),
        )
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(user_id="user-2"),
        )

        resp = client.get("/api/v1/episodes/recall/user-1")
        data = resp.json()
        assert data["raw_count"] == 1

    def test_recall_episode_has_full_fields(self, client: TestClient):
        """Each episode entry in recall has all required fields."""
        client.post(
            "/api/v1/episodes/observations",
            json=_observation_payload(),
        )

        resp = client.get("/api/v1/episodes/recall/user-1")
        entry = resp.json()["entries"][0]
        assert entry["entry_type"] == "episode"
        ep = entry["episode"]
        assert "id" in ep
        assert "user_id" in ep
        assert "timestamp" in ep
        assert "raw_text" in ep
        assert "intent" in ep
        assert "emotion" in ep
        assert "entities" in ep
        assert "is_compressed" in ep

    def test_recall_summary_has_full_fields(self, client: TestClient):
        """Each summary entry in recall has all required fields."""
        # Create and compress
        ids = []
        for i in range(2):
            resp = client.post(
                "/api/v1/episodes/observations",
                json=_observation_payload(raw_text=f"ep {i}"),
            )
            ids.append(resp.json()["episode"]["id"])

        client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ids,
                "summary_text": "A test summary.",
            },
        )

        resp = client.get("/api/v1/episodes/recall/user-1")
        summaries = [e for e in resp.json()["entries"] if e["entry_type"] == "summary"]
        assert len(summaries) == 1
        s = summaries[0]["summary"]
        assert "id" in s
        assert "user_id" in s
        assert "period_start" in s
        assert "period_end" in s
        assert "episode_count" in s
        assert "summary_text" in s
        assert s["summary_text"] == "A test summary."


# ---------------------------------------------------------------------------
# Pipeline integrity — observation flows through store → retrieve → compress → recall
# ---------------------------------------------------------------------------


class TestPipelineIntegrity:
    def test_full_pipeline_flow(self, client: TestClient):
        """End-to-end: store observations → verify retrieval → compress → recall."""
        # 1. Store observations
        ids = []
        for i in range(5):
            resp = client.post(
                "/api/v1/episodes/observations",
                json=_observation_payload(
                    raw_text=f"Pipeline test {i}",
                    intent="task" if i < 3 else "idea",
                    entities=[
                        {"name": "Sarah", "entity_type": "person", "confidence": 0.95}
                    ],
                ),
            )
            assert resp.status_code == 201
            ids.append(resp.json()["episode"]["id"])

        # 2. Verify all stored and retrievable
        resp = client.get("/api/v1/episodes/user/user-1")
        assert len(resp.json()["episodes"]) == 5

        # 3. Compress first 3 episodes
        compress_resp = client.post(
            "/api/v1/episodes/compress",
            json={
                "user_id": "user-1",
                "episode_ids": ids[:3],
                "summary_text": "Three task observations about Sarah.",
            },
        )
        assert compress_resp.status_code == 201
        summary_data = compress_resp.json()
        assert summary_data["episode_count"] == 3
        assert summary_data["entity_mentions"]["Sarah"] == 3
        assert summary_data["intent_distribution"]["task"] == 3

        # 4. Verify compressed episodes are marked
        for eid in ids[:3]:
            ep_resp = client.get(f"/api/v1/episodes/{eid}")
            assert ep_resp.json()["is_compressed"] is True

        # 5. Uncompressed episodes still visible
        resp = client.get("/api/v1/episodes/user/user-1")
        uncompressed = resp.json()["episodes"]
        assert len(uncompressed) == 2  # only the 2 uncompressed

        # 6. Full recall shows both raw and summaries
        recall_resp = client.get("/api/v1/episodes/recall/user-1")
        recall_data = recall_resp.json()
        assert recall_data["summary_count"] == 1
        assert recall_data["total_count"] > 0

        # 7. Summaries endpoint shows the summary
        summaries_resp = client.get("/api/v1/episodes/summaries/user-1")
        assert summaries_resp.json()["count"] == 1

    def test_no_data_loss_through_pipeline(self, client: TestClient):
        """Every observation stored can be retrieved — no data loss in the pipeline."""
        stored_ids = set()
        for i in range(10):
            resp = client.post(
                "/api/v1/episodes/observations",
                json=_observation_payload(raw_text=f"Data integrity {i}"),
            )
            stored_ids.add(resp.json()["episode"]["id"])

        # Verify each by direct get
        for eid in stored_ids:
            resp = client.get(f"/api/v1/episodes/{eid}")
            assert resp.status_code == 200
            assert resp.json()["id"] == eid

        # Verify total count
        resp = client.get("/api/v1/episodes/user/user-1")
        assert resp.json()["total_count"] == 10
