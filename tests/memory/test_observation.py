"""Tests for the Observation model, factory functions, and ObservationRepository.

Covers:
- Observation creation and immutability
- Factory functions (observe_voice, observe_text)
- Pipeline enrichment (with_classification, with_emotion, with_entities, with_embedding)
- Conversion to Episode (to_episode)
- ObservationRepository append-only semantics
- Batch recording
- Metadata preservation through the full pipeline
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from blurt.memory.episodic import (
    BehavioralSignal,
    EmotionSnapshot,
    EntityRef,
    InMemoryEpisodicStore,
    InputModality,
)
from blurt.memory.observation import (
    Observation,
    ObservationMetadata,
    ObservationRepository,
    observe_text,
    observe_voice,
)


# ---------------------------------------------------------------------------
# Observation creation
# ---------------------------------------------------------------------------


class TestObservationCreation:
    def test_default_observation_has_uuid(self):
        obs = Observation(user_id="u1", raw_text="hello")
        assert obs.id  # non-empty
        assert len(obs.id) == 36  # UUID4 format

    def test_default_observation_has_utc_timestamp(self):
        before = datetime.now(timezone.utc)
        obs = Observation(user_id="u1", raw_text="hello")
        after = datetime.now(timezone.utc)
        assert before <= obs.timestamp <= after

    def test_default_modality_is_voice(self):
        obs = Observation(user_id="u1", raw_text="hello")
        assert obs.modality == InputModality.VOICE

    def test_observation_is_frozen(self):
        obs = Observation(user_id="u1", raw_text="hello")
        with pytest.raises(AttributeError):
            obs.raw_text = "modified"  # type: ignore[misc]

    def test_observation_with_full_metadata(self):
        meta = ObservationMetadata(
            session_id="sess-1",
            time_of_day="afternoon",
            day_of_week="wednesday",
            device_type="mobile",
            locale="en-US",
            audio_duration_ms=3200,
            transcription_confidence=0.95,
        )
        obs = Observation(user_id="u1", raw_text="hello", metadata=meta)
        assert obs.metadata.session_id == "sess-1"
        assert obs.metadata.device_type == "mobile"
        assert obs.metadata.audio_duration_ms == 3200


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_observe_voice_creates_voice_observation(self):
        obs = observe_voice(
            "user-1",
            "Pick up groceries",
            session_id="sess-1",
            audio_duration_ms=2500,
            transcription_confidence=0.92,
            time_of_day="morning",
            day_of_week="tuesday",
        )
        assert obs.user_id == "user-1"
        assert obs.raw_text == "Pick up groceries"
        assert obs.modality == InputModality.VOICE
        assert obs.metadata.session_id == "sess-1"
        assert obs.metadata.audio_duration_ms == 2500
        assert obs.metadata.transcription_confidence == 0.92

    def test_observe_text_creates_text_observation(self):
        obs = observe_text(
            "user-1",
            "Actually, pick up milk too",
            session_id="sess-1",
        )
        assert obs.modality == InputModality.TEXT
        assert obs.raw_text == "Actually, pick up milk too"
        assert obs.metadata.audio_duration_ms is None

    def test_observe_voice_with_preceding_episode(self):
        obs = observe_voice(
            "user-1",
            "And also get bread",
            session_id="sess-1",
            preceding_episode_id="ep-abc",
            active_task_id="task-xyz",
        )
        assert obs.metadata.preceding_episode_id == "ep-abc"
        assert obs.metadata.active_task_id == "task-xyz"


# ---------------------------------------------------------------------------
# Pipeline enrichment (immutable with_ methods)
# ---------------------------------------------------------------------------


class TestPipelineEnrichment:
    def test_with_classification(self):
        obs = observe_voice("u1", "Buy groceries")
        enriched = obs.with_classification("task", 0.95)
        assert enriched.intent == "task"
        assert enriched.intent_confidence == 0.95
        # Original unchanged
        assert obs.intent is None
        # Identity preserved
        assert enriched.id == obs.id
        assert enriched.timestamp == obs.timestamp

    def test_with_emotion(self):
        obs = observe_voice("u1", "I'm so happy")
        emo = EmotionSnapshot(primary="joy", intensity=2.5, valence=0.9, arousal=0.7)
        enriched = obs.with_emotion(emo)
        assert enriched.emotion.primary == "joy"
        assert enriched.emotion.intensity == 2.5
        assert obs.emotion.primary == "trust"  # original default unchanged

    def test_with_entities(self):
        obs = observe_voice("u1", "Meeting with Sarah about Project X")
        entities = [
            EntityRef(name="Sarah", entity_type="person"),
            EntityRef(name="Project X", entity_type="project"),
        ]
        enriched = obs.with_entities(entities)
        assert len(enriched.entities) == 2
        assert enriched.entities[0].name == "Sarah"
        assert len(obs.entities) == 0  # original unchanged

    def test_with_embedding(self):
        obs = observe_voice("u1", "hello world")
        enriched = obs.with_embedding([0.1, 0.2, 0.3])
        assert enriched.embedding == (0.1, 0.2, 0.3)
        assert obs.embedding is None  # original unchanged

    def test_chained_enrichment(self):
        """Pipeline enrichment can be chained immutably."""
        obs = observe_voice("u1", "Lunch with Sarah tomorrow", session_id="sess-1")
        enriched = (
            obs.with_classification("event", 0.88)
            .with_emotion(EmotionSnapshot(primary="anticipation", intensity=1.5, valence=0.4, arousal=0.5))
            .with_entities([EntityRef(name="Sarah", entity_type="person")])
            .with_embedding([0.1, 0.2, 0.3, 0.4])
        )
        assert enriched.intent == "event"
        assert enriched.emotion.primary == "anticipation"
        assert len(enriched.entities) == 1
        assert enriched.embedding is not None
        assert enriched.id == obs.id  # same observation throughout


# ---------------------------------------------------------------------------
# Conversion to Episode
# ---------------------------------------------------------------------------


class TestToEpisode:
    def test_basic_conversion(self):
        obs = observe_voice("u1", "Buy milk", session_id="sess-1")
        enriched = obs.with_classification("task", 0.95)
        episode = enriched.to_episode()

        assert episode.id == obs.id
        assert episode.user_id == "u1"
        assert episode.raw_text == "Buy milk"
        assert episode.modality == InputModality.VOICE
        assert episode.intent == "task"
        assert episode.intent_confidence == 0.95
        assert episode.context.session_id == "sess-1"
        assert episode.is_compressed is False

    def test_conversion_preserves_timestamp(self):
        obs = observe_voice("u1", "hello")
        episode = obs.to_episode()
        assert episode.timestamp == obs.timestamp

    def test_conversion_preserves_entities(self):
        obs = observe_voice("u1", "Call Sarah").with_entities(
            [EntityRef(name="Sarah", entity_type="person", entity_id="ent-1")]
        )
        episode = obs.to_episode()
        assert len(episode.entities) == 1
        assert episode.entities[0].name == "Sarah"
        assert episode.entities[0].entity_id == "ent-1"

    def test_conversion_preserves_emotion(self):
        emo = EmotionSnapshot(primary="joy", intensity=2.0, valence=0.8, arousal=0.6)
        obs = observe_voice("u1", "great news").with_emotion(emo)
        episode = obs.to_episode()
        assert episode.emotion.primary == "joy"
        assert episode.emotion.intensity == 2.0

    def test_conversion_preserves_embedding(self):
        obs = observe_voice("u1", "hello").with_embedding([0.1, 0.2])
        episode = obs.to_episode()
        assert episode.embedding == [0.1, 0.2]

    def test_conversion_without_classification_defaults_to_task(self):
        obs = observe_voice("u1", "something")
        episode = obs.to_episode()
        assert episode.intent == "task"  # default when not classified

    def test_conversion_preserves_behavioral_signal(self):
        obs = Observation(
            user_id="u1",
            raw_text="done",
            behavioral_signal=BehavioralSignal.COMPLETED,
            surfaced_task_id="task-1",
        )
        episode = obs.to_episode()
        assert episode.behavioral_signal == BehavioralSignal.COMPLETED
        assert episode.surfaced_task_id == "task-1"

    def test_metadata_to_episode_context(self):
        meta = ObservationMetadata(
            session_id="sess-1",
            time_of_day="evening",
            day_of_week="friday",
            preceding_episode_id="ep-prev",
            active_task_id="task-active",
            device_type="mobile",
            audio_duration_ms=5000,
        )
        ctx = meta.to_episode_context()
        assert ctx.session_id == "sess-1"
        assert ctx.time_of_day == "evening"
        assert ctx.day_of_week == "friday"
        assert ctx.preceding_episode_id == "ep-prev"
        assert ctx.active_task_id == "task-active"


# ---------------------------------------------------------------------------
# ObservationRepository
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryEpisodicStore:
    return InMemoryEpisodicStore()


@pytest.fixture
def repo(store: InMemoryEpisodicStore) -> ObservationRepository:
    return ObservationRepository(store)


class TestObservationRepository:
    @pytest.mark.asyncio
    async def test_record_observation(self, repo: ObservationRepository):
        obs = (
            observe_voice("u1", "Buy groceries", session_id="sess-1")
            .with_classification("task", 0.95)
            .with_emotion(EmotionSnapshot(primary="anticipation", intensity=1.0, valence=0.3, arousal=0.4))
        )
        episode = await repo.record(obs)
        assert episode.id == obs.id
        assert episode.raw_text == "Buy groceries"
        assert episode.intent == "task"

    @pytest.mark.asyncio
    async def test_record_increments_count(self, repo: ObservationRepository):
        assert repo.observation_count == 0
        await repo.record(observe_voice("u1", "one"))
        assert repo.observation_count == 1
        await repo.record(observe_voice("u1", "two"))
        assert repo.observation_count == 2

    @pytest.mark.asyncio
    async def test_record_is_append_only(self, repo: ObservationRepository):
        obs = observe_voice("u1", "hello")
        await repo.record(obs)
        # Same observation ID cannot be recorded twice
        with pytest.raises(ValueError, match="already exists"):
            await repo.record(obs)

    @pytest.mark.asyncio
    async def test_record_many(self, repo: ObservationRepository):
        observations = [
            observe_voice("u1", f"observation {i}", session_id="sess-1")
            for i in range(5)
        ]
        episodes = await repo.record_many(observations)
        assert len(episodes) == 5
        assert repo.observation_count == 5
        for i, ep in enumerate(episodes):
            assert ep.raw_text == f"observation {i}"

    @pytest.mark.asyncio
    async def test_get_episode_after_record(self, repo: ObservationRepository):
        obs = observe_voice("u1", "hello")
        episode = await repo.record(obs)
        retrieved = await repo.get_episode(episode.id)
        assert retrieved is not None
        assert retrieved.id == episode.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_episode(self, repo: ObservationRepository):
        assert await repo.get_episode("nonexistent") is None

    @pytest.mark.asyncio
    async def test_count_user_episodes(self, repo: ObservationRepository):
        await repo.record(observe_voice("u1", "one"))
        await repo.record(observe_voice("u1", "two"))
        await repo.record(observe_voice("u2", "three"))
        assert await repo.count_user_episodes("u1") == 2
        assert await repo.count_user_episodes("u2") == 1

    @pytest.mark.asyncio
    async def test_store_property(self, repo: ObservationRepository, store: InMemoryEpisodicStore):
        assert repo.store is store


# ---------------------------------------------------------------------------
# End-to-end: voice blurt -> observation -> enrichment -> episode -> store
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, repo: ObservationRepository):
        """Simulate a complete blurt pipeline: capture -> classify -> detect -> extract -> store."""
        # 1. Capture observation from voice
        obs = observe_voice(
            "u1",
            "Meeting with Sarah about Q2 planning tomorrow at 3pm",
            session_id="sess-1",
            audio_duration_ms=4200,
            transcription_confidence=0.94,
            time_of_day="afternoon",
            day_of_week="wednesday",
        )

        # 2. Classify intent
        obs = obs.with_classification("event", 0.91)

        # 3. Detect emotion
        obs = obs.with_emotion(
            EmotionSnapshot(primary="anticipation", intensity=1.2, valence=0.4, arousal=0.5)
        )

        # 4. Extract entities
        obs = obs.with_entities([
            EntityRef(name="Sarah", entity_type="person"),
            EntityRef(name="Q2 planning", entity_type="project"),
        ])

        # 5. Generate embedding
        obs = obs.with_embedding([0.1, 0.2, 0.3, 0.4, 0.5])

        # 6. Store
        episode = await repo.record(obs)

        # Verify everything persisted
        assert episode.raw_text == "Meeting with Sarah about Q2 planning tomorrow at 3pm"
        assert episode.intent == "event"
        assert episode.intent_confidence == 0.91
        assert episode.emotion.primary == "anticipation"
        assert len(episode.entities) == 2
        assert episode.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert episode.context.session_id == "sess-1"
        assert episode.context.time_of_day == "afternoon"
        assert episode.is_compressed is False

        # Verify retrieval
        retrieved = await repo.get_episode(episode.id)
        assert retrieved is not None
        assert retrieved.raw_text == episode.raw_text

    @pytest.mark.asyncio
    async def test_text_correction_pipeline(self, repo: ObservationRepository):
        """Text input for corrections follows the same pipeline."""
        obs = observe_text("u1", "Actually make it 4pm not 3pm", session_id="sess-1")
        obs = obs.with_classification("update", 0.87)
        episode = await repo.record(obs)
        assert episode.modality == InputModality.TEXT
        assert episode.intent == "update"
