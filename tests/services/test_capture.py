"""Tests for the BlurtCapturePipeline — zero-drop capture of everything said.

Covers:
- Every utterance is captured (casual remarks, empty strings, filler words)
- Pipeline enrichment stages (classify, extract, detect, embed)
- Graceful degradation when enrichment stages fail
- Safe fallback to journal on classification failure
- Stats tracking (including casual capture count)
- Voice and text modality support
- Pipeline integrity: no data loss through the capture funnel
"""

from __future__ import annotations


import pytest

from blurt.memory.episodic import (
    EmotionSnapshot,
    EntityRef,
    InMemoryEpisodicStore,
    InputModality,
)
from blurt.services.capture import (
    BlurtCapturePipeline,
    CaptureStage,
    _default_classify,
    _default_detect_emotion,
    _default_embed,
    _default_extract_entities,
)


# ---------------------------------------------------------------------------
# Test helpers — mock enrichment functions
# ---------------------------------------------------------------------------


async def mock_classify(text: str) -> tuple[str, float]:
    """Smart mock classifier that handles casual remarks."""
    text_lower = text.lower().strip()

    # Casual/throwaway remarks → journal
    casual_patterns = [
        "huh", "hmm", "oh", "nice", "cool", "yeah", "okay", "ok",
        "interesting", "whatever", "sure", "right", "wow",
    ]
    if not text_lower or any(text_lower.startswith(p) for p in casual_patterns):
        return ("journal", 0.90)

    # Task-like
    if any(w in text_lower for w in ["need to", "buy", "call", "finish", "submit"]):
        return ("task", 0.92)

    # Event-like
    if any(w in text_lower for w in ["meeting", "dinner", "at 3pm", "tomorrow at"]):
        return ("event", 0.89)

    # Question-like
    if text_lower.startswith(("what", "when", "how", "why", "where", "who")):
        return ("question", 0.88)

    # Default to journal for anything else
    return ("journal", 0.85)


async def mock_extract_entities(text: str) -> list[EntityRef]:
    """Simple entity extraction — finds capitalized words as names."""
    entities = []
    for word in text.split():
        if word[0].isupper() and len(word) > 1 and word.isalpha():
            entities.append(EntityRef(name=word, entity_type="person", confidence=0.8))
    return entities


async def mock_detect_emotion(text: str) -> EmotionSnapshot:
    """Simple emotion detection based on keywords."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["happy", "great", "awesome", "love"]):
        return EmotionSnapshot(primary="joy", intensity=2.0, valence=0.8, arousal=0.6)
    if any(w in text_lower for w in ["sad", "tough", "hard", "awful"]):
        return EmotionSnapshot(primary="sadness", intensity=1.5, valence=-0.6, arousal=0.3)
    return EmotionSnapshot(primary="trust", intensity=0.5, valence=0.1, arousal=0.2)


async def mock_embed(text: str) -> list[float]:
    """Simple embedding — hash-based fake vector."""
    if not text:
        return []
    h = hash(text) % 1000
    return [h / 1000, (h + 1) / 1000, (h + 2) / 1000, (h + 3) / 1000]


async def failing_classify(text: str) -> tuple[str, float]:
    raise RuntimeError("Classifier service unavailable")


async def failing_extract(text: str) -> list[EntityRef]:
    raise RuntimeError("Entity extractor unavailable")


async def failing_detect(text: str) -> EmotionSnapshot:
    raise RuntimeError("Emotion detector unavailable")


async def failing_embed(text: str) -> list[float]:
    raise RuntimeError("Embedder unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryEpisodicStore:
    return InMemoryEpisodicStore()


@pytest.fixture
def pipeline(store: InMemoryEpisodicStore) -> BlurtCapturePipeline:
    return BlurtCapturePipeline(
        store,
        classifier=mock_classify,
        entity_extractor=mock_extract_entities,
        emotion_detector=mock_detect_emotion,
        embedder=mock_embed,
    )


@pytest.fixture
def default_pipeline(store: InMemoryEpisodicStore) -> BlurtCapturePipeline:
    """Pipeline with default (no-op) enrichment functions."""
    return BlurtCapturePipeline(store)


# ---------------------------------------------------------------------------
# Core: Everything gets captured
# ---------------------------------------------------------------------------


class TestEverythingCaptured:
    """Verify that ALL input is captured — nothing is dropped or filtered."""

    @pytest.mark.asyncio
    async def test_casual_remark_captured(self, pipeline: BlurtCapturePipeline):
        """Casual throwaway remarks are stored, not dropped."""
        result = await pipeline.capture_voice("u1", "huh, interesting")
        assert result.was_stored
        assert result.episode.raw_text == "huh, interesting"

    @pytest.mark.asyncio
    async def test_filler_words_captured(self, pipeline: BlurtCapturePipeline):
        """Filler words like 'hmm', 'yeah', 'ok' are captured."""
        for text in ["hmm", "yeah", "ok", "sure", "right"]:
            result = await pipeline.capture_voice("u1", text)
            assert result.was_stored, f"'{text}' was not stored!"
            assert result.episode.raw_text == text

    @pytest.mark.asyncio
    async def test_empty_string_captured(self, pipeline: BlurtCapturePipeline):
        """Even empty strings are captured (user spoke but nothing transcribed)."""
        result = await pipeline.capture_voice("u1", "")
        assert result.was_stored
        assert result.episode.raw_text == ""

    @pytest.mark.asyncio
    async def test_weather_comment_captured(self, pipeline: BlurtCapturePipeline):
        """Off-hand observations about weather are stored."""
        result = await pipeline.capture_voice("u1", "nice weather today")
        assert result.was_stored
        assert result.episode.raw_text == "nice weather today"

    @pytest.mark.asyncio
    async def test_half_thought_captured(self, pipeline: BlurtCapturePipeline):
        """Incomplete thoughts are captured."""
        result = await pipeline.capture_voice("u1", "I was thinking maybe we could...")
        assert result.was_stored
        assert "thinking maybe" in result.episode.raw_text

    @pytest.mark.asyncio
    async def test_reaction_captured(self, pipeline: BlurtCapturePipeline):
        """Quick reactions are captured."""
        result = await pipeline.capture_voice("u1", "oh wow, that's cool")
        assert result.was_stored

    @pytest.mark.asyncio
    async def test_long_rambling_captured(self, pipeline: BlurtCapturePipeline):
        """Long rambling input is captured in full."""
        long_text = "So I was walking down the street and I saw this dog and " * 10
        result = await pipeline.capture_voice("u1", long_text)
        assert result.was_stored
        assert result.episode.raw_text == long_text

    @pytest.mark.asyncio
    async def test_actionable_input_captured(self, pipeline: BlurtCapturePipeline):
        """Normal actionable input (tasks, events) is also captured."""
        result = await pipeline.capture_voice("u1", "I need to buy groceries")
        assert result.was_stored
        assert result.episode.intent == "task"

    @pytest.mark.asyncio
    async def test_multiple_casual_remarks_all_captured(self, pipeline: BlurtCapturePipeline):
        """Every casual remark in a sequence is captured."""
        remarks = [
            "huh, interesting",
            "nice weather",
            "oh well",
            "whatever",
            "cool",
            "that's funny",
        ]
        for remark in remarks:
            result = await pipeline.capture_voice("u1", remark)
            assert result.was_stored, f"'{remark}' was not stored!"

        # Verify all are retrievable
        count = await pipeline.repo.count_user_episodes("u1")
        assert count == len(remarks)


# ---------------------------------------------------------------------------
# Pipeline enrichment
# ---------------------------------------------------------------------------


class TestPipelineEnrichment:
    @pytest.mark.asyncio
    async def test_classification_applied(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "I need to call the dentist")
        assert result.classification_applied
        assert result.episode.intent == "task"
        assert result.episode.intent_confidence > 0.5

    @pytest.mark.asyncio
    async def test_casual_classified_as_journal(self, pipeline: BlurtCapturePipeline):
        """Casual remarks are classified as journal entries."""
        result = await pipeline.capture_voice("u1", "hmm, that's interesting")
        assert result.episode.intent == "journal"

    @pytest.mark.asyncio
    async def test_entities_extracted(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "Meeting with Sarah about the project")
        assert result.entities_extracted >= 1  # "Sarah" should be found

    @pytest.mark.asyncio
    async def test_emotion_detected(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "I'm so happy today!")
        assert result.emotion_detected
        assert result.episode.emotion.primary == "joy"

    @pytest.mark.asyncio
    async def test_embedding_generated(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "Some text to embed")
        assert result.embedding_generated
        assert result.episode.embedding is not None

    @pytest.mark.asyncio
    async def test_fully_enriched(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "I need to buy groceries")
        assert result.fully_enriched

    @pytest.mark.asyncio
    async def test_stages_tracked(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "test input")
        assert CaptureStage.RECEIVED in result.stages_completed
        assert CaptureStage.OBSERVED in result.stages_completed
        assert CaptureStage.CLASSIFIED in result.stages_completed
        assert CaptureStage.STORED in result.stages_completed


# ---------------------------------------------------------------------------
# Graceful degradation — enrichment failures don't prevent storage
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_classification_failure_still_stores(self, store: InMemoryEpisodicStore):
        """When the classifier fails, the observation is still stored as journal."""
        pipeline = BlurtCapturePipeline(
            store,
            classifier=failing_classify,
            entity_extractor=mock_extract_entities,
            emotion_detector=mock_detect_emotion,
            embedder=mock_embed,
        )
        result = await pipeline.capture_voice("u1", "I need to do something")
        assert result.was_stored
        assert result.episode.intent == "journal"  # Safe fallback
        assert len(result.warnings) > 0
        assert "Classification failed" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_entity_extraction_failure_still_stores(self, store: InMemoryEpisodicStore):
        pipeline = BlurtCapturePipeline(
            store,
            classifier=mock_classify,
            entity_extractor=failing_extract,
            emotion_detector=mock_detect_emotion,
            embedder=mock_embed,
        )
        result = await pipeline.capture_voice("u1", "Meeting with Sarah")
        assert result.was_stored
        assert result.entities_extracted == 0
        assert "Entity extraction failed" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_emotion_detection_failure_still_stores(self, store: InMemoryEpisodicStore):
        pipeline = BlurtCapturePipeline(
            store,
            classifier=mock_classify,
            entity_extractor=mock_extract_entities,
            emotion_detector=failing_detect,
            embedder=mock_embed,
        )
        result = await pipeline.capture_voice("u1", "I'm feeling great")
        assert result.was_stored
        assert not result.emotion_detected
        assert "Emotion detection failed" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_embedding_failure_still_stores(self, store: InMemoryEpisodicStore):
        pipeline = BlurtCapturePipeline(
            store,
            classifier=mock_classify,
            entity_extractor=mock_extract_entities,
            emotion_detector=mock_detect_emotion,
            embedder=failing_embed,
        )
        result = await pipeline.capture_voice("u1", "some text")
        assert result.was_stored
        assert not result.embedding_generated
        assert "Embedding generation failed" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_all_enrichment_fails_still_stores(self, store: InMemoryEpisodicStore):
        """Even if EVERY enrichment stage fails, the observation is stored."""
        pipeline = BlurtCapturePipeline(
            store,
            classifier=failing_classify,
            entity_extractor=failing_extract,
            emotion_detector=failing_detect,
            embedder=failing_embed,
        )
        result = await pipeline.capture_voice("u1", "total failure scenario")
        assert result.was_stored
        assert result.episode.raw_text == "total failure scenario"
        assert len(result.warnings) == 4  # All 4 stages warned
        assert not result.fully_enriched


# ---------------------------------------------------------------------------
# Default pipeline (no-op enrichment)
# ---------------------------------------------------------------------------


class TestDefaultPipeline:
    @pytest.mark.asyncio
    async def test_default_classify_returns_journal(self):
        intent, conf = await _default_classify("anything")
        assert intent == "journal"
        assert conf == 1.0

    @pytest.mark.asyncio
    async def test_default_extract_returns_empty(self):
        entities = await _default_extract_entities("text")
        assert entities == []

    @pytest.mark.asyncio
    async def test_default_detect_returns_neutral(self):
        emotion = await _default_detect_emotion("text")
        assert emotion.primary == "trust"

    @pytest.mark.asyncio
    async def test_default_embed_returns_empty(self):
        embedding = await _default_embed("text")
        assert embedding == []

    @pytest.mark.asyncio
    async def test_default_pipeline_captures_casual(self, default_pipeline: BlurtCapturePipeline):
        """Default pipeline captures casual remarks with journal intent."""
        result = await default_pipeline.capture_voice("u1", "oh, whatever")
        assert result.was_stored
        assert result.episode.intent == "journal"
        assert result.episode.raw_text == "oh, whatever"


# ---------------------------------------------------------------------------
# Voice and text modality
# ---------------------------------------------------------------------------


class TestModalities:
    @pytest.mark.asyncio
    async def test_voice_modality(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "hello")
        assert result.episode.modality == InputModality.VOICE

    @pytest.mark.asyncio
    async def test_text_modality(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_text("u1", "correction here")
        assert result.episode.modality == InputModality.TEXT

    @pytest.mark.asyncio
    async def test_voice_with_audio_metadata(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice(
            "u1",
            "casual remark",
            audio_duration_ms=1200,
            transcription_confidence=0.88,
        )
        assert result.was_stored

    @pytest.mark.asyncio
    async def test_text_captures_casual_too(self, pipeline: BlurtCapturePipeline):
        """Text modality also captures casual input."""
        result = await pipeline.capture_text("u1", "hmm ok")
        assert result.was_stored
        assert result.episode.raw_text == "hmm ok"


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------


class TestStats:
    @pytest.mark.asyncio
    async def test_total_captured_increments(self, pipeline: BlurtCapturePipeline):
        assert pipeline.stats.total_captured == 0
        await pipeline.capture_voice("u1", "one")
        assert pipeline.stats.total_captured == 1
        await pipeline.capture_voice("u1", "two")
        assert pipeline.stats.total_captured == 2

    @pytest.mark.asyncio
    async def test_voice_text_counts(self, pipeline: BlurtCapturePipeline):
        await pipeline.capture_voice("u1", "voice one")
        await pipeline.capture_voice("u1", "voice two")
        await pipeline.capture_text("u1", "text one")
        assert pipeline.stats.voice_count == 2
        assert pipeline.stats.text_count == 1

    @pytest.mark.asyncio
    async def test_casual_capture_counted(self, pipeline: BlurtCapturePipeline):
        """Casual remarks are tracked in stats."""
        await pipeline.capture_voice("u1", "hmm, interesting")
        await pipeline.capture_voice("u1", "cool")
        await pipeline.capture_voice("u1", "I need to buy groceries")  # Not casual
        assert pipeline.stats.casual_capture_count >= 2  # At least the 2 casual ones

    @pytest.mark.asyncio
    async def test_drop_rate_always_zero(self, pipeline: BlurtCapturePipeline):
        """Drop rate is ALWAYS zero — by design."""
        for i in range(10):
            await pipeline.capture_voice("u1", f"blurt {i}")
        assert pipeline.stats.drop_rate == 0.0

    @pytest.mark.asyncio
    async def test_intent_distribution_tracked(self, pipeline: BlurtCapturePipeline):
        await pipeline.capture_voice("u1", "I need to buy groceries")  # task
        await pipeline.capture_voice("u1", "hmm, interesting")  # journal
        await pipeline.capture_voice("u1", "what time is it?")  # question
        dist = pipeline.stats.intent_distribution
        assert "task" in dist
        assert "journal" in dist
        assert "question" in dist

    @pytest.mark.asyncio
    async def test_stats_to_dict(self, pipeline: BlurtCapturePipeline):
        await pipeline.capture_voice("u1", "test")
        d = pipeline.stats.to_dict()
        assert "total_captured" in d
        assert "drop_rate" in d
        assert d["drop_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_classification_failure_stat(self, store: InMemoryEpisodicStore):
        pipeline = BlurtCapturePipeline(
            store, classifier=failing_classify
        )
        await pipeline.capture_voice("u1", "test")
        assert pipeline.stats.classification_failures == 1


# ---------------------------------------------------------------------------
# CaptureResult model
# ---------------------------------------------------------------------------


class TestCaptureResult:
    @pytest.mark.asyncio
    async def test_to_dict(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "test input")
        d = result.to_dict()
        assert "episode_id" in d
        assert "stages_completed" in d
        assert "fully_enriched" in d
        assert "latency_ms" in d
        assert "warnings" in d

    @pytest.mark.asyncio
    async def test_observation_id_matches_episode(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "test")
        assert result.observation_id == result.episode.id


# ---------------------------------------------------------------------------
# Pipeline integrity — capture_raw
# ---------------------------------------------------------------------------


class TestCaptureRaw:
    @pytest.mark.asyncio
    async def test_capture_raw_observation(self, pipeline: BlurtCapturePipeline):
        from blurt.memory.observation import observe_voice as ov

        obs = ov("u1", "pre-built observation", session_id="sess-1")
        result = await pipeline.capture_raw(obs)
        assert result.was_stored
        assert result.episode.raw_text == "pre-built observation"


# ---------------------------------------------------------------------------
# Session context preserved
# ---------------------------------------------------------------------------


class TestContextPreserved:
    @pytest.mark.asyncio
    async def test_session_id_preserved(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice(
            "u1", "casual remark", session_id="sess-42"
        )
        assert result.episode.context.session_id == "sess-42"

    @pytest.mark.asyncio
    async def test_time_context_preserved(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice(
            "u1",
            "whatever",
            time_of_day="evening",
            day_of_week="friday",
        )
        assert result.episode.context.time_of_day == "evening"
        assert result.episode.context.day_of_week == "friday"

    @pytest.mark.asyncio
    async def test_preceding_episode_linked(self, pipeline: BlurtCapturePipeline):
        r1 = await pipeline.capture_voice("u1", "first thing")
        r2 = await pipeline.capture_voice(
            "u1", "oh and also", preceding_episode_id=r1.episode.id
        )
        assert r2.episode.context.preceding_episode_id == r1.episode.id


# ---------------------------------------------------------------------------
# Zero-drop guarantee: mixed input types
# ---------------------------------------------------------------------------


class TestZeroDropMixed:
    """Test that a realistic mixed stream of inputs all gets captured."""

    @pytest.mark.asyncio
    async def test_realistic_conversation_stream(self, pipeline: BlurtCapturePipeline):
        """Simulate a real conversation with mixed casual and actionable inputs."""
        inputs = [
            ("voice", "hmm"),
            ("voice", "oh right, I need to call Sarah"),
            ("voice", "nice weather today"),
            ("voice", "meeting with the team at 3pm"),
            ("voice", "whatever"),
            ("text", "actually make it 4pm"),
            ("voice", "I was thinking about that project"),
            ("voice", "huh"),
            ("voice", "oh well"),
            ("voice", "remind me to buy milk"),
        ]

        results = []
        for modality, text in inputs:
            if modality == "voice":
                r = await pipeline.capture_voice("u1", text, session_id="sess-1")
            else:
                r = await pipeline.capture_text("u1", text, session_id="sess-1")
            results.append(r)

        # Every single input was captured
        assert len(results) == len(inputs)
        for i, result in enumerate(results):
            assert result.was_stored, f"Input {i} ('{inputs[i][1]}') was not stored!"
            assert result.episode.raw_text == inputs[i][1]

        # Total count matches
        count = await pipeline.repo.count_user_episodes("u1")
        assert count == len(inputs)

        # Stats reflect everything
        assert pipeline.stats.total_captured == len(inputs)
        assert pipeline.stats.drop_rate == 0.0
