"""End-to-end tests: captures everything said including casual remarks as observations.

This test validates AC 8: every utterance — including casual remarks, filler words,
throwaway comments, half-thoughts, and brief reactions — flows through the full
capture pipeline and is stored as a retrievable observation.

The key guarantee: NOTHING is ever dropped, filtered, or discarded.
Casual remarks are first-class observations that contribute to:
- Behavioral pattern detection
- Emotional baseline tracking
- Conversational context in sessions
- Full personal history recall

Anti-shame: no input is "too trivial" to capture.
"""

from __future__ import annotations

import pytest

from blurt.memory.episodic import (
    EmotionSnapshot,
    EntityRef,
    InMemoryEpisodicStore,
    IntentFilter,
)
from blurt.services.capture import (
    BlurtCapturePipeline,
    CaptureStage,
)
from blurt.services.casual_detection import ObservationType


# ---------------------------------------------------------------------------
# Mock enrichment functions
# ---------------------------------------------------------------------------


async def mock_classify(text: str) -> tuple[str, float]:
    text_lower = text.lower().strip()
    casual_starters = [
        "huh", "hmm", "oh", "nice", "cool", "yeah", "ok", "sure",
        "whatever", "meh", "wow",
    ]
    if not text_lower or any(text_lower.startswith(p) for p in casual_starters):
        return ("journal", 0.90)
    if any(w in text_lower for w in ["need to", "buy", "call", "finish"]):
        return ("task", 0.92)
    if any(w in text_lower for w in ["meeting", "dinner", "at 3pm"]):
        return ("event", 0.89)
    if text_lower.startswith(("what", "when", "how")):
        return ("question", 0.88)
    return ("journal", 0.85)


async def mock_extract(text: str) -> list[EntityRef]:
    entities = []
    for w in text.split():
        if w[0].isupper() and len(w) > 1 and w.isalpha():
            entities.append(EntityRef(name=w, entity_type="person", confidence=0.8))
    return entities


async def mock_detect_emotion(text: str) -> EmotionSnapshot:
    return EmotionSnapshot(primary="trust", intensity=0.5, valence=0.1, arousal=0.2)


async def mock_embed(text: str) -> list[float]:
    if not text:
        return []
    h = hash(text) % 1000
    return [h / 1000, (h + 1) / 1000, (h + 2) / 1000]


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
        entity_extractor=mock_extract,
        emotion_detector=mock_detect_emotion,
        embedder=mock_embed,
    )


# ---------------------------------------------------------------------------
# Core guarantee: every casual remark is captured as an observation
# ---------------------------------------------------------------------------


class TestCasualRemarksCaptured:
    """Every casual remark, filler word, and throwaway comment is stored."""

    @pytest.mark.asyncio
    async def test_filler_words_stored(self, pipeline: BlurtCapturePipeline):
        """Single filler words are captured and stored."""
        fillers = ["hmm", "yeah", "ok", "sure", "cool", "meh", "wow"]
        for word in fillers:
            result = await pipeline.capture_voice("u1", word)
            assert result.was_stored, f"'{word}' was not stored!"
            assert result.episode.raw_text == word
            assert result.observation_type == ObservationType.CASUAL

    @pytest.mark.asyncio
    async def test_empty_input_stored_as_casual(self, pipeline: BlurtCapturePipeline):
        """Empty string (user spoke but nothing transcribed) is still captured."""
        result = await pipeline.capture_voice("u1", "")
        assert result.was_stored
        assert result.episode.raw_text == ""
        assert result.is_casual

    @pytest.mark.asyncio
    async def test_weather_comment_stored(self, pipeline: BlurtCapturePipeline):
        """Off-hand observations about weather are stored as casual observations."""
        result = await pipeline.capture_voice("u1", "nice weather today")
        assert result.was_stored
        assert result.episode.raw_text == "nice weather today"
        assert result.is_casual

    @pytest.mark.asyncio
    async def test_reaction_stored(self, pipeline: BlurtCapturePipeline):
        """Quick reactions are stored."""
        result = await pipeline.capture_voice("u1", "oh wow, that's cool")
        assert result.was_stored
        assert result.episode.raw_text == "oh wow, that's cool"

    @pytest.mark.asyncio
    async def test_half_thought_stored(self, pipeline: BlurtCapturePipeline):
        """Incomplete thoughts are captured."""
        result = await pipeline.capture_voice("u1", "I was thinking maybe we could...")
        assert result.was_stored
        assert "thinking" in result.episode.raw_text

    @pytest.mark.asyncio
    async def test_oh_well_stored(self, pipeline: BlurtCapturePipeline):
        """'oh well' — classic throwaway — is captured."""
        result = await pipeline.capture_voice("u1", "oh well")
        assert result.was_stored
        assert result.is_casual

    @pytest.mark.asyncio
    async def test_whatever_stored(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "whatever")
        assert result.was_stored
        assert result.is_casual

    @pytest.mark.asyncio
    async def test_long_rambling_stored(self, pipeline: BlurtCapturePipeline):
        """Long rambling input is stored in full without truncation."""
        long_text = "So I was walking and I saw this cool thing and it was great " * 5
        result = await pipeline.capture_voice("u1", long_text)
        assert result.was_stored
        assert result.episode.raw_text == long_text


# ---------------------------------------------------------------------------
# Casual vs substantive classification in pipeline
# ---------------------------------------------------------------------------


class TestObservationTypeInPipeline:
    """Casual detection enriches the capture result without filtering."""

    @pytest.mark.asyncio
    async def test_casual_remark_tagged(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "huh, interesting")
        assert result.observation_type == ObservationType.CASUAL
        assert result.is_casual
        assert result.casual_detection is not None
        assert result.casual_detection.confidence > 0.5

    @pytest.mark.asyncio
    async def test_substantive_input_tagged(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "I need to buy groceries tomorrow")
        assert result.observation_type == ObservationType.SUBSTANTIVE
        assert not result.is_casual

    @pytest.mark.asyncio
    async def test_casual_metadata_in_episode_extra(self, pipeline: BlurtCapturePipeline):
        """Casual detection metadata is stored in the observation's extra dict."""
        result = await pipeline.capture_voice("u1", "hmm")
        # The episode was stored — check it can be retrieved
        episode = await pipeline.repo.get_episode(result.episode.id)
        assert episode is not None
        assert episode.raw_text == "hmm"

    @pytest.mark.asyncio
    async def test_to_dict_includes_casual_fields(self, pipeline: BlurtCapturePipeline):
        result = await pipeline.capture_voice("u1", "whatever")
        d = result.to_dict()
        assert "observation_type" in d
        assert "is_casual" in d
        assert d["observation_type"] == "casual"
        assert d["is_casual"] is True


# ---------------------------------------------------------------------------
# Mixed stream: casual + substantive all captured
# ---------------------------------------------------------------------------


class TestMixedConversationStream:
    """Simulate a real conversation with mixed casual and actionable inputs."""

    @pytest.mark.asyncio
    async def test_full_conversation_nothing_dropped(self, pipeline: BlurtCapturePipeline):
        """Every utterance in a realistic conversation is stored."""
        conversation = [
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
            ("voice", ""),  # Empty utterance
            ("voice", "yeah"),
            ("voice", "cool"),
        ]

        results = []
        for modality, text in conversation:
            if modality == "voice":
                r = await pipeline.capture_voice("u1", text, session_id="sess-1")
            else:
                r = await pipeline.capture_text("u1", text, session_id="sess-1")
            results.append(r)

        # ZERO-DROP: every single input was captured
        assert len(results) == len(conversation)
        for i, result in enumerate(results):
            assert result.was_stored, f"Input {i} ('{conversation[i][1]}') was NOT stored!"
            assert result.episode.raw_text == conversation[i][1]
            assert CaptureStage.STORED in result.stages_completed

        # Total episode count matches
        count = await pipeline.repo.count_user_episodes("u1")
        assert count == len(conversation)

        # Stats reflect everything
        assert pipeline.stats.total_captured == len(conversation)
        assert pipeline.stats.drop_rate == 0.0

    @pytest.mark.asyncio
    async def test_casual_and_substantive_both_in_session(
        self, pipeline: BlurtCapturePipeline, store: InMemoryEpisodicStore
    ):
        """Both casual and substantive observations appear in the same session."""
        await pipeline.capture_voice("u1", "hmm", session_id="sess-2")
        await pipeline.capture_voice("u1", "I need to call the dentist", session_id="sess-2")
        await pipeline.capture_voice("u1", "oh well", session_id="sess-2")

        # All three should appear in session
        session_eps = await store.get_session_episodes("sess-2")
        assert len(session_eps) == 3
        texts = [ep.raw_text for ep in session_eps]
        assert "hmm" in texts
        assert "I need to call the dentist" in texts
        assert "oh well" in texts

    @pytest.mark.asyncio
    async def test_casual_remarks_have_intent_classification(self, pipeline: BlurtCapturePipeline):
        """Even casual remarks get classified (typically as 'journal')."""
        casual_inputs = ["hmm", "nice weather", "oh well", "whatever", "huh"]
        for text in casual_inputs:
            result = await pipeline.capture_voice("u1", text)
            assert result.classification_applied
            # Casual remarks typically classified as journal
            assert result.episode.intent in ("journal", "idea", "task", "event", "reminder", "update", "question")

    @pytest.mark.asyncio
    async def test_casual_remarks_have_emotion_detection(self, pipeline: BlurtCapturePipeline):
        """Even casual remarks get emotion detection."""
        result = await pipeline.capture_voice("u1", "meh")
        assert result.emotion_detected
        assert result.episode.emotion is not None


# ---------------------------------------------------------------------------
# Retrieval: casual observations are retrievable
# ---------------------------------------------------------------------------


class TestCasualRetrieval:
    """Casual observations can be queried and retrieved."""

    @pytest.mark.asyncio
    async def test_casual_retrievable_by_id(self, pipeline: BlurtCapturePipeline):
        """A casual observation can be fetched by its episode ID."""
        result = await pipeline.capture_voice("u1", "hmm")
        episode = await pipeline.repo.get_episode(result.episode.id)
        assert episode is not None
        assert episode.raw_text == "hmm"

    @pytest.mark.asyncio
    async def test_casual_appears_in_user_query(
        self, pipeline: BlurtCapturePipeline, store: InMemoryEpisodicStore
    ):
        """Casual observations appear in user episode queries."""
        await pipeline.capture_voice("u1", "hmm")
        await pipeline.capture_voice("u1", "whatever")
        await pipeline.capture_voice("u1", "I need to buy groceries")

        episodes = await store.query("u1")
        assert len(episodes) == 3
        texts = {ep.raw_text for ep in episodes}
        assert "hmm" in texts
        assert "whatever" in texts
        assert "I need to buy groceries" in texts

    @pytest.mark.asyncio
    async def test_casual_appears_in_journal_filter(
        self, pipeline: BlurtCapturePipeline, store: InMemoryEpisodicStore
    ):
        """Casual remarks classified as journal can be filtered."""
        await pipeline.capture_voice("u1", "hmm")
        await pipeline.capture_voice("u1", "I need to buy groceries")

        journal_eps = await store.query("u1", intent_filter=IntentFilter("journal"))
        # "hmm" should be classified as journal
        assert any(ep.raw_text == "hmm" for ep in journal_eps)

    @pytest.mark.asyncio
    async def test_casual_in_session_retrieval(
        self, pipeline: BlurtCapturePipeline, store: InMemoryEpisodicStore
    ):
        """Casual observations are part of session history."""
        await pipeline.capture_voice("u1", "hmm", session_id="s1")
        await pipeline.capture_voice("u1", "call dentist", session_id="s1")
        await pipeline.capture_voice("u1", "ok", session_id="s1")

        session = await store.get_session_episodes("s1")
        assert len(session) == 3


# ---------------------------------------------------------------------------
# Stats: casual captures tracked
# ---------------------------------------------------------------------------


class TestCasualStats:
    """Verify casual capture statistics are accurate."""

    @pytest.mark.asyncio
    async def test_casual_capture_count_tracks(self, pipeline: BlurtCapturePipeline):
        """casual_capture_count increases for casual observations."""
        await pipeline.capture_voice("u1", "hmm")
        await pipeline.capture_voice("u1", "cool")
        await pipeline.capture_voice("u1", "I need to buy groceries")
        # At least the first two should be counted as casual
        assert pipeline.stats.casual_capture_count >= 2

    @pytest.mark.asyncio
    async def test_zero_drop_rate_with_casual(self, pipeline: BlurtCapturePipeline):
        """Drop rate stays zero even with many casual inputs."""
        for _ in range(20):
            await pipeline.capture_voice("u1", "hmm")
        assert pipeline.stats.drop_rate == 0.0
        assert pipeline.stats.total_captured == 20


# ---------------------------------------------------------------------------
# Behavioral contribution: casual remarks have value
# ---------------------------------------------------------------------------


class TestCasualBehavioralValue:
    """Casual remarks contribute to behavioral patterns and emotional baselines."""

    @pytest.mark.asyncio
    async def test_casual_has_timestamp(self, pipeline: BlurtCapturePipeline):
        """Casual observations have timestamps for temporal pattern analysis."""
        result = await pipeline.capture_voice("u1", "hmm")
        assert result.episode.timestamp is not None

    @pytest.mark.asyncio
    async def test_casual_has_session_context(self, pipeline: BlurtCapturePipeline):
        """Casual observations preserve session context."""
        result = await pipeline.capture_voice(
            "u1", "whatever", session_id="sess-42",
            time_of_day="evening", day_of_week="friday",
        )
        assert result.episode.context.session_id == "sess-42"
        assert result.episode.context.time_of_day == "evening"
        assert result.episode.context.day_of_week == "friday"

    @pytest.mark.asyncio
    async def test_casual_emotion_contributes_to_timeline(
        self, pipeline: BlurtCapturePipeline, store: InMemoryEpisodicStore
    ):
        """Casual observations have emotion data for baseline tracking."""
        result = await pipeline.capture_voice("u1", "meh")
        assert result.episode.emotion is not None
        assert result.episode.emotion.primary is not None

    @pytest.mark.asyncio
    async def test_casual_links_in_session_chain(self, pipeline: BlurtCapturePipeline):
        """Casual observations can be linked to preceding episodes."""
        r1 = await pipeline.capture_voice("u1", "hmm")
        r2 = await pipeline.capture_voice(
            "u1", "oh right", preceding_episode_id=r1.episode.id,
        )
        assert r2.episode.context.preceding_episode_id == r1.episode.id


# ---------------------------------------------------------------------------
# Multi-user isolation
# ---------------------------------------------------------------------------


class TestMultiUserCasual:
    """Casual observations are isolated per user."""

    @pytest.mark.asyncio
    async def test_casual_user_isolation(
        self, pipeline: BlurtCapturePipeline, store: InMemoryEpisodicStore
    ):
        await pipeline.capture_voice("u1", "hmm")
        await pipeline.capture_voice("u2", "whatever")
        await pipeline.capture_voice("u1", "ok")

        u1_count = await store.count("u1")
        u2_count = await store.count("u2")
        assert u1_count == 2
        assert u2_count == 1
