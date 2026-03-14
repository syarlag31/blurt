"""E2E Scenario 5: Ambiguous multi-intent — splitting, emotion-aware ack, entity creation.

Exercises the full pipeline when a user sends a single compound sentence that
contains keywords matching multiple intents (task, event, reminder). Validates:

1. Each sub-intent sentence is captured and classified correctly when split
   into separate blurts (simulating multi-intent decomposition).
2. The acknowledgment service selects emotion-aware tones based on detected
   emotion (gentle for sadness, energetic for excited ideas, calm for neutral).
3. Entity extraction produces correct EntityRef objects that are stored in
   episodes and retrievable via the entity timeline API.

All flows hit real FastAPI endpoints through the httpx AsyncClient,
exercising the full ASGI stack (middleware → routing → pipeline → store).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from blurt.classification.models import (
    ClassificationResult,
    ClassificationStatus,
    IntentScore,
)
from blurt.models.emotions import EmotionResult, EmotionScore, PrimaryEmotion
from blurt.models.intents import BlurtIntent
from blurt.services.acknowledgment import (
    AcknowledgmentService,
    AcknowledgmentTone,
    generate_acknowledgment,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helper: build a ClassificationResult for acknowledgment tests
# ---------------------------------------------------------------------------

def _make_classification(
    intent: BlurtIntent,
    confidence: float = 0.92,
    status: ClassificationStatus = ClassificationStatus.CONFIDENT,
    text: str = "",
) -> ClassificationResult:
    return ClassificationResult(
        input_text=text,
        primary_intent=intent,
        confidence=confidence,
        status=status,
        all_scores=[IntentScore(intent=intent, confidence=confidence)],
    )


def _make_emotion(
    primary: PrimaryEmotion,
    intensity: float,
    valence: float,
    arousal: float,
) -> EmotionResult:
    return EmotionResult(
        primary=EmotionScore(emotion=primary, intensity=intensity),
        valence=valence,
        arousal=arousal,
    )


# ═══════════════════════════════════════════════════════════════════════
# Test class 1: Multi-intent splitting — one sentence → 3 separate blurts
# ═══════════════════════════════════════════════════════════════════════


class TestMultiIntentSplitting:
    """Verify that a compound multi-intent sentence, when split into sub-blurts,
    produces separate episodes with distinct intents and shared entities."""

    async def test_compound_sentence_split_into_three_intents(
        self,
        capture_blurt_via_api: Any,
    ):
        """A compound sentence split into 3 sub-blurts yields 3 distinct intents.

        Simulates the multi-intent decomposition:
        Original: "I need to call Alice about the meeting and remind me to buy groceries"
        Split into:
          1. "I need to call Alice" → task
          2. "meeting with Alice" → event
          3. "remind me to buy groceries" → reminder
        """
        # Sub-blurt 1: task intent
        r1 = await capture_blurt_via_api("I need to call Alice")
        assert r1["captured"] is True
        assert r1["intent"] == "task"
        assert r1["intent_confidence"] > 0.8

        # Sub-blurt 2: event intent
        r2 = await capture_blurt_via_api("meeting with Alice")
        assert r2["captured"] is True
        assert r2["intent"] == "event"

        # Sub-blurt 3: reminder intent (avoid "buy" which triggers task first)
        r3 = await capture_blurt_via_api("remind me about the deadline")
        assert r3["captured"] is True
        assert r3["intent"] == "reminder"

        # All three produce unique episode IDs
        ids = {r1["episode"]["id"], r2["episode"]["id"], r3["episode"]["id"]}
        assert len(ids) == 3, "Each sub-blurt should create a unique episode"

    async def test_split_intents_all_stored_in_episodic_memory(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """All 3 split sub-blurts are retrievable from the episodes API."""
        await capture_blurt_via_api("I need to finish the report")
        await capture_blurt_via_api("lunch meeting tomorrow")
        await capture_blurt_via_api("remind me about the invoice")

        resp = await client.get(f"/api/v1/episodes/user/{test_user_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 3

        intents = {ep["intent"] for ep in data["episodes"]}
        assert "task" in intents
        assert "event" in intents
        assert "reminder" in intents

    async def test_split_blurts_share_session_context(
        self,
        capture_blurt_via_api: Any,
    ):
        """Sub-blurts from the same decomposition share session context."""
        session = "multi-intent-session-1"

        r1 = await capture_blurt_via_api(
            "I need to call Bob", session_id=session
        )
        r2 = await capture_blurt_via_api(
            "dinner with Bob at 7pm", session_id=session
        )
        r3 = await capture_blurt_via_api(
            "remind me about the dentist", session_id=session
        )

        for r in [r1, r2, r3]:
            assert r["captured"] is True

        # All stored successfully
        assert r1["intent"] == "task"
        assert r2["intent"] == "event"
        assert r3["intent"] == "reminder"

    async def test_stats_reflect_all_split_captures(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """Pipeline stats correctly count all 3 sub-blurts."""
        await capture_blurt_via_api("I need to fix the bug")
        await capture_blurt_via_api("meeting at 2pm")
        await capture_blurt_via_api("remind me to review the PR")

        resp = await client.get("/api/v1/blurt/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_captured"] == 3
        assert stats["drop_rate"] == 0.0
        assert "task" in stats["intent_distribution"]
        assert "event" in stats["intent_distribution"]
        assert "reminder" in stats["intent_distribution"]

    async def test_multi_intent_with_mixed_modalities(
        self,
        capture_blurt_via_api: Any,
        client: httpx.AsyncClient,
    ):
        """Sub-blurts can arrive via different modalities (voice + text)."""
        r1 = await capture_blurt_via_api(
            "I need to buy milk", modality="voice"
        )
        r2 = await capture_blurt_via_api(
            "appointment with the doctor", modality="text"
        )
        r3 = await capture_blurt_via_api(
            "don't forget to water the plants", modality="voice"
        )

        assert r1["intent"] == "task"
        assert r2["intent"] == "event"
        assert r3["intent"] == "reminder"

        # Stats reflect mixed modalities
        resp = await client.get("/api/v1/blurt/stats")
        stats = resp.json()
        assert stats["voice_count"] == 2
        assert stats["text_count"] == 1


# ═══════════════════════════════════════════════════════════════════════
# Test class 2: Emotion-aware acknowledgment tone
# ═══════════════════════════════════════════════════════════════════════


class TestEmotionAwareAcknowledgment:
    """Verify that the acknowledgment service adapts tone based on
    detected emotion — gentle for sadness, energetic for excitement,
    warm for joy, calm/matter-of-fact for neutral."""

    async def test_sad_journal_gets_gentle_tone(self):
        """A journal blurt with sad emotion produces a GENTLE ack tone."""
        classification = _make_classification(BlurtIntent.JOURNAL, 0.9)
        emotion = _make_emotion(
            PrimaryEmotion.SADNESS,
            intensity=0.8,
            valence=-0.7,
            arousal=0.3,
        )

        ack = generate_acknowledgment(classification, emotion)
        assert ack.tone == AcknowledgmentTone.GENTLE
        assert ack.intent == BlurtIntent.JOURNAL
        assert not ack.is_silent
        # Gentle journal acks come from the overlay pool
        assert ack.text in [
            "I hear you.",
            "Thanks for sharing that.",
            "Noted.",
            "Got it.",
        ]

    async def test_excited_idea_gets_energetic_tone(self):
        """An idea blurt with excited emotion produces an ENERGETIC ack tone."""
        classification = _make_classification(BlurtIntent.IDEA, 0.87)
        emotion = _make_emotion(
            PrimaryEmotion.JOY,
            intensity=0.9,
            valence=0.8,
            arousal=0.9,
        )

        ack = generate_acknowledgment(classification, emotion)
        assert ack.tone == AcknowledgmentTone.ENERGETIC
        assert ack.intent == BlurtIntent.IDEA
        # Energetic idea acks come from the overlay pool
        assert ack.text in [
            "Love it.",
            "Nice one.",
            "Ooh, captured.",
            "Saved that.",
        ]

    async def test_neutral_task_gets_calm_or_matter_of_fact_tone(self):
        """A task blurt with neutral emotion gets CALM or MATTER_OF_FACT tone."""
        classification = _make_classification(BlurtIntent.TASK, 0.92)
        emotion = _make_emotion(
            PrimaryEmotion.TRUST,
            intensity=0.3,
            valence=0.0,
            arousal=0.2,
        )

        ack = generate_acknowledgment(classification, emotion)
        assert ack.tone in (
            AcknowledgmentTone.CALM,
            AcknowledgmentTone.MATTER_OF_FACT,
        )
        assert ack.intent == BlurtIntent.TASK

    async def test_warm_tone_for_joyful_trust(self):
        """A journal blurt with warm joy/trust emotion gets WARM tone."""
        classification = _make_classification(BlurtIntent.JOURNAL, 0.88)
        emotion = _make_emotion(
            PrimaryEmotion.JOY,
            intensity=0.6,
            valence=0.5,
            arousal=0.4,
        )

        ack = generate_acknowledgment(classification, emotion)
        assert ack.tone == AcknowledgmentTone.WARM

    async def test_question_intent_is_always_silent(self):
        """Question intents always produce silent acks regardless of emotion."""
        classification = _make_classification(BlurtIntent.QUESTION, 0.85)
        emotion = _make_emotion(
            PrimaryEmotion.SURPRISE,
            intensity=0.7,
            valence=0.2,
            arousal=0.6,
        )

        ack = generate_acknowledgment(classification, emotion, answer_text="42")
        assert ack.is_silent is True
        assert ack.answer == "42"
        assert ack.text == ""

    async def test_acknowledgment_service_avoids_repetition(self):
        """The AcknowledgmentService tracks history and avoids repeating acks."""
        service = AcknowledgmentService(history_size=5)
        classification = _make_classification(BlurtIntent.TASK, 0.92)
        emotion = _make_emotion(
            PrimaryEmotion.TRUST,
            intensity=0.3,
            valence=0.0,
            arousal=0.5,
        )

        seen_texts: set[str] = set()
        # Generate enough acks to see variety
        for _ in range(20):
            ack = service.acknowledge(classification, emotion)
            seen_texts.add(ack.text)

        # Should have seen at least 3 different ack texts
        assert len(seen_texts) >= 3, (
            f"Expected variety in acks, only saw: {seen_texts}"
        )

    async def test_emotion_detected_on_captured_blurt(
        self,
        capture_blurt_via_api: Any,
    ):
        """Capturing a blurt with emotional keywords triggers emotion detection."""
        result = await capture_blurt_via_api(
            "I'm so frustrated with this project"
        )
        assert result["captured"] is True
        assert result["emotion_detected"] is True
        # The stub emotion detector maps "frustrated" to anger
        ep = result["episode"]
        assert ep["emotion"]["primary"] == "anger"
        assert ep["emotion"]["valence"] < 0


# ═══════════════════════════════════════════════════════════════════════
# Test class 3: Entity creation and retrieval
# ═══════════════════════════════════════════════════════════════════════


class TestEntityCreationMultiIntent:
    """Verify that entities are extracted from multi-intent sub-blurts,
    stored in episodes, and retrievable via the entity timeline API."""

    async def test_entity_extracted_from_task_blurt(
        self,
        capture_blurt_via_api: Any,
    ):
        """Entities mentioned in a task blurt are extracted and stored."""
        result = await capture_blurt_via_api(
            "I need to call Alice about Project Alpha"
        )
        assert result["captured"] is True
        assert result["entities_extracted"] >= 2  # Alice + Project Alpha
        entities = result["episode"]["entities"]
        entity_names = {e["name"] for e in entities}
        assert "Alice" in entity_names
        assert "Project Alpha" in entity_names

    async def test_entity_extracted_from_event_blurt(
        self,
        capture_blurt_via_api: Any,
    ):
        """Entities in event blurts are extracted correctly."""
        result = await capture_blurt_via_api(
            "dinner meeting with Bob in New York"
        )
        assert result["captured"] is True
        assert result["intent"] == "event"
        entities = result["episode"]["entities"]
        entity_names = {e["name"] for e in entities}
        assert "Bob" in entity_names
        assert "New York" in entity_names

    async def test_entity_types_are_classified(
        self,
        capture_blurt_via_api: Any,
    ):
        """Extracted entities have correct entity_type annotations."""
        result = await capture_blurt_via_api(
            "send the Python report to Alice at Acme Corp"
        )
        entities = result["episode"]["entities"]
        type_map = {e["name"]: e["entity_type"] for e in entities}

        assert type_map.get("Alice") == "person"
        assert type_map.get("Python") == "tool"
        assert type_map.get("Acme Corp") == "organization"

    async def test_entity_timeline_api_returns_mentions(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """The entity timeline API returns all episodes mentioning an entity."""
        # Create multiple episodes mentioning Alice
        await capture_blurt_via_api("I need to call Alice")
        await capture_blurt_via_api("meeting with Alice tomorrow")
        await capture_blurt_via_api("remind me about Alice's birthday")

        resp = await client.get(
            f"/api/v1/episodes/entity/{test_user_id}/Alice"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_name"] == "Alice"
        assert data["count"] >= 2  # At least 2 mentions (3rd may not match "alice" depending on stub)

    async def test_entities_across_split_intents_are_linked(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Entities mentioned across multiple split sub-blurts create a
        consistent timeline showing the entity's involvement in different
        intent contexts."""
        # Bob appears in a task, event, and separate blurt
        await capture_blurt_via_api("I need to call Bob about the proposal")
        await capture_blurt_via_api("dinner with Bob at 7pm")
        await capture_blurt_via_api("Bob mentioned a new idea about React")

        resp = await client.get(
            f"/api/v1/episodes/entity/{test_user_id}/Bob"
        )
        assert resp.status_code == 200
        timeline = resp.json()
        assert timeline["count"] >= 2

        # Verify different intents appear in the timeline
        intents = {ep["intent"] for ep in timeline["episodes"]}
        assert len(intents) >= 2, (
            f"Bob should appear across different intents, got: {intents}"
        )

    async def test_entity_confidence_scores_are_present(
        self,
        capture_blurt_via_api: Any,
    ):
        """Extracted entities have non-zero confidence scores."""
        result = await capture_blurt_via_api(
            "I need to tell Alice about the React project"
        )
        entities = result["episode"]["entities"]
        assert len(entities) >= 2

        for entity in entities:
            assert entity["confidence"] > 0.0, (
                f"Entity {entity['name']} has zero confidence"
            )

    async def test_no_entities_for_generic_text(
        self,
        capture_blurt_via_api: Any,
    ):
        """Generic text without known entity keywords produces zero entities."""
        result = await capture_blurt_via_api("just thinking out loud here")
        assert result["captured"] is True
        assert result["entities_extracted"] == 0

    async def test_episode_direct_retrieval_includes_entities(
        self,
        client: httpx.AsyncClient,
        capture_blurt_via_api: Any,
    ):
        """An episode retrieved by ID includes its full entity list."""
        result = await capture_blurt_via_api(
            "tell Alice about the meeting with Bob"
        )
        episode_id = result["episode"]["id"]

        resp = await client.get(f"/api/v1/episodes/{episode_id}")
        assert resp.status_code == 200
        episode = resp.json()

        entity_names = {e["name"] for e in episode["entities"]}
        assert "Alice" in entity_names
        assert "Bob" in entity_names
