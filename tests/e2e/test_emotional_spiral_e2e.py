"""E2E Scenario 10: Emotional spiral — emotion trend detection, pattern flagging,
proactive empathy response, and mood-gated task surfacing.

Validates that Blurt correctly:
1. Detects escalating negative emotion trends across successive blurts
2. Flags emotional patterns (sustained sadness, anger spirals)
3. Adjusts task surfacing based on the user's emotional state (mood-gating)
4. Records emotion snapshots in episodic memory with correct valence/arousal
5. Filters episodes by emotion and builds emotion timelines
6. Surfaces lower-cognitive-load tasks when valence is negative
7. Avoids surfacing high-arousal tasks during emotional distress (anti-shame)

Tests exercise the full pipeline: capture → emotion detection → episodic
storage → pattern creation → task scoring with emotional alignment.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from blurt.memory.episodic import (
    EmotionFilter,
    InMemoryEpisodicStore,
)
from blurt.services.task_surfacing import (
    EnergyLevel,
    SurfaceableTask,
    SurfacingWeights,
    TaskScoringEngine,
    UserContext,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPIRAL_BLURTS: list[dict[str, str]] = [
    # Neutral → slightly negative → escalating sadness → anger
    {"text": "Just finished my morning coffee", "expected_emotion": "trust"},
    {"text": "Feeling a bit down today", "expected_emotion": "sadness"},
    {"text": "Really sad about how things are going", "expected_emotion": "sadness"},
    {"text": "I'm so frustrated with this project", "expected_emotion": "anger"},
    {"text": "Everything is making me angry and annoyed", "expected_emotion": "anger"},
]


class TestEmotionTrendDetection:
    """Capture a sequence of blurts with escalating negative emotions and
    verify that the episodic store records the trend accurately."""

    async def test_escalating_negative_emotions_captured(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """A sequence of increasingly negative blurts stores correct emotions."""
        results = []
        for blurt in SPIRAL_BLURTS:
            result = await capture_blurt_via_api(raw_text=blurt["text"])
            results.append(result)

        # Verify all episodes were stored
        count = await episodic_store.count(test_user_id)
        assert count == len(SPIRAL_BLURTS)

        # Retrieve episodes and check emotion primaries
        episodes = await episodic_store.query(test_user_id, limit=50)
        # Episodes are returned newest-first; reverse for chronological order
        episodes.reverse()

        for i, blurt in enumerate(SPIRAL_BLURTS):
            assert episodes[i].emotion.primary == blurt["expected_emotion"], (
                f"Episode {i} expected emotion '{blurt['expected_emotion']}', "
                f"got '{episodes[i].emotion.primary}'"
            )

    async def test_valence_decreases_over_spiral(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Valence should decrease as the emotional spiral deepens."""
        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        episodes = await episodic_store.query(test_user_id, limit=50)
        episodes.reverse()  # chronological order

        # First episode is neutral (trust) → last episodes are negative
        first_valence = episodes[0].emotion.valence
        last_valence = episodes[-1].emotion.valence
        assert last_valence < first_valence, (
            f"Expected valence to decrease: first={first_valence}, last={last_valence}"
        )

    async def test_negative_valence_episodes_have_is_negative(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Episodes with negative valence should report is_negative=True."""
        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        episodes = await episodic_store.query(test_user_id, limit=50)
        negative_episodes = [ep for ep in episodes if ep.emotion.is_negative]
        # At least the sad and angry episodes should be negative
        assert len(negative_episodes) >= 3, (
            f"Expected at least 3 negative episodes, got {len(negative_episodes)}"
        )


class TestEmotionFiltering:
    """Verify episodic store emotion filters work for emotional spiral data."""

    async def test_filter_episodes_by_primary_emotion(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Filter episodes by primary emotion returns correct subset."""
        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        # Filter for sadness episodes
        sad_episodes = await episodic_store.query(
            test_user_id,
            emotion_filter=EmotionFilter(primary="sadness"),
        )
        assert len(sad_episodes) == 2
        for ep in sad_episodes:
            assert ep.emotion.primary == "sadness"

    async def test_filter_by_negative_valence_range(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Filter episodes by negative valence range."""
        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        negative = await episodic_store.query(
            test_user_id,
            emotion_filter=EmotionFilter(valence_range=(-1.0, -0.2)),
        )
        # Sad and angry episodes should fall in this range
        assert len(negative) >= 3
        for ep in negative:
            assert ep.emotion.valence <= -0.2

    async def test_filter_by_min_intensity(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Filter by minimum emotion intensity."""
        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        intense = await episodic_store.query(
            test_user_id,
            emotion_filter=EmotionFilter(min_intensity=1.0),
        )
        # Sad (1.5) and angry (1.5) episodes should match
        assert len(intense) >= 2
        for ep in intense:
            assert ep.emotion.intensity >= 1.0


class TestEmotionTimeline:
    """Verify emotion timeline retrieval for spiral analysis."""

    async def test_emotion_timeline_returns_chronological_order(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Emotion timeline returns episodes in chronological order."""
        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)

        timeline = await episodic_store.get_emotion_timeline(
            test_user_id, start, end
        )

        assert len(timeline) == len(SPIRAL_BLURTS)
        # Verify chronological order
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp

    async def test_timeline_shows_emotion_progression(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Timeline shows progression from neutral → sad → angry."""
        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        now = datetime.now(timezone.utc)
        timeline = await episodic_store.get_emotion_timeline(
            test_user_id,
            now - timedelta(hours=1),
            now + timedelta(hours=1),
        )

        emotions_sequence = [ep.emotion.primary for ep in timeline]
        # Should start with trust, contain sadness, end with anger
        assert emotions_sequence[0] == "trust"
        assert "sadness" in emotions_sequence
        assert emotions_sequence[-1] == "anger"


class TestPatternFlagging:
    """Create emotion patterns from spiral data and verify flagging."""

    async def test_create_negative_emotion_pattern(
        self,
        create_pattern_via_api: Any,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """A sustained negative emotion trend can be flagged as a pattern."""
        # First capture the spiral
        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        # Retrieve negative episodes as evidence
        negative_episodes = await episodic_store.query(
            test_user_id,
            emotion_filter=EmotionFilter(valence_range=(-1.0, -0.2)),
        )
        evidence_ids = [ep.id for ep in negative_episodes]

        # Create an emotion pattern flagging the spiral
        pattern = await create_pattern_via_api(
            pattern_type="mood",
            description="Sustained negative emotion spiral detected — sadness escalating to anger",
            parameters={
                "trend": "escalating_negative",
                "primary_emotions": ["sadness", "anger"],
                "episode_count": len(evidence_ids),
            },
            confidence=0.75,
            observation_count=len(evidence_ids),
            supporting_evidence=evidence_ids,
        )

        assert pattern["pattern_type"] == "mood_cycle"
        assert pattern["confidence"] == 0.75
        assert pattern["is_active"] is True
        assert pattern["observation_count"] == len(evidence_ids)

    async def test_reinforce_emotion_pattern_with_new_evidence(
        self,
        client: httpx.AsyncClient,
        create_pattern_via_api: Any,
        capture_blurt_via_api: Any,
        test_user_id: str,
    ):
        """Reinforcing an emotion pattern increases its confidence."""
        pattern = await create_pattern_via_api(
            pattern_type="mood",
            description="Recurring sadness pattern",
            confidence=0.5,
        )
        pattern_id = pattern["id"]

        # Capture another sad blurt to reinforce
        await capture_blurt_via_api(raw_text="I feel really sad again today")

        # Reinforce the pattern
        resp = await client.put(
            f"/api/v1/users/{test_user_id}/patterns/{pattern_id}/reinforce",
            json={
                "evidence": "another sad episode detected",
                "confidence_boost": 0.15,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == pytest.approx(0.65, abs=0.01)
        assert data["observation_count"] == 2


class TestMoodGatedSurfacing:
    """Verify that task surfacing respects the user's emotional state
    and avoids overwhelming the user during emotional distress."""

    async def test_low_valence_suppresses_high_arousal_tasks(self):
        """When user valence is negative, high-arousal tasks score lower."""
        engine = TaskScoringEngine()

        high_arousal_task = SurfaceableTask(
            id="task-intense",
            content="Give critical presentation to board",
            capture_arousal=0.9,
            capture_valence=-0.2,
            estimated_energy=EnergyLevel.HIGH,
        )

        low_arousal_task = SurfaceableTask(
            id="task-gentle",
            content="Organize desktop files",
            capture_arousal=0.2,
            capture_valence=0.3,
            estimated_energy=EnergyLevel.LOW,
        )

        # User is in a negative emotional state (post-spiral)
        distressed_context = UserContext(
            energy=EnergyLevel.LOW,
            current_valence=-0.6,
            current_arousal=0.3,
        )

        score_intense = engine.score_single(
            high_arousal_task, distressed_context, use_thompson=False
        )
        score_gentle = engine.score_single(
            low_arousal_task, distressed_context, use_thompson=False
        )

        # Gentle task should score higher when user is distressed
        emotional_score_intense = dict(
            (s.signal.value, s.value) for s in score_intense.signal_scores
        ).get("emotional_alignment", 0)
        emotional_score_gentle = dict(
            (s.signal.value, s.value) for s in score_gentle.signal_scores
        ).get("emotional_alignment", 0)

        assert emotional_score_gentle > emotional_score_intense, (
            f"Gentle task emotional alignment ({emotional_score_gentle}) should be "
            f"higher than intense task ({emotional_score_intense}) when user is distressed"
        )

    async def test_positive_valence_allows_all_tasks(self):
        """When user mood is positive, all tasks score well on emotional alignment."""
        engine = TaskScoringEngine()

        task = SurfaceableTask(
            id="task-any",
            content="Work on complex analysis",
            capture_arousal=0.7,
            capture_valence=0.5,
            estimated_energy=EnergyLevel.HIGH,
        )

        happy_context = UserContext(
            energy=EnergyLevel.HIGH,
            current_valence=0.7,
            current_arousal=0.6,
        )

        scored = engine.score_single(task, happy_context, use_thompson=False)
        emotional_score = dict(
            (s.signal.value, s.value) for s in scored.signal_scores
        ).get("emotional_alignment", 0)

        # Positive mood → good emotional alignment for any task
        assert emotional_score >= 0.5

    async def test_mood_gating_filters_in_ranking(self):
        """Full ranking pass with mixed tasks respects mood gating."""
        engine = TaskScoringEngine(
            weights=SurfacingWeights(emotional_alignment=0.5),
            min_score=0.0,
        )

        tasks = [
            SurfaceableTask(
                id="hard-task",
                content="Debate with stakeholders",
                capture_arousal=0.9,
                capture_valence=-0.3,
                estimated_energy=EnergyLevel.HIGH,
            ),
            SurfaceableTask(
                id="easy-task",
                content="Water the plants",
                capture_arousal=0.1,
                capture_valence=0.5,
                estimated_energy=EnergyLevel.LOW,
            ),
        ]

        sad_context = UserContext(
            energy=EnergyLevel.LOW,
            current_valence=-0.5,
            current_arousal=0.2,
        )

        result = engine.score_and_rank(tasks, sad_context)
        assert result.has_tasks
        # The easy/positive task should rank higher when user is sad
        top = result.top_task
        assert top is not None
        assert top.task.id == "easy-task"


class TestProactiveEmpathyResponse:
    """Verify that capture responses during negative emotional states
    reflect empathetic, anti-shame design."""

    async def test_sad_blurt_response_contains_emotion(
        self,
        capture_blurt_via_api: Any,
    ):
        """Capturing a sad blurt records sadness emotion correctly."""
        result = await capture_blurt_via_api(
            raw_text="I'm feeling really sad and down about everything"
        )
        assert result["episode"]["emotion"]["primary"] == "sadness"
        assert result["episode"]["emotion"]["valence"] < 0

    async def test_angry_blurt_response_records_high_arousal(
        self,
        capture_blurt_via_api: Any,
    ):
        """Angry blurts record high arousal."""
        result = await capture_blurt_via_api(
            raw_text="I'm so angry and frustrated right now"
        )
        assert result["episode"]["emotion"]["primary"] == "anger"
        assert result["episode"]["emotion"]["arousal"] > 0.5

    async def test_happy_blurt_after_spiral_shows_recovery(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """A positive blurt after a negative spiral marks emotional recovery."""
        # Build a sad spiral
        await capture_blurt_via_api(raw_text="I'm feeling sad today")
        await capture_blurt_via_api(raw_text="Everything is making me frustrated")
        # Recovery blurt
        await capture_blurt_via_api(
            raw_text="Actually I'm feeling happy and excited now!"
        )

        episodes = await episodic_store.query(test_user_id, limit=10)
        episodes.reverse()  # chronological

        # Last episode should show positive recovery
        recovery = episodes[-1]
        assert recovery.emotion.primary == "joy"
        assert recovery.emotion.valence > 0

        # Earlier episodes should be negative
        assert episodes[0].emotion.primary == "sadness"
        assert episodes[0].emotion.valence < 0


class TestEmotionalSpiralSummary:
    """Verify that episode summaries correctly aggregate emotional data
    from a spiral."""

    async def test_build_summary_captures_dominant_emotions(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Summary of spiral episodes captures dominant emotions."""
        from blurt.memory.episodic import build_summary

        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        episodes = await episodic_store.query(test_user_id, limit=50)

        summary = build_summary(
            user_id=test_user_id,
            episodes=episodes,
            summary_text="Emotional spiral: trust → sadness → anger over 5 blurts",
        )

        assert summary.episode_count == len(SPIRAL_BLURTS)
        # Dominant emotions should include sadness and anger (2 each)
        dominant_primaries = [e.primary for e in summary.dominant_emotions]
        assert "anger" in dominant_primaries
        assert "sadness" in dominant_primaries

    async def test_compress_spiral_marks_episodes(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Compressing spiral episodes marks them and stores the summary."""
        from blurt.memory.episodic import compress_episodes

        for blurt in SPIRAL_BLURTS:
            await capture_blurt_via_api(raw_text=blurt["text"])

        episodes = await episodic_store.query(test_user_id, limit=50)

        summary = await compress_episodes(
            store=episodic_store,
            user_id=test_user_id,
            episodes=episodes,
            summary_text="Compressed emotional spiral",
        )

        assert summary.episode_count == len(SPIRAL_BLURTS)

        # Episodes should be marked as compressed
        for ep in episodes:
            stored = await episodic_store.get(ep.id)
            assert stored is not None
            assert stored.is_compressed is True
            assert stored.compressed_into_id == summary.id

        # Summary should be retrievable
        summaries = await episodic_store.get_summaries(test_user_id)
        assert len(summaries) == 1
        assert summaries[0].id == summary.id


class TestCrossCuttingEmotionConcerns:
    """Cross-cutting concerns: session isolation, multi-emotion handling,
    and emotion data in API responses."""

    async def test_emotion_data_in_episode_api_response(
        self,
        create_episode_via_api: Any,
        client: httpx.AsyncClient,
        test_user_id: str,
    ):
        """Episode API response includes full emotion snapshot."""
        episode = await create_episode_via_api(
            raw_text="I'm very worried and anxious about the deadline",
            emotion_primary="fear",
            emotion_intensity=1.2,
            emotion_valence=-0.5,
            emotion_arousal=0.7,
        )

        resp = await client.get(
            f"/api/v1/episodes/{episode['id']}",
            params={"user_id": test_user_id},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["emotion"]["primary"] == "fear"
        assert data["emotion"]["valence"] == -0.5
        assert data["emotion"]["arousal"] == 0.7

    async def test_session_isolates_emotion_spirals(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Emotion spirals in different sessions are independently queryable."""
        # Session 1: sad spiral
        await capture_blurt_via_api(
            raw_text="I'm really sad", session_id="session-sad"
        )
        await capture_blurt_via_api(
            raw_text="Still feeling down and depressed", session_id="session-sad"
        )

        # Session 2: happy session
        await capture_blurt_via_api(
            raw_text="I'm so happy and excited!", session_id="session-happy"
        )

        sad_session = await episodic_store.get_session_episodes("session-sad")
        happy_session = await episodic_store.get_session_episodes("session-happy")

        assert len(sad_session) == 2
        assert len(happy_session) == 1

        # All sad session episodes should have negative valence
        for ep in sad_session:
            assert ep.emotion.valence < 0

        # Happy session should have positive valence
        assert happy_session[0].emotion.valence > 0

    async def test_arousal_properties_on_emotion_snapshot(
        self,
        capture_blurt_via_api: Any,
        episodic_store: InMemoryEpisodicStore,
        test_user_id: str,
    ):
        """Verify is_negative and is_high_arousal properties on stored episodes."""
        await capture_blurt_via_api(
            raw_text="I'm so angry and frustrated!"
        )

        episodes = await episodic_store.query(test_user_id, limit=1)
        assert len(episodes) == 1

        ep = episodes[0]
        assert ep.emotion.primary == "anger"
        assert ep.emotion.is_negative is True
        assert ep.emotion.is_high_arousal is True
