"""Tests for the episodic memory store.

Covers: append-only semantics, multi-dimensional filtering (time, entity,
emotion, intent, behavioral, session), pagination, semantic search,
entity/emotion timelines, compression, and summary aggregation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blurt.memory.episodic import (
    BehavioralFilter,
    BehavioralSignal,
    EmotionFilter,
    EmotionSnapshot,
    EntityFilter,
    EntityRef,
    Episode,
    EpisodeContext,
    EpisodeSummary,
    InMemoryEpisodicStore,
    InputModality,
    IntentFilter,
    SessionFilter,
    TimeRangeFilter,
    build_summary,
    compress_episodes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(
    session_id: str = "session-1",
    time_of_day: str = "morning",
    day_of_week: str = "monday",
) -> EpisodeContext:
    return EpisodeContext(
        time_of_day=time_of_day,
        day_of_week=day_of_week,
        session_id=session_id,
    )


def _emo(
    primary: str = "joy",
    intensity: float = 1.0,
    valence: float = 0.5,
    arousal: float = 0.5,
) -> EmotionSnapshot:
    return EmotionSnapshot(
        primary=primary, intensity=intensity, valence=valence, arousal=arousal
    )


def _ep(
    user_id: str = "user-1",
    raw_text: str = "Pick up groceries",
    intent: str = "task",
    confidence: float = 0.95,
    emotion: EmotionSnapshot | None = None,
    entities: list[EntityRef] | None = None,
    context: EpisodeContext | None = None,
    timestamp: datetime | None = None,
    behavioral_signal: BehavioralSignal = BehavioralSignal.NONE,
    embedding: list[float] | None = None,
) -> Episode:
    ep = Episode(
        user_id=user_id,
        raw_text=raw_text,
        modality=InputModality.VOICE,
        intent=intent,
        intent_confidence=confidence,
        emotion=emotion or _emo(),
        entities=entities or [],
        context=context or _ctx(),
        behavioral_signal=behavioral_signal,
        embedding=embedding,
    )
    if timestamp:
        ep.timestamp = timestamp
    return ep


@pytest.fixture
def store() -> InMemoryEpisodicStore:
    return InMemoryEpisodicStore()


# ---------------------------------------------------------------------------
# Append & Retrieve
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_and_get(store: InMemoryEpisodicStore):
    episode = _ep()
    stored = await store.append(episode)
    assert stored.id == episode.id
    assert stored.raw_text == "Pick up groceries"

    retrieved = await store.get(stored.id)
    assert retrieved is not None
    assert retrieved.id == stored.id


@pytest.mark.asyncio
async def test_append_is_append_only(store: InMemoryEpisodicStore):
    episode = _ep()
    await store.append(episode)
    with pytest.raises(ValueError, match="already exists"):
        await store.append(episode)


@pytest.mark.asyncio
async def test_get_nonexistent(store: InMemoryEpisodicStore):
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_count(store: InMemoryEpisodicStore):
    assert await store.count("user-1") == 0
    await store.append(_ep(user_id="user-1", raw_text="one"))
    await store.append(_ep(user_id="user-1", raw_text="two"))
    await store.append(_ep(user_id="user-2", raw_text="three"))
    assert await store.count("user-1") == 2
    assert await store.count("user-2") == 1


# ---------------------------------------------------------------------------
# Episode serialization
# ---------------------------------------------------------------------------


def test_episode_to_dict():
    ep = _ep(entities=[EntityRef(name="Sarah", entity_type="person")])
    d = ep.to_dict()
    assert d["raw_text"] == "Pick up groceries"
    assert d["intent"] == "task"
    assert d["emotion"]["primary"] == "joy"
    assert len(d["entities"]) == 1
    assert d["entities"][0]["name"] == "Sarah"
    assert d["is_compressed"] is False


# ---------------------------------------------------------------------------
# Query filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_by_time_range(store: InMemoryEpisodicStore):
    now = datetime.now(timezone.utc)
    await store.append(_ep(raw_text="old", timestamp=now - timedelta(days=7)))
    await store.append(_ep(raw_text="recent", timestamp=now - timedelta(hours=1)))
    await store.append(_ep(raw_text="now", timestamp=now))

    results = await store.query(
        "user-1",
        time_range=TimeRangeFilter(start=now - timedelta(hours=2)),
    )
    assert len(results) == 2
    assert results[0].raw_text == "now"  # newest first


@pytest.mark.asyncio
async def test_query_by_entity(store: InMemoryEpisodicStore):
    sarah = EntityRef(name="Sarah", entity_type="person")
    jake = EntityRef(name="Jake", entity_type="person")

    await store.append(_ep(raw_text="talk to Sarah", entities=[sarah]))
    await store.append(_ep(raw_text="talk to Jake", entities=[jake]))
    await store.append(_ep(raw_text="Sarah and Jake", entities=[sarah, jake]))

    results = await store.query(
        "user-1", entity_filter=EntityFilter(entity_name="Sarah")
    )
    assert len(results) == 2
    texts = {r.raw_text for r in results}
    assert "talk to Sarah" in texts
    assert "Sarah and Jake" in texts


@pytest.mark.asyncio
async def test_query_by_emotion(store: InMemoryEpisodicStore):
    await store.append(
        _ep(raw_text="happy", emotion=_emo("joy", intensity=2.0, valence=0.8))
    )
    await store.append(
        _ep(raw_text="sad", emotion=_emo("sadness", intensity=2.0, valence=-0.7))
    )
    await store.append(
        _ep(raw_text="slightly happy", emotion=_emo("joy", intensity=0.5, valence=0.3))
    )

    results = await store.query(
        "user-1",
        emotion_filter=EmotionFilter(primary="joy", min_intensity=2.0),
    )
    assert len(results) == 1
    assert results[0].raw_text == "happy"


@pytest.mark.asyncio
async def test_query_by_emotion_valence_range(store: InMemoryEpisodicStore):
    await store.append(_ep(raw_text="positive", emotion=_emo(valence=0.8)))
    await store.append(
        _ep(raw_text="negative", emotion=_emo(primary="sadness", valence=-0.6))
    )

    results = await store.query(
        "user-1",
        emotion_filter=EmotionFilter(valence_range=(-1.0, -0.2)),
    )
    assert len(results) == 1
    assert results[0].raw_text == "negative"


@pytest.mark.asyncio
async def test_query_by_intent(store: InMemoryEpisodicStore):
    await store.append(_ep(raw_text="task one", intent="task"))
    await store.append(_ep(raw_text="idea one", intent="idea"))
    await store.append(_ep(raw_text="journal", intent="journal"))

    results = await store.query("user-1", intent_filter=IntentFilter("idea"))
    assert len(results) == 1
    assert results[0].raw_text == "idea one"


@pytest.mark.asyncio
async def test_query_combined_filters(store: InMemoryEpisodicStore):
    sarah = EntityRef(name="Sarah", entity_type="person")

    await store.append(
        _ep(
            raw_text="happy about Sarah",
            intent="journal",
            entities=[sarah],
            emotion=_emo("joy", intensity=2.0),
        )
    )
    await store.append(
        _ep(
            raw_text="Sarah task",
            intent="task",
            entities=[sarah],
            emotion=_emo("anticipation", intensity=1.0),
        )
    )
    await store.append(
        _ep(
            raw_text="journal no sarah",
            intent="journal",
            emotion=_emo("joy", intensity=2.0),
        )
    )

    results = await store.query(
        "user-1",
        intent_filter=IntentFilter("journal"),
        entity_filter=EntityFilter(entity_name="Sarah"),
        emotion_filter=EmotionFilter(primary="joy"),
    )
    assert len(results) == 1
    assert results[0].raw_text == "happy about Sarah"


@pytest.mark.asyncio
async def test_query_excludes_compressed_by_default(store: InMemoryEpisodicStore):
    ep = _ep(raw_text="will be compressed")
    await store.append(ep)
    await store.mark_compressed([ep.id], "summary-1")

    results = await store.query("user-1")
    assert len(results) == 0

    results_with = await store.query("user-1", include_compressed=True)
    assert len(results_with) == 1


@pytest.mark.asyncio
async def test_query_pagination(store: InMemoryEpisodicStore):
    now = datetime.now(timezone.utc)
    for i in range(10):
        await store.append(
            _ep(
                raw_text=f"episode {i}",
                timestamp=now + timedelta(seconds=i),
            )
        )

    page1 = await store.query("user-1", limit=3, offset=0)
    page2 = await store.query("user-1", limit=3, offset=3)

    assert len(page1) == 3
    assert len(page2) == 3
    assert page1[0].raw_text != page2[0].raw_text


# ---------------------------------------------------------------------------
# Session retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_episodes(store: InMemoryEpisodicStore):
    now = datetime.now(timezone.utc)
    for i in range(3):
        await store.append(
            _ep(
                raw_text=f"msg {i}",
                context=_ctx(session_id="sess-A"),
                timestamp=now + timedelta(seconds=i),
            )
        )
    await store.append(
        _ep(raw_text="other session", context=_ctx(session_id="sess-B"))
    )

    session_eps = await store.get_session_episodes("sess-A")
    assert len(session_eps) == 3
    assert session_eps[0].raw_text == "msg 0"  # chronological
    assert session_eps[2].raw_text == "msg 2"


# ---------------------------------------------------------------------------
# Entity timeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_timeline(store: InMemoryEpisodicStore):
    now = datetime.now(timezone.utc)
    sarah = EntityRef(name="Sarah", entity_type="person")

    for i in range(5):
        await store.append(
            _ep(
                raw_text=f"Sarah mention {i}",
                entities=[sarah],
                timestamp=now + timedelta(hours=i),
            )
        )

    timeline = await store.get_entity_timeline("user-1", "Sarah", limit=3)
    assert len(timeline) == 3
    assert timeline[0].raw_text == "Sarah mention 4"  # newest first


@pytest.mark.asyncio
async def test_entity_timeline_case_insensitive(store: InMemoryEpisodicStore):
    sarah = EntityRef(name="Sarah", entity_type="person")
    await store.append(_ep(raw_text="about Sarah", entities=[sarah]))

    assert len(await store.get_entity_timeline("user-1", "sarah")) == 1
    assert len(await store.get_entity_timeline("user-1", "SARAH")) == 1


# ---------------------------------------------------------------------------
# Emotion timeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emotion_timeline(store: InMemoryEpisodicStore):
    now = datetime.now(timezone.utc)
    emotions = ["joy", "sadness", "anger", "fear", "anticipation"]

    for i in range(5):
        await store.append(
            _ep(
                raw_text=f"emotion {i}",
                emotion=_emo(primary=emotions[i]),
                timestamp=now + timedelta(hours=i),
            )
        )

    timeline = await store.get_emotion_timeline(
        "user-1",
        start=now + timedelta(hours=1),
        end=now + timedelta(hours=3),
    )
    assert len(timeline) == 3
    assert timeline[0].raw_text == "emotion 1"  # chronological


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_search(store: InMemoryEpisodicStore):
    await store.append(_ep(raw_text="about cooking", embedding=[1.0, 0.0, 0.0]))
    await store.append(_ep(raw_text="about coding", embedding=[0.0, 1.0, 0.0]))
    await store.append(_ep(raw_text="about baking", embedding=[0.9, 0.1, 0.0]))
    await store.append(_ep(raw_text="no embedding"))

    results = await store.semantic_search(
        "user-1",
        query_embedding=[1.0, 0.0, 0.0],
        min_similarity=0.5,
    )
    assert len(results) >= 2
    assert results[0][0].raw_text == "about cooking"
    assert results[0][1] == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_semantic_search_min_similarity(store: InMemoryEpisodicStore):
    await store.append(_ep(raw_text="match", embedding=[1.0, 0.0, 0.0]))
    await store.append(_ep(raw_text="no match", embedding=[0.0, 1.0, 0.0]))

    results = await store.semantic_search(
        "user-1",
        query_embedding=[1.0, 0.0, 0.0],
        min_similarity=0.99,
    )
    assert len(results) == 1
    assert results[0][0].raw_text == "match"


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_compressed(store: InMemoryEpisodicStore):
    ep1 = _ep(raw_text="ep1")
    ep2 = _ep(raw_text="ep2")
    await store.append(ep1)
    await store.append(ep2)

    count = await store.mark_compressed([ep1.id, ep2.id], "summary-1")
    assert count == 2

    retrieved = await store.get(ep1.id)
    assert retrieved is not None
    assert retrieved.is_compressed is True
    assert retrieved.compressed_into_id == "summary-1"


@pytest.mark.asyncio
async def test_build_summary():
    now = datetime.now(timezone.utc)
    sarah = EntityRef(name="Sarah", entity_type="person")
    project = EntityRef(name="Q2 Deck", entity_type="project")

    episodes = [
        _ep(
            raw_text="happy about Sarah",
            intent="journal",
            entities=[sarah],
            emotion=_emo("joy", intensity=2.0),
            timestamp=now,
        ),
        _ep(
            raw_text="Sarah and Q2 deck task",
            intent="task",
            entities=[sarah, project],
            emotion=_emo("anticipation", intensity=1.0),
            behavioral_signal=BehavioralSignal.COMPLETED,
            timestamp=now + timedelta(hours=1),
        ),
    ]

    summary = build_summary(
        "user-1", episodes, "User journaled about Sarah and completed a Q2 task"
    )
    assert summary.user_id == "user-1"
    assert summary.episode_count == 2
    assert summary.period_start == now
    assert len(summary.source_episode_ids) == 2
    assert summary.entity_mentions["Sarah"] == 2
    assert summary.entity_mentions["Q2 Deck"] == 1
    assert summary.intent_distribution["journal"] == 1
    assert summary.intent_distribution["task"] == 1
    assert summary.behavioral_signals["completed"] == 1


@pytest.mark.asyncio
async def test_build_summary_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        build_summary("user-1", [], "empty")


@pytest.mark.asyncio
async def test_compress_episodes_end_to_end(store: InMemoryEpisodicStore):
    now = datetime.now(timezone.utc)
    episodes = []
    for i in range(5):
        ep = _ep(raw_text=f"episode {i}", timestamp=now + timedelta(hours=i))
        await store.append(ep)
        episodes.append(ep)

    summary = await compress_episodes(
        store, "user-1", episodes, "Five episodes covering the morning"
    )
    assert summary.episode_count == 5

    for ep in episodes:
        stored_ep = await store.get(ep.id)
        assert stored_ep is not None
        assert stored_ep.is_compressed is True
        assert stored_ep.compressed_into_id == summary.id

    summaries = await store.get_summaries("user-1")
    assert len(summaries) == 1
    assert summaries[0].id == summary.id


@pytest.mark.asyncio
async def test_summaries_filterable_by_time(store: InMemoryEpisodicStore):
    now = datetime.now(timezone.utc)

    s1 = EpisodeSummary(
        user_id="user-1",
        period_start=now - timedelta(days=7),
        period_end=now - timedelta(days=6),
        source_episode_ids=["a"],
        episode_count=1,
        summary_text="last week",
    )
    s2 = EpisodeSummary(
        user_id="user-1",
        period_start=now - timedelta(days=1),
        period_end=now,
        source_episode_ids=["b"],
        episode_count=1,
        summary_text="yesterday",
    )
    await store.store_summary(s1)
    await store.store_summary(s2)

    recent = await store.get_summaries("user-1", start=now - timedelta(days=2))
    assert len(recent) == 1
    assert recent[0].summary_text == "yesterday"


# ---------------------------------------------------------------------------
# Emotion model helpers
# ---------------------------------------------------------------------------


def test_emotion_snapshot_properties():
    negative = _emo(valence=-0.5)
    assert negative.is_negative is True

    positive = _emo(valence=0.5)
    assert positive.is_negative is False

    high_arousal = _emo(arousal=0.8)
    assert high_arousal.is_high_arousal is True

    low_arousal = _emo(arousal=0.3)
    assert low_arousal.is_high_arousal is False


# ---------------------------------------------------------------------------
# Behavioral signal filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_by_behavioral_signal(store: InMemoryEpisodicStore):
    await store.append(
        _ep(raw_text="completed task", behavioral_signal=BehavioralSignal.COMPLETED)
    )
    await store.append(
        _ep(raw_text="skipped task", behavioral_signal=BehavioralSignal.SKIPPED)
    )
    await store.append(_ep(raw_text="no signal"))

    results = await store.query(
        "user-1",
        behavioral_filter=BehavioralFilter(BehavioralSignal.COMPLETED),
    )
    assert len(results) == 1
    assert results[0].raw_text == "completed task"


# ---------------------------------------------------------------------------
# Session filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_by_session_filter(store: InMemoryEpisodicStore):
    await store.append(_ep(raw_text="sess A", context=_ctx(session_id="A")))
    await store.append(_ep(raw_text="sess B", context=_ctx(session_id="B")))

    results = await store.query(
        "user-1", session_filter=SessionFilter("A")
    )
    assert len(results) == 1
    assert results[0].raw_text == "sess A"


# ---------------------------------------------------------------------------
# User isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_isolation(store: InMemoryEpisodicStore):
    await store.append(_ep(user_id="user-1", raw_text="u1"))
    await store.append(_ep(user_id="user-2", raw_text="u2"))

    r1 = await store.query("user-1")
    r2 = await store.query("user-2")

    assert len(r1) == 1
    assert r1[0].raw_text == "u1"
    assert len(r2) == 1
    assert r2[0].raw_text == "u2"
