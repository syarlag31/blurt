"""Tests for the working memory tier."""

from __future__ import annotations

import asyncio
import time

import pytest

from blurt.memory.working import (
    EmotionLabel,
    EmotionState,
    IntentType,
    WorkingMemory,
    WorkingMemoryEntry,
)


# ---------------------------------------------------------------------------
# WorkingMemoryEntry unit tests
# ---------------------------------------------------------------------------


class TestWorkingMemoryEntry:
    def test_default_construction(self):
        entry = WorkingMemoryEntry()
        assert entry.content == ""
        assert entry.intent is None
        assert entry.ttl_seconds == 300.0
        assert entry.source == "voice"
        assert len(entry.id) == 36  # UUID format

    def test_expiry_calculation(self):
        now = time.time()
        entry = WorkingMemoryEntry(created_at=now, ttl_seconds=60.0)
        assert entry.expires_at == pytest.approx(now + 60.0, abs=0.01)
        assert not entry.is_expired
        assert entry.remaining_ttl == pytest.approx(60.0, abs=1.0)

    def test_expired_entry(self):
        entry = WorkingMemoryEntry(created_at=time.time() - 400, ttl_seconds=300.0)
        assert entry.is_expired
        assert entry.remaining_ttl == 0.0

    def test_to_dict_roundtrip(self):
        entry = WorkingMemoryEntry(
            content="Call Sarah",
            intent=IntentType.TASK,
            confidence=0.92,
            entities=[{"name": "Sarah", "type": "person"}],
            emotion=EmotionState(
                primary=EmotionLabel.JOY, intensity=1.5, valence=0.7, arousal=0.6
            ),
            source="text",
        )
        d = entry.to_dict()
        assert d["content"] == "Call Sarah"
        assert d["intent"] == "task"
        assert d["confidence"] == 0.92
        assert d["emotion"]["primary"] == "joy"
        assert d["emotion"]["intensity"] == 1.5
        assert d["source"] == "text"
        assert "id" in d
        assert "remaining_ttl" in d

    def test_to_dict_no_intent(self):
        entry = WorkingMemoryEntry(content="hmm")
        d = entry.to_dict()
        assert d["intent"] is None


# ---------------------------------------------------------------------------
# WorkingMemory core operations
# ---------------------------------------------------------------------------


class TestWorkingMemoryCore:
    @pytest.fixture
    def wm(self):
        return WorkingMemory(session_id="test-session", max_entries=10, default_ttl=300.0)

    @pytest.mark.asyncio
    async def test_add_and_retrieve(self, wm):
        entry = await wm.add("hello world", intent=IntentType.JOURNAL)
        assert entry.content == "hello world"
        assert entry.intent == IntentType.JOURNAL

        active = await wm.get_active_entries()
        assert len(active) == 1
        assert active[0].id == entry.id

    @pytest.mark.asyncio
    async def test_size(self, wm):
        assert await wm.size() == 0
        await wm.add("one")
        await wm.add("two")
        assert await wm.size() == 2

    @pytest.mark.asyncio
    async def test_get_entry_by_id(self, wm):
        entry = await wm.add("find me")
        found = await wm.get_entry(entry.id)
        assert found is not None
        assert found.content == "find me"

    @pytest.mark.asyncio
    async def test_get_entry_not_found(self, wm):
        result = await wm.get_entry("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove_entry(self, wm):
        entry = await wm.add("remove me")
        assert await wm.remove(entry.id)
        assert await wm.size() == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, wm):
        assert not await wm.remove("nope")

    @pytest.mark.asyncio
    async def test_clear(self, wm):
        await wm.add("a")
        await wm.add("b")
        cleared = await wm.clear()
        assert cleared == 2
        assert await wm.size() == 0

    @pytest.mark.asyncio
    async def test_clear_empty(self, wm):
        cleared = await wm.clear()
        assert cleared == 0


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------


class TestTTLExpiration:
    @pytest.mark.asyncio
    async def test_expired_entries_pruned_on_get(self):
        wm = WorkingMemory(default_ttl=0.1)  # 100ms TTL
        await wm.add("will expire")
        await asyncio.sleep(0.15)
        active = await wm.get_active_entries()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_mixed_expiry(self):
        wm = WorkingMemory(default_ttl=300.0)
        # Add one that expires immediately
        await wm.add("old", ttl_seconds=0.05)
        # Add one that lasts
        entry2 = await wm.add("new", ttl_seconds=300.0)
        await asyncio.sleep(0.1)
        active = await wm.get_active_entries()
        assert len(active) == 1
        assert active[0].id == entry2.id

    @pytest.mark.asyncio
    async def test_expired_entry_not_found_by_id(self):
        wm = WorkingMemory(default_ttl=0.05)
        entry = await wm.add("temp")
        await asyncio.sleep(0.1)
        assert await wm.get_entry(entry.id) is None

    @pytest.mark.asyncio
    async def test_custom_ttl_per_entry(self):
        wm = WorkingMemory(default_ttl=300.0)
        await wm.add("short", ttl_seconds=0.05)
        long_ = await wm.add("long", ttl_seconds=600.0)
        await asyncio.sleep(0.1)
        active = await wm.get_active_entries()
        assert len(active) == 1
        assert active[0].id == long_.id


# ---------------------------------------------------------------------------
# Capacity eviction
# ---------------------------------------------------------------------------


class TestCapacityEviction:
    @pytest.mark.asyncio
    async def test_oldest_evicted_at_capacity(self):
        wm = WorkingMemory(max_entries=3, default_ttl=300.0)
        e1 = await wm.add("first")
        e2 = await wm.add("second")
        e3 = await wm.add("third")
        e4 = await wm.add("fourth")  # should evict "first"

        active = await wm.get_active_entries()
        ids = [e.id for e in active]
        assert e1.id not in ids
        assert e2.id in ids
        assert e3.id in ids
        assert e4.id in ids
        assert len(active) == 3

    @pytest.mark.asyncio
    async def test_expired_pruned_before_eviction(self):
        wm = WorkingMemory(max_entries=2, default_ttl=300.0)
        await wm.add("will expire", ttl_seconds=0.05)
        await wm.add("stays", ttl_seconds=300.0)
        await asyncio.sleep(0.1)
        # Buffer is "full" (2 entries) but one expired
        # Adding should prune expired first, avoiding eviction of "stays"
        await wm.add("new")
        active = await wm.get_active_entries()
        assert len(active) == 2
        contents = [e.content for e in active]
        assert "stays" in contents
        assert "new" in contents


# ---------------------------------------------------------------------------
# Context aggregation
# ---------------------------------------------------------------------------


class TestContextAggregation:
    @pytest.mark.asyncio
    async def test_empty_context(self):
        wm = WorkingMemory(session_id="ctx-test")
        ctx = await wm.get_context()
        assert ctx.session_id == "ctx-test"
        assert ctx.entry_count == 0
        assert ctx.last_interaction_at is None
        assert ctx.recent_intents == []
        assert ctx.recent_entities == []

    @pytest.mark.asyncio
    async def test_mood_from_latest_entry(self):
        wm = WorkingMemory()
        await wm.add(
            "I'm happy",
            emotion=EmotionState(primary=EmotionLabel.SADNESS, intensity=2.0),
        )
        await wm.add(
            "Actually great!",
            emotion=EmotionState(primary=EmotionLabel.JOY, intensity=2.5, valence=0.8),
        )
        ctx = await wm.get_context()
        assert ctx.current_mood.primary == EmotionLabel.JOY
        assert ctx.current_mood.intensity == 2.5

    @pytest.mark.asyncio
    async def test_recent_intents_deduplicated(self):
        wm = WorkingMemory()
        await wm.add("t1", intent=IntentType.TASK)
        await wm.add("e1", intent=IntentType.EVENT)
        await wm.add("t2", intent=IntentType.TASK)
        ctx = await wm.get_context()
        # Should have TASK and EVENT, not duplicated
        assert IntentType.TASK in ctx.recent_intents
        assert IntentType.EVENT in ctx.recent_intents
        assert len(ctx.recent_intents) == 2

    @pytest.mark.asyncio
    async def test_recent_entities_collected(self):
        wm = WorkingMemory()
        await wm.add(
            "Call Sarah",
            entities=[{"name": "Sarah", "type": "person"}],
        )
        await wm.add(
            "Meeting at Google",
            entities=[{"name": "Google", "type": "organization"}],
        )
        ctx = await wm.get_context()
        names = [e["name"] for e in ctx.recent_entities]
        assert "Sarah" in names
        assert "Google" in names

    @pytest.mark.asyncio
    async def test_entity_deduplication(self):
        wm = WorkingMemory()
        await wm.add("a", entities=[{"name": "Sarah", "type": "person"}])
        await wm.add("b", entities=[{"name": "Sarah", "type": "person"}])
        ctx = await wm.get_context()
        names = [e["name"] for e in ctx.recent_entities]
        assert names.count("Sarah") == 1

    @pytest.mark.asyncio
    async def test_energy_level_bounds(self):
        wm = WorkingMemory()
        await wm.add(
            "extreme",
            emotion=EmotionState(valence=1.0, arousal=1.0),
        )
        ctx = await wm.get_context()
        assert 0.0 <= ctx.energy_level <= 1.0

        wm2 = WorkingMemory()
        await wm2.add(
            "low",
            emotion=EmotionState(valence=-1.0, arousal=0.0),
        )
        ctx2 = await wm2.get_context()
        assert 0.0 <= ctx2.energy_level <= 1.0

    @pytest.mark.asyncio
    async def test_active_task_in_context(self):
        wm = WorkingMemory()
        await wm.add("working on something")
        await wm.set_active_task("task-123")
        ctx = await wm.get_context()
        assert ctx.active_task_id == "task-123"

    @pytest.mark.asyncio
    async def test_context_to_dict(self):
        wm = WorkingMemory()
        await wm.add("test", intent=IntentType.IDEA)
        ctx = await wm.get_context()
        d = ctx.to_dict()
        assert "session_id" in d
        assert "current_mood" in d
        assert "energy_level" in d
        assert d["entry_count"] == 1


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    @pytest.mark.asyncio
    async def test_get_recent_content(self):
        wm = WorkingMemory()
        for i in range(5):
            await wm.add(f"message {i}")
        recent = await wm.get_recent_content(limit=3)
        assert len(recent) == 3
        assert recent == ["message 2", "message 3", "message 4"]

    @pytest.mark.asyncio
    async def test_get_recent_content_fewer_than_limit(self):
        wm = WorkingMemory()
        await wm.add("only one")
        recent = await wm.get_recent_content(limit=10)
        assert recent == ["only one"]

    @pytest.mark.asyncio
    async def test_get_entries_by_intent(self):
        wm = WorkingMemory()
        await wm.add("task 1", intent=IntentType.TASK)
        await wm.add("an idea", intent=IntentType.IDEA)
        await wm.add("task 2", intent=IntentType.TASK)

        tasks = await wm.get_entries_by_intent(IntentType.TASK)
        assert len(tasks) == 2
        assert all(e.intent == IntentType.TASK for e in tasks)

        ideas = await wm.get_entries_by_intent(IntentType.IDEA)
        assert len(ideas) == 1


# ---------------------------------------------------------------------------
# Session ID and defaults
# ---------------------------------------------------------------------------


class TestSessionConfig:
    def test_auto_generated_session_id(self):
        wm = WorkingMemory()
        assert len(wm.session_id) == 36  # UUID

    def test_custom_session_id(self):
        wm = WorkingMemory(session_id="custom-123")
        assert wm.session_id == "custom-123"

    def test_default_settings(self):
        wm = WorkingMemory()
        assert wm.max_entries == 100
        assert wm.default_ttl == 300.0

    @pytest.mark.asyncio
    async def test_set_and_clear_active_task(self):
        wm = WorkingMemory()
        await wm.set_active_task("t1")
        ctx = await wm.get_context()
        # Need an entry for get_context not to short-circuit
        await wm.add("x")
        ctx = await wm.get_context()
        assert ctx.active_task_id == "t1"

        await wm.set_active_task(None)
        ctx = await wm.get_context()
        assert ctx.active_task_id is None

    @pytest.mark.asyncio
    async def test_clear_resets_active_task(self):
        wm = WorkingMemory()
        await wm.set_active_task("t1")
        await wm.add("x")
        await wm.clear()
        ctx = await wm.get_context()
        assert ctx.active_task_id is None


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_intent_values(self):
        assert len(IntentType) == 7
        expected = {"task", "event", "reminder", "idea", "journal", "update", "question"}
        assert {i.value for i in IntentType} == expected

    def test_emotion_values(self):
        assert len(EmotionLabel) == 8
        expected = {
            "joy", "trust", "fear", "surprise",
            "sadness", "disgust", "anger", "anticipation",
        }
        assert {e.value for e in EmotionLabel} == expected

    def test_intent_is_string(self):
        assert IntentType.TASK == "task"

    def test_emotion_is_string(self):
        assert EmotionLabel.JOY == "joy"
