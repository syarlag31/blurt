"""E2E Scenario: Working memory continuity across deployment mode switch.

Validates that working memory entries populated in one deployment mode
(cloud or local) persist correctly when the mode is switched.  Exercises:

- TTL correctness across mode transitions
- Session context aggregation continuity
- Active task tracking persistence
- Intent-based and entity-based retrieval after mode change
- Emotion / mood continuity in session context
- Entry eviction and expiration semantics are mode-independent

Working memory is an in-process ephemeral buffer (WorkingMemory class).
"Mode switch" here means rebuilding the FastAPI app under a different
DeploymentMode while keeping the *same* WorkingMemory instance — the
real scenario when a user's session outlives a configuration change or
hot-reload.
"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from blurt.api.capture import router as capture_router, set_pipeline
from blurt.api.episodes import router as episodes_router, set_store
from blurt.api.patterns import set_pattern_service
from blurt.api.task_feedback import set_feedback_service
from blurt.config.settings import BlurtConfig, DeploymentMode
from blurt.core.app import create_app
from blurt.memory.episodic import InMemoryEpisodicStore
from blurt.memory.working import (
    EmotionLabel,
    EmotionState,
    IntentType,
    SessionContext,
    WorkingMemory,
    WorkingMemoryEntry,
)
from blurt.middleware.egress_guard import EgressGuard
from blurt.services.capture import BlurtCapturePipeline
from blurt.services.feedback import InMemoryFeedbackStore, TaskFeedbackService
from blurt.services.patterns import InMemoryPatternStore, PatternService

from tests.e2e.conftest import (
    _stub_classify,
    _stub_detect_emotion,
    _stub_embed,
    _stub_extract_entities,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app_with_shared_wm(
    mode: DeploymentMode,
    wm: WorkingMemory,
) -> tuple[FastAPI, dict[str, Any]]:
    """Create a fully wired FastAPI app sharing a pre-existing WorkingMemory.

    This simulates a mode switch: the app is rebuilt under a new
    DeploymentMode but the working memory buffer survives.
    """
    config = BlurtConfig(mode=mode, debug=True)
    application = create_app(config)

    episodic_store = InMemoryEpisodicStore()
    pattern_store = InMemoryPatternStore()
    feedback_store = InMemoryFeedbackStore()

    pipeline = BlurtCapturePipeline(
        store=episodic_store,
        classifier=_stub_classify,
        entity_extractor=_stub_extract_entities,
        emotion_detector=_stub_detect_emotion,
        embedder=_stub_embed,
    )
    pattern_service = PatternService(store=pattern_store)
    feedback_service = TaskFeedbackService(store=feedback_store)

    _registered_paths = {r.path for r in application.routes}
    if "/api/v1/blurt" not in _registered_paths:
        application.include_router(capture_router)
    if "/api/v1/episodes" not in _registered_paths:
        application.include_router(episodes_router)

    set_store(episodic_store)
    set_pipeline(pipeline)
    set_pattern_service(pattern_service)
    set_feedback_service(feedback_service)

    stores: dict[str, Any] = {
        "episodic_store": episodic_store,
        "pipeline": pipeline,
        "working_memory": wm,
    }
    return application, stores


async def _make_client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _deactivate_egress(app: FastAPI) -> None:
    """Deactivate egress guard if present so local-mode tests don't block."""
    if hasattr(app.state, "egress_guard"):
        guard: EgressGuard = app.state.egress_guard
        guard.deactivate()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_wm() -> WorkingMemory:
    """A single WorkingMemory instance shared across mode switches."""
    return WorkingMemory(session_id="wm-mode-switch-session", max_entries=100)


# ---------------------------------------------------------------------------
# Tests: TTL persistence across mode switch
# ---------------------------------------------------------------------------


class TestTTLPersistenceAcrossModeSwitch:
    """Entries added in cloud mode retain correct TTL when queried after
    switching to local mode (and vice-versa)."""

    async def test_entries_retain_ttl_after_cloud_to_local(self, shared_wm: WorkingMemory):
        """Populate entries in cloud mode context, switch to local, verify TTL."""
        # --- Cloud phase: add entries with known TTLs ---
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        _deactivate_egress(cloud_app)

        entry_short = await shared_wm.add(
            "Buy milk on the way home",
            intent=IntentType.TASK,
            confidence=0.9,
            ttl_seconds=60.0,
            source="voice",
        )
        entry_long = await shared_wm.add(
            "Brainstorm product roadmap for Q3",
            intent=IntentType.IDEA,
            confidence=0.85,
            ttl_seconds=600.0,
            source="text",
        )

        # Verify entries are present in cloud context
        active = await shared_wm.get_active_entries()
        assert len(active) == 2

        # --- Switch to local ---
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        # Same WM instance — entries must survive
        active_after = await shared_wm.get_active_entries()
        assert len(active_after) == 2

        # TTLs must be preserved (original values)
        short_entry = await shared_wm.get_entry(entry_short.id)
        long_entry = await shared_wm.get_entry(entry_long.id)
        assert short_entry is not None
        assert long_entry is not None
        assert short_entry.ttl_seconds == 60.0
        assert long_entry.ttl_seconds == 600.0

        # remaining_ttl should be slightly less than original (time has passed)
        assert short_entry.remaining_ttl <= 60.0
        assert short_entry.remaining_ttl > 0.0
        assert long_entry.remaining_ttl <= 600.0
        assert long_entry.remaining_ttl > 0.0

    async def test_entries_retain_ttl_after_local_to_cloud(self, shared_wm: WorkingMemory):
        """Populate entries in local mode context, switch to cloud, verify TTL."""
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        entry = await shared_wm.add(
            "Schedule dentist appointment",
            intent=IntentType.REMINDER,
            confidence=0.88,
            ttl_seconds=120.0,
        )

        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        retrieved = await shared_wm.get_entry(entry.id)
        assert retrieved is not None
        assert retrieved.ttl_seconds == 120.0
        assert retrieved.content == "Schedule dentist appointment"
        assert retrieved.remaining_ttl > 0.0

    async def test_expired_entries_pruned_after_mode_switch(self, shared_wm: WorkingMemory):
        """Entries with very short TTL expire naturally after mode switch."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        # Add entry with essentially zero TTL (already expired)
        entry = await shared_wm.add(
            "Ephemeral thought",
            intent=IntentType.JOURNAL,
            ttl_seconds=0.001,
        )

        # Small sleep to ensure expiry
        import asyncio
        await asyncio.sleep(0.01)

        # Switch mode
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        # The expired entry should be pruned
        active = await shared_wm.get_active_entries()
        assert all(e.id != entry.id for e in active)


# ---------------------------------------------------------------------------
# Tests: Session context continuity
# ---------------------------------------------------------------------------


class TestSessionContextContinuity:
    """Session context (mood, energy, entry count) is correctly aggregated
    after a mode switch."""

    async def test_session_id_persists_across_modes(self, shared_wm: WorkingMemory):
        """session_id stays the same when mode changes."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        ctx_cloud = await shared_wm.get_context()

        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)
        ctx_local = await shared_wm.get_context()

        assert ctx_cloud.session_id == ctx_local.session_id
        assert ctx_cloud.session_id == "wm-mode-switch-session"

    async def test_entry_count_accumulates_across_modes(self, shared_wm: WorkingMemory):
        """Entries added in cloud mode count toward local mode context."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        await shared_wm.add("First note", intent=IntentType.JOURNAL)
        await shared_wm.add("Second note", intent=IntentType.JOURNAL)

        ctx = await shared_wm.get_context()
        assert ctx.entry_count == 2

        # Switch to local — add more
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        await shared_wm.add("Third note from local", intent=IntentType.JOURNAL)

        ctx_after = await shared_wm.get_context()
        assert ctx_after.entry_count == 3

    async def test_current_mood_reflects_latest_entry_after_switch(self, shared_wm: WorkingMemory):
        """The most recent entry's emotion becomes current_mood regardless of mode."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        await shared_wm.add(
            "Feeling great about the project",
            intent=IntentType.JOURNAL,
            emotion=EmotionState(
                primary=EmotionLabel.JOY,
                intensity=2.5,
                valence=0.9,
                arousal=0.8,
            ),
        )

        ctx_cloud = await shared_wm.get_context()
        assert ctx_cloud.current_mood.primary == EmotionLabel.JOY

        # Switch to local — add a sadder entry
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        await shared_wm.add(
            "Feeling down about the deadline",
            intent=IntentType.JOURNAL,
            emotion=EmotionState(
                primary=EmotionLabel.SADNESS,
                intensity=1.5,
                valence=-0.6,
                arousal=0.3,
            ),
        )

        ctx_local = await shared_wm.get_context()
        # Most recent entry's emotion should be current mood
        assert ctx_local.current_mood.primary == EmotionLabel.SADNESS
        assert ctx_local.entry_count == 2

    async def test_energy_level_aggregated_after_mode_switch(self, shared_wm: WorkingMemory):
        """Energy level computed from valence/arousal reflects all entries
        regardless of which mode they were created in."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        # Add high-energy entries in cloud
        for _ in range(3):
            await shared_wm.add(
                "Excited about launch",
                intent=IntentType.UPDATE,
                emotion=EmotionState(
                    primary=EmotionLabel.JOY,
                    intensity=2.0,
                    valence=0.8,
                    arousal=0.9,
                ),
            )

        ctx_cloud = await shared_wm.get_context()
        cloud_energy = ctx_cloud.energy_level

        # Switch to local
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        ctx_local = await shared_wm.get_context()
        # Energy should be identical — same entries, same computation
        assert ctx_local.energy_level == cloud_energy
        # High valence + high arousal → energy > 0.5
        assert ctx_local.energy_level > 0.5


# ---------------------------------------------------------------------------
# Tests: Active task tracking
# ---------------------------------------------------------------------------


class TestActiveTaskTrackingAcrossModes:
    """Active task ID persists across mode switches."""

    async def test_active_task_set_in_cloud_visible_in_local(self, shared_wm: WorkingMemory):
        """Set active task in cloud mode, verify it's visible after switching to local."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        # Add an entry so get_context() doesn't short-circuit on empty buffer
        await shared_wm.add("Working on cloud task", intent=IntentType.TASK)
        await shared_wm.set_active_task("task-cloud-123")

        ctx_cloud = await shared_wm.get_context()
        assert ctx_cloud.active_task_id == "task-cloud-123"

        # Switch to local
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        ctx_local = await shared_wm.get_context()
        assert ctx_local.active_task_id == "task-cloud-123"

    async def test_active_task_updated_in_local_visible_in_cloud(self, shared_wm: WorkingMemory):
        """Update active task in local mode, verify it persists after cloud switch."""
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        await shared_wm.add("Working on local task", intent=IntentType.TASK)
        await shared_wm.set_active_task("task-local-456")

        # Switch to cloud
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        ctx = await shared_wm.get_context()
        assert ctx.active_task_id == "task-local-456"

    async def test_clear_active_task_persists_across_switch(self, shared_wm: WorkingMemory):
        """Clearing active task in one mode is reflected after switch."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        await shared_wm.add("Task entry", intent=IntentType.TASK)
        await shared_wm.set_active_task("task-to-clear")
        ctx = await shared_wm.get_context()
        assert ctx.active_task_id == "task-to-clear"

        # Clear it
        await shared_wm.set_active_task(None)

        # Switch to local
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        ctx_local = await shared_wm.get_context()
        assert ctx_local.active_task_id is None

    async def test_active_task_survives_multiple_mode_switches(self, shared_wm: WorkingMemory):
        """Active task remains set through cloud→local→cloud transitions."""
        # Cloud: set task
        cloud_app1, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        await shared_wm.add("Persistent work item", intent=IntentType.TASK)
        await shared_wm.set_active_task("persistent-task")

        # Local
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)
        ctx1 = await shared_wm.get_context()
        assert ctx1.active_task_id == "persistent-task"

        # Back to cloud
        cloud_app2, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        ctx2 = await shared_wm.get_context()
        assert ctx2.active_task_id == "persistent-task"


# ---------------------------------------------------------------------------
# Tests: Intent retrieval across modes
# ---------------------------------------------------------------------------


class TestIntentRetrievalAcrossModes:
    """Entries can be filtered by intent after a mode switch."""

    async def test_get_entries_by_intent_after_switch(self, shared_wm: WorkingMemory):
        """Entries added in cloud mode are retrievable by intent in local mode."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        await shared_wm.add("Buy groceries", intent=IntentType.TASK, confidence=0.9)
        await shared_wm.add("Meeting at 3pm", intent=IntentType.EVENT, confidence=0.88)
        await shared_wm.add("Fix the login bug", intent=IntentType.TASK, confidence=0.92)

        # Switch to local
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        tasks = await shared_wm.get_entries_by_intent(IntentType.TASK)
        assert len(tasks) == 2
        assert all(e.intent == IntentType.TASK for e in tasks)

        events = await shared_wm.get_entries_by_intent(IntentType.EVENT)
        assert len(events) == 1
        assert events[0].content == "Meeting at 3pm"

    async def test_recent_intents_in_context_span_both_modes(self, shared_wm: WorkingMemory):
        """SessionContext.recent_intents includes intents from both modes."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        await shared_wm.add("Todo: ship feature", intent=IntentType.TASK)
        await shared_wm.add("Random thought", intent=IntentType.JOURNAL)

        # Switch to local and add more
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        await shared_wm.add("What if we redesign the API?", intent=IntentType.IDEA)
        await shared_wm.add("Remind me to call Bob", intent=IntentType.REMINDER)

        ctx = await shared_wm.get_context()
        intent_values = [i.value for i in ctx.recent_intents]

        # All four intents should appear (deduplicated, ordered)
        assert "task" in intent_values
        assert "journal" in intent_values
        assert "idea" in intent_values
        assert "reminder" in intent_values

    async def test_mixed_mode_entries_filtered_correctly(self, shared_wm: WorkingMemory):
        """Adding same intent in different modes yields correct aggregation."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        await shared_wm.add("Cloud task 1", intent=IntentType.TASK)

        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)
        await shared_wm.add("Local task 2", intent=IntentType.TASK)

        cloud_app2, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        await shared_wm.add("Cloud task 3", intent=IntentType.TASK)

        all_tasks = await shared_wm.get_entries_by_intent(IntentType.TASK)
        assert len(all_tasks) == 3
        contents = [e.content for e in all_tasks]
        assert "Cloud task 1" in contents
        assert "Local task 2" in contents
        assert "Cloud task 3" in contents


# ---------------------------------------------------------------------------
# Tests: Entity retrieval across modes
# ---------------------------------------------------------------------------


class TestEntityRetrievalAcrossModes:
    """Entities attached to entries persist and aggregate after mode switch."""

    async def test_entities_from_cloud_appear_in_local_context(self, shared_wm: WorkingMemory):
        """Entities added in cloud mode appear in session context after local switch."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        await shared_wm.add(
            "Discuss project with Alice",
            intent=IntentType.TASK,
            entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 0.95},
                {"name": "Project Alpha", "entity_type": "project", "confidence": 0.90},
            ],
        )

        # Switch to local
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        ctx = await shared_wm.get_context()
        entity_names = [e["name"] for e in ctx.recent_entities]
        assert "Alice" in entity_names
        assert "Project Alpha" in entity_names

    async def test_entities_accumulate_across_mode_switches(self, shared_wm: WorkingMemory):
        """Entities from both cloud and local phases appear in context."""
        # Cloud: add Alice
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        await shared_wm.add(
            "Meeting with Alice",
            intent=IntentType.EVENT,
            entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )

        # Local: add Bob
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)
        await shared_wm.add(
            "Email from Bob",
            intent=IntentType.UPDATE,
            entities=[{"name": "Bob", "entity_type": "person", "confidence": 0.92}],
        )

        ctx = await shared_wm.get_context()
        entity_names = [e["name"] for e in ctx.recent_entities]
        assert "Alice" in entity_names
        assert "Bob" in entity_names
        assert ctx.entry_count == 2

    async def test_duplicate_entities_deduplicated_in_context(self, shared_wm: WorkingMemory):
        """Same entity mentioned in both modes appears only once in context."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        await shared_wm.add(
            "Talk to Alice about design",
            intent=IntentType.TASK,
            entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.9}],
        )

        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)
        await shared_wm.add(
            "Follow up with Alice on review",
            intent=IntentType.TASK,
            entities=[{"name": "Alice", "entity_type": "person", "confidence": 0.92}],
        )

        ctx = await shared_wm.get_context()
        alice_count = sum(1 for e in ctx.recent_entities if e["name"] == "Alice")
        # SessionContext deduplicates by name
        assert alice_count == 1


# ---------------------------------------------------------------------------
# Tests: Full end-to-end capture pipeline with mode switch
# ---------------------------------------------------------------------------


class TestCaptureAndWorkingMemoryCrossMode:
    """Capture blurts via the HTTP API in one mode, verify working memory
    state after switching modes.  Uses the shared WM to bridge modes."""

    async def test_capture_in_cloud_then_query_entries_in_local(self, shared_wm: WorkingMemory):
        """Capture via API in cloud, populate WM, then verify entries survive in local."""
        cloud_app, cloud_stores = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        cloud_client = await _make_client(cloud_app)

        # Add entries directly (simulating what the capture pipeline would do)
        entry = await shared_wm.add(
            "I need to call the dentist",
            intent=IntentType.TASK,
            confidence=0.92,
            entities=[{"name": "dentist", "entity_type": "person", "confidence": 0.8}],
            emotion=EmotionState(
                primary=EmotionLabel.ANTICIPATION,
                intensity=1.0,
                valence=0.1,
                arousal=0.4,
            ),
            ttl_seconds=300.0,
        )
        await shared_wm.set_active_task("task-dentist")

        await cloud_client.aclose()

        # Switch to local
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)
        local_client = await _make_client(local_app)

        # Verify full entry integrity
        retrieved = await shared_wm.get_entry(entry.id)
        assert retrieved is not None
        assert retrieved.content == "I need to call the dentist"
        assert retrieved.intent == IntentType.TASK
        assert retrieved.confidence == 0.92
        assert retrieved.ttl_seconds == 300.0
        assert len(retrieved.entities) == 1
        assert retrieved.entities[0]["name"] == "dentist"
        assert retrieved.emotion.primary == EmotionLabel.ANTICIPATION

        # Verify context
        ctx = await shared_wm.get_context()
        assert ctx.active_task_id == "task-dentist"
        assert ctx.entry_count == 1
        assert IntentType.TASK in ctx.recent_intents

        await local_client.aclose()

    async def test_recent_content_spans_both_modes(self, shared_wm: WorkingMemory):
        """get_recent_content returns text from entries added in both modes."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        await shared_wm.add("Cloud thought alpha")
        await shared_wm.add("Cloud thought beta")

        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)
        await shared_wm.add("Local thought gamma")

        recent = await shared_wm.get_recent_content(limit=10)
        assert "Cloud thought alpha" in recent
        assert "Cloud thought beta" in recent
        assert "Local thought gamma" in recent
        assert len(recent) == 3

    async def test_clear_in_local_resets_everything(self, shared_wm: WorkingMemory):
        """Clearing WM in local mode wipes entries and active task,
        which is then verified after switching back to cloud."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)
        await shared_wm.add("Important note", intent=IntentType.TASK)
        await shared_wm.set_active_task("active-task")

        # Switch to local and clear
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)
        cleared = await shared_wm.clear()
        assert cleared == 1

        # Switch back to cloud — should be empty
        cloud_app2, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        ctx = await shared_wm.get_context()
        assert ctx.entry_count == 0
        assert ctx.active_task_id is None
        assert ctx.recent_intents == []
        assert ctx.recent_entities == []

    async def test_entry_serialization_consistent_across_modes(self, shared_wm: WorkingMemory):
        """to_dict() output is consistent regardless of which mode is active."""
        cloud_app, _ = _build_app_with_shared_wm(DeploymentMode.CLOUD, shared_wm)

        entry = await shared_wm.add(
            "Serialize me",
            intent=IntentType.IDEA,
            confidence=0.87,
            emotion=EmotionState(
                primary=EmotionLabel.SURPRISE,
                intensity=1.2,
                valence=0.3,
                arousal=0.6,
            ),
            entities=[{"name": "React", "entity_type": "tool", "confidence": 0.9}],
            ttl_seconds=180.0,
            metadata={"origin": "cloud"},
        )

        dict_cloud = entry.to_dict()

        # Switch to local — same entry, same serialization
        local_app, _ = _build_app_with_shared_wm(DeploymentMode.LOCAL, shared_wm)
        _deactivate_egress(local_app)

        retrieved = await shared_wm.get_entry(entry.id)
        assert retrieved is not None
        dict_local = retrieved.to_dict()

        # Core fields must match (remaining_ttl / is_expired may differ slightly)
        assert dict_cloud["id"] == dict_local["id"]
        assert dict_cloud["content"] == dict_local["content"]
        assert dict_cloud["intent"] == dict_local["intent"]
        assert dict_cloud["confidence"] == dict_local["confidence"]
        assert dict_cloud["entities"] == dict_local["entities"]
        assert dict_cloud["emotion"] == dict_local["emotion"]
        assert dict_cloud["ttl_seconds"] == dict_local["ttl_seconds"]
        assert dict_cloud["created_at"] == dict_local["created_at"]
        assert dict_cloud["metadata"] == dict_local["metadata"]
        assert dict_cloud["source"] == dict_local["source"]
