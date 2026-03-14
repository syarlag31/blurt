"""Tests for intent-specific metadata models and BlurtClassificationOutput.

Validates the 7 intent metadata schemas, the create_intent_metadata factory,
the unified BlurtClassificationOutput model, and serialization round-trips.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from blurt.models.intents import (
    INTENT_METADATA_MODELS,
    SYNCABLE_INTENTS,
    BlurtClassificationOutput,
    BlurtIntent,
    EventMetadata,
    IdeaMetadata,
    JournalMetadata,
    QuestionMetadata,
    ReminderMetadata,
    TaskMetadata,
    UpdateMetadata,
    create_intent_metadata,
)


# ── Intent Metadata Model Coverage ──────────────────────────────────


class TestIntentMetadataModels:
    """Every intent must have a corresponding metadata model."""

    def test_all_seven_intents_have_metadata_models(self) -> None:
        """INTENT_METADATA_MODELS must cover all 7 intents."""
        for intent in BlurtIntent:
            assert intent in INTENT_METADATA_MODELS, (
                f"{intent} missing from INTENT_METADATA_MODELS"
            )

    def test_metadata_models_count(self) -> None:
        assert len(INTENT_METADATA_MODELS) == 7

    def test_each_model_has_intent_type_literal(self) -> None:
        """Each metadata model must have an intent_type field matching its intent."""
        for intent, model_cls in INTENT_METADATA_MODELS.items():
            instance = model_cls()
            assert instance.intent_type == intent.value

    def test_each_model_has_extra_field(self) -> None:
        """Each metadata model captures overflow in an 'extra' dict."""
        for intent, model_cls in INTENT_METADATA_MODELS.items():
            instance = model_cls()
            assert hasattr(instance, "extra")
            assert isinstance(instance.extra, dict)


# ── TaskMetadata ────────────────────────────────────────────────────


class TestTaskMetadata:
    def test_defaults(self) -> None:
        meta = TaskMetadata()
        assert meta.intent_type == "task"
        assert meta.action_summary == ""
        assert meta.deadline is None
        assert meta.deadline_flexibility == "flexible"
        assert meta.assignee is None
        assert meta.priority_hint is None
        assert meta.tags == []
        assert meta.sync_targets == []

    def test_full_population(self) -> None:
        deadline = datetime(2026, 3, 20, 17, 0, tzinfo=timezone.utc)
        meta = TaskMetadata(
            action_summary="Buy groceries",
            deadline=deadline,
            deadline_flexibility="soft",
            assignee="me",
            priority_hint="medium",
            project_ref="weekly-errands",
            depends_on=["pick-up-car"],
            sync_targets=["notion"],
            tags=["personal", "errands"],
        )
        assert meta.action_summary == "Buy groceries"
        assert meta.deadline == deadline
        assert meta.priority_hint == "medium"
        assert meta.sync_targets == ["notion"]

    def test_serialization_roundtrip(self) -> None:
        meta = TaskMetadata(action_summary="Test task", tags=["work"])
        data = meta.model_dump()
        restored = TaskMetadata.model_validate(data)
        assert restored.action_summary == "Test task"
        assert restored.tags == ["work"]
        assert restored.intent_type == "task"


# ── EventMetadata ───────────────────────────────────────────────────


class TestEventMetadata:
    def test_defaults(self) -> None:
        meta = EventMetadata()
        assert meta.intent_type == "event"
        assert meta.title == ""
        assert meta.start_time is None
        assert meta.all_day is False
        assert meta.attendees == []

    def test_full_event(self) -> None:
        start = datetime(2026, 3, 15, 19, 0, tzinfo=timezone.utc)
        meta = EventMetadata(
            title="Dinner with Sarah",
            start_time=start,
            duration_minutes=90,
            location="Italian place",
            attendees=["Sarah"],
        )
        assert meta.title == "Dinner with Sarah"
        assert meta.duration_minutes == 90
        assert meta.attendees == ["Sarah"]


# ── ReminderMetadata ────────────────────────────────────────────────


class TestReminderMetadata:
    def test_defaults(self) -> None:
        meta = ReminderMetadata()
        assert meta.intent_type == "reminder"
        assert meta.recurring is False
        assert meta.trigger_time is None

    def test_relative_trigger(self) -> None:
        meta = ReminderMetadata(
            content="Take meds",
            trigger_relative="in 2 hours",
        )
        assert meta.content == "Take meds"
        assert meta.trigger_relative == "in 2 hours"


# ── IdeaMetadata ────────────────────────────────────────────────────


class TestIdeaMetadata:
    def test_defaults(self) -> None:
        meta = IdeaMetadata()
        assert meta.intent_type == "idea"
        assert meta.domain is None
        assert meta.related_ideas == []

    def test_full_idea(self) -> None:
        meta = IdeaMetadata(
            summary="Use embeddings for cooking recipes",
            domain="technical",
            potential_connections=["recipe-project", "ml-experiments"],
            tags=["ai", "cooking"],
        )
        assert meta.summary == "Use embeddings for cooking recipes"
        assert len(meta.potential_connections) == 2


# ── JournalMetadata ─────────────────────────────────────────────────


class TestJournalMetadata:
    def test_defaults(self) -> None:
        meta = JournalMetadata()
        assert meta.intent_type == "journal"
        assert meta.energy_level is None
        assert meta.mood_keywords == []

    def test_journal_with_emotion(self) -> None:
        meta = JournalMetadata(
            summary="Tough day at work",
            mood_keywords=["stressed", "tired"],
            energy_level="low",
            people_mentioned=["boss"],
            time_context="today",
        )
        assert "stressed" in meta.mood_keywords
        assert meta.energy_level == "low"


# ── UpdateMetadata ──────────────────────────────────────────────────


class TestUpdateMetadata:
    def test_defaults(self) -> None:
        meta = UpdateMetadata()
        assert meta.intent_type == "update"
        assert meta.update_type == "modify"
        assert meta.sync_required is False

    def test_reschedule_update(self) -> None:
        meta = UpdateMetadata(
            update_type="reschedule",
            target_description="the meeting",
            previous_value="2pm",
            new_value="3pm",
            changes={"time": "3pm"},
            sync_required=True,
        )
        assert meta.update_type == "reschedule"
        assert meta.sync_required is True
        assert meta.changes["time"] == "3pm"


# ── QuestionMetadata ────────────────────────────────────────────────


class TestQuestionMetadata:
    def test_defaults(self) -> None:
        meta = QuestionMetadata()
        assert meta.intent_type == "question"
        assert meta.query_type == "recall"
        assert meta.entity_references == []

    def test_temporal_question(self) -> None:
        meta = QuestionMetadata(
            query_text="When is Sarah's birthday?",
            query_type="temporal",
            entity_references=["Sarah"],
            expected_answer_type="date",
        )
        assert meta.query_type == "temporal"
        assert meta.expected_answer_type == "date"


# ── create_intent_metadata factory ──────────────────────────────────


class TestCreateIntentMetadata:
    def test_creates_correct_type_for_each_intent(self) -> None:
        expected_types = {
            BlurtIntent.TASK: TaskMetadata,
            BlurtIntent.EVENT: EventMetadata,
            BlurtIntent.REMINDER: ReminderMetadata,
            BlurtIntent.IDEA: IdeaMetadata,
            BlurtIntent.JOURNAL: JournalMetadata,
            BlurtIntent.UPDATE: UpdateMetadata,
            BlurtIntent.QUESTION: QuestionMetadata,
        }
        for intent, expected_cls in expected_types.items():
            meta = create_intent_metadata(intent)
            assert isinstance(meta, expected_cls), (
                f"create_intent_metadata({intent}) returned {type(meta)}, "
                f"expected {expected_cls}"
            )

    def test_creates_with_raw_data(self) -> None:
        meta = create_intent_metadata(
            BlurtIntent.TASK,
            {"action_summary": "Buy milk", "tags": ["grocery"]},
        )
        assert isinstance(meta, TaskMetadata)
        assert meta.action_summary == "Buy milk"
        assert meta.tags == ["grocery"]

    def test_unknown_keys_go_to_extra(self) -> None:
        meta = create_intent_metadata(
            BlurtIntent.IDEA,
            {"summary": "Cool idea", "custom_field": "preserved"},
        )
        assert isinstance(meta, IdeaMetadata)
        assert meta.summary == "Cool idea"
        assert meta.extra["custom_field"] == "preserved"

    def test_none_raw_data_returns_defaults(self) -> None:
        meta = create_intent_metadata(BlurtIntent.JOURNAL, None)
        assert isinstance(meta, JournalMetadata)
        assert meta.summary == ""

    def test_empty_dict_returns_defaults(self) -> None:
        meta = create_intent_metadata(BlurtIntent.REMINDER, {})
        assert isinstance(meta, ReminderMetadata)
        assert meta.content == ""

    def test_intent_type_field_not_overridden(self) -> None:
        """Passing intent_type in raw_data shouldn't corrupt the model."""
        meta = create_intent_metadata(
            BlurtIntent.TASK,
            {"intent_type": "event", "action_summary": "Test"},
        )
        assert meta.intent_type == "task"  # Should stay task, not event


# ── BlurtClassificationOutput ───────────────────────────────────────


class TestBlurtClassificationOutput:
    def test_defaults(self) -> None:
        output = BlurtClassificationOutput()
        assert output.primary_intent == BlurtIntent.JOURNAL  # safe default
        assert output.confidence == 0.0
        assert output.secondary_intent is None
        assert output.intent_metadata is None
        assert output.entity_ids == []
        assert output.emotion_label == "neutral"
        assert output.input_source == "voice"
        assert output.is_multi_intent is False
        assert output.id  # should have a UUID

    def test_high_confidence_property(self) -> None:
        high = BlurtClassificationOutput(
            primary_intent=BlurtIntent.TASK, confidence=0.92,
        )
        assert high.is_high_confidence is True

        low = BlurtClassificationOutput(
            primary_intent=BlurtIntent.TASK, confidence=0.80,
        )
        assert low.is_high_confidence is False

        boundary = BlurtClassificationOutput(
            primary_intent=BlurtIntent.TASK, confidence=0.85,
        )
        assert boundary.is_high_confidence is True

    def test_is_syncable_property(self) -> None:
        for intent in BlurtIntent:
            output = BlurtClassificationOutput(primary_intent=intent)
            assert output.is_syncable == (intent in SYNCABLE_INTENTS)

    def test_downstream_actions_property(self) -> None:
        output = BlurtClassificationOutput(primary_intent=BlurtIntent.TASK)
        actions = output.downstream_actions
        assert "extract_entities" in actions
        assert "sync_task_tool" in actions

    def test_full_output_with_metadata(self) -> None:
        task_meta = TaskMetadata(
            action_summary="Buy groceries",
            priority_hint="low",
        )
        output = BlurtClassificationOutput(
            blurt_id="blurt-123",
            user_id="user-456",
            primary_intent=BlurtIntent.TASK,
            confidence=0.93,
            all_intent_scores={
                "task": 0.93, "event": 0.02, "reminder": 0.02,
                "idea": 0.01, "journal": 0.01, "update": 0.005, "question": 0.005,
            },
            intent_metadata=task_meta,
            raw_text="I need to buy groceries",
            entity_names=["groceries"],
            model_used="flash-lite",
            pipeline_latency_ms=45.2,
        )
        assert output.primary_intent == BlurtIntent.TASK
        assert output.confidence == 0.93
        assert isinstance(output.intent_metadata, TaskMetadata)
        assert output.intent_metadata.action_summary == "Buy groceries"
        assert output.is_high_confidence is True
        assert output.is_syncable is True

    def test_serialization_roundtrip(self) -> None:
        """Full model should serialize to dict and back."""
        meta = EventMetadata(
            title="Team standup",
            start_time=datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc),
            attendees=["Alice", "Bob"],
        )
        output = BlurtClassificationOutput(
            primary_intent=BlurtIntent.EVENT,
            confidence=0.91,
            intent_metadata=meta,
            raw_text="Team standup tomorrow at 9am",
            entity_names=["Alice", "Bob"],
            emotion_valence=0.2,
        )
        data = output.model_dump()
        restored = BlurtClassificationOutput.model_validate(data)
        assert restored.primary_intent == BlurtIntent.EVENT
        assert restored.confidence == 0.91
        assert restored.entity_names == ["Alice", "Bob"]
        assert restored.raw_text == "Team standup tomorrow at 9am"

    def test_json_serialization(self) -> None:
        output = BlurtClassificationOutput(
            primary_intent=BlurtIntent.QUESTION,
            confidence=0.88,
            raw_text="When is Sarah's birthday?",
            intent_metadata=QuestionMetadata(
                query_text="When is Sarah's birthday?",
                query_type="temporal",
            ),
        )
        json_str = output.model_dump_json()
        assert '"question"' in json_str
        assert "0.88" in json_str

    def test_confidence_validation(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            BlurtClassificationOutput(confidence=1.5)
        with pytest.raises(Exception):
            BlurtClassificationOutput(confidence=-0.1)

    def test_multi_intent_output(self) -> None:
        output = BlurtClassificationOutput(
            primary_intent=BlurtIntent.TASK,
            confidence=0.75,
            is_multi_intent=True,
            sub_intents=[
                {"intent": "task", "confidence": 0.75, "segment": "buy groceries"},
                {"intent": "reminder", "confidence": 0.70, "segment": "remind me tomorrow"},
            ],
            was_escalated=True,
            model_used="flash",
        )
        assert output.is_multi_intent is True
        assert len(output.sub_intents) == 2
        assert output.was_escalated is True

    def test_created_at_auto_set(self) -> None:
        output = BlurtClassificationOutput()
        assert output.created_at is not None
        assert output.created_at.tzinfo is not None

    def test_emotion_fields(self) -> None:
        output = BlurtClassificationOutput(
            primary_intent=BlurtIntent.JOURNAL,
            confidence=0.95,
            emotion_valence=-0.6,
            emotion_arousal=0.8,
            emotion_label="sadness",
        )
        assert output.emotion_valence == -0.6
        assert output.emotion_arousal == 0.8
        assert output.emotion_label == "sadness"
