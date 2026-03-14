"""Tests for encryption integration across all data persistence layers.

Verifies that all data is encrypted before write and decrypted after read
for: episodic memory, knowledge graph (semantic memory), and sync state.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from blurt.core.encryption import DataEncryptor
from blurt.core.encrypted_storage import (
    EncryptedEpisodicStore,
    EncryptedSemanticMemoryStore,
    EncryptedSyncStateStore,
    _dict_to_episode,
    _dict_to_summary,
    _episode_to_dict,
    _summary_to_dict,
)
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
    InputModality,
    IntentFilter,
    SessionFilter,
    TimeRangeFilter,
)
from blurt.models.entities import (
    EntityNode,
    EntityType,
    Fact,
    FactType,
    LearnedPattern,
    PatternType,
    RelationshipEdge,
    RelationshipType,
)
from blurt.models.sync import (
    ConflictRecord,
    ConflictResolutionStrategy,
    SyncDirection,
    SyncOperation,
    SyncProvider,
    SyncRecord,
    SyncStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emo(
    primary: str = "joy",
    intensity: float = 1.0,
    valence: float = 0.5,
    arousal: float = 0.5,
) -> EmotionSnapshot:
    return EmotionSnapshot(
        primary=primary, intensity=intensity, valence=valence, arousal=arousal
    )


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
def encryptor() -> DataEncryptor:
    return DataEncryptor()


@pytest.fixture
def enc_store(encryptor: DataEncryptor) -> EncryptedEpisodicStore:
    return EncryptedEpisodicStore(encryptor)


@pytest.fixture
def enc_semantic(encryptor: DataEncryptor) -> EncryptedSemanticMemoryStore:
    return EncryptedSemanticMemoryStore("user-1", encryptor)


@pytest.fixture
def enc_sync(encryptor: DataEncryptor) -> EncryptedSyncStateStore:
    return EncryptedSyncStateStore(encryptor)


# ===========================================================================
# Episodic Memory Encryption Tests
# ===========================================================================


class TestEncryptedEpisodicStore:
    """Verify episodic memory is encrypted at rest and decrypted on read."""

    @pytest.mark.asyncio
    async def test_append_and_get_round_trip(self, enc_store: EncryptedEpisodicStore):
        """Episodes survive encrypt/decrypt round-trip with all fields intact."""
        episode = _ep(
            raw_text="Call Sarah about Q2 deck",
            entities=[EntityRef(name="Sarah", entity_type="person")],
            emotion=_emo("anticipation", 2.0, 0.6, 0.7),
        )
        stored = await enc_store.append(episode)
        assert stored.id == episode.id

        retrieved = await enc_store.get(stored.id)
        assert retrieved is not None
        assert retrieved.id == episode.id
        assert retrieved.raw_text == "Call Sarah about Q2 deck"
        assert retrieved.intent == "task"
        assert retrieved.intent_confidence == 0.95
        assert retrieved.emotion.primary == "anticipation"
        assert retrieved.emotion.intensity == 2.0
        assert len(retrieved.entities) == 1
        assert retrieved.entities[0].name == "Sarah"
        assert retrieved.context.session_id == "session-1"

    @pytest.mark.asyncio
    async def test_data_is_actually_encrypted(self, enc_store: EncryptedEpisodicStore):
        """Raw stored bytes must not contain plaintext user data."""
        episode = _ep(raw_text="super secret personal diary entry 12345")
        await enc_store.append(episode)

        # Access raw encrypted bytes
        encrypted_bytes = enc_store._encrypted_episodes[episode.id]
        assert b"super secret personal diary entry" not in encrypted_bytes
        assert b"12345" not in encrypted_bytes
        assert b"raw_text" not in encrypted_bytes

    @pytest.mark.asyncio
    async def test_wrong_key_cannot_decrypt(self, encryptor: DataEncryptor):
        """Data encrypted with one key cannot be decrypted with another."""
        store1 = EncryptedEpisodicStore(encryptor)
        store2 = EncryptedEpisodicStore(DataEncryptor())  # different key

        episode = _ep(raw_text="sensitive data")
        await store1.append(episode)

        # Copy encrypted data to store2
        store2._encrypted_episodes = dict(store1._encrypted_episodes)
        store2._user_ids = dict(store1._user_ids)

        with pytest.raises(Exception):  # DecryptionError
            await store2.get(episode.id)

    @pytest.mark.asyncio
    async def test_append_only_semantics(self, enc_store: EncryptedEpisodicStore):
        """Duplicate appends are rejected."""
        episode = _ep()
        await enc_store.append(episode)
        with pytest.raises(ValueError, match="already exists"):
            await enc_store.append(episode)

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, enc_store: EncryptedEpisodicStore):
        assert await enc_store.get("nope") is None

    @pytest.mark.asyncio
    async def test_count(self, enc_store: EncryptedEpisodicStore):
        assert await enc_store.count("user-1") == 0
        await enc_store.append(_ep(user_id="user-1", raw_text="one"))
        await enc_store.append(_ep(user_id="user-1", raw_text="two"))
        await enc_store.append(_ep(user_id="user-2", raw_text="three"))
        assert await enc_store.count("user-1") == 2
        assert await enc_store.count("user-2") == 1

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, enc_store: EncryptedEpisodicStore):
        now = datetime.now(timezone.utc)
        await enc_store.append(_ep(raw_text="old", timestamp=now - timedelta(days=7)))
        await enc_store.append(_ep(raw_text="recent", timestamp=now - timedelta(hours=1)))
        await enc_store.append(_ep(raw_text="now", timestamp=now))

        results = await enc_store.query(
            "user-1",
            time_range=TimeRangeFilter(start=now - timedelta(hours=2)),
        )
        assert len(results) == 2
        assert results[0].raw_text == "now"

    @pytest.mark.asyncio
    async def test_query_by_entity(self, enc_store: EncryptedEpisodicStore):
        sarah = EntityRef(name="Sarah", entity_type="person")
        jake = EntityRef(name="Jake", entity_type="person")

        await enc_store.append(_ep(raw_text="talk to Sarah", entities=[sarah]))
        await enc_store.append(_ep(raw_text="talk to Jake", entities=[jake]))
        await enc_store.append(_ep(raw_text="Sarah and Jake", entities=[sarah, jake]))

        results = await enc_store.query(
            "user-1", entity_filter=EntityFilter(entity_name="Sarah")
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_emotion(self, enc_store: EncryptedEpisodicStore):
        await enc_store.append(
            _ep(raw_text="happy", emotion=_emo("joy", intensity=2.0, valence=0.8))
        )
        await enc_store.append(
            _ep(raw_text="sad", emotion=_emo("sadness", intensity=2.0, valence=-0.7))
        )
        await enc_store.append(
            _ep(raw_text="slightly happy", emotion=_emo("joy", intensity=0.5, valence=0.3))
        )

        results = await enc_store.query(
            "user-1",
            emotion_filter=EmotionFilter(primary="joy", min_intensity=2.0),
        )
        assert len(results) == 1
        assert results[0].raw_text == "happy"

    @pytest.mark.asyncio
    async def test_query_by_intent(self, enc_store: EncryptedEpisodicStore):
        await enc_store.append(_ep(raw_text="task one", intent="task"))
        await enc_store.append(_ep(raw_text="idea one", intent="idea"))

        results = await enc_store.query("user-1", intent_filter=IntentFilter("idea"))
        assert len(results) == 1
        assert results[0].raw_text == "idea one"

    @pytest.mark.asyncio
    async def test_query_by_behavioral_signal(self, enc_store: EncryptedEpisodicStore):
        await enc_store.append(
            _ep(raw_text="done", behavioral_signal=BehavioralSignal.COMPLETED)
        )
        await enc_store.append(
            _ep(raw_text="skip", behavioral_signal=BehavioralSignal.SKIPPED)
        )

        results = await enc_store.query(
            "user-1",
            behavioral_filter=BehavioralFilter(BehavioralSignal.COMPLETED),
        )
        assert len(results) == 1
        assert results[0].raw_text == "done"

    @pytest.mark.asyncio
    async def test_query_by_session(self, enc_store: EncryptedEpisodicStore):
        await enc_store.append(_ep(raw_text="A", context=_ctx(session_id="A")))
        await enc_store.append(_ep(raw_text="B", context=_ctx(session_id="B")))

        results = await enc_store.query(
            "user-1", session_filter=SessionFilter("A")
        )
        assert len(results) == 1
        assert results[0].raw_text == "A"

    @pytest.mark.asyncio
    async def test_query_excludes_compressed(self, enc_store: EncryptedEpisodicStore):
        ep = _ep(raw_text="will be compressed")
        await enc_store.append(ep)
        await enc_store.mark_compressed([ep.id], "summary-1")

        results = await enc_store.query("user-1")
        assert len(results) == 0

        results_with = await enc_store.query("user-1", include_compressed=True)
        assert len(results_with) == 1

    @pytest.mark.asyncio
    async def test_query_pagination(self, enc_store: EncryptedEpisodicStore):
        now = datetime.now(timezone.utc)
        for i in range(10):
            await enc_store.append(
                _ep(raw_text=f"ep {i}", timestamp=now + timedelta(seconds=i))
            )

        page1 = await enc_store.query("user-1", limit=3, offset=0)
        page2 = await enc_store.query("user-1", limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].raw_text != page2[0].raw_text

    @pytest.mark.asyncio
    async def test_session_episodes(self, enc_store: EncryptedEpisodicStore):
        now = datetime.now(timezone.utc)
        for i in range(3):
            await enc_store.append(
                _ep(
                    raw_text=f"msg {i}",
                    context=_ctx(session_id="sess-A"),
                    timestamp=now + timedelta(seconds=i),
                )
            )
        eps = await enc_store.get_session_episodes("sess-A")
        assert len(eps) == 3
        assert eps[0].raw_text == "msg 0"

    @pytest.mark.asyncio
    async def test_semantic_search(self, enc_store: EncryptedEpisodicStore):
        await enc_store.append(_ep(raw_text="cooking", embedding=[1.0, 0.0, 0.0]))
        await enc_store.append(_ep(raw_text="coding", embedding=[0.0, 1.0, 0.0]))
        await enc_store.append(_ep(raw_text="baking", embedding=[0.9, 0.1, 0.0]))

        results = await enc_store.semantic_search(
            "user-1",
            query_embedding=[1.0, 0.0, 0.0],
            min_similarity=0.5,
        )
        assert len(results) >= 2
        assert results[0][0].raw_text == "cooking"

    @pytest.mark.asyncio
    async def test_entity_timeline(self, enc_store: EncryptedEpisodicStore):
        now = datetime.now(timezone.utc)
        sarah = EntityRef(name="Sarah", entity_type="person")
        for i in range(5):
            await enc_store.append(
                _ep(
                    raw_text=f"Sarah {i}",
                    entities=[sarah],
                    timestamp=now + timedelta(hours=i),
                )
            )
        timeline = await enc_store.get_entity_timeline("user-1", "Sarah", limit=3)
        assert len(timeline) == 3
        assert timeline[0].raw_text == "Sarah 4"

    @pytest.mark.asyncio
    async def test_emotion_timeline(self, enc_store: EncryptedEpisodicStore):
        now = datetime.now(timezone.utc)
        for i in range(5):
            await enc_store.append(
                _ep(
                    raw_text=f"emo {i}",
                    emotion=_emo(primary="joy"),
                    timestamp=now + timedelta(hours=i),
                )
            )
        timeline = await enc_store.get_emotion_timeline(
            "user-1",
            start=now + timedelta(hours=1),
            end=now + timedelta(hours=3),
        )
        assert len(timeline) == 3

    @pytest.mark.asyncio
    async def test_mark_compressed_re_encrypts(self, enc_store: EncryptedEpisodicStore):
        """Mark compressed must decrypt, update, and re-encrypt."""
        ep = _ep(raw_text="to compress")
        await enc_store.append(ep)

        original_bytes = enc_store._encrypted_episodes[ep.id]
        count = await enc_store.mark_compressed([ep.id], "summary-1")
        assert count == 1

        new_bytes = enc_store._encrypted_episodes[ep.id]
        # Re-encrypted, so bytes differ
        assert new_bytes != original_bytes

        retrieved = await enc_store.get(ep.id)
        assert retrieved is not None
        assert retrieved.is_compressed is True
        assert retrieved.compressed_into_id == "summary-1"

    @pytest.mark.asyncio
    async def test_summary_encryption(self, enc_store: EncryptedEpisodicStore):
        """Summaries are encrypted at rest and decrypted on read."""
        now = datetime.now(timezone.utc)
        summary = EpisodeSummary(
            user_id="user-1",
            period_start=now - timedelta(days=1),
            period_end=now,
            source_episode_ids=["ep-1", "ep-2"],
            episode_count=2,
            summary_text="User worked on Q2 deck with Sarah",
        )
        await enc_store.store_summary(summary)

        # Verify encrypted
        encrypted_bytes = enc_store._encrypted_summaries[summary.id]
        assert b"Q2 deck" not in encrypted_bytes
        assert b"Sarah" not in encrypted_bytes

        # Verify decryption
        summaries = await enc_store.get_summaries("user-1")
        assert len(summaries) == 1
        assert summaries[0].summary_text == "User worked on Q2 deck with Sarah"
        assert summaries[0].episode_count == 2

    @pytest.mark.asyncio
    async def test_user_isolation(self, enc_store: EncryptedEpisodicStore):
        """Episodes are isolated by user_id."""
        await enc_store.append(_ep(user_id="user-1", raw_text="u1"))
        await enc_store.append(_ep(user_id="user-2", raw_text="u2"))

        r1 = await enc_store.query("user-1")
        r2 = await enc_store.query("user-2")
        assert len(r1) == 1 and r1[0].raw_text == "u1"
        assert len(r2) == 1 and r2[0].raw_text == "u2"

    @pytest.mark.asyncio
    async def test_embeddings_preserved_in_cleartext(self, enc_store: EncryptedEpisodicStore):
        """Embeddings are stored in cleartext for vector search."""
        emb = [0.1, 0.2, 0.3, 0.4]
        ep = _ep(raw_text="embedded", embedding=emb)
        await enc_store.append(ep)

        # Embeddings accessible without decryption
        assert enc_store._embeddings[ep.id] == emb

        # But still returned after decryption
        retrieved = await enc_store.get(ep.id)
        assert retrieved is not None
        assert retrieved.embedding == emb


# ===========================================================================
# Semantic Memory (Knowledge Graph) Encryption Tests
# ===========================================================================


class TestEncryptedSemanticMemoryStore:
    """Verify knowledge graph data is encrypted at rest."""

    def test_entity_encrypt_decrypt_round_trip(self, enc_semantic: EncryptedSemanticMemoryStore):
        """Entity nodes survive encryption round-trip."""
        entity = EntityNode(
            user_id="user-1",
            name="Sarah",
            normalized_name="sarah",
            entity_type=EntityType.PERSON,
            aliases=["sar"],
            attributes={"role": "manager"},
            mention_count=5,
            embedding=[0.1, 0.2, 0.3],
        )

        enc_semantic.store_entity(entity)
        loaded = enc_semantic.load_entity(entity.id)

        assert loaded is not None
        assert loaded.id == entity.id
        assert loaded.name == "Sarah"
        assert loaded.normalized_name == "sarah"
        assert loaded.entity_type == EntityType.PERSON
        assert loaded.aliases == ["sar"]
        assert loaded.attributes == {"role": "manager"}
        assert loaded.mention_count == 5
        assert loaded.embedding == [0.1, 0.2, 0.3]

    def test_entity_data_encrypted_at_rest(self, enc_semantic: EncryptedSemanticMemoryStore):
        """Raw encrypted bytes must not contain entity names."""
        entity = EntityNode(
            user_id="user-1",
            name="Sarah Johnson",
            entity_type=EntityType.PERSON,
        )
        enc_semantic.store_entity(entity)

        encrypted = enc_semantic._encrypted_entities[entity.id]
        assert b"Sarah Johnson" not in encrypted
        assert b"person" not in encrypted

    def test_entity_lookup_by_name(self, enc_semantic: EncryptedSemanticMemoryStore):
        entity = EntityNode(
            user_id="user-1",
            name="Project Alpha",
            entity_type=EntityType.PROJECT,
            aliases=["alpha"],
        )
        enc_semantic.store_entity(entity)

        assert enc_semantic.get_entity_id_by_name("project alpha") == entity.id
        assert enc_semantic.get_entity_id_by_name("alpha") == entity.id
        assert enc_semantic.get_entity_id_by_name("beta") is None

    def test_entity_filter_by_type(self, enc_semantic: EncryptedSemanticMemoryStore):
        enc_semantic.store_entity(EntityNode(
            user_id="user-1", name="Sarah", entity_type=EntityType.PERSON
        ))
        enc_semantic.store_entity(EntityNode(
            user_id="user-1", name="Office", entity_type=EntityType.PLACE
        ))

        people = enc_semantic.get_all_entity_ids(EntityType.PERSON)
        places = enc_semantic.get_all_entity_ids(EntityType.PLACE)
        all_entities = enc_semantic.get_all_entity_ids()

        assert len(people) == 1
        assert len(places) == 1
        assert len(all_entities) == 2

    def test_relationship_encrypt_decrypt_round_trip(self, enc_semantic: EncryptedSemanticMemoryStore):
        """Relationships survive encryption round-trip."""
        rel = RelationshipEdge(
            user_id="user-1",
            source_entity_id="ent-1",
            target_entity_id="ent-2",
            relationship_type=RelationshipType.WORKS_WITH,
            strength=15.0,
            co_mention_count=3,
            context_snippets=["Sarah and Jake on Q2 deck"],
        )
        enc_semantic.store_relationship(rel)
        loaded = enc_semantic.load_relationship(rel.id)

        assert loaded is not None
        assert loaded.source_entity_id == "ent-1"
        assert loaded.target_entity_id == "ent-2"
        assert loaded.relationship_type == RelationshipType.WORKS_WITH
        assert loaded.strength == 15.0
        assert loaded.co_mention_count == 3
        assert loaded.context_snippets == ["Sarah and Jake on Q2 deck"]

    def test_relationship_data_encrypted(self, enc_semantic: EncryptedSemanticMemoryStore):
        rel = RelationshipEdge(
            user_id="user-1",
            source_entity_id="ent-1",
            target_entity_id="ent-2",
            relationship_type=RelationshipType.MENTIONED_WITH,
            context_snippets=["secret project meeting notes"],
        )
        enc_semantic.store_relationship(rel)

        encrypted = enc_semantic._encrypted_relationships[rel.id]
        assert b"secret project meeting" not in encrypted

    def test_relationship_indexed_by_entity(self, enc_semantic: EncryptedSemanticMemoryStore):
        rel = RelationshipEdge(
            user_id="user-1",
            source_entity_id="ent-1",
            target_entity_id="ent-2",
            relationship_type=RelationshipType.WORKS_WITH,
        )
        enc_semantic.store_relationship(rel)

        assert rel.id in enc_semantic.get_relationship_ids_for_entity("ent-1")
        assert rel.id in enc_semantic.get_relationship_ids_for_entity("ent-2")

    def test_fact_encrypt_decrypt_round_trip(self, enc_semantic: EncryptedSemanticMemoryStore):
        """Facts survive encryption round-trip."""
        fact = Fact(
            user_id="user-1",
            fact_type=FactType.PREFERENCE,
            subject_entity_id="ent-1",
            content="I prefer morning meetings",
            confidence=0.9,
            source_blurt_ids=["blurt-1"],
            embedding=[0.5, 0.6, 0.7],
        )
        enc_semantic.store_fact(fact)
        loaded = enc_semantic.load_fact(fact.id)

        assert loaded is not None
        assert loaded.content == "I prefer morning meetings"
        assert loaded.fact_type == FactType.PREFERENCE
        assert loaded.confidence == 0.9
        assert loaded.embedding == [0.5, 0.6, 0.7]

    def test_fact_data_encrypted(self, enc_semantic: EncryptedSemanticMemoryStore):
        fact = Fact(
            user_id="user-1",
            fact_type=FactType.ATTRIBUTE,
            content="Sarah is my direct manager at Acme Corp",
        )
        enc_semantic.store_fact(fact)

        encrypted = enc_semantic._encrypted_facts[fact.id]
        assert b"Sarah" not in encrypted
        assert b"Acme Corp" not in encrypted
        assert b"manager" not in encrypted

    def test_fact_indexed_by_entity(self, enc_semantic: EncryptedSemanticMemoryStore):
        fact = Fact(
            user_id="user-1",
            fact_type=FactType.ATTRIBUTE,
            content="fact about entity",
            subject_entity_id="ent-1",
        )
        enc_semantic.store_fact(fact)

        assert fact.id in enc_semantic.get_fact_ids_for_entity("ent-1")

    def test_fact_active_index(self, enc_semantic: EncryptedSemanticMemoryStore):
        active = Fact(user_id="user-1", fact_type=FactType.ATTRIBUTE, content="active")
        inactive = Fact(
            user_id="user-1", fact_type=FactType.ATTRIBUTE,
            content="inactive", is_active=False
        )
        enc_semantic.store_fact(active)
        enc_semantic.store_fact(inactive)

        active_ids = enc_semantic.get_all_active_fact_ids()
        assert active.id in active_ids
        assert inactive.id not in active_ids

    def test_pattern_encrypt_decrypt_round_trip(self, enc_semantic: EncryptedSemanticMemoryStore):
        """Patterns survive encryption round-trip."""
        pattern = LearnedPattern(
            user_id="user-1",
            pattern_type=PatternType.TIME_OF_DAY,
            description="User is most productive in the morning",
            parameters={"peak_hour": 9, "productivity_score": 0.85},
            confidence=0.8,
            observation_count=10,
            embedding=[0.1, 0.2],
        )
        enc_semantic.store_pattern(pattern)
        loaded = enc_semantic.load_pattern(pattern.id)

        assert loaded is not None
        assert loaded.description == "User is most productive in the morning"
        assert loaded.pattern_type == PatternType.TIME_OF_DAY
        assert loaded.parameters == {"peak_hour": 9, "productivity_score": 0.85}
        assert loaded.confidence == 0.8
        assert loaded.embedding == [0.1, 0.2]

    def test_pattern_data_encrypted(self, enc_semantic: EncryptedSemanticMemoryStore):
        pattern = LearnedPattern(
            user_id="user-1",
            pattern_type=PatternType.MOOD_CYCLE,
            description="User tends to feel anxious on Sunday evenings",
        )
        enc_semantic.store_pattern(pattern)

        encrypted = enc_semantic._encrypted_patterns[pattern.id]
        assert b"anxious" not in encrypted
        assert b"Sunday" not in encrypted

    def test_pattern_active_filter(self, enc_semantic: EncryptedSemanticMemoryStore):
        active = LearnedPattern(
            user_id="user-1",
            pattern_type=PatternType.TIME_OF_DAY,
            description="active",
            is_active=True,
        )
        inactive = LearnedPattern(
            user_id="user-1",
            pattern_type=PatternType.MOOD_CYCLE,
            description="inactive",
            is_active=False,
        )
        enc_semantic.store_pattern(active)
        enc_semantic.store_pattern(inactive)

        active_ids = enc_semantic.get_active_pattern_ids()
        assert active.id in active_ids
        assert inactive.id not in active_ids

        time_patterns = enc_semantic.get_active_pattern_ids(PatternType.TIME_OF_DAY)
        assert active.id in time_patterns

    def test_graph_stats(self, enc_semantic: EncryptedSemanticMemoryStore):
        enc_semantic.store_entity(EntityNode(
            user_id="user-1", name="A", entity_type=EntityType.PERSON
        ))
        enc_semantic.store_entity(EntityNode(
            user_id="user-1", name="B", entity_type=EntityType.PROJECT
        ))
        enc_semantic.store_relationship(RelationshipEdge(
            user_id="user-1",
            source_entity_id="a", target_entity_id="b",
            relationship_type=RelationshipType.WORKS_WITH,
        ))
        enc_semantic.store_fact(Fact(
            user_id="user-1", fact_type=FactType.ATTRIBUTE, content="test"
        ))

        stats = enc_semantic.get_stats()
        assert stats["total_entities"] == 2
        assert stats["total_relationships"] == 1
        assert stats["total_facts"] == 1
        assert stats["entity_type_counts"]["person"] == 1
        assert stats["entity_type_counts"]["project"] == 1

    def test_embeddings_in_cleartext(self, enc_semantic: EncryptedSemanticMemoryStore):
        """Embeddings must be accessible without decryption for vector search."""
        entity = EntityNode(
            user_id="user-1", name="Test", entity_type=EntityType.TOPIC,
            embedding=[0.1, 0.2, 0.3],
        )
        enc_semantic.store_entity(entity)
        assert enc_semantic._entity_embeddings[entity.id] == [0.1, 0.2, 0.3]

        fact = Fact(
            user_id="user-1", fact_type=FactType.ATTRIBUTE,
            content="test", embedding=[0.4, 0.5],
        )
        enc_semantic.store_fact(fact)
        assert enc_semantic._fact_embeddings[fact.id] == [0.4, 0.5]

    def test_wrong_key_cannot_decrypt_entity(self):
        """Entity encrypted with one key cannot be decrypted with another."""
        enc1 = DataEncryptor()
        enc2 = DataEncryptor()
        store1 = EncryptedSemanticMemoryStore("user-1", enc1)
        store2 = EncryptedSemanticMemoryStore("user-1", enc2)

        entity = EntityNode(
            user_id="user-1", name="Secret", entity_type=EntityType.PERSON
        )
        store1.store_entity(entity)

        # Copy encrypted data to store2
        store2._encrypted_entities = dict(store1._encrypted_entities)

        with pytest.raises(Exception):
            store2.load_entity(entity.id)


# ===========================================================================
# Sync State Encryption Tests
# ===========================================================================


class TestEncryptedSyncStateStore:
    """Verify sync state is encrypted at rest."""

    def test_sync_record_round_trip(self, enc_sync: EncryptedSyncStateStore):
        record = SyncRecord(
            blurt_id="blurt-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            external_id="gcal-event-123",
            direction=SyncDirection.BIDIRECTIONAL,
            status=SyncStatus.PENDING,
        )
        enc_sync.upsert_sync_record(record)
        loaded = enc_sync.get_sync_record(record.id)

        assert loaded is not None
        assert loaded.blurt_id == "blurt-1"
        assert loaded.provider == SyncProvider.GOOGLE_CALENDAR
        assert loaded.external_id == "gcal-event-123"
        assert loaded.status == SyncStatus.PENDING

    def test_sync_record_data_encrypted(self, enc_sync: EncryptedSyncStateStore):
        record = SyncRecord(
            blurt_id="blurt-secret-id",
            provider=SyncProvider.NOTION,
            external_id="notion-page-xyz",
        )
        enc_sync.upsert_sync_record(record)

        encrypted = enc_sync._encrypted_records[record.id]
        assert b"blurt-secret-id" not in encrypted
        assert b"notion-page-xyz" not in encrypted

    def test_lookup_by_blurt_id(self, enc_sync: EncryptedSyncStateStore):
        record = SyncRecord(
            blurt_id="b1",
            provider=SyncProvider.GOOGLE_CALENDAR,
        )
        enc_sync.upsert_sync_record(record)

        found = enc_sync.get_sync_record_by_blurt_id("b1", SyncProvider.GOOGLE_CALENDAR)
        assert found is not None
        assert found.id == record.id

        not_found = enc_sync.get_sync_record_by_blurt_id("b1", SyncProvider.NOTION)
        assert not_found is None

    def test_lookup_by_external_id(self, enc_sync: EncryptedSyncStateStore):
        record = SyncRecord(
            blurt_id="b1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            external_id="ext-1",
        )
        enc_sync.upsert_sync_record(record)

        found = enc_sync.get_sync_record_by_external_id("ext-1", SyncProvider.GOOGLE_CALENDAR)
        assert found is not None
        assert found.id == record.id

    def test_mark_synced(self, enc_sync: EncryptedSyncStateStore):
        record = SyncRecord(
            blurt_id="b1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            status=SyncStatus.PENDING,
        )
        enc_sync.upsert_sync_record(record)

        synced = enc_sync.mark_synced(record.id, external_id="ext-1", external_version="v2")
        assert synced is not None
        assert synced.status == SyncStatus.COMPLETED
        assert synced.external_id == "ext-1"

        # Verify re-encryption
        loaded = enc_sync.get_sync_record(record.id)
        assert loaded is not None
        assert loaded.status == SyncStatus.COMPLETED

    def test_mark_failed(self, enc_sync: EncryptedSyncStateStore):
        record = SyncRecord(
            blurt_id="b1",
            provider=SyncProvider.GOOGLE_CALENDAR,
        )
        enc_sync.upsert_sync_record(record)

        failed = enc_sync.mark_failed(record.id, "API error")
        assert failed is not None
        assert failed.status == SyncStatus.FAILED
        assert failed.error_message == "API error"
        assert failed.retry_count == 1

    def test_mark_conflict(self, enc_sync: EncryptedSyncStateStore):
        record = SyncRecord(
            blurt_id="b1",
            provider=SyncProvider.GOOGLE_CALENDAR,
        )
        enc_sync.upsert_sync_record(record)

        conflicted = enc_sync.mark_conflict(record.id)
        assert conflicted is not None
        assert conflicted.status == SyncStatus.CONFLICT

    def test_operation_round_trip(self, enc_sync: EncryptedSyncStateStore):
        op = SyncOperation(
            sync_record_id="rec-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            direction=SyncDirection.OUTBOUND,
            operation_type="create",
            payload={"title": "Meeting with Sarah", "time": "2pm"},
        )
        enc_sync.add_operation(op)

        loaded = enc_sync.get_operation(op.id)
        assert loaded is not None
        assert loaded.operation_type == "create"
        assert loaded.payload["title"] == "Meeting with Sarah"

    def test_operation_data_encrypted(self, enc_sync: EncryptedSyncStateStore):
        op = SyncOperation(
            sync_record_id="rec-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            direction=SyncDirection.OUTBOUND,
            operation_type="create",
            payload={"title": "Secret meeting with CEO"},
        )
        enc_sync.add_operation(op)

        encrypted = enc_sync._encrypted_operations[op.id]
        assert b"Secret meeting" not in encrypted
        assert b"CEO" not in encrypted

    def test_complete_operation(self, enc_sync: EncryptedSyncStateStore):
        op = SyncOperation(
            sync_record_id="rec-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            direction=SyncDirection.OUTBOUND,
            operation_type="create",
        )
        enc_sync.add_operation(op)

        completed = enc_sync.complete_operation(op.id, result={"event_id": "evt-1"})
        assert completed is not None
        assert completed.status == SyncStatus.COMPLETED
        assert completed.result == {"event_id": "evt-1"}

    def test_fail_operation(self, enc_sync: EncryptedSyncStateStore):
        op = SyncOperation(
            sync_record_id="rec-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            direction=SyncDirection.OUTBOUND,
            operation_type="update",
        )
        enc_sync.add_operation(op)

        failed = enc_sync.fail_operation(op.id, "Network error")
        assert failed is not None
        assert failed.status == SyncStatus.FAILED
        assert failed.error_message == "Network error"

    def test_conflict_round_trip(self, enc_sync: EncryptedSyncStateStore):
        conflict = ConflictRecord(
            sync_record_id="rec-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            blurt_data={"title": "My version of the event"},
            external_data={"title": "Their version of the event"},
            resolution_strategy=ConflictResolutionStrategy.LATEST_WINS,
        )
        enc_sync.add_conflict(conflict)

        loaded = enc_sync.get_conflict(conflict.id)
        assert loaded is not None
        assert loaded.blurt_data["title"] == "My version of the event"
        assert loaded.external_data["title"] == "Their version of the event"
        assert not loaded.resolved

    def test_conflict_data_encrypted(self, enc_sync: EncryptedSyncStateStore):
        conflict = ConflictRecord(
            sync_record_id="rec-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            blurt_data={"secret": "blurt-side confidential data"},
            external_data={"secret": "external-side confidential data"},
        )
        enc_sync.add_conflict(conflict)

        encrypted = enc_sync._encrypted_conflicts[conflict.id]
        assert b"confidential" not in encrypted

    def test_resolve_conflict(self, enc_sync: EncryptedSyncStateStore):
        conflict = ConflictRecord(
            sync_record_id="rec-1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            blurt_data={"v": 1},
            external_data={"v": 2},
        )
        enc_sync.add_conflict(conflict)

        resolved = enc_sync.resolve_conflict(conflict.id, {"v": 2, "merged": True})
        assert resolved is not None
        assert resolved.resolved is True
        assert resolved.resolution_result == {"v": 2, "merged": True}

    def test_unresolved_conflicts(self, enc_sync: EncryptedSyncStateStore):
        c1 = ConflictRecord(
            sync_record_id="r1",
            provider=SyncProvider.GOOGLE_CALENDAR,
            blurt_data={}, external_data={},
        )
        c2 = ConflictRecord(
            sync_record_id="r2",
            provider=SyncProvider.NOTION,
            blurt_data={}, external_data={},
        )
        enc_sync.add_conflict(c1)
        enc_sync.add_conflict(c2)
        enc_sync.resolve_conflict(c1.id, {"done": True})

        unresolved = enc_sync.get_unresolved_conflicts()
        assert len(unresolved) == 1
        assert unresolved[0].id == c2.id

    def test_stats(self, enc_sync: EncryptedSyncStateStore):
        enc_sync.upsert_sync_record(SyncRecord(
            blurt_id="b1", provider=SyncProvider.GOOGLE_CALENDAR,
            status=SyncStatus.COMPLETED,
        ))
        enc_sync.upsert_sync_record(SyncRecord(
            blurt_id="b2", provider=SyncProvider.NOTION,
            status=SyncStatus.PENDING,
        ))
        enc_sync.add_conflict(ConflictRecord(
            sync_record_id="r1", provider=SyncProvider.GOOGLE_CALENDAR,
            blurt_data={}, external_data={},
        ))

        stats = enc_sync.stats()
        assert stats["total_records"] == 2
        assert stats["total_conflicts"] == 1
        assert stats["unresolved_conflicts"] == 1
        assert stats["status_counts"]["completed"] == 1
        assert stats["status_counts"]["pending"] == 1

    def test_wrong_key_cannot_decrypt(self):
        """Sync data encrypted with one key cannot be read with another."""
        store1 = EncryptedSyncStateStore(DataEncryptor())
        store2 = EncryptedSyncStateStore(DataEncryptor())

        record = SyncRecord(
            blurt_id="b1", provider=SyncProvider.GOOGLE_CALENDAR,
        )
        store1.upsert_sync_record(record)

        store2._encrypted_records = dict(store1._encrypted_records)

        with pytest.raises(Exception):
            store2.get_sync_record(record.id)


# ===========================================================================
# Serialization Round-Trip Tests
# ===========================================================================


class TestSerializationHelpers:
    """Test episode/summary serialization helpers."""

    def test_episode_serialization_round_trip(self):
        ep = _ep(
            raw_text="Test serialization",
            entities=[EntityRef(name="Sarah", entity_type="person", entity_id="e-1")],
            emotion=_emo("fear", 1.5, -0.3, 0.8),
            embedding=[0.1, 0.2],
        )
        d = _episode_to_dict(ep)
        restored = _dict_to_episode(d, ep.embedding)

        assert restored.id == ep.id
        assert restored.raw_text == "Test serialization"
        assert restored.emotion.primary == "fear"
        assert restored.emotion.intensity == 1.5
        assert restored.entities[0].name == "Sarah"
        assert restored.entities[0].entity_id == "e-1"
        assert restored.embedding == [0.1, 0.2]

    def test_summary_serialization_round_trip(self):
        now = datetime.now(timezone.utc)
        summary = EpisodeSummary(
            user_id="user-1",
            period_start=now - timedelta(days=1),
            period_end=now,
            source_episode_ids=["a", "b"],
            episode_count=2,
            summary_text="Summary text",
            dominant_emotions=[_emo("joy"), _emo("anticipation")],
            entity_mentions={"Sarah": 3, "Q2": 1},
            intent_distribution={"task": 2},
            behavioral_signals={"completed": 1},
            embedding=[0.5, 0.6],
        )
        d = _summary_to_dict(summary)
        restored = _dict_to_summary(d, summary.embedding)

        assert restored.user_id == "user-1"
        assert restored.episode_count == 2
        assert restored.summary_text == "Summary text"
        assert len(restored.dominant_emotions) == 2
        assert restored.entity_mentions["Sarah"] == 3
        assert restored.embedding == [0.5, 0.6]
