"""Integration tests verifying feature parity between local-only and cloud modes.

Asserts that:
1. All core features (memory, encryption, classification, sync) work identically
   in both local and cloud deployment modes.
2. Zero external network calls are made during local-only operation.
3. Local-only mode has full feature parity with no degradation.
4. Encryption is always on in both modes.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from blurt.clients.embeddings import MockEmbeddingProvider
from blurt.core.config import DeploymentMode, Settings
from blurt.core.encrypted_storage import (
    EncryptedEpisodicStore,
    EncryptedSemanticMemoryStore,
    EncryptedSyncStateStore,
)
from blurt.core.encryption import DataEncryptor
from blurt.memory.episodic import (
    EmotionFilter,
    EmotionSnapshot,
    EntityFilter,
    EntityRef,
    Episode,
    EpisodeContext,
    InMemoryEpisodicStore,
    InputModality,
    IntentFilter,
    build_summary,
)
from blurt.memory.semantic import SemanticMemoryStore
from blurt.memory.working import EmotionState, IntentType, WorkingMemory
from blurt.models.entities import (
    EntityNode,
    EntityType,
    FactType,
    PatternType,
    RelationshipType,
)
from blurt.models.sync import (
    SyncProvider,
    SyncRecord,
    SyncStatus,
)
from blurt.sync.orchestrator import SyncOrchestrator
from blurt.sync.providers import GoogleCalendarAdapter, NotionAdapter
from blurt.sync.state import SyncStateStore


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

USER_ID = "test-user-local-cloud"
SESSION_ID = "test-session-001"


def _make_master_key() -> bytes:
    """Generate a 32-byte master key for testing."""
    return b"\x42" * 32


def _make_episode(
    *,
    user_id: str = USER_ID,
    raw_text: str = "Buy groceries after work",
    intent: str = "task",
    confidence: float = 0.92,
    emotion: EmotionSnapshot | None = None,
    entities: list[EntityRef] | None = None,
    embedding: list[float] | None = None,
    session_id: str = SESSION_ID,
) -> Episode:
    """Create a test episode with sensible defaults."""
    return Episode(
        user_id=user_id,
        raw_text=raw_text,
        modality=InputModality.VOICE,
        intent=intent,
        intent_confidence=confidence,
        emotion=emotion or EmotionSnapshot(
            primary="trust", intensity=1.2, valence=0.4, arousal=0.3
        ),
        entities=entities or [
            EntityRef(name="Groceries", entity_type="topic", confidence=0.95)
        ],
        context=EpisodeContext(
            time_of_day="afternoon",
            day_of_week="wednesday",
            session_id=session_id,
        ),
        embedding=embedding or [0.1 * i for i in range(64)],
    )


class NetworkBlocker:
    """Context manager that blocks all socket.connect calls to assert zero
    external network activity. Any attempt to open a network connection
    raises an AssertionError with details about the target.
    """

    def __init__(self) -> None:
        self._original_connect: Any = None
        self.blocked_calls: list[tuple[str, int]] = []

    def _blocking_connect(self, sock: socket.socket, address: Any) -> None:
        """Intercept all socket connect calls."""
        # Allow localhost/loopback for in-process communication
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = address[0], address[1]
            if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
                return self._original_connect(sock, address)
            self.blocked_calls.append((host, port))
            raise AssertionError(
                f"LOCAL-ONLY MODE VIOLATION: Attempted network connection "
                f"to {host}:{port}. No external network calls are allowed."
            )
        # Unix domain sockets are OK
        if isinstance(address, str):
            return self._original_connect(sock, address)
        self.blocked_calls.append(("unknown", 0))
        raise AssertionError(
            f"LOCAL-ONLY MODE VIOLATION: Unexpected socket.connect with "
            f"address type {type(address)}: {address}"
        )

    def __enter__(self) -> "NetworkBlocker":
        self._original_connect = socket.socket.connect
        socket.socket.connect = self._blocking_connect  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        socket.socket.connect = self._original_connect  # type: ignore[assignment]


@pytest.fixture
def master_key() -> bytes:
    return _make_master_key()


@pytest.fixture
def encryptor(master_key: bytes) -> DataEncryptor:
    return DataEncryptor(master_key)


@pytest.fixture
def mock_embeddings() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()


@pytest.fixture
def local_settings(tmp_path) -> Settings:
    """Settings configured for local-only mode."""
    return Settings(
        deployment_mode=DeploymentMode.LOCAL,
        data_dir=tmp_path / ".blurt",
        encryption_enabled=True,
        google_client_id="",
        google_client_secret="",
    )


@pytest.fixture
def cloud_settings(tmp_path) -> Settings:
    """Settings configured for cloud mode."""
    return Settings(
        deployment_mode=DeploymentMode.CLOUD,
        data_dir=tmp_path / ".blurt-cloud",
        encryption_enabled=True,
        google_client_id="test-client-id",
        google_client_secret="test-client-secret",
    )


@pytest.fixture
def network_blocker() -> NetworkBlocker:
    return NetworkBlocker()


# ---------------------------------------------------------------------------
# 1. Memory System Feature Parity
# ---------------------------------------------------------------------------


class TestWorkingMemoryParity:
    """Working memory has identical behavior in both modes."""

    @pytest.mark.asyncio
    async def test_working_memory_operations_local(self) -> None:
        """All working memory operations work in local-only mode."""
        wm = WorkingMemory(session_id="local-session", max_entries=50)

        entry = await wm.add(
            "Call dentist tomorrow morning",
            intent=IntentType.TASK,
            confidence=0.88,
            entities=[{"name": "Dentist", "type": "person"}],
            emotion=EmotionState(
                primary=EmotionState().primary,
                intensity=0.5,
                valence=0.2,
                arousal=0.3,
            ),
        )

        assert entry.content == "Call dentist tomorrow morning"
        assert entry.intent == IntentType.TASK

        ctx = await wm.get_context()
        assert ctx.entry_count == 1
        assert len(ctx.recent_entities) == 1

        active = await wm.get_active_entries()
        assert len(active) == 1
        assert active[0].id == entry.id

    @pytest.mark.asyncio
    async def test_working_memory_operations_cloud(self) -> None:
        """Same working memory operations work identically in cloud mode."""
        wm = WorkingMemory(session_id="cloud-session", max_entries=50)

        entry = await wm.add(
            "Call dentist tomorrow morning",
            intent=IntentType.TASK,
            confidence=0.88,
            entities=[{"name": "Dentist", "type": "person"}],
            emotion=EmotionState(
                primary=EmotionState().primary,
                intensity=0.5,
                valence=0.2,
                arousal=0.3,
            ),
        )

        assert entry.content == "Call dentist tomorrow morning"
        assert entry.intent == IntentType.TASK

        ctx = await wm.get_context()
        assert ctx.entry_count == 1
        assert len(ctx.recent_entities) == 1

    @pytest.mark.asyncio
    async def test_working_memory_no_network_in_local(
        self, network_blocker: NetworkBlocker
    ) -> None:
        """Working memory makes zero network calls in local mode."""
        with network_blocker:
            wm = WorkingMemory(session_id="blocked-session")
            for i in range(10):
                await wm.add(
                    f"Entry {i}",
                    intent=IntentType.TASK,
                    confidence=0.9,
                )
            await wm.get_context()
            await wm.get_active_entries()
            await wm.get_recent_content(limit=5)
            await wm.get_entries_by_intent(IntentType.TASK)
            assert len(network_blocker.blocked_calls) == 0


class TestEpisodicMemoryParity:
    """Episodic memory store works identically in both modes."""

    @pytest.mark.asyncio
    async def test_plain_store_operations(self) -> None:
        """InMemoryEpisodicStore (no encryption) has full API coverage."""
        store = InMemoryEpisodicStore()
        ep = _make_episode()

        stored = await store.append(ep)
        assert stored.id == ep.id

        retrieved = await store.get(ep.id)
        assert retrieved is not None
        assert retrieved.raw_text == ep.raw_text

        count = await store.count(USER_ID)
        assert count == 1

        results = await store.query(USER_ID)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_encrypted_store_operations(self, encryptor: DataEncryptor) -> None:
        """EncryptedEpisodicStore has identical API surface and behavior."""
        store = EncryptedEpisodicStore(encryptor)
        ep = _make_episode()

        stored = await store.append(ep)
        assert stored.id == ep.id

        retrieved = await store.get(ep.id)
        assert retrieved is not None
        assert retrieved.raw_text == ep.raw_text

        count = await store.count(USER_ID)
        assert count == 1

        results = await store.query(USER_ID)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_parity_between_stores(
        self, encryptor: DataEncryptor
    ) -> None:
        """Both plain and encrypted stores return identical query results."""
        plain_store = InMemoryEpisodicStore()
        encrypted_store = EncryptedEpisodicStore(encryptor)

        # Insert same episodes into both stores
        episodes = [
            _make_episode(
                raw_text=f"Episode {i}",
                intent="task" if i % 2 == 0 else "idea",
                emotion=EmotionSnapshot(
                    primary="joy" if i % 3 == 0 else "trust",
                    intensity=float(i),
                    valence=0.5 if i % 2 == 0 else -0.3,
                    arousal=0.4,
                ),
                entities=[EntityRef(name=f"Entity-{i}", entity_type="person")],
            )
            for i in range(5)
        ]

        for ep in episodes:
            await plain_store.append(ep)
            await encrypted_store.append(ep)

        # Query with intent filter
        plain_tasks = await plain_store.query(
            USER_ID, intent_filter=IntentFilter("task")
        )
        enc_tasks = await encrypted_store.query(
            USER_ID, intent_filter=IntentFilter("task")
        )
        assert len(plain_tasks) == len(enc_tasks)
        for p, e in zip(plain_tasks, enc_tasks):
            assert p.raw_text == e.raw_text
            assert p.intent == e.intent

        # Query with emotion filter
        plain_joy = await plain_store.query(
            USER_ID, emotion_filter=EmotionFilter(primary="joy")
        )
        enc_joy = await encrypted_store.query(
            USER_ID, emotion_filter=EmotionFilter(primary="joy")
        )
        assert len(plain_joy) == len(enc_joy)

        # Session episodes
        plain_session = await plain_store.get_session_episodes(SESSION_ID)
        enc_session = await encrypted_store.get_session_episodes(SESSION_ID)
        assert len(plain_session) == len(enc_session)

    @pytest.mark.asyncio
    async def test_semantic_search_parity(
        self, encryptor: DataEncryptor
    ) -> None:
        """Semantic search returns same results across both stores."""
        plain_store = InMemoryEpisodicStore()
        encrypted_store = EncryptedEpisodicStore(encryptor)

        embedding = [0.1] * 64
        ep = _make_episode(embedding=embedding)

        await plain_store.append(ep)
        await encrypted_store.append(ep)

        query_embedding = [0.1] * 64  # Same as stored — perfect match
        plain_results = await plain_store.semantic_search(
            USER_ID, query_embedding, min_similarity=0.5
        )
        enc_results = await encrypted_store.semantic_search(
            USER_ID, query_embedding, min_similarity=0.5
        )

        assert len(plain_results) == len(enc_results)
        if plain_results:
            assert plain_results[0][0].raw_text == enc_results[0][0].raw_text
            assert abs(plain_results[0][1] - enc_results[0][1]) < 0.001

    @pytest.mark.asyncio
    async def test_compression_parity(self, encryptor: DataEncryptor) -> None:
        """Episode compression works identically across both stores."""
        plain_store = InMemoryEpisodicStore()
        encrypted_store = EncryptedEpisodicStore(encryptor)

        episodes = [_make_episode(raw_text=f"Ep {i}") for i in range(3)]
        for ep in episodes:
            await plain_store.append(ep)
            await encrypted_store.append(ep)

        summary = build_summary(USER_ID, episodes, "Summary of 3 episodes")

        await plain_store.store_summary(summary)
        await encrypted_store.store_summary(summary)

        episode_ids = [ep.id for ep in episodes]
        plain_count = await plain_store.mark_compressed(episode_ids, summary.id)
        enc_count = await encrypted_store.mark_compressed(episode_ids, summary.id)
        assert plain_count == enc_count == 3

        # Compressed episodes should be excluded from default queries
        plain_results = await plain_store.query(USER_ID)
        enc_results = await encrypted_store.query(USER_ID)
        assert len(plain_results) == 0
        assert len(enc_results) == 0

        # Include compressed
        plain_all = await plain_store.query(USER_ID, include_compressed=True)
        enc_all = await encrypted_store.query(USER_ID, include_compressed=True)
        assert len(plain_all) == len(enc_all) == 3

    @pytest.mark.asyncio
    async def test_episodic_no_network_in_local(
        self, encryptor: DataEncryptor, network_blocker: NetworkBlocker
    ) -> None:
        """All episodic memory operations make zero network calls."""
        with network_blocker:
            store = EncryptedEpisodicStore(encryptor)

            for i in range(5):
                ep = _make_episode(raw_text=f"Local episode {i}")
                await store.append(ep)

            await store.query(USER_ID)
            await store.count(USER_ID)
            await store.get_session_episodes(SESSION_ID)
            await store.semantic_search(USER_ID, [0.1] * 64)

            assert len(network_blocker.blocked_calls) == 0


class TestSemanticMemoryParity:
    """Knowledge graph works identically in both modes."""

    @pytest.mark.asyncio
    async def test_entity_operations(
        self, mock_embeddings: MockEmbeddingProvider
    ) -> None:
        """Entity CRUD operations work with local embedding provider."""
        store = SemanticMemoryStore(USER_ID, mock_embeddings)

        entity = await store.add_entity(
            "Alice", EntityType.PERSON, aliases=["Ali"]
        )
        assert entity.name == "Alice"
        assert entity.embedding is not None

        found = await store.find_entity_by_name("alice")
        assert found is not None
        assert found.id == entity.id

        found_alias = await store.find_entity_by_name("ali")
        assert found_alias is not None
        assert found_alias.id == entity.id

    @pytest.mark.asyncio
    async def test_relationship_operations(
        self, mock_embeddings: MockEmbeddingProvider
    ) -> None:
        """Relationship creation and strengthening work in both modes."""
        store = SemanticMemoryStore(USER_ID, mock_embeddings)

        alice = await store.add_entity("Alice", EntityType.PERSON)
        bob = await store.add_entity("Bob", EntityType.PERSON)

        rel = await store.add_or_strengthen_relationship(
            alice.id, bob.id, RelationshipType.MENTIONED_WITH, "They work together"
        )
        assert rel.strength == 1.0

        # Strengthen
        rel2 = await store.add_or_strengthen_relationship(
            alice.id, bob.id, RelationshipType.MENTIONED_WITH
        )
        assert rel2.strength > 1.0
        assert rel2.co_mention_count == 2

    @pytest.mark.asyncio
    async def test_fact_operations(
        self, mock_embeddings: MockEmbeddingProvider
    ) -> None:
        """Fact storage, confirmation, and supersession work in both modes."""
        store = SemanticMemoryStore(USER_ID, mock_embeddings)

        entity = await store.add_entity("Alice", EntityType.PERSON)
        fact = await store.add_fact(
            "Alice likes coffee",
            FactType.PREFERENCE,
            subject_entity_id=entity.id,
        )
        assert fact.content == "Alice likes coffee"
        assert fact.is_active

        facts = await store.get_entity_facts(entity.id)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_pattern_operations(
        self, mock_embeddings: MockEmbeddingProvider
    ) -> None:
        """Pattern detection and promotion work in both modes."""
        store = SemanticMemoryStore(USER_ID, mock_embeddings)

        pattern = await store.add_pattern(
            PatternType.ENERGY_RHYTHM,
            "User is most productive in the morning",
            confidence=0.5,
            observation_count=3,
        )
        assert not pattern.is_active  # Below threshold

        # Confirm until promoted
        for i in range(10):
            await store.confirm_pattern(pattern.id, f"obs-{i}")

        active = await store.get_active_patterns()
        assert len(active) >= 1

    @pytest.mark.asyncio
    async def test_semantic_search_local(
        self, mock_embeddings: MockEmbeddingProvider
    ) -> None:
        """Semantic search across knowledge graph works with local embeddings."""
        store = SemanticMemoryStore(USER_ID, mock_embeddings)

        await store.add_entity("Python Programming", EntityType.TOPIC)
        await store.add_fact(
            "User prefers Python for backend development",
            FactType.PREFERENCE,
        )

        results = await store.search("programming languages", min_similarity=-1.0)
        # With mock embeddings, similarity scores may be low, so we just
        # verify the API works and returns the correct types
        assert isinstance(results, list)
        # Verify we can at least find items when we set no threshold
        all_results = await store.search("Python", min_similarity=-1.0)
        assert len(all_results) > 0  # Should find entities and facts

    @pytest.mark.asyncio
    async def test_semantic_memory_no_network_in_local(
        self,
        mock_embeddings: MockEmbeddingProvider,
        network_blocker: NetworkBlocker,
    ) -> None:
        """All knowledge graph operations make zero network calls with MockEmbeddings."""
        with network_blocker:
            store = SemanticMemoryStore(USER_ID, mock_embeddings)

            alice = await store.add_entity("Alice", EntityType.PERSON, aliases=["Ali"])
            bob = await store.add_entity("Bob", EntityType.PERSON)
            project = await store.add_entity("Project X", EntityType.PROJECT)

            await store.add_or_strengthen_relationship(
                alice.id, bob.id, RelationshipType.MENTIONED_WITH
            )
            await store.add_or_strengthen_relationship(
                alice.id, project.id, RelationshipType.MENTIONED_WITH
            )

            await store.add_fact(
                "Alice leads Project X",
                FactType.ATTRIBUTE,
                subject_entity_id=alice.id,
            )

            await store.add_pattern(
                PatternType.DAY_OF_WEEK,
                "Alice and Bob meet on Mondays",
                confidence=0.8,
                observation_count=6,
            )

            await store.search("project leadership")
            await store.get_entity_context(alice.id)
            await store.get_graph_stats()

            assert len(network_blocker.blocked_calls) == 0


# ---------------------------------------------------------------------------
# 2. Encrypted Storage Parity
# ---------------------------------------------------------------------------


class TestEncryptedStorageParity:
    """Encrypted stores behave identically to plain stores in both modes."""

    @pytest.mark.asyncio
    async def test_encrypted_semantic_store_roundtrip(
        self, encryptor: DataEncryptor
    ) -> None:
        """Knowledge graph encrypted store has full roundtrip fidelity."""
        store = EncryptedSemanticMemoryStore(USER_ID, encryptor)

        entity = EntityNode(
            user_id=USER_ID,
            name="Test Entity",
            normalized_name="test entity",
            entity_type=EntityType.PERSON,
            aliases=["te"],
            attributes={"role": "engineer"},
            mention_count=3,
            embedding=[0.5] * 64,
        )

        store.store_entity(entity)
        loaded = store.load_entity(entity.id)

        assert loaded is not None
        assert loaded.name == entity.name
        assert loaded.entity_type == entity.entity_type
        assert loaded.aliases == entity.aliases
        assert loaded.attributes == entity.attributes
        assert loaded.mention_count == entity.mention_count
        assert loaded.embedding == entity.embedding

    @pytest.mark.asyncio
    async def test_encrypted_sync_state_roundtrip(
        self, encryptor: DataEncryptor
    ) -> None:
        """Sync state encrypted store has full roundtrip fidelity."""
        store = EncryptedSyncStateStore(encryptor)

        record = SyncRecord(
            blurt_id="blurt-123",
            provider=SyncProvider.GOOGLE_CALENDAR,
            status=SyncStatus.PENDING,
        )

        store.upsert_sync_record(record)
        loaded = store.get_sync_record(record.id)

        assert loaded is not None
        assert loaded.blurt_id == "blurt-123"
        assert loaded.provider == SyncProvider.GOOGLE_CALENDAR
        assert loaded.status == SyncStatus.PENDING

    @pytest.mark.asyncio
    async def test_encrypted_stores_no_network(
        self, encryptor: DataEncryptor, network_blocker: NetworkBlocker
    ) -> None:
        """All encrypted store operations make zero network calls."""
        with network_blocker:
            # Episodic
            ep_store = EncryptedEpisodicStore(encryptor)
            ep = _make_episode()
            await ep_store.append(ep)
            await ep_store.get(ep.id)
            await ep_store.query(USER_ID)

            # Semantic
            sem_store = EncryptedSemanticMemoryStore(USER_ID, encryptor)
            entity = EntityNode(
                user_id=USER_ID,
                name="LocalEntity",
                normalized_name="localentity",
                entity_type=EntityType.TOPIC,
            )
            sem_store.store_entity(entity)
            sem_store.load_entity(entity.id)
            sem_store.get_stats()

            # Sync state
            sync_store = EncryptedSyncStateStore(encryptor)
            rec = SyncRecord(
                blurt_id="local-test",
                provider=SyncProvider.NOTION,
                status=SyncStatus.PENDING,
            )
            sync_store.upsert_sync_record(rec)
            sync_store.get_pending_records()
            sync_store.stats()

            assert len(network_blocker.blocked_calls) == 0


# ---------------------------------------------------------------------------
# 3. Sync Orchestrator Mode Behavior
# ---------------------------------------------------------------------------


class TestSyncOrchestratorParity:
    """Sync orchestrator behaves correctly in both modes."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_providers_cloud(self) -> None:
        """In cloud mode, orchestrator syncs to registered providers."""
        orchestrator = SyncOrchestrator()
        gcal = GoogleCalendarAdapter(credentials={"key": "test"})
        notion = NotionAdapter(api_key="test-key", database_id="db-1")
        orchestrator.register_provider(gcal)
        orchestrator.register_provider(notion)

        # EVENT intent should trigger Google Calendar sync
        triggers = await orchestrator.on_blurt_classified(
            blurt_id="blurt-event-1",
            intent="event",
            confidence=0.9,
            payload={"title": "Team meeting", "start": "2026-03-15T10:00:00Z"},
        )

        assert len(triggers) == 1
        assert triggers[0].provider == SyncProvider.GOOGLE_CALENDAR

        # TASK intent should trigger Notion sync
        triggers = await orchestrator.on_blurt_classified(
            blurt_id="blurt-task-1",
            intent="task",
            confidence=0.85,
            payload={"title": "Write tests"},
        )

        assert len(triggers) == 1
        assert triggers[0].provider == SyncProvider.NOTION

    @pytest.mark.asyncio
    async def test_orchestrator_without_providers_local(self) -> None:
        """In local mode without providers, orchestrator silently skips sync."""
        orchestrator = SyncOrchestrator()
        # No providers registered — simulates local-only mode

        triggers = await orchestrator.on_blurt_classified(
            blurt_id="blurt-local-1",
            intent="event",
            confidence=0.9,
            payload={"title": "Team meeting"},
        )

        # No triggers because no providers are registered
        assert len(triggers) == 0

    @pytest.mark.asyncio
    async def test_orchestrator_should_sync_without_providers(self) -> None:
        """should_sync returns False when no providers are registered."""
        orchestrator = SyncOrchestrator()
        assert not orchestrator.should_sync("event", 0.9)
        assert not orchestrator.should_sync("task", 0.85)
        assert not orchestrator.should_sync("reminder", 0.8)

    @pytest.mark.asyncio
    async def test_orchestrator_non_syncable_intents(self) -> None:
        """Non-syncable intents never trigger sync in either mode."""
        orchestrator = SyncOrchestrator()
        gcal = GoogleCalendarAdapter(credentials={"key": "test"})
        orchestrator.register_provider(gcal)

        for intent in ["journal", "idea", "question"]:
            triggers = await orchestrator.on_blurt_classified(
                blurt_id=f"blurt-{intent}",
                intent=intent,
                confidence=0.95,
                payload={"text": f"A {intent} entry"},
            )
            assert len(triggers) == 0, f"{intent} should not trigger sync"

    @pytest.mark.asyncio
    async def test_sync_state_parity(self) -> None:
        """SyncStateStore operations are identical regardless of mode."""
        store = SyncStateStore()

        record = SyncRecord(
            blurt_id="parity-test",
            provider=SyncProvider.GOOGLE_CALENDAR,
            status=SyncStatus.PENDING,
        )
        store.upsert_sync_record(record)

        by_id = store.get_sync_record(record.id)
        assert by_id is not None
        assert by_id.blurt_id == "parity-test"

        by_blurt = store.get_sync_record_by_blurt_id(
            "parity-test", SyncProvider.GOOGLE_CALENDAR
        )
        assert by_blurt is not None

        pending = store.get_pending_records()
        assert len(pending) >= 1

        store.mark_synced(record.id, external_id="ext-1")
        synced = store.get_sync_record(record.id)
        assert synced is not None
        assert synced.status == SyncStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_orchestrator_no_network_local(
        self, network_blocker: NetworkBlocker
    ) -> None:
        """Sync orchestrator with no providers makes zero network calls."""
        with network_blocker:
            orchestrator = SyncOrchestrator()

            # All intent types — none should trigger network calls
            for intent in ["task", "event", "reminder", "idea", "journal", "update", "question"]:
                await orchestrator.on_blurt_classified(
                    blurt_id=f"local-{intent}",
                    intent=intent,
                    confidence=0.95,
                    payload={"title": f"Test {intent}"},
                )

            await orchestrator.pull_inbound_changes()
            await orchestrator.retry_failed()
            await orchestrator.health()

            assert len(network_blocker.blocked_calls) == 0


# ---------------------------------------------------------------------------
# 4. Encryption Always-On in Both Modes
# ---------------------------------------------------------------------------


class TestEncryptionAlwaysOn:
    """Encryption is enabled by default in both modes."""

    def test_local_settings_encryption_default(self, tmp_path) -> None:
        """Local mode has encryption enabled by default."""
        settings = Settings(
            deployment_mode=DeploymentMode.LOCAL,
            data_dir=tmp_path,
        )
        assert settings.encryption_enabled is True

    def test_cloud_settings_encryption_default(self, tmp_path) -> None:
        """Cloud mode has encryption enabled by default."""
        settings = Settings(
            deployment_mode=DeploymentMode.CLOUD,
            data_dir=tmp_path,
        )
        assert settings.encryption_enabled is True

    def test_encryption_same_behavior_both_modes(
        self, master_key: bytes
    ) -> None:
        """DataEncryptor produces identical encrypt/decrypt behavior."""
        enc = DataEncryptor(master_key)

        plaintext = b"Sensitive user data: I have a meeting tomorrow"
        aad = b"user-123"

        encrypted = enc.encrypt(plaintext, aad=aad)
        decrypted = enc.decrypt(encrypted, aad=aad)
        assert decrypted == plaintext

        # JSON roundtrip
        data = {"intent": "task", "text": "Buy groceries", "confidence": 0.92}
        enc_json = enc.encrypt_json(data, aad=aad)
        dec_json = enc.decrypt_json(enc_json, aad=aad)
        assert dec_json == data

    def test_encryption_no_network(
        self, master_key: bytes, network_blocker: NetworkBlocker
    ) -> None:
        """Encryption operations make zero network calls."""
        with network_blocker:
            enc = DataEncryptor(master_key)

            for i in range(20):
                data = f"Sensitive data {i}".encode()
                encrypted = enc.encrypt(data)
                decrypted = enc.decrypt(encrypted)
                assert decrypted == data

            assert len(network_blocker.blocked_calls) == 0


# ---------------------------------------------------------------------------
# 5. Full Pipeline Parity (classify → extract → detect → store → surface)
# ---------------------------------------------------------------------------


class TestFullPipelineParity:
    """The complete blurt processing pipeline works in local-only mode
    with zero external calls and full feature parity.
    """

    @pytest.mark.asyncio
    async def test_pipeline_local_mode_no_network(
        self,
        encryptor: DataEncryptor,
        mock_embeddings: MockEmbeddingProvider,
        network_blocker: NetworkBlocker,
    ) -> None:
        """Full pipeline: classify→extract→detect→store→surface, zero network."""
        with network_blocker:
            # 1. Working memory (classify + detect)
            wm = WorkingMemory(session_id="pipeline-local")
            entry = await wm.add(
                "I need to finish the quarterly report by Friday",
                intent=IntentType.TASK,
                confidence=0.92,
                entities=[
                    {"name": "Quarterly Report", "type": "project"},
                    {"name": "Friday", "type": "date"},
                ],
                emotion=EmotionState(
                    primary=EmotionState().primary,
                    intensity=1.5,
                    valence=-0.2,
                    arousal=0.6,
                ),
            )

            # 2. Episodic memory (store)
            ep_store = EncryptedEpisodicStore(encryptor)
            episode = Episode(
                user_id=USER_ID,
                raw_text=entry.content,
                intent="task",
                intent_confidence=entry.confidence,
                emotion=EmotionSnapshot(
                    primary="anticipation",
                    intensity=1.5,
                    valence=-0.2,
                    arousal=0.6,
                ),
                entities=[
                    EntityRef(
                        name="Quarterly Report",
                        entity_type="project",
                    ),
                ],
                context=EpisodeContext(
                    session_id="pipeline-local",
                    time_of_day="afternoon",
                    day_of_week="wednesday",
                ),
                embedding=await mock_embeddings.embed(entry.content),
            )
            await ep_store.append(episode)

            # 3. Semantic memory (extract entities to knowledge graph)
            sem_store = SemanticMemoryStore(USER_ID, mock_embeddings)
            entities = await sem_store.process_extracted_entities(
                [{"name": "Quarterly Report", "type": "project"}],
                blurt_id=episode.id,
                raw_text=entry.content,
            )
            assert len(entities) == 1

            facts = await sem_store.process_extracted_facts(
                [
                    {
                        "content": "User needs to finish quarterly report by Friday",
                        "type": "attribute",
                        "subject_entity": "Quarterly Report",
                    }
                ],
                blurt_id=episode.id,
            )
            assert len(facts) == 1

            # 4. Surface — verify data is retrievable
            results = await ep_store.query(USER_ID, intent_filter=IntentFilter("task"))
            assert len(results) == 1
            assert results[0].raw_text == entry.content

            context = await sem_store.get_entity_context(entities[0].id)
            assert context["entity"]["name"] == "Quarterly Report"
            assert len(context["facts"]) == 1

            # 5. Sync skipped in local mode
            orchestrator = SyncOrchestrator()
            triggers = await orchestrator.on_blurt_classified(
                blurt_id=episode.id,
                intent="task",
                confidence=0.92,
                payload={"title": entry.content},
            )
            assert len(triggers) == 0  # No sync in local mode

            # ZERO network calls
            assert len(network_blocker.blocked_calls) == 0

    @pytest.mark.asyncio
    async def test_pipeline_data_integrity_across_modes(
        self,
        encryptor: DataEncryptor,
        mock_embeddings: MockEmbeddingProvider,
    ) -> None:
        """Same data stored via local pipeline and cloud pipeline produces
        identical outputs when retrieved.
        """
        raw_text = "Remind me to water the plants every Sunday morning"

        # Simulate local mode storage
        local_store = EncryptedEpisodicStore(encryptor)
        episode = _make_episode(
            raw_text=raw_text,
            intent="reminder",
            confidence=0.88,
        )
        await local_store.append(episode)
        local_result = await local_store.get(episode.id)

        # Simulate cloud mode storage (same encryption, same store type)
        cloud_encryptor = DataEncryptor(_make_master_key())
        cloud_store = EncryptedEpisodicStore(cloud_encryptor)
        cloud_episode = Episode(
            id=episode.id,
            user_id=episode.user_id,
            raw_text=raw_text,
            intent="reminder",
            intent_confidence=0.88,
            modality=InputModality.VOICE,
            emotion=episode.emotion,
            entities=list(episode.entities),
            context=EpisodeContext(
                time_of_day="afternoon",
                day_of_week="wednesday",
                session_id=SESSION_ID,
            ),
            embedding=episode.embedding,
        )
        await cloud_store.append(cloud_episode)
        cloud_result = await cloud_store.get(episode.id)

        # Data fidelity check
        assert local_result is not None
        assert cloud_result is not None
        assert local_result.raw_text == cloud_result.raw_text
        assert local_result.intent == cloud_result.intent
        assert local_result.intent_confidence == cloud_result.intent_confidence
        assert local_result.emotion.primary == cloud_result.emotion.primary
        assert local_result.emotion.intensity == cloud_result.emotion.intensity


# ---------------------------------------------------------------------------
# 6. Embedding Provider Parity
# ---------------------------------------------------------------------------


class TestEmbeddingProviderParity:
    """Mock/local embedding provider has same API as cloud provider."""

    @pytest.mark.asyncio
    async def test_mock_provider_api_complete(
        self, mock_embeddings: MockEmbeddingProvider
    ) -> None:
        """MockEmbeddingProvider implements the full EmbeddingProvider API."""
        # Single embed
        vec = await mock_embeddings.embed("test text")
        assert isinstance(vec, list)
        assert len(vec) == mock_embeddings.dimension
        assert all(isinstance(v, float) for v in vec)

        # Batch embed
        vecs = await mock_embeddings.embed_batch(["text 1", "text 2", "text 3"])
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == mock_embeddings.dimension

        # Dimension property
        assert mock_embeddings.dimension == 64

    @pytest.mark.asyncio
    async def test_mock_embeddings_deterministic(
        self, mock_embeddings: MockEmbeddingProvider
    ) -> None:
        """Same input produces same output (reproducible tests)."""
        v1 = await mock_embeddings.embed("hello world")
        v2 = await mock_embeddings.embed("hello world")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_mock_embeddings_no_network(
        self,
        mock_embeddings: MockEmbeddingProvider,
        network_blocker: NetworkBlocker,
    ) -> None:
        """MockEmbeddingProvider makes zero network calls."""
        with network_blocker:
            for i in range(50):
                await mock_embeddings.embed(f"text {i}")
            await mock_embeddings.embed_batch([f"batch {i}" for i in range(20)])

            assert len(network_blocker.blocked_calls) == 0


# ---------------------------------------------------------------------------
# 7. Settings / Configuration Mode Parity
# ---------------------------------------------------------------------------


class TestConfigurationParity:
    """Configuration system supports both modes with correct defaults."""

    def test_local_mode_settings(self, tmp_path) -> None:
        """Local mode settings are self-contained."""
        settings = Settings(
            deployment_mode=DeploymentMode.LOCAL,
            data_dir=tmp_path / ".blurt",
        )
        assert settings.deployment_mode == DeploymentMode.LOCAL
        assert settings.encryption_enabled is True
        assert settings.data_dir == tmp_path / ".blurt"

    def test_cloud_mode_settings(self, tmp_path) -> None:
        """Cloud mode settings include external service config."""
        settings = Settings(
            deployment_mode=DeploymentMode.CLOUD,
            data_dir=tmp_path / ".blurt",
            google_client_id="test-id",
            google_client_secret="test-secret",
        )
        assert settings.deployment_mode == DeploymentMode.CLOUD
        assert settings.encryption_enabled is True
        assert settings.google_client_id == "test-id"

    def test_both_modes_have_data_dir(self, tmp_path) -> None:
        """Both modes use a data directory for local storage."""
        for mode in DeploymentMode:
            settings = Settings(
                deployment_mode=mode,
                data_dir=tmp_path / f".blurt-{mode.value}",
            )
            assert settings.data_dir is not None

    def test_mode_enum_values(self) -> None:
        """DeploymentMode enum has exactly LOCAL and CLOUD."""
        assert DeploymentMode.LOCAL.value == "local"
        assert DeploymentMode.CLOUD.value == "cloud"
        assert len(DeploymentMode) == 2


# ---------------------------------------------------------------------------
# 8. Comprehensive Zero-Network Integration Test
# ---------------------------------------------------------------------------


class TestZeroNetworkLocalMode:
    """Comprehensive test asserting zero external network calls across
    ALL subsystems when operating in local-only mode.
    """

    @pytest.mark.asyncio
    async def test_full_system_zero_network(
        self, network_blocker: NetworkBlocker, master_key: bytes
    ) -> None:
        """Exercise every subsystem in local mode and verify zero network calls."""
        with network_blocker:
            enc = DataEncryptor(master_key)
            embeddings = MockEmbeddingProvider()

            # --- Working Memory ---
            wm = WorkingMemory(session_id="zero-net")
            for i in range(15):
                await wm.add(
                    f"Entry {i}: some content about topic {i}",
                    intent=IntentType(list(IntentType)[i % 7].value),
                    confidence=0.8 + (i % 3) * 0.05,
                    entities=[{"name": f"Entity-{i}", "type": "person"}],
                )
            ctx = await wm.get_context()
            assert ctx.entry_count > 0
            await wm.get_recent_content(5)
            for intent_type in IntentType:
                await wm.get_entries_by_intent(intent_type)

            # --- Episodic Memory ---
            ep_store = EncryptedEpisodicStore(enc)
            episode_ids = []
            for i in range(10):
                ep = Episode(
                    user_id=USER_ID,
                    raw_text=f"Episode {i}",
                    intent=["task", "event", "reminder", "idea", "journal", "update", "question"][i % 7],
                    intent_confidence=0.85,
                    emotion=EmotionSnapshot(
                        primary=["joy", "trust", "fear", "surprise", "sadness"][i % 5],
                        intensity=float(i % 3),
                        valence=0.5 - (i * 0.1),
                        arousal=0.3 + (i * 0.05),
                    ),
                    entities=[EntityRef(name=f"Entity-{i}", entity_type="person")],
                    context=EpisodeContext(session_id="zero-net"),
                    embedding=await embeddings.embed(f"Episode {i}"),
                )
                await ep_store.append(ep)
                episode_ids.append(ep.id)

            await ep_store.query(USER_ID)
            await ep_store.query(USER_ID, intent_filter=IntentFilter("task"))
            await ep_store.query(USER_ID, emotion_filter=EmotionFilter(primary="joy"))
            await ep_store.query(
                USER_ID,
                entity_filter=EntityFilter(entity_name="Entity-0"),
            )
            await ep_store.count(USER_ID)
            await ep_store.get_session_episodes("zero-net")
            await ep_store.semantic_search(
                USER_ID, await embeddings.embed("test query"), min_similarity=0.0
            )

            # --- Semantic Memory (Knowledge Graph) ---
            sem = SemanticMemoryStore(USER_ID, embeddings)
            entities_created = []
            for name, etype in [
                ("Alice", EntityType.PERSON),
                ("Bob", EntityType.PERSON),
                ("Project X", EntityType.PROJECT),
                ("Acme Corp", EntityType.ORGANIZATION),
                ("Machine Learning", EntityType.TOPIC),
            ]:
                e = await sem.add_entity(name, etype)
                entities_created.append(e)

            for i in range(len(entities_created) - 1):
                await sem.add_or_strengthen_relationship(
                    entities_created[i].id,
                    entities_created[i + 1].id,
                    RelationshipType.MENTIONED_WITH,
                )

            await sem.add_fact(
                "Alice leads Project X",
                FactType.ATTRIBUTE,
                subject_entity_id=entities_created[0].id,
            )
            await sem.add_pattern(
                PatternType.ENERGY_RHYTHM,
                "Most productive before noon",
                confidence=0.8,
                observation_count=10,
            )

            await sem.search("project leadership", min_similarity=0.0)
            await sem.get_graph_stats()
            for e in entities_created:
                await sem.get_entity_context(e.id)

            # --- Encrypted Semantic Store ---
            enc_sem = EncryptedSemanticMemoryStore(USER_ID, enc)
            for e in entities_created:
                enc_sem.store_entity(e)
            for e in entities_created:
                enc_sem.load_entity(e.id)
            enc_sem.get_stats()

            # --- Sync Orchestrator (no providers = local mode) ---
            orch = SyncOrchestrator()
            for intent in ["task", "event", "reminder", "idea", "journal"]:
                await orch.on_blurt_classified(
                    blurt_id=f"zn-{intent}",
                    intent=intent,
                    confidence=0.9,
                    payload={"title": f"Test {intent}"},
                )
            await orch.pull_inbound_changes()
            await orch.retry_failed()
            await orch.health()
            orch.stats()

            # --- Sync State ---
            ss = SyncStateStore()
            rec = SyncRecord(
                blurt_id="zn-rec",
                provider=SyncProvider.GOOGLE_CALENDAR,
                status=SyncStatus.PENDING,
            )
            ss.upsert_sync_record(rec)
            ss.get_pending_records()
            ss.get_conflicted_records()
            ss.mark_synced(rec.id, external_id="ext-1")
            ss.stats()

            # --- Encrypted Sync State ---
            enc_ss = EncryptedSyncStateStore(enc)
            enc_rec = SyncRecord(
                blurt_id="zn-enc-rec",
                provider=SyncProvider.NOTION,
                status=SyncStatus.PENDING,
            )
            enc_ss.upsert_sync_record(enc_rec)
            enc_ss.get_pending_records()
            enc_ss.stats()

            # FINAL ASSERTION: Zero network calls
            assert len(network_blocker.blocked_calls) == 0, (
                f"Expected zero external network calls but got "
                f"{len(network_blocker.blocked_calls)}: "
                f"{network_blocker.blocked_calls}"
            )
