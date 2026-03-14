"""Encryption integration for all data persistence layers.

Wraps episodic memory, semantic memory (knowledge graph), and sync state
stores with transparent encrypt-before-write and decrypt-after-read.

Data is encrypted at the serialization boundary so that:
- In-memory representations are always plaintext for processing
- Persisted/serialized forms are always encrypted
- Encryption is transparent to business logic
- Embeddings are stored in cleartext for vector search (non-sensitive)
- User IDs are used as AAD (additional authenticated data) to bind
  ciphertext to the correct user context

The encrypted stores follow the Decorator pattern: they wrap any existing
store implementation and add encryption transparently.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from blurt.core.encryption import DataEncryptor
from blurt.memory.episodic import (
    BehavioralFilter,
    BehavioralSignal,
    EmotionFilter,
    EmotionSnapshot,
    EntityFilter,
    EntityRef,
    Episode,
    EpisodeContext,
    EpisodicMemoryStore,
    EpisodeSummary,
    InputModality,
    IntentFilter,
    SessionFilter,
    TimeRangeFilter,
    _cosine_similarity,
)
from blurt.models.entities import (
    EntityNode,
    EntityType,
    Fact,
    LearnedPattern,
    PatternType,
    RelationshipEdge,
)
from blurt.models.sync import (
    ConflictRecord,
    SyncOperation,
    SyncProvider,
    SyncRecord,
    SyncStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_value(v: Any) -> Any:
    """Convert non-JSON-serializable values to JSON-friendly types."""
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, Enum):
        return v.value
    if is_dataclass(v) and not isinstance(v, type):
        return {k: _serialize_value(val) for k, val in asdict(v).items()}
    if isinstance(v, dict):
        return {str(k): _serialize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize_value(item) for item in v]
    return v


def _episode_to_dict(ep: Episode) -> dict[str, Any]:
    """Serialize an Episode to a dictionary for encryption."""
    return {
        "id": ep.id,
        "user_id": ep.user_id,
        "timestamp": ep.timestamp.isoformat(),
        "raw_text": ep.raw_text,
        "modality": ep.modality.value,
        "intent": ep.intent,
        "intent_confidence": ep.intent_confidence,
        "emotion": {
            "primary": ep.emotion.primary,
            "intensity": ep.emotion.intensity,
            "valence": ep.emotion.valence,
            "arousal": ep.emotion.arousal,
            "secondary": ep.emotion.secondary,
        },
        "entities": [
            {
                "name": e.name,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "confidence": e.confidence,
            }
            for e in ep.entities
        ],
        "behavioral_signal": ep.behavioral_signal.value,
        "surfaced_task_id": ep.surfaced_task_id,
        "context": {
            "time_of_day": ep.context.time_of_day,
            "day_of_week": ep.context.day_of_week,
            "session_id": ep.context.session_id,
            "preceding_episode_id": ep.context.preceding_episode_id,
            "active_task_id": ep.context.active_task_id,
        },
        "is_compressed": ep.is_compressed,
        "compressed_into_id": ep.compressed_into_id,
        "source_working_id": ep.source_working_id,
        # Embeddings stored separately (cleartext for vector search)
    }


def _dict_to_episode(d: dict[str, Any], embedding: list[float] | None = None) -> Episode:
    """Deserialize a dictionary back to an Episode."""
    emotion_data = d.get("emotion", {})
    context_data = d.get("context", {})

    ep = Episode(
        id=d["id"],
        user_id=d["user_id"],
        raw_text=d["raw_text"],
        modality=InputModality(d.get("modality", "voice")),
        intent=d.get("intent", "task"),
        intent_confidence=d.get("intent_confidence", 0.0),
        emotion=EmotionSnapshot(
            primary=emotion_data.get("primary", "trust"),
            intensity=emotion_data.get("intensity", 0.0),
            valence=emotion_data.get("valence", 0.0),
            arousal=emotion_data.get("arousal", 0.0),
            secondary=emotion_data.get("secondary"),
        ),
        entities=[
            EntityRef(
                name=e["name"],
                entity_type=e["entity_type"],
                entity_id=e.get("entity_id"),
                confidence=e.get("confidence", 1.0),
            )
            for e in d.get("entities", [])
        ],
        behavioral_signal=BehavioralSignal(d.get("behavioral_signal", "none")),
        surfaced_task_id=d.get("surfaced_task_id"),
        context=EpisodeContext(
            time_of_day=context_data.get("time_of_day", "morning"),
            day_of_week=context_data.get("day_of_week", "monday"),
            session_id=context_data.get("session_id", ""),
            preceding_episode_id=context_data.get("preceding_episode_id"),
            active_task_id=context_data.get("active_task_id"),
        ),
        is_compressed=d.get("is_compressed", False),
        compressed_into_id=d.get("compressed_into_id"),
        embedding=embedding,
        source_working_id=d.get("source_working_id"),
    )
    ep.timestamp = datetime.fromisoformat(d["timestamp"])
    return ep


def _summary_to_dict(s: EpisodeSummary) -> dict[str, Any]:
    """Serialize an EpisodeSummary to a dictionary for encryption."""
    return {
        "id": s.id,
        "user_id": s.user_id,
        "created_at": s.created_at.isoformat(),
        "period_start": s.period_start.isoformat(),
        "period_end": s.period_end.isoformat(),
        "source_episode_ids": s.source_episode_ids,
        "episode_count": s.episode_count,
        "summary_text": s.summary_text,
        "dominant_emotions": [
            {
                "primary": e.primary,
                "intensity": e.intensity,
                "valence": e.valence,
                "arousal": e.arousal,
                "secondary": e.secondary,
            }
            for e in s.dominant_emotions
        ],
        "entity_mentions": s.entity_mentions,
        "intent_distribution": s.intent_distribution,
        "behavioral_signals": s.behavioral_signals,
    }


def _dict_to_summary(d: dict[str, Any], embedding: list[float] | None = None) -> EpisodeSummary:
    """Deserialize a dictionary back to an EpisodeSummary."""
    return EpisodeSummary(
        id=d["id"],
        user_id=d["user_id"],
        created_at=datetime.fromisoformat(d["created_at"]),
        period_start=datetime.fromisoformat(d["period_start"]),
        period_end=datetime.fromisoformat(d["period_end"]),
        source_episode_ids=d.get("source_episode_ids", []),
        episode_count=d.get("episode_count", 0),
        summary_text=d.get("summary_text", ""),
        dominant_emotions=[
            EmotionSnapshot(
                primary=e.get("primary", "trust"),
                intensity=e.get("intensity", 0.0),
                valence=e.get("valence", 0.0),
                arousal=e.get("arousal", 0.0),
                secondary=e.get("secondary"),
            )
            for e in d.get("dominant_emotions", [])
        ],
        entity_mentions=d.get("entity_mentions", {}),
        intent_distribution=d.get("intent_distribution", {}),
        behavioral_signals=d.get("behavioral_signals", {}),
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Encrypted Episodic Memory Store
# ---------------------------------------------------------------------------


class EncryptedEpisodicStore(EpisodicMemoryStore):
    """Episodic memory store with transparent encrypt-on-write / decrypt-on-read.

    Wraps any EpisodicMemoryStore implementation (in-memory, SQLite, cloud)
    with AES-256-GCM encryption. All episode data (raw_text, entities,
    emotions, context) is encrypted before storage and decrypted on retrieval.

    Embeddings are stored in cleartext for vector similarity search since
    they don't contain directly identifiable information and are required
    for unencrypted computation.

    The user_id is used as AAD (additional authenticated data) to bind
    the ciphertext to the user context, preventing cross-user data swaps.
    """

    def __init__(self, encryptor: DataEncryptor) -> None:
        self._enc = encryptor
        # Encrypted storage: episode_id -> encrypted_bytes
        self._encrypted_episodes: dict[str, bytes] = {}
        # Cleartext indexes needed for filtering/search
        self._embeddings: dict[str, list[float]] = {}  # episode_id -> embedding
        self._user_index: dict[str, list[str]] = defaultdict(list)
        self._session_index: dict[str, list[str]] = defaultdict(list)
        self._entity_index: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Minimal metadata indexes for filtering without decryption
        self._timestamps: dict[str, datetime] = {}
        self._intents: dict[str, str] = {}
        self._compressed: dict[str, bool] = {}
        self._compressed_into: dict[str, str | None] = {}
        self._user_ids: dict[str, str] = {}
        self._emotions: dict[str, EmotionSnapshot] = {}
        self._behavioral_signals: dict[str, BehavioralSignal] = {}
        self._session_ids: dict[str, str] = {}
        # Entity names per episode for entity filtering
        self._episode_entities: dict[str, list[EntityRef]] = {}
        # Summaries
        self._encrypted_summaries: dict[str, bytes] = {}
        self._summary_embeddings: dict[str, list[float]] = {}
        self._user_summaries: dict[str, list[str]] = defaultdict(list)
        self._summary_periods: dict[str, tuple[datetime, datetime]] = {}

    def _encrypt_episode(self, episode: Episode) -> bytes:
        """Encrypt episode data, using user_id as AAD."""
        data = _episode_to_dict(episode)
        aad = episode.user_id.encode("utf-8") if episode.user_id else None
        return self._enc.encrypt_json(data, aad=aad)

    def _decrypt_episode(self, episode_id: str) -> Episode:
        """Decrypt an episode by ID."""
        encrypted = self._encrypted_episodes[episode_id]
        user_id = self._user_ids[episode_id]
        aad = user_id.encode("utf-8") if user_id else None
        data = self._enc.decrypt_json(encrypted, aad=aad)
        embedding = self._embeddings.get(episode_id)
        return _dict_to_episode(data, embedding)

    def _index_episode(self, episode: Episode) -> None:
        """Build cleartext indexes from episode metadata."""
        eid = episode.id
        self._timestamps[eid] = episode.timestamp
        self._intents[eid] = episode.intent
        self._compressed[eid] = episode.is_compressed
        self._compressed_into[eid] = episode.compressed_into_id
        self._user_ids[eid] = episode.user_id
        self._emotions[eid] = episode.emotion
        self._behavioral_signals[eid] = episode.behavioral_signal
        self._session_ids[eid] = episode.context.session_id
        self._episode_entities[eid] = list(episode.entities)

        self._user_index[episode.user_id].append(eid)
        self._session_index[episode.context.session_id].append(eid)

        if episode.embedding is not None:
            self._embeddings[eid] = episode.embedding

        for entity in episode.entities:
            key = entity.name.lower()
            self._entity_index[episode.user_id][key].append(eid)

    async def append(self, episode: Episode) -> Episode:
        if episode.id in self._encrypted_episodes:
            raise ValueError(f"Episode {episode.id} already exists (append-only)")

        # Encrypt and store
        self._encrypted_episodes[episode.id] = self._encrypt_episode(episode)
        self._index_episode(episode)
        return episode

    async def get(self, episode_id: str) -> Episode | None:
        if episode_id not in self._encrypted_episodes:
            return None
        return self._decrypt_episode(episode_id)

    async def query(
        self,
        user_id: str,
        *,
        time_range: TimeRangeFilter | None = None,
        entity_filter: EntityFilter | None = None,
        emotion_filter: EmotionFilter | None = None,
        intent_filter: IntentFilter | None = None,
        behavioral_filter: BehavioralFilter | None = None,
        session_filter: SessionFilter | None = None,
        limit: int = 50,
        offset: int = 0,
        include_compressed: bool = False,
    ) -> list[Episode]:
        # Filter using indexes first (no decryption needed)
        candidate_ids = self._user_index.get(user_id, [])

        # Apply index-based filters
        filtered_ids = []
        for eid in candidate_ids:
            if not include_compressed and self._compressed.get(eid, False):
                continue
            if intent_filter and self._intents.get(eid) != intent_filter.intent:
                continue
            if time_range:
                ts = self._timestamps.get(eid)
                if ts:
                    if time_range.start and ts < time_range.start:
                        continue
                    if time_range.end and ts > time_range.end:
                        continue
            if behavioral_filter:
                if self._behavioral_signals.get(eid) != behavioral_filter.signal:
                    continue
            if session_filter:
                if self._session_ids.get(eid) != session_filter.session_id:
                    continue
            if emotion_filter:
                em = self._emotions.get(eid)
                if em:
                    if emotion_filter.primary and em.primary != emotion_filter.primary:
                        continue
                    if em.intensity < emotion_filter.min_intensity:
                        continue
                    if emotion_filter.valence_range:
                        lo, hi = emotion_filter.valence_range
                        if not (lo <= em.valence <= hi):
                            continue
            if entity_filter:
                entities = self._episode_entities.get(eid, [])
                match = False
                for ent in entities:
                    if entity_filter.entity_id and ent.entity_id == entity_filter.entity_id:
                        match = True
                        break
                    if entity_filter.entity_name and ent.name.lower() == entity_filter.entity_name:
                        match = True
                        break
                if not match:
                    continue

            filtered_ids.append(eid)

        # Sort by timestamp descending (newest first)
        filtered_ids.sort(key=lambda eid: self._timestamps.get(eid, datetime.min.replace(tzinfo=timezone.utc)), reverse=True)

        # Paginate, then decrypt only what's needed
        page_ids = filtered_ids[offset: offset + limit]
        return [self._decrypt_episode(eid) for eid in page_ids]

    async def count(self, user_id: str) -> int:
        return len(self._user_index.get(user_id, []))

    async def get_session_episodes(self, session_id: str) -> list[Episode]:
        episode_ids = self._session_index.get(session_id, [])
        episodes = [self._decrypt_episode(eid) for eid in episode_ids]
        episodes.sort(key=lambda ep: ep.timestamp)
        return episodes

    async def semantic_search(
        self,
        user_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> list[tuple[Episode, float]]:
        episode_ids = self._user_index.get(user_id, [])
        scored: list[tuple[str, float]] = []

        for eid in episode_ids:
            if self._compressed.get(eid, False):
                continue
            emb = self._embeddings.get(eid)
            if emb is None:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= min_similarity:
                scored.append((eid, sim))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        # Decrypt only top results
        return [(self._decrypt_episode(eid), sim) for eid, sim in scored[:limit]]

    async def get_entity_timeline(
        self,
        user_id: str,
        entity_name: str,
        limit: int = 20,
    ) -> list[Episode]:
        key = entity_name.lower()
        episode_ids = self._entity_index.get(user_id, {}).get(key, [])
        non_compressed = [eid for eid in episode_ids if not self._compressed.get(eid, False)]
        non_compressed.sort(
            key=lambda eid: self._timestamps.get(eid, datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )
        return [self._decrypt_episode(eid) for eid in non_compressed[:limit]]

    async def get_emotion_timeline(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> list[Episode]:
        episode_ids = self._user_index.get(user_id, [])
        matching = []
        for eid in episode_ids:
            ts = self._timestamps.get(eid)
            if ts and start <= ts <= end and not self._compressed.get(eid, False):
                matching.append(eid)
        matching.sort(key=lambda eid: self._timestamps[eid])
        return [self._decrypt_episode(eid) for eid in matching]

    async def mark_compressed(
        self, episode_ids: list[str], summary_id: str
    ) -> int:
        count = 0
        for eid in episode_ids:
            if eid in self._encrypted_episodes and not self._compressed.get(eid, False):
                # Decrypt, update, re-encrypt
                episode = self._decrypt_episode(eid)
                episode.is_compressed = True
                episode.compressed_into_id = summary_id
                self._encrypted_episodes[eid] = self._encrypt_episode(episode)
                self._compressed[eid] = True
                self._compressed_into[eid] = summary_id
                count += 1
        return count

    async def store_summary(self, summary: EpisodeSummary) -> EpisodeSummary:
        data = _summary_to_dict(summary)
        aad = summary.user_id.encode("utf-8") if summary.user_id else None
        self._encrypted_summaries[summary.id] = self._enc.encrypt_json(data, aad=aad)
        if summary.embedding is not None:
            self._summary_embeddings[summary.id] = summary.embedding
        self._user_summaries[summary.user_id].append(summary.id)
        self._summary_periods[summary.id] = (summary.period_start, summary.period_end)
        return summary

    async def get_summaries(
        self,
        user_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[EpisodeSummary]:
        summary_ids = self._user_summaries.get(user_id, [])
        results = []
        aad = user_id.encode("utf-8") if user_id else None

        for sid in summary_ids:
            period = self._summary_periods.get(sid)
            if period:
                p_start, p_end = period
                if start and p_end < start:
                    continue
                if end and p_start > end:
                    continue

            data = self._enc.decrypt_json(self._encrypted_summaries[sid], aad=aad)
            embedding = self._summary_embeddings.get(sid)
            results.append(_dict_to_summary(data, embedding))

        results.sort(key=lambda s: s.period_start, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Encrypted Semantic Memory Store (Knowledge Graph)
# ---------------------------------------------------------------------------


class EncryptedSemanticMemoryStore:
    """Knowledge graph store with transparent encryption at the persistence boundary.

    All entity nodes, relationship edges, facts, and patterns are encrypted
    before storage and decrypted on read. Embeddings remain in cleartext
    for vector search operations.

    This wraps the data storage layer, not the SemanticMemoryStore business
    logic. It provides encrypt/decrypt operations for each data type that
    can be used by any persistent backend (SQLite, PostgreSQL, cloud).
    """

    def __init__(self, user_id: str, encryptor: DataEncryptor) -> None:
        self.user_id = user_id
        self._enc = encryptor

        # Encrypted storage
        self._encrypted_entities: dict[str, bytes] = {}
        self._encrypted_relationships: dict[str, bytes] = {}
        self._encrypted_facts: dict[str, bytes] = {}
        self._encrypted_patterns: dict[str, bytes] = {}

        # Cleartext indexes for search/lookup
        self._entity_embeddings: dict[str, list[float]] = {}
        self._fact_embeddings: dict[str, list[float]] = {}
        self._pattern_embeddings: dict[str, list[float]] = {}

        # Index lookups (non-sensitive metadata only)
        self._entity_by_name: dict[str, str] = {}  # normalized_name -> id
        self._entity_by_alias: dict[str, str] = {}  # alias -> entity_id
        self._entity_types: dict[str, EntityType] = {}
        self._entity_names: dict[str, str] = {}  # id -> name
        self._relationships_by_entity: dict[str, list[str]] = defaultdict(list)
        self._facts_by_entity: dict[str, list[str]] = defaultdict(list)
        self._fact_active: dict[str, bool] = {}
        self._pattern_active: dict[str, bool] = {}
        self._pattern_types: dict[str, PatternType] = {}

    @property
    def _aad(self) -> bytes:
        return self.user_id.encode("utf-8")

    # ── Entity persistence ────────────────────────────────────────────

    def store_entity(self, entity: EntityNode) -> None:
        """Encrypt and store an entity node."""
        data = entity.model_dump(exclude={"embedding"})
        # Serialize datetimes
        for key in ("first_seen", "last_seen", "created_at", "updated_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_entities[entity.id] = self._enc.encrypt_json(data, aad=self._aad)

        # Update cleartext indexes
        if entity.embedding is not None:
            self._entity_embeddings[entity.id] = entity.embedding
        self._entity_by_name[entity.normalized_name] = entity.id
        self._entity_types[entity.id] = entity.entity_type
        self._entity_names[entity.id] = entity.name
        for alias in entity.aliases:
            self._entity_by_alias[alias] = entity.id

    def load_entity(self, entity_id: str) -> EntityNode | None:
        """Decrypt and return an entity node."""
        encrypted = self._encrypted_entities.get(entity_id)
        if encrypted is None:
            return None
        data = self._enc.decrypt_json(encrypted, aad=self._aad)
        # Restore embedding from cleartext store
        data["embedding"] = self._entity_embeddings.get(entity_id)
        return EntityNode(**data)

    def get_entity_id_by_name(self, name: str) -> str | None:
        """Look up entity ID by normalized name or alias (no decryption needed)."""
        normalized = name.lower().strip()
        return self._entity_by_name.get(normalized) or self._entity_by_alias.get(normalized)

    def get_all_entity_ids(self, entity_type: EntityType | None = None) -> list[str]:
        """Get all entity IDs, optionally filtered by type."""
        if entity_type is None:
            return list(self._encrypted_entities.keys())
        return [eid for eid, et in self._entity_types.items() if et == entity_type]

    # ── Relationship persistence ──────────────────────────────────────

    def store_relationship(self, rel: RelationshipEdge) -> None:
        """Encrypt and store a relationship edge."""
        data = rel.model_dump()
        for key in ("first_seen", "last_seen", "created_at", "updated_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_relationships[rel.id] = self._enc.encrypt_json(data, aad=self._aad)

        # Index by both entities
        if rel.id not in self._relationships_by_entity.get(rel.source_entity_id, []):
            self._relationships_by_entity[rel.source_entity_id].append(rel.id)
        if rel.id not in self._relationships_by_entity.get(rel.target_entity_id, []):
            self._relationships_by_entity[rel.target_entity_id].append(rel.id)

    def load_relationship(self, rel_id: str) -> RelationshipEdge | None:
        """Decrypt and return a relationship edge."""
        encrypted = self._encrypted_relationships.get(rel_id)
        if encrypted is None:
            return None
        data = self._enc.decrypt_json(encrypted, aad=self._aad)
        return RelationshipEdge(**data)

    def get_relationship_ids_for_entity(self, entity_id: str) -> list[str]:
        """Get relationship IDs for an entity (no decryption needed)."""
        return self._relationships_by_entity.get(entity_id, [])

    # ── Fact persistence ──────────────────────────────────────────────

    def store_fact(self, fact: Fact) -> None:
        """Encrypt and store a fact."""
        data = fact.model_dump(exclude={"embedding"})
        for key in ("first_learned", "last_confirmed", "created_at", "updated_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_facts[fact.id] = self._enc.encrypt_json(data, aad=self._aad)

        if fact.embedding is not None:
            self._fact_embeddings[fact.id] = fact.embedding
        self._fact_active[fact.id] = fact.is_active
        if fact.subject_entity_id:
            if fact.id not in self._facts_by_entity.get(fact.subject_entity_id, []):
                self._facts_by_entity[fact.subject_entity_id].append(fact.id)

    def load_fact(self, fact_id: str) -> Fact | None:
        """Decrypt and return a fact."""
        encrypted = self._encrypted_facts.get(fact_id)
        if encrypted is None:
            return None
        data = self._enc.decrypt_json(encrypted, aad=self._aad)
        data["embedding"] = self._fact_embeddings.get(fact_id)
        return Fact(**data)

    def get_fact_ids_for_entity(self, entity_id: str) -> list[str]:
        """Get fact IDs for an entity."""
        return self._facts_by_entity.get(entity_id, [])

    def get_all_active_fact_ids(self) -> list[str]:
        """Get all active fact IDs."""
        return [fid for fid, active in self._fact_active.items() if active]

    # ── Pattern persistence ───────────────────────────────────────────

    def store_pattern(self, pattern: LearnedPattern) -> None:
        """Encrypt and store a learned pattern."""
        data = pattern.model_dump(exclude={"embedding"})
        for key in ("first_detected", "last_confirmed", "created_at", "updated_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_patterns[pattern.id] = self._enc.encrypt_json(data, aad=self._aad)

        if pattern.embedding is not None:
            self._pattern_embeddings[pattern.id] = pattern.embedding
        self._pattern_active[pattern.id] = pattern.is_active
        self._pattern_types[pattern.id] = pattern.pattern_type

    def load_pattern(self, pattern_id: str) -> LearnedPattern | None:
        """Decrypt and return a learned pattern."""
        encrypted = self._encrypted_patterns.get(pattern_id)
        if encrypted is None:
            return None
        data = self._enc.decrypt_json(encrypted, aad=self._aad)
        data["embedding"] = self._pattern_embeddings.get(pattern_id)
        return LearnedPattern(**data)

    def get_active_pattern_ids(self, pattern_type: PatternType | None = None) -> list[str]:
        """Get active pattern IDs."""
        ids = [pid for pid, active in self._pattern_active.items() if active]
        if pattern_type:
            ids = [pid for pid in ids if self._pattern_types.get(pid) == pattern_type]
        return ids

    # ── Graph stats (no decryption needed) ────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics from indexes (no decryption)."""
        entity_type_counts: dict[str, int] = {}
        for et in self._entity_types.values():
            entity_type_counts[et.value] = entity_type_counts.get(et.value, 0) + 1

        return {
            "total_entities": len(self._encrypted_entities),
            "total_relationships": len(self._encrypted_relationships),
            "total_facts": len(self._encrypted_facts),
            "active_facts": sum(1 for a in self._fact_active.values() if a),
            "total_patterns": len(self._encrypted_patterns),
            "active_patterns": sum(1 for a in self._pattern_active.values() if a),
            "entity_type_counts": entity_type_counts,
        }


# ---------------------------------------------------------------------------
# Encrypted Sync State Store
# ---------------------------------------------------------------------------


class EncryptedSyncStateStore:
    """Sync state store with encryption at rest.

    Sync records, operations, and conflict records are encrypted before
    storage. Provider and status metadata is kept in cleartext indexes
    for efficient filtering.
    """

    def __init__(self, encryptor: DataEncryptor) -> None:
        self._enc = encryptor

        # Encrypted storage
        self._encrypted_records: dict[str, bytes] = {}
        self._encrypted_operations: dict[str, bytes] = {}
        self._encrypted_conflicts: dict[str, bytes] = {}

        # Cleartext indexes for filtering
        self._record_status: dict[str, SyncStatus] = {}
        self._record_provider: dict[str, SyncProvider] = {}
        self._blurt_provider_index: dict[str, str] = {}  # "blurt_id:provider" -> record_id
        self._external_id_index: dict[str, str] = {}  # "external_id:provider" -> record_id
        self._record_blurt_ids: dict[str, str] = {}  # record_id -> blurt_id
        self._op_record_index: dict[str, list[str]] = defaultdict(list)  # record_id -> [op_ids]
        self._op_status: dict[str, SyncStatus] = {}
        self._op_created_at: dict[str, datetime] = {}
        self._conflict_resolved: dict[str, bool] = {}
        self._conflict_record_index: dict[str, str] = {}  # conflict_id -> record_id

    def _index_key(self, blurt_id: str, provider: SyncProvider) -> str:
        return f"{blurt_id}:{provider.value}"

    def _ext_key(self, external_id: str, provider: SyncProvider) -> str:
        return f"{external_id}:{provider.value}"

    # --- Sync Records ---

    def upsert_sync_record(self, record: SyncRecord) -> SyncRecord:
        """Encrypt and store a sync record."""
        data = record.model_dump()
        for key in ("last_synced_at", "last_blurt_modified_at",
                     "last_external_modified_at", "created_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_records[record.id] = self._enc.encrypt_json(data)

        # Update indexes
        self._record_status[record.id] = record.status
        self._record_provider[record.id] = record.provider
        self._record_blurt_ids[record.id] = record.blurt_id
        key = self._index_key(record.blurt_id, record.provider)
        self._blurt_provider_index[key] = record.id
        if record.external_id:
            ext_key = self._ext_key(record.external_id, record.provider)
            self._external_id_index[ext_key] = record.id
        return record

    def get_sync_record(self, record_id: str) -> SyncRecord | None:
        """Decrypt and return a sync record."""
        encrypted = self._encrypted_records.get(record_id)
        if encrypted is None:
            return None
        data = self._enc.decrypt_json(encrypted)
        return SyncRecord(**data)

    def get_sync_record_by_blurt_id(
        self, blurt_id: str, provider: SyncProvider
    ) -> SyncRecord | None:
        key = self._index_key(blurt_id, provider)
        record_id = self._blurt_provider_index.get(key)
        if record_id:
            return self.get_sync_record(record_id)
        return None

    def get_sync_record_by_external_id(
        self, external_id: str, provider: SyncProvider
    ) -> SyncRecord | None:
        ext_key = self._ext_key(external_id, provider)
        record_id = self._external_id_index.get(ext_key)
        if record_id:
            return self.get_sync_record(record_id)
        return None

    def get_pending_records(self, provider: SyncProvider | None = None) -> list[SyncRecord]:
        """Get records needing sync (decrypts only matching records)."""
        results = []
        for rid, status in self._record_status.items():
            if provider and self._record_provider.get(rid) != provider:
                continue
            if status in (SyncStatus.PENDING, SyncStatus.FAILED):
                record = self.get_sync_record(rid)
                if record and record.can_retry:
                    results.append(record)
            elif status == SyncStatus.COMPLETED:
                record = self.get_sync_record(rid)
                if record and (record.needs_outbound_sync or record.needs_inbound_sync):
                    results.append(record)
        return results

    def get_conflicted_records(self) -> list[SyncRecord]:
        """Get records with unresolved conflicts."""
        return [
            record
            for rid, status in self._record_status.items()
            if status == SyncStatus.CONFLICT
            for record in (self.get_sync_record(rid),)
            if record is not None
        ]

    def mark_synced(
        self, record_id: str,
        external_id: str | None = None,
        external_version: str | None = None,
    ) -> SyncRecord | None:
        record = self.get_sync_record(record_id)
        if record is None:
            return None
        now = datetime.now(timezone.utc)
        record.status = SyncStatus.COMPLETED
        record.last_synced_at = now
        record.error_message = None
        record.retry_count = 0
        if external_id:
            record.external_id = external_id
        if external_version:
            record.external_version = external_version
        return self.upsert_sync_record(record)

    def mark_failed(self, record_id: str, error: str) -> SyncRecord | None:
        record = self.get_sync_record(record_id)
        if record is None:
            return None
        record.status = SyncStatus.FAILED
        record.error_message = error
        record.retry_count += 1
        return self.upsert_sync_record(record)

    def mark_conflict(self, record_id: str) -> SyncRecord | None:
        record = self.get_sync_record(record_id)
        if record is None:
            return None
        record.status = SyncStatus.CONFLICT
        return self.upsert_sync_record(record)

    # --- Operations ---

    def add_operation(self, operation: SyncOperation) -> SyncOperation:
        data = operation.model_dump()
        for key in ("created_at", "completed_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_operations[operation.id] = self._enc.encrypt_json(data)
        self._op_record_index[operation.sync_record_id].append(operation.id)
        self._op_status[operation.id] = operation.status
        self._op_created_at[operation.id] = operation.created_at
        return operation

    def get_operation(self, operation_id: str) -> SyncOperation | None:
        encrypted = self._encrypted_operations.get(operation_id)
        if encrypted is None:
            return None
        data = self._enc.decrypt_json(encrypted)
        return SyncOperation(**data)

    def get_operations_for_record(self, sync_record_id: str) -> list[SyncOperation]:
        op_ids = self._op_record_index.get(sync_record_id, [])
        ops = []
        for oid in op_ids:
            op = self.get_operation(oid)
            if op:
                ops.append(op)
        return sorted(ops, key=lambda o: o.created_at)

    def complete_operation(
        self, operation_id: str, result: dict[str, Any] | None = None
    ) -> SyncOperation | None:
        op = self.get_operation(operation_id)
        if op is None:
            return None
        op.status = SyncStatus.COMPLETED
        op.completed_at = datetime.now(timezone.utc)
        op.result = result
        # Re-encrypt
        data = op.model_dump()
        for key in ("created_at", "completed_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_operations[op.id] = self._enc.encrypt_json(data)
        self._op_status[op.id] = op.status
        return op

    def fail_operation(self, operation_id: str, error: str) -> SyncOperation | None:
        op = self.get_operation(operation_id)
        if op is None:
            return None
        op.status = SyncStatus.FAILED
        op.error_message = error
        data = op.model_dump()
        for key in ("created_at", "completed_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_operations[op.id] = self._enc.encrypt_json(data)
        self._op_status[op.id] = op.status
        return op

    # --- Conflicts ---

    def add_conflict(self, conflict: ConflictRecord) -> ConflictRecord:
        data = conflict.model_dump()
        for key in ("created_at", "resolved_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_conflicts[conflict.id] = self._enc.encrypt_json(data)
        self._conflict_resolved[conflict.id] = conflict.resolved
        self._conflict_record_index[conflict.id] = conflict.sync_record_id
        return conflict

    def get_conflict(self, conflict_id: str) -> ConflictRecord | None:
        encrypted = self._encrypted_conflicts.get(conflict_id)
        if encrypted is None:
            return None
        data = self._enc.decrypt_json(encrypted)
        return ConflictRecord(**data)

    def get_unresolved_conflicts(self) -> list[ConflictRecord]:
        return [
            conflict
            for cid, resolved in self._conflict_resolved.items()
            if not resolved
            for conflict in (self.get_conflict(cid),)
            if conflict is not None
        ]

    def resolve_conflict(
        self, conflict_id: str, result: dict[str, Any]
    ) -> ConflictRecord | None:
        conflict = self.get_conflict(conflict_id)
        if conflict is None:
            return None
        conflict.resolved = True
        conflict.resolved_at = datetime.now(timezone.utc)
        conflict.resolution_result = result
        # Re-encrypt
        data = conflict.model_dump()
        for key in ("created_at", "resolved_at"):
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        self._encrypted_conflicts[conflict.id] = self._enc.encrypt_json(data)
        self._conflict_resolved[conflict.id] = True
        return conflict

    # --- Stats ---

    def stats(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for status in self._record_status.values():
            status_counts[status.value] = status_counts.get(status.value, 0) + 1

        return {
            "total_records": len(self._encrypted_records),
            "total_operations": len(self._encrypted_operations),
            "total_conflicts": len(self._encrypted_conflicts),
            "unresolved_conflicts": sum(
                1 for resolved in self._conflict_resolved.values() if not resolved
            ),
            "status_counts": status_counts,
        }
