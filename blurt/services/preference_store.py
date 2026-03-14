"""Per-user Beta distribution persistence with temporal decay.

Provides a multi-user storage layer for Thompson Sampling Beta distribution
parameters, with built-in exponential decay for adapting to non-stationary
user preferences. Each user maintains independent Beta(alpha, beta) parameters
per task category, ensuring complete isolation.

AC 11 Sub-AC 4: Persistence layer for per-user Beta distribution parameters.

Design principles:
- Per-user isolation: each user has independent preference parameters
- Decay for non-stationarity: old preferences fade so the model adapts
- Anti-shame: no category is ever fully suppressed (decay toward uniform prior)
- Cold start: new users start with optimistic Beta(1,1) uniform prior
- Thread-safe: all mutations go through controlled methods
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from blurt.services.thompson_sampling import (
    BetaParams,
    DecayConfig,
    DEFAULT_CATEGORIES,
    FeedbackType,
    FeedbackWeights,
)


@dataclass
class UserPreferenceSnapshot:
    """Snapshot of a user's complete preference state.

    Captures all Beta distribution parameters for a user at a point in
    time, suitable for persistence, debugging, or analytics.

    Attributes:
        user_id: The user's unique identifier.
        params: Per-category Beta distribution parameters.
        created_at: When this user's preferences were first created.
        last_interaction: When the user last provided feedback.
        total_feedback_count: Total feedback events across all categories.
        version: Schema version for forward-compatible serialization.
    """

    user_id: str
    params: dict[str, BetaParams] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_interaction: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_feedback_count: int = 0
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "params": {
                cat: p.to_dict() for cat, p in self.params.items()
            },
            "created_at": self.created_at.isoformat(),
            "last_interaction": self.last_interaction.isoformat(),
            "total_feedback_count": self.total_feedback_count,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserPreferenceSnapshot:
        """Deserialize from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        last_interaction = data.get("last_interaction")
        if isinstance(last_interaction, str):
            last_interaction = datetime.fromisoformat(last_interaction)
        elif last_interaction is None:
            last_interaction = datetime.now(timezone.utc)

        params = {}
        for cat, pdata in data.get("params", {}).items():
            params[cat] = BetaParams.from_dict(pdata)

        return cls(
            user_id=data.get("user_id", ""),
            params=params,
            created_at=created_at,
            last_interaction=last_interaction,
            total_feedback_count=int(data.get("total_feedback_count", 0)),
            version=int(data.get("version", 1)),
        )


class PreferenceStoreBackend(Protocol):
    """Protocol for pluggable persistence backends.

    Implementations could be in-memory, SQLite, Redis, S3, etc.
    The backend only handles raw storage — decay logic is in the store layer.
    """

    def load_snapshot(self, user_id: str) -> UserPreferenceSnapshot | None:
        """Load a user's preference snapshot, or None if not found."""
        ...

    def save_snapshot(self, snapshot: UserPreferenceSnapshot) -> None:
        """Persist a user's preference snapshot."""
        ...

    def delete_snapshot(self, user_id: str) -> bool:
        """Delete a user's preferences. Returns True if found and deleted."""
        ...

    def list_users(self) -> list[str]:
        """List all user IDs with stored preferences."""
        ...


class InMemoryPreferenceBackend:
    """In-memory backend for development and testing.

    Thread-safe via a lock. Data is lost on process restart.
    """

    def __init__(self) -> None:
        self._data: dict[str, UserPreferenceSnapshot] = {}
        self._lock = threading.Lock()

    def load_snapshot(self, user_id: str) -> UserPreferenceSnapshot | None:
        with self._lock:
            snap = self._data.get(user_id)
            if snap is None:
                return None
            # Return a deep copy to prevent external mutation
            return UserPreferenceSnapshot.from_dict(snap.to_dict())

    def save_snapshot(self, snapshot: UserPreferenceSnapshot) -> None:
        with self._lock:
            # Store a deep copy
            self._data[snapshot.user_id] = (
                UserPreferenceSnapshot.from_dict(snapshot.to_dict())
            )

    def delete_snapshot(self, user_id: str) -> bool:
        with self._lock:
            if user_id in self._data:
                del self._data[user_id]
                return True
            return False

    def list_users(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())


class UserPreferenceStore:
    """Per-user Beta distribution persistence with temporal decay.

    Manages Thompson Sampling Beta parameters for multiple users with:
    - Automatic decay on read (lazy decay for efficiency)
    - Per-category tracking aligned with Blurt's 7 intent types
    - Feedback recording that updates parameters in-place
    - Snapshot-based persistence through pluggable backends

    The decay mechanism ensures the model adapts to changing user behavior:
    - Parameters decay toward the uniform prior Beta(1,1) over time
    - Decay follows exponential half-life: param = prior + (param - prior) * 2^(-t/half_life)
    - Decay is applied lazily on read, not continuously

    Anti-shame design:
    - No category is ever fully suppressed (min params = prior)
    - Decay ensures old dismissals don't permanently penalize categories
    - New users start with optimistic uniform priors

    Usage:
        store = UserPreferenceStore()
        store.record_feedback("user-1", "task", FeedbackType.ACCEPTED)
        params = store.get_params("user-1", "task")  # Decayed + updated
        snapshot = store.get_snapshot("user-1")  # Full state
    """

    def __init__(
        self,
        backend: PreferenceStoreBackend | None = None,
        categories: list[str] | None = None,
        decay_config: DecayConfig | None = None,
        feedback_weights: FeedbackWeights | None = None,
    ) -> None:
        """Initialize the per-user preference store.

        Args:
            backend: Persistence backend (defaults to in-memory).
            categories: Task categories to track (defaults to 7 Blurt intents).
            decay_config: Temporal decay configuration.
            feedback_weights: How feedback types update parameters.
        """
        self._backend = backend or InMemoryPreferenceBackend()
        self._categories = categories or list(DEFAULT_CATEGORIES)
        self._decay_config = decay_config or DecayConfig()
        self._feedback_weights = feedback_weights or FeedbackWeights()

    @property
    def categories(self) -> list[str]:
        """List of tracked task categories."""
        return list(self._categories)

    @property
    def decay_config(self) -> DecayConfig:
        """Current decay configuration."""
        return self._decay_config

    def _ensure_user(self, user_id: str) -> UserPreferenceSnapshot:
        """Load or create a user's preference snapshot.

        New users are initialized with uniform Beta(1,1) priors for all
        categories — optimistic cold start.
        """
        snapshot = self._backend.load_snapshot(user_id)
        if snapshot is not None:
            # Ensure all categories exist (handles category additions)
            for cat in self._categories:
                if cat not in snapshot.params:
                    snapshot.params[cat] = BetaParams()
            return snapshot

        # Create new user with uniform priors
        now = datetime.now(timezone.utc)
        snapshot = UserPreferenceSnapshot(
            user_id=user_id,
            params={cat: BetaParams() for cat in self._categories},
            created_at=now,
            last_interaction=now,
            total_feedback_count=0,
        )
        self._backend.save_snapshot(snapshot)
        return snapshot

    def _apply_decay(
        self,
        params: BetaParams,
        now: datetime,
    ) -> BetaParams:
        """Apply temporal decay to a single BetaParams instance.

        Decay moves parameters toward the prior Beta(1,1) over time using
        exponential decay with configurable half-life.

        The formula:
            param_new = prior + (param_old - prior) * 2^(-elapsed / half_life)

        Args:
            params: The Beta parameters to decay.
            now: Current timestamp.

        Returns:
            New BetaParams with decay applied (original is not mutated).
        """
        config = self._decay_config
        elapsed_hours = (now - params.last_updated).total_seconds() / 3600.0

        if elapsed_hours < config.decay_interval_hours:
            return params  # Too soon for decay

        if elapsed_hours <= 0:
            return params

        # Exponential decay factor: 2^(-elapsed / half_life)
        decay_factor = math.pow(2.0, -elapsed_hours / config.half_life_hours)

        new_alpha = config.min_alpha + (params.alpha - config.min_alpha) * decay_factor
        new_beta = config.min_beta + (params.beta - config.min_beta) * decay_factor

        return BetaParams(
            alpha=max(config.min_alpha, new_alpha),
            beta=max(config.min_beta, new_beta),
            last_updated=params.last_updated,  # Preserve original update time
            total_observations=params.total_observations,
        )

    def _apply_decay_all(
        self,
        snapshot: UserPreferenceSnapshot,
        now: datetime | None = None,
    ) -> UserPreferenceSnapshot:
        """Apply decay to all categories in a snapshot.

        Returns a new snapshot with decayed parameters. Does not persist.
        """
        now = now or datetime.now(timezone.utc)
        decayed_params = {}
        for cat, params in snapshot.params.items():
            decayed_params[cat] = self._apply_decay(params, now)

        return UserPreferenceSnapshot(
            user_id=snapshot.user_id,
            params=decayed_params,
            created_at=snapshot.created_at,
            last_interaction=snapshot.last_interaction,
            total_feedback_count=snapshot.total_feedback_count,
            version=snapshot.version,
        )

    def get_params(
        self,
        user_id: str,
        category: str,
        apply_decay: bool = True,
        now: datetime | None = None,
    ) -> BetaParams:
        """Get Beta parameters for a user-category pair.

        Applies temporal decay by default so returned values reflect
        current relevance of historical preferences.

        Args:
            user_id: The user's identifier.
            category: The task category.
            apply_decay: Whether to apply temporal decay (default True).
            now: Current timestamp (defaults to UTC now).

        Returns:
            BetaParams for the user-category pair.

        Raises:
            KeyError: If category is not a tracked category.
        """
        if category not in self._categories:
            raise KeyError(
                f"Unknown category '{category}'. "
                f"Known categories: {self._categories}"
            )

        snapshot = self._ensure_user(user_id)
        params = snapshot.params.get(category, BetaParams())

        if apply_decay:
            now = now or datetime.now(timezone.utc)
            params = self._apply_decay(params, now)

        return params

    def get_all_params(
        self,
        user_id: str,
        apply_decay: bool = True,
        now: datetime | None = None,
    ) -> dict[str, BetaParams]:
        """Get all Beta parameters for a user, optionally with decay.

        Args:
            user_id: The user's identifier.
            apply_decay: Whether to apply temporal decay.
            now: Current timestamp.

        Returns:
            Dict of category -> BetaParams.
        """
        snapshot = self._ensure_user(user_id)

        if apply_decay:
            now = now or datetime.now(timezone.utc)
            snapshot = self._apply_decay_all(snapshot, now)

        return dict(snapshot.params)

    def record_feedback(
        self,
        user_id: str,
        category: str,
        feedback: FeedbackType,
        weight_multiplier: float = 1.0,
        now: datetime | None = None,
    ) -> BetaParams:
        """Record user feedback and update Beta parameters.

        Applies temporal decay before the update, then records new
        evidence. The updated parameters are persisted.

        Args:
            user_id: The user's identifier.
            category: The task category that received feedback.
            feedback: Type of user feedback.
            weight_multiplier: Optional multiplier for the update weight.
            now: Current timestamp.

        Returns:
            Updated BetaParams for the category.

        Raises:
            KeyError: If category is not tracked.
        """
        if category not in self._categories:
            raise KeyError(
                f"Unknown category '{category}'. "
                f"Known categories: {self._categories}"
            )

        now = now or datetime.now(timezone.utc)
        snapshot = self._ensure_user(user_id)

        # Apply decay before update so we're working with current-relevance params
        params = snapshot.params.get(category, BetaParams())
        params = self._apply_decay(params, now)

        # Compute deltas based on feedback type
        fw = self._feedback_weights
        alpha_delta = 0.0
        beta_delta = 0.0

        if feedback == FeedbackType.ACCEPTED:
            alpha_delta = fw.accepted_alpha * weight_multiplier
        elif feedback == FeedbackType.COMPLETED:
            alpha_delta = fw.completed_alpha * weight_multiplier
        elif feedback == FeedbackType.DISMISSED:
            beta_delta = fw.dismissed_beta * weight_multiplier
        elif feedback == FeedbackType.SNOOZED:
            alpha_delta = fw.snoozed_alpha * weight_multiplier
            beta_delta = fw.snoozed_beta * weight_multiplier
        elif feedback == FeedbackType.IGNORED:
            beta_delta = fw.ignored_beta * weight_multiplier

        # Apply update
        updated = BetaParams(
            alpha=params.alpha + alpha_delta,
            beta=params.beta + beta_delta,
            last_updated=now,
            total_observations=params.total_observations + 1,
        )

        # Persist
        snapshot.params[category] = updated
        snapshot.last_interaction = now
        snapshot.total_feedback_count += 1
        self._backend.save_snapshot(snapshot)

        return BetaParams(
            alpha=updated.alpha,
            beta=updated.beta,
            last_updated=updated.last_updated,
            total_observations=updated.total_observations,
        )

    def batch_feedback(
        self,
        user_id: str,
        feedbacks: list[tuple[str, FeedbackType]],
        weight_multiplier: float = 1.0,
        now: datetime | None = None,
    ) -> dict[str, BetaParams]:
        """Record multiple feedback events for a user in one batch.

        More efficient than individual record_feedback calls as it
        loads/saves the snapshot only once.

        Args:
            user_id: The user's identifier.
            feedbacks: List of (category, feedback_type) tuples.
            weight_multiplier: Optional weight multiplier for all updates.
            now: Current timestamp.

        Returns:
            Dict of category -> updated BetaParams for affected categories.
        """
        now = now or datetime.now(timezone.utc)
        snapshot = self._ensure_user(user_id)
        results: dict[str, BetaParams] = {}
        fw = self._feedback_weights

        for category, feedback in feedbacks:
            if category not in self._categories:
                raise KeyError(
                    f"Unknown category '{category}'. "
                    f"Known categories: {self._categories}"
                )

            params = snapshot.params.get(category, BetaParams())
            params = self._apply_decay(params, now)

            alpha_delta = 0.0
            beta_delta = 0.0

            if feedback == FeedbackType.ACCEPTED:
                alpha_delta = fw.accepted_alpha * weight_multiplier
            elif feedback == FeedbackType.COMPLETED:
                alpha_delta = fw.completed_alpha * weight_multiplier
            elif feedback == FeedbackType.DISMISSED:
                beta_delta = fw.dismissed_beta * weight_multiplier
            elif feedback == FeedbackType.SNOOZED:
                alpha_delta = fw.snoozed_alpha * weight_multiplier
                beta_delta = fw.snoozed_beta * weight_multiplier
            elif feedback == FeedbackType.IGNORED:
                beta_delta = fw.ignored_beta * weight_multiplier

            updated = BetaParams(
                alpha=params.alpha + alpha_delta,
                beta=params.beta + beta_delta,
                last_updated=now,
                total_observations=params.total_observations + 1,
            )

            snapshot.params[category] = updated
            snapshot.total_feedback_count += 1
            results[category] = BetaParams(
                alpha=updated.alpha,
                beta=updated.beta,
                last_updated=updated.last_updated,
                total_observations=updated.total_observations,
            )

        snapshot.last_interaction = now
        self._backend.save_snapshot(snapshot)
        return results

    def get_snapshot(
        self,
        user_id: str,
        apply_decay: bool = True,
        now: datetime | None = None,
    ) -> UserPreferenceSnapshot:
        """Get the full preference snapshot for a user.

        Args:
            user_id: The user's identifier.
            apply_decay: Whether to apply decay to all params.
            now: Current timestamp.

        Returns:
            UserPreferenceSnapshot with current state.
        """
        snapshot = self._ensure_user(user_id)
        if apply_decay:
            now = now or datetime.now(timezone.utc)
            snapshot = self._apply_decay_all(snapshot, now)
        return snapshot

    def get_category_rankings(
        self,
        user_id: str,
        apply_decay: bool = True,
        now: datetime | None = None,
    ) -> list[tuple[str, float]]:
        """Get categories ranked by expected engagement for a user.

        Returns:
            List of (category, mean) tuples sorted by mean descending.
        """
        all_params = self.get_all_params(
            user_id, apply_decay=apply_decay, now=now
        )
        rankings = [
            (cat, params.mean) for cat, params in all_params.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def reset_user(self, user_id: str) -> None:
        """Reset a user's preferences to uniform priors.

        This is a soft reset — the user record is preserved but all
        parameters return to Beta(1,1).
        """
        now = datetime.now(timezone.utc)
        snapshot = UserPreferenceSnapshot(
            user_id=user_id,
            params={cat: BetaParams() for cat in self._categories},
            created_at=now,
            last_interaction=now,
            total_feedback_count=0,
        )
        self._backend.save_snapshot(snapshot)

    def delete_user(self, user_id: str) -> bool:
        """Permanently delete a user's preference data.

        Returns True if the user existed and was deleted.
        """
        return self._backend.delete_snapshot(user_id)

    def list_users(self) -> list[str]:
        """List all users with stored preferences."""
        return self._backend.list_users()

    def export_snapshot(self, user_id: str) -> dict[str, Any]:
        """Export a user's preferences as a serializable dict.

        Useful for backup, migration, or debugging.
        """
        snapshot = self.get_snapshot(user_id, apply_decay=False)
        return snapshot.to_dict()

    def import_snapshot(self, data: dict[str, Any]) -> str:
        """Import a user's preferences from a serialized dict.

        Args:
            data: Serialized UserPreferenceSnapshot.

        Returns:
            The imported user_id.
        """
        snapshot = UserPreferenceSnapshot.from_dict(data)
        self._backend.save_snapshot(snapshot)
        return snapshot.user_id
