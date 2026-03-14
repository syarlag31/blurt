"""Thompson Sampling engine for intelligent task surfacing.

Uses Beta distribution tracking for each task category to learn which types
of tasks the user is most likely to engage with at any given time. The engine
balances exploration (trying different categories) with exploitation (favoring
categories the user historically engages with).

AC 11: Behavioral learning — Thompson Sampling with Beta priors.

Design principles:
- Anti-shame: dismissed/snoozed feedback gently updates priors, never punishes
- Decay over time: old interactions fade so the model adapts to changing behavior
- No forced engagement: sampling can return None if no category scores well
- Shame-free: no streaks, no guilt, no overdue counters
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FeedbackType(str, Enum):
    """User feedback on a surfaced task."""

    ACCEPTED = "accepted"      # User engaged with the task
    DISMISSED = "dismissed"    # User explicitly dismissed
    SNOOZED = "snoozed"       # User deferred to later
    COMPLETED = "completed"   # User completed the task
    IGNORED = "ignored"       # Task was surfaced but no interaction


# Default task categories aligned with Blurt's 7 intents
DEFAULT_CATEGORIES = [
    "task",
    "event",
    "reminder",
    "idea",
    "journal",
    "update",
    "question",
]


@dataclass
class BetaParams:
    """Beta distribution parameters for a single category.

    The Beta(alpha, beta) distribution models our belief about the
    probability that a user will engage with a surfaced item from
    this category. Higher alpha relative to beta means higher
    expected engagement.

    Attributes:
        alpha: Success parameter (pseudo-count of positive outcomes).
        beta: Failure parameter (pseudo-count of negative outcomes).
        last_updated: Timestamp of the last parameter update.
        total_observations: Total number of feedback events recorded.
    """

    alpha: float = 1.0  # Start with uniform prior Beta(1,1)
    beta: float = 1.0
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_observations: int = 0

    @property
    def mean(self) -> float:
        """Expected value of the Beta distribution: alpha / (alpha + beta)."""
        total = self.alpha + self.beta
        if total == 0:
            return 0.5
        return self.alpha / total

    @property
    def variance(self) -> float:
        """Variance of the Beta distribution."""
        a, b = self.alpha, self.beta
        total = a + b
        if total == 0 or total + 1 == 0:
            return 0.0
        return (a * b) / (total * total * (total + 1))

    @property
    def confidence(self) -> float:
        """Confidence in the estimate (0-1). Higher with more observations.

        Uses inverse variance as a proxy — more observations → lower variance
        → higher confidence.
        """
        if self.total_observations == 0:
            return 0.0
        # Sigmoid-like: approaches 1.0 as observations grow
        return 1.0 - 1.0 / (1.0 + 0.1 * self.total_observations)

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw a sample from the Beta(alpha, beta) distribution.

        Args:
            rng: Optional random number generator for reproducibility.

        Returns:
            A float in [0, 1] drawn from Beta(alpha, beta).
        """
        r = rng or random.Random()

        # Clamp parameters to valid range (must be > 0)
        a = max(self.alpha, 0.01)
        b = max(self.beta, 0.01)

        return r.betavariate(a, b)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "last_updated": self.last_updated.isoformat(),
            "total_observations": self.total_observations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BetaParams:
        """Deserialize from dictionary."""
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        elif last_updated is None:
            last_updated = datetime.now(timezone.utc)

        return cls(
            alpha=float(data.get("alpha", 1.0)),
            beta=float(data.get("beta", 1.0)),
            last_updated=last_updated,
            total_observations=int(data.get("total_observations", 0)),
        )


@dataclass
class SamplingResult:
    """Result of a Thompson Sampling round.

    Attributes:
        category: The selected category.
        sampled_value: The sampled value from the Beta distribution.
        params: The Beta parameters at sampling time.
        all_samples: All category samples for transparency/debugging.
    """

    category: str
    sampled_value: float
    params: BetaParams
    all_samples: dict[str, float] = field(default_factory=dict)


@dataclass
class DecayConfig:
    """Configuration for temporal decay of Beta parameters.

    Decay prevents stale historical data from dominating current behavior.
    Parameters decay toward the prior (Beta(1,1)) over time.

    Attributes:
        half_life_hours: Hours until parameters decay by half toward prior.
        min_alpha: Minimum alpha after decay (never goes below prior).
        min_beta: Minimum beta after decay (never goes below prior).
        decay_interval_hours: Minimum hours between decay applications.
    """

    half_life_hours: float = 168.0  # 1 week
    min_alpha: float = 1.0  # Don't decay below uniform prior
    min_beta: float = 1.0
    decay_interval_hours: float = 1.0  # Apply decay at most once per hour


@dataclass
class FeedbackWeights:
    """How much each feedback type updates the Beta parameters.

    Positive feedback (accepted/completed) increases alpha.
    Negative feedback (dismissed) increases beta.
    Snoozed is treated as mild negative — the user wants it, just not now.
    Ignored is very mild negative — we don't want to over-penalize silence.

    Anti-shame design: even negative feedback has small weights to avoid
    punishing the user for not engaging.

    Attributes:
        accepted_alpha: Alpha increment for accepted feedback.
        completed_alpha: Alpha increment for completed feedback.
        dismissed_beta: Beta increment for dismissed feedback.
        snoozed_beta: Beta increment for snoozed feedback.
        ignored_beta: Beta increment for ignored feedback.
        snoozed_alpha: Small alpha increment for snoozed (user wants it, just not now).
    """

    accepted_alpha: float = 1.0
    completed_alpha: float = 1.5  # Completion is strong positive signal
    dismissed_beta: float = 0.5   # Gentle — don't over-penalize
    snoozed_beta: float = 0.2     # Very gentle — they want it, just not now
    snoozed_alpha: float = 0.3    # Partial positive — intent to engage later
    ignored_beta: float = 0.1     # Minimal — silence isn't rejection


class ThompsonSamplingEngine:
    """Thompson Sampling engine for task category selection.

    Maintains Beta distribution parameters for each task category and uses
    Thompson Sampling to select which category of tasks to surface. Learns
    from user feedback (accepted/dismissed/snoozed) and applies temporal
    decay so the model adapts to changing behavior.

    Usage:
        engine = ThompsonSamplingEngine()
        result = engine.sample()  # Pick best category
        # ... surface a task from result.category ...
        engine.update("task", FeedbackType.ACCEPTED)  # User engaged

    Anti-shame principles:
    - Dismissed feedback has lower weight than accepted
    - Snoozed counts as partial positive (user wants it, just not now)
    - Decay ensures old dismissals don't permanently suppress categories
    - No category is ever fully suppressed (min params = prior)
    """

    def __init__(
        self,
        categories: list[str] | None = None,
        decay_config: DecayConfig | None = None,
        feedback_weights: FeedbackWeights | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the Thompson Sampling engine.

        Args:
            categories: Task categories to track. Defaults to Blurt's 7 intents.
            decay_config: Temporal decay configuration.
            feedback_weights: How feedback types update parameters.
            seed: Random seed for reproducibility (testing).
        """
        self._categories = categories or list(DEFAULT_CATEGORIES)
        self._decay_config = decay_config or DecayConfig()
        self._feedback_weights = feedback_weights or FeedbackWeights()
        self._rng = random.Random(seed)

        # Initialize Beta(1,1) uniform priors for each category
        self._params: dict[str, BetaParams] = {
            cat: BetaParams() for cat in self._categories
        }

        # Track feedback history for analysis
        self._feedback_history: list[dict[str, Any]] = []

    @property
    def categories(self) -> list[str]:
        """List of tracked categories."""
        return list(self._categories)

    @property
    def params(self) -> dict[str, BetaParams]:
        """Current Beta parameters for all categories (read-only view)."""
        return dict(self._params)

    def get_params(self, category: str) -> BetaParams:
        """Get Beta parameters for a specific category.

        Args:
            category: The task category.

        Returns:
            BetaParams for the category.

        Raises:
            KeyError: If category is not tracked.
        """
        if category not in self._params:
            raise KeyError(
                f"Unknown category '{category}'. "
                f"Known categories: {self._categories}"
            )
        return self._params[category]

    def add_category(self, category: str, params: BetaParams | None = None) -> None:
        """Add a new category with optional initial parameters.

        Args:
            category: New category name.
            params: Initial Beta parameters (defaults to uniform prior).
        """
        if category not in self._params:
            self._categories.append(category)
        self._params[category] = params or BetaParams()

    def sample(
        self,
        eligible_categories: list[str] | None = None,
        apply_decay: bool = True,
    ) -> SamplingResult:
        """Perform Thompson Sampling to select a task category.

        Draws a sample from each category's Beta distribution and returns
        the category with the highest sample. This naturally balances
        exploration (uncertain categories have high variance) with
        exploitation (high-engagement categories have high mean).

        Args:
            eligible_categories: Subset of categories to sample from.
                If None, samples from all categories.
            apply_decay: Whether to apply temporal decay before sampling.

        Returns:
            SamplingResult with the selected category and all samples.
        """
        candidates = eligible_categories or self._categories

        # Validate categories
        valid_candidates = [c for c in candidates if c in self._params]
        if not valid_candidates:
            # Fallback: use all categories
            valid_candidates = self._categories

        # Apply temporal decay if configured
        if apply_decay:
            now = datetime.now(timezone.utc)
            for cat in valid_candidates:
                self._apply_decay(cat, now)

        # Draw samples from each category's Beta distribution
        samples: dict[str, float] = {}
        for cat in valid_candidates:
            params = self._params[cat]
            samples[cat] = params.sample(self._rng)

        # Select category with highest sample
        best_category = max(samples, key=lambda c: samples[c])

        return SamplingResult(
            category=best_category,
            sampled_value=samples[best_category],
            params=BetaParams(
                alpha=self._params[best_category].alpha,
                beta=self._params[best_category].beta,
                last_updated=self._params[best_category].last_updated,
                total_observations=self._params[best_category].total_observations,
            ),
            all_samples=samples,
        )

    def sample_top_k(
        self,
        k: int = 3,
        eligible_categories: list[str] | None = None,
        apply_decay: bool = True,
    ) -> list[SamplingResult]:
        """Sample and return top-k categories ranked by sampled value.

        Useful when you want to surface tasks from multiple categories
        at once, ranked by Thompson Sampling preference.

        Args:
            k: Number of top categories to return.
            eligible_categories: Subset of categories to sample from.
            apply_decay: Whether to apply temporal decay before sampling.

        Returns:
            List of SamplingResults, sorted by sampled value descending.
        """
        candidates = eligible_categories or self._categories
        valid_candidates = [c for c in candidates if c in self._params]
        if not valid_candidates:
            valid_candidates = self._categories

        # Apply decay once
        if apply_decay:
            now = datetime.now(timezone.utc)
            for cat in valid_candidates:
                self._apply_decay(cat, now)

        # Draw samples
        samples: dict[str, float] = {}
        for cat in valid_candidates:
            params = self._params[cat]
            samples[cat] = params.sample(self._rng)

        # Sort by sample value descending, take top k
        sorted_cats = sorted(samples, key=lambda c: samples[c], reverse=True)
        top_k = sorted_cats[:k]

        results = []
        for cat in top_k:
            results.append(
                SamplingResult(
                    category=cat,
                    sampled_value=samples[cat],
                    params=BetaParams(
                        alpha=self._params[cat].alpha,
                        beta=self._params[cat].beta,
                        last_updated=self._params[cat].last_updated,
                        total_observations=self._params[cat].total_observations,
                    ),
                    all_samples=samples,
                )
            )

        return results

    def update(
        self,
        category: str,
        feedback: FeedbackType,
        weight_multiplier: float = 1.0,
    ) -> BetaParams:
        """Update Beta parameters based on user feedback.

        Positive feedback (accepted/completed) increases alpha.
        Negative feedback (dismissed/ignored) increases beta.
        Snoozed gets both a small alpha and beta increase (partial positive).

        Args:
            category: The task category that received feedback.
            feedback: Type of user feedback.
            weight_multiplier: Optional multiplier for the update weight
                (e.g., higher for strong signals, lower for weak).

        Returns:
            Updated BetaParams for the category.

        Raises:
            KeyError: If category is not tracked.
        """
        if category not in self._params:
            raise KeyError(
                f"Unknown category '{category}'. "
                f"Known categories: {self._categories}"
            )

        params = self._params[category]
        now = datetime.now(timezone.utc)
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
            # Snoozed is nuanced: user wants it, just not now
            # Small alpha (intent) + small beta (not now)
            alpha_delta = fw.snoozed_alpha * weight_multiplier
            beta_delta = fw.snoozed_beta * weight_multiplier
        elif feedback == FeedbackType.IGNORED:
            beta_delta = fw.ignored_beta * weight_multiplier

        # Apply updates
        params.alpha += alpha_delta
        params.beta += beta_delta
        params.last_updated = now
        params.total_observations += 1

        # Record in history
        self._feedback_history.append({
            "category": category,
            "feedback": feedback.value,
            "alpha_delta": alpha_delta,
            "beta_delta": beta_delta,
            "timestamp": now.isoformat(),
            "resulting_alpha": params.alpha,
            "resulting_beta": params.beta,
        })

        return BetaParams(
            alpha=params.alpha,
            beta=params.beta,
            last_updated=params.last_updated,
            total_observations=params.total_observations,
        )

    def batch_update(
        self,
        updates: list[tuple[str, FeedbackType]],
        weight_multiplier: float = 1.0,
    ) -> dict[str, BetaParams]:
        """Apply multiple feedback updates at once.

        Args:
            updates: List of (category, feedback_type) tuples.
            weight_multiplier: Optional weight multiplier for all updates.

        Returns:
            Dict of category → updated BetaParams for all affected categories.
        """
        results: dict[str, BetaParams] = {}
        for category, feedback in updates:
            results[category] = self.update(category, feedback, weight_multiplier)
        return results

    def _apply_decay(self, category: str, now: datetime) -> None:
        """Apply temporal decay to a category's Beta parameters.

        Decay moves parameters toward the prior (Beta(1,1)) over time,
        preventing stale data from dominating. Uses exponential decay
        with configurable half-life.

        The decay formula:
            param_new = prior + (param_old - prior) * 2^(-elapsed / half_life)

        This ensures:
        - Recent feedback has full weight
        - Old feedback gradually fades
        - Parameters never drop below the prior (uniform)
        - The model stays adaptive to changing behavior

        Args:
            category: The category to decay.
            now: Current timestamp.
        """
        params = self._params[category]
        config = self._decay_config

        # Check if enough time has passed since last decay
        elapsed_hours = (
            now - params.last_updated
        ).total_seconds() / 3600.0

        if elapsed_hours < config.decay_interval_hours:
            return  # Too soon for another decay

        if elapsed_hours <= 0:
            return

        # Exponential decay factor: 2^(-elapsed / half_life)
        decay_factor = math.pow(2.0, -elapsed_hours / config.half_life_hours)

        # Decay toward prior (1.0 for both alpha and beta)
        prior_alpha = config.min_alpha
        prior_beta = config.min_beta

        new_alpha = prior_alpha + (params.alpha - prior_alpha) * decay_factor
        new_beta = prior_beta + (params.beta - prior_beta) * decay_factor

        # Enforce minimums
        params.alpha = max(config.min_alpha, new_alpha)
        params.beta = max(config.min_beta, new_beta)

        # Note: we don't update last_updated here to avoid resetting
        # the decay clock. last_updated reflects the last feedback event.

    def apply_decay_all(self, now: datetime | None = None) -> None:
        """Apply temporal decay to all categories.

        Args:
            now: Current timestamp. Defaults to UTC now.
        """
        now = now or datetime.now(timezone.utc)
        for category in self._categories:
            self._apply_decay(category, now)

    def get_category_rankings(self) -> list[tuple[str, float]]:
        """Get categories ranked by their expected engagement rate.

        Returns:
            List of (category, mean) tuples sorted by mean descending.
            This represents the exploitation-only ranking (no exploration).
        """
        rankings = [
            (cat, params.mean)
            for cat, params in self._params.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_category_stats(self) -> dict[str, dict[str, Any]]:
        """Get detailed statistics for all categories.

        Returns:
            Dict of category → stats including mean, variance,
            confidence, and parameter values.
        """
        stats: dict[str, dict[str, Any]] = {}
        for cat, params in self._params.items():
            stats[cat] = {
                "alpha": params.alpha,
                "beta": params.beta,
                "mean": params.mean,
                "variance": params.variance,
                "confidence": params.confidence,
                "total_observations": params.total_observations,
                "last_updated": params.last_updated.isoformat(),
            }
        return stats

    def reset_category(self, category: str) -> None:
        """Reset a category to uniform prior Beta(1,1).

        Args:
            category: Category to reset.

        Raises:
            KeyError: If category is not tracked.
        """
        if category not in self._params:
            raise KeyError(f"Unknown category '{category}'")
        self._params[category] = BetaParams()

    def reset_all(self) -> None:
        """Reset all categories to uniform prior Beta(1,1)."""
        for cat in self._categories:
            self._params[cat] = BetaParams()
        self._feedback_history.clear()

    @property
    def feedback_history(self) -> list[dict[str, Any]]:
        """Get the feedback history (read-only copy)."""
        return list(self._feedback_history)

    def to_dict(self) -> dict[str, Any]:
        """Serialize engine state for persistence.

        Returns:
            Dict containing all parameters and configuration.
        """
        return {
            "categories": self._categories,
            "params": {
                cat: params.to_dict()
                for cat, params in self._params.items()
            },
            "decay_config": {
                "half_life_hours": self._decay_config.half_life_hours,
                "min_alpha": self._decay_config.min_alpha,
                "min_beta": self._decay_config.min_beta,
                "decay_interval_hours": self._decay_config.decay_interval_hours,
            },
            "feedback_weights": {
                "accepted_alpha": self._feedback_weights.accepted_alpha,
                "completed_alpha": self._feedback_weights.completed_alpha,
                "dismissed_beta": self._feedback_weights.dismissed_beta,
                "snoozed_beta": self._feedback_weights.snoozed_beta,
                "snoozed_alpha": self._feedback_weights.snoozed_alpha,
                "ignored_beta": self._feedback_weights.ignored_beta,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], seed: int | None = None) -> ThompsonSamplingEngine:
        """Deserialize engine state from a dictionary.

        Args:
            data: Serialized engine state.
            seed: Optional random seed.

        Returns:
            Restored ThompsonSamplingEngine instance.
        """
        decay_data = data.get("decay_config", {})
        decay_config = DecayConfig(
            half_life_hours=decay_data.get("half_life_hours", 168.0),
            min_alpha=decay_data.get("min_alpha", 1.0),
            min_beta=decay_data.get("min_beta", 1.0),
            decay_interval_hours=decay_data.get("decay_interval_hours", 1.0),
        )

        fw_data = data.get("feedback_weights", {})
        feedback_weights = FeedbackWeights(
            accepted_alpha=fw_data.get("accepted_alpha", 1.0),
            completed_alpha=fw_data.get("completed_alpha", 1.5),
            dismissed_beta=fw_data.get("dismissed_beta", 0.5),
            snoozed_beta=fw_data.get("snoozed_beta", 0.2),
            snoozed_alpha=fw_data.get("snoozed_alpha", 0.3),
            ignored_beta=fw_data.get("ignored_beta", 0.1),
        )

        engine = cls(
            categories=data.get("categories", DEFAULT_CATEGORIES),
            decay_config=decay_config,
            feedback_weights=feedback_weights,
            seed=seed,
        )

        # Restore per-category parameters
        params_data = data.get("params", {})
        for cat, pdata in params_data.items():
            engine._params[cat] = BetaParams.from_dict(pdata)
            if cat not in engine._categories:
                engine._categories.append(cat)

        return engine
