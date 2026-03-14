"""Thompson Sampling for adaptive task surfacing weight adjustment.

Uses Beta-Bernoulli Thompson Sampling to learn which scoring signal
dimensions (time, energy, mood, etc.) best predict user engagement.
Weights adjust over time based on what the user actually completes
vs. skips — all on-device, zero API cost.

Design:
- Each signal dimension has a Beta(alpha, beta) prior.
- When a surfaced task is completed, we update alpha for the signals
  that contributed most to its ranking (reward).
- When a surfaced task is dismissed/ignored, we update beta (no reward).
- Sampling from these distributions produces exploration-exploitation
  balanced weights for the scoring engine.

Anti-shame principles:
- Dismissals are neutral feedback, not punishment.
- Cold start uses uniform, optimistic priors (Beta(1,1) = uniform).
- Weight adjustments are gentle — no dramatic swings.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class ArmState:
    """Beta distribution parameters for one signal arm.

    Alpha counts successes (task completed when this signal ranked high).
    Beta counts failures (task dismissed when this signal ranked high).
    """

    alpha: float = 1.0  # successes + prior
    beta: float = 1.0   # failures + prior

    @property
    def mean(self) -> float:
        """Expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def total_observations(self) -> float:
        """Total observations (alpha + beta - 2 for the prior)."""
        return self.alpha + self.beta - 2.0

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw a sample from Beta(alpha, beta).

        Uses the provided RNG for reproducibility in tests.
        """
        r = rng or random
        # random.betavariate requires alpha > 0 and beta > 0
        return r.betavariate(self.alpha, self.beta)


@dataclass
class ThompsonSampler:
    """Thompson Sampling bandit for adaptive surfacing weight learning.

    Each arm corresponds to a signal dimension from the scoring engine.
    The sampler learns which signals best predict user engagement.

    Attributes:
        arms: Mapping of signal name → Beta distribution state.
        decay_factor: Exponential decay applied to old observations to
            allow the model to adapt to changing user behavior. 1.0 = no
            decay, 0.99 = slow forgetting, 0.95 = faster adaptation.
        min_weight: Floor weight to prevent any signal from being fully
            ignored (ensures exploration).
    """

    arms: dict[str, ArmState] = field(default_factory=dict)
    decay_factor: float = 1.0
    min_weight: float = 0.05

    # Default signal dimensions matching the scoring engine
    DEFAULT_SIGNALS: tuple[str, ...] = (
        "time_relevance",
        "energy_match",
        "context_relevance",
        "emotional_alignment",
        "momentum",
        "freshness",
    )

    def __post_init__(self) -> None:
        """Ensure all default signals have arms."""
        for signal in self.DEFAULT_SIGNALS:
            if signal not in self.arms:
                self.arms[signal] = ArmState()

    def sample_weights(self, rng: random.Random | None = None) -> dict[str, float]:
        """Sample weights from the current posterior distributions.

        Returns normalized weights suitable for the scoring engine.
        Each weight is drawn from its arm's Beta distribution, then
        normalized to sum to 1.0. A minimum floor is applied to ensure
        exploration.

        Args:
            rng: Optional random.Random instance for reproducibility.

        Returns:
            Dict mapping signal name to weight in (0, 1), summing to 1.0.
        """
        raw = {}
        for name, arm in self.arms.items():
            sample = arm.sample(rng)
            raw[name] = max(self.min_weight, sample)

        total = sum(raw.values())
        if total == 0:
            # Shouldn't happen with min_weight, but safety fallback
            n = len(raw)
            return {k: 1.0 / n for k in raw}

        return {k: v / total for k, v in raw.items()}

    def get_expected_weights(self) -> dict[str, float]:
        """Get expected (mean) weights without sampling randomness.

        Useful for inspecting the learned weights deterministically.

        Returns:
            Dict mapping signal name to normalized expected weight.
        """
        raw = {name: max(self.min_weight, arm.mean) for name, arm in self.arms.items()}
        total = sum(raw.values())
        if total == 0:
            n = len(raw)
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}

    def record_feedback(
        self,
        signal_contributions: dict[str, float],
        reward: bool,
        learning_rate: float = 1.0,
    ) -> None:
        """Record user feedback (completion or dismissal) for a surfaced task.

        Updates the Beta distribution parameters for each signal arm
        proportional to how much that signal contributed to the task's
        ranking.

        Args:
            signal_contributions: Mapping of signal name → normalized
                contribution (0-1) to the task's composite score. Higher
                contribution means the signal was more responsible for
                surfacing this task.
            reward: True if the user completed/engaged with the task,
                False if dismissed/ignored.
            learning_rate: Multiplier on the update magnitude. Default 1.0.
                Lower values make learning more gradual.
        """
        if not signal_contributions:
            return

        # Apply decay to existing observations (forgetting old data)
        if self.decay_factor < 1.0:
            self._apply_decay()

        for signal_name, contribution in signal_contributions.items():
            if signal_name not in self.arms:
                self.arms[signal_name] = ArmState()

            arm = self.arms[signal_name]
            # Update proportional to contribution
            update = contribution * learning_rate

            if reward:
                arm.alpha += update
            else:
                arm.beta += update

    def record_completion(
        self,
        signal_scores: dict[str, float],
        composite_score: float,
        learning_rate: float = 1.0,
    ) -> None:
        """Convenience: record a task completion with its signal breakdown.

        Computes signal contributions from the score breakdown and
        records a positive reward.

        Args:
            signal_scores: The signal breakdown from a ScoredTask.
            composite_score: The composite score of the completed task.
            learning_rate: Learning rate multiplier.
        """
        contributions = self._compute_contributions(signal_scores, composite_score)
        self.record_feedback(contributions, reward=True, learning_rate=learning_rate)

    def record_dismissal(
        self,
        signal_scores: dict[str, float],
        composite_score: float,
        learning_rate: float = 1.0,
    ) -> None:
        """Convenience: record a task dismissal with its signal breakdown.

        Computes signal contributions from the score breakdown and
        records a negative reward (no engagement).

        Args:
            signal_scores: The signal breakdown from a ScoredTask.
            composite_score: The composite score of the dismissed task.
            learning_rate: Learning rate multiplier.
        """
        contributions = self._compute_contributions(signal_scores, composite_score)
        self.record_feedback(contributions, reward=False, learning_rate=learning_rate)

    def _compute_contributions(
        self,
        signal_scores: dict[str, float],
        composite_score: float,
    ) -> dict[str, float]:
        """Compute each signal's relative contribution to the composite score.

        Returns a normalized dict where values sum to ~1.0.
        """
        if not signal_scores or composite_score == 0:
            # Equal contribution when we can't differentiate
            n = len(signal_scores) if signal_scores else 1
            return {k: 1.0 / n for k in signal_scores}

        total = sum(signal_scores.values())
        if total == 0:
            n = len(signal_scores)
            return {k: 1.0 / n for k in signal_scores}

        return {k: v / total for k, v in signal_scores.items()}

    def _apply_decay(self) -> None:
        """Apply exponential decay to all arms.

        This moves observations slightly toward the prior, allowing
        the model to forget old patterns and adapt to new behavior.
        The prior (1.0, 1.0) is preserved — decay only affects the
        accumulated evidence.
        """
        for arm in self.arms.values():
            # Decay only the evidence portion (above prior)
            excess_alpha = arm.alpha - 1.0
            excess_beta = arm.beta - 1.0
            arm.alpha = 1.0 + excess_alpha * self.decay_factor
            arm.beta = 1.0 + excess_beta * self.decay_factor

    def reset(self) -> None:
        """Reset all arms to uniform prior (cold start)."""
        for arm in self.arms.values():
            arm.alpha = 1.0
            arm.beta = 1.0

    def get_state(self) -> dict[str, dict[str, float]]:
        """Serialize sampler state for persistence.

        Returns:
            Dict of arm_name → {alpha, beta, mean, variance}.
        """
        return {
            name: {
                "alpha": arm.alpha,
                "beta": arm.beta,
                "mean": arm.mean,
                "variance": arm.variance,
            }
            for name, arm in self.arms.items()
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, dict[str, float]],
        decay_factor: float = 1.0,
        min_weight: float = 0.05,
    ) -> ThompsonSampler:
        """Restore a sampler from serialized state.

        Args:
            state: Dict of arm_name → {alpha, beta}.
            decay_factor: Decay factor for the restored sampler.
            min_weight: Minimum weight floor.

        Returns:
            Restored ThompsonSampler instance.
        """
        arms = {}
        for name, params in state.items():
            arms[name] = ArmState(
                alpha=params.get("alpha", 1.0),
                beta=params.get("beta", 1.0),
            )
        return cls(arms=arms, decay_factor=decay_factor, min_weight=min_weight)
