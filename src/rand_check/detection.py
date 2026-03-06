"""Bayesian sequential detection: H₀ (IID Bernoulli) vs H₁ (Markov human).

Maintains a running posterior π_n = P(H₁ | x₁ … x_n) that updates after
every hand.  Uses Bayes' rule with:

    L_n(H₀) = P^{x_n} · (1-P)^{1-x_n}            (IID Bernoulli)
    L_n(H₁) = P(x_n | x_{n-1}, Markov(1) posterior)  (human model)

The detection engine wraps a BayesianMarkovUpdater and exposes
the posterior π_n as a live confidence meter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from rand_check.bayesian_markov import BayesianMarkovUpdater


@dataclass
class DetectionEngine:
    """Sequential H₀ vs H₁ detector.

    Attributes
    ----------
    gto_prob : float
        GTO baseline P for the current spot.
    prior_h1 : float
        Prior probability of H₁ (human pattern) before any data.
        Default 0.5 is uninformative.
    markov : BayesianMarkovUpdater
        The Markov(1) model representing H₁.
    """
    gto_prob: float = 0.25
    prior_h1: float = 0.5

    # Internal state
    _pi: float = field(init=False)
    _markov: BayesianMarkovUpdater = field(init=False)
    _log_lr_cumulative: float = field(default=0.0, init=False)  # cumulative log LR
    _n_obs: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._pi = self.prior_h1
        self._markov = BayesianMarkovUpdater(gto_prob=self.gto_prob)

    @property
    def posterior_h1(self) -> float:
        """Current π_n = P(H₁ | data)."""
        return self._pi

    @property
    def log_likelihood_ratio(self) -> float:
        """Cumulative log(L(H₁) / L(H₀))."""
        return self._log_lr_cumulative

    @property
    def markov(self) -> BayesianMarkovUpdater:
        return self._markov

    def update(self, action: int) -> float:
        """Process a new observation and return updated π_n.

        Parameters
        ----------
        action : int
            0 (fold/call) or 1 (3bet).

        Returns
        -------
        float
            Updated posterior probability of H₁.
        """
        # Likelihood under H₀: IID Bernoulli(P)
        p = self.gto_prob
        if action == 1:
            log_l_h0 = math.log(max(p, 1e-15))
        else:
            log_l_h0 = math.log(max(1.0 - p, 1e-15))

        # Likelihood under H₁: Markov(1) predictive
        log_l_h1 = self._markov.log_likelihood_observation(action)

        # Update the Markov model *after* computing its likelihood
        self._markov.update(action)
        self._n_obs += 1

        # Accumulate log likelihood ratio
        log_lr = log_l_h1 - log_l_h0
        self._log_lr_cumulative += log_lr

        # Bayesian update of π_n
        # π_n = π_{n-1} · L(H₁) / [ π_{n-1}·L(H₁) + (1-π_{n-1})·L(H₀) ]
        # In log-space for numerical stability:
        pi = self._pi
        if pi <= 0.0:
            self._pi = 0.0
            return self._pi
        if pi >= 1.0:
            self._pi = 1.0
            return self._pi

        log_odds = math.log(pi / (1.0 - pi)) + log_lr
        # Clamp to avoid overflow
        log_odds = max(min(log_odds, 20.0), -20.0)
        self._pi = 1.0 / (1.0 + math.exp(-log_odds))

        return self._pi

    def reset(self, gto_prob: Optional[float] = None) -> None:
        """Reset detection state (e.g. after a changepoint)."""
        if gto_prob is not None:
            self.gto_prob = gto_prob
        self._pi = self.prior_h1
        self._log_lr_cumulative = 0.0
        self._n_obs = 0
        self._markov.reset(gto_prob=self.gto_prob)

    def interpret(self) -> str:
        """Human-readable interpretation of current π_n."""
        pi = self._pi
        if pi < 0.3:
            return "Looks like genuine RNG — no exploitation yet"
        elif pi < 0.6:
            return "Mild evidence of patterns — watch closely"
        elif pi < 0.85:
            return "Likely human-generated — begin exploitative adjustments"
        else:
            return "High confidence human pattern — strong exploitation recommended"
