"""Bayesian sequential detection: H0 (IID Bernoulli) vs H1 (Markov patterned).

Maintains a running posterior pi_n = P(H1 | x1 ... xn) that updates after
every observation.  Uses Bayes' rule with:

    L_n(H0) = P^{x_n} * (1-P)^{1-x_n}            (IID Bernoulli)
    L_n(H1) = P(x_n | x_{n-1}, Markov(1) posterior)  (patterned model)

The detection engine wraps a BayesianMarkovUpdater and exposes
the posterior pi_n as a live confidence meter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from rand_check.bayesian_markov import BayesianMarkovUpdater


@dataclass
class DetectionEngine:
    """Sequential H0 vs H1 detector.

    Attributes
    ----------
    baseline_prob : float
        Baseline probability P for the null hypothesis.
    prior_h1 : float
        Prior probability of H1 (patterned) before any data.
        Default 0.5 is uninformative.
    markov : BayesianMarkovUpdater
        The Markov(1) model representing H1.
    """
    baseline_prob: float = 0.50
    prior_h1: float = 0.5

    # Internal state
    _pi: float = field(init=False)
    _markov: BayesianMarkovUpdater = field(init=False)
    _log_lr_cumulative: float = field(default=0.0, init=False)
    _n_obs: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._pi = self.prior_h1
        self._markov = BayesianMarkovUpdater(baseline_prob=self.baseline_prob)

    @property
    def posterior_h1(self) -> float:
        """Current pi_n = P(H1 | data)."""
        return self._pi

    @property
    def log_likelihood_ratio(self) -> float:
        """Cumulative log(L(H1) / L(H0))."""
        return self._log_lr_cumulative

    @property
    def markov(self) -> BayesianMarkovUpdater:
        return self._markov

    def update(self, action: int) -> float:
        """Process a new observation and return updated pi_n.

        Parameters
        ----------
        action : int
            0 or 1.

        Returns
        -------
        float
            Updated posterior probability of H1.
        """
        # Likelihood under H0: IID Bernoulli(P)
        p = self.baseline_prob
        if action == 1:
            log_l_h0 = math.log(max(p, 1e-15))
        else:
            log_l_h0 = math.log(max(1.0 - p, 1e-15))

        # Likelihood under H1: Markov(1) predictive
        log_l_h1 = self._markov.log_likelihood_observation(action)

        # Update the Markov model *after* computing its likelihood
        self._markov.update(action)
        self._n_obs += 1

        # Accumulate log likelihood ratio
        log_lr = log_l_h1 - log_l_h0
        self._log_lr_cumulative += log_lr

        # Bayesian update of pi_n
        pi = self._pi
        if pi <= 0.0:
            self._pi = 0.0
            return self._pi
        if pi >= 1.0:
            self._pi = 1.0
            return self._pi

        log_odds = math.log(pi / (1.0 - pi)) + log_lr
        log_odds = max(min(log_odds, 20.0), -20.0)
        self._pi = 1.0 / (1.0 + math.exp(-log_odds))

        return self._pi

    def reset(self, baseline_prob: Optional[float] = None) -> None:
        """Reset detection state (e.g. after a changepoint)."""
        if baseline_prob is not None:
            self.baseline_prob = baseline_prob
        self._pi = self.prior_h1
        self._log_lr_cumulative = 0.0
        self._n_obs = 0
        self._markov.reset(baseline_prob=self.baseline_prob)

    def interpret(self) -> str:
        """Human-readable interpretation of current pi_n."""
        pi = self._pi
        if pi < 0.3:
            return "Consistent with randomness -- no sequential pattern detected"
        elif pi < 0.6:
            return "Weak signal of sequential dependence -- more data needed"
        elif pi < 0.85:
            return "Probable pattern -- sequential dependencies detected"
        else:
            return "High-confidence pattern detected"
