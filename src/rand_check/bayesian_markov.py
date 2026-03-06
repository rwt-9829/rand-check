"""Bayesian Markov(1) updater with conjugate Beta-Binomial priors.

This is the core live-update engine.  After every hand it performs an O(1)
update of transition counts and returns the posterior predictive probability
of the next action being a 3bet, conditioned on the most recent action.

The prior structure encodes known human cognitive biases:
  - Alternation bias: δ ≈ 0.10 added to P(switch) transitions
  - Run-aversion:   δ subtracted from P(repeat) transitions

The prior *strength* κ controls how many pseudo-observations the prior
is worth.  κ = 15 means "the prior dominates for the first ~15 hands,
then data takes over" — exactly matching the design spec.

Mathematical basis:
    P(x_n = 1 | x_{n-1} = s, data) = (n_{s,1} + α_{s,1}) /
                                       (n_{s,0} + n_{s,1} + α_{s,0} + α_{s,1})

    where  α_{s,j} = prior pseudo-count for transition (s → j)
           n_{s,j} = observed count of transition   (s → j)

    This is the standard Beta-Binomial conjugate posterior predictive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from scipy.stats import beta as beta_dist


@dataclass
class BayesianMarkovUpdater:
    """Markov(1) model with informative Beta priors for 3bet prediction.

    Parameters
    ----------
    gto_prob : float
        The GTO baseline 3bet probability P for the current spot.
    kappa : float
        Prior pseudo-count strength.  Larger → prior dominates longer.
    delta : float
        Alternation bias offset.  Humans over-alternate by roughly this much.
    """
    gto_prob: float = 0.25
    kappa: float = 15.0
    delta: float = 0.10

    # ── Internal state ───────────────────────────────────────────────
    # Transition counts  n[prev_action][curr_action]
    _counts: list = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
    _last_action: Optional[int] = field(default=None, repr=False)
    _n_updates: int = field(default=0, repr=False)

    # Prior hyperparameters (set in __post_init__)
    _alpha: list = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])

    def __post_init__(self) -> None:
        self._setup_priors()

    def _setup_priors(self) -> None:
        """Compute informative Beta prior hyperparameters.

        Given GTO prob P, pseudo-count strength κ, and alternation bias δ:

        From state 0 (prev = fold/call):
            α_{0,1} = P · κ            (base rate of 3betting)
            α_{0,0} = (1 - P) · κ

        From state 1 (prev = 3bet):
            α_{1,0} = (1 - P + δ) · κ  (humans more likely to switch away)
            α_{1,1} = (P - δ) · κ      (humans less likely to repeat 3bet)
        """
        p = self.gto_prob
        k = self.kappa
        d = self.delta

        # Clamp (P - δ) so prior pseudo-count stays positive
        p_minus_d = max(p - d, 0.01)
        one_minus_p_plus_d = min(1.0 - p + d, 0.99)

        self._alpha = [
            [(1.0 - p) * k, p * k],            # from state 0
            [one_minus_p_plus_d * k, p_minus_d * k],  # from state 1
        ]

    def reset(self, gto_prob: Optional[float] = None) -> None:
        """Soft reset: keep priors, discard observed counts.

        Optionally update the GTO probability (e.g. after a position change).
        """
        self._counts = [[0.0, 0.0], [0.0, 0.0]]
        self._last_action = None
        self._n_updates = 0
        if gto_prob is not None:
            self.gto_prob = gto_prob
            self._setup_priors()

    def update(self, action: int) -> None:
        """Incorporate a new observation.

        Parameters
        ----------
        action : int
            0 = fold/call, 1 = 3bet.
        """
        if self._last_action is not None:
            s = self._last_action
            self._counts[s][action] += 1.0
        self._last_action = action
        self._n_updates += 1

    # ── Posterior predictive queries ─────────────────────────────────

    def predict(self, given_last: Optional[int] = None) -> float:
        """Return P(next = 3bet | last action, data, priors).

        If *given_last* is supplied it overrides the internally tracked
        last action (useful for hypothetical queries).
        """
        s = given_last if given_last is not None else self._last_action
        if s is None:
            # No history yet → use marginal GTO prior
            return self.gto_prob

        n0 = self._counts[s][0]
        n1 = self._counts[s][1]
        a0 = self._alpha[s][0]
        a1 = self._alpha[s][1]

        return (n1 + a1) / (n0 + n1 + a0 + a1)

    def credible_interval(self, level: float = 0.95, given_last: Optional[int] = None) -> Tuple[float, float]:
        """Posterior credible interval for P(3bet | last action).

        Uses the Beta posterior directly.
        """
        s = given_last if given_last is not None else self._last_action
        if s is None:
            # Uninformative — wide interval
            tail = (1.0 - level) / 2.0
            return (tail, 1.0 - tail)

        alpha_post = self._counts[s][1] + self._alpha[s][1]
        beta_post = self._counts[s][0] + self._alpha[s][0]

        tail = (1.0 - level) / 2.0
        lo = float(beta_dist.ppf(tail, alpha_post, beta_post))
        hi = float(beta_dist.ppf(1.0 - tail, alpha_post, beta_post))
        return (lo, hi)

    def log_likelihood_observation(self, action: int) -> float:
        """Log-likelihood of *action* under the Markov(1) predictive model.

        Used by the detection engine to compute the likelihood ratio.
        """
        p = self.predict()
        if action == 1:
            return math.log(max(p, 1e-15))
        else:
            return math.log(max(1.0 - p, 1e-15))

    @property
    def transition_matrix(self) -> list:
        """Current posterior transition matrix (2×2).

        Element [i][j] = P(curr = j | prev = i).
        """
        mat = []
        for s in (0, 1):
            n0 = self._counts[s][0]
            n1 = self._counts[s][1]
            a0 = self._alpha[s][0]
            a1 = self._alpha[s][1]
            total = n0 + n1 + a0 + a1
            mat.append([
                (n0 + a0) / total,
                (n1 + a1) / total,
            ])
        return mat

    @property
    def counts_total(self) -> float:
        return sum(self._counts[s][a] for s in (0, 1) for a in (0, 1))

    @property
    def effective_sample_size(self) -> float:
        """Total observations + prior pseudo-counts."""
        return self.counts_total + 2.0 * self.kappa
