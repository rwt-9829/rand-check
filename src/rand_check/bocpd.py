"""Bayesian Online Changepoint Detection (BOCPD).

Implements the algorithm of Adams & MacKay (2007, arXiv:0710.3742).

The model maintains a posterior distribution over the "run length" r_t —
the number of observations since the last changepoint.  After each
observation a message-passing step computes:

    P(r_t | x_{1:t})

When P(r_t = 0 | x_{1:t}) (i.e., a *new* run just started) exceeds a
threshold, we flag a changepoint.

For our application the underlying predictive model (UPM) within each
run is a Beta-Bernoulli model, and the hazard function (probability of
a changepoint at any step) is constant:

    H(τ) = 1 / expected_run_length

Design choice: ``expected_run_length = 50`` encodes "the subject adjusts
behaviour at most once per ~50 observations".

When a changepoint is detected the prediction engine should soft-reset
its Markov transition counts and detection posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BOCPDDetector:
    """Online changepoint detector with Beta-Bernoulli UPM.

    Parameters
    ----------
    hazard_rate : float
        Constant hazard rate = 1 / expected_run_length.
    threshold : float
        If P(run_length = 0 | data) exceeds this, flag a changepoint.
    prior_a : float
        Beta prior α for the within-run Bernoulli model.
    prior_b : float
        Beta prior β for the within-run Bernoulli model.
    max_run_length : int
        Truncate the run-length distribution at this value to bound
        memory; older runs are absorbed into the tail.
    """
    hazard_rate: float = 0.02   # 1/50
    threshold: float = 0.10
    prior_a: float = 1.0
    prior_b: float = 1.0
    max_run_length: int = 300

    # ── Internal state ───────────────────────────────────────────────
    # _joint[r] ∝ P(r_t = r, x_{1:t})  (un-normalized joint)
    _joint: np.ndarray = field(init=False, repr=False)
    _alphas: np.ndarray = field(init=False, repr=False)  # per-run α accumulators
    _betas: np.ndarray = field(init=False, repr=False)   # per-run β accumulators
    _t: int = field(default=0, init=False)
    _changepoint_log: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._joint = np.zeros(self.max_run_length + 1)
        self._joint[0] = 1.0  # start with run length 0
        self._alphas = np.full(self.max_run_length + 1, self.prior_a)
        self._betas = np.full(self.max_run_length + 1, self.prior_b)

    @property
    def changepoint_probability(self) -> float:
        """P(changepoint at current step) = P(r_t = 0 | data)."""
        total = self._joint.sum()
        if total <= 0:
            return 0.0
        return float(self._joint[0] / total)

    @property
    def most_likely_run_length(self) -> int:
        """The run length with highest posterior probability."""
        return int(np.argmax(self._joint))

    @property
    def mean_run_length(self) -> float:
        """Posterior mean run length."""
        total = self._joint.sum()
        if total <= 0:
            return 0.0
        indices = np.arange(len(self._joint))
        return float(np.dot(indices, self._joint) / total)

    @property
    def changepoints_detected(self) -> list[int]:
        """List of observation numbers where changepoints were flagged."""
        return list(self._changepoint_log)

    def update(self, observation: int) -> bool:
        """Process a new binary observation.

        Parameters
        ----------
        observation : int
            0 or 1.

        Returns
        -------
        bool
            True if a changepoint was detected at this step.
        """
        self._t += 1
        H = self.hazard_rate

        # ── Step 1: Predictive probabilities for each run length ─────
        # P(x_t | r_{t-1}) under Beta-Bernoulli with accumulated counts
        pred_probs = (self._alphas / (self._alphas + self._betas))  # P(x=1 | r)
        if observation == 1:
            pi_arr = pred_probs
        else:
            pi_arr = 1.0 - pred_probs

        # ── Step 2: Growth probabilities ─────────────────────────────
        # Grow each run by 1 (multiply by (1-H) and run-conditioned likelihood)
        growth = self._joint * pi_arr * (1.0 - H)

        # ── Step 3: Changepoint probability ──────────────────────────
        # Sum the mass that goes to run_length = 0 (new run). In standard
        # Adams–MacKay BOCPD this branch uses the segment prior predictive,
        # not the run-conditioned predictive used for growth.
        prior_pred = self.prior_a / (self.prior_a + self.prior_b)
        cp_predictive = prior_pred if observation == 1 else (1.0 - prior_pred)
        changepoint_mass = float(np.sum(self._joint * H) * cp_predictive)

        # ── Step 4: Shift the distribution ───────────────────────────
        new_joint = np.zeros_like(self._joint)
        new_joint[0] = changepoint_mass
        new_joint[1:] = growth[:-1]

        # Update sufficient statistics
        new_alphas = np.zeros_like(self._alphas)
        new_betas = np.zeros_like(self._betas)

        # New run starts at r_t = 0 *after observing x_t*, so its sufficient
        # statistics must already include the current observation.
        new_alphas[0] = self.prior_a + (1.0 if observation == 1 else 0.0)
        new_betas[0] = self.prior_b + (1.0 if observation == 0 else 0.0)

        # Existing runs: accumulate counts
        if observation == 1:
            new_alphas[1:] = self._alphas[:-1] + 1.0
            new_betas[1:] = self._betas[:-1]
        else:
            new_alphas[1:] = self._alphas[:-1]
            new_betas[1:] = self._betas[:-1] + 1.0

        self._joint = new_joint
        self._alphas = new_alphas
        self._betas = new_betas

        # Normalize to prevent underflow (keep relative proportions)
        total = self._joint.sum()
        if total > 0:
            self._joint /= total

        # ── Step 5: Check changepoint ────────────────────────────────
        detected = self._t > 2 and self.changepoint_probability >= self.threshold

        if detected:
            self._changepoint_log.append(self._t)

        return detected

    def reset(self) -> None:
        """Full reset of the detector."""
        self._joint = np.zeros(self.max_run_length + 1)
        self._joint[0] = 1.0
        self._alphas = np.full(self.max_run_length + 1, self.prior_a)
        self._betas = np.full(self.max_run_length + 1, self.prior_b)
        self._t = 0
        self._changepoint_log.clear()
