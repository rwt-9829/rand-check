"""Core data models and type definitions for binary sequence analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Observation:
    """A single observed binary outcome.

    Attributes:
        action: The observed value (0 or 1).
        index: Sequential observation number in the session.
    """
    action: int
    index: int = 0


@dataclass
class AnalysisPackage:
    """Output of the prediction engine -- the full analysis-ready package.

    Attributes:
        detection_confidence: pi_n in [0,1], posterior probability that the
            sequence is *not* truly random (H1 = patterned).
        predicted_prob: q_n, the adjusted probability of the next outcome
            being 1, replacing the baseline probability.
        baseline_prob: The assumed baseline probability of a 1 under
            the null hypothesis (default 0.5).
        edge: q_n - baseline_prob, signed deviation from baseline.
        confidence_interval: (lo, hi) 95% posterior credible interval.
        changepoint_flag: True if BOCPD detected a recent behaviour shift.
        observations: Total observations processed so far.
        observations_to_reliable: Estimated observations until
            high-confidence regime (~0).
        suggestion: Human-readable recommendation string.
    """
    detection_confidence: float = 0.0
    predicted_prob: float = 0.0
    baseline_prob: float = 0.0
    edge: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    changepoint_flag: bool = False
    observations: int = 0
    observations_to_reliable: int = 40
    suggestion: str = ""


@dataclass
class TransitionCounts:
    """Markov(1) transition counters with Beta priors.

    Maintains four counters n_{ij} where i = previous value, j = current value.
    Also stores the Beta prior hyperparameters.
    """
    # Observed counts
    n00: float = 0.0  # prev=0, curr=0
    n01: float = 0.0  # prev=0, curr=1
    n10: float = 0.0  # prev=1, curr=0
    n11: float = 0.0  # prev=1, curr=1

    # Beta prior hyperparameters for P(1 | prev=0)
    alpha_01: float = 1.0
    beta_01: float = 1.0

    # Beta prior hyperparameters for P(0 | prev=1)
    alpha_10: float = 1.0
    beta_10: float = 1.0

    @property
    def total_transitions(self) -> float:
        return self.n00 + self.n01 + self.n10 + self.n11


@dataclass
class SessionState:
    """Mutable state for a live session tracker.

    This aggregates all running state needed across the various engines.
    """
    history: List[int] = field(default_factory=list)
    transition_counts: TransitionCounts = field(default_factory=TransitionCounts)
    detection_posterior: float = 0.5  # pi_n, prior = 0.5 (uninformative)
    observations_processed: int = 0
    current_baseline_prob: float = 0.5
