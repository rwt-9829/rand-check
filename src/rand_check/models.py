"""Core data models and type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


class Action(Enum):
    """Binary action at a 3bet decision point."""
    FOLD_CALL = 0  # Did not 3bet (fold or flat call)
    THREE_BET = 1  # 3bet


class Position(Enum):
    """Table positions (6-max)."""
    UTG = auto()
    HJ = auto()
    CO = auto()
    BTN = auto()
    SB = auto()
    BB = auto()


@dataclass(frozen=True)
class HandResult:
    """A single observed hand at a 3bet decision point.

    Attributes:
        action: Whether the player 3bet or not.
        position: The player's table position.
        stack_bb: Effective stack in big blinds.
        hand_number: Sequential hand number in the session.
        villain_position: The opener's position (the one facing the 3bet).
    """
    action: Action
    position: Position
    stack_bb: float = 100.0
    hand_number: int = 0
    villain_position: Optional[Position] = None


@dataclass
class DecisionPackage:
    """Output of the prediction engine — the full exploitation-ready package.

    Attributes:
        detection_confidence: π_n ∈ [0,1], posterior probability that the
            player is *not* using true RNG (H₁ = human-generated).
        predicted_3bet_prob: q_n, the adjusted probability to use for
            decision-making, replacing the GTO baseline P.
        gto_3bet_prob: P, the solver-derived baseline 3bet frequency.
        edge: q_n − P, signed deviation from GTO.
        confidence_interval: (q_lo, q_hi) 95 % posterior credible interval.
        changepoint_flag: True if BOCPD detected a recent style shift.
        hands_observed: Total hands processed so far.
        hands_to_reliable: Estimated hands until high-confidence regime (~0).
        exploitation_suggestion: Human-readable recommendation string.
    """
    detection_confidence: float = 0.0
    predicted_3bet_prob: float = 0.0
    gto_3bet_prob: float = 0.0
    edge: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    changepoint_flag: bool = False
    hands_observed: int = 0
    hands_to_reliable: int = 40
    exploitation_suggestion: str = ""


@dataclass
class TransitionCounts:
    """Markov(1) transition counters with Beta priors.

    Maintains four counters n_{ij} where i = previous action, j = current action.
    Also stores the Beta prior hyperparameters α_{ij}, β_{ij}.
    """
    # Observed counts
    n00: float = 0.0  # prev=fold, curr=fold
    n01: float = 0.0  # prev=fold, curr=3bet
    n10: float = 0.0  # prev=3bet, curr=fold
    n11: float = 0.0  # prev=3bet, curr=3bet

    # Beta prior hyperparameters for P(3bet | prev=fold)
    alpha_01: float = 1.0
    beta_01: float = 1.0

    # Beta prior hyperparameters for P(fold | prev=3bet)
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
    detection_posterior: float = 0.5  # π_n, prior = 0.5 (uninformative)
    hands_processed: int = 0
    current_position: Position = Position.BTN
    current_gto_prob: float = 0.25
