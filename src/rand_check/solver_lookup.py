"""GTO solver lookup table for 3bet frequencies.

Maps (position, villain_position, stack_depth_bucket) → GTO 3bet probability P.

These values are representative of well-studied 6-max NL Hold'em solver
solutions at typical stack depths.  A production system would import these
from an actual solver database; here we provide sensible defaults that
cover the most common scenarios.
"""

from __future__ import annotations

from rand_check.models import Position

# ──────────────────────────────────────────────────────────────────────
# GTO 3bet frequencies: (defender_position, opener_position) → P
#
# Values are approximate for 100bb 6-max cash games, derived from
# publicly available solver aggregates.
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_3BET_FREQ: dict[tuple[Position, Position], float] = {
    # vs UTG open
    (Position.HJ, Position.UTG): 0.06,
    (Position.CO, Position.UTG): 0.07,
    (Position.BTN, Position.UTG): 0.09,
    (Position.SB, Position.UTG): 0.08,
    (Position.BB, Position.UTG): 0.07,
    # vs HJ open
    (Position.CO, Position.HJ): 0.08,
    (Position.BTN, Position.HJ): 0.11,
    (Position.SB, Position.HJ): 0.09,
    (Position.BB, Position.HJ): 0.08,
    # vs CO open
    (Position.BTN, Position.CO): 0.14,
    (Position.SB, Position.CO): 0.11,
    (Position.BB, Position.CO): 0.10,
    # vs BTN open
    (Position.SB, Position.BTN): 0.16,
    (Position.BB, Position.BTN): 0.13,
    # vs SB open (from BB)
    (Position.BB, Position.SB): 0.18,
}

# Stack depth adjustments: multiplier on base freq
# Shorter stacks → wider 3bet (more shove equity)
# Deeper stacks → tighter 3bet (more positional disadvantage)
_STACK_DEPTH_MULTIPLIER: dict[str, float] = {
    "short":  1.25,   # < 40bb
    "medium": 1.00,   # 40-120bb
    "deep":   0.85,   # > 120bb
}


def _stack_bucket(stack_bb: float) -> str:
    if stack_bb < 40:
        return "short"
    elif stack_bb <= 120:
        return "medium"
    else:
        return "deep"


def get_gto_3bet_prob(
    defender: Position,
    opener: Position,
    stack_bb: float = 100.0,
) -> float:
    """Look up the GTO 3bet probability for a given spot.

    Parameters
    ----------
    defender : Position
        The player considering the 3bet.
    opener : Position
        The player who opened (raised first).
    stack_bb : float
        Effective stack size in big blinds.

    Returns
    -------
    float
        The solver-derived 3bet probability P ∈ (0, 1).
    """
    base = _DEFAULT_3BET_FREQ.get((defender, opener))
    if base is None:
        # Fallback: if the position combo is unusual or same-position,
        # return a reasonable default.
        base = 0.10

    multiplier = _STACK_DEPTH_MULTIPLIER[_stack_bucket(stack_bb)]
    return min(base * multiplier, 0.60)  # cap at 60 %


def get_default_3bet_prob(position: Position) -> float:
    """Get a position-aggregated default 3bet frequency.

    Useful when the opener's position is unknown.
    """
    _POSITION_DEFAULTS = {
        Position.UTG: 0.05,
        Position.HJ: 0.07,
        Position.CO: 0.10,
        Position.BTN: 0.13,
        Position.SB: 0.14,
        Position.BB: 0.12,
    }
    return _POSITION_DEFAULTS.get(position, 0.10)
