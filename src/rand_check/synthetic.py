"""Synthetic data generators for calibration and validation.

Generates binary sequences from known cognitive models so the system
can be calibrated before touching real data.

Generators:
  1. IID Bernoulli(P) — true null (H₀).
  2. Markov(1) with alternation bias — the simplest human model.
  3. Gambler's fallacy — P(3bet) increases after long runs of non-3bet.
  4. Counter model — player self-corrects toward perceived target freq.
  5. Mixture model — some % of the time genuine RNG, rest is biased.
  6. Run-averse model — hard cap on consecutive same-actions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GeneratedSequence:
    """A synthetic binary sequence with known ground truth."""
    sequence: list
    model_name: str
    gto_prob: float
    is_human: bool    # True = human model (H₁), False = IID (H₀)
    parameters: dict  # Model-specific parameters for reproducibility


def generate_iid_bernoulli(
    n: int,
    p: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedSequence:
    """Pure IID Bernoulli(P) — the null hypothesis H₀."""
    rng = rng or np.random.default_rng()
    seq = rng.binomial(1, p, size=n).tolist()
    return GeneratedSequence(
        sequence=seq,
        model_name="iid_bernoulli",
        gto_prob=p,
        is_human=False,
        parameters={"p": p},
    )


def generate_markov_alternation(
    n: int,
    p: float = 0.25,
    alternation_rate: float = 0.60,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedSequence:
    """Markov(1) chain with alternation bias.

    Parameters
    ----------
    alternation_rate : float
        Overall probability of switching (human range: 0.55–0.70).
        Under IID Bernoulli(0.25) the expected alternation rate is
        2·0.25·0.75 = 0.375.  Humans produce 0.55–0.70.
    """
    rng = rng or np.random.default_rng()

    # Derive transition probabilities from alternation_rate and
    # the constraint that the stationary distribution ≈ (1-p, p).
    # P(switch from 0→1) = p01
    # P(switch from 1→0) = p10
    # Stationary: π₁ = p01 / (p01 + p10)
    # Alternation rate = π₀ · p01 + π₁ · p10
    # ≈ (1-p)·p01 + p·p10

    # Simple parametrization: scale both switch probs by alternation_rate/expected
    expected_alt = 2.0 * p * (1.0 - p)
    scale = alternation_rate / max(expected_alt, 0.01)
    p01 = min(p * scale, 0.95)
    p10 = min((1.0 - p) * scale, 0.95)

    seq = [rng.binomial(1, p)]
    for _ in range(1, n):
        if seq[-1] == 0:
            seq.append(1 if rng.random() < p01 else 0)
        else:
            seq.append(0 if rng.random() < p10 else 1)

    return GeneratedSequence(
        sequence=seq,
        model_name="markov_alternation",
        gto_prob=p,
        is_human=True,
        parameters={"p": p, "alternation_rate": alternation_rate, "p01": p01, "p10": p10},
    )


def generate_gamblers_fallacy(
    n: int,
    p: float = 0.25,
    pressure_rate: float = 0.05,
    max_pressure: float = 0.40,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedSequence:
    """Gambler's fallacy model.

    After k consecutive non-3bets, the probability of 3betting increases:
        P(3bet | last k = fold) = min(p + k · pressure_rate, max_pressure)

    After a 3bet, the probability resets to (p − offset) to model
    "I just 3bet, I shouldn't do it again right away."
    """
    rng = rng or np.random.default_rng()

    seq: list[int] = []
    consecutive_folds = 0

    for _ in range(n):
        if consecutive_folds > 0:
            prob = min(p + consecutive_folds * pressure_rate, max_pressure)
        else:
            prob = max(p * 0.6, 0.02)  # suppressed after recent 3bet

        action = 1 if rng.random() < prob else 0
        seq.append(action)

        if action == 0:
            consecutive_folds += 1
        else:
            consecutive_folds = 0

    return GeneratedSequence(
        sequence=seq,
        model_name="gamblers_fallacy",
        gto_prob=p,
        is_human=True,
        parameters={"p": p, "pressure_rate": pressure_rate, "max_pressure": max_pressure},
    )


def generate_counter_model(
    n: int,
    p: float = 0.25,
    window: int = 15,
    correction_strength: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedSequence:
    """Counter model — player mentally tracks frequency and self-corrects.

    The player maintains a rough mental count over the last `window`
    hands, and adjusts their 3bet probability to correct perceived
    over- or under-3betting toward their target frequency.

    P(3bet) = P + correction_strength · (P − observed_freq_in_window)
    """
    rng = rng or np.random.default_rng()

    seq: list[int] = []
    for i in range(n):
        if i < 3:
            prob = p
        else:
            recent = seq[max(0, i - window):i]
            observed_freq = sum(recent) / len(recent)
            correction = correction_strength * (p - observed_freq)
            prob = max(0.02, min(0.95, p + correction))

        action = 1 if rng.random() < prob else 0
        seq.append(action)

    return GeneratedSequence(
        sequence=seq,
        model_name="counter",
        gto_prob=p,
        is_human=True,
        parameters={"p": p, "window": window, "correction_strength": correction_strength},
    )


def generate_run_averse(
    n: int,
    p: float = 0.25,
    max_consecutive: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedSequence:
    """Run-averse model — never does the same thing more than K times in a row.

    After max_consecutive identical actions, forces a switch.
    """
    rng = rng or np.random.default_rng()

    seq: list[int] = []
    for i in range(n):
        if i >= max_consecutive:
            recent = seq[-max_consecutive:]
            if all(x == 1 for x in recent):
                seq.append(0)  # force fold after K 3bets
                continue
            elif all(x == 0 for x in recent):
                seq.append(1)  # force 3bet after K folds
                continue

        action = 1 if rng.random() < p else 0
        seq.append(action)

    return GeneratedSequence(
        sequence=seq,
        model_name="run_averse",
        gto_prob=p,
        is_human=True,
        parameters={"p": p, "max_consecutive": max_consecutive},
    )


def generate_mixture(
    n: int,
    p: float = 0.25,
    rng_fraction: float = 0.30,
    human_alternation: float = 0.60,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedSequence:
    """Mixture model — some fraction of decisions use genuine RNG.

    With probability rng_fraction, the action is Bernoulli(P).
    Otherwise, it follows the Markov alternation model.
    """
    rng = rng or np.random.default_rng()

    # Pre-generate the human Markov chain
    human = generate_markov_alternation(n, p, human_alternation, rng)

    seq: list[int] = []
    for i in range(n):
        if rng.random() < rng_fraction:
            # Genuine RNG
            seq.append(1 if rng.random() < p else 0)
        else:
            # Human pattern
            seq.append(human.sequence[i])

    return GeneratedSequence(
        sequence=seq,
        model_name="mixture",
        gto_prob=p,
        is_human=True,
        parameters={"p": p, "rng_fraction": rng_fraction, "human_alternation": human_alternation},
    )


def generate_changepoint(
    n: int,
    p: float = 0.25,
    changepoint_at: Optional[int] = None,
    pre_alternation: float = 0.60,
    post_alternation: float = 0.40,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedSequence:
    """Sequence with a mid-session style change (changepoint).

    Before the changepoint: strong alternation bias.
    After: reduced alternation (or even streaky behaviour).
    """
    rng = rng or np.random.default_rng()
    if changepoint_at is None:
        changepoint_at = n // 2

    pre = generate_markov_alternation(changepoint_at, p, pre_alternation, rng)
    post = generate_markov_alternation(n - changepoint_at, p, post_alternation, rng)

    return GeneratedSequence(
        sequence=pre.sequence + post.sequence,
        model_name="changepoint",
        gto_prob=p,
        is_human=True,
        parameters={
            "p": p,
            "changepoint_at": changepoint_at,
            "pre_alternation": pre_alternation,
            "post_alternation": post_alternation,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Batch generation for calibration
# ═══════════════════════════════════════════════════════════════════════

def generate_calibration_dataset(
    n_per_model: int = 100,
    seq_length: int = 80,
    gto_prob: float = 0.25,
    seed: int = 42,
) -> list[GeneratedSequence]:
    """Generate a balanced calibration dataset with all models.

    Returns n_per_model sequences from each generator (7 generators):
    - IID Bernoulli (H₀)
    - Markov alternation (H₁)
    - Gambler's fallacy (H₁)
    - Counter model (H₁)
    - Run-averse (H₁)
    - Mixture (H₁)
    - Changepoint (H₁)
    """
    rng = np.random.default_rng(seed)
    dataset: list[GeneratedSequence] = []

    for _ in range(n_per_model):
        dataset.append(generate_iid_bernoulli(seq_length, gto_prob, rng))

    for _ in range(n_per_model):
        alt = rng.uniform(0.55, 0.70)
        dataset.append(generate_markov_alternation(seq_length, gto_prob, alt, rng))

    for _ in range(n_per_model):
        pr = rng.uniform(0.03, 0.08)
        dataset.append(generate_gamblers_fallacy(seq_length, gto_prob, pr, rng=rng))

    for _ in range(n_per_model):
        cs = rng.uniform(1.5, 3.0)
        dataset.append(generate_counter_model(seq_length, gto_prob, correction_strength=cs, rng=rng))

    for _ in range(n_per_model):
        mc = rng.choice([2, 3])
        dataset.append(generate_run_averse(seq_length, gto_prob, int(mc), rng))

    for _ in range(n_per_model):
        rf = rng.uniform(0.20, 0.40)
        dataset.append(generate_mixture(seq_length, gto_prob, rf, rng=rng))

    for _ in range(n_per_model):
        lo, hi = 20, seq_length - 20
        if lo >= hi:
            # Sequence too short for randomized changepoint — use midpoint
            cp = seq_length // 2
        else:
            cp = rng.integers(lo, hi)
        dataset.append(generate_changepoint(seq_length, gto_prob, int(cp), rng=rng))

    return dataset
