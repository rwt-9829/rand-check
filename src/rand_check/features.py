"""Post-session feature extraction suite.

Computes the six diagnostic features from the full binary sequence:

  1. Log Likelihood Ratio (LLR)  — Markov(1) MLE vs IID Bernoulli(P)
  2. Alternation Rate Deviation  — ΔA = A − 2P(1−P)
  3. Run Length Score             — −Σ log P(run ≥ k | Bernoulli)
  4. Normalized LZ Complexity     — LZC / LZC_expected
  5. Frequency Drift              — variance of windowed 3bet frequency
  6. Lag-1 Serial Correlation     — Corr(x_i, x_{i−1})

All features are designed to detect human-generated departures from IID
Bernoulli randomness, even at small sample sizes (n ≈ 30–100).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureVector:
    """The six detection features computed from a full session."""
    log_likelihood_ratio: float
    alternation_deviation: float
    run_length_score: float
    normalized_lz_complexity: float
    frequency_drift: float
    serial_correlation: float
    n_observations: int

    def as_dict(self) -> dict[str, float]:
        return {
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "alternation_deviation": self.alternation_deviation,
            "run_length_score": self.run_length_score,
            "normalized_lz_complexity": self.normalized_lz_complexity,
            "frequency_drift": self.frequency_drift,
            "serial_correlation": self.serial_correlation,
        }

    def detection_summary(self, gto_prob: float, detailed: bool = False) -> str:
        """Human-readable summary of which biases are detected.

        Parameters
        ----------
        gto_prob : float
            The GTO baseline probability.
        detailed : bool
            If True, show technical metric names. If False, use plain English.
        """
        if detailed:
            return self._technical_summary(gto_prob)
        return self._friendly_summary(gto_prob)

    def _friendly_summary(self, gto_prob: float) -> str:
        """Plain-English feature summary."""
        lines = []
        lines.append(f"  PATTERN ANALYSIS ({self.n_observations} hands, baseline: {gto_prob * 100:.1f}%)")
        lines.append("  " + "-" * 50)

        findings = []

        # Alternation
        if self.alternation_deviation > 0.05:
            findings.append(
                f"  Alternation:    Too much switching back and forth "
                f"(+{self.alternation_deviation:.1%} above expected)"
            )
        elif self.alternation_deviation < -0.05:
            findings.append(
                f"  Alternation:    Too streaky -- tends to repeat the same action "
                f"({self.alternation_deviation:.1%} below expected)"
            )
        else:
            findings.append(f"  Alternation:    Normal")

        # Serial correlation
        if self.serial_correlation < -0.1:
            findings.append(
                f"  Action link:    Each action tends to be the OPPOSITE of the last one"
            )
        elif self.serial_correlation > 0.1:
            findings.append(
                f"  Action link:    Each action tends to REPEAT the last one"
            )
        else:
            findings.append(f"  Action link:    Actions look independent of each other")

        # Run length
        if self.run_length_score > 5.0:
            findings.append(
                f"  Streaks:        Runs of same action are shorter than random would produce"
            )
        else:
            findings.append(f"  Streaks:        Normal streak lengths")

        # LZ complexity
        if self.normalized_lz_complexity < 0.85:
            findings.append(
                f"  Complexity:     Sequence is more predictable/structured than random"
            )
        elif self.normalized_lz_complexity > 1.05:
            findings.append(
                f"  Complexity:     Sequence is unusually complex"
            )
        else:
            findings.append(f"  Complexity:     Normal")

        # LLR
        if self.log_likelihood_ratio > 1.0:
            findings.append(
                f"  Pattern signal: Strong evidence of sequential patterns"
            )
        elif self.log_likelihood_ratio > 0.0:
            findings.append(
                f"  Pattern signal: Mild evidence of sequential patterns"
            )
        else:
            findings.append(
                f"  Pattern signal: Consistent with random play"
            )

        lines.extend(findings)
        return "\n".join(lines)

    def _technical_summary(self, gto_prob: float) -> str:
        """Technical summary with metric abbreviations."""
        lines = [f"  Session Feature Analysis (n={self.n_observations}, P={gto_prob:.3f})"]
        lines.append("  " + "=" * 60)

        # LLR
        if self.log_likelihood_ratio > 1.0:
            lines.append(f"    LLR = {self.log_likelihood_ratio:+.2f}  <- Strong Markov signal")
        elif self.log_likelihood_ratio > 0.0:
            lines.append(f"    LLR = {self.log_likelihood_ratio:+.2f}  <- Mild Markov signal")
        else:
            lines.append(f"    LLR = {self.log_likelihood_ratio:+.2f}  <- Consistent with IID")

        # Alternation
        if self.alternation_deviation > 0.05:
            lines.append(f"    dA  = {self.alternation_deviation:+.3f}  <- Over-alternation (human bias)")
        elif self.alternation_deviation < -0.05:
            lines.append(f"    dA  = {self.alternation_deviation:+.3f}  <- Under-alternation (streaky)")
        else:
            lines.append(f"    dA  = {self.alternation_deviation:+.3f}  <- Normal range")

        # Run length
        lines.append(f"    RLS = {self.run_length_score:.2f}")
        if self.run_length_score > 5.0:
            lines[-1] += "  <- Runs shorter than expected (human)"

        # LZ complexity
        if self.normalized_lz_complexity < 0.85:
            lines.append(f"    LZC = {self.normalized_lz_complexity:.3f}  <- Compressible (patterned)")
        elif self.normalized_lz_complexity > 1.05:
            lines.append(f"    LZC = {self.normalized_lz_complexity:.3f}  <- Over-complex")
        else:
            lines.append(f"    LZC = {self.normalized_lz_complexity:.3f}  <- Normal complexity")

        # Frequency drift
        lines.append(f"    FD  = {self.frequency_drift:.4f}")

        # Serial correlation
        if self.serial_correlation < -0.1:
            lines.append(f"    r1  = {self.serial_correlation:+.3f}  <- Negative autocorrelation (alternation)")
        elif self.serial_correlation > 0.1:
            lines.append(f"    r1  = {self.serial_correlation:+.3f}  <- Positive autocorrelation (streaky)")
        else:
            lines.append(f"    r1  = {self.serial_correlation:+.3f}  <- Near zero (IID-consistent)")

        return "\n".join(lines)


def compute_features(sequence: list[int], gto_prob: float) -> FeatureVector:
    """Compute all six features from a binary sequence.

    Parameters
    ----------
    sequence : list[int]
        Binary sequence of actions (0 = fold/call, 1 = 3bet).
    gto_prob : float
        The GTO baseline probability P.

    Returns
    -------
    FeatureVector
    """
    n = len(sequence)
    if n < 3:
        return FeatureVector(0, 0, 0, 1, 0, 0, n)

    arr = np.array(sequence, dtype=float)
    p = gto_prob

    llr = _log_likelihood_ratio(sequence, p)
    alt_dev = _alternation_deviation(sequence, p)
    rls = _run_length_score(sequence, p)
    lzc = _normalized_lz_complexity(sequence, n)
    fd = _frequency_drift(arr, window=20)
    sc = _serial_correlation(arr)

    return FeatureVector(
        log_likelihood_ratio=llr,
        alternation_deviation=alt_dev,
        run_length_score=rls,
        normalized_lz_complexity=lzc,
        frequency_drift=fd,
        serial_correlation=sc,
        n_observations=n,
    )


# ═══════════════════════════════════════════════════════════════════════
# Feature 1: Log Likelihood Ratio
# ═══════════════════════════════════════════════════════════════════════

def _log_likelihood_ratio(seq: list[int], p: float) -> float:
    """LLR = log P(data | Markov(1) MLE) − log P(data | IID Bernoulli(P)).

    Positive → Markov model fits better → human-generated signal.
    """
    n = len(seq)
    if n < 2:
        return 0.0

    # IID log-likelihood
    k = sum(seq)
    log_iid = k * math.log(max(p, 1e-15)) + (n - k) * math.log(max(1.0 - p, 1e-15))

    # Markov(1) MLE log-likelihood
    counts = [[0, 0], [0, 0]]
    for i in range(1, n):
        counts[seq[i - 1]][seq[i]] += 1

    log_markov = 0.0
    for s in (0, 1):
        total = counts[s][0] + counts[s][1]
        if total == 0:
            continue
        for a in (0, 1):
            if counts[s][a] > 0:
                p_sa = counts[s][a] / total
                log_markov += counts[s][a] * math.log(p_sa)

    return log_markov - log_iid


# ═══════════════════════════════════════════════════════════════════════
# Feature 2: Alternation Rate Deviation
# ═══════════════════════════════════════════════════════════════════════

def _alternation_deviation(seq: list[int], p: float) -> float:
    """ΔA = observed_alternation_rate − expected_alternation_rate.

    Expected under IID Bernoulli(P): A* = 2P(1−P).
    Humans produce ΔA > 0 (over-alternate).
    """
    n = len(seq)
    if n < 2:
        return 0.0

    alternations = sum(1 for i in range(1, n) if seq[i] != seq[i - 1])
    observed = alternations / (n - 1)
    expected = 2.0 * p * (1.0 - p)
    return observed - expected


# ═══════════════════════════════════════════════════════════════════════
# Feature 3: Run Length Score
# ═══════════════════════════════════════════════════════════════════════

def _run_length_score(seq: list[int], p: float) -> float:
    """Score = −Σ log P(run ≥ k | IID Bernoulli(P)).

    Higher score → runs are shorter than expected → human-generated.
    """
    if len(seq) < 2:
        return 0.0

    runs = _extract_runs(seq)
    score = 0.0
    for value, length in runs:
        # P(run of value ≥ k) = q^k where q = p if value=1, (1-p) otherwise
        q = p if value == 1 else (1.0 - p)
        if q <= 0 or q >= 1:
            continue
        # −log P(run ≥ length) = −length · log(q)
        score -= length * math.log(q)

    return score


def _extract_runs(seq: list[int]) -> list[tuple[int, int]]:
    """Extract (value, length) pairs for each run in the sequence."""
    if not seq:
        return []
    runs: list[tuple[int, int]] = []
    current_val = seq[0]
    current_len = 1
    for i in range(1, len(seq)):
        if seq[i] == current_val:
            current_len += 1
        else:
            runs.append((current_val, current_len))
            current_val = seq[i]
            current_len = 1
    runs.append((current_val, current_len))
    return runs


# ═══════════════════════════════════════════════════════════════════════
# Feature 4: Normalized Lempel-Ziv Complexity
# ═══════════════════════════════════════════════════════════════════════

def _lz_complexity(seq: list[int]) -> int:
    """Compute the Lempel-Ziv complexity C(S) of a binary sequence.

    Implements the algorithm from Lempel & Ziv (1976).
    """
    n = len(seq)
    if n == 0:
        return 0
    if n == 1:
        return 1

    i = 0
    c = 1
    u = 1
    v = 1
    v_max = v

    while u + v <= n:
        if seq[i + v - 1] == seq[u + v - 1]:  # 0-indexed adjustment
            v += 1
        else:
            v_max = max(v, v_max)
            i += 1
            if i == u:
                c += 1
                u += v_max
                v = 1
                i = 0
                v_max = v
            else:
                v = 1

    if v != 1:
        c += 1

    return c


def _normalized_lz_complexity(seq: list[int], n: int) -> float:
    """LZC normalized to [0, ~1] where 1 ≈ truly random.

    Normalization: C(S) / (n / log₂(n))
    For a random binary string, LZC ≈ n / log₂(n).
    """
    if n < 3:
        return 1.0

    c = _lz_complexity(seq)
    expected = n / math.log2(n)
    if expected <= 0:
        return 1.0
    return c / expected


# ═══════════════════════════════════════════════════════════════════════
# Feature 5: Frequency Drift
# ═══════════════════════════════════════════════════════════════════════

def _frequency_drift(arr: np.ndarray, window: int = 20) -> float:
    """Variance of local 3bet frequency in sliding windows.

    RNG is stationary → low drift.  Humans drift as their "mental
    account" resets.
    """
    n = len(arr)
    if n < window:
        return 0.0

    windows = n - window + 1
    freqs = np.array([arr[i:i + window].mean() for i in range(windows)])
    return float(np.var(freqs))


# ═══════════════════════════════════════════════════════════════════════
# Feature 6: Lag-1 Serial Correlation
# ═══════════════════════════════════════════════════════════════════════

def _serial_correlation(arr: np.ndarray) -> float:
    """ρ₁ = Corr(x_i, x_{i-1}).

    Expected under IID: 0.
    Humans: ρ₁ < 0 (negative autocorrelation — alternation).
    """
    n = len(arr)
    if n < 3:
        return 0.0

    x = arr[1:]
    y = arr[:-1]
    mean_x = x.mean()
    mean_y = y.mean()
    std_x = x.std()
    std_y = y.std()

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0

    cov = np.mean((x - mean_x) * (y - mean_y))
    return float(cov / (std_x * std_y))
