"""Tests for synthetic data generators."""

import numpy as np
import pytest

from rand_check.synthetic import (
    GeneratedSequence,
    generate_iid_bernoulli,
    generate_markov_alternation,
    generate_gamblers_fallacy,
    generate_counter_model,
    generate_run_averse,
    generate_mixture,
    generate_changepoint,
    generate_calibration_dataset,
)


def _alternation_rate(sequence: list[int]) -> float:
    return sum(
        1 for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1]
    ) / (len(sequence) - 1)


def _p_one_after_zero_run(sequence: list[int], run_length: int) -> float:
    matches = []
    for i in range(run_length, len(sequence)):
        if all(x == 0 for x in sequence[i - run_length:i]):
            matches.append(sequence[i])
    return sum(matches) / len(matches)


class TestSyntheticGenerators:

    def test_iid_bernoulli_length(self):
        seq = generate_iid_bernoulli(100, 0.50)
        assert len(seq.sequence) == 100
        assert seq.model_name == "iid_bernoulli"
        assert not seq.is_human

    def test_iid_bernoulli_frequency(self):
        """Average should be near P over many samples."""
        rng = np.random.default_rng(42)
        seq = generate_iid_bernoulli(10000, 0.50, rng)
        freq = sum(seq.sequence) / len(seq.sequence)
        assert abs(freq - 0.50) < 0.03

    def test_markov_alternation(self):
        seq = generate_markov_alternation(100, 0.50, 0.60)
        assert len(seq.sequence) == 100
        assert seq.is_human
        assert seq.model_name == "markov_alternation"

    def test_markov_alternation_rate(self):
        """Should have higher alternation than IID."""
        rng = np.random.default_rng(42)
        seq = generate_markov_alternation(1000, 0.50, 0.65, rng)
        alt_rate = _alternation_rate(seq.sequence)
        # Should be near the target alternation rate
        assert alt_rate > 0.45

    def test_iid_alternation_rate_stays_near_half(self):
        """IID Bernoulli(.5) should keep alternation near the 0.5 null rate.

        This follows the subjective-randomness literature: over-alternation is
        the human signature, so the IID null should stay near 0.5 on long runs.
        """
        seq = generate_iid_bernoulli(20000, 0.50, np.random.default_rng(42))
        alt_rate = _alternation_rate(seq.sequence)
        assert 0.47 < alt_rate < 0.53

    def test_markov_alternation_tracks_requested_target_rate(self):
        """The alternation generator should honor the requested target rate."""
        target = 0.65
        seq = generate_markov_alternation(20000, 0.50, target, np.random.default_rng(42))
        alt_rate = _alternation_rate(seq.sequence)
        assert abs(alt_rate - target) < 0.04

    def test_gamblers_fallacy(self):
        seq = generate_gamblers_fallacy(100, 0.50)
        assert len(seq.sequence) == 100
        assert seq.is_human

    def test_gamblers_fallacy_increases_reversal_probability_after_longer_zero_runs(self):
        """Law-of-small-numbers style reversal pressure should grow with zero streaks.

        Literature on subjective randomness predicts stronger reversal pressure
        after longer streaks, so P(1 | 000) should exceed P(1 | 00), which in
        turn should exceed P(1 | 0).
        """
        seq = generate_gamblers_fallacy(50000, 0.50, rng=np.random.default_rng(42))
        p_after_0 = _p_one_after_zero_run(seq.sequence, 1)
        p_after_00 = _p_one_after_zero_run(seq.sequence, 2)
        p_after_000 = _p_one_after_zero_run(seq.sequence, 3)

        assert p_after_0 > 0.50
        assert p_after_0 < p_after_00 < p_after_000
        assert p_after_000 > 0.60

    def test_counter_model(self):
        seq = generate_counter_model(100, 0.50)
        assert len(seq.sequence) == 100
        assert seq.is_human

    def test_counter_model_frequency_stable(self):
        """Counter model should keep frequency close to P."""
        rng = np.random.default_rng(42)
        seq = generate_counter_model(500, 0.50, correction_strength=3.0, rng=rng)
        freq = sum(seq.sequence) / len(seq.sequence)
        assert abs(freq - 0.50) < 0.10

    def test_run_averse(self):
        seq = generate_run_averse(100, 0.50, max_consecutive=2)
        assert len(seq.sequence) == 100
        # Check no runs > 2
        run_len = 1
        for i in range(1, len(seq.sequence)):
            if seq.sequence[i] == seq.sequence[i-1]:
                run_len += 1
                assert run_len <= 2 + 1  # +1 because the cap triggers after max_consecutive
            else:
                run_len = 1

    def test_run_averse_forces_immediate_reversal_after_hitting_cap(self):
        seq = generate_run_averse(2000, 0.50, max_consecutive=2, rng=np.random.default_rng(42))
        for i in range(2, len(seq.sequence)):
            if seq.sequence[i - 2] == seq.sequence[i - 1]:
                assert seq.sequence[i] != seq.sequence[i - 1]

    def test_mixture(self):
        seq = generate_mixture(100, 0.50)
        assert len(seq.sequence) == 100
        assert seq.is_human

    def test_changepoint(self):
        seq = generate_changepoint(100, 0.50, changepoint_at=50)
        assert len(seq.sequence) == 100
        assert seq.is_human
        assert seq.parameters["changepoint_at"] == 50

    def test_calibration_dataset(self):
        dataset = generate_calibration_dataset(n_per_model=5, seq_length=30)
        # 7 models x 5 each = 35
        assert len(dataset) == 35
        # Check mix of patterned and IID
        n_human = sum(1 for s in dataset if s.is_human)
        n_iid = sum(1 for s in dataset if not s.is_human)
        assert n_iid == 5
        assert n_human == 30

    def test_deterministic_with_seed(self):
        seq1 = generate_iid_bernoulli(50, 0.50, np.random.default_rng(42))
        seq2 = generate_iid_bernoulli(50, 0.50, np.random.default_rng(42))
        assert seq1.sequence == seq2.sequence
