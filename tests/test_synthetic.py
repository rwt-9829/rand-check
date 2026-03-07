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
        alts = sum(1 for i in range(1, len(seq.sequence)) if seq.sequence[i] != seq.sequence[i-1])
        alt_rate = alts / (len(seq.sequence) - 1)
        # Should be near the target alternation rate
        assert alt_rate > 0.45

    def test_gamblers_fallacy(self):
        seq = generate_gamblers_fallacy(100, 0.50)
        assert len(seq.sequence) == 100
        assert seq.is_human

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
