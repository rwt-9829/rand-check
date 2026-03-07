"""Tests for the post-session feature extraction."""

import numpy as np
import pytest

from rand_check.features import (
    FeatureVector,
    compute_features,
    _lz_complexity,
    _extract_runs,
)


class TestFeatures:

    def test_compute_features_returns_feature_vector(self):
        seq = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        fv = compute_features(seq, gto_prob=0.25)
        assert isinstance(fv, FeatureVector)
        assert fv.n_observations == 10

    def test_alternation_deviation_positive_for_alternating(self):
        """Pure alternation should have positive ΔA."""
        seq = [0, 1] * 30
        fv = compute_features(seq, gto_prob=0.25)
        assert fv.alternation_deviation > 0.3

    def test_alternation_deviation_normal_for_iid(self):
        """IID data should have ΔA near 0."""
        rng = np.random.default_rng(42)
        seq = rng.binomial(1, 0.25, size=500).tolist()
        fv = compute_features(seq, gto_prob=0.25)
        assert abs(fv.alternation_deviation) < 0.10

    def test_serial_correlation_negative_for_alternating(self):
        seq = [0, 1] * 30
        fv = compute_features(seq, gto_prob=0.50)
        assert fv.serial_correlation < -0.5

    def test_serial_correlation_positive_for_streaky(self):
        seq = [0] * 20 + [1] * 20
        fv = compute_features(seq, gto_prob=0.50)
        assert fv.serial_correlation > 0.3

    def test_lz_complexity_binary(self):
        # All same → lowest complexity
        assert _lz_complexity([0] * 20) <= 3
        # Alternating → moderate complexity
        alt = [0, 1] * 10
        c_alt = _lz_complexity(alt)
        assert c_alt > 1

    def test_normalized_lzc_near_one_for_random(self):
        rng = np.random.default_rng(42)
        seq = rng.binomial(1, 0.5, size=500).tolist()
        fv = compute_features(seq, gto_prob=0.50)
        assert 0.7 < fv.normalized_lz_complexity < 1.3

    def test_extract_runs(self):
        runs = _extract_runs([0, 0, 1, 1, 1, 0])
        assert runs == [(0, 2), (1, 3), (0, 1)]

    def test_extract_runs_empty(self):
        assert _extract_runs([]) == []

    def test_run_length_score_higher_for_short_runs(self):
        """Sequence with short runs (alternating) should have high RLS."""
        short_runs = [0, 1] * 20
        long_runs = [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10
        fv_short = compute_features(short_runs, gto_prob=0.50)
        fv_long = compute_features(long_runs, gto_prob=0.50)
        # Short runs → higher score (more surprising under IID)
        assert fv_short.run_length_score < fv_long.run_length_score

    def test_llr_positive_for_markov_data(self):
        """Markov data should have positive LLR (Markov fits better than IID)."""
        # Generate alternating (strongly Markov)
        seq = [0, 1] * 40
        fv = compute_features(seq, gto_prob=0.50)
        assert fv.log_likelihood_ratio > 0

    def test_feature_vector_as_dict(self):
        fv = compute_features([0, 1, 0, 1, 0, 0], gto_prob=0.25)
        d = fv.as_dict()
        assert "log_likelihood_ratio" in d
        assert "alternation_deviation" in d
        assert len(d) == 6

    def test_short_sequence_doesnt_crash(self):
        fv = compute_features([0], gto_prob=0.25)
        assert fv.n_observations == 1

    def test_detection_summary_string(self):
        seq = [0, 1] * 20
        fv = compute_features(seq, gto_prob=0.25)
        summary = fv.detection_summary(0.25)
        assert "Alternation" in summary
        assert "Pattern signal" in summary

    def test_detection_summary_detailed(self):
        seq = [0, 1] * 20
        fv = compute_features(seq, gto_prob=0.25)
        summary = fv.detection_summary(0.25, detailed=True)
        assert "LLR" in summary
        assert "dA" in summary
