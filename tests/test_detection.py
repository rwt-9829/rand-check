"""Tests for the detection engine."""

import pytest

from rand_check.detection import DetectionEngine


class TestDetectionEngine:

    def test_initial_posterior_is_prior(self):
        eng = DetectionEngine(baseline_prob=0.50, prior_h1=0.5)
        assert eng.posterior_h1 == 0.5

    def test_iid_sequence_keeps_low_posterior(self):
        """IID Bernoulli data should not drive pi_n high."""
        import numpy as np
        rng = np.random.default_rng(42)
        eng = DetectionEngine(baseline_prob=0.50, prior_h1=0.5)
        for _ in range(50):
            action = int(rng.random() < 0.50)
            eng.update(action)
        # pi_n should not be very high for genuinely random data
        assert eng.posterior_h1 < 0.95

    def test_alternating_sequence_raises_posterior(self):
        """Strongly alternating data should increase pi_n toward 1."""
        eng = DetectionEngine(baseline_prob=0.50, prior_h1=0.5)
        for _ in range(30):
            eng.update(0)
            eng.update(1)
        # Strong alternation -> patterned signal
        assert eng.posterior_h1 > 0.5

    def test_reset_restores_prior(self):
        eng = DetectionEngine(baseline_prob=0.50, prior_h1=0.5)
        for i in range(20):
            eng.update(i % 2)
        assert eng.posterior_h1 != 0.5
        eng.reset()
        assert eng.posterior_h1 == 0.5

    def test_interpret_low(self):
        eng = DetectionEngine(baseline_prob=0.50, prior_h1=0.1)
        assert "randomness" in eng.interpret()

    def test_interpret_high(self):
        eng = DetectionEngine(baseline_prob=0.50, prior_h1=0.9)
        assert "High-confidence" in eng.interpret()

    def test_log_likelihood_ratio_updates(self):
        eng = DetectionEngine(baseline_prob=0.50)
        # Feed an alternating pattern -- enough to diverge from IID
        for _ in range(10):
            eng.update(0)
            eng.update(1)
        # LLR should be non-zero after enough data
        assert eng.log_likelihood_ratio != 0.0

    def test_posterior_bounded_0_1(self):
        eng = DetectionEngine(baseline_prob=0.50)
        for _ in range(100):
            eng.update(0)
            eng.update(1)
        assert 0.0 <= eng.posterior_h1 <= 1.0
