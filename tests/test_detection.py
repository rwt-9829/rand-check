"""Tests for the detection engine."""

import pytest

from rand_check.detection import DetectionEngine


class TestDetectionEngine:

    def test_initial_posterior_is_prior(self):
        eng = DetectionEngine(gto_prob=0.25, prior_h1=0.5)
        assert eng.posterior_h1 == 0.5

    def test_iid_sequence_keeps_low_posterior(self):
        """IID Bernoulli data should not drive π_n high."""
        import numpy as np
        rng = np.random.default_rng(42)
        eng = DetectionEngine(gto_prob=0.25, prior_h1=0.5)
        for _ in range(50):
            action = int(rng.random() < 0.25)
            eng.update(action)
        # π_n should not be very high for genuinely random data
        # (may fluctuate, but shouldn't consistently be > 0.9)
        assert eng.posterior_h1 < 0.95

    def test_alternating_sequence_raises_posterior(self):
        """Strongly alternating data should increase π_n toward 1."""
        eng = DetectionEngine(gto_prob=0.25, prior_h1=0.5)
        for _ in range(30):
            eng.update(0)
            eng.update(1)
        # Strong alternation → human signal
        assert eng.posterior_h1 > 0.5

    def test_reset_restores_prior(self):
        eng = DetectionEngine(gto_prob=0.25, prior_h1=0.5)
        for i in range(20):
            eng.update(i % 2)
        assert eng.posterior_h1 != 0.5
        eng.reset()
        assert eng.posterior_h1 == 0.5

    def test_interpret_low(self):
        eng = DetectionEngine(gto_prob=0.25, prior_h1=0.1)
        assert "genuine RNG" in eng.interpret()

    def test_interpret_high(self):
        eng = DetectionEngine(gto_prob=0.25, prior_h1=0.9)
        assert "High confidence" in eng.interpret()

    def test_log_likelihood_ratio_updates(self):
        eng = DetectionEngine(gto_prob=0.25)
        # Feed an alternating pattern — enough to diverge from IID
        for _ in range(10):
            eng.update(0)
            eng.update(1)
        # LLR should be non-zero after enough data
        assert eng.log_likelihood_ratio != 0.0

    def test_posterior_bounded_0_1(self):
        eng = DetectionEngine(gto_prob=0.25)
        for _ in range(100):
            eng.update(0)
            eng.update(1)
        assert 0.0 <= eng.posterior_h1 <= 1.0
