"""Tests for Context Tree Weighting."""

import math
import pytest

from rand_check.ctw import ContextTreeWeighting


class TestCTW:

    def test_initial_prediction_near_half(self):
        """With no data, KT estimator starts near 0.5."""
        ctw = ContextTreeWeighting(max_depth=3, activation_threshold=0)
        pred = ctw.predict()
        assert 0.3 < pred < 0.7

    def test_prediction_after_all_ones(self):
        """After many 1s, should predict high P(next=1)."""
        ctw = ContextTreeWeighting(max_depth=2, activation_threshold=0)
        for _ in range(30):
            ctw.update(1)
        assert ctw.predict() > 0.7

    def test_prediction_after_all_zeros(self):
        """After many 0s, should predict low P(next=1)."""
        ctw = ContextTreeWeighting(max_depth=2, activation_threshold=0)
        for _ in range(30):
            ctw.update(0)
        assert ctw.predict() < 0.3

    def test_alternating_pattern(self):
        """After 0101..., context depth should pick up the pattern."""
        ctw = ContextTreeWeighting(max_depth=2, activation_threshold=0)
        for _ in range(40):
            ctw.update(0)
            ctw.update(1)
        # Last was 1, so next should be 0 → predict low
        pred = ctw.predict()
        assert pred < 0.45

    def test_is_active_property(self):
        ctw = ContextTreeWeighting(max_depth=3, activation_threshold=60)
        assert not ctw.is_active
        for _ in range(59):
            ctw.update(0)
        assert not ctw.is_active
        ctw.update(0)
        assert ctw.is_active

    def test_reset(self):
        ctw = ContextTreeWeighting(max_depth=2, activation_threshold=0)
        for _ in range(20):
            ctw.update(1)
        ctw.reset()
        assert ctw.n_observations == 0

    def test_prediction_is_valid_probability(self):
        """Prediction should always be in (0, 1)."""
        ctw = ContextTreeWeighting(max_depth=3, activation_threshold=0)
        for val in [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0]:
            ctw.update(val)
            p = ctw.predict()
            assert 0.0 < p < 1.0

    def test_log_loss_is_positive(self):
        ctw = ContextTreeWeighting(max_depth=2, activation_threshold=0)
        for _ in range(10):
            ctw.update(0)
            ctw.update(1)
        assert ctw.log_loss() > 0
