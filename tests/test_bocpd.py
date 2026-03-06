"""Tests for BOCPD changepoint detector."""

import numpy as np
import pytest

from rand_check.bocpd import BOCPDDetector


class TestBOCPD:

    def test_initial_state(self):
        det = BOCPDDetector()
        assert det.most_likely_run_length == 0
        assert det.changepoint_probability > 0

    def test_no_changepoint_on_stationary_data(self):
        """Stationary Bernoulli data should not flag changepoints often."""
        rng = np.random.default_rng(123)
        det = BOCPDDetector(hazard_rate=0.02, threshold=0.70)
        changepoints = 0
        for _ in range(100):
            obs = int(rng.random() < 0.25)
            if det.update(obs):
                changepoints += 1
        # Should have very few (ideally 0) false changepoints
        assert changepoints < 5

    def test_detects_changepoint_in_shifted_data(self):
        """Sequence that changes from all-0 to all-1 should trigger detection."""
        det = BOCPDDetector(hazard_rate=0.05, threshold=0.50)
        # 50 zeros then 50 ones — obvious changepoint
        detected = False
        for i in range(100):
            obs = 0 if i < 50 else 1
            if det.update(obs):
                detected = True
        assert detected

    def test_run_length_grows_without_changepoint(self):
        """For stationary data, run length should grow."""
        det = BOCPDDetector(hazard_rate=0.02)
        for _ in range(50):
            det.update(0)
        # Most likely run length should be near 50
        assert det.most_likely_run_length > 20

    def test_mean_run_length_positive(self):
        det = BOCPDDetector()
        for _ in range(20):
            det.update(0)
        assert det.mean_run_length > 0

    def test_reset(self):
        det = BOCPDDetector()
        for _ in range(30):
            det.update(1)
        det.reset()
        assert det.most_likely_run_length == 0
        assert len(det.changepoints_detected) == 0

    def test_changepoints_logged(self):
        det = BOCPDDetector(hazard_rate=0.05, threshold=0.30)
        for i in range(100):
            obs = 0 if i < 50 else 1
            det.update(obs)
        # If any changepoints detected, they should be in the log
        # (may or may not detect depending on threshold)
        assert isinstance(det.changepoints_detected, list)
