"""Tests for the validation framework."""

import pytest

from rand_check.validation import ValidationRunner, ValidationMetrics


class TestValidation:

    def test_run_produces_metrics(self):
        runner = ValidationRunner(checkpoints=[20, 40])
        metrics = runner.run(n_per_model=5, seq_length=50, gto_prob=0.25, seed=42)
        assert isinstance(metrics, ValidationMetrics)

    def test_metrics_have_correct_counts(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=5, seq_length=30, gto_prob=0.25, seed=42)
        assert metrics.n_sequences == 35  # 7 models × 5
        assert metrics.n_iid == 5
        assert metrics.n_human == 30

    def test_brier_score_bounded(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=10, seq_length=30, gto_prob=0.25, seed=42)
        assert 0.0 <= metrics.brier_score <= 1.0

    def test_log_loss_positive(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=10, seq_length=30, gto_prob=0.25, seed=42)
        assert metrics.log_loss > 0
        assert metrics.log_loss_baseline > 0

    def test_detection_power_between_0_and_1(self):
        runner = ValidationRunner(checkpoints=[20, 40])
        metrics = runner.run(n_per_model=10, seq_length=50, gto_prob=0.25, seed=42)
        for cp, power in metrics.detection_power.items():
            assert 0.0 <= power <= 1.0

    def test_false_positive_rate_between_0_and_1(self):
        runner = ValidationRunner(checkpoints=[20, 40])
        metrics = runner.run(n_per_model=10, seq_length=50, gto_prob=0.25, seed=42)
        for cp, fpr in metrics.false_positive_rate.items():
            assert 0.0 <= fpr <= 1.0

    def test_summary_string(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=5, seq_length=30, gto_prob=0.25, seed=42)
        summary = metrics.summary()
        assert "Brier score" in summary
        assert "Detection Power" in summary
