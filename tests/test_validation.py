"""Tests for the validation framework."""

import numpy as np
import pytest

from rand_check.synthetic import generate_iid_bernoulli, generate_markov_alternation
from rand_check.validation import (
    ValidationRunner,
    ValidationMetrics,
    CalibrationMetrics,
    ChangepointPerformance,
)


def _structured_validation_dataset() -> list:
    """Controlled validation set with a clean literature-style alternation signal.

    Using strong alternation against IID makes the expected power progression and
    proper-score improvements stable enough for regression testing.
    """
    rng = np.random.default_rng(7)
    return [
        *[generate_iid_bernoulli(60, 0.50, rng) for _ in range(12)],
        *[generate_markov_alternation(60, 0.50, 0.68, rng) for _ in range(12)],
    ]


class TestValidation:

    def test_run_produces_metrics(self):
        runner = ValidationRunner(checkpoints=[20, 40])
        metrics = runner.run(n_per_model=5, seq_length=50, baseline_prob=0.50, seed=42)
        assert isinstance(metrics, ValidationMetrics)

    def test_metrics_have_correct_counts(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=5, seq_length=30, baseline_prob=0.50, seed=42)
        assert metrics.n_sequences == 35  # 7 models x 5
        assert metrics.n_iid == 5
        assert metrics.n_human == 30

    def test_brier_score_bounded(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=10, seq_length=30, baseline_prob=0.50, seed=42)
        assert 0.0 <= metrics.brier_score <= 1.0

    def test_log_loss_positive(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=10, seq_length=30, baseline_prob=0.50, seed=42)
        assert metrics.log_loss > 0
        assert metrics.log_loss_baseline > 0

    def test_detection_power_between_0_and_1(self):
        runner = ValidationRunner(checkpoints=[20, 40])
        metrics = runner.run(n_per_model=10, seq_length=50, baseline_prob=0.50, seed=42)
        for cp, power in metrics.detection_power.items():
            assert 0.0 <= power <= 1.0

    def test_false_positive_rate_between_0_and_1(self):
        runner = ValidationRunner(checkpoints=[20, 40])
        metrics = runner.run(n_per_model=10, seq_length=50, baseline_prob=0.50, seed=42)
        for cp, fpr in metrics.false_positive_rate.items():
            assert 0.0 <= fpr <= 1.0

    def test_summary_string(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=5, seq_length=30, baseline_prob=0.50, seed=42)
        summary = metrics.summary()
        assert "Brier score" in summary
        assert "CALIBRATION" in summary
        assert "DETECTION POWER" in summary
        assert "CHANGEPOINT PERFORMANCE" in summary
        assert "FNR=" in summary

    def test_calibration_metrics_present_and_bounded(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=5, seq_length=30, baseline_prob=0.50, seed=42)

        assert isinstance(metrics.calibration, CalibrationMetrics)
        assert 0.0 <= metrics.calibration.expected_calibration_error <= 1.0
        assert 0.0 <= metrics.calibration.max_calibration_error <= 1.0
        assert metrics.calibration.reliability >= 0.0
        assert metrics.calibration.resolution >= 0.0
        assert metrics.calibration.uncertainty >= 0.0

    def test_changepoint_performance_present_for_full_synthetic_suite(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=5, seq_length=80, baseline_prob=0.50, seed=42)

        assert isinstance(metrics.changepoint_performance, ChangepointPerformance)
        assert metrics.changepoint_performance.n_sequences == 5
        assert 0.0 <= metrics.changepoint_performance.detection_rate <= 1.0
        assert 0.0 <= metrics.changepoint_performance.localized_detection_rate <= 1.0
        assert 0.0 <= metrics.changepoint_performance.false_alarm_rate <= 1.0
        assert 0.0 <= metrics.changepoint_performance.mean_peak_probability <= 1.0

    def test_checkpoint_metrics_are_internally_consistent(self):
        runner = ValidationRunner(checkpoints=[20, 40])
        metrics = runner.run(n_per_model=5, seq_length=40, baseline_prob=0.50, seed=42)

        cp20 = metrics.checkpoint_metrics[20]
        assert cp20.true_positive + cp20.false_negative == metrics.n_human
        assert cp20.true_negative + cp20.false_positive == metrics.n_iid
        assert cp20.false_negative_rate == pytest.approx(1.0 - cp20.true_positive_rate)
        assert 0.0 <= cp20.roc_auc <= 1.0

    def test_per_model_metrics_cover_all_generators(self):
        runner = ValidationRunner(checkpoints=[20])
        metrics = runner.run(n_per_model=3, seq_length=20, baseline_prob=0.50, seed=42)

        assert set(metrics.per_model_metrics) == {
            "iid_bernoulli",
            "markov_alternation",
            "gamblers_fallacy",
            "counter",
            "run_averse",
            "mixture",
            "changepoint",
        }
        assert metrics.per_model_metrics["iid_bernoulli"].is_human is False
        assert metrics.per_model_metrics["markov_alternation"].is_human is True

    def test_validation_beats_baseline_on_structured_dataset(self):
        runner = ValidationRunner(checkpoints=[20, 40, 60])
        metrics = runner.run(dataset=_structured_validation_dataset())

        assert metrics.log_loss < metrics.log_loss_baseline
        assert metrics.brier_score < 0.25
        assert "better than the naive baseline" in metrics.summary()
        assert metrics.changepoint_performance is None

    def test_detection_power_increases_with_more_observations(self):
        runner = ValidationRunner(checkpoints=[20, 40, 60])
        metrics = runner.run(dataset=_structured_validation_dataset())

        assert metrics.detection_power[20] < metrics.detection_power[40] < metrics.detection_power[60]

    def test_false_positive_rate_stays_low_on_controlled_iid_subset(self):
        runner = ValidationRunner(checkpoints=[20, 40, 60])
        metrics = runner.run(dataset=_structured_validation_dataset())

        assert all(fpr <= 0.05 for fpr in metrics.false_positive_rate.values())
