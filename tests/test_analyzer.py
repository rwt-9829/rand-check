"""Tests for post-session analysis.

The stronger fingerprint checks are motivated by the classic subjective-
randomness literature, especially Tversky & Kahneman (1971) and Falk & Konold
(1997): human-made sequences tend to over-alternate and show local reversal
pressure.
"""

import numpy as np

from rand_check.analyzer import PatternFingerprint, PostSessionAnalyzer, SessionReport
from rand_check.features import FeatureVector
from rand_check.models import AnalysisPackage
from rand_check.synthetic import generate_gamblers_fallacy


def _make_report() -> SessionReport:
    return SessionReport(
        features=FeatureVector(
            log_likelihood_ratio=2.5,
            alternation_deviation=0.14,
            run_length_score=6.0,
            normalized_lz_complexity=0.82,
            frequency_drift=0.01,
            serial_correlation=-0.30,
            n_observations=40,
        ),
        fingerprint=PatternFingerprint(
            primary_bias="alternation_bias",
            confidence=0.85,
            bias_scores={"alternation_bias": 0.85, "run_aversion": 0.40},
            description="Subject over-alternates between 0 and 1.",
        ),
        final_prediction=AnalysisPackage(
            detection_confidence=0.82,
            predicted_prob=0.62,
            baseline_prob=0.50,
            edge=0.12,
            confidence_interval=(0.54, 0.70),
            changepoint_flag=False,
            observations=40,
            observations_to_reliable=0,
            suggestion="Subject favors 1 over baseline (+12.0% deviation)",
        ),
        timeline=[0.20, 0.35, 0.60, 0.82],
        prediction_timeline=[0.50, 0.54, 0.59, 0.62],
        baseline_prob=0.50,
        n_observations=40,
        changepoints=[21],
    )


class TestPostSessionAnalyzer:

    def test_analyze_alternating_sequence_identifies_alternation_bias(self):
        report = PostSessionAnalyzer().analyze([0, 1] * 30, baseline_prob=0.50)

        assert report.fingerprint.primary_bias == "alternation_bias"
        assert len(report.timeline) == 60
        assert len(report.prediction_timeline) == 60
        assert report.final_prediction.observations == 60

    def test_short_sequence_defaults_to_none_detected(self):
        report = PostSessionAnalyzer().analyze([0, 1, 0, 1, 1], baseline_prob=0.50)

        assert report.fingerprint.primary_bias == "none_detected"
        assert report.fingerprint.confidence == 0.0

    def test_gambler_fallacy_score_is_positive_for_reversal_biased_generator(self):
        sequence = generate_gamblers_fallacy(120, 0.50, rng=np.random.default_rng(42)).sequence
        score = PostSessionAnalyzer._gambler_fallacy_score(sequence, 0.50)
        assert score > 0.10

    def test_get_runs_groups_consecutive_values(self):
        runs = PostSessionAnalyzer._get_runs([0, 0, 1, 1, 1, 0])
        assert runs == [(0, 2), (1, 3), (0, 1)]

    def test_friendly_summary_mentions_bias_and_changepoint(self):
        summary = _make_report().summary()
        assert "SESSION REPORT" in summary
        assert "Alternation bias" in summary
        assert "Regime changes detected" in summary
        assert "ASSESSMENT" in summary

    def test_detailed_summary_includes_technical_metrics(self):
        summary = _make_report().summary(detailed=True)
        assert "POST-SESSION ANALYSIS (DETAILED)" in summary
        assert "Detection confidence (pi_n)" in summary
        assert "Pattern Fingerprint" in summary
