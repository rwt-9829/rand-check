"""Tests for the command-line interface."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rand_check import cli
from rand_check.models import AnalysisPackage
from rand_check.synthetic import GeneratedSequence
from rand_check.validation import ValidationMetrics


class TestCliHelpers:

    def test_parse_sequence_accepts_multiple_formats(self):
        assert cli._parse_sequence("0101") == [0, 1, 0, 1]
        assert cli._parse_sequence("0 1 0 1") == [0, 1, 0, 1]
        assert cli._parse_sequence("0,1,0,1") == [0, 1, 0, 1]

    def test_parse_sequence_rejects_invalid_input(self, capsys):
        with pytest.raises(SystemExit):
            cli._parse_sequence("01201")

        out = capsys.readouterr().out
        assert "sequence must contain only 0s and 1s" in out

    def test_resolve_model_supports_prefixes(self):
        assert cli._resolve_model("alt") == "markov_alternation"
        assert cli._resolve_model("random") == "iid"

    def test_main_without_command_prints_welcome(self, capsys):
        cli.main([])
        assert "rand-check -- Binary Sequence Pattern Detector" in capsys.readouterr().out


class TestCliCommands:

    def test_simulate_command_outputs_binary_string(self, capsys):
        cli.main(["simulate", "random", "12", "--seed", "1"])
        out = capsys.readouterr().out.strip()

        assert len(out) == 12
        assert set(out) <= {"0", "1"}

    def test_analyze_command_prints_report(self, monkeypatch, capsys):
        class FakeReport:
            def summary(self, detailed: bool = False) -> str:
                return "stub detailed summary" if detailed else "stub summary"

        class FakeAnalyzer:
            def analyze(self, sequence, baseline_prob=0.50):
                assert sequence == [0, 1, 0, 1, 0]
                assert baseline_prob == 0.50
                return FakeReport()

        monkeypatch.setattr("rand_check.analyzer.PostSessionAnalyzer", FakeAnalyzer)

        cli.main(["analyze", "01010"])
        out = capsys.readouterr().out

        assert "ANALYZING 5 OBSERVATIONS" in out
        assert "stub summary" in out

    def test_validate_command_prints_metrics_summary(self, monkeypatch, capsys):
        class FakeRunner:
            def run(self, **kwargs):
                return ValidationMetrics(
                    brier_score=0.20,
                    log_loss=0.65,
                    log_loss_baseline=0.69,
                    detection_power={20: 0.75},
                    false_positive_rate={20: 0.0},
                    n_sequences=14,
                    n_human=12,
                    n_iid=2,
                )

        monkeypatch.setattr("rand_check.validation.ValidationRunner", FakeRunner)

        cli.main(["validate", "--sequences", "2", "--length", "10"])
        out = capsys.readouterr().out

        assert "DETECTION ENGINE ACCURACY REPORT" in out
        assert "Completed in" in out

    def test_demo_command_prints_live_walkthrough(self, monkeypatch, capsys):
        class FakeDetector:
            def interpret(self) -> str:
                return "Probable pattern -- sequential dependencies detected"

        class FakeEngine:
            def __init__(self, default_baseline_prob=0.50):
                self._detector = FakeDetector()
                self._obs = 0

            def process_action(self, action: int) -> AnalysisPackage:
                self._obs += 1
                return AnalysisPackage(
                    detection_confidence=min(0.25 * self._obs, 0.9),
                    predicted_prob=0.62,
                    baseline_prob=0.50,
                    edge=0.12,
                    confidence_interval=(0.54, 0.70),
                    changepoint_flag=(self._obs == 2),
                    observations=self._obs,
                    observations_to_reliable=max(0, 40 - self._obs),
                    suggestion="Subject favors 1 over baseline (+12.0% deviation)",
                )

            def _build_package(self) -> AnalysisPackage:
                return AnalysisPackage(
                    detection_confidence=0.75,
                    predicted_prob=0.62,
                    baseline_prob=0.50,
                    edge=0.12,
                    confidence_interval=(0.54, 0.70),
                    changepoint_flag=False,
                    observations=self._obs,
                    observations_to_reliable=max(0, 40 - self._obs),
                    suggestion="Subject favors 1 over baseline (+12.0% deviation)",
                )

        class FakeAnalyzer:
            def analyze(self, sequence, baseline_prob=0.50):
                return SimpleNamespace(summary=lambda detailed=False: "stub post-session summary")

        def fake_generate(*args, **kwargs):
            return GeneratedSequence(
                sequence=[0, 1, 0],
                model_name="markov_alternation",
                baseline_prob=0.50,
                is_human=True,
                parameters={},
            )

        monkeypatch.setattr("rand_check.prediction.PredictionEngine", FakeEngine)
        monkeypatch.setattr("rand_check.analyzer.PostSessionAnalyzer", FakeAnalyzer)
        monkeypatch.setattr("rand_check.synthetic.generate_markov_alternation", fake_generate)

        cli.main(["demo", "--model", "alternator", "--length", "3"])
        out = capsys.readouterr().out

        assert "DEMO: Watching a \"alternator\" sequence" in out
        assert "RESULT AFTER 3 OBSERVATIONS" in out
        assert "stub post-session summary" in out
