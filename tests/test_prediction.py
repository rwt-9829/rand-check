"""Tests for the prediction engine."""

import pytest
import numpy as np

from rand_check.prediction import PredictionEngine
from rand_check.models import AnalysisPackage


class TestPredictionEngine:

    def test_process_action_returns_package(self):
        engine = PredictionEngine(default_baseline_prob=0.50)
        pkg = engine.process_action(0)
        assert isinstance(pkg, AnalysisPackage)

    def test_observations_processed_increments(self):
        engine = PredictionEngine()
        engine.process_action(0)
        engine.process_action(1)
        assert engine.observations_processed == 2

    def test_history_tracked(self):
        engine = PredictionEngine()
        engine.process_action(0)
        engine.process_action(1)
        engine.process_action(0)
        assert engine.history == [0, 1, 0]

    def test_process_sequence(self):
        engine = PredictionEngine(default_baseline_prob=0.50)
        seq = [0, 1, 0, 0, 1, 0, 1, 0]
        pkg = engine.process_sequence(seq)
        assert pkg.observations == 8

    def test_edge_calculation(self):
        engine = PredictionEngine(default_baseline_prob=0.50)
        # Feed lots of 1s -> predicted prob should go up -> positive edge
        for _ in range(30):
            pkg = engine.process_action(1)
        assert pkg.edge > 0

    def test_confidence_interval_in_package(self):
        engine = PredictionEngine()
        pkg = engine.process_action(0)
        lo, hi = pkg.confidence_interval
        assert lo < hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_detection_confidence_accessible(self):
        engine = PredictionEngine()
        for _ in range(10):
            engine.process_action(0)
            engine.process_action(1)
        assert 0.0 <= engine.detection_confidence <= 1.0

    def test_predict_next(self):
        engine = PredictionEngine(default_baseline_prob=0.50)
        engine.process_action(0)
        pred = engine.predict_next()
        assert 0.0 < pred < 1.0

    def test_reset(self):
        engine = PredictionEngine()
        for _ in range(20):
            engine.process_action(0)
        engine.reset()
        assert engine.observations_processed == 0
        assert engine.history == []

    def test_suggestion_present(self):
        engine = PredictionEngine()
        for _ in range(30):
            engine.process_action(1)
        pkg = engine.process_action(1)
        assert pkg.suggestion != ""

    def test_observations_to_reliable_decreases(self):
        engine = PredictionEngine()
        pkg1 = engine.process_action(0)
        for _ in range(39):
            pkg = engine.process_action(0)
        assert pkg.observations_to_reliable == 0

    def test_ctw_blend_activates_after_threshold(self):
        engine = PredictionEngine(ctw_activation=10)
        for _ in range(5):
            engine.process_action(0)
        assert not engine._ctw.is_active
        for _ in range(10):
            engine.process_action(0)
        assert engine._ctw.is_active
