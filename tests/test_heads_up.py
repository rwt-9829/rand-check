"""Tests for Heads-Up (HU) deployment configuration."""

from __future__ import annotations

import unittest

from rand_check.models import Position
from rand_check.prediction import PredictionEngine
from rand_check.solver_lookup import GameFormat, get_gto_3bet_prob, get_default_3bet_prob


class TestHUGTOLookup(unittest.TestCase):
    """Verify HU solver table returns correct values."""

    def test_bb_vs_btn_base_frequency(self):
        p = get_gto_3bet_prob(Position.BB, Position.BTN, 100.0, GameFormat.HEADS_UP)
        self.assertAlmostEqual(p, 0.23, places=2)

    def test_bb_vs_sb_alias(self):
        """BTN and SB are the same seat in HU — should give same result."""
        p_btn = get_gto_3bet_prob(Position.BB, Position.BTN, 100.0, GameFormat.HEADS_UP)
        p_sb = get_gto_3bet_prob(Position.BB, Position.SB, 100.0, GameFormat.HEADS_UP)
        self.assertAlmostEqual(p_btn, p_sb)

    def test_short_stack_increases_frequency(self):
        p_short = get_gto_3bet_prob(Position.BB, Position.BTN, 20.0, GameFormat.HEADS_UP)
        p_med = get_gto_3bet_prob(Position.BB, Position.BTN, 60.0, GameFormat.HEADS_UP)
        self.assertGreater(p_short, p_med)

    def test_deep_stack_decreases_frequency(self):
        p_deep = get_gto_3bet_prob(Position.BB, Position.BTN, 150.0, GameFormat.HEADS_UP)
        p_med = get_gto_3bet_prob(Position.BB, Position.BTN, 60.0, GameFormat.HEADS_UP)
        self.assertLess(p_deep, p_med)

    def test_default_hu_prob(self):
        p = get_default_3bet_prob(Position.BB, GameFormat.HEADS_UP)
        self.assertAlmostEqual(p, 0.23)

    def test_sixmax_unchanged(self):
        """6-max lookups should be unaffected by HU additions."""
        p = get_gto_3bet_prob(Position.BB, Position.BTN, 100.0, GameFormat.SIXMAX)
        self.assertAlmostEqual(p, 0.13, places=2)


class TestHUPredictionEngine(unittest.TestCase):
    """Verify the HU factory and engine behaviour."""

    def test_for_heads_up_factory(self):
        engine = PredictionEngine.for_heads_up()
        self.assertEqual(engine.game_format, GameFormat.HEADS_UP)
        self.assertAlmostEqual(engine.default_gto_prob, 0.23, places=2)

    def test_for_heads_up_custom_stack(self):
        engine = PredictionEngine.for_heads_up(stack_bb=25.0)
        # Short stack → higher GTO freq
        self.assertGreater(engine.default_gto_prob, 0.23)

    def test_ctw_activation_earlier_in_hu(self):
        engine = PredictionEngine.for_heads_up()
        self.assertEqual(engine.ctw_activation, 45)

    def test_process_action_uses_hu_table(self):
        engine = PredictionEngine.for_heads_up()
        pkg = engine.process_action(1, position=Position.BB, villain_position=Position.BTN)
        self.assertAlmostEqual(pkg.gto_3bet_prob, 0.23, places=2)

    def test_process_sequence_hu(self):
        engine = PredictionEngine.for_heads_up()
        seq = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        pkg = engine.process_sequence(seq, position=Position.BB)
        self.assertEqual(pkg.hands_observed, 10)
        self.assertTrue(0.0 <= pkg.predicted_3bet_prob <= 1.0)

    def test_edge_sign_for_tight_opponent(self):
        """An opponent who never 3-bets should give negative edge."""
        engine = PredictionEngine.for_heads_up()
        for _ in range(40):
            pkg = engine.process_action(0, position=Position.BB, villain_position=Position.BTN)
        self.assertLess(pkg.edge, 0.0, "edge should be negative for under-3betting opponent")


if __name__ == "__main__":
    unittest.main()
