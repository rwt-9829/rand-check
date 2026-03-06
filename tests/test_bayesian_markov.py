"""Tests for the Bayesian Markov(1) updater."""

import math
import pytest

from rand_check.bayesian_markov import BayesianMarkovUpdater


class TestBayesianMarkovUpdater:
    """Core tests for the Bayesian Markov updater."""

    def test_initial_prediction_equals_gto(self):
        """Before any data, prediction should be near the GTO prior."""
        updater = BayesianMarkovUpdater(gto_prob=0.25)
        # No last action → returns gto_prob
        assert updater.predict() == 0.25

    def test_update_increments_counts(self):
        updater = BayesianMarkovUpdater(gto_prob=0.25)
        updater.update(0)  # first action, no transition counted
        updater.update(1)  # transition 0→1
        assert updater._counts[0][1] == 1.0
        assert updater.counts_total == 1.0

    def test_prediction_after_alternating_sequence(self):
        """After observing many alternations (0,1,0,1,...), the model
        should predict a high P(switch) — i.e. after a 0, high P(3bet)."""
        updater = BayesianMarkovUpdater(gto_prob=0.25, kappa=5.0)
        for _ in range(20):
            updater.update(0)
            updater.update(1)
        # After the last 1, predict P(next=0 | last=1) should be high
        p_fold_after_3bet = 1.0 - updater.predict(given_last=1)
        assert p_fold_after_3bet > 0.7

    def test_prediction_after_streaky_sequence(self):
        """After observing many repeats (1,1,1,...), the model should
        predict high P(repeat)."""
        updater = BayesianMarkovUpdater(gto_prob=0.25, kappa=5.0)
        updater.update(0)
        for _ in range(30):
            updater.update(1)
        # P(3bet | last=3bet) should be high
        p = updater.predict(given_last=1)
        assert p > 0.6

    def test_credible_interval_narrows_with_data(self):
        updater = BayesianMarkovUpdater(gto_prob=0.25, kappa=5.0)
        updater.update(0)
        updater.update(1)
        ci_early = updater.credible_interval()
        width_early = ci_early[1] - ci_early[0]

        for _ in range(50):
            updater.update(0)
            updater.update(1)
        ci_late = updater.credible_interval()
        width_late = ci_late[1] - ci_late[0]

        assert width_late < width_early

    def test_reset_clears_counts_but_keeps_priors(self):
        updater = BayesianMarkovUpdater(gto_prob=0.25)
        for i in range(20):
            updater.update(i % 2)
        assert updater.counts_total > 0

        updater.reset()
        assert updater.counts_total == 0.0
        assert updater._alpha[0][1] > 0  # priors still set

    def test_reset_with_new_gto(self):
        updater = BayesianMarkovUpdater(gto_prob=0.25)
        updater.update(0)
        updater.reset(gto_prob=0.40)
        assert updater.gto_prob == 0.40
        assert updater.predict() == 0.40  # no last action → gto

    def test_log_likelihood_observation(self):
        updater = BayesianMarkovUpdater(gto_prob=0.25)
        updater.update(0)
        ll1 = updater.log_likelihood_observation(1)
        ll0 = updater.log_likelihood_observation(0)
        # Both should be negative (log of prob < 1)
        assert ll1 < 0
        assert ll0 < 0
        # P(fold) should be higher than P(3bet) from state 0 with default priors
        assert ll0 > ll1

    def test_transition_matrix_sums_to_one(self):
        updater = BayesianMarkovUpdater(gto_prob=0.25)
        for i in range(50):
            updater.update(i % 3 == 0)
        mat = updater.transition_matrix
        for row in mat:
            assert abs(sum(row) - 1.0) < 1e-10

    def test_kappa_controls_prior_strength(self):
        """Higher kappa = prior dominates longer."""
        seq = [0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
        updater_low = BayesianMarkovUpdater(gto_prob=0.25, kappa=3.0)
        updater_high = BayesianMarkovUpdater(gto_prob=0.25, kappa=30.0)
        for a in seq:
            updater_low.update(a)
            updater_high.update(a)
        # With high kappa, prediction should be closer to prior (gto)
        pred_low = updater_low.predict(given_last=0)
        pred_high = updater_high.predict(given_last=0)
        assert abs(pred_high - 0.25) < abs(pred_low - 0.25)
