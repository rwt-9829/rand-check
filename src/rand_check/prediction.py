"""Live prediction engine -- the main orchestrator.

Combines all layers:
  1. Baseline probability      -- user-specified P(outcome=1) under null
  2. BayesianMarkovUpdater     -- posterior predictive q_n
  3. DetectionEngine           -- pi_n (is this patterned?)
  4. ContextTreeWeighting      -- higher-order prediction (post-60 observations)
  5. BOCPDDetector             -- changepoint detection

Produces an AnalysisPackage after every observation -- the full
analysis-ready output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rand_check.bayesian_markov import BayesianMarkovUpdater
from rand_check.bocpd import BOCPDDetector
from rand_check.ctw import ContextTreeWeighting
from rand_check.detection import DetectionEngine
from rand_check.models import AnalysisPackage
from rand_check.solver_lookup import DEFAULT_BASELINE_PROB
from typing import List


@dataclass
class PredictionEngine:
    """The full live system -- processes observations and produces predictions.

    Parameters
    ----------
    default_baseline_prob : float
        Baseline probability of outcome=1 under the null hypothesis.
    kappa : float
        Prior pseudo-count strength for the Bayesian Markov model.
    delta : float
        Alternation bias offset.
    hazard_rate : float
        BOCPD hazard rate (1/expected_run_length).
    ctw_max_depth : int
        Maximum context depth for CTW.
    ctw_activation : int
        Minimum observations before CTW predictions are blended in.
    ctw_blend_weight : float
        Weight given to CTW prediction when active (rest to Markov(1)).
    exploitation_threshold : float
        Minimum |edge| required before suggesting action.
    """
    default_baseline_prob: float = DEFAULT_BASELINE_PROB
    kappa: float = 15.0
    delta: float = 0.10
    hazard_rate: float = 0.02
    ctw_max_depth: int = 3
    ctw_activation: int = 60
    ctw_blend_weight: float = 0.35
    exploitation_threshold: float = 0.03
    markov_weight: float = 0.7
    counter_window: int = 12
    counter_strength: float = 1.25

    # -- Sub-engines (initialized in __post_init__) --------------------
    _detector: DetectionEngine = field(init=False, repr=False)
    _ctw: ContextTreeWeighting = field(init=False, repr=False)
    _bocpd: BOCPDDetector = field(init=False, repr=False)
    _history: List[int] = field(default_factory=list, init=False, repr=False)
    _observations_processed: int = field(default=0, init=False)
    _current_baseline_prob: float = field(init=False)
    _last_changepoint_flag: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._current_baseline_prob = self.default_baseline_prob
        self._detector = DetectionEngine(
            baseline_prob=self._current_baseline_prob,
            markov_kappa=self.kappa,
            markov_delta=self.delta,
            markov_weight=self.markov_weight,
            counter_window=self.counter_window,
            counter_strength=self.counter_strength,
        )
        self._ctw = ContextTreeWeighting(
            max_depth=self.ctw_max_depth,
            activation_threshold=self.ctw_activation,
        )
        self._bocpd = BOCPDDetector(hazard_rate=self.hazard_rate)

    # -- Public API ----------------------------------------------------

    def process_action(self, action: int) -> AnalysisPackage:
        """Process a single binary observation and return the analysis package.

        Parameters
        ----------
        action : int
            0 or 1.
        """
        # Update all sub-engines
        pi_n = self._detector.update(action)
        self._ctw.update(action)
        changepoint = self._bocpd.update(action)

        self._history.append(action)
        self._observations_processed += 1

        # Handle changepoint
        if changepoint:
            self._last_changepoint_flag = True
            self._soft_reset()
        else:
            self._last_changepoint_flag = False

        return self._build_package()

    def process_sequence(self, sequence: List[int]) -> AnalysisPackage:
        """Process an entire sequence and return the final package.

        Useful for batch analysis of historical data.
        """
        pkg = AnalysisPackage()
        for action in sequence:
            pkg = self.process_action(action)
        return pkg

    def predict_next(self) -> float:
        """Return the current predicted P(next = 1)."""
        return self._get_blended_prediction()

    @property
    def detection_confidence(self) -> float:
        """Current pi_n."""
        return self._detector.posterior_h1

    @property
    def history(self) -> List[int]:
        return list(self._history)

    @property
    def observations_processed(self) -> int:
        return self._observations_processed

    def reset(self) -> None:
        """Full reset of all state."""
        self._detector.reset(baseline_prob=self._current_baseline_prob)
        self._ctw.reset()
        self._bocpd.reset()
        self._history.clear()
        self._observations_processed = 0

    # -- Private helpers -----------------------------------------------

    def _soft_reset(self) -> None:
        """Soft reset after changepoint: keep priors, discard accumulated data."""
        self._detector.reset(baseline_prob=self._current_baseline_prob)
        self._ctw.reset()

    def _get_pattern_prediction(self) -> float:
        """Pattern-model prediction under H1.

        This is the higher-capacity sequential model: a Markov(1) base rate
        optionally blended with CTW once enough data exists.
        """
        base_pattern_pred = self._detector.patterned_predict()

        if self._ctw.is_active:
            ctw_pred = self._ctw.predict()
            w = self.ctw_blend_weight
            return w * ctw_pred + (1.0 - w) * base_pattern_pred

        return base_pattern_pred

    def _get_blended_prediction(self) -> float:
        """Bayesian model average between H0 and H1 predictions.

        The detection engine maintains ``pi_n = P(H1 | data)``. Prediction
        should therefore average the null model (IID Bernoulli baseline) and
        the sequential H1 model rather than always trusting the patterned model.
        This improves calibration on genuinely random sequences while still
        exploiting structure once the evidence for H1 grows.
        """
        pi_n = self._detector.posterior_h1
        h0_pred = self._current_baseline_prob
        h1_pred = self._get_pattern_prediction()
        return (1.0 - pi_n) * h0_pred + pi_n * h1_pred

    def _build_package(self) -> AnalysisPackage:
        """Construct the full AnalysisPackage."""
        pi_n = self._detector.posterior_h1
        q_n = self._get_blended_prediction()
        p = self._current_baseline_prob
        edge = q_n - p

        ci = self._detector.markov.credible_interval(0.95)

        # Suggestion
        suggestion = self._generate_suggestion(edge, pi_n)

        # Observations to reliable
        obs_to_reliable = max(0, 40 - self._observations_processed)

        return AnalysisPackage(
            detection_confidence=pi_n,
            predicted_prob=q_n,
            baseline_prob=p,
            edge=edge,
            confidence_interval=ci,
            changepoint_flag=self._last_changepoint_flag,
            observations=self._observations_processed,
            observations_to_reliable=obs_to_reliable,
            suggestion=suggestion,
        )

    def _generate_suggestion(self, edge: float, pi_n: float) -> str:
        """Generate human-readable analysis recommendation."""
        if pi_n < 0.1:
            return "Insufficient data -- continue observing"

        adjusted_threshold = self.exploitation_threshold / max(pi_n, 0.1)

        if abs(edge) < adjusted_threshold:
            return "No significant deviation from baseline detected"

        if edge > 0:
            # Subject produces 1s more than baseline
            magnitude = "strongly " if edge > 0.08 else ("" if edge > 0.04 else "slightly ")
            return f"Subject {magnitude}favors 1 over baseline ({edge:+.1%} deviation)"
        else:
            # Subject produces 0s more than baseline
            magnitude = "strongly " if edge < -0.08 else ("" if edge < -0.04 else "slightly ")
            return f"Subject {magnitude}favors 0 over baseline ({edge:+.1%} deviation)"
