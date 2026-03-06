"""Live prediction engine — the main orchestrator.

Combines all layers:
  1. Solver lookup          → GTO baseline P
  2. BayesianMarkovUpdater  → posterior predictive q_n
  3. DetectionEngine        → π_n (is this human?)
  4. ContextTreeWeighting   → higher-order prediction (post-60 hands)
  5. BOCPDDetector          → changepoint detection

Produces a DecisionPackage after every hand — the full
exploitation-ready output described in the design spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rand_check.bayesian_markov import BayesianMarkovUpdater
from rand_check.bocpd import BOCPDDetector
from rand_check.ctw import ContextTreeWeighting
from rand_check.detection import DetectionEngine
from rand_check.models import (
    Action,
    DecisionPackage,
    HandResult,
    Position,
)
from rand_check.solver_lookup import get_default_3bet_prob, get_gto_3bet_prob
from typing import Optional, List


@dataclass
class PredictionEngine:
    """The full live system — processes hands and produces predictions.

    Parameters
    ----------
    default_gto_prob : float
        Fallback GTO 3bet probability when position info is incomplete.
    kappa : float
        Prior pseudo-count strength for the Bayesian Markov model.
    delta : float
        Alternation bias offset.
    hazard_rate : float
        BOCPD hazard rate (1/expected_run_length).
    ctw_max_depth : int
        Maximum context depth for CTW.
    ctw_activation : int
        Minimum hands before CTW predictions are blended in.
    ctw_blend_weight : float
        Weight given to CTW prediction when active (rest to Markov(1)).
    exploitation_threshold : float
        Minimum |edge| required before suggesting exploitation.
    """
    default_gto_prob: float = 0.25
    kappa: float = 15.0
    delta: float = 0.10
    hazard_rate: float = 0.02
    ctw_max_depth: int = 3
    ctw_activation: int = 60
    ctw_blend_weight: float = 0.35
    exploitation_threshold: float = 0.03

    # ── Sub-engines (initialized in __post_init__) ───────────────────
    _detector: DetectionEngine = field(init=False, repr=False)
    _ctw: ContextTreeWeighting = field(init=False, repr=False)
    _bocpd: BOCPDDetector = field(init=False, repr=False)
    _history: List[int] = field(default_factory=list, init=False, repr=False)
    _hands_processed: int = field(default=0, init=False)
    _current_gto_prob: float = field(init=False)
    _last_changepoint_flag: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._current_gto_prob = self.default_gto_prob
        self._detector = DetectionEngine(gto_prob=self._current_gto_prob)
        self._ctw = ContextTreeWeighting(
            max_depth=self.ctw_max_depth,
            activation_threshold=self.ctw_activation,
        )
        self._bocpd = BOCPDDetector(hazard_rate=self.hazard_rate)

    # ── Public API ───────────────────────────────────────────────────

    def process_hand(self, hand: HandResult) -> DecisionPackage:
        """Process a single hand and return the full decision package.

        This is the main entry point called after every hand.
        """
        action_int = hand.action.value  # 0 or 1

        # Update GTO baseline for this spot
        if hand.villain_position is not None:
            self._current_gto_prob = get_gto_3bet_prob(
                hand.position, hand.villain_position, hand.stack_bb
            )
        else:
            self._current_gto_prob = get_default_3bet_prob(hand.position)

        # Update all sub-engines
        pi_n = self._detector.update(action_int)
        self._ctw.update(action_int)
        changepoint = self._bocpd.update(action_int)

        self._history.append(action_int)
        self._hands_processed += 1

        # Handle changepoint
        if changepoint:
            self._last_changepoint_flag = True
            self._soft_reset()
        else:
            self._last_changepoint_flag = False

        return self._build_package()

    def process_action(self, action: int, position: Position = Position.BTN,
                       villain_position: Optional[Position] = None,
                       stack_bb: float = 100.0) -> DecisionPackage:
        """Convenience method — process a raw action without a HandResult."""
        hand = HandResult(
            action=Action(action),
            position=position,
            stack_bb=stack_bb,
            hand_number=self._hands_processed + 1,
            villain_position=villain_position,
        )
        return self.process_hand(hand)

    def process_sequence(self, sequence: List[int],
                         position: Position = Position.BTN) -> DecisionPackage:
        """Process an entire sequence and return the final package.

        Useful for batch analysis of historical data.
        """
        pkg = DecisionPackage()
        for action in sequence:
            pkg = self.process_action(action, position)
        return pkg

    def predict_next(self) -> float:
        """Return the current predicted P(next = 3bet)."""
        return self._get_blended_prediction()

    @property
    def detection_confidence(self) -> float:
        """Current π_n."""
        return self._detector.posterior_h1

    @property
    def history(self) -> List[int]:
        return list(self._history)

    @property
    def hands_processed(self) -> int:
        return self._hands_processed

    def reset(self) -> None:
        """Full reset of all state."""
        self._detector.reset(gto_prob=self._current_gto_prob)
        self._ctw.reset()
        self._bocpd.reset()
        self._history.clear()
        self._hands_processed = 0

    # ── Private helpers ──────────────────────────────────────────────

    def _soft_reset(self) -> None:
        """Soft reset after changepoint: keep priors, discard accumulated data."""
        self._detector.reset(gto_prob=self._current_gto_prob)
        # CTW is NOT reset — it adapts on its own
        # BOCPD is NOT reset — it continues tracking run lengths

    def _get_blended_prediction(self) -> float:
        """Blend Markov(1) and CTW predictions."""
        markov_pred = self._detector.markov.predict()

        if self._ctw.is_active:
            ctw_pred = self._ctw.predict()
            w = self.ctw_blend_weight
            return w * ctw_pred + (1.0 - w) * markov_pred
        else:
            return markov_pred

    def _build_package(self) -> DecisionPackage:
        """Construct the full DecisionPackage."""
        pi_n = self._detector.posterior_h1
        q_n = self._get_blended_prediction()
        p = self._current_gto_prob
        edge = q_n - p

        ci = self._detector.markov.credible_interval(0.95)

        # Exploitation suggestion
        suggestion = self._exploitation_suggestion(edge, pi_n)

        # Hands to reliable
        hands_to_reliable = max(0, 40 - self._hands_processed)

        return DecisionPackage(
            detection_confidence=pi_n,
            predicted_3bet_prob=q_n,
            gto_3bet_prob=p,
            edge=edge,
            confidence_interval=ci,
            changepoint_flag=self._last_changepoint_flag,
            hands_observed=self._hands_processed,
            hands_to_reliable=hands_to_reliable,
            exploitation_suggestion=suggestion,
        )

    def _exploitation_suggestion(self, edge: float, pi_n: float) -> str:
        """Generate human-readable exploitation recommendation."""
        # Scale threshold by inverse confidence
        if pi_n < 0.1:
            return "Insufficient data — play GTO"

        adjusted_threshold = self.exploitation_threshold / max(pi_n, 0.1)

        if abs(edge) < adjusted_threshold:
            return "No significant edge detected — continue GTO play"

        if edge > 0:
            # Opponent 3bets more than GTO
            suggestions = []
            if edge > 0.08:
                suggestions.append("STRONG: Tighten calling range significantly")
                suggestions.append("Widen 4bet bluff range to exploit their wide 3bets")
            elif edge > 0.04:
                suggestions.append("Fold more marginal hands vs their 3bets")
                suggestions.append("Consider widening 4bet range")
            else:
                suggestions.append("Slight tightening of call range")
            return " | ".join(suggestions)
        else:
            # Opponent 3bets less than GTO
            suggestions = []
            if edge < -0.08:
                suggestions.append("STRONG: Widen flatting range aggressively")
                suggestions.append("Reduce 4bet bluffing (they rarely 3bet)")
            elif edge < -0.04:
                suggestions.append("Widen flatting range — defend lighter")
                suggestions.append("Reduce 4bet bluff frequency")
            else:
                suggestions.append("Slight widening of defend range")
            return " | ".join(suggestions)
