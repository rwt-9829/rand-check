"""
rand-check: 3bet Randomness Detector & Exploitative Predictor

A Bayesian system for detecting non-random patterns in poker 3bet decisions
and producing exploitative predictions in real-time.

Key algorithms:
- Bayesian Markov(1) updater with conjugate Beta-Binomial priors
- Context Tree Weighting (CTW) for variable-order Markov prediction
- Bayesian Online Changepoint Detection (BOCPD)
- Six-feature post-session analysis suite

References:
- Adams & MacKay (2007). "Bayesian Online Changepoint Detection." arXiv:0710.3742
- Willems, Shtarkov & Tjalkens (1995). "The Context-Tree Weighting Method."
  IEEE Trans. IT, 41(3):653-664
- Lempel & Ziv (1976). "On the Complexity of Finite Sequences."
  IEEE Trans. IT, 22(1):75-81
- Tversky & Kahneman (1971). "Belief in the law of small numbers."
  Psychological Bulletin, 76(2):105-110
"""

__version__ = "0.1.0"

from rand_check.models import (
    Action,
    HandResult,
    Position,
    DecisionPackage,
)
from rand_check.prediction import PredictionEngine
from rand_check.solver_lookup import GameFormat

__all__ = [
    "Action",
    "HandResult",
    "Position",
    "DecisionPackage",
    "PredictionEngine",
    "GameFormat",
]
