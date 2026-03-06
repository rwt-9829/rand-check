"""Context Tree Weighting (CTW) — variable-order Markov prediction.

Implements the CTW algorithm of Willems, Shtarkov & Tjalkens (1995) for
binary sequences.  Maintains a depth-D suffix tree where each node stores
Beta-Binomial sufficient statistics.  The predicted probability is a
Bayesian mixture over all Markov orders 0 … D, with weights determined
by the data likelihood at each depth — higher-order models gain weight
only when they demonstrably improve prediction.

This layer activates after ~60 hands (configurable) and can capture
patterns up to Markov(D) that the simple Markov(1) model misses, e.g.:
  - "Never 3bets three times in a row" (order 2)
  - "After fold-fold-fold, always 3bets" (order 3)

Reference:
    Willems, Shtarkov & Tjalkens (1995). "The Context-Tree Weighting
    Method: Basic Properties."  IEEE Trans. IT, 41(3):653-664.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _CTWNode:
    """A single node in the context tree.

    Each node maintains:
      - a, b: counts of 0s and 1s that *arrived at this context*.
      - log_pe: log of the KT-estimated probability at this node.
      - log_pw: log of the *weighted* (CTW-mixed) probability.
    """
    a: float = 0.0   # count of 0s
    b: float = 0.0   # count of 1s
    log_pe: float = 0.0  # log KT estimate (local)
    log_pw: float = 0.0  # log weighted probability (CTW mixture)
    children: dict[int, "_CTWNode"] = field(default_factory=dict)


def _kt_update(a: float, b: float, symbol: int) -> tuple[float, float, float]:
    """Krichevsky-Trofimov sequential estimator update.

    Returns (new_a, new_b, log_ratio) where log_ratio is the
    log-probability of the new symbol under KT.

    KT estimate: P(x_{t+1} = 1 | a zeros, b ones) = (b + 0.5) / (a + b + 1)
    """
    total = a + b + 1.0
    if symbol == 1:
        log_ratio = math.log((b + 0.5) / total)
        return a, b + 1.0, log_ratio
    else:
        log_ratio = math.log((a + 0.5) / total)
        return a + 1.0, b, log_ratio


class ContextTreeWeighting:
    """Binary CTW predictor with configurable max depth.

    Parameters
    ----------
    max_depth : int
        Maximum context depth (Markov order).  D=3 is standard for
        this application — covers up to "last 3 actions" patterns.
    activation_threshold : int
        Minimum observations before CTW predictions are used.
    """

    def __init__(self, max_depth: int = 3, activation_threshold: int = 60) -> None:
        self.max_depth = max_depth
        self.activation_threshold = activation_threshold
        self._root = _CTWNode()
        self._history: list[int] = []
        self._n_updates = 0

    @property
    def is_active(self) -> bool:
        """Whether enough data has been collected to trust CTW."""
        return self._n_updates >= self.activation_threshold

    @property
    def n_observations(self) -> int:
        return self._n_updates

    def reset(self) -> None:
        """Full reset — discard all data."""
        self._root = _CTWNode()
        self._history.clear()
        self._n_updates = 0

    # ── Public API ───────────────────────────────────────────────────

    def update(self, symbol: int) -> None:
        """Process a new binary symbol (0 or 1).

        Updates all nodes along the context path in the suffix tree,
        then recomputes weighted probabilities bottom-up.
        """
        context = self._get_context()
        self._update_tree(self._root, context, 0, symbol)
        self._history.append(symbol)
        self._n_updates += 1

    def predict(self) -> float:
        """Return P(next = 1 | history) under the CTW mixture.

        This marginalizes over all Markov orders 0 … D.
        """
        context = self._get_context()

        # Compute log P(next=1) and log P(next=0) by simulating updates
        # and taking the ratio of weighted probabilities.
        log_pw_before = self._root.log_pw

        # Simulate update with symbol=1
        log_pw_after_1 = self._simulate_update(self._root, context, 0, 1)

        # Simulate update with symbol=0
        log_pw_after_0 = self._simulate_update(self._root, context, 0, 0)

        # P(1) = exp(log_pw_after_1) / (exp(log_pw_after_1) + exp(log_pw_after_0))
        # Use log-sum-exp for numerical stability
        max_log = max(log_pw_after_1, log_pw_after_0)
        denom = max_log + math.log(
            math.exp(log_pw_after_1 - max_log) + math.exp(log_pw_after_0 - max_log)
        )
        prob_1 = math.exp(log_pw_after_1 - denom)

        # Clamp to valid probability
        return max(min(prob_1, 1.0 - 1e-10), 1e-10)

    def log_loss(self) -> float:
        """Total log-loss (negative log-likelihood) of all observations."""
        return -self._root.log_pw

    # ── Tree operations (private) ────────────────────────────────────

    def _get_context(self) -> list[int]:
        """Most recent symbols, reversed (deepest context first)."""
        d = min(self.max_depth, len(self._history))
        return list(reversed(self._history[-d:])) if d > 0 else []

    def _update_tree(
        self, node: _CTWNode, context: list[int], depth: int, symbol: int
    ) -> None:
        """Recursively update the context tree with a new symbol.

        Traverses the tree from root to the deepest context node,
        updates the KT estimates, then propagates the weighted
        probability back up (bottom-up pass).
        """
        # Update local KT estimate
        node.a, node.b, log_kt_ratio = _kt_update(node.a, node.b, symbol)
        node.log_pe += log_kt_ratio

        if depth < self.max_depth and depth < len(context):
            # Recurse into the child indexed by the next context symbol
            ctx_sym = context[depth]
            if ctx_sym not in node.children:
                node.children[ctx_sym] = _CTWNode()
            child = node.children[ctx_sym]
            self._update_tree(child, context, depth + 1, symbol)

            # CTW weighted probability:
            # P_w(node) = 0.5 · P_e(node) + 0.5 · Π_{children} P_w(child)
            # In log space:
            log_children_pw = sum(
                c.log_pw for c in node.children.values()
            )
            # For missing children, their contribution is 0 in log space
            # (they have pw=1 initially)

            node.log_pw = _log_add(
                math.log(0.5) + node.log_pe,
                math.log(0.5) + log_children_pw,
            )
        else:
            # Leaf node: weighted prob = KT estimate
            node.log_pw = node.log_pe

    def _simulate_update(
        self, node: _CTWNode, context: list[int], depth: int, symbol: int
    ) -> float:
        """Non-destructively compute what log_pw would be after observing symbol."""
        # Local KT update (simulated)
        new_a, new_b, log_kt_ratio = _kt_update(node.a, node.b, symbol)
        new_log_pe = node.log_pe + log_kt_ratio

        if depth < self.max_depth and depth < len(context):
            ctx_sym = context[depth]
            if ctx_sym in node.children:
                child = node.children[ctx_sym]
                child_log_pw = self._simulate_update(child, context, depth + 1, symbol)
            else:
                # New child would get its first symbol
                child = _CTWNode()
                _, _, child_log_kt = _kt_update(0.0, 0.0, symbol)
                child_log_pw = child_log_kt  # leaf: log_pw = log_pe

            # Sum log_pw of all children (only the target child changes)
            log_children_pw = child_log_pw
            for sym, c in node.children.items():
                if sym != ctx_sym:
                    log_children_pw += c.log_pw

            return _log_add(
                math.log(0.5) + new_log_pe,
                math.log(0.5) + log_children_pw,
            )
        else:
            return new_log_pe


def _log_add(log_a: float, log_b: float) -> float:
    """Compute log(exp(log_a) + exp(log_b)) in a numerically stable way."""
    if log_a == float("-inf"):
        return log_b
    if log_b == float("-inf"):
        return log_a
    mx = max(log_a, log_b)
    return mx + math.log(math.exp(log_a - mx) + math.exp(log_b - mx))
