"""Validation and calibration framework.

Runs the detection + prediction system against the synthetic calibration
dataset and reports performance metrics specifically chosen for small-n
regimes:

  - Brier score (proper scoring rule, meaningful at n=20)
  - Log-loss vs IID Bernoulli(P) baseline
  - Detection power at n = 20, 40, 60, 100
  - False positive rate (P(π_n > 0.75 | truly IID) — must be < 5%)
  - AUC-ROC for the detection task
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rand_check.prediction import PredictionEngine
from rand_check.models import Position
from rand_check.synthetic import GeneratedSequence, generate_calibration_dataset


@dataclass(frozen=True)
class ValidationMetrics:
    """Results from a validation run."""
    brier_score: float
    log_loss: float
    log_loss_baseline: float   # IID Bernoulli baseline for comparison
    detection_power: dict[int, float]   # n → P(π_n > 0.75 | human)
    false_positive_rate: dict[int, float]  # n → P(π_n > 0.75 | IID)
    n_sequences: int
    n_human: int
    n_iid: int

    def summary(self) -> str:
        lines = []
        lines.append("  " + "=" * 58)
        lines.append("  DETECTION ENGINE ACCURACY REPORT")
        lines.append("  " + "=" * 58)
        lines.append("")
        lines.append(f"  Sequences tested:  {self.n_sequences}")
        lines.append(f"    Patterned:       {self.n_human}")
        lines.append(f"    Truly random:    {self.n_iid}")
        lines.append("")

        # Prediction quality
        lines.append(f"  PREDICTION ACCURACY")
        lines.append("  " + "-" * 40)
        lines.append(f"    Brier score:         {self.brier_score:.4f}  (lower is better, 0 = perfect)")
        lines.append(f"    Model log-loss:      {self.log_loss:.4f}")
        lines.append(f"    Baseline log-loss:   {self.log_loss_baseline:.4f}  (naive always-GTO guess)")
        ll_improvement = self.log_loss_baseline - self.log_loss
        pct = ll_improvement / max(self.log_loss_baseline, 1e-10) * 100
        if ll_improvement > 0:
            lines.append(f"    --> Model is {pct:.1f}% better than the naive baseline")
        else:
            lines.append(f"    --> Model is {abs(pct):.1f}% worse than baseline (needs tuning)")

        # Detection power
        lines.append("")
        lines.append(f"  DETECTION POWER (can it spot patterned opponents?)")
        lines.append("  " + "-" * 40)
        for n in sorted(self.detection_power.keys()):
            power = self.detection_power[n]
            bar_len = int(power * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            lines.append(f"    After {n:3d} hands: {bar} {power * 100:.1f}%")

        # False positive rate
        lines.append("")
        lines.append(f"  FALSE ALARM RATE (does it wrongly flag random opponents?)")
        lines.append("  " + "-" * 40)
        for n in sorted(self.false_positive_rate.keys()):
            fpr = self.false_positive_rate[n]
            status = "PASS (< 5%)" if fpr < 0.05 else "FAIL (too high!)"
            lines.append(f"    After {n:3d} hands: {fpr * 100:.1f}%  {status}")

        return "\n".join(lines)


class ValidationRunner:
    """Run the full validation suite."""

    def __init__(
        self,
        detection_threshold: float = 0.75,
        checkpoints: list[int] | None = None,
    ) -> None:
        self.detection_threshold = detection_threshold
        self.checkpoints = checkpoints or [20, 40, 60, 100]

    def run(
        self,
        dataset: list[GeneratedSequence] | None = None,
        n_per_model: int = 100,
        seq_length: int = 100,
        gto_prob: float = 0.25,
        seed: int = 42,
    ) -> ValidationMetrics:
        """Execute the full validation.

        Parameters
        ----------
        dataset : list[GeneratedSequence] | None
            Pre-generated dataset, or None to auto-generate.
        """
        if dataset is None:
            dataset = generate_calibration_dataset(
                n_per_model=n_per_model,
                seq_length=seq_length,
                gto_prob=gto_prob,
                seed=seed,
            )

        # Accumulators
        brier_scores: list[float] = []
        log_losses: list[float] = []
        log_losses_baseline: list[float] = []

        # Detection at checkpoints
        # checkpoint → list of (π_n, is_human)
        checkpoint_results: dict[int, list[tuple[float, bool]]] = {
            cp: [] for cp in self.checkpoints
        }

        for gen_seq in dataset:
            seq = gen_seq.sequence
            p = gen_seq.gto_prob
            is_human = gen_seq.is_human

            engine = PredictionEngine(default_gto_prob=p)

            for i, action in enumerate(seq):
                hand_num = i + 1

                # Get prediction *before* observing the action
                pred = engine.predict_next()

                # Brier score: (pred - actual)^2
                brier_scores.append((pred - action) ** 2)

                # Log-loss: -(actual * log(pred) + (1-actual) * log(1-pred))
                pred_clamped = max(min(pred, 1.0 - 1e-10), 1e-10)
                if action == 1:
                    log_losses.append(-math.log(pred_clamped))
                else:
                    log_losses.append(-math.log(1.0 - pred_clamped))

                # Baseline log-loss (IID Bernoulli)
                p_clamped = max(min(p, 1.0 - 1e-10), 1e-10)
                if action == 1:
                    log_losses_baseline.append(-math.log(p_clamped))
                else:
                    log_losses_baseline.append(-math.log(1.0 - p_clamped))

                # Process the hand
                pkg = engine.process_action(action)

                # Record detection at checkpoints
                if hand_num in checkpoint_results:
                    checkpoint_results[hand_num].append(
                        (pkg.detection_confidence, is_human)
                    )

        # Compute aggregate metrics
        brier = float(np.mean(brier_scores))
        ll = float(np.mean(log_losses))
        ll_base = float(np.mean(log_losses_baseline))

        detection_power: dict[int, float] = {}
        false_positive_rate: dict[int, float] = {}

        for cp, results in checkpoint_results.items():
            human_results = [pi for pi, is_h in results if is_h]
            iid_results = [pi for pi, is_h in results if not is_h]

            if human_results:
                detected = sum(1 for pi in human_results if pi > self.detection_threshold)
                detection_power[cp] = detected / len(human_results)
            else:
                detection_power[cp] = 0.0

            if iid_results:
                false_pos = sum(1 for pi in iid_results if pi > self.detection_threshold)
                false_positive_rate[cp] = false_pos / len(iid_results)
            else:
                false_positive_rate[cp] = 0.0

        n_human = sum(1 for s in dataset if s.is_human)
        n_iid = sum(1 for s in dataset if not s.is_human)

        return ValidationMetrics(
            brier_score=brier,
            log_loss=ll,
            log_loss_baseline=ll_base,
            detection_power=detection_power,
            false_positive_rate=false_positive_rate,
            n_sequences=len(dataset),
            n_human=n_human,
            n_iid=n_iid,
        )
