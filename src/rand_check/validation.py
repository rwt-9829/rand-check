"""Validation and calibration framework.

Runs the detection + prediction system against the synthetic calibration
dataset and reports performance metrics specifically chosen for small-n
regimes:

  - Brier score (proper scoring rule, meaningful at n=20)
  - Log-loss vs IID Bernoulli(P) baseline
  - Detection power at n = 20, 40, 60, 100
  - False positive rate (P(pi_n > 0.75 | truly IID) -- must be < 5%)
    - Full checkpoint classification metrics (TP/FP/TN/FN, FNR, precision,
        specificity, balanced accuracy, F1, ROC AUC)
    - Per-model prediction and detection breakdowns
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rand_check.prediction import PredictionEngine
from rand_check.synthetic import GeneratedSequence, generate_calibration_dataset


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _roc_auc(scores: list[float], labels: list[int]) -> float:
    """Compute ROC AUC from scores without external ML dependencies."""
    n = len(scores)
    if n == 0:
        return 0.0

    positives = sum(labels)
    negatives = n - positives
    if positives == 0 or negatives == 0:
        return 0.0

    order = sorted(range(n), key=lambda idx: scores[idx])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i + 1
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1

        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    rank_sum_pos = sum(ranks[idx] for idx, label in enumerate(labels) if label == 1)
    u_stat = rank_sum_pos - positives * (positives + 1) / 2.0
    return float(u_stat / (positives * negatives))


def _expected_calibration_error(
    predictions: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> tuple[float, float]:
    """Return (ECE, MCE) for binary probabilistic predictions."""
    if not predictions or not outcomes or len(predictions) != len(outcomes):
        return 0.0, 0.0

    ece = 0.0
    mce = 0.0
    total = len(predictions)

    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        if bin_idx == n_bins - 1:
            members = [i for i, pred in enumerate(predictions) if lo <= pred <= hi]
        else:
            members = [i for i, pred in enumerate(predictions) if lo <= pred < hi]
        if not members:
            continue

        mean_pred = float(np.mean([predictions[i] for i in members]))
        mean_obs = float(np.mean([outcomes[i] for i in members]))
        gap = abs(mean_pred - mean_obs)
        weight = len(members) / total
        ece += weight * gap
        mce = max(mce, gap)

    return ece, mce


def _brier_decomposition(
    predictions: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> tuple[float, float, float]:
    """Return Murphy decomposition: (reliability, resolution, uncertainty)."""
    if not predictions or not outcomes or len(predictions) != len(outcomes):
        return 0.0, 0.0, 0.0

    overall_rate = float(np.mean(outcomes))
    uncertainty = overall_rate * (1.0 - overall_rate)
    reliability = 0.0
    resolution = 0.0
    total = len(predictions)

    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        if bin_idx == n_bins - 1:
            members = [i for i, pred in enumerate(predictions) if lo <= pred <= hi]
        else:
            members = [i for i, pred in enumerate(predictions) if lo <= pred < hi]
        if not members:
            continue

        weight = len(members) / total
        mean_pred = float(np.mean([predictions[i] for i in members]))
        mean_obs = float(np.mean([outcomes[i] for i in members]))
        reliability += weight * (mean_pred - mean_obs) ** 2
        resolution += weight * (mean_obs - overall_rate) ** 2

    return reliability, resolution, uncertainty


@dataclass(frozen=True)
class CheckpointMetrics:
    """Classification metrics at a specific observation checkpoint."""
    checkpoint: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    true_positive_rate: float
    false_positive_rate: float
    false_negative_rate: float
    true_negative_rate: float
    precision: float
    negative_predictive_value: float
    accuracy: float
    balanced_accuracy: float
    f1_score: float
    roc_auc: float
    mean_pattern_confidence_human: float
    mean_pattern_confidence_iid: float


@dataclass(frozen=True)
class ModelPerformance:
    """Per-generator performance breakdown for deeper analysis."""
    model_name: str
    is_human: bool
    n_sequences: int
    avg_log_loss: float
    avg_brier_score: float
    mean_final_detection_confidence: float
    final_positive_rate: float


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration diagnostics for next-step probabilistic predictions."""
    expected_calibration_error: float
    max_calibration_error: float
    reliability: float
    resolution: float
    uncertainty: float


@dataclass(frozen=True)
class ChangepointPerformance:
    """Localization and delay metrics on synthetic changepoint sequences."""
    n_sequences: int
    detection_rate: float
    localized_detection_rate: float
    false_alarm_rate: float
    mean_detection_delay: float
    mean_absolute_error: float
    mean_peak_probability: float


@dataclass(frozen=True)
class ValidationMetrics:
    """Results from a validation run."""
    brier_score: float
    log_loss: float
    log_loss_baseline: float   # IID Bernoulli baseline for comparison
    detection_power: dict[int, float]   # n -> P(pi_n > 0.75 | patterned)
    false_positive_rate: dict[int, float]  # n -> P(pi_n > 0.75 | IID)
    n_sequences: int
    n_human: int
    n_iid: int
    checkpoint_metrics: dict[int, CheckpointMetrics] = field(default_factory=dict)
    per_model_metrics: dict[str, ModelPerformance] = field(default_factory=dict)
    calibration: CalibrationMetrics | None = None
    changepoint_performance: ChangepointPerformance | None = None

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
        lines.append(f"    Baseline log-loss:   {self.log_loss_baseline:.4f}  (naive always-baseline guess)")
        ll_improvement = self.log_loss_baseline - self.log_loss
        pct = ll_improvement / max(self.log_loss_baseline, 1e-10) * 100
        if ll_improvement > 0:
            lines.append(f"    --> Model is {pct:.1f}% better than the naive baseline")
        else:
            lines.append(f"    --> Model is {abs(pct):.1f}% worse than baseline (needs tuning)")

        if self.calibration is not None:
            cal = self.calibration
            lines.append("")
            lines.append("  CALIBRATION")
            lines.append("  " + "-" * 40)
            lines.append(f"    ECE:                 {cal.expected_calibration_error:.4f}")
            lines.append(f"    Max calibration err: {cal.max_calibration_error:.4f}")
            lines.append(
                f"    Brier decomposition: reliability={cal.reliability:.4f}  "
                f"resolution={cal.resolution:.4f}  uncertainty={cal.uncertainty:.4f}"
            )

        # Detection power
        lines.append("")
        lines.append(f"  DETECTION POWER (sensitivity to patterned sequences)")
        lines.append("  " + "-" * 40)
        for n in sorted(self.detection_power.keys()):
            power = self.detection_power[n]
            bar_len = int(power * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            lines.append(f"    After {n:3d} observations: {bar} {power * 100:.1f}%")

        # False positive rate
        lines.append("")
        lines.append(f"  FALSE ALARM RATE (false positives on random sequences)")
        lines.append("  " + "-" * 40)
        for n in sorted(self.false_positive_rate.keys()):
            fpr = self.false_positive_rate[n]
            status = "PASS (< 5%)" if fpr < 0.05 else "FAIL (exceeds 5% threshold)"
            lines.append(f"    After {n:3d} observations: {fpr * 100:.1f}%  {status}")

        if self.checkpoint_metrics:
            lines.append("")
            lines.append("  CHECKPOINT CLASSIFICATION METRICS")
            lines.append("  " + "-" * 40)
            for n in sorted(self.checkpoint_metrics.keys()):
                cp = self.checkpoint_metrics[n]
                lines.append(
                    f"    n={n:3d}: TP={cp.true_positive} FP={cp.false_positive} "
                    f"TN={cp.true_negative} FN={cp.false_negative}"
                )
                lines.append(
                    " " * 10
                    + f"Recall/Power={cp.true_positive_rate * 100:.1f}%  "
                    + f"FNR={cp.false_negative_rate * 100:.1f}%  "
                    + f"Specificity={cp.true_negative_rate * 100:.1f}%  "
                    + f"Precision={cp.precision * 100:.1f}%  "
                    + f"F1={cp.f1_score:.3f}  "
                    + f"Acc={cp.accuracy * 100:.1f}%  "
                    + f"BalAcc={cp.balanced_accuracy * 100:.1f}%  "
                    + f"AUC={cp.roc_auc:.3f}"
                )
                lines.append(
                    " " * 10
                    + f"Mean confidence: patterned={cp.mean_pattern_confidence_human:.3f}, "
                    + f"IID={cp.mean_pattern_confidence_iid:.3f}"
                )

        if self.per_model_metrics:
            lines.append("")
            lines.append("  PER-MODEL BREAKDOWN")
            lines.append("  " + "-" * 40)
            for name in sorted(self.per_model_metrics.keys()):
                model = self.per_model_metrics[name]
                label = "patterned" if model.is_human else "iid"
                rate_label = "detect" if model.is_human else "false alarm"
                lines.append(
                    f"    {name:18s} [{label}]  log-loss={model.avg_log_loss:.4f}  "
                    f"brier={model.avg_brier_score:.4f}  final_conf={model.mean_final_detection_confidence:.3f}  "
                    f"final_{rate_label}={model.final_positive_rate * 100:.1f}%"
                )

        if self.changepoint_performance is not None:
            cp = self.changepoint_performance
            lines.append("")
            lines.append("  CHANGEPOINT PERFORMANCE")
            lines.append("  " + "-" * 40)
            lines.append(f"    Sequences:           {cp.n_sequences}")
            lines.append(f"    Detection rate:      {cp.detection_rate * 100:.1f}%")
            lines.append(f"    Localized (±5):      {cp.localized_detection_rate * 100:.1f}%")
            lines.append(f"    False-alarm rate:    {cp.false_alarm_rate * 100:.1f}%")
            lines.append(f"    Mean delay:          {cp.mean_detection_delay:.2f}")
            lines.append(f"    Mean abs. error:     {cp.mean_absolute_error:.2f}")
            lines.append(f"    Mean peak cp-prob:   {cp.mean_peak_probability:.3f}")

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
        baseline_prob: float = 0.50,
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
                baseline_prob=baseline_prob,
                seed=seed,
            )

        # Accumulators
        brier_scores: list[float] = []
        log_losses: list[float] = []
        log_losses_baseline: list[float] = []
        all_predictions: list[float] = []
        all_outcomes: list[int] = []
        per_model_stats: dict[str, dict[str, list[float] | bool | int]] = {}
        changepoint_stats = {
            "n_sequences": 0,
            "detected": 0,
            "localized": 0,
            "false_alarms": 0,
            "delays": [],
            "absolute_errors": [],
            "peak_probs": [],
        }

        # Detection at checkpoints
        checkpoint_results: dict[int, list[tuple[float, bool]]] = {
            cp: [] for cp in self.checkpoints
        }

        for gen_seq in dataset:
            seq = gen_seq.sequence
            p = gen_seq.baseline_prob
            is_human = gen_seq.is_human
            model_name = gen_seq.model_name

            engine = PredictionEngine(default_baseline_prob=p)
            cp_prob_timeline: list[float] = []

            if model_name not in per_model_stats:
                per_model_stats[model_name] = {
                    "is_human": is_human,
                    "n_sequences": 0,
                    "log_losses": [],
                    "brier_scores": [],
                    "final_confidences": [],
                }

            model_bucket = per_model_stats[model_name]
            model_bucket["n_sequences"] = int(model_bucket["n_sequences"]) + 1

            for i, action in enumerate(seq):
                hand_num = i + 1

                # Get prediction *before* observing the action
                pred = engine.predict_next()

                # Brier score: (pred - actual)^2
                brier = (pred - action) ** 2
                brier_scores.append(brier)
                all_predictions.append(pred)
                all_outcomes.append(int(action))
                model_bucket["brier_scores"].append(brier)  # type: ignore[index]

                # Log-loss
                pred_clamped = max(min(pred, 1.0 - 1e-10), 1e-10)
                if action == 1:
                    log_loss = -math.log(pred_clamped)
                else:
                    log_loss = -math.log(1.0 - pred_clamped)
                log_losses.append(log_loss)
                model_bucket["log_losses"].append(log_loss)  # type: ignore[index]

                # Baseline log-loss (IID Bernoulli)
                p_clamped = max(min(p, 1.0 - 1e-10), 1e-10)
                if action == 1:
                    log_losses_baseline.append(-math.log(p_clamped))
                else:
                    log_losses_baseline.append(-math.log(1.0 - p_clamped))

                # Process the observation
                pkg = engine.process_action(action)
                cp_prob_timeline.append(engine._bocpd.changepoint_probability)

                # Record detection at checkpoints
                if hand_num in checkpoint_results:
                    checkpoint_results[hand_num].append(
                        (pkg.detection_confidence, is_human)
                    )

            if model_name == "changepoint":
                true_cp = int(gen_seq.parameters.get("changepoint_at", len(seq) // 2))
                detections = engine._bocpd.changepoints_detected

                changepoint_stats["n_sequences"] += 1
                changepoint_stats["peak_probs"].append(
                    max(cp_prob_timeline[max(0, true_cp - 6): min(len(cp_prob_timeline), true_cp + 5)], default=0.0)
                )  # type: ignore[index]
                if any(d < true_cp for d in detections):
                    changepoint_stats["false_alarms"] += 1

                post_change = [d for d in detections if d >= true_cp]
                if post_change:
                    first = post_change[0]
                    delay = first - true_cp
                    changepoint_stats["detected"] += 1
                    changepoint_stats["delays"].append(delay)  # type: ignore[index]
                    changepoint_stats["absolute_errors"].append(abs(delay))  # type: ignore[index]
                    if abs(delay) <= 5:
                        changepoint_stats["localized"] += 1

            model_bucket["final_confidences"].append(engine.detection_confidence)  # type: ignore[index]

        # Compute aggregate metrics
        brier = float(np.mean(brier_scores))
        ll = float(np.mean(log_losses))
        ll_base = float(np.mean(log_losses_baseline))
        ece, mce = _expected_calibration_error(all_predictions, all_outcomes)
        reliability, resolution, uncertainty = _brier_decomposition(all_predictions, all_outcomes)

        detection_power: dict[int, float] = {}
        false_positive_rate: dict[int, float] = {}
        checkpoint_metrics: dict[int, CheckpointMetrics] = {}

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

            tp = sum(1 for pi in human_results if pi > self.detection_threshold)
            fn = len(human_results) - tp
            fp = sum(1 for pi in iid_results if pi > self.detection_threshold)
            tn = len(iid_results) - fp

            tpr = _safe_div(tp, tp + fn)
            fpr = _safe_div(fp, fp + tn)
            fnr = _safe_div(fn, tp + fn)
            tnr = _safe_div(tn, fp + tn)
            precision = _safe_div(tp, tp + fp)
            npv = _safe_div(tn, tn + fn)
            accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
            balanced_accuracy = 0.5 * (tpr + tnr)
            f1 = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
            scores = human_results + iid_results
            labels = [1] * len(human_results) + [0] * len(iid_results)

            checkpoint_metrics[cp] = CheckpointMetrics(
                checkpoint=cp,
                true_positive=tp,
                false_positive=fp,
                true_negative=tn,
                false_negative=fn,
                true_positive_rate=tpr,
                false_positive_rate=fpr,
                false_negative_rate=fnr,
                true_negative_rate=tnr,
                precision=precision,
                negative_predictive_value=npv,
                accuracy=accuracy,
                balanced_accuracy=balanced_accuracy,
                f1_score=f1,
                roc_auc=_roc_auc(scores, labels),
                mean_pattern_confidence_human=float(np.mean(human_results)) if human_results else 0.0,
                mean_pattern_confidence_iid=float(np.mean(iid_results)) if iid_results else 0.0,
            )

        per_model_metrics: dict[str, ModelPerformance] = {}
        for model_name, stats in per_model_stats.items():
            final_confidences = stats["final_confidences"]  # type: ignore[index]
            per_model_metrics[model_name] = ModelPerformance(
                model_name=model_name,
                is_human=bool(stats["is_human"]),
                n_sequences=int(stats["n_sequences"]),
                avg_log_loss=float(np.mean(stats["log_losses"])),  # type: ignore[arg-type]
                avg_brier_score=float(np.mean(stats["brier_scores"])),  # type: ignore[arg-type]
                mean_final_detection_confidence=float(np.mean(final_confidences)),  # type: ignore[arg-type]
                final_positive_rate=float(
                    np.mean([1.0 if pi > self.detection_threshold else 0.0 for pi in final_confidences])  # type: ignore[arg-type]
                ),
            )

        n_human = sum(1 for s in dataset if s.is_human)
        n_iid = sum(1 for s in dataset if not s.is_human)

        calibration = CalibrationMetrics(
            expected_calibration_error=ece,
            max_calibration_error=mce,
            reliability=reliability,
            resolution=resolution,
            uncertainty=uncertainty,
        )

        n_cp = int(changepoint_stats["n_sequences"])
        changepoint_performance = None
        if n_cp > 0:
            delays = changepoint_stats["delays"]  # type: ignore[assignment]
            absolute_errors = changepoint_stats["absolute_errors"]  # type: ignore[assignment]
            peak_probs = changepoint_stats["peak_probs"]  # type: ignore[assignment]
            changepoint_performance = ChangepointPerformance(
                n_sequences=n_cp,
                detection_rate=float(changepoint_stats["detected"]) / n_cp,
                localized_detection_rate=float(changepoint_stats["localized"]) / n_cp,
                false_alarm_rate=float(changepoint_stats["false_alarms"]) / n_cp,
                mean_detection_delay=float(np.mean(delays)) if delays else 0.0,
                mean_absolute_error=float(np.mean(absolute_errors)) if absolute_errors else 0.0,
                mean_peak_probability=float(np.mean(peak_probs)) if peak_probs else 0.0,
            )

        return ValidationMetrics(
            brier_score=brier,
            log_loss=ll,
            log_loss_baseline=ll_base,
            detection_power=detection_power,
            false_positive_rate=false_positive_rate,
            n_sequences=len(dataset),
            n_human=n_human,
            n_iid=n_iid,
            checkpoint_metrics=checkpoint_metrics,
            per_model_metrics=per_model_metrics,
            calibration=calibration,
            changepoint_performance=changepoint_performance,
        )
