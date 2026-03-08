"""Post-session analyzer -- rich analysis of a completed binary sequence.

Uses the full feature vector to produce a comprehensive report,
including pattern fingerprinting (which cognitive bias dominates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rand_check.features import FeatureVector, compute_features
from rand_check.prediction import PredictionEngine
from rand_check.models import AnalysisPackage


@dataclass(frozen=True)
class PatternFingerprint:
    """Identifies which cognitive bias dominates the subject's behaviour."""
    primary_bias: str
    confidence: float        # 0-1
    bias_scores: dict[str, float]
    description: str


@dataclass
class SessionReport:
    """Complete post-session analysis report."""
    features: FeatureVector
    fingerprint: PatternFingerprint
    final_prediction: AnalysisPackage
    timeline: list[float]       # pi_n at each observation
    prediction_timeline: list[float]  # q_n at each observation
    baseline_prob: float
    n_observations: int
    changepoints: list[int]

    def summary(self, detailed: bool = False) -> str:
        """Generate a formatted summary string.

        Parameters
        ----------
        detailed : bool
            If True, include full technical metrics. If False (default),
            show a simplified, plain-English report.
        """
        if detailed:
            return self._detailed_summary()
        return self._friendly_summary()

    def _friendly_summary(self) -> str:
        """Plain-English summary for non-technical users."""
        lines = []
        pi = self.final_prediction.detection_confidence
        pred = self.final_prediction.predicted_prob
        edge = self.final_prediction.edge

        # Verdict
        if pi < 0.3:
            verdict = "VERDICT: Sequence is consistent with genuine randomness."
            verdict_detail = "No exploitable pattern detected."
        elif pi < 0.6:
            verdict = "VERDICT: Weak evidence of patterning -- insufficient data for certainty."
            verdict_detail = "Additional observations will improve confidence."
        elif pi < 0.85:
            verdict = "VERDICT: Probable sequential pattern detected."
            verdict_detail = "Sequential dependencies have been identified."
        else:
            verdict = "VERDICT: High-confidence pattern detected."
            verdict_detail = "This sequence strongly deviates from random generation."

        lines.append("  " + "=" * 58)
        lines.append("  SESSION REPORT")
        lines.append("  " + "=" * 58)
        lines.append("")
        lines.append(f"  {verdict}")
        lines.append(f"  {verdict_detail}")
        lines.append("")
        lines.append(f"  Observations analyzed: {self.n_observations}")
        lines.append(f"  Pattern confidence:    {pi * 100:.1f}%")

        # Prediction
        lines.append("")
        lines.append(f"  Predicted next P(1):  {pred * 100:.1f}%")
        lines.append(f"  Baseline P(1):        {self.baseline_prob * 100:.1f}%")
        if abs(edge) > 0.03:
            direction = "more 1s" if edge > 0 else "fewer 1s"
            lines.append(f"  --> Subject produces {direction} than baseline ({abs(edge) * 100:.1f}% deviation)")

        # Strategy shifts
        if self.changepoints:
            lines.append("")
            shifts = ", ".join(str(h) for h in self.changepoints)
            lines.append(f"  Regime changes detected at observation(s): {shifts}")
            lines.append("  (A behavioural shift was detected mid-sequence)")

        # Pattern fingerprint
        lines.append("")
        lines.append("  DETECTED BIAS")
        lines.append("  " + "-" * 40)

        fp = self.fingerprint
        bias_labels = {
            "alternation_bias": "Alternation bias",
            "gamblers_fallacy": "Gambler's fallacy",
            "run_aversion": "Streak aversion",
            "frequency_tracking": "Frequency tracking",
            "compressible_pattern": "Repeating pattern",
            "none_detected": "None detected",
        }

        if fp.primary_bias == "none_detected":
            lines.append("  No significant bias found.")
        else:
            label = bias_labels.get(fp.primary_bias, fp.primary_bias)
            lines.append(f"  Primary bias: {label} ({fp.confidence * 100:.0f}% match)")
            lines.append(f"  {fp.description}")

        if fp.bias_scores:
            lines.append("")
            lines.append("  Bias decomposition:")
            for bias, score in sorted(fp.bias_scores.items(), key=lambda x: -x[1]):
                label = bias_labels.get(bias, bias)
                bar_len = int(score * 20)
                bar = "#" * bar_len + "." * (20 - bar_len)
                lines.append(f"    {label:22s} {bar} {score * 100:.0f}%")

        # Suggestion
        lines.append("")
        sugg = self.final_prediction.suggestion
        if sugg:
            lines.append(f"  ASSESSMENT: {sugg}")

        return "\n".join(lines)

    def _detailed_summary(self) -> str:
        """Full technical summary for advanced users."""
        lines = []
        lines.append("  " + "=" * 58)
        lines.append("  POST-SESSION ANALYSIS (DETAILED)")
        lines.append("  " + "=" * 58)
        lines.append("")
        lines.append(f"  Observations analyzed: {self.n_observations}")
        lines.append(f"  Baseline P: {self.baseline_prob:.3f}")
        lines.append(f"  Detection confidence (pi_n): {self.final_prediction.detection_confidence:.3f}")
        lines.append(f"  Predicted P(1) (q_n):        {self.final_prediction.predicted_prob:.3f}")
        lines.append(f"  Edge over baseline: {self.final_prediction.edge:+.3f}")
        lines.append("")

        if self.changepoints:
            lines.append(f"  Changepoints detected at observations: {self.changepoints}")
        else:
            lines.append("  No changepoints detected")

        lines.append("")
        lines.append(self.features.detection_summary(self.baseline_prob, detailed=True))
        lines.append("")
        lines.append("  Pattern Fingerprint:")
        lines.append(f"    Primary bias: {self.fingerprint.primary_bias}")
        lines.append(f"    Confidence:   {self.fingerprint.confidence:.2f}")
        lines.append(f"    {self.fingerprint.description}")
        lines.append("")

        if self.fingerprint.bias_scores:
            lines.append("    Bias breakdown:")
            for bias, score in sorted(self.fingerprint.bias_scores.items(),
                                       key=lambda x: -x[1]):
                bar_len = int(score * 20)
                bar = "#" * bar_len
                lines.append(f"      {bias:20s}: {score:.2f} {bar}")

        lines.append("")
        lines.append(f"  Assessment: {self.final_prediction.suggestion}")

        return "\n".join(lines)


class PostSessionAnalyzer:
    """Analyze a completed binary session sequence."""

    def analyze(
        self,
        sequence: list[int],
        baseline_prob: float = 0.50,
    ) -> SessionReport:
        """Run full post-session analysis.

        Parameters
        ----------
        sequence : list[int]
            Complete binary session history.
        baseline_prob : float
            Baseline probability of outcome 1 under the null hypothesis.

        Returns
        -------
        SessionReport
        """
        n = len(sequence)

        # Compute features from the raw sequence
        features = compute_features(sequence, baseline_prob)

        # Run the prediction engine over the full sequence to get timelines
        engine = PredictionEngine(default_baseline_prob=baseline_prob)
        detection_timeline: list[float] = []
        prediction_timeline: list[float] = []

        pkg = None
        for action in sequence:
            pkg = engine.process_action(action)
            detection_timeline.append(pkg.detection_confidence)
            prediction_timeline.append(pkg.predicted_prob)

        if pkg is None:
            from rand_check.models import AnalysisPackage
            pkg = AnalysisPackage()

        changepoints = engine._bocpd.changepoints_detected

        # Compute pattern fingerprint
        fingerprint = self._compute_fingerprint(features, sequence, baseline_prob)

        return SessionReport(
            features=features,
            fingerprint=fingerprint,
            final_prediction=pkg,
            timeline=detection_timeline,
            prediction_timeline=prediction_timeline,
            baseline_prob=baseline_prob,
            n_observations=n,
            changepoints=changepoints,
        )

    def _compute_fingerprint(
        self, features: FeatureVector, sequence: list[int], baseline_prob: float
    ) -> PatternFingerprint:
        """Identify the dominant cognitive bias from the feature vector."""
        if len(sequence) < 10:
            return PatternFingerprint(
                primary_bias="none_detected",
                confidence=0.0,
                bias_scores={},
                description="Sequence is too short for a reliable bias fingerprint",
            )

        scores: dict[str, float] = {}

        # Alternation bias: high alternation deviation + negative serial correlation
        alt_signal = max(0.0, features.alternation_deviation / 0.20)
        neg_corr_signal = max(0.0, -features.serial_correlation / 0.30)
        scores["alternation_bias"] = min(1.0, (alt_signal + neg_corr_signal) / 2.0)

        # Gambler's fallacy: reversal pressure increases after longer streaks
        gf_signal = self._gambler_fallacy_score(sequence, baseline_prob)
        scores["gamblers_fallacy"] = gf_signal

        # Run aversion: very short max runs compared to expected
        runs = self._get_runs(sequence)
        if runs:
            max_run = max(r[1] for r in runs)
            q = max(baseline_prob, 1.0 - baseline_prob)
            n = len(sequence)
            if q > 0 and n > 1:
                expected_max = max(1.0, np.log(n) / (-np.log(q)))
                if max_run < expected_max * 0.5:
                    scores["run_aversion"] = min(1.0, (expected_max - max_run) / expected_max)
                else:
                    scores["run_aversion"] = 0.0
            else:
                scores["run_aversion"] = 0.0
        else:
            scores["run_aversion"] = 0.0

        # Frequency tracking / counter: low frequency drift = over-correcting
        fd_signal = max(0.0, 1.0 - features.frequency_drift / 0.01) if features.frequency_drift > 0 else 0.5
        scores["frequency_tracking"] = min(1.0, fd_signal * 0.5)

        # Pattern compressibility: low LZC
        if features.normalized_lz_complexity < 0.90:
            scores["compressible_pattern"] = min(1.0, (0.90 - features.normalized_lz_complexity) / 0.30)
        else:
            scores["compressible_pattern"] = 0.0

        # Determine primary bias
        if not scores or max(scores.values()) < 0.1:
            return PatternFingerprint(
                primary_bias="none_detected",
                confidence=0.0,
                bias_scores=scores,
                description="No significant cognitive bias detected -- consistent with RNG",
            )

        primary = max(scores, key=scores.get)  # type: ignore
        confidence = scores[primary]

        descriptions = {
            "alternation_bias": "Subject over-alternates between 0 and 1 -- a well-documented cognitive randomization artifact",
            "gamblers_fallacy": "Subject shows increasing reversal pressure after longer streaks -- gambler's fallacy / local representativeness",
            "run_aversion": "Subject avoids consecutive identical values -- truncated run lengths",
            "frequency_tracking": "Subject tracks cumulative frequency and self-corrects toward a target rate",
            "compressible_pattern": "Sequence is highly structured -- low Lempel-Ziv complexity indicates a repeating pattern",
        }

        return PatternFingerprint(
            primary_bias=primary,
            confidence=confidence,
            bias_scores=scores,
            description=descriptions.get(primary, ""),
        )

    @staticmethod
    def _gambler_fallacy_score(sequence: list[int], baseline_prob: float) -> float:
        """Score indicating gambler's fallacy behaviour.

        Check whether reversal probability increases with streak length for
        either symbol. This captures symmetric "reversal pressure" rather than
        only the special case of zeros followed by ones.
        """
        if len(sequence) < 10:
            return 0.0

        streak_reversal: dict[int, dict[int, list[int]]] = {0: {}, 1: {}}

        current_value = sequence[0]
        streak_len = 1
        for action in sequence[1:]:
            bucket = min(streak_len, 5)
            reversed_now = 1 if action != current_value else 0
            streak_reversal[current_value].setdefault(bucket, []).append(reversed_now)

            if action == current_value:
                streak_len += 1
            else:
                current_value = action
                streak_len = 1

        signal_strengths: list[float] = []
        for streak_value in (0, 1):
            rates = []
            for k in sorted(streak_reversal[streak_value].keys()):
                obs = streak_reversal[streak_value][k]
                if len(obs) >= 5:
                    rates.append((k, sum(obs) / len(obs)))

            if len(rates) < 2:
                continue

            low_rate = rates[0][1]
            high_rate = rates[-1][1]
            if high_rate > low_rate + 0.05:
                signal_strengths.append(min(1.0, (high_rate - low_rate) / 0.30))

        if not signal_strengths:
            return 0.0
        return float(sum(signal_strengths) / len(signal_strengths))

    @staticmethod
    def _get_runs(seq: list[int]) -> list[tuple[int, int]]:
        if not seq:
            return []
        runs: list[tuple[int, int]] = []
        current_val = seq[0]
        current_len = 1
        for i in range(1, len(seq)):
            if seq[i] == current_val:
                current_len += 1
            else:
                runs.append((current_val, current_len))
                current_val = seq[i]
                current_len = 1
        runs.append((current_val, current_len))
        return runs
