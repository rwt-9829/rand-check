"""Command-line interface for rand-check.

A CLI for detecting sequential patterns in binary (0/1) sequences.
Accessible without prior statistical knowledge.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time


# -- Friendly model names -> technical names ---------------------------
MODEL_ALIASES = {
    "random":       "iid",
    "alternator":   "markov_alternation",
    "gamblers":     "gamblers_fallacy",
    "counter":      "counter",
    "streak-averse": "run_averse",
    "mixed":        "mixture",
    "shifter":      "changepoint",
    # Also accept the raw technical names
    "iid":                  "iid",
    "markov_alternation":   "markov_alternation",
    "gamblers_fallacy":     "gamblers_fallacy",
    "run_averse":           "run_averse",
    "mixture":              "mixture",
    "changepoint":          "changepoint",
}

MODEL_DESCRIPTIONS = {
    "random":        "Truly random (IID Bernoulli -- no exploitable structure)",
    "alternator":    "Over-alternates between 0 and 1 (most common cognitive bias)",
    "gamblers":      "Avoids repeating recent outcomes -- the gambler's fallacy",
    "counter":       "Tracks cumulative frequency and self-corrects toward a target rate",
    "streak-averse": "Truncates runs -- avoids producing the same value consecutively",
    "mixed":         "Alternates randomly between two underlying strategies",
    "shifter":       "Follows one regime, then shifts abruptly mid-sequence",
}

FRIENDLY_NAMES = list(MODEL_DESCRIPTIONS.keys())


def _resolve_model(name: str) -> str:
    """Turn a friendly or technical model name into the internal name."""
    key = name.lower().replace("_", "-").strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    # Fuzzy match: check if it's a prefix
    matches = [k for k in MODEL_ALIASES if k.startswith(key)]
    if len(matches) == 1:
        return MODEL_ALIASES[matches[0]]
    valid = ", ".join(FRIENDLY_NAMES)
    print(f"Unknown model type '{name}'. Choose from: {valid}")
    sys.exit(1)


def _friendly_model_name(technical: str) -> str:
    """Get the user-friendly name for a technical model name."""
    reverse = {v: k for k, v in MODEL_ALIASES.items() if k in MODEL_DESCRIPTIONS}
    return reverse.get(technical, technical)


# -- Banner & help -----------------------------------------------------

WELCOME = r"""
  rand-check -- Binary Sequence Pattern Detector

  Determine whether a binary sequence is truly random
  or exhibits exploitable cognitive patterns.

  COMMANDS
  --------
  rand-check demo               See a live example with a simulated sequence
  rand-check analyze SEQUENCE   Analyze a real binary sequence
  rand-check simulate TYPE N    Generate a fake sequence for testing
  rand-check validate           Run accuracy tests on the detection engine

  QUICK START
  -----------
  1. Run  rand-check demo  to see how it works.
  2. Record your binary decisions as 1s and 0s.
  3. Run  rand-check analyze 010100101001  to get a report.

  Use  rand-check <command> --help  for details on any command.
"""


class _PreservingHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Help formatter that preserves newlines and indentation."""
    pass


# -- Parsing helpers ---------------------------------------------------

def _parse_sequence(raw: str) -> list[int]:
    """Parse a sequence from various user-friendly formats.

    Accepts:
        "010110010"         binary string
        "0 1 0 1 1 0 0 1"  space-separated
        "0,1,0,1,1,0,0,1"  comma-separated
    """
    s = raw.strip().strip('"').strip("'")

    # Remove separators
    s = s.replace(",", "").replace(" ", "").replace("-", "")

    if not s:
        print("Error: empty sequence.")
        sys.exit(1)
    if not all(c in "01" for c in s):
        print(f"Error: sequence must contain only 0s and 1s.")
        print(f"  Got: {raw[:60]}{'...' if len(raw) > 60 else ''}")
        print(f"\n  Examples:")
        print(f"    rand-check analyze 010100101001")
        print(f"    rand-check analyze \"0 1 0 1 0 0 1 0\"")
        sys.exit(1)

    return [int(c) for c in s]


# -- Main entry point --------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rand-check",
        description="Detect patterns in binary (0/1) sequences.",
        formatter_class=_PreservingHelpFormatter,
        add_help=True,
    )
    sub = parser.add_subparsers(dest="command")

    # -- demo ----------------------------------------------------------
    model_list = "\n".join(
        f"    {name:15s} {desc}" for name, desc in MODEL_DESCRIPTIONS.items()
    )
    demo_p = sub.add_parser(
        "demo",
        help="Watch the engine detect patterns in a simulated sequence",
        formatter_class=_PreservingHelpFormatter,
        epilog=textwrap.dedent(f"""\
        model types (--model):
        {model_list}

        examples:
          rand-check demo
          rand-check demo --model gamblers --length 120
        """),
    )
    demo_p.add_argument(
        "--length", "-n", type=int, default=80,
        help="Number of observations to simulate (default: 80)",
    )
    demo_p.add_argument(
        "--baseline", "-p", type=float, default=0.50,
        help="Baseline probability of outcome 1 (default: 0.50)",
    )
    demo_p.add_argument(
        "--model", default="alternator",
        metavar="TYPE",
        help=f"Type of simulated pattern (default: alternator). "
             f"Choices: {', '.join(FRIENDLY_NAMES)}",
    )
    demo_p.add_argument(
        "--detailed", action="store_true",
        help="Show full technical detail (for advanced users)",
    )

    # -- analyze -------------------------------------------------------
    ana_p = sub.add_parser(
        "analyze",
        help="Analyze a binary sequence for patterns",
        formatter_class=_PreservingHelpFormatter,
        epilog=textwrap.dedent("""\
        input formats:
          Binary string:    rand-check analyze 010100101001
          With spaces:      rand-check analyze "0 1 0 1 0 0 1 0"
          Comma-separated:  rand-check analyze "0,1,0,1,0,0,1,0"

        examples:
          rand-check analyze 010100101001010010100101001010
          rand-check analyze "1,0,1,0,0,1" --baseline 0.30
        """),
    )
    ana_p.add_argument(
        "sequence", type=str,
        help="Sequence of binary values: 0s and 1s",
    )
    ana_p.add_argument(
        "--baseline", "-p", type=float, default=0.50,
        help="Baseline probability of outcome 1 (default: 0.50)",
    )
    ana_p.add_argument(
        "--detailed", action="store_true",
        help="Show full technical detail",
    )

    # -- simulate ------------------------------------------------------
    sim_p = sub.add_parser(
        "simulate",
        help="Generate a fake sequence for testing",
        formatter_class=_PreservingHelpFormatter,
        epilog=textwrap.dedent(f"""\
        model types:
        {model_list}

        examples:
          rand-check simulate alternator 100
          rand-check simulate gamblers 200 --seed 42
          rand-check simulate random 80
        """),
    )
    sim_p.add_argument(
        "model", metavar="TYPE",
        help=f"Type of pattern. Choices: {', '.join(FRIENDLY_NAMES)}",
    )
    sim_p.add_argument("length", type=int, metavar="N", help="Number of values to generate")
    sim_p.add_argument(
        "--baseline", "-p", type=float, default=0.50,
        help="Baseline probability of outcome 1 (default: 0.50)",
    )
    sim_p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # -- validate ------------------------------------------------------
    val_p = sub.add_parser(
        "validate",
        help="Test the detection engine's accuracy",
        formatter_class=_PreservingHelpFormatter,
        epilog=textwrap.dedent("""\
        Generates many fake sequences and checks how well the engine
        detects patterns. Reports accuracy and false-positive rates.

        examples:
          rand-check validate
          rand-check validate --sequences 200 --length 150
        """),
    )
    val_p.add_argument(
        "--sequences", "--n-per-model", type=int, default=50,
        help="Number of test sequences per model type (default: 50)",
    )
    val_p.add_argument(
        "--length", "--seq-length", type=int, default=80,
        help="Observations per sequence (default: 80)",
    )
    val_p.add_argument(
        "--baseline", "-p", type=float, default=0.50,
        help="Baseline probability of outcome 1 (default: 0.50)",
    )
    val_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args(argv)

    if args.command == "demo":
        _run_demo(args)
    elif args.command == "validate":
        _run_validate(args)
    elif args.command == "analyze":
        _run_analyze(args)
    elif args.command == "simulate":
        _run_simulate(args)
    else:
        print(WELCOME)


# -- Formatting helpers ------------------------------------------------

def _bar(value: float, width: int = 20, fill: str = "#", empty: str = ".") -> str:
    """Render a progress bar."""
    filled = int(value * width)
    return fill * filled + empty * (width - filled)


def _pct(value: float) -> str:
    """Format as percentage."""
    return f"{value * 100:.1f}%"


def _verdict_icon(pi: float) -> str:
    """Return a verdict indicator based on detection confidence."""
    if pi < 0.3:
        return "[OK]"
    elif pi < 0.6:
        return "[??]"
    elif pi < 0.85:
        return "[!]"
    else:
        return "[!!]"


def _confidence_label(pi: float) -> str:
    """Plain-English detection confidence label."""
    if pi < 0.3:
        return "Consistent with randomness"
    elif pi < 0.6:
        return "Weak signal -- more data needed"
    elif pi < 0.85:
        return "Probable pattern detected"
    else:
        return "High-confidence pattern detected"


# -- Command implementations -------------------------------------------

def _run_demo(args: argparse.Namespace) -> None:
    import numpy as np
    from rand_check.prediction import PredictionEngine
    from rand_check.analyzer import PostSessionAnalyzer
    from rand_check import synthetic

    model_name = _resolve_model(args.model)
    friendly_name = _friendly_model_name(model_name)
    n = args.length
    p = args.baseline

    rng = np.random.default_rng(42)
    generators = {
        "iid": lambda: synthetic.generate_iid_bernoulli(n, p, rng),
        "markov_alternation": lambda: synthetic.generate_markov_alternation(n, p, 0.62, rng),
        "gamblers_fallacy": lambda: synthetic.generate_gamblers_fallacy(n, p, rng=rng),
        "counter": lambda: synthetic.generate_counter_model(n, p, rng=rng),
        "run_averse": lambda: synthetic.generate_run_averse(n, p, rng=rng),
        "mixture": lambda: synthetic.generate_mixture(n, p, rng=rng),
        "changepoint": lambda: synthetic.generate_changepoint(n, p, rng=rng),
    }

    gen_seq = generators[model_name]()
    seq = gen_seq.sequence

    truth = "HAS PATTERNS" if gen_seq.is_human else "TRULY RANDOM"

    print()
    print("  +" + "-"*60 + "+")
    print(f"  |  DEMO: Watching a \"{friendly_name}\" sequence")
    print(f"  |  Length: {n}   Baseline P(1): {_pct(p)}")
    print(f"  |  Ground truth: {truth}")
    print("  +" + "-"*60 + "+")
    print()
    print(f"  The engine processes each observation to determine")
    print(f"  whether the sequence is random or structurally patterned.")
    print()

    engine = PredictionEngine(default_baseline_prob=p)

    if args.detailed:
        # Full technical table for advanced users
        print(f"  {'#':>5} | {'Val':>3} | {'Detect':>8} | {'Next P':>8} | {'Base':>6} | {'Edge':>7} | {'95% CI':>14} | Status")
        print("  " + "-" * 85)

        for i, action in enumerate(seq):
            pkg = engine.process_action(action)
            ci_str = f"[{pkg.confidence_interval[0]:.2f},{pkg.confidence_interval[1]:.2f}]"
            flag = " <-- regime change" if pkg.changepoint_flag else ""
            print(
                f"  {i+1:5d} |   {action} | "
                f"{pkg.detection_confidence:8.3f} | {pkg.predicted_prob:8.3f} | "
                f"{pkg.baseline_prob:6.3f} | {pkg.edge:+7.3f} | {ci_str:>14} | "
                f"{engine._detector.interpret()}{flag}"
            )
    else:
        # Simplified, friendly output
        print(f"  {'#':>5} | {'Value':>5} | {'Pattern?':>24} | {'Next P(1)':>10} | Status")
        print("  " + "-" * 75)

        milestones = {20, 40, 60, n}
        for i, action in enumerate(seq):
            pkg = engine.process_action(action)
            detect_bar = _bar(pkg.detection_confidence, width=15)
            detect_pct = _pct(pkg.detection_confidence)
            next_pct = _pct(pkg.predicted_prob)
            flag = " <-- regime change detected" if pkg.changepoint_flag else ""

            print(
                f"  {i+1:5d} |     {action} | {detect_bar} {detect_pct:>5} | {next_pct:>10} | "
                f"{_confidence_label(pkg.detection_confidence)}{flag}"
            )

            # Print milestone summaries
            obs_num = i + 1
            if obs_num in milestones and obs_num < n:
                pi = pkg.detection_confidence
                icon = _verdict_icon(pi)
                print()
                print(f"  {icon} After {obs_num} observations: {_confidence_label(pi)} (confidence: {_pct(pi)})")
                if pi > 0.6:
                    print(f"      Assessment: {pkg.suggestion}")
                print()

    # Final summary
    final_pkg = engine._build_package()

    pi = final_pkg.detection_confidence
    icon = _verdict_icon(pi)

    print()
    print("  +" + "-"*60 + "+")
    print(f"  |  {icon} RESULT AFTER {n} OBSERVATIONS")
    print("  |")
    print(f"  |  Detection:    {_confidence_label(pi)}")
    print(f"  |  Confidence:   {_bar(pi)} {_pct(pi)}")
    print(f"  |  Next P(1):    {_pct(final_pkg.predicted_prob)}  (baseline: {_pct(final_pkg.baseline_prob)})")
    print(f"  |  Edge:         {final_pkg.edge:+.3f}")
    print("  +" + "-"*60 + "+")
    if final_pkg.suggestion:
        print(f"\n  --> {final_pkg.suggestion}")

    print(f"\n  {'='*60}")
    print("  POST-SESSION ANALYSIS")
    print(f"  {'='*60}\n")

    analyzer = PostSessionAnalyzer()
    report = analyzer.analyze(seq, baseline_prob=p)
    print(report.summary(detailed=args.detailed))


def _run_validate(args: argparse.Namespace) -> None:
    from rand_check.validation import ValidationRunner

    n_per = args.sequences
    seq_len = args.length
    p = args.baseline
    print()
    print(f"  Evaluating detection engine performance...")
    print(f"  {n_per} sequences × 7 model types × {seq_len} observations each")
    print(f"  Baseline P(1): {_pct(p)}")
    print(f"  This may take a moment.\n")

    runner = ValidationRunner()
    t0 = time.time()
    metrics = runner.run(
        n_per_model=n_per,
        seq_length=seq_len,
        baseline_prob=p,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print(metrics.summary())
    print(f"\n  Completed in {elapsed:.1f}s")


def _run_analyze(args: argparse.Namespace) -> None:
    from rand_check.analyzer import PostSessionAnalyzer

    seq = _parse_sequence(args.sequence)
    if len(seq) < 5:
        print("Error: need at least 5 values to analyze.")
        print("  Record more data and try again.")
        sys.exit(1)

    n = len(seq)
    ones = sum(seq)
    zeros = n - ones
    rate = ones / n
    p = args.baseline

    print()
    print("  +" + "-"*60 + "+")
    print(f"  |  ANALYZING {n} OBSERVATIONS")
    print("  |")
    print(f"  |  1s: {ones}   0s: {zeros}   Rate of 1s: {_pct(rate)}")
    print(f"  |  Baseline rate: {_pct(p)}")

    if rate > p + 0.05:
        print(f"  |  --> Observed rate exceeds baseline ({_pct(rate)} vs {_pct(p)})")
    elif rate < p - 0.05:
        print(f"  |  --> Observed rate falls below baseline ({_pct(rate)} vs {_pct(p)})")
    else:
        print(f"  |  --> Observed rate is consistent with baseline")

    print("  +" + "-"*60 + "+")
    print()

    analyzer = PostSessionAnalyzer()
    report = analyzer.analyze(seq, baseline_prob=p)
    print(report.summary(detailed=args.detailed))


def _run_simulate(args: argparse.Namespace) -> None:
    import numpy as np
    from rand_check import synthetic

    model_name = _resolve_model(args.model)
    n = args.length
    p = args.baseline
    rng = np.random.default_rng(args.seed)
    generators = {
        "iid": lambda: synthetic.generate_iid_bernoulli(n, p, rng),
        "markov_alternation": lambda: synthetic.generate_markov_alternation(n, p, 0.62, rng),
        "gamblers_fallacy": lambda: synthetic.generate_gamblers_fallacy(n, p, rng=rng),
        "counter": lambda: synthetic.generate_counter_model(n, p, rng=rng),
        "run_averse": lambda: synthetic.generate_run_averse(n, p, rng=rng),
        "mixture": lambda: synthetic.generate_mixture(n, p, rng=rng),
        "changepoint": lambda: synthetic.generate_changepoint(n, p, rng=rng),
    }

    gen_seq = generators[model_name]()
    seq_str = "".join(str(x) for x in gen_seq.sequence)
    print(seq_str)


if __name__ == "__main__":
    main()
