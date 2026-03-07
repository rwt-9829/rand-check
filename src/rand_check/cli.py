"""Command-line interface for rand-check.

A user-friendly CLI for detecting patterns in poker 3-bet decisions.
Designed so you don't need to know any statistics to use it.
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
    "random":        "Truly random (no pattern -- like a coin flip)",
    "alternator":    "Over-alternates between 3-bet and fold (most common human bias)",
    "gamblers":      "Avoids repeating recent outcomes (\"I just 3-bet, so I shouldn't again\")",
    "counter":       "Mentally tracks frequency and self-corrects (\"1 in 4 hands\")",
    "streak-averse": "Breaks up runs early -- never 3-bets several times in a row",
    "mixed":         "Switches randomly between two different strategies",
    "shifter":       "Plays one strategy, then abruptly changes mid-session",
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
    print(f"Unknown opponent type '{name}'. Choose from: {valid}")
    sys.exit(1)


def _friendly_model_name(technical: str) -> str:
    """Get the user-friendly name for a technical model name."""
    reverse = {v: k for k, v in MODEL_ALIASES.items() if k in MODEL_DESCRIPTIONS}
    return reverse.get(technical, technical)


# -- Banner & help -----------------------------------------------------

WELCOME = r"""
  rand-check -- Poker 3-Bet Pattern Detector

  Detect if your opponent is truly randomizing their 3-bets
  or following exploitable human patterns.

  COMMANDS
  --------
  rand-check demo               See a live example with a fake opponent
  rand-check analyze SEQUENCE   Analyze a real sequence of 3-bet decisions
  rand-check simulate TYPE N    Generate a fake sequence for testing
  rand-check validate           Run accuracy tests on the detection engine

  QUICK START
  -----------
  1. Run  rand-check demo  to see how it works.
  2. Record your opponent's decisions as 1s and 0s:
       1 = they 3-bet,  0 = they folded or called
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
        "BFFBFBFB"          B/F notation (B=3bet, F=fold)
    """
    s = raw.strip().strip('"').strip("'")

    # B/F notation
    if s and s[0].upper() in ("B", "F"):
        result = []
        for ch in s.upper():
            if ch == "B":
                result.append(1)
            elif ch == "F":
                result.append(0)
            elif ch in (" ", ",", "-"):
                continue
            else:
                print(f"Error: unexpected character '{ch}' in B/F sequence.")
                print("  Use B for 3-bet and F for fold, e.g.: BFFBFBFB")
                sys.exit(1)
        return result

    # Remove separators
    s = s.replace(",", "").replace(" ", "").replace("-", "")

    if not s:
        print("Error: empty sequence.")
        sys.exit(1)
    if not all(c in "01" for c in s):
        print(f"Error: sequence must contain only 0s and 1s (or B/F notation).")
        print(f"  Got: {raw[:60]}{'...' if len(raw) > 60 else ''}")
        print(f"\n  Examples:")
        print(f"    rand-check analyze 010100101001")
        print(f"    rand-check analyze BFFBFBFB")
        sys.exit(1)

    return [int(c) for c in s]


# -- Main entry point --------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rand-check",
        description="Detect patterns in your poker opponent's 3-bet decisions.",
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
        help="Watch the engine detect patterns in a simulated opponent",
        formatter_class=_PreservingHelpFormatter,
        epilog=textwrap.dedent(f"""\
        opponent types (--opponent):
        {model_list}

        examples:
          rand-check demo
          rand-check demo --opponent gamblers --hands 120
          rand-check demo --hu
        """),
    )
    demo_p.add_argument(
        "--hands", "-n", type=int, default=80,
        help="Number of hands to simulate (default: 80)",
    )
    demo_p.add_argument(
        "--baseline", "-p", type=float, default=None,
        help="GTO 3-bet probability -- how often a balanced player would 3-bet "
             "(default: 0.25 for 6-max, 0.23 for heads-up)",
    )
    demo_p.add_argument(
        "--opponent", "--model", default="alternator",
        metavar="TYPE",
        help=f"Type of simulated opponent (default: alternator). "
             f"Choices: {', '.join(FRIENDLY_NAMES)}",
    )
    demo_p.add_argument(
        "--hu", "--heads-up", action="store_true",
        help="Use heads-up mode instead of 6-max",
    )
    demo_p.add_argument(
        "--detailed", action="store_true",
        help="Show full technical detail (for advanced users)",
    )

    # -- analyze -------------------------------------------------------
    ana_p = sub.add_parser(
        "analyze",
        help="Analyze real 3-bet decisions from a session",
        formatter_class=_PreservingHelpFormatter,
        epilog=textwrap.dedent("""\
        input formats:
          Binary string:    rand-check analyze 010100101001
          With spaces:      rand-check analyze "0 1 0 1 0 0 1 0"
          B/F notation:     rand-check analyze BFFBFBFB
            (B = 3-bet, F = fold)

        examples:
          rand-check analyze 010100101001010010100101001010
          rand-check analyze BFFBFBFFBFBF --hu
          rand-check analyze "1,0,1,0,0,1" --baseline 0.20
        """),
    )
    ana_p.add_argument(
        "sequence", type=str,
        help="Sequence of decisions: 1 or B = 3-bet, 0 or F = fold/call",
    )
    ana_p.add_argument(
        "--baseline", "-p", type=float, default=None,
        help="GTO 3-bet probability baseline (default: 0.25 / 0.23 for HU)",
    )
    ana_p.add_argument(
        "--hu", "--heads-up", action="store_true",
        help="Use heads-up mode",
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
        opponent types:
        {model_list}

        examples:
          rand-check simulate alternator 100
          rand-check simulate gamblers 200 --seed 42
          rand-check simulate random 80
        """),
    )
    sim_p.add_argument(
        "opponent", metavar="TYPE",
        help=f"Type of opponent. Choices: {', '.join(FRIENDLY_NAMES)}",
    )
    sim_p.add_argument("hands", type=int, metavar="N", help="Number of hands to generate")
    sim_p.add_argument(
        "--baseline", "-p", type=float, default=None,
        help="GTO 3-bet probability (default: 0.25 / 0.23 for HU)",
    )
    sim_p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    sim_p.add_argument("--hu", "--heads-up", action="store_true", help="Use heads-up mode")

    # -- validate ------------------------------------------------------
    val_p = sub.add_parser(
        "validate",
        help="Test the detection engine's accuracy",
        formatter_class=_PreservingHelpFormatter,
        epilog=textwrap.dedent("""\
        Generates many fake opponents and checks how well the engine
        detects patterns. Reports accuracy and false-positive rates.

        examples:
          rand-check validate
          rand-check validate --sequences 200 --hands 150
        """),
    )
    val_p.add_argument(
        "--sequences", "--n-per-model", type=int, default=50,
        help="Number of test sequences per opponent type (default: 50)",
    )
    val_p.add_argument(
        "--hands", "--seq-length", type=int, default=80,
        help="Hands per sequence (default: 80)",
    )
    val_p.add_argument(
        "--baseline", "-p", type=float, default=None,
        help="GTO 3-bet probability (default: auto)",
    )
    val_p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    val_p.add_argument("--hu", "--heads-up", action="store_true", help="Use heads-up mode")

    args = parser.parse_args(argv)

    # Resolve GTO prob from --hu flag if not explicitly given
    hu = getattr(args, "hu", False)
    baseline = getattr(args, "baseline", None)
    if baseline is None:
        if hasattr(args, "baseline"):
            args.baseline = 0.23 if hu else 0.25
    # Backward compat: expose as .p for internal code
    if hasattr(args, "baseline"):
        args.p = args.baseline

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
        return "Looks random"
    elif pi < 0.6:
        return "Possibly patterned -- keep watching"
    elif pi < 0.85:
        return "Likely patterned"
    else:
        return "Strong pattern detected"


# -- Command implementations -------------------------------------------

def _run_demo(args: argparse.Namespace) -> None:
    import numpy as np
    from rand_check.prediction import PredictionEngine
    from rand_check.analyzer import PostSessionAnalyzer
    from rand_check import synthetic

    model_name = _resolve_model(args.opponent)
    friendly_name = _friendly_model_name(model_name)
    n = args.hands

    rng = np.random.default_rng(42)
    generators = {
        "iid": lambda: synthetic.generate_iid_bernoulli(n, args.p, rng),
        "markov_alternation": lambda: synthetic.generate_markov_alternation(n, args.p, 0.62, rng),
        "gamblers_fallacy": lambda: synthetic.generate_gamblers_fallacy(n, args.p, rng=rng),
        "counter": lambda: synthetic.generate_counter_model(n, args.p, rng=rng),
        "run_averse": lambda: synthetic.generate_run_averse(n, args.p, rng=rng),
        "mixture": lambda: synthetic.generate_mixture(n, args.p, rng=rng),
        "changepoint": lambda: synthetic.generate_changepoint(n, args.p, rng=rng),
    }

    gen_seq = generators[model_name]()
    seq = gen_seq.sequence

    fmt_label = "Heads-Up" if args.hu else "6-max"
    truth = "HAS PATTERNS" if gen_seq.is_human else "TRULY RANDOM"

    print()
    print("  +" + "-"*60 + "+")
    print(f"  |  DEMO: Watching a \"{friendly_name}\" opponent")
    print(f"  |  Format: {fmt_label}   Hands: {n}   Baseline: {_pct(args.p)}")
    print(f"  |  Ground truth: {truth}")
    print("  +" + "-"*60 + "+")
    print()
    print(f"  The engine observes each hand and tries to figure out")
    print(f"  if this opponent is truly randomizing or following a pattern.")
    print()

    if args.hu:
        engine = PredictionEngine.for_heads_up()
    else:
        engine = PredictionEngine(default_gto_prob=args.p)

    if args.detailed:
        # Full technical table for advanced users
        print(f"  {'Hand':>5} | {'Did':>5} | {'Detect':>8} | {'Next P':>8} | {'GTO':>6} | {'Edge':>7} | {'95% CI':>14} | Status")
        print("  " + "-" * 85)

        for i, action in enumerate(seq):
            pkg = engine.process_action(action)
            ci_str = f"[{pkg.confidence_interval[0]:.2f},{pkg.confidence_interval[1]:.2f}]"
            flag = " <-- SHIFT!" if pkg.changepoint_flag else ""
            act_label = "3-BET" if action else "fold"
            print(
                f"  {i+1:5d} | {act_label:>5} | "
                f"{pkg.detection_confidence:8.3f} | {pkg.predicted_3bet_prob:8.3f} | "
                f"{pkg.gto_3bet_prob:6.3f} | {pkg.edge:+7.3f} | {ci_str:>14} | "
                f"{engine._detector.interpret()}{flag}"
            )
    else:
        # Simplified, friendly output
        print(f"  {'Hand':>5} | {'Action':>7} | {'Pattern?':>24} | {'Next 3bet':>10} | Status")
        print("  " + "-" * 75)

        milestones = {20, 40, 60, n}
        for i, action in enumerate(seq):
            pkg = engine.process_action(action)
            act_label = " 3-BET" if action else "  fold"
            detect_bar = _bar(pkg.detection_confidence, width=15)
            detect_pct = _pct(pkg.detection_confidence)
            next_pct = _pct(pkg.predicted_3bet_prob)
            flag = " <-- Strategy shift!" if pkg.changepoint_flag else ""

            print(
                f"  {i+1:5d} | {act_label:>7} | {detect_bar} {detect_pct:>5} | {next_pct:>10} | "
                f"{_confidence_label(pkg.detection_confidence)}{flag}"
            )

            # Print milestone summaries
            hand_num = i + 1
            if hand_num in milestones and hand_num < n:
                pi = pkg.detection_confidence
                icon = _verdict_icon(pi)
                print()
                print(f"  {icon} After {hand_num} hands: {_confidence_label(pi)} (confidence: {_pct(pi)})")
                if pi > 0.6:
                    print(f"      Suggestion: {pkg.exploitation_suggestion}")
                print()

    # Final summary
    final_pkg = engine._build_package()

    pi = final_pkg.detection_confidence
    icon = _verdict_icon(pi)

    print()
    print("  +" + "-"*60 + "+")
    print(f"  |  {icon} RESULT AFTER {n} HANDS")
    print("  |")
    print(f"  |  Detection:    {_confidence_label(pi)}")
    print(f"  |  Confidence:   {_bar(pi)} {_pct(pi)}")
    print(f"  |  Next 3-bet:   {_pct(final_pkg.predicted_3bet_prob)}  (GTO baseline: {_pct(final_pkg.gto_3bet_prob)})")
    print(f"  |  Edge:         {final_pkg.edge:+.3f}")
    print("  +" + "-"*60 + "+")
    if final_pkg.exploitation_suggestion:
        print(f"\n  -> {final_pkg.exploitation_suggestion}")

    print(f"\n  {'='*60}")
    print("  POST-SESSION ANALYSIS")
    print(f"  {'='*60}\n")

    analyzer = PostSessionAnalyzer()
    report = analyzer.analyze(seq, gto_prob=args.p)
    print(report.summary(detailed=args.detailed))


def _run_validate(args: argparse.Namespace) -> None:
    from rand_check.validation import ValidationRunner

    n_per = args.sequences
    seq_len = args.hands
    print()
    print(f"  Testing detection engine accuracy...")
    print(f"  {n_per} sequences x 7 opponent types x {seq_len} hands each")
    print(f"  Baseline 3-bet rate: {_pct(args.p)}")
    print(f"  This may take a moment...\n")

    runner = ValidationRunner()
    t0 = time.time()
    metrics = runner.run(
        n_per_model=n_per,
        seq_length=seq_len,
        gto_prob=args.p,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print(metrics.summary())
    print(f"\n  Completed in {elapsed:.1f}s")


def _run_analyze(args: argparse.Namespace) -> None:
    from rand_check.analyzer import PostSessionAnalyzer

    seq = _parse_sequence(args.sequence)
    if len(seq) < 5:
        print("Error: need at least 5 decisions to analyze.")
        print("  Record more hands and try again.")
        sys.exit(1)

    n = len(seq)
    threebets = sum(seq)
    folds = n - threebets
    rate = threebets / n

    print()
    print("  +" + "-"*60 + "+")
    print(f"  |  ANALYZING {n} HANDS")
    print("  |")
    print(f"  |  3-bets: {threebets}   Folds: {folds}   3-bet rate: {_pct(rate)}")
    print(f"  |  Balanced (GTO) rate: {_pct(args.p)}")

    if rate > args.p + 0.05:
        print(f"  |  -> Opponent 3-bets MORE than balanced ({_pct(rate)} vs {_pct(args.p)})")
    elif rate < args.p - 0.05:
        print(f"  |  -> Opponent 3-bets LESS than balanced ({_pct(rate)} vs {_pct(args.p)})")
    else:
        print(f"  |  -> 3-bet rate is close to balanced")

    print("  +" + "-"*60 + "+")
    print()

    analyzer = PostSessionAnalyzer()
    report = analyzer.analyze(seq, gto_prob=args.p)
    print(report.summary(detailed=args.detailed))


def _run_simulate(args: argparse.Namespace) -> None:
    import numpy as np
    from rand_check import synthetic

    model_name = _resolve_model(args.opponent)
    n = args.hands
    rng = np.random.default_rng(args.seed)
    generators = {
        "iid": lambda: synthetic.generate_iid_bernoulli(n, args.p, rng),
        "markov_alternation": lambda: synthetic.generate_markov_alternation(n, args.p, 0.62, rng),
        "gamblers_fallacy": lambda: synthetic.generate_gamblers_fallacy(n, args.p, rng=rng),
        "counter": lambda: synthetic.generate_counter_model(n, args.p, rng=rng),
        "run_averse": lambda: synthetic.generate_run_averse(n, args.p, rng=rng),
        "mixture": lambda: synthetic.generate_mixture(n, args.p, rng=rng),
        "changepoint": lambda: synthetic.generate_changepoint(n, args.p, rng=rng),
    }

    gen_seq = generators[model_name]()
    seq_str = "".join(str(x) for x in gen_seq.sequence)
    print(seq_str)


if __name__ == "__main__":
    main()
