"""Command-line interface for rand-check.

Usage:
    rand-check demo              Run a live demo with synthetic data
    rand-check validate          Run the full validation/calibration suite
    rand-check analyze SEQ       Analyze a binary string (e.g. "010110010")
    rand-check simulate MODEL N  Generate a synthetic sequence
"""

from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rand-check",
        description="3bet Randomness Detector & Exploitative Predictor",
    )
    sub = parser.add_subparsers(dest="command")

    # ── demo ─────────────────────────────────────────────────────────
    demo_p = sub.add_parser("demo", help="Run a live demo with synthetic data")
    demo_p.add_argument("-n", type=int, default=80, help="Sequence length")
    demo_p.add_argument("-p", type=float, default=0.25, help="GTO 3bet prob")
    demo_p.add_argument("--model", default="markov_alternation",
                        choices=["iid", "markov_alternation", "gamblers_fallacy",
                                 "counter", "run_averse", "mixture", "changepoint"],
                        help="Synthetic model to demo")

    # ── validate ─────────────────────────────────────────────────────
    val_p = sub.add_parser("validate", help="Run calibration/validation suite")
    val_p.add_argument("--n-per-model", type=int, default=50, help="Sequences per model")
    val_p.add_argument("--seq-length", type=int, default=80, help="Sequence length")
    val_p.add_argument("-p", type=float, default=0.25, help="GTO 3bet prob")
    val_p.add_argument("--seed", type=int, default=42, help="RNG seed")

    # ── analyze ──────────────────────────────────────────────────────
    ana_p = sub.add_parser("analyze", help="Analyze a binary string")
    ana_p.add_argument("sequence", type=str, help="Binary string, e.g. '010110010'")
    ana_p.add_argument("-p", type=float, default=0.25, help="GTO 3bet prob")

    # ── simulate ─────────────────────────────────────────────────────
    sim_p = sub.add_parser("simulate", help="Generate a synthetic sequence")
    sim_p.add_argument("model", choices=["iid", "markov_alternation", "gamblers_fallacy",
                                         "counter", "run_averse", "mixture", "changepoint"])
    sim_p.add_argument("n", type=int, help="Sequence length")
    sim_p.add_argument("-p", type=float, default=0.25, help="GTO 3bet prob")
    sim_p.add_argument("--seed", type=int, default=None, help="RNG seed")

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
        parser.print_help()


def _run_demo(args: argparse.Namespace) -> None:
    import numpy as np
    from rand_check.prediction import PredictionEngine
    from rand_check.analyzer import PostSessionAnalyzer
    from rand_check import synthetic

    rng = np.random.default_rng(42)
    generators = {
        "iid": lambda: synthetic.generate_iid_bernoulli(args.n, args.p, rng),
        "markov_alternation": lambda: synthetic.generate_markov_alternation(args.n, args.p, 0.62, rng),
        "gamblers_fallacy": lambda: synthetic.generate_gamblers_fallacy(args.n, args.p, rng=rng),
        "counter": lambda: synthetic.generate_counter_model(args.n, args.p, rng=rng),
        "run_averse": lambda: synthetic.generate_run_averse(args.n, args.p, rng=rng),
        "mixture": lambda: synthetic.generate_mixture(args.n, args.p, rng=rng),
        "changepoint": lambda: synthetic.generate_changepoint(args.n, args.p, rng=rng),
    }

    gen_seq = generators[args.model]()
    seq = gen_seq.sequence

    print(f"\n{'='*60}")
    print(f"  LIVE DEMO: {gen_seq.model_name} (n={args.n}, P={args.p})")
    print(f"  Ground truth: {'HUMAN' if gen_seq.is_human else 'IID'}")
    print(f"  Parameters: {gen_seq.parameters}")
    print(f"{'='*60}\n")

    engine = PredictionEngine(default_gto_prob=args.p)

    print(f"{'Hand':>5} │ {'Act':>3} │ {'π_n':>6} │ {'q_n':>6} │ {'P':>5} │ {'Edge':>6} │ {'CI':>13} │ Interpretation")
    print("─" * 90)

    for i, action in enumerate(seq):
        pkg = engine.process_action(action)
        ci_str = f"[{pkg.confidence_interval[0]:.2f},{pkg.confidence_interval[1]:.2f}]"
        flag = " ⚠CP" if pkg.changepoint_flag else ""
        print(
            f"{i+1:5d} │ {'3B' if action else '  ':>3} │ "
            f"{pkg.detection_confidence:6.3f} │ {pkg.predicted_3bet_prob:6.3f} │ "
            f"{pkg.gto_3bet_prob:5.3f} │ {pkg.edge:+6.3f} │ {ci_str:>13} │ "
            f"{engine._detector.interpret()}{flag}"
        )

    print(f"\n{'='*60}")
    print("  POST-SESSION ANALYSIS")
    print(f"{'='*60}\n")

    analyzer = PostSessionAnalyzer()
    report = analyzer.analyze(seq, gto_prob=args.p)
    print(report.summary())


def _run_validate(args: argparse.Namespace) -> None:
    from rand_check.validation import ValidationRunner

    print(f"\nRunning validation (n_per_model={args.n_per_model}, "
          f"seq_length={args.seq_length}, P={args.p}, seed={args.seed})...")
    print("This may take a moment...\n")

    runner = ValidationRunner()
    t0 = time.time()
    metrics = runner.run(
        n_per_model=args.n_per_model,
        seq_length=args.seq_length,
        gto_prob=args.p,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print(metrics.summary())
    print(f"\n  Completed in {elapsed:.1f}s")


def _run_analyze(args: argparse.Namespace) -> None:
    from rand_check.analyzer import PostSessionAnalyzer

    # Parse binary string
    seq_str = args.sequence.strip()
    if not all(c in "01" for c in seq_str):
        print("Error: sequence must contain only '0' and '1' characters.")
        sys.exit(1)

    seq = [int(c) for c in seq_str]
    if len(seq) < 5:
        print("Error: sequence must be at least 5 characters long.")
        sys.exit(1)

    print(f"\nAnalyzing sequence of length {len(seq)} (P={args.p})...")
    print(f"  Sequence: {seq_str[:60]}{'...' if len(seq_str) > 60 else ''}")
    print(f"  3bet rate: {sum(seq)/len(seq):.3f} (GTO: {args.p:.3f})\n")

    analyzer = PostSessionAnalyzer()
    report = analyzer.analyze(seq, gto_prob=args.p)
    print(report.summary())


def _run_simulate(args: argparse.Namespace) -> None:
    import numpy as np
    from rand_check import synthetic

    rng = np.random.default_rng(args.seed)
    generators = {
        "iid": lambda: synthetic.generate_iid_bernoulli(args.n, args.p, rng),
        "markov_alternation": lambda: synthetic.generate_markov_alternation(args.n, args.p, 0.62, rng),
        "gamblers_fallacy": lambda: synthetic.generate_gamblers_fallacy(args.n, args.p, rng=rng),
        "counter": lambda: synthetic.generate_counter_model(args.n, args.p, rng=rng),
        "run_averse": lambda: synthetic.generate_run_averse(args.n, args.p, rng=rng),
        "mixture": lambda: synthetic.generate_mixture(args.n, args.p, rng=rng),
        "changepoint": lambda: synthetic.generate_changepoint(args.n, args.p, rng=rng),
    }

    gen_seq = generators[args.model]()
    seq_str = "".join(str(x) for x in gen_seq.sequence)
    print(seq_str)


if __name__ == "__main__":
    main()
