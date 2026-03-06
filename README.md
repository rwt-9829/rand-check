# rand-check — 3bet Randomness Detector & Exploitative Predictor

A Bayesian detection system that determines whether an opponent's 3-bet decisions in poker are truly random (GTO-aligned) or exhibit human cognitive biases, then exploits the detected patterns in real time.

## Architecture

```
┌────────────────┐     ┌──────────────┐     ┌──────────────┐
│ Solver Lookup  │────>│ Bayesian     │────>│ Detection    │
│ (GTO P)        │     │ Markov(1)    │     │ Engine       │
└────────────────┘     │ Beta-Binom   │     │ H₀ vs H₁    │
                       └──────────────┘     └──────┬───────┘
                                                   │
┌────────────────┐     ┌──────────────┐            │
│ CTW Layer      │────>│ Prediction   │<───────────┘
│ (depth ≤ 3)    │     │ Engine       │
└────────────────┘     └──────┬───────┘
                              │
┌────────────────┐            │
│ BOCPD          │────────────┘
│ (changepoint)  │
└────────────────┘
```

### Components

| Module | Description |
|--------|-------------|
| `models.py` | Core data types: `Action`, `Position`, `HandResult`, `DecisionPackage` |
| `solver_lookup.py` | GTO 3-bet frequency lookup tables for 6-max positions |
| `bayesian_markov.py` | Markov(1) updater with conjugate Beta-Binomial priors encoding alternation bias (δ≈0.10) |
| `detection.py` | Sequential Bayesian H₀ (IID) vs H₁ (Markov human) detection via log-odds posterior |
| `ctw.py` | Context Tree Weighting (Willems et al. 1995) — Bayesian mixture over variable-order Markov models |
| `bocpd.py` | Bayesian Online Changepoint Detection (Adams & MacKay 2007) with Beta-Bernoulli UPM |
| `features.py` | 6 features: LLR, alternation deviation, run-length score, LZ complexity, frequency drift, serial correlation |
| `synthetic.py` | 7 generators: IID, Markov alternation, gambler's fallacy, counter, run-averse, mixture, changepoint |
| `prediction.py` | Live prediction engine combining all layers into `DecisionPackage` per hand |
| `analyzer.py` | Post-session analysis with cognitive-bias fingerprinting |
| `validation.py` | Calibration framework: Brier score, log-loss, detection power, FPR |
| `cli.py` | CLI with `demo`, `validate`, `analyze`, and `simulate` commands |

## Theoretical Foundation

- **Bayesian Markov(1)**: Conjugate Beta-Binomial model with informative priors capturing known human randomness biases (alternation bias ~60%, per Tversky & Kahneman 1974). O(1) updates.
- **Context Tree Weighting**: Optimal Bayesian mixture over all Markov models up to depth D (Willems, Shtarkov & Tjalkens 1995). Activates after 60 hands.
- **BOCPD**: Online detection of strategy shifts using the run-length posterior distribution (Adams & MacKay 2007). Triggers soft-resets when players change strategies.
- **Lempel-Ziv Complexity**: Measures sequence compressibility; human sequences are significantly more compressible than true RNG (Lempel & Ziv 1976).

## Installation

```bash
# Requires Python >= 3.9
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from rand_check.prediction import PredictionEngine

engine = PredictionEngine(gto_prob=0.25)

# Process live observations (1 = 3bet, 0 = fold/call)
for action in [0, 1, 0, 0, 1, 0, 1, 1, 0, 0]:
    result = engine.process_hand(action)
    print(f"P(3bet) = {result.predicted_3bet_prob:.3f}  "
          f"Detection = {result.detection_confidence:.3f}  "
          f"Edge = {result.edge:+.3f}")
```

### CLI

```bash
# Interactive demo with synthetic data
rand-check demo

# Full calibration suite
rand-check validate --models 50

# Analyze a binary sequence
rand-check analyze --sequence "010010110010"

# Generate synthetic sequences
rand-check simulate --model markov_alternation --length 200
```

### Post-session Analysis

```python
from rand_check.analyzer import PostSessionAnalyzer

analyzer = PostSessionAnalyzer()
report = analyzer.analyze(
    sequence=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
              0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    gto_prob=0.25,
)
print(report.summary())
```

## Testing

```bash
# Run all 79 tests
pytest

# With coverage
pytest --cov=rand_check --cov-report=term-missing
```

## Key Outputs — `DecisionPackage`

Each call to `engine.process_hand()` returns a `DecisionPackage` with:

| Field | Type | Description |
|-------|------|-------------|
| `predicted_3bet_prob` | float | Blended probability of next 3-bet |
| `detection_confidence` | float | π_n — posterior probability of human pattern (H₁) |
| `edge` | float | `gto_prob - predicted_3bet_prob` — positive = opponent under-3bets |
| `confidence_interval` | (float, float) | 95% credible interval for 3-bet probability |
| `changepoint_flag` | bool | True if BOCPD detected a strategy shift |
| `exploitation_suggestion` | str | Human-readable action recommendation |

## Design Decisions

- **O(1) per hand** for core Markov layer — suitable for real-time play
- **Informative priors** (κ=15, δ=0.10) encode human alternation bias — converges to accurate predictions in ~20 hands
- **CTW activation threshold** at 60 hands — avoids overfitting with limited data
- **Constant hazard rate** (H=0.02, ~1 changepoint per 50 hands) — calibrated to typical poker session dynamics
- **Log-odds space** for posterior updates — prevents numerical underflow with many observations

## References

1. Adams, R. P. & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection*. arXiv:0710.3742.
2. Willems, F. M. J., Shtarkov, Y. M. & Tjalkens, T. J. (1995). *The Context-Tree Weighting Method: Basic Properties*. IEEE Trans. Info. Theory.
3. Lempel, A. & Ziv, J. (1976). *On the Complexity of Finite Sequences*. IEEE Trans. Info. Theory.
4. Tversky, A. & Kahneman, D. (1974). *Judgment Under Uncertainty: Heuristics and Biases*. Science, 185(4157).

## License

MIT
