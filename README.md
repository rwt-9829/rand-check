# rand-check

**Detect patterns in your poker opponents' 3-bet decisions and exploit them in real time.**

Most players *think* they're randomizing their 3-bets, but humans are bad at being random. They alternate too much, avoid long streaks, or fall into the gambler's fallacy. **rand-check** catches these biases and tells you how to adjust.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [Python API](#python-api)
- [Understanding the Output](#understanding-the-output)
- [Testing](#testing)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

---

## What It Does

1. **Watches** your opponent's 3-bet/fold decisions hand by hand.
2. **Detects** whether those decisions are truly random or follow human patterns.
3. **Predicts** the probability of their next 3-bet.
4. **Recommends** how to exploit any detected bias (e.g. "widen your calling range").

Works for both **6-max** and **heads-up** formats.

---

## Installation

**Requirements:** Python 3.9 or newer.

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/rand-check.git
cd rand-check
```

### Step 2 — Create a virtual environment and install

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

> **Execution-policy error?** Run this once and try again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

</details>

<details>
<summary><strong>Windows (Command Prompt)</strong></summary>

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

</details>

<details>
<summary><strong>macOS / Linux</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

</details>

> **Important:** Activate the virtual environment every time you open a new terminal before running `rand-check`. Look for `(.venv)` at the start of your prompt.

---

## Quick Start

After installing, try the interactive demo:

```
rand-check demo
```

This generates a fake opponent with known biases and shows the detection engine working hand by hand. You'll see a progress bar for each hand showing how confident the engine is that the opponent is following a pattern.

To analyze a real session, record your opponent's decisions as 1s and 0s (1 = 3-bet, 0 = fold/call), then:

```
rand-check analyze 010100101001010010100101001010
```

Or use B/F notation if you find that easier:

```
rand-check analyze BFFBFBFFBFBFBFFBFBFFBFBFBFFBFB
```

---

## CLI Commands

rand-check has four commands: **demo**, **analyze**, **simulate**, and **validate**.

### `rand-check demo` — See a live example

Runs a walkthrough using a simulated opponent so you can see detection working hand by hand. Great for understanding what the tool does.

```
rand-check demo [OPTIONS]
```

| Option | Default | What it means |
|--------|---------|-------------|
| `--hands N` | `80` | How many hands to simulate |
| `--baseline P` | `0.25` (6-max) / `0.23` (HU) | How often a balanced player would 3-bet |
| `--opponent TYPE` | `alternator` | What kind of fake opponent to generate |
| `--hu` | off | Heads-up mode |
| `--detailed` | off | Show full technical output |

**Opponent types** (use with `--opponent`):

| Name | What it simulates |
|------|-------------------|
| `random` | Truly random (no exploitable pattern) |
| `alternator` | Over-alternates between 3-bet and fold (most common human bias) |
| `gamblers` | Avoids repeating recent outcomes ("I just 3-bet, so I shouldn't again") |
| `counter` | Mentally tracks frequency ("1 in 4 hands") |
| `streak-averse` | Never 3-bets several times in a row |
| `mixed` | Randomly switches between two strategies |
| `shifter` | Plays one way, then suddenly changes mid-session |

> **Note:** The old technical model names (`iid`, `markov_alternation`, `gamblers_fallacy`, `run_averse`, `mixture`, `changepoint`) still work too.

**Examples:**

```bash
# Default demo — alternation-biased opponent, 80 hands
rand-check demo

# Heads-up demo with a gambler's fallacy opponent over 120 hands
rand-check demo --hu --opponent gamblers --hands 120

# Full technical output for advanced users
rand-check demo --detailed
```

---

### `rand-check analyze` — Analyze a real session

Feed in your opponent's 3-bet decisions and get a full bias report.

```
rand-check analyze SEQUENCE [OPTIONS]
```

**Input formats** — use whichever is easiest:

| Format | Example |
|--------|---------|
| Binary string | `rand-check analyze 010010110010` |
| Space-separated | `rand-check analyze "0 1 0 0 1 0 1 1"` |
| Comma-separated | `rand-check analyze "0,1,0,0,1,0,1,1"` |
| B/F notation | `rand-check analyze BFFBFBFF` |

Where: **1** or **B** = they 3-bet, **0** or **F** = they folded or called.

| Option | Default | What it means |
|--------|---------|-------------|
| `--baseline P` | `0.25` / `0.23` (HU) | How often a balanced player would 3-bet |
| `--hu` | off | Heads-up mode |
| `--detailed` | off | Show full technical metrics |

**Examples:**

```bash
# Analyze a 30-hand 6-max session
rand-check analyze 010100101001010010100101001010

# Heads-up session with a specific baseline
rand-check analyze BFFBFBFFBF --hu --baseline 0.20

# Full technical breakdown
rand-check analyze 010100101001 --detailed
```

The report tells you:
- Whether the opponent appears to be **randomizing** or **following a pattern**
- What kind of bias was detected (alternation, gambler's fallacy, etc.)
- How their 3-bet rate compares to balanced play
- **What to do about it** — plain-English suggestions

---

### `rand-check simulate` — Generate fake sequences

Outputs a binary string from a simulated opponent. Useful for testing or piping into `analyze`.

```
rand-check simulate TYPE N [OPTIONS]
```

| Option | Default | What it means |
|--------|---------|-------------|
| `TYPE` | *(required)* | Opponent type (same names as `--opponent` above) |
| `N` | *(required)* | Number of hands |
| `--baseline P` | auto | 3-bet probability |
| `--seed N` | random | Seed for reproducibility |
| `--hu` | off | Heads-up mode |

**Examples:**

```bash
# Generate 200 hands of alternation-biased play
rand-check simulate alternator 200

# Reproducible gambler's fallacy sequence
rand-check simulate gamblers 100 --seed 123

# Pipe into analyze (PowerShell)
rand-check analyze $(rand-check simulate counter 150)
```

---

### `rand-check validate` — Test the engine's accuracy

Generates many sequences from every opponent type and measures detection performance.

```
rand-check validate [OPTIONS]
```

| Option | Default | What it means |
|--------|---------|-------------|
| `--sequences N` | `50` | Test sequences per opponent type |
| `--hands N` | `80` | Hands per sequence |
| `--baseline P` | auto | 3-bet probability |
| `--seed N` | `42` | Random seed |
| `--hu` | off | Heads-up mode |

**Examples:**

```bash
rand-check validate
rand-check validate --sequences 200 --hands 150
```

---

## Python API

You can also use rand-check as a library.

### Live prediction (hand by hand)

```python
from rand_check.prediction import PredictionEngine

engine = PredictionEngine(default_gto_prob=0.25)

# For heads-up:
# engine = PredictionEngine.for_heads_up()

actions = [0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
for action in actions:
    result = engine.process_action(action)

    print(f"P(next 3bet) = {result.predicted_3bet_prob:.3f}")
    print(f"Detection    = {result.detection_confidence:.3f}")
    print(f"Edge         = {result.edge:+.3f}")
    print(f"Suggestion   = {result.exploitation_suggestion}")
    print()
```

### Post-session analysis

```python
from rand_check.analyzer import PostSessionAnalyzer

analyzer = PostSessionAnalyzer()
report = analyzer.analyze(
    sequence=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
              0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
              0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    gto_prob=0.25,
)

# Friendly output (default)
print(report.summary())

# Technical output
print(report.summary(detailed=True))
```

---

## Understanding the Output

### The simple view (default)

After each hand, you see:

| Column | What it means |
|--------|---------------|
| **Hand** | Hand number in the session |
| **Action** | What the opponent did (3-BET or fold) |
| **Pattern?** | A progress bar showing how confident the engine is that the opponent is following a pattern. 0% = looks random. 100% = definitely patterned. |
| **Next 3bet** | The engine's prediction: how likely is the opponent to 3-bet on the next hand? |
| **Status** | Plain-English summary of what the engine thinks |

### The post-session report

The report at the end tells you:

- **Verdict** — Is this opponent randomizing or patterned?
- **Detected bias** — What kind of pattern (alternation, gambler's fallacy, etc.)
- **What to do** — Exploitation suggestion in plain English

### The detailed view (`--detailed`)

For advanced users, `--detailed` shows the raw detection metrics:

| Metric | What it means |
|--------|---------------|
| Detect | Detection confidence (0 = random, 1 = patterned) |
| Next P | Predicted 3-bet probability for next hand |
| GTO | Balanced 3-bet rate for this spot |
| Edge | Difference between predicted and balanced (positive = over-3betting) |
| 95% CI | Confidence interval around the prediction |

---

## Testing

```bash
pytest
pytest --cov=rand_check --cov-report=term-missing
```

---

## How It Works

rand-check combines three Bayesian layers that run simultaneously on every hand:

### Layer 1 — Pattern Detection (always active)

A statistical model that learns whether the opponent's next action depends on their previous action (e.g. "after a 3-bet, they tend to fold next"). Converges to accurate predictions in ~20 hands.

### Layer 2 — Complex Patterns (activates after 60 hands)

Catches more complex patterns like "they only 3-bet after two consecutive folds." Needs more data to be reliable, so it kicks in after 60 observations.

### Layer 3 — Strategy Shift Detection (always active)

Watches for sudden strategy changes — e.g. the opponent goes on tilt or starts playing differently. When a shift is detected, the engine resets its beliefs and adapts quickly.

### Technical details

- O(1) per hand for the core layer — fast enough for real-time play
- Informative priors converge in ~20 hands
- Built on: Bayesian Markov models, Context Tree Weighting (Willems et al. 1995), and Bayesian Online Changepoint Detection (Adams & MacKay 2007)

---

## Project Structure

| File | What it does |
|------|--------------|
| `cli.py` | Command-line interface |
| `models.py` | Core data types |
| `prediction.py` | Main prediction engine that combines all layers |
| `detection.py` | Human-vs-random pattern detector |
| `bayesian_markov.py` | Statistical model for sequential patterns |
| `ctw.py` | Complex multi-step pattern detection |
| `bocpd.py` | Strategy shift detection |
| `features.py` | Feature extraction (alternation, streaks, complexity) |
| `analyzer.py` | Post-session analysis with bias fingerprinting |
| `synthetic.py` | Fake opponent generators for testing and demos |
| `solver_lookup.py` | GTO 3-bet frequency tables |
| `validation.py` | Accuracy testing framework |

---

## References

1. Adams, R. P. & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection*. arXiv:0710.3742.
2. Willems, F. M. J., Shtarkov, Y. M. & Tjalkens, T. J. (1995). *The Context-Tree Weighting Method: Basic Properties*. IEEE Trans. Info. Theory.
3. Lempel, A. & Ziv, J. (1976). *On the Complexity of Finite Sequences*. IEEE Trans. Info. Theory.
4. Tversky, A. & Kahneman, D. (1974). *Judgment Under Uncertainty: Heuristics and Biases*. Science, 185(4157).

---

## License

MIT
