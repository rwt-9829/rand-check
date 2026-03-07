# rand-check

**Detect patterns in binary (0/1) sequences and predict the next outcome.**

Humans are bad at producing random sequences. They over-alternate, avoid streaks, and fall into predictable patterns. **rand-check** catches these biases using Bayesian inference and tells you what to expect next.

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

1. **Observes** a sequence of binary outcomes (0s and 1s) one at a time.
2. **Detects** whether the sequence is truly random or follows human-generated patterns.
3. **Predicts** the probability that the next value will be 1.
4. **Reports** any detected bias with plain-English explanations.

---

## Installation

**Requirements:** Python 3.9 or newer.

### Step 1 --- Clone the repository

```bash
git clone https://github.com/your-username/rand-check.git
cd rand-check
```

### Step 2 --- Create a virtual environment and install

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

This generates a synthetic sequence with known biases and shows the detection engine working observation by observation. You will see a progress bar showing how confident the engine is that a pattern is present.

To analyze a sequence you already have, enter it as 0s and 1s:

```
rand-check analyze 010100101001010010100101001010
```

---

## CLI Commands

rand-check has four commands: **demo**, **analyze**, **simulate**, and **validate**.

### `rand-check demo` --- See a live example

Runs a walkthrough using a simulated sequence so you can see detection in action.

```
rand-check demo [OPTIONS]
```

| Option | Default | What it means |
|--------|---------|-------------|
| `--length N` | `80` | How many observations to simulate |
| `--baseline P` | `0.50` | Expected probability of a 1 under the null hypothesis |
| `--model TYPE` | `alternator` | What kind of synthetic sequence to generate |
| `--detailed` | off | Show full technical output |

**Model types** (use with `--model`):

| Name | What it simulates |
|------|-------------------|
| `random` | Truly random (no exploitable pattern) |
| `alternator` | Over-alternates between 0 and 1 (most common human bias) |
| `gamblers` | Avoids repeating recent outcomes |
| `counter` | Mentally tracks frequency to hit a target rate |
| `streak-averse` | Never produces long runs of the same value |
| `mixed` | Randomly switches between two strategies |
| `shifter` | Uses one strategy, then suddenly changes mid-sequence |

**Examples:**

```bash
# Default demo --- alternation-biased, 80 observations
rand-check demo

# Gambler's fallacy model over 120 observations, baseline 0.3
rand-check demo --model gamblers --length 120 --baseline 0.3

# Full technical output
rand-check demo --detailed
```

---

### `rand-check analyze` --- Analyze a sequence

Feed in a binary sequence and get a full bias report.

```
rand-check analyze SEQUENCE [OPTIONS]
```

**Input formats** --- use whichever is easiest:

| Format | Example |
|--------|---------|
| Binary string | `rand-check analyze 010010110010` |
| Space-separated | `rand-check analyze "0 1 0 0 1 0 1 1"` |
| Comma-separated | `rand-check analyze "0,1,0,0,1,0,1,1"` |

| Option | Default | What it means |
|--------|---------|-------------|
| `--baseline P` | `0.50` | Expected probability of a 1 |
| `--detailed` | off | Show full technical metrics |

**Examples:**

```bash
# Analyze a 30-observation sequence
rand-check analyze 010100101001010010100101001010

# Custom baseline probability
rand-check analyze 010100101001 --baseline 0.3

# Full technical breakdown
rand-check analyze 010100101001 --detailed
```

The report tells you:
- Whether the sequence appears to be **random** or **patterned**
- What kind of bias was detected (alternation, gambler's fallacy, etc.)
- How the observed frequency compares to the baseline
- A plain-English summary

---

### `rand-check simulate` --- Generate synthetic sequences

Outputs a binary string from a synthetic model. Useful for testing or piping into `analyze`.

```
rand-check simulate TYPE N [OPTIONS]
```

| Option | Default | What it means |
|--------|---------|-------------|
| `TYPE` | *(required)* | Model type (same names as `--model` above) |
| `N` | *(required)* | Number of observations |
| `--baseline P` | `0.50` | Probability of a 1 |
| `--seed N` | random | Seed for reproducibility |

**Examples:**

```bash
# Generate 200 alternation-biased observations
rand-check simulate alternator 200

# Reproducible gambler's fallacy sequence
rand-check simulate gamblers 100 --seed 123

# Pipe into analyze (PowerShell)
rand-check analyze $(rand-check simulate counter 150)
```

---

### `rand-check validate` --- Test the engine's accuracy

Generates many sequences from every model type and measures detection performance.

```
rand-check validate [OPTIONS]
```

| Option | Default | What it means |
|--------|---------|-------------|
| `--sequences N` | `50` | Test sequences per model type |
| `--length N` | `80` | Observations per sequence |
| `--baseline P` | `0.50` | Probability of a 1 |
| `--seed N` | `42` | Random seed |

**Examples:**

```bash
rand-check validate
rand-check validate --sequences 200 --length 150
```

---

## Python API

You can also use rand-check as a library.

### Live prediction (observation by observation)

```python
from rand_check.prediction import PredictionEngine

engine = PredictionEngine(default_baseline_prob=0.5)

actions = [0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
for action in actions:
    result = engine.process_action(action)

    print(f"P(next=1)    = {result.predicted_prob:.3f}")
    print(f"Detection    = {result.detection_confidence:.3f}")
    print(f"Edge         = {result.edge:+.3f}")
    print(f"Suggestion   = {result.suggestion}")
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
    baseline_prob=0.5,
)

# Friendly output (default)
print(report.summary())

# Technical output
print(report.summary(detailed=True))
```

---

## Understanding the Output

### The simple view (default)

After each observation, you see:

| Column | What it means |
|--------|---------------|
| **#** | Observation number |
| **Value** | What was observed (0 or 1) |
| **Pattern?** | Progress bar showing detection confidence. 0% = looks random. 100% = definitely patterned. |
| **Next P(1)** | Predicted probability the next value will be 1 |
| **Status** | Plain-English summary |

### The post-session report

The report at the end tells you:

- **Verdict** --- Is this sequence random or patterned?
- **Detected bias** --- What kind of pattern (alternation, gambler's fallacy, etc.)
- **Summary** --- Plain-English explanation of what was found

### The detailed view (`--detailed`)

For advanced users, `--detailed` shows the raw detection metrics:

| Metric | What it means |
|--------|---------------|
| Detect | Detection confidence (0 = random, 1 = patterned) |
| Next P | Predicted probability for next observation |
| Base | Baseline probability (null hypothesis) |
| Edge | Difference between predicted and baseline |
| 95% CI | Confidence interval around the prediction |

---

## Testing

```bash
pytest
pytest --cov=rand_check --cov-report=term-missing
```

---

## How It Works

rand-check combines three Bayesian layers that run simultaneously on every observation:

### Layer 1 --- Pattern Detection (always active)

A Bayesian Markov model that learns whether the next value depends on the previous value (e.g. "after a 1, the next value tends to be 0"). Converges in ~20 observations.

### Layer 2 --- Complex Patterns (activates after 60 observations)

Catches higher-order patterns like "never produces three 1s in a row." Uses Context Tree Weighting to model patterns up to depth 3. Needs more data to be reliable, so it activates after 60 observations.

### Layer 3 --- Changepoint Detection (always active)

Watches for sudden strategy changes --- e.g. the sequence generator switches behavior mid-stream. When a shift is detected, the engine resets its beliefs and adapts quickly.

### Technical details

- O(1) per observation for the core layer
- Informative priors converge in ~20 observations
- Built on: Bayesian Markov models, Context Tree Weighting (Willems et al. 1995), and Bayesian Online Changepoint Detection (Adams & MacKay 2007)

---

## Project Structure

| File | What it does |
|------|--------------|
| `cli.py` | Command-line interface |
| `models.py` | Core data types |
| `prediction.py` | Main prediction engine combining all layers |
| `detection.py` | Pattern vs. random detector |
| `bayesian_markov.py` | Markov(1) model with Beta-Binomial priors |
| `ctw.py` | Context Tree Weighting for complex patterns |
| `bocpd.py` | Bayesian Online Changepoint Detection |
| `features.py` | Feature extraction (alternation rate, streaks, complexity) |
| `analyzer.py` | Post-session analysis with bias fingerprinting |
| `synthetic.py` | Synthetic sequence generators for testing and demos |
| `solver_lookup.py` | Baseline probability constant |
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
