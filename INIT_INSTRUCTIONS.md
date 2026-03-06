Perfect. This is a very clean setup. Preflop 3bet decisions are:
- **High frequency** (every hand generates a data point)
- **Binary** (3bet vs. call/fold)
- **GTO-solved** (P is known exactly from solver for each position/stack/format)
- **Psychologically loaded** (players genuinely struggle to randomize bluff-catchers and thin 3bets)

Let me now do the full system design.

---

## Full System Design: 3bet Randomness Detector & Exploitative Predictor

---

### Framing the Problem Precisely

At each decision point, GTO prescribes the opponent should 3bet with probability **P** (known, position/stack-dependent). Over a session of N hands, we observe a binary string **x₁, x₂, ..., xₙ** where xᵢ = 1 (3bet) or 0 (call/fold).

**Part 1**: Is this string consistent with IID Bernoulli(P), or is it human-generated (structured/biased)?

**Part 2**: Given it's human-generated, what is P(xₙ₊₁ = 1 | x₁...xₙ), updated live after every hand?

The critical constraint: **we need useful signal from as few as 15-20 hands, and maximum signal by hand 100.** This rules out any approach that requires large n to converge.

---

### Part 1 — Detection: "Is This Player Actually Randomizing?"

#### The Core Difficulty at Small N

With n=30 hands and P=0.25 (typical BTN 3bet frequency), you expect only ~7-8 threes. Classical tests like NIST require n≥100 to have power. This forces a fundamentally **Bayesian** approach — you need priors, not p-values.

#### The Signal Structure

Under GTO (null hypothesis H₀): `x ~ IID Bernoulli(P)`

Under human generation (H₁): the player exhibits some combination of:
- **Alternation bias**: after a 3bet, less likely to 3bet again soon
- **Gambler's fallacy**: after N consecutive non-3bets, "due" pressure to 3bet
- **Tilt/momentum**: after losing a 3bet pot, avoidance; after winning, repetition
- **Run aversion**: never 3bets 3 hands in a row regardless of holdings
- **Count tracking**: unconsciously self-correcting toward their perceived target frequency

The single most powerful distinguishing feature at small n is **run-length distribution** — specifically, the probability of observing the longest run you've seen. Under Bernoulli(P), this has a known distribution. Humans dramatically under-represent long runs of either 0s or 1s.

#### The Bayesian Sequential Detection Framework

Rather than a single test at session end, we maintain a **running posterior** that updates after every hand:

```
π_n = P(H₁ | x₁...xₙ)
```

Using Bayes' rule:

```
π_n = π_{n-1} · L_n(H₁) / [π_{n-1}·L_n(H₁) + (1-π_{n-1})·L_n(H₀)]
```

Where `L_n(Hₖ)` is the likelihood of the new observation xₙ under each hypothesis.

**Under H₀**: `L_n(H₀) = P^xₙ · (1-P)^(1-xₙ)` — simple Bernoulli

**Under H₁**: `L_n(H₁)` comes from a **Markov(1) model** — the minimum viable cognitive model:

```
L_n(H₁) = P(xₙ | xₙ₋₁, θ_human)
```

where `θ_human = {p₀₀, p₀₁, p₁₀, p₁₁}` are the transition probabilities estimated from data so far, with **informative priors** encoding the known human biases:

```
Prior on p₀₁ (P(3bet | prev: no 3bet)):  Beta(α₀₁, β₀₁) 
  where α₀₁/( α₀₁+β₀₁) = P  (centered at GTO)
  but with extra mass above P (humans sometimes overcorrect after long folds)

Prior on p₁₀ (P(no 3bet | prev: 3bet)):  Beta(α₁₀, β₁₀)
  where α₁₀/(α₁₀+β₁₀) > (1-P)  (humans avoid repeating 3bets — alternation bias)
```

The prior strength (pseudo-count total α+β) encodes how many hands worth of prior knowledge you're injecting. Setting α+β ≈ 10 means the prior dominates for the first ~10 hands, then data takes over — exactly the right behavior.

#### What π_n Means at the Table

| π_n | Interpretation | Action |
|-----|---------------|--------|
| < 0.3 | Looks like genuine RNG | No exploitation yet; keep collecting |
| 0.3 – 0.6 | Mild evidence of patterns | Light adjustments, watch closely |
| 0.6 – 0.85 | Likely human-generated | Begin exploitative adjustments |
| > 0.85 | High confidence human pattern | Strong exploitation; trust the model |

This gives you a **live confidence meter** from hand 1, not a binary verdict at session end.

#### The Detection Feature Set (for Post-Session Analysis)

For the richer post-session analysis where you have the full string, compute the full feature vector and run a proper test:

**Feature 1 — Log Likelihood Ratio (LLR)**:
```
LLR = log P(x₁...xₙ | Markov(1) MLE) − log P(x₁...xₙ | IID Bernoulli(P))
```
Positive LLR → data fits Markov better than IID → human-generated signal.

**Feature 2 — Alternation Rate Deviation**:
```
A = #{i: xᵢ ≠ xᵢ₋₁} / (n-1)
Expected under H₀: A* = 2P(1-P)
Deviation: ΔA = A - A*
```
Humans produce ΔA > 0 (over-alternate).

**Feature 3 — Run Length Score**:
```
For each observed run of length k, compute 
  P(run ≥ k | IID Bernoulli(P))
Score = -Σ log P(observed run lengths)
```
High score → runs are shorter than expected → human avoiding repetition.

**Feature 4 — Normalized LZC**: Compressibility of the string. Humans produce LZC < 1 (more compressible than true random).

**Feature 5 — Frequency Drift**: Variance of local 3bet frequency in sliding windows of 20 hands. RNG is stationary; humans drift as their "mental account" resets.

**Feature 6 — Lag-1 Serial Correlation**:
```
ρ₁ = Corr(xᵢ, xᵢ₋₁) 
Expected under H₀: 0
Humans: ρ₁ < 0 (negative autocorrelation — alternation)
```

All six features feed into a **Beta-Binomial hypothesis test** for small n, and a **gradient boosted classifier** for n > 80 where ML becomes reliable.

---

### Part 2 — Prediction: P(next = 3bet | history)

This is the live exploitation engine. The output is a single number — the **adjusted probability** you should use in your own decision-making, replacing GTO's assumed P.

#### The Core Model: Bayesian Markov with Conjugate Updates

Maintain four counters, updated after every hand:

```
n₀₀ = # times (prev: fold, curr: fold)
n₀₁ = # times (prev: fold, curr: 3bet)  
n₁₀ = # times (prev: 3bet, curr: fold)
n₁₁ = # times (prev: 3bet, curr: 3bet)
```

With Beta priors `Beta(αᵢⱼ, βᵢⱼ)` encoding human biases, the posterior predictive after observing the last action s is:

```
P(next = 3bet | last = s) = (nₛ₁ + αₛ₁) / (nₛ₀ + nₛ₁ + αₛ₀ + αₛ₁)
```

This is a **closed-form, O(1) update** — exactly what you need live. After every hand, increment one counter, recompute. No retraining, no batch updates.

#### The Prior Design — Critical for Small N

The priors encode everything we know before hand 1:

```
α₀₁ = P · κ      β₀₁ = (1-P) · κ      [base rate prior]
α₁₀ = (1-P+δ)·κ  β₁₀ = (P-δ)·κ        [alternation prior: δ≈0.1]
α₁₁ = (P-δ)·κ    β₁₁ = (1-P+δ)·κ      [run-avoidance prior]
```

Where **κ = 15** (pseudo-count strength) means: "the prior is worth 15 hands of data." By hand 30, data is roughly equal weight with the prior. By hand 100, data dominates.

Choosing κ is the key calibration decision — it should be tuned on historical population data of human players at similar stakes.

#### The Context Tree Extension (When N > 60)

Once you have enough data, upgrade from Markov(1) to **Context Tree Weighting (CTW)**, which maintains Bayesian posteriors over all Markov orders simultaneously:

```
P_CTW(next | x₁...xₙ) = mixture over:
  - IID Bernoulli(P)           [order 0: no memory]
  - Markov(1) posterior        [order 1: last action matters]  
  - Markov(2) posterior        [order 2: last two actions]
  - Markov(3) posterior        [order 3: streaks of 3]
```

Weights are automatically Bayesian — higher-order models get more weight only when they've earned it through prediction accuracy. This prevents overfitting at small n while capturing richer patterns at large n.

#### Changepoint Detection — The Adaptation Problem

The most dangerous failure mode: you've correctly identified a pattern, but the opponent **adjusts** mid-session (notices they're being exploited, or just runs well and changes their approach). You must detect this.

Use **Bayesian Online Changepoint Detection (BOCPD)**:

```
At each step, maintain P(changepoint at step t)
If P(changepoint) > 0.7:
  → Reset transition counters (keep priors, discard data)
  → Reset detection confidence π_n
  → Flag "opponent adjusted" in HUD
```

The changepoint prior should be calibrated to roughly "a player adjusts at most once per 50 hands" — so the hazard rate is ~0.02 per hand.

#### The Output: Exploitation-Ready Probability

The final output at hand n+1 is not just a number but a **decision package**:

```
{
  detection_confidence: π_n,          // 0-1, how sure we are they're not RNG
  predicted_3bet_prob: q_n,           // adjusted probability to use
  gto_3bet_prob: P,                   // baseline
  edge: q_n - P,                      // how much to adjust
  confidence_interval: [q_lo, q_hi],  // posterior 95% interval
  changepoint_flag: bool,             // did they just adjust?
  hands_to_reliable: max(0, 40 - n)  // hands until high-confidence regime
}
```

**How you use `edge` at the table:**

- If `q_n > P + threshold` (opponent 3bets more than GTO): tighten calling range, fold more marginals, widen 4bet bluff range
- If `q_n < P - threshold` (opponent 3bets less than GTO): widen flatting range, defend lighter, reduce 4bet bluffing

The threshold should scale with confidence: `threshold = base_threshold / π_n` — require more edge when detection confidence is low.

---

### Part 3 — The Full Live System Architecture

```
Per-hand input:
  · action xₙ ∈ {0, 1}
  · position (→ determines P from solver lookup)
  · stack depth (→ modifies P)
          │
          ▼
┌─────────────────────────────────┐
│  SOLVER LOOKUP TABLE            │
│  (position, stack) → GTO P      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  BAYESIAN MARKOV UPDATER        │
│  · Increment n_{s,xₙ}           │
│  · Compute q_n (posterior pred) │
│  · Update π_n (detection score) │
│  O(1) per hand                  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  BOCPD LAYER                    │
│  · Compute P(changepoint)       │
│  · If triggered: soft reset     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  CTW LAYER (activates at n>60)  │
│  · Blend orders 0-3             │
│  · Reweight by predictive acc.  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  OUTPUT PACKAGE                 │
│  · HUD display (live)           │
│  · Exploitation recommendation  │
│  · Confidence interval          │
│  · Hands-to-reliable estimate   │
└─────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  POST-SESSION ANALYZER          │
│  · Full feature vector          │
│  · LLR test + 6-feature report  │
│  · Session timeline chart       │
│  · Pattern fingerprint (which   │
│    cognitive bias dominates?)   │
└─────────────────────────────────┘
```

---

### Part 4 — Calibration & Validation Plan

Since you can't collect infinite live data, calibration must be done on **synthetic data** generated from known cognitive models, then validated on whatever real hand histories you have.

**Synthetic data generation protocol:**
- Markov(1) chains with alternation rate 0.55–0.70 (covers known human range)
- Gambler's fallacy models: P(3bet | last k hands all non-3bet) increases with k
- "Counter" model: player mentally tracks their 3bet frequency and self-corrects
- Mixture models: 30% of the time player is genuinely randomizing (spot-check sessions)

**Validation metrics — specifically chosen for small n:**
- **Brier score** of next-bit prediction (proper scoring rule, meaningful at n=20)
- **Log-loss** vs. baseline Bernoulli(P)
- **Detection power at n=20, 40, 60, 100**: P(π_n > 0.75 | truly human-generated)
- **False positive rate**: P(π_n > 0.75 | truly IID Bernoulli(P)) — must be < 5%

---

### The Key Design Decisions Summarized

The entire system rests on four choices that make it work under poker constraints:

**1. Bayesian not frequentist** — priors allow meaningful inference at n=15 where p-values are useless.

**2. Conjugate Beta-Binomial updates** — O(1) per hand, no retraining, works live.

**3. Informative priors from cognitive science** — the known alternation bias δ≈0.1 is free information that massively improves small-n performance.

**4. Continuous confidence score not binary verdict** — you don't need certainty to exploit; you need calibrated probability, and you can start adjusting as soon as π_n > 0.4.

Ready to start building this? The natural first step is implementing the Bayesian Markov updater and synthetic data generator so we can validate the small-n performance before writing a single line of HUD code.
