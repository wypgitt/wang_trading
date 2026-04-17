# Phase 4 — Backtesting Framework

Operator reference for everything under [`src/backtesting/`](../src/backtesting).
Each section answers three questions:

- **Why** the safeguard exists (the failure mode it prevents).
- **How** it works (math + concrete examples, not just prose).
- **What to do** with it (operator-level knobs, interpretation, failure modes).

---

## Table of contents

1. [Why most backtests are worthless](#1-why-most-backtests-are-worthless)
2. [Gate 1 — Combinatorial Purged CV (CPCV)](#2-gate-1--combinatorial-purged-cv-cpcv)
3. [Gate 2 — Deflated Sharpe Ratio (DSR)](#3-gate-2--deflated-sharpe-ratio-dsr)
4. [Gate 3 — Probability of Backtest Overfitting (PBO)](#4-gate-3--probability-of-backtest-overfitting-pbo)
5. [Transaction cost model](#5-transaction-cost-model)
6. [Walk-forward vs. standard backtest](#6-walk-forward-vs-standard-backtest)
7. [Strategy gate — what to do when a gate fails](#7-strategy-gate--what-to-do-when-a-gate-fails)

---

## 1. Why most backtests are worthless

AFML devotes four chapters to this. The short version:

| failure mode | what it is | how it sneaks in |
|---|---|---|
| **Multiple testing** | Running N strategies and picking the best inflates the observed Sharpe by pure luck. With σ_SR ≈ 1, the expected best-of-1000 Sharpe exceeds 3 under H₀. | Grid search, hyperparameter tuning, reading a blog post and running the first idea that sounds promising, publication bias in papers. |
| **Look-ahead bias** | Training data leaks information from the test period. Triple-barrier labels overlap in time; standard k-fold shuffles through them. Feature autocorrelation leaks even after purging. | Using label-period features, normalising features with statistics computed over the full sample, cross-validation without purging + embargo. |
| **Survivorship bias** | Universe built from symbols that exist today excludes the delisted, the acquired, the bankrupt. The surviving names beat a randomly-drawn sample. | Using current S&P 500 members as the historical universe, ignoring crypto tokens that de-pegged / exchanges that went insolvent. |
| **Overfitting to in-sample** | A strategy with many tunable knobs will land on a knob combination that matches the noise in the IS sample. The noise doesn't repeat OOS. | Training on 100% of history then backtesting on the same 100%; picking hyperparameters on a validation set that the model has seen 100× via early stopping. |
| **Transaction cost optimism** | Assuming you get filled at the close price with zero commission/slippage. For a high-turnover strategy a 2 bps miss per trade can turn a Sharpe-1 strategy into a Sharpe-0 strategy. | Backtesting with free fills; ignoring market impact; using a constant spread when the live strategy actually moves markets. |

Our three gates address the first four directly; realistic cost modelling ([§5](#5-transaction-cost-model)) addresses the fifth.

---

## 2. Gate 1 — Combinatorial Purged CV (CPCV)

**Module**: [`src.backtesting.cpcv`](../src/backtesting/cpcv.py). Design-doc §9.1.

### Why

A single train/test split evaluates the strategy on one time slice. If that slice happens to be a bull market for momentum signals, your momentum model looks great. If it happens to be a chop, your model looks broken. A single number has huge variance.

**CPCV** (AFML Ch. 12) extends purged k-fold by enumerating every combination of test folds. With N=10 groups and k=2 test groups per combination, there are `C(10, 2) = 45` train/test splits — 45 different "if you had deployed this model, here's how it would have performed" paths.

### How

1. Split the time axis into N contiguous groups.
2. Enumerate every way of picking k groups as test and the rest as train.
3. For each combination:
   - Purge: drop training samples whose label period overlaps the test span.
   - Embargo: drop an additional forward window to handle autocorrelated features.
   - Fit the model on the purged train set.
   - Predict on the test set; compute per-path metrics (Sharpe, return, DD).

**Default config**: `n_groups=10, n_test_groups=2, embargo_pct=0.01`. This yields 45 paths, each covering ~20% of the data.

### Interpreting the 45 paths

| paths positive | verdict |
|---|---|
| ≥ 27 / 45 (60%) | **pass gate 1** — the strategy makes money in most regimes |
| 20–26 / 45 | borderline — may be regime-dependent; inspect per-path stats |
| < 20 / 45 | fail — the strategy only works in a minority of periods, usually overfitting |

The point is to see the **distribution** of outcomes, not just the central tendency. A strategy with a mean Sharpe of 1.5 but a 10th-percentile Sharpe of −2.0 is not a Sharpe-1.5 strategy — it's a "usually wins, occasionally craters" strategy, which is very different to drawdown-sensitive capital.

### What to do

- Use `CPCVEngine.get_path_statistics()` to pull per-path Sharpe, return, DD, win rate. Look at the full distribution — `path_stats.describe()` is your friend.
- `CPCVEngine.assemble_equity_curves()` returns a (time × 45) DataFrame. Plot the confidence bands: the 5th/50th/95th percentile of NAV across paths tells you what a "lucky" vs "unlucky" deployment would look like.
- If < 60% of paths are positive, **do not tune the gate threshold**. Go back to the signal layer and the feature layer (§7).

---

## 3. Gate 2 — Deflated Sharpe Ratio (DSR)

**Module**: [`src.backtesting.deflated_sharpe`](../src/backtesting/deflated_sharpe.py). Design-doc §9.2.

### Why

The Sharpe ratio of the **best** strategy out of N trials is biased upward by selection. Even if all N strategies have true Sharpe of zero, the best one will, by luck, look impressive. DSR (Bailey & López de Prado 2014) quantifies how much of your observed Sharpe is "real" versus "selection-bias residue."

### How (the math)

Two pieces:

**1. Expected max Sharpe under H₀** (eq. 13 of the paper):

```
E[max(SR)] ≈ σ_SR · [(1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e))]
```

where:
- `σ_SR` is the cross-sectional std of Sharpes across the trial pool (default 1.0)
- `γ ≈ 0.5772` is the Euler-Mascheroni constant
- `Φ⁻¹` is the inverse normal CDF

For `N=1`, `E[max] = 0` (no selection bias). For `N=1000` with `σ_SR=1`, `E[max] ≈ 3.26` — you'd expect the best of 1000 zero-Sharpe strategies to look like a Sharpe-3.26 strategy.

**2. Deflated Sharpe statistic** (Mertens variance correction):

```
σ̂_SR = sqrt((1 − γ₃·SR + (γ₄ − 1)/4 · SR²) / T)
DSR   = (SR_obs − E[max(SR)]) / σ̂_SR
p     = 1 − Φ(DSR)
```

Where `γ₃` is return skewness, `γ₄` is non-excess kurtosis (3.0 for Gaussian), and `T` is the observation count. Fat tails (`γ₄ > 3`) widen σ̂_SR, making it harder to clear the gate — exactly the right behaviour, since fat-tailed returns produce more extreme Sharpes by chance.

### Worked example

Your meta-labeler has observed Sharpe 2.0 from a 1000-trial hyperparameter grid, 252 daily returns, skewness −0.1, kurtosis 5.0.

- `E[max(SR)] ≈ 3.26` (from the 1000-trial formula above)
- `σ̂_SR = sqrt((1 − (−0.1)·2.0 + (5 − 1)/4 · 4) / 252) = sqrt(5.2 / 252) ≈ 0.144`
- `DSR = (2.0 − 3.26) / 0.144 ≈ −8.75` → p ≈ 1.0 → **fails**

Same strategy, but only 10 trials:
- `E[max] ≈ 1.58`
- `DSR = (2.0 − 1.58) / 0.144 ≈ 2.9` → p ≈ 0.002 → **passes**

The lesson: *a Sharpe of 2.0 is either brilliant or cheap, depending on how many you picked from*. Write down `n_trials` honestly — every hyperparameter you tuned, every feature you swapped in and out, every threshold you nudged — or the DSR is a lie.

### What to do

- Log every hyperparameter trial. `scripts/retrain_model.py --tune` already does this via MLflow. The trial count that goes into DSR should include tuning trials plus any manual experiments.
- When DSR fails with p between 0.05 and 0.30, consider: longer history (raises T, tightens σ̂_SR), simpler model (fewer hyperparameters), or honestly admitting the strategy may not be real.
- When DSR fails with p > 0.50, the strategy is almost certainly noise. Do not proceed.

---

## 4. Gate 3 — Probability of Backtest Overfitting (PBO)

**Module**: [`src.backtesting.pbo`](../src/backtesting/pbo.py). Design-doc §9.3.

### Why

CPCV and DSR judge a **single** strategy. PBO (Bailey et al. 2017) judges the **selection process** itself: when you pick the IS best among N variants, does that choice generalise OOS?

### How (CSCV algorithm)

1. Take a `(T × N)` matrix of returns: one column per strategy variant (e.g. each row of a hyperparameter grid).
2. Split the time axis into S equal partitions (S even — default 10).
3. For each way of labelling S/2 partitions as IS and the rest as OOS:
   a. Compute each variant's Sharpe on the IS concatenation.
   b. Pick the IS champion.
   c. Rank the champion's OOS Sharpe among all variants.
   d. `logit = log(rank / (N − rank))` — negative ⇒ champion finished below OOS median.
4. `PBO = fraction of splits with logit < 0`.

### Interpreting the logit distribution

The full logit distribution across splits is the real output. Its shape says:

- **Narrow, far-right of 0** (PBO small): IS ranking reliably identifies OOS winners. Your selection process works.
- **Centred on 0** (PBO ≈ 0.5): IS ranking is coin-flip — no information. You're picking lucky noise.
- **Narrow, far-left of 0** (PBO > 0.5): IS ranking is *anti*-informative — the IS champion systematically underperforms OOS. Usually means your IS/OOS boundary has structural differences the model is overfitting across (regime change, instrument change, liquidity shift).

### Thresholds

| PBO | verdict |
|---|---|
| < 0.40 | **pass gate 3** — selection is informative |
| 0.40–0.50 | fail — meaningful overfitting risk |
| > 0.50 | fail — selection is *worse than random*; the IS champion loses OOS systematically |

### What to do

- When PBO fails, the fix is almost never "add more variants" — more variants usually makes it worse. The fix is to **reduce the variant space**: fewer hyperparameters, coarser grids, simpler model.
- Inspect `details[details.logit < 0].is_best_strategy` — if the IS champion is disproportionately the most complex variant, that's the diagnosis: the model has too many degrees of freedom.

---

## 5. Transaction cost model

**Module**: [`src.backtesting.transaction_costs`](../src/backtesting/transaction_costs.py). Design-doc §9.4, §10.1.

### Four components

```
total = commission + spread_cost + slippage + market_impact
```

| component | formula | what it models |
|---|---|---|
| commission | `max(commission_per_share · qty, min_commission)` | broker fee; floored for small orders |
| spread_cost | `spread_bps/10⁴ · notional` | crossing the bid-ask spread (half-spread per leg) |
| slippage | `slippage_bps/10⁴ · notional` | execution price drift between decision and fill |
| market_impact | `coefficient · σ · sqrt(qty / ADV) · notional` | Johnson's square-root impact: impact grows with √(participation rate) |

The square-root law (Almgren et al. 2005; empirically robust) is why a 10% ADV order isn't 100× as expensive as a 0.1% order — it's ~10× per-share, 1000× total dollars. Impact dominates for large orders; spread + commission dominate for small retail orders.

### Typical ranges by asset class

| asset class | commission | spread (bps) | slippage (bps) | impact coef |
|---|---|---|---|---|
| equities (IBKR tiered) | $0.005/share, $1 min | 1–3 | 0.5–2 | 0.05–0.15 |
| crypto (CCXT) | 0–10 bps taker | 2–5 | 1–3 | 0.10–0.20 |
| futures | $1.25/contract | 0.5–1 tick | 0.5 tick | 0.05–0.10 |

These are the presets in `EQUITIES_COSTS`, `CRYPTO_COSTS`, `FUTURES_COSTS`. Override per-symbol when live TCA data becomes available.

### Calibrating from live TCA (Phase 5)

Once paper trading runs, the execution engine logs `(decision_price, fill_price, qty, ADV, vol)` per fill. The plan:

1. Regress `(fill − decision) / decision · 10⁴` (bps slippage) on `sqrt(qty/ADV) · σ` — the slope estimates the market-impact coefficient per asset class.
2. Residual spread = median bps miss when participation is below the noise floor.
3. Refit monthly; update `EQUITIES_COSTS` / `CRYPTO_COSTS` / `FUTURES_COSTS` and rerun the backtest gate.

---

## 6. Walk-forward vs. standard backtest

**Module**: [`src.backtesting.walk_forward`](../src/backtesting/walk_forward.py). Design-doc §9.4.

### The difference

- **Standard backtest**: train on 2020–2022, test on 2023–2024. One shot. Gives you one number.
- **Walk-forward (expanding window)**: train on 2020–2021, predict Q1 2022, train on 2020–Q1 2022, predict Q2 2022, … Repeat. Produces a stitched-together out-of-sample trajectory *and* forces the model to retrain the same way it would in production.

Walk-forward matters because the model in production is not the model you trained once on history — it's a sequence of models, each fit on the data available at that point in time. Walk-forward measures **that** deployment experience, including the lag between market changes and model adaptation.

### Retrain frequency

The backtester's `run_expanding_window` has `retrain_interval` (bars between retrains) and `initial_train_size` (warm-up).

| frequency | pros | cons |
|---|---|---|
| daily (retrain_interval=1) | tracks drift instantly | expensive; high model-update churn; Sharpe looks too good because every day's model "knew" about yesterday |
| weekly (5) | typical for slow-moving equity strategies | may miss crypto regime flips |
| monthly (21) | cheap; stable | clearly lags in fast-moving regimes |
| quarterly (63) | rare; reserved for LSTM/autoencoder | will be noticeably stale |

Default is `retrain_interval=252` for annual refits in backtest + daily refits in production (see design-doc §7.1.2). The **research** retrain is daily to show best-case performance; the **production** cadence is defined in [scripts/retrain_model.py](../scripts/retrain_model.py) (daily refit with saved params, weekly full retune).

---

## 7. Strategy gate — what to do when a gate fails

**Module**: [`src.backtesting.gate_orchestrator`](../src/backtesting/gate_orchestrator.py).

### Golden rule

**Do not tune the gate thresholds.** They are external constraints, not knobs. Tuning the threshold to pass is the purest form of backtest overfitting — you are literally optimising against the overfitting detector.

### Failure playbook

| gate | typical root causes | what to do |
|---|---|---|
| **CPCV < 60% positive** | regime-dependent signal; overfit to IS period | Add features that work across regimes (e.g. GARCH vol scaling, regime detector output). Simplify the signal. Increase training history. |
| **DSR p ≥ 0.05** | too many hyperparameter trials; observed Sharpe not impressive enough to beat selection haircut | Reduce trial count (coarser grid). Collect more observations (longer history, higher-frequency bars). Use a simpler model with fewer hyperparameters. |
| **PBO ≥ 40%** | variant space too large; IS/OOS regime mismatch | Shrink the hyperparameter grid. Check whether IS and OOS cover different market regimes — if so, the model isn't learning a stable pattern. |
| **all three fail** | the strategy doesn't work | Drop it. Move on. A failed strategy that you patch into passing is a liability, not an asset. |

### Quick iteration mode

During development, `StrategyGate.quick_validate` runs only DSR (no CPCV, no PBO). Use it to filter out obvious losers before spending the minutes of a full validation. When `quick_validate` passes, run the full `validate` before promoting.

### Audit trail

Every `StrategyGate.validate` call can emit a full `BacktestReport` (see [§9.5 report.py](../src/backtesting/report.py)). Save it alongside the MLflow run:

```python
result = gate.validate(...)
if not result["passed"]:
    result["report"].save_report(f"data/reports/rejected/{symbol}_{run_id}")
```

Rejected reports are as important as accepted ones — they're the record of what *didn't* work, which is the only thing that keeps you honest about what does.
