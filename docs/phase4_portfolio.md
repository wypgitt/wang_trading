# Phase 4 — Portfolio Construction

Operator reference for [`src/portfolio/`](../src/portfolio) — HRP, factor
risk, risk parity, and the multi-strategy allocator that sits on top.

---

## Table of contents

1. [Portfolio construction hierarchy](#1-portfolio-construction-hierarchy)
2. [HRP vs. Markowitz vs. Risk Parity](#2-hrp-vs-markowitz-vs-risk-parity)
3. [Factor risk model — PCA interpretation](#3-factor-risk-model--pca-interpretation)
4. [Rebalancing cadence — the cost vs drift tradeoff](#4-rebalancing-cadence--the-cost-vs-drift-tradeoff)
5. [Regime-conditional tilting — evidence for and against](#5-regime-conditional-tilting--evidence-for-and-against)

---

## 1. Portfolio construction hierarchy

**Module**: [`src.portfolio.multi_strategy`](../src/portfolio/multi_strategy.py).

```
┌──────────────────────────────────────────────────┐
│  L1  strategy-level optimiser                    │
│       → weight across signal families            │
│       (default: HRP on per-family return series) │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│  L2  within-strategy instrument optimiser        │
│       → weight across instruments in each family │
│       (default: HRP on instrument returns)       │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│  L3  combine + sign + modulate                   │
│       final_w = strategy_w · within_w ·          │
│                 sign(signal) · |bet_size|        │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│  L4  risk-budget enforcement (design-doc §8.5)   │
│       • max_instrument_weight (10%)              │
│       • max_strategy_weight (30%)                │
│       • max_gross_exposure (150%)                │
│       • max_crypto_pct (30%)                     │
│       • optional regime tilt                     │
└──────────────────────────────────────────────────┘
```

### Why split L1 / L2

Treating "allocate capital across 7 families × 20 instruments each" as a
single 140-asset optimization has two problems:

1. **Fragile covariance**. Estimating a 140×140 covariance from even 2
   years of daily data (500 obs) is noisy; Markowitz would concentrate in
   whichever assets had the flukiest Sharpes. HRP mitigates this at any
   scale, but splitting reduces the per-solve dimension to ~20, which
   gives much cleaner covariances.
2. **Operator intent**. You want to say "carry is 20% of the book." A
   flat solver might decide carry is 5% because its recent Sharpe was
   mediocre, even if structurally you want the exposure. The L1/L2 split
   makes the strategy-level allocation an explicit, inspectable number.

### L4 ordering

Constraints are enforced in this order: per-instrument cap → crypto cap
(proportional scale) → gross exposure cap (proportional scale). Strategy
caps are applied at L1 *before* the instrument-level allocation so the
within-strategy solver sees a realistic budget. The cap is a hard
ceiling, **not** redistributed — if 3 strategies each want 40% with cap
30%, the result is 3 × 30% = 90%, with 10% in cash. Redistributing
would defeat the purpose of the cap.

---

## 2. HRP vs. Markowitz vs. Risk Parity

| optimizer | strength | weakness | when to use |
|---|---|---|---|
| **Markowitz** | theoretically optimal given correct inputs | requires expected returns (you don't have them reliably); inverts a near-singular covariance; concentrates aggressively on fluky Sharpes | rarely — only when you have a strong structural expected-return view you trust |
| **HRP** | no matrix inversion; handles correlated assets naturally via clustering; stable weights; allocates hierarchically so correlated blocks share risk budget | doesn't target equal risk contribution; "optimal" only in a heuristic sense | **default**. Robust, cheap, reproducible. |
| **Risk Parity** | explicit equal-risk guarantee; transparent | assumes all assets should contribute equally (no "bet more on better signals"); sensitive to covariance estimation; requires an iterative solver | when you genuinely want equal risk by mandate (e.g. multi-manager fund-of-funds) |

### Why HRP is the default (design-doc §8.4.1)

HRP (López de Prado 2016) replaces Markowitz's matrix inversion with:
1. Hierarchical clustering on the correlation matrix.
2. Quasi-diagonalization (reorder so correlated assets are adjacent).
3. Recursive bisection, splitting the sorted list in halves and
   allocating inverse to each half's cluster variance.

This keeps the math numerically stable even on 50+ correlated assets,
produces long-only weights by construction, and requires no expected-
return inputs — the single biggest source of forecast error in Markowitz.

### When Risk Parity wins

The one case where RP beats HRP materially: when the asset universe is
composed of *uncorrelated strategies* and you want each one to contribute
the same risk. HRP's clustering layer does nothing useful on near-
orthogonal inputs; RP gives you the clean equal-risk answer.

Our live system (design-doc §8.4.3) runs both and picks the one with
better trailing 6-month risk-adjusted performance; see
[`RiskParityOptimizer.compare_with_hrp`](../src/portfolio/risk_parity.py).

### HRP leaf-level ≠ risk parity

A common confusion: HRP's recursive bisection uses inverse-variance
weighting **within each leaf cluster**, which is the optimal (minimum-
variance) weighting for uncorrelated assets. Risk parity on the same
uncorrelated assets gives inverse-*volatility* (`w ∝ 1/σ`), **not**
inverse-variance. The two differ by a square root.

---

## 3. Factor risk model — PCA interpretation

**Module**: [`src.portfolio.factor_risk`](../src/portfolio/factor_risk.py).
Design-doc §8.4.2.

### What the PCA factors represent

In a PCA factor model with K factors, the covariance decomposes as
`Σ ≈ B Σ_f Bᵀ + D` where B is N×K loadings, Σ_f is the diagonal factor
covariance, and D is idiosyncratic variance. PCA picks the top K
eigenvectors of Σ as B; they have no pre-assigned economic name.

Typical patterns in an equity universe:

| factor | loading signature | economic name |
|---|---|---|
| Factor 1 | all-positive, similar magnitudes | the *market* factor (SPX-like) |
| Factor 2 | bipolar (growth vs value; small vs large) | *style* or *size* |
| Factor 3 | sector-specific (e.g. tech vs defensive) | *sector rotation* |
| Factors 4–5 | idiosyncratic clusters | harder to interpret — usually residual structure |
| Factors 6+ | small eigenvalues | noise; unreliable |

For crypto the first factor is almost always "BTC-correlated"; for
futures it splits market risk from roll/carry.

### How to neutralize unwanted exposures

`FactorRiskModel.neutralize_factors(weights, factors_to_neutralize=[0])`
projects the weights onto the null space of the chosen factor loadings.
Closed form:

```
w_new = w − Bₖᵀ (Bₖ Bₖᵀ)⁻¹ Bₖ w
```

where `Bₖ` stacks the rows of B for the selected factors. This gives the
closest-in-L₂ weight vector with zero exposure to those factors.

**Common uses**:

- Market-neutralise the book: `neutralize_factors(w, [0])`. Removes the
  pure market beta so the remaining P&L is whatever the strategies were
  actually supposed to capture.
- Strip out a suspected overcrowded factor: if factors 2 and 3 happen to
  look like size/momentum and that exposure was accidental, neutralise
  both.

`detect_unintended_tilts` flags any factor whose portfolio exposure
exceeds `threshold_std × std(loadings)` — the σ reference is the cross-
sectional std of loadings, so a concentrated portfolio (which has
exposure = its single asset's loadings) reliably trips the detector.

### Operator discipline

- Refit the factor model when the universe changes (new instruments, large
  delistings). Stale loadings are worse than no loadings.
- `n_factors = 5` is a good default for equity-crypto-futures mixed books.
  Check `explained_variance_ratio` — if the top 5 factors explain < 60%,
  you likely need more factors or a better-quality universe. If the top 1
  factor explains > 90%, your universe is dominated by a single driver and
  factor-level neutralization is load-bearing.

---

## 4. Rebalancing cadence — the cost vs drift tradeoff

| cadence | pros | cons |
|---|---|---|
| **daily** | tracks target weights tightly; good for high-turnover signals | turnover cost scales with NAV → cost drag can dominate returns; vulnerable to noise-driven rebalances |
| **weekly (5 bars)** | industry standard for equity long-short | misses fast regime flips in crypto |
| **monthly (21)** | cheap; most crypto arb strategies don't need higher | noticeable drift on trending days |
| **event-driven** | rebalance only when `Σ|weight_change| > threshold` | harder to reason about; requires guardrails to avoid never rebalancing |

Our default `rebalance_frequency=5` (weekly-ish on daily bars) sits at the
cost-drift sweet spot for most equity strategies. Crypto can usually
handle daily. Futures runs depend on roll schedules.

### The `min_trade_fraction` filter

`MultiStrategyAllocator.compute_rebalance_trades` applies a floor: any
position-weight delta below `min_trade_fraction` (default 1%) is
suppressed. This is the single most cost-effective knob — a 1% threshold
typically cuts turnover by 40–60% with negligible tracking error.

---

## 5. Regime-conditional tilting — evidence for and against

`MultiStrategyAllocator.compute_target_portfolio(..., regime="trending")`
boosts momentum/trend family weights by `regime_boost` (default 20%)
before renormalisation. Symmetric logic for `regime="mean_reverting"`.

### The case for

- Signal families are differentially exposed to macro regimes. Momentum
  works in trending markets and gets eaten alive in chop. Mean reversion
  is the opposite.
- Regime detectors (LSTM classifier in [`src.ml_layer.regime_detector`](../src/ml_layer/regime_detector.py))
  can achieve ~60% accuracy on a binary trending/reverting classification.
- If you're willing to accept 60% accuracy × 20% tilt, you're still
  ex-ante better off than the equal-weight baseline in expectation.

### The case against

- Regime misclassification is correlated with the worst drawdowns —
  *exactly* when you want to trust the detector is when it's most likely
  to be wrong, because the regime is changing.
- Tilts amplify the detector's bias: if your LSTM systematically says
  "trending" 60% of the time, you'll systematically overweight momentum
  even when you shouldn't.
- Adds a tunable knob (regime_boost) to the optimizer, which is a
  Phase 4 "most backtests are worthless" red flag — another degree of
  freedom that CPCV/DSR/PBO will have to cover.

### Our current stance

Tilting is available but **off by default** (`regime=None`). Turn it on
only after:

1. The regime detector has paper-tracked for ≥ 6 months with recorded
   predictions.
2. Walk-forward backtest with the tilt beats walk-forward without, after
   DSR with the extra `n_trials` spent on regime-boost tuning.
3. The incremental Sharpe is worth the additional complexity for
   operator interpretability.

Until all three conditions hold, treat `regime_boost` as a research
feature, not a production feature.
