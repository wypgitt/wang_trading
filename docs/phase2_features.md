# Phase 2 — Feature Factory

Reference for every feature family produced by
[`src/feature_factory/`](../src/feature_factory). Each section lists:

- **What it measures** — the economic / statistical object the feature tracks.
- **Formula** — the exact definition as implemented.
- **Source** — the book chapter (AFML = López de Prado, *Advances in
  Financial Machine Learning*) or paper the construction comes from.
- **Range** — the numerical interval outputs live in, after warmup.
- **Interpretation** — what high / low values mean for trading.

---

## Table of contents

1. [Fractional differentiation (FFD)](#1-fractional-differentiation-ffd)
2. [Structural-break features](#2-structural-break-features)
3. [Entropy features](#3-entropy-features)
4. [Microstructural features](#4-microstructural-features)
5. [GARCH / realized-vol features](#5-garch--realized-vol-features)
6. [NLP sentiment features](#6-nlp-sentiment-features)
7. [On-chain (crypto) features](#7-on-chain-crypto-features)
8. [Autoencoder latent features](#8-autoencoder-latent-features)
9. [Classical indicators](#9-classical-indicators)
10. [Feature correlation — what is OK, what is not](#10-feature-correlation)
11. [Feature stationarity & FFD d-selection](#11-feature-stationarity--ffd-d-selection)

---

## 1. Fractional differentiation (FFD)

**Module**: [`fractional_diff.py`](../src/feature_factory/fractional_diff.py)

### What it measures
The smallest amount of differencing needed to make a time series stationary
while preserving as much long-range memory as possible. Integer differencing
(returns) destroys memory; fractional differencing finds the minimum `d ∈ [0,1]`
that produces a stationary series.

### Formula
Fixed-Width Window (FFD) uses the binomial-series weights
`w_k = -w_{k-1} * (d - k + 1) / k`, with `w_0 = 1`, truncated once
`|w_k| < threshold` (default `1e-5`). Then
`X_t^d = Σ_{k=0}^{K-1} w_k · X_{t-k}`.

### Source
AFML Ch. 5 (Fractional Differentiation).

### Range
Unbounded. For `d=0` the output equals the input (identity). For `d=1` it
equals first differences. Typical optimal d values:

| series type | typical optimal d |
|---|---|
| equity / futures prices | 0.3 – 0.7 |
| volumes | 0.1 – 0.5 |
| volatility measures | 0.0 – 0.3 |
| already-stationary features | 0.0 |

### Interpretation
- Low d value achieving stationarity → strong long-memory series; much
  historical information is being preserved.
- High d value required → series is highly non-stationary; less history
  usable for prediction.

---

## 2. Structural-break features

**Module**: [`structural_breaks.py`](../src/feature_factory/structural_breaks.py)

Continuous-valued features (for the ML layer), not event triggers. Three
distinct detectors:

### 2.1 `cusum_stat` — rolling CUSUM magnitude
- **Formula**: over a trailing window, track both one-sided
  `S_pos = max(0, S_pos + r_t)` and `S_neg = min(0, S_neg + r_t)` on
  log-returns; emit `max(sup S_pos, -inf S_neg)`.
- **Source**: AFML Ch. 2 (CUSUM filter) adapted to a rolling feature.
- **Range**: `≥ 0`. Larger = stronger evidence of a mean shift inside the
  window.

### 2.2 `sadf_stat` — Supremum Augmented Dickey-Fuller
- **Formula**: for each bar `t`, the max ADF t-statistic across all trailing
  window sizes `w ∈ [min_window, t]`. In `fast=True` mode, window sizes
  are sub-sampled log-spaced (≤ 12 per bar) so runtime stays tractable.
- **Source**: Phillips, Shi, Yu (2015); AFML Ch. 17 treatment.
- **Range**: roughly `[-5, +5]`. Large **positive** values signal
  explosive / bubble-like behavior.

### 2.3 `chow_stat` — rolling Chow F-test for trend change
- **Formula**: split the series at each candidate break `t`, fit two OLS
  trends, report `F = ((RSS_pooled - RSS_split)/k) / (RSS_split/(n - 2k))`
  with `k=2`.
- **Source**: AFML Ch. 17 Chow-type structural break discussion.
- **Range**: `≥ 0`. Peaks near true break points; small elsewhere.

### 2.4 `gsadf_stat` (opt-in, slow)
Generalized SADF: scans both start and end indices rather than a fixed end.
Sub-sampled for tractability. Off by default because of cost.

---

## 3. Entropy features

**Module**: [`entropy.py`](../src/feature_factory/entropy.py)

Low entropy → predictable patterns (tradeable). High entropy → randomness.

### 3.1 `shannon_entropy`
Per-window `H = -Σ p_i log₂ p_i` over **globally** quantile-binned log
returns. Global edges are used so a rolling window whose distribution is
narrow concentrates into few bins and scores low entropy.
- **Range**: `[0, log₂(n_bins)]`. With `n_bins=10`: `[0, 3.32]`.

### 3.2 `lz_entropy`
LZ76 complexity of binarized return sign, normalized by Kontoyiannis:
`h = c(n) · log₂(n) / n`.
- **Source**: AFML Ch. 18.
- **Range**: typically `[0, ~1]`. `≈ 1` for i.i.d. uniform binary;
  `≪ 1` for periodic/repetitive patterns.

### 3.3 `sample_entropy`
Richman & Moorman (2000) bias-corrected ApEn. `SampEn = -log(A/B)` where
A and B count template matches of length `m+1` and `m` respectively,
**excluding self-matches**.
- **Range**: `≥ 0`. Lower = more regular; higher = more random.

### 3.4 `approx_entropy` (opt-in)
Pincus 1991 Approximate Entropy. Kept behind a flag because SampEn is
strictly preferred (unbiased).

---

## 4. Microstructural features

**Module**: [`microstructure.py`](../src/feature_factory/microstructure.py)

All measure some flavor of liquidity or informed-trading probability.

| feature | formula (rolling window) | source | interpretation |
|---|---|---|---|
| `kyle_lambda` | `Cov(Δp, sv) / Var(sv)` | Kyle (1985), AFML Ch. 19 | ↑ = less liquid |
| `amihud_lambda` | `mean(|r| / dollar_volume)` | Amihud (2002) | ↑ = less liquid |
| `roll_spread` | `2 √(-Cov(Δp_t, Δp_{t-1}))` (clipped at 0) | Roll (1984) | ↑ = wider effective spread |
| `vpin` | `mean(|buy − sell| / total)` | Easley et al. (2012) | ↑ = informed trading / vol spike warning |
| `order_flow_imbalance` | `Σ(buy−sell) / Σ(buy+sell)` | Johnson (DMA) | `∈ [-1, +1]`, sign = direction |
| `trade_intensity` | `tick_count / bar_duration_seconds` | industry standard | ↑ = more active trading |
| `hasbrouck_lambda` (opt-in) | cumulative IRF of dp to sv shock | Hasbrouck (1991), AFML Ch. 19 | ↑ = larger permanent impact |

VPIN is the star of this group: elevated VPIN precedes volatility events
and is used by the Bet Sizing layer as a size throttle.

---

## 5. GARCH / realized-vol features

**Module**: [`volatility.py`](../src/feature_factory/volatility.py)

Source: Sinclair (*Volatility Trading*).

| feature | formula | interpretation |
|---|---|---|
| `garch_vol` | conditional σ from rolling GARCH(1,1) refit every `refit_interval` bars; recursed in between via `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}` | current forecast vol |
| `vol_term_structure` | `RV(5) / RV(30)` | `> 1` = inverted curve / vol spike |
| `vol_of_vol` | rolling std of `garch_vol` | `↑` = regime transition risk |
| `rv_iv_spread` (opt-in) | `IV − RV` over the long window | positive = short-vol opportunity (VRP) |

**Range**: all non-negative (except `rv_iv_spread`). Annualization is the
default (`sqrt(252)` scaling).

**Interpretation**: high `vol_of_vol` is particularly dangerous for
trend-following; high `vol_term_structure` favors mean reversion over
momentum per the VRP signal's regime modifier.

---

## 6. NLP sentiment features

**Module**: [`sentiment.py`](../src/feature_factory/sentiment.py)

Scores from ProsusAI/finbert on per-bar fetched news articles, aggregated
with exponential decay.

| feature | description | range |
|---|---|---|
| `sentiment_score` | decay-weighted mean of `positive − negative` | `[-1, +1]` |
| `sentiment_mom_1d` | `sentiment_t − sentiment_{t-1}` | `[-2, +2]` |
| `sentiment_mom_3d` | `sentiment_t − sentiment_{t-3}` | `[-2, +2]` |
| `article_count` | news coverage volume in the lookback | `≥ 0` |

Source: Jansen Ch. 14–16. Article count itself is informative: coverage
spikes often precede large moves.

---

## 7. On-chain (crypto) features

**Module**: [`onchain.py`](../src/feature_factory/onchain.py)

Derived from Glassnode's exchange-flow and network metrics.

| feature | formula | range | interpretation |
|---|---|---|---|
| `net_flow` | `outflow − inflow` | any | `> 0` = accumulation (coins leaving exchanges) |
| `flow_ratio` | `inflow / outflow` | `≥ 0` | `> 1` = net selling pressure |
| `net_flow_zscore` | rolling z-score of `net_flow` | any | extremes = directional bets |
| `whale_tx_count` | # bars exceeding rolling-quantile cutoff | `≥ 0` | spikes precede large price moves |
| `whale_volume_ratio` | `Σ whale vol / Σ total vol` (window) | `[0, 1]` | whale dominance of recent flow |
| `nvm_ratio` | `price / active_addresses²` | `≥ 0` | low = undervalued relative to network |
| `addr_price_divergence` | rolling Pearson of addresses vs price | `[-1, +1]` | near 0 or negative = mean-reversion signal |
| `ssr` | `stablecoin_mcap / btc_mcap` | `≥ 0` | high = dry powder on sidelines |

---

## 8. Autoencoder latent features

**Module**: [`autoencoder.py`](../src/feature_factory/autoencoder.py)

Source: Jansen Ch. 20.

A denoising autoencoder trained monthly on the hand-crafted feature
matrix produces `encoding_dim` latent features (default 8) named
`ae_latent_0 .. ae_latent_{k-1}`. Bottleneck activation is `tanh`, so each
latent is bounded in `[-1, +1]`.

These AUGMENT, never replace, interpretable features. They capture
nonlinear combinations the meta-labeler might not find on its own.

---

## 9. Classical indicators

**Module**: [`assembler.py`](../src/feature_factory/assembler.py) (`_classical_block`)

| feature | formula | range |
|---|---|---|
| `rsi_14` | Wilder 14-period RSI | `[0, 100]` |
| `bbw_20` | `(2·2·σ) / SMA` (Bollinger width as fraction of mid) | `≥ 0` |
| `ret_z_5 / 10 / 20` | rolling z-score of close-to-close returns | any |

Still useful. RSI and Bollinger width survive the ML pipeline because
they are cheap, well-understood, and carry genuine marginal information
after controlling for fancier features.

---

## 10. Feature correlation

Some pairs of features *should* be correlated by construction. That's fine
as long as the meta-labeler can distinguish them; the concern is not
redundancy but collinearity so severe that regression-based models
destabilize.

### Expected strongly correlated groups

| group | features | reason |
|---|---|---|
| liquidity | `kyle_lambda`, `amihud_lambda`, `roll_spread` | all measure price impact per unit volume |
| informed flow | `vpin`, `order_flow_imbalance`, `hasbrouck_lambda` | all track directional trade pressure |
| volatility | `garch_vol`, `vol_of_vol`, `bbw_20` | share the underlying σ² state |
| structural breaks | `cusum_stat`, `sadf_stat`, `chow_stat` | all fire during regime change |
| entropy | `shannon_entropy`, `lz_entropy`, `sample_entropy` | share the same "randomness" dimension |

### Why this is OK

1. Tree-based models (LightGBM/XGBoost, our Phase 3 meta-labeler) are
   robust to multicollinearity — the split picks the best-performing
   variant and the others carry negligible weight.
2. Our MDA feature-importance pass (Phase 3) quantifies redundancy
   statistically and prunes correlated features whose drop degrades
   accuracy <1%.
3. The autoencoder block (`ae_latent_*`) naturally compresses redundant
   directions into fewer latents.

### When to worry

- Pairwise correlation > 0.95 across > 5% of the history.
- MDA importance of a feature = 0 for 5 consecutive retraining cycles.
- Linear models in downstream phases (e.g. factor risk model) — switch
  to PCA before regressing.

---

## 11. Feature stationarity & FFD d-selection

### Why FFD

All features fed to the ML layer must be stationary (bounded moments,
constant distribution) so that training-time and live-time distributions
match. Integer differencing destroys memory; FFD retains it.

### Default d values — what to expect

Running `FeatureAssembler.get_optimal_d_values()` after an assemble on
clean daily data typically returns:

| column | typical optimal d |
|---|---|
| `close` (price) | 0.3 – 0.5 |
| `volume` | 0.0 – 0.3 |
| `dollar_volume` | 0.3 – 0.5 |
| microstructure features | 0 (already stationary) |
| GARCH vol | 0 |
| entropy features | 0 |

### Post-hoc stationarisation pass

The assembler runs a two-pass `_stationarise`:

1. **Pass 1** — For each column failing ADF (p ≥ 0.05), search `[0.05, 1]`
   for the smallest `d` that brings it below threshold. If FFD still
   fails, try plain first differences.
2. **Pass 2** — After the common `dropna`, re-verify on the trimmed
   matrix; any straggler gets one more first-difference.

If a column still fails after both passes, the assembler logs a warning
(`could not stationarise {col}`) and leaves the column alone. These are
candidates for removal in the MDA pruning step.

### Diagnosing a feature that keeps failing

1. **Too little data** — ADF has weak power below ~200 bars. Check slice length.
2. **Bounded feature with rare outliers** — VPIN, RSI, entropy. The ADF
   test can reject stationarity spuriously. Visually inspect; if bounded
   and stationary in spirit, keep.
3. **Slow-moving rolling mean** — e.g. a 100-bar rolling quantile.
   Differencing often works; consider shortening the window.
4. **Regime-dependent** — sentiment score shifts with macro cycle. Use a
   shorter decay half-life or subtract a rolling mean.
