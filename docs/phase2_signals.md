# Phase 2 — Signal Battery

Reference for every primary-alpha signal in
[`src/signal_battery/`](../src/signal_battery). The Signal Battery is
Narang's *Alpha Model* layer: many uncorrelated modest-edge signals, all
run simultaneously, with arbitration pushed to the Phase-3 meta-labeler.

Each section covers:

- **Logic** — how the signal generates directional predictions.
- **Entry / Exit rules** — when the signal emits `side = ±1` or `0`.
- **Expected Sharpe** — standalone, pre-meta-labeling.
- **Asset class fit** — where the anomaly is strongest.
- **Best / worst regimes** — when to expect good and bad P&L.

Common `Signal` dataclass contract: `(timestamp, symbol, family, side,
confidence, metadata)`. Confidence is always in `[0, 1]` and passes
through to Bet Sizing.

---

## Table of contents

1. [Time-series momentum](#1-time-series-momentum)
2. [Cross-sectional momentum](#2-cross-sectional-momentum)
3. [Mean reversion (Ornstein-Uhlenbeck)](#3-mean-reversion-ornstein-uhlenbeck)
4. [Statistical arbitrage / pairs](#4-statistical-arbitrage--pairs)
5. [Trend following — MA crossover](#5-trend-following--ma-crossover)
6. [Trend following — Donchian breakout](#6-trend-following--donchian-breakout)
7. [Carry — futures roll yield](#7-carry--futures-roll-yield)
8. [Carry — crypto funding rate](#8-carry--crypto-funding-rate)
9. [Cross-exchange arbitrage](#9-cross-exchange-arbitrage)
10. [Volatility risk premium](#10-volatility-risk-premium)
11. [Signal correlation matrix](#11-signal-correlation-matrix)

---

## 1. Time-series momentum

**Module**: [`momentum.py`](../src/signal_battery/momentum.py) — `TimeSeriesMomentumSignal`

### Logic
Per-asset. For each lookback `L ∈ {21, 63, 126, 252}` bars (≈1, 3, 6, 12
months), compute the Sharpe-like score `z_L = r_L / (σ_daily · √L)`,
clip to `±3`, rescale to `[-1, +1]`. The aggregate is a weighted mean
across lookbacks (equal weights by default).

### Entry / Exit
- Every bar past warmup emits a Signal.
- `side = sign(aggregate)`, `confidence = |aggregate|`.

### Expected Sharpe
0.5 – 0.8 per the design doc (before meta-labeling).

### Asset class fit
All: equities, crypto, futures. Most robust anomaly in finance.

### Best / worst regimes
- **Best**: sustained one-way markets (early trend).
- **Worst**: regime transitions, choppy markets, mean-reverting ranges.
  Particularly bad right after a strong rally inverts.

### Source
Jegadeesh & Titman (1993); AQR "Time Series Momentum" literature;
AFML Ch. 17 sign-of-return framing.

---

## 2. Cross-sectional momentum

**Module**: [`momentum.py`](../src/signal_battery/momentum.py) — `CrossSectionalMomentumSignal`

### Logic
Panel-level. At each rebalance, rank symbols by 12-month return **skipping
the last 1 month** (skip-month avoids the short-term reversal effect).
Top decile → long; bottom decile → short.

### Entry / Exit
- Monthly cadence (typically).
- `side ∈ {-1, 0, +1}` per symbol; `confidence = 2·|rank − 0.5|`.
- Neutral positions explicitly have confidence 0.

### Expected Sharpe
0.3 – 0.6.

### Asset class fit
Equities primarily (requires ≥ 20 liquid symbols). Also works on futures
baskets and large-cap crypto.

### Best / worst regimes
- **Best**: late-cycle trending markets with strong dispersion.
- **Worst**: momentum crashes (2009, 2020 Q2). Can lose 20% in weeks.

### Source
Jegadeesh & Titman (1993).

---

## 3. Mean reversion (Ornstein-Uhlenbeck)

**Module**: [`mean_reversion.py`](../src/signal_battery/mean_reversion.py) — `MeanReversionSignal`

### Logic
1. Fit an O-U process to the price series, estimate half-life `h`.
2. Filter out: non-stationary (ADF p > 0.05), half-life outside `[1, 100]`.
3. Rolling z-score of price with window = `round(h)`.

### Entry / Exit
- Entry: `|z| > 2.0` → `side = -sign(z)` (fade the move),
  `confidence = min(|z|/4, 1)`.
- Exit: `|z| < 0.5` → `side = 0`.
- In between: hold (no signal emitted).

### Expected Sharpe
0.4 – 0.7.

### Asset class fit
Equity pairs, ETFs, crypto alts. Bad on trending futures.

### Best / worst regimes
- **Best**: range-bound, high-vol markets (wide O-U amplitude).
- **Worst**: sustained trends (will signal shorts all the way up).

### Source
Chan, *Quantitative Trading* / *Algorithmic Trading*.

---

## 4. Statistical arbitrage / pairs

**Module**: [`stat_arb.py`](../src/signal_battery/stat_arb.py) — `StatArbSignal`

### Logic
1. Identify cointegrated pairs (Engle-Granger or Johansen).
2. Track hedge ratio via Kalman filter (adaptive, slow δ).
3. Spread = `y − β_t · x`; compute rolling z-score with window = spread
   half-life.
4. Reject pairs whose spread fails ADF (Kalman smoothing can make
   anything look mean-reverting).

### Entry / Exit
- Entry: `|z| > 2.0` → short the overperformer, long the underperformer;
  `confidence = min(|z|/4, 1)`.
- Exit: `|z| < 0.5` → flatten.

### Expected Sharpe
0.5 – 1.0 per pair (higher than most single-asset signals because the
pair is market-neutral).

### Asset class fit
Equity pairs (sector- or factor-matched), ETF/components, some crypto.

### Best / worst regimes
- **Best**: stable cointegration regimes; earnings-season dispersion.
- **Worst**: cointegration breaks (merger, index exclusion, regulatory
  shock). Kalman filter will track the drift but lose money while doing so.

### Source
Chan, *Algorithmic Trading*; Elliott et al. Kalman-pairs literature.

---

## 5. Trend following — MA crossover

**Module**: [`trend_following.py`](../src/signal_battery/trend_following.py) — `MovingAverageCrossoverSignal`

### Logic
EMA-based dual (fast/slow) or triple (fast/medium/slow) MA crossover.
Triple-MA uses 2-of-3 voting: side is the direction a majority of the
pairwise comparisons (`fast > med`, `fast > slow`, `med > slow`) agree on.

### Entry / Exit
- Every bar past warmup emits a Signal.
- `side = +1` if fast EMA above slow (or majority-up); `-1` otherwise.
- `confidence = |fast − slow| / price`, clipped at 1.

### Expected Sharpe
0.5 – 0.9 on futures; lower on equities.

### Asset class fit
Futures and crypto shine. Adapt MA lengths per asset class — faster on
crypto (momentum decays faster), slower on bonds.

### Best / worst regimes
- **Best**: strong directional futures markets (oil, rates, bonds).
- **Worst**: whipsaw ranges — you'll flip long/short every few bars and
  pay friction on every flip.

### Source
Clenow, *Following the Trend*.

---

## 6. Trend following — Donchian breakout

**Module**: [`trend_following.py`](../src/signal_battery/trend_following.py) — `DonchianBreakoutSignal`

### Logic
Turtle-style. Enter on a breakout of the prior `entry_period` high/low
(default 55 bars); exit on a breakout of the prior `exit_period` high/low
(default 20 bars) in the opposite direction.

### Entry / Exit
- Stateful tracker: each signal is an explicit `"entry"` or `"exit"` event.
- Confidence = normalized distance from channel midpoint
  (`2 · |price − mid| / (high − low)`), clipped at 1.

### Expected Sharpe
0.5 – 0.9. ATR-based position sizing (`atr_position_size`) equalizes
dollar risk across instruments.

### Asset class fit
Futures and crypto. Historically the Turtle system's workhorse.

### Best / worst regimes
- **Best**: trending markets breaking multi-month ranges.
- **Worst**: narrow, mean-reverting ranges where every breakout fails.

### Source
Clenow, *Following the Trend*; Turtle Traders' rules.

---

## 7. Carry — futures roll yield

**Module**: [`carry.py`](../src/signal_battery/carry.py) — `FuturesCarrySignal`

### Logic
Annualized roll yield per contract:
`carry = (front − back) / front · (365 / days_between)`.

### Entry / Exit
- `carry > 0` (backwardation) → long (+1).
- `carry < 0` (contango) → short (-1).
- Confidence = `|carry| / rolling_max(|carry|, window=252)`, clipped.

### Expected Sharpe
0.3 – 0.6. Slow signal; diversifies momentum and mean reversion nicely.

### Asset class fit
Futures only: commodities (oil, metals, ags) and rates (bond rolls).

### Best / worst regimes
- **Best**: stable supply/demand regimes with consistent curve shape.
- **Worst**: curve inversion shocks (e.g. oil in 2014, 2020).

### Source
Clenow carry models; widely-traded CTA staple.

---

## 8. Carry — crypto funding rate

**Module**: [`carry.py`](../src/signal_battery/carry.py) — `FundingRateArbSignal`

### Logic
Delta-neutral: long spot + short perpetual. When funding is positive and
annualized funding exceeds entry threshold, enter; exit when funding
falls below exit threshold.

### Entry / Exit
- Entry (not in position, ann. funding > 10%) → `side = +1` (always
  "long" the yield).
- Exit (in position, ann. funding < 2%) → `side = 0`.
- Confidence = `clip(ann_funding / 0.5, 0, 1)`.

### Expected Sharpe
1.0+ in bull markets (funding can annualize to 15–40%). 0 in bear / flat
markets (funding flips negative).

### Asset class fit
Crypto perpetuals only.

### Best / worst regimes
- **Best**: parabolic bull runs where retail longs pay shorts generously.
- **Worst**: bear markets (negative funding → strategy dormant) and
  exchange-risk events (insolvency, withdrawal halts).

### Source
Crypto industry practice; academic coverage in Soska et al. and Alexander
"Crypto Options Market" (2021).

---

## 9. Cross-exchange arbitrage

**Module**: [`cross_exchange_arb.py`](../src/signal_battery/cross_exchange_arb.py) — `CrossExchangeArbSignal`

### Logic
For each bar, find max-price and min-price venue across exchanges. If
`spread_bps > min_spread_bps + fee_estimate_bps`, flag as an arb with
side `+1` (buy cheap, sell rich).

### Entry / Exit
- `MultiExchangePriceTracker` provides real-time true-bid/true-ask
  tracking with a `stale_after` filter (default 5 seconds).
- `side = +1` always active; `confidence = clip((spread − fees)/100, 0, 1)`.

### Expected Sharpe
1.0+ when opportunities are available. Zero most of the time on major
pairs (BTC/USDT), occasionally non-zero on mid-cap alts.

### Asset class fit
Crypto. Requires pre-positioned capital on multiple venues (transfer
latency is the killer).

### Best / worst regimes
- **Best**: news-driven dislocations; illiquid altcoins; exchange outages
  creating sustained price gaps.
- **Worst**: calm markets with tight cross-venue pricing. Also: expect
  zero alpha on BTC/ETH — those markets are arbitraged by HFT within ms.

### Source
Industry practice; academic treatment in Makarov & Schoar
"Trading and Arbitrage in Cryptocurrency Markets" (2020).

---

## 10. Volatility risk premium

**Module**: [`volatility_signal.py`](../src/signal_battery/volatility_signal.py) — `VolatilityRiskPremiumSignal`

### Logic
VRP = IV − RV. Compute rolling percentile rank over `vrp_lookback`
(default 30). Very high VRP → short vol; very low → long vol. Emits a
`regime_modifier` dict in metadata that boosts momentum's confidence
(1.2×) in high-VRP regimes and mean-reversion's confidence in low-VRP
regimes; this is consumed by the meta-labeler / bet sizer.

### Entry / Exit
- Percentile > 75 → `side = -1` (short vol / sell premium).
- Percentile < 25 → `side = +1` (long vol / buy protection).
- Otherwise: no signal.

### Expected Sharpe
0.3 – 0.5 on the direct short-vol trade; much larger value as a regime
modulator that improves other families.

### Asset class fit
Equities (VIX-based), BTC/ETH (Deribit IV). Not applicable to assets
without liquid options markets.

### Best / worst regimes
- **Best**: stable markets where IV consistently overprices realized
  risk.
- **Worst**: volatility spikes (Feb 2018 XIV blowup, March 2020). The
  short-vol trade loses explosively and the VRP signal amplifies
  momentum right into the turn.

### Source
Sinclair, *Volatility Trading*.

---

## 11. Signal correlation matrix

The premise of the Signal Battery is that **combined Sharpe** (via
HRP + meta-labeling) is much larger than any individual family's Sharpe
because the correlations across families are low or negative.

### Expected correlation structure

| pair | correlation | reason |
|---|---|---|
| TS momentum ↔ trend following (MA, Donchian) | **+0.6 to +0.8** | all three bet in the trend direction; trend following is essentially discrete TS momentum |
| CS momentum ↔ TS momentum | +0.3 to +0.5 | same anomaly, different framing; CS is cross-sectional so less correlated to any single name |
| momentum ↔ mean reversion | **−0.3 to −0.5** | directly opposing hypotheses; this is the main diversifier |
| stat-arb ↔ everything directional | ≈ 0 | pair trades are market-neutral by construction |
| carry ↔ momentum | +0.1 to +0.3 | carry trends tend to persist, overlapping with TS momentum |
| cross-exchange arb ↔ everything | ≈ 0 | structurally independent (market-neutral, latency-based) |
| VRP signals ↔ momentum | slight + | high-VRP regimes = low realized vol = trending markets |
| VRP signals ↔ mean reversion | slight + (low-VRP side) | low-VRP regimes = high realized vol = mean-reverting |

### Why this diversification matters

A single strategy with Sharpe 0.7 drawn down 20% hurts. Ten strategies
with Sharpe 0.4 each but low cross-correlation can compose to Sharpe
1.5+ **with a much smaller max drawdown**, because losses are unlikely
to align in time. That is the design doc's target and the entire
rationale for the Signal Battery's width.

### Negative-correlation gold

The **momentum vs mean-reversion pairing** is the single most valuable
one. During regime changes — when momentum is about to whipsaw — mean
reversion typically outperforms. The meta-labeler (Phase 3) uses
regime-detection features plus the VRP `regime_modifier` to shift
capital between them dynamically. A standalone Sharpe-0.5 mean-reversion
system cutting 50 % of momentum's drawdown during whipsaws is what pushes
the composite Sharpe past 1.5.

### How the system uses this

1. **HRP portfolio optimizer** (Phase 4) uses the realized signal
   correlation matrix to cluster families and allocate risk inversely to
   cluster variance. Negatively correlated families end up on opposite
   sides of the dendrogram → complementary risk budgets.
2. **Meta-labeler** (Phase 3) sees the per-family signal IDs and
   confidence scores as features; it learns regime-dependent weights.
3. **Risk-budget enforcer** (Phase 4) caps exposure at 30 % per family
   so no single anomaly can dominate the portfolio.
