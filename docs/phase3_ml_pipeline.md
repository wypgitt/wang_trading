# Phase 3 — Labeling Engine, Core ML, and Bet Sizing

Operator reference for everything under [`src/labeling/`](../src/labeling),
[`src/ml_layer/`](../src/ml_layer), and [`src/bet_sizing/`](../src/bet_sizing).
Each section answers three questions:

- **Why** the component exists (the problem with the naïve alternative).
- **How** it works (formulas + concrete examples, not just prose).
- **What to do** with it (operator-level knobs, failure modes, diagnostics).

---

## Table of contents

1. [Meta-labeling: splitting side from size](#1-meta-labeling-splitting-side-from-size)
2. [Triple-barrier labeling](#2-triple-barrier-labeling)
3. [Sample weights — why overlapping labels break ML](#3-sample-weights--why-overlapping-labels-break-ml)
4. [Purged k-fold CV + embargo](#4-purged-k-fold-cv--embargo)
5. [Meta-labeler (Tier 1 model)](#5-meta-labeler-tier-1-model)
6. [Feature importance — MDI / MDA / SFI / SHAP](#6-feature-importance--mdi--mda--sfi--shap)
7. [Bet-sizing cascade](#7-bet-sizing-cascade)
8. [Model retraining protocol](#8-model-retraining-protocol)

---

## 1. Meta-labeling: splitting side from size

**Modules**: [`src/signal_battery/`](../src/signal_battery) (side),
[`src/labeling/meta_labeler_pipeline.py`](../src/labeling/meta_labeler_pipeline.py),
[`src/ml_layer/meta_labeler.py`](../src/ml_layer/meta_labeler.py) (size).

### Why
A single ML model that predicts both *"is this a long or a short?"* AND
*"how much should I bet?"* is two hard problems mashed together. You need
a lot of labeled data to learn both jointly, and when the model underfits
it tends to compromise on both.

AFML's trick is to split the problem:

- **Primary (side) model** = the Signal Battery (Phase 2).
  Many cheap generators (ts-momentum, mean-reversion, stat-arb, …) each
  commit to a direction with **high recall, low precision** — they mostly
  fire when they should, but also fire a lot when they shouldn't.
- **Secondary (size) model** = the meta-labeler (LightGBM gradient booster).
  Given the side already chosen by the primary and the full feature matrix,
  it answers one binary question:
  **"will this trade work out?"**
  The output probability p is used to SCALE the bet, not to pick its direction.

### How (flow)

```
┌──────────────────────────┐    ┌──────────────────────────┐
│  Signal Battery          │    │  Feature Factory         │
│  (primary, cheap models) │    │  (FFD, entropy,          │
│  → side ∈ {-1, 0, +1}    │    │   microstructure, GARCH, │
│                          │    │   sentiment, …)          │
└────────────┬─────────────┘    └────────────┬─────────────┘
             │                               │
             ▼                               ▼
       ┌──────────────────────────────────────────┐
       │  MetaLabelingPipeline                    │
       │  - triple-barrier labels (P/L sign)      │
       │  - sample weights (uniqueness × ret)     │
       │  - features + one-hot(family) + side +   │
       │    confidence                            │
       └─────────────────────┬────────────────────┘
                             │
                             ▼
       ┌──────────────────────────────────────────┐
       │  MetaLabeler (LightGBM)                  │
       │  → P(profitable | side, features)        │
       │  → isotonic calibration on OOF preds     │
       └─────────────────────┬────────────────────┘
                             │
                             ▼
                    Bet Sizing Cascade
```

### What to do
- A primary signal emits `side ∈ {-1, 0, +1}`. `side = 0` means neutral —
  the meta-labeler never sees it; the bet-sizing cascade returns size 0.
- The meta-labeler's target y is binary: `1` if the triple-barrier trade was
  profitable, else `0`. The probability p fed to sizing is always P(profitable).
- `p < 0.5` means **skip the trade**, NOT "reverse it". The cascade enforces
  this by clipping raw AFML sizes to `[0, max_size]`; the sign comes from
  `side` (see §7).

### Sanity check
If the meta-labeler's CV accuracy isn't above 50%, something's wrong with
either the features, the triple-barrier horizon, or the signal families —
revisit [§6](#6-feature-importance--mdi--mda--sfi--shap) and
[§8](#8-model-retraining-protocol).

---

## 2. Triple-barrier labeling

**Module**: [`src/labeling/triple_barrier.py`](../src/labeling/triple_barrier.py).

### Why
Fixed-horizon labels (*"is the return after N bars positive?"*) ignore the
path the price took. A trade that drifted down 3 σ then recovered gets the
same label as one that climbed smoothly. Path-dependent risk drops out of
the loss function, so the model can't learn to avoid it.

The **triple-barrier method** captures the trade outcome the way a real
stop-loss / take-profit bracket would:

### How (barriers)

For every CUSUM-triggered event with side s ∈ {−1, +1}:

1. **Upper barrier** (physical upper):
   `entry × (1 + upper_multiplier × σ)`
2. **Lower barrier** (physical lower):
   `entry × (1 − lower_multiplier × σ)`
3. **Vertical barrier** (time expiry):
   event timestamp + `max_holding_period` bars (clamped to last bar).

σ is the daily volatility estimate (EWMA std of returns; we also support
swapping in GARCH conditional vol). Walk the price forward until one barrier
is hit; the label is:

| first touched | label | interpretation |
|---|---|---|
| upper | `+1` if `side = +1`, else `-1` | (long) take-profit / (short) stop-loss |
| lower | `-1` if `side = +1`, else `+1` | (long) stop-loss / (short) take-profit |
| vertical | `sign((exit/entry − 1) × side)` | held to expiry; sign of the P/L |

### Asymmetric multipliers by strategy type

Different signal families have different natural return distributions.
[`configure_barrier_asymmetry`](../src/labeling/meta_labeler_pipeline.py)
picks `(upper_mult, lower_mult)` accordingly:

| family (substring-matched) | profile | `upper × lower` | shape |
|---|---|---|---|
| `carry`, `funding` | very wide TP, tight SL | 3.0 × 1.0 | let winners run |
| `mean_reversion` | tight TP, wider SL | 1.0 × 1.5 | many small wins |
| `momentum`, `trend`, `crossover`, `donchian`, `breakout` | wide TP, tight SL | 2.5 × 1.0 | occasional big wins |
| `arb` (stat / cross-exchange) | symmetric | 1.5 × 1.5 | |
| default | symmetric | 2.0 × 2.0 | |

### Worked example — momentum long, σ = 2%

```
 entry = $100,  side = +1,  upper_mult = 2.5,  lower_mult = 1.0
 upper barrier = 100 × (1 + 2.5 × 0.02) = $105
 lower barrier = 100 × (1 − 1.0 × 0.02) = $98

 bar  price  upper?  lower?
  0   100.0    -       -
  1    99.5    -       -
  2   101.0    -       -
  3   104.8    -       -         ← approaching upper
  4   105.4    YES     -         ← upper touched first  → label = +1
```
The meta-labeler learns: "in this regime, with these features,
momentum-long signals like this one hit +2.5σ before hitting −1σ".

### Short worked example (mirror)
```
 entry = $100, side = −1, upper_mult = 2.5, lower_mult = 1.0
 Same physical barriers at 105 and 98.
 If price touches $98 first  → label = +1 (short profited)
 If price touches $105 first → label = −1 (short stopped out)
```

### Edge case — vertical expiry
When neither horizontal fires, we use `sign((exit/entry − 1) × side)`.
A flat ride (exit = entry) produces `label = 0`, which collapses to
`meta_label = 0` for the meta-labeler's binary target.

### Meta-labels vs triple-barrier labels
- **`label` ∈ {−1, 0, +1}** — triple-barrier output; sign-of-P/L.
- **`meta_label` ∈ {0, 1}** — `(label > 0).astype(int)`; the binary target
  for the meta-labeler. Only this downstream.

---

## 3. Sample weights — why overlapping labels break ML

**Module**: [`src/labeling/sample_weights.py`](../src/labeling/sample_weights.py).

### Why
Classical ML assumes samples are i.i.d. Triple-barrier labels are not: two
events fired at adjacent bars almost certainly share most of their label
period. Giving each sample weight 1 over-counts evidence — the model
"sees" the same price path three or four times and convinces itself the
edge is stronger than it is.

Four weight components together produce a principled per-sample weight:

### 3.1 Average uniqueness
Let `c_t` = number of events alive at bar t. Each event's uniqueness at
bar t is `1 / c_t`. The event's **average uniqueness** is the mean of
`1 / c_t` across bars where it's alive.

| scenario | uniqueness |
|---|---|
| solo event (no overlap) | 1.0 |
| two events overlap entirely | 0.5 each |
| three events overlap entirely | 0.33 each |
| partial overlap (A [5,15] & B [10,20]) | each gets `(5·1 + 6·0.5)/11 ≈ 0.773` |

Implementation uses a difference-array for O(N+T) concurrent counting and a
cumulative-sum trick for O(N+T) per-event means — fast enough for 10k+ events.

### 3.2 Sequential bootstrap (AFML Algorithm 4.5.2)
Naïve bootstrap draws samples i.i.d. — on overlapping labels this
double-counts the shared price path, same as the weighting issue above.
Sequential bootstrap re-weights **after each draw** so that overlapping
samples become less likely to be picked again:

```
φ = []   # drawn indices so far
for each draw:
    for each candidate i:
        conditional uniqueness = mean over t∈[start_i,end_i]
                                 of 1 / (1 + Σ_{k∈φ} 1_{t,k})
    sample ∝ conditional uniqueness
    append to φ, update the overlap counter
```

Our implementation uses cumulative-sum updates so one draw costs O(T + N),
not the O(N·T) of the naïve formulation.

### 3.3 Return attribution
Samples that sat through big price moves should count more than samples
that sat through tiny ones. The weight is:

```
w_i = | Σ_{t ∈ [s_i, e_i]}  r_t / c_t |
```

where `r_t` is the per-bar return. Dividing by `c_t` again avoids double-
counting across overlapping events. Finally rescale so the weights sum to N.

### 3.4 Time decay
Bias toward recent regimes. Rank events chronologically (0 = oldest,
1 = newest) and linearly interpolate:

```
decay(i) = oldest_weight + (newest_weight − oldest_weight) × rank_i
```

`oldest_weight = 1.0` disables decay (the default). Crypto runs typically
use `oldest_weight ≈ 0.25` (aggressive decay — half-life ≈ 3 months at
daily bars); equities can use `0.5` or `0.75`.

### 3.5 The composition
`compute_sample_weights` multiplies all three per-event pieces and
renormalises to sum to N:

```
sample_weight = uniqueness × return_attribution × time_decay
sample_weight ← sample_weight × N / Σ sample_weight
```

This drops straight into LightGBM / XGBoost / sklearn's `sample_weight=...`
argument.

---

## 4. Purged k-fold CV + embargo

**Module**: [`src/ml_layer/purged_cv.py`](../src/ml_layer/purged_cv.py).
Design-doc §7.4.

### Why
Standard k-fold CV picks random folds. On time-series with overlapping
labels, the training fold ends up holding samples whose label periods
OVERLAP the test fold's labels — the model literally trains on the same
price path it's supposed to be evaluated against. Measured accuracy
balloons; the strategy collapses live.

### Information-leakage example
```
 event index:  0  1  2  3  4  5  6  7  8  9
 event labels  span 5 bars each:

 event 4 label period: bars [4, 5, 6, 7, 8]
 event 6 label period: bars [6, 7, 8, 9, 10]

 If standard KFold picks TEST = {events 6, 7, 8, 9}
 and TRAIN = {events 0, 1, 2, 3, 4, 5}:

   event 4's LABEL is determined by bars 4-8
   event 6 (in test) has LABEL determined by bars 6-10
   → they share bars 6, 7, 8

 The model trained on event 4 has already "seen" most of
 the price path that determines event 6's label. Test accuracy
 ceases to be out-of-sample.
```

### Purge
Remove from the training set any sample whose label period
`[event_start, event_end]` overlaps the **union** of test-fold label
periods. In the example above, events 4 and 5 would be purged.

We use the merged span `[min(test_event_start), max(test_event_end)]` rather
than per-sample intersection — equivalent when the test fold is time-
contiguous (always true here; folds are position-slices of a time-ordered
panel), and faster.

### Embargo
Even after purging, autocorrelated features can leak information. A training
sample whose label period ends right before the test starts doesn't overlap,
but its features were computed from price data that touches the test block.

Solution: drop an additional **forward embargo** of `embargo_pct × N` samples
after each test fold's end. `embargo_pct = 0.01` on 10 000 samples = 100
samples dropped after each fold.

### Comparison, end-to-end

```
 Standard KFold                 Purged KFold (no embargo)         Purged KFold + 1% embargo
 ─────────────────              ─────────────────────────         ──────────────────────────
 train   [0 1 2 3 4 5]          train   [0 1 2 3]                 train   [0 1 2 3]    ← (4-5 purged)
 test    [6 7 8 9]              purged  [4 5]                     purged  [4 5]
 → LEAKY                        test    [6 7 8 9]                 embargo [10 11 …]
                                → clean                           test    [6 7 8 9]
                                                                  → even cleaner
```

### What to do
- Always use `PurgedKFoldCV`, never sklearn's `KFold`. `cross_val_score_purged`
  covers the common case and supports `accuracy`, `f1`, `roc_auc`,
  `neg_log_loss`, `log_loss`, etc.
- `n_splits=5`, `embargo_pct=0.01` are good defaults. For short runs
  (< 1000 samples) drop to `n_splits=3`.
- The `MetaLabeler.fit(..., labels_df=...)` path uses this internally for
  early stopping — pass `labels_df` to opt in; omit for a plain refit.

---

## 5. Meta-labeler (Tier 1 model)

**Module**: [`src/ml_layer/meta_labeler.py`](../src/ml_layer/meta_labeler.py).
Design-doc §7.1.

### Choice
Gradient-boosted trees (LightGBM primary, XGBoost fallback,
RandomForest as a sanity-check baseline). AFML recommended RandomForest;
industry consensus since has moved to GBMs for tabular financial data —
faster to train, more flexible regularisation knobs, native early stopping.

### Training flow
```
MetaLabeler.fit(X, y, sample_weight, labels_df) does:
  1. Run PurgedKFoldCV(5 folds, 1% embargo)
  2. For each fold:
     - fit with early stopping on the purged validation slice
     - record best_iteration + OOF predictions
  3. Refit on full data with n_estimators = round(mean(best_iters))
  4. Fit the isotonic calibrator on OOF preds (honest calibration —
     never uses in-sample probs if labels_df is supplied).
```

### Probability calibration (why)
Raw GBM probabilities are rank-correct but frequency-wrong. A raw output
of 0.8 doesn't necessarily mean 80% empirical win rate. Isotonic regression
learns the monotone mapping from raw → calibrated so the bet-sizing
cascade can treat p as a real probability.

Calibrator training uses **out-of-fold** predictions when `labels_df` is
supplied — don't calibrate on in-sample or the correction is a no-op.

### Hyperparameter tuning
`src/ml_layer/tuning.py`:
- Optuna TPE sampler (Bayesian), MedianPruner (kills weak trials after a
  few folds).
- Search space per design-doc §7.1:
  `learning_rate ∈ log[0.005, 0.3]`, `n_estimators ∈ [100, 2000]`,
  `max_depth ∈ [3, 10]`, `min_child_weight ∈ [1, 50]`,
  `subsample ∈ [0.5, 1]`, `colsample_bytree ∈ [0.3, 1]`,
  `reg_alpha`, `reg_lambda` ∈ log[1e-4, 10].
- Per-fold scoring + `trial.report(running_mean, step=fold)` so the pruner
  can kill before burning the full trial budget.

Typical budgets: 50 trials / 10 min on a 5-year daily dataset. Pass
`--tune` to `retrain_model.py` to run it; results persist to
`data/best_params/<symbol>_<model_type>.json`.

---

## 6. Feature importance — MDI / MDA / SFI / SHAP

**Module**: [`src/ml_layer/feature_importance.py`](../src/ml_layer/feature_importance.py).
Design-doc §7.5.

Four methods, each answering a different question. Use them together —
disagreement between them is the interesting signal.

### 6.1 MDI — Mean Decrease Impurity
In-sample gini-decrease (RF) or split-gain (GBM) aggregated across trees.

| when | why |
|---|---|
| ✅ always, after every fit | Free, fast, available in every backend. |
| ❌ never as the sole gate | Biased toward high-cardinality features; doesn't reflect out-of-sample value. |

### 6.2 MDA — Mean Decrease Accuracy (permutation under purged CV)
For each feature: refit the model per fold, record baseline score, shuffle
the feature column in the validation set, record the permuted score,
accumulate `baseline − permuted` as an MDA sample. One-sided t-test on
the resulting distribution gives a p-value.

**This is the most trustworthy importance measure** — it reflects what
the model actually loses when the feature's information is destroyed,
out-of-sample, accounting for interactions.

| when | why |
|---|---|
| ✅ before dropping features | Honest OOF signal. |
| ✅ weekly during retraining | Flags features that have stopped working. |
| ⚠️ expensive | O(n_features × n_repeats × n_splits) fit-equivalents. Ours uses the "fit once per fold, re-score many times" pattern to amortise. |

### 6.3 SFI — Single Feature Importance
Train a model on each feature ALONE, measure purged-CV accuracy.

| when | why |
|---|---|
| ✅ complement to MDA | A feature with high MDA but low SFI is interaction-powered; high SFI but low MDA means another correlated feature is stealing its gain. |
| ❌ solo | Ignores everything the model could learn with feature combinations. |

### 6.4 SHAP — per-row attribution
`shap.TreeExplainer` values. Each prediction gets a vector of feature
contributions that sum to the (log-odds) prediction.

| when | why |
|---|---|
| ✅ audit trail for individual trades | Regulatory / post-mortem: *why did the model size up this trade?* |
| ✅ global importance via `mean(|SHAP|)` | Often agrees with MDA but is cheaper on small datasets. |
| ❌ training-time feature selection | Too slow to run during every retrain. |

### How to decide what to drop — `select_features`
Keep a feature if **either** bar passes:

```
keep if MDA p-value < mda_pvalue_threshold (default 0.05)
     OR SFI score   > min_sfi_score         (default 0.51)
```

The union rule preserves **both** interaction-powered features (high MDA,
low SFI) AND standalone-strong features (low MDA because of correlation
absorption, high SFI). Continuous pruning over 5+ retrain cycles — flag
features whose MDA is never significant — is the design-doc recommendation.

---

## 7. Bet-sizing cascade

**Module**: [`src/bet_sizing/cascade.py`](../src/bet_sizing/cascade.py).
Design-doc §8.

Five layers. Each layer can only SHRINK the size — never grow it. The
function is monotonically non-increasing, so partial application (skipping
a layer when inputs are missing) is always safe.

```
 Meta-labeler p(profitable)           ●  Layer 1 — AFML sizing
          │
          ▼                        size₁ = (2Φ(z(p)) − 1) · max_size
 afml_size  (clipped ≥ 0)          z(p)  = (p − 0.5) / √(p(1−p))
          │
          ▼
 kelly_capped = min(afml_size,     ●  Layer 2 — Kelly cap
                    fractional_kelly(p, W, L))
          │                        f* = (pW − qL) / (WL), clipped to [0,1]
          ▼                        fractional = 0.25 · f*
 vol_adjusted = kelly_capped ×     ●  Layer 3 — Vol adjust (Sinclair/Clenow)
                 (avg_vol/cur_vol)
                 × (1 − 0.25 · 𝟙{VRP top-quartile})
          │
          ▼
 atr_capped = min(vol_adjusted,    ●  Layer 4 — ATR cap (Clenow, trend/futures)
                  max_atr_fraction)
          │                        max_frac = risk_per_trade · price /
          ▼                                     (atr × atr_multiplier)
 final_size = ±apply_risk_budget   ●  Layer 5 — Risk budget enforcement
               (magnitude caps)    · max_single_position (10% NAV)
          │                        · max_family_allocation (30%)
          ▼                        · max_gross_exposure (150%)
 signed fraction of NAV            · max_crypto_allocation (30%)
                                   · max_sector_exposure (20% — equities)
```

### What each layer assumes
| layer | input required | skipped when | constraint tag |
|---|---|---|---|
| 1 | `prob`, `side` | never — always runs | — |
| 2 | `family_stats[family]` | family unknown (warns + skips) | `kelly_cap` |
| 3 | `current_vol`, `avg_vol` | both equal or missing → pass-through | `vol_scaling`, `vrp_haircut` |
| 4 | `atr`, `price`, `point_value` | non-trend family AND not futures, OR missing ATR/price | `atr_cap` |
| 5 | `portfolio_nav`, optionally `current_positions` | single-cap always applies; others skipped without positions | `max_single_position`, `max_family_allocation`, `max_gross_exposure`, `max_crypto_allocation`, `max_sector_exposure` |

The returned dict carries ALL intermediate sizes plus
`constraints_applied: list[str]` — the audit trail for every trade.

### What happens when layers disagree
Each layer ratchets the size down only; the tightest constraint always
wins. If Layer 2 (Kelly) says 3%, Layer 4 (ATR) says 8%, and Layer 5 (family
cap) says 1%, the final size is 1%.

The `constraints_applied` list records EVERY tag that fired, not just the
binding one, so you can trace back *why* a borderline trade was pared down.

### Example — the short story of a trade
```
prob = 0.85  side = +1  family = "ts_momentum"  symbol = "AAPL"
current_vol = 0.04  avg_vol = 0.02  (stormy)
portfolio_nav = 1 000 000  NAV already 25% ts_momentum
family_stats[ts_momentum] = (avg_win=2%, avg_loss=1.5%)

 1. afml_size        = 2·Φ((0.35)/sqrt(0.1275)) − 1 ≈ 0.73
 2. kelly_capped     = min(0.73, 0.25·full_kelly(0.85, 0.02, 0.015)) ≈ 0.25
 3. vol_adjusted     = 0.25 · (0.02/0.04) = 0.125                       ← vol_scaling
 4. atr_capped       = 0.125  (ATR not supplied)
 5. single-pos cap   = min(0.125, 0.10) = 0.10                          ← max_single_position
    family cap        → 30% − 25% = 5% headroom → 0.05                   ← max_family_allocation
 final_size          = +1 · 0.05 = 0.05

 constraints_applied = ["vol_scaling", "max_single_position", "max_family_allocation"]
```

### Operator notes
- `family_stats` is where historical `avg_win` / `avg_loss` magnitudes
  from the triple-barrier backtest live. Rebuild weekly as part of the
  retrain job.
- Do **not** change `max_single_position_pct` under the hard 10% cap
  without updating the design-doc risk table.
- `--dry-run` at the retrain script level logs everything but doesn't
  promote — use it when tuning the cascade knobs.

---

## 8. Model retraining protocol

**Script**: [`scripts/retrain_model.py`](../scripts/retrain_model.py).
**Make targets**: `make retrain SYMBOL=...`, `make retrain-tune SYMBOL=...`,
`make retrain-all`.

### Frequency
| cadence | what runs | why |
|---|---|---|
| daily | `--use-best-params` refit per symbol | keeps calibration current with fresh bars |
| weekly | full Optuna retune on active symbols | rediscover drifting optima |
| monthly | autoencoder retrain (Phase 2) feeding back into the feature matrix | latent-feature refresh |
| quarterly | FinBERT sentiment retrain (Phase 2) | drift mitigation |

### Triggers — beyond the schedule
Retrain out of band when any of these fire:

- Live paper-trading hit rate drops >10pp below the retrain-time CV accuracy.
- Feature drift alarm (Phase 5 monitoring) — mean of any feature column
  shifts > 3σ from its training distribution.
- A new signal family is added to the battery (changes the one-hot vocab).
- MDA p-values flag ≥ 20% of features as "no-longer-significant" in the
  weekly importance report.

### Promotion gate
Implemented in `_promotion_gate_phase4_stub` — swap the body when Phase 4
lands. Phase 3 rule:

```
if incumbent is None:
    promote (first model ever)
elif challenger.mean_cv_score > incumbent.mean_cv_score + 0.01:
    promote
else:
    keep incumbent
```

Phase 4 replaces this with the three-gate CPCV / DSR / PBO check (design-doc §9):
- ≥ 60% of CPCV's 45 paths must show positive net returns.
- Deflated-Sharpe p-value < 0.05 (adjusts for strategy-search bias).
- Probability of Backtest Overfitting (PBO) < 40%.

### How to run
```bash
# Single symbol, fast path
python scripts/retrain_model.py --symbol AAPL --use-best-params

# Single symbol with retuning
python scripts/retrain_model.py --symbol AAPL --tune --n-trials 50

# Nightly cron (universe sweep, no tuning)
python scripts/retrain_model.py --all-symbols --use-best-params

# Weekend cron (universe sweep, with tuning)
python scripts/retrain_model.py --all-symbols --tune --timeout 3600

# Inspect-only (no promotion)
python scripts/retrain_model.py --symbol AAPL --use-best-params --dry-run
```

### What gets logged (MLflow)
Every run writes:
- **Params**: all tuned hyperparameters + `model_type`, `n_features`,
  `n_samples`, `calibrate`, `has_sample_weight`.
- **Metrics**: `mean_cv_score`, `std_cv_score`, per-fold `cv_fold_i`,
  `train_accuracy`.
- **Artifacts**: joblib-dumped MetaLabeler, `feature_names.json`, per-method
  importance CSVs under `importances/`.

Query the best run at any time:
```python
from src.ml_layer.model_registry import ModelRegistry

reg = ModelRegistry(experiment_name="meta-labeler-AAPL")
top = reg.get_best_model(metric="mean_cv_score", n=5)
```

---

## Appendix — module pointers

| concern | entry point |
|---|---|
| triple-barrier labels | [`src.labeling.triple_barrier`](../src/labeling/triple_barrier.py) |
| meta-labeling pipeline | [`src.labeling.meta_labeler_pipeline`](../src/labeling/meta_labeler_pipeline.py) |
| sample weights | [`src.labeling.sample_weights`](../src/labeling/sample_weights.py) |
| purged CV | [`src.ml_layer.purged_cv`](../src/ml_layer/purged_cv.py) |
| LightGBM / XGBoost meta-labeler | [`src.ml_layer.meta_labeler`](../src/ml_layer/meta_labeler.py) |
| Optuna tuning | [`src.ml_layer.tuning`](../src/ml_layer/tuning.py) |
| feature importance | [`src.ml_layer.feature_importance`](../src/ml_layer/feature_importance.py) |
| MLflow registry | [`src.ml_layer.model_registry`](../src/ml_layer/model_registry.py) |
| LSTM regime detector | [`src.ml_layer.regime_detector`](../src/ml_layer/regime_detector.py) |
| AFML probability→size | [`src.bet_sizing.afml_sizing`](../src/bet_sizing/afml_sizing.py) |
| Kelly | [`src.bet_sizing.kelly`](../src/bet_sizing/kelly.py) |
| full cascade | [`src.bet_sizing.cascade`](../src/bet_sizing/cascade.py) |
| retrain orchestrator | [`scripts.retrain_model`](../scripts/retrain_model.py) |
