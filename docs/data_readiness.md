# Engine Data Readiness — Ground Truth for Aperture v1

Status: audit result · 2026-06-13 · Owner: YW
Companion to [aperture_app_design.md](aperture_app_design.md) (product) and
[web_app_design_v2.md](web_app_design_v2.md) (depth spec).

## What this is

A field-by-field, screen-by-screen map of **what the `wang_trading` engine actually
produces and persists today** vs. what the design docs assume. It exists to stop us
building beautiful UI for data that doesn't exist yet.

**Method.** A 25-agent audit read the real implementation across all 12 subsystems
(`src/data_engine`, `feature_factory`, `signal_battery`, `labeling`, `ml_layer`,
`bet_sizing`, `portfolio`, `execution`, `monitoring`, `backtesting`, `ui/trade_ideas`,
`web`). Every optimistic claim was re-checked by an adversarial verifier against source,
and the five most load-bearing claims were spot-checked by hand. Classifications:

- **PRODUCED** — computed as real runtime output *and* wired into a runnable path.
- **PARTIAL** — implemented but incomplete / gated off / depends on data not flowing.
- **SCAFFOLD** — signature exists, body is stub / placeholder / returns a constant.
- **ABSENT** — not in code.

## The one-paragraph verdict

**Only two things the engine produces are persisted and HTTP-reachable today: the
TimescaleDB `bars` hypertable and the tmpfs trade-ideas snapshot.** Everything else —
features, signals, the 5-layer sizing cascade, the entire `src/portfolio` risk library,
every backtest metric, all pipeline monitoring metrics, regime, SHAP, cost, track
record — is **computed in-process and discarded**. `FeatureStore.save_features` has zero
callers; the deployed trade-ideas path stops before order routing (`live_orders_sent=0`,
[trade_ideas.py:206](src/ui/trade_ideas.py:206)), so no orders/fills/positions/audit rows
are ever written. So a *read-only BFF* is enough for a narrow real v1 (Markets, Trade
Ideas list, Symbol candles), but every "impressive" screen is gated on **net-new
persistence/wiring work in the engine**, not on frontend effort.

## The single most important picture

| Layer | Persisted today? | Backs which screens |
|---|---|---|
| `bars` hypertable (OHLCV + microstructure cols) | ✅ TimescaleDB | Markets, Symbol candles |
| tmpfs `trade_ideas.json` snapshot (21 real fields) | ✅ tmpfs (overwritten each cycle) | Trade Ideas list |
| Everything else (features, signals, sizing, risk, backtests, metrics, audit) | ❌ ephemeral / never written | *(blocked until persisted)* |

## TradeIdea field readiness

The decision-chain drawer is the flagship — and it's mostly hollow today. Of the 29
v2 DTO fields:

**✅ PRODUCED (real, today)** — `symbol`, `action`, `target_weight`, `target_notional`,
`estimated_quantity`, `latest_price`, `latest_bar_at`, `bar_type`, `bars_loaded`,
`feature_rows` (count only), `signal_count`, `top_signal_family`, `top_signal_side`,
`top_signal_confidence`, `avg_signal_confidence`, `bet_size`, `strategy`, `reason`,
`stage_latency_seconds`, `errors`.

**⚠️ PARTIAL** — `meta_probability` / `calibrated_probability`: real GBM output **only
when an MLflow production model is loaded** (else null → `MODEL_REQUIRED`). And the two
are **collapsed to the same value** — the live pipeline writes one already-calibrated
scalar to both, so there is no raw-vs-calibrated split to show.

**❌ ABSENT (hardcoded null at the BFF boundary —
[trade_ideas_service.py:358-367](src/web/services/trade_ideas_service.py:358))** —
`regime`, `regime_fit_score`, `sizing_constraints_applied`, `expected_cost_bps`,
`top_shap_feature`, `track_record_win_rate`, `track_record_n`. Each has *real engine code
behind it that is never invoked or never persisted*: `RegimeDetector` has zero runtime
callers; the cascade emits `constraints_applied` but `trade_ideas.py` drops it;
`shap_importance` (TreeSHAP) is real but never called; there is no cost service and no
persisted call-history store.

> So the decision drawer's real content today = **signals + latency + (conditional)
> meta-prob**. Regime-fit, the sizing-cascade waterfall, SHAP, and cost are all blocked.

## Screen readiness

| Screen | Buildable now? | Gate |
|---|---|---|
| **Markets** | ✅ yes | none — bars hypertable |
| **Trade Ideas (list)** | ✅ yes | none — tmpfs snapshot via registered route ([app.py:204](src/web/app.py:204)) |
| **Symbol Detail** (candles) | ⚠️ partial | candles ✅; live-feature/microstructure overlay ❌ (features ephemeral) |
| **Trade Ideas drawer** | ⚠️ partial | signals + latency ✅; meta/cal conditional; regime-fit / cascade / SHAP / cost ❌; `GET /trade-ideas/{symbol}` is a 501 stub |
| **Overview** | ⚠️ partial | top-ideas/movers ✅; NAV / equity curve / key stats / regime ❌ (no persisted NAV series; `/overview` is a stub) |
| **Strategies grid** | ⚠️ partial | live ideas ✅; per-strategy Sharpe/win/PnL ❌ (backtest metrics never persisted); 6/10 families inert |
| **Model & Features** | ⚠️ partial | meta-prob histogram + thin MLflow timeline possible; calibration/drift/RL-shadow/importance ❌ |
| **Preflight & Go-Live** | ⚠️ partial → cheap win | rewire the BFF stub to the real `PreflightChecker` + `InfrastructureProbe` (both runnable); no persistence needed |
| **Portfolio & Risk** | ❌ no | no positions/NAV ever persisted; `src/portfolio` risk library has zero production callers |
| **Research & Backtests** | ❌ no | no backtest-run persistence; **retrain gate is hard-broken** (see below) |
| **Execution & TCA** | ❌ no | deployed path never routes orders; zero orders/fills/TCA rows written |
| **Monitoring & Alerts** | ❌ no | pipeline metrics live on a registry **nothing scrapes**; BFF `/metrics` is a disjoint registry |
| **Scenarios & Stress** | ❌ no | `ScenarioService` returns hardcoded mock numbers; `factor_risk` not even imported |
| **Track Record** | ❌ no | no append-only call history (snapshot overwritten each publish) |
| **Replay / Time Travel** | ❌ no | `ComplianceAuditLogger` never instantiated; no audit rows ever written |

## Corrected v1 build order

Each wave is gated only on what's named.

**Wave 1 — the one real spine (no new engine work).**
Markets · Trade Ideas (list) · Symbol Detail (candles + bar-level microstructure cols).
Backed by the `bars` hypertable + tmpfs snapshot. *This is shippable now.*

**Wave 2 — conditional on a loaded MLflow production model (no new persistence).**
Decision drawer (signals + latency + meta/cal only) · Model & Features (meta-prob
histogram + thin retrain timeline). Note the retrain timeline shows a placeholder
`cv_score=0.0` and no real gate verdicts until the engine bug below is fixed.

**Wave 3 — cheap on-demand probes (rewire one BFF stub).**
Preflight & Go-Live (live blocker checks + infra probes) · Overview (top-ideas/movers
tiles only). Just point `PreflightService` at the real checker.

**Wave 4 — gated on a feature-persistence bridge.**
Symbol live-feature overlay · drawer sizing-cascade waterfall + SHAP. Needs a
`save_features` caller, surfacing `constraints_applied` at the idea boundary, and
persisting `shap_importance`.

**Wave 5 — gated on execution/portfolio/backtest persistence.**
Portfolio & Risk · Execution & TCA · Research & Backtests · Overview NAV/equity tiles.
Needs an order-routing path that writes `ExecutionStorage`, backtest-run persistence,
the broken retrain gate fixed, and the `factor_risk` library actually wired.

**Wave 6 — gated on net-new services/stores.**
Monitoring (expose metrics over HTTP + persist alerts) · Scenarios (real engine) ·
Track Record (call-history store) · Replay (write the audit chain) · regime everywhere
(invoke + persist `RegimeDetector`) · calibration/drift/RL-shadow sections.

## Engine bugs & wiring gaps the audit surfaced (not UI)

These are pre-existing engine issues, independent of the app. Worth tracking:

1. **Retrain gate is hard-broken.** `_run_gate` falls through to
   `{"passed": False, "failing_gates": ["gate_unavailable"]}`
   ([retrain_pipeline.py:265](src/ml_layer/retrain_pipeline.py:265)) on every retrain —
   the gate fns are invoked with a mismatched signature and always throw. The only
   persisted backtest signal (3 MLflow gate flags) is therefore uniformly hardcoded
   `0/False`. CPCV and PBO never compute in any runnable entrypoint; only DSR runs via a
   CLI quick path.
2. **meta vs. calibrated probability collapsed** to one value in the live pipeline.
3. **Monitoring metrics are never scraped** — `start_server` runs on a fresh empty
   collector; the BFF `/metrics` serves a disjoint registry with only `bff_*` self-metrics.
4. **Drift emits a hardcoded `1.0`**, never sets a baseline; `get_drifted_features()`
   always returns `[]`.
5. **6 of 10 signal families are dead-on-arrival** — no caller supplies their
   panel/pair/exchange/futures context, so only ~4 bars-families (ts_momentum,
   mean_reversion, ma_crossover, donchian) can fire, on a single symbol, no history.
6. **`FeatureStore.save_features` has zero callers** — features are never persisted to
   Parquet despite the machinery existing.

## Implication for the design docs

[web_app_design_v2.md](web_app_design_v2.md) describes a destination, not the current
system. Its 15 pages, ~40 endpoints, Replay/Scenarios/Track-Record/Preflight, and the
full decision drawer assume persistence and wiring the engine **has not built yet**. That
doesn't make it wrong — it makes it the *roadmap*. Aperture v1 should be honest: ship
Waves 1-3 on real data, label the rest "coming as the engine lands it," and never render
a placeholder as if it were a live number.
