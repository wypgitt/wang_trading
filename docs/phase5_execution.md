# Phase 5 — Execution Engine + Paper Trading + Monitoring

This document is the operator runbook for Phase 5. It covers execution
algorithm selection, circuit breakers, TCA interpretation, the paper
trading protocol, monitoring setup, daily operations, and troubleshooting.

---

## 1. Execution Algorithm Selection

`select_execution_algo(order, adv, urgency, asset_class, order_book_depth)`
routes each order to one of four algorithms based on order size relative
to ADV and urgency. The decision tree:

```
urgency == "high" ────────────────────────────► ImmediateAlgo (market fallback)
│
asset_class == "crypto" and size / depth > 0.5%
                     ────────────────────────────► IcebergAlgo
│
size / ADV < 0.1%   ────────────────────────────► ImmediateAlgo (limit-at-mid)
│
0.1% ≤ size / ADV ≤ 1%
                     ────────────────────────────► TWAPAlgo
│
size / ADV > 1%     ────────────────────────────► VWAPAlgo
```

**When to use each**

| Algorithm | Use case | Key parameter |
|-----------|----------|---------------|
| `ImmediateAlgo` | Small orders (< 0.1% ADV); urgent entries/exits | `timeout_seconds` (default 30 s), `use_market_fallback` |
| `TWAPAlgo` | Medium orders (0.1%–1% ADV); minimizes timing risk | `n_slices`, `duration_minutes`, `child_timeout_seconds` |
| `VWAPAlgo` | Large orders (> 1% ADV); matches market volume curve | `volume_profile` (default U-shape), `duration_minutes` |
| `IcebergAlgo` | Large orders on thin crypto books | `visible_size` (default 10% of total), `price_offset` |

**Limit-at-mid vs market fallback.** `ImmediateAlgo` first places a limit
order at the mid-price and waits `timeout_seconds`. If unfilled, it
cancels and either retries the remainder at market (`use_market_fallback=
True`, the default when urgency is "high") or lets the caller decide.
Use market fallback only for stop-loss exits and circuit-breaker
liquidations — never for entries.

---

## 2. Circuit Breaker Reference

All breakers live in `src/execution/circuit_breakers.py`. They split into
two tiers: **pre-trade** (synchronous, run on every `submit_order`) and
**portfolio-health** (periodic, emit `CircuitBreakerAction`).

### 2.1 Pre-trade breakers (reject the order)

| Breaker | Trigger | Parameter | How to reset |
|---------|---------|-----------|--------------|
| Fat finger | order notional > 5% NAV | `max_order_pct` | resize the order |
| Daily loss limit | `daily_pnl` < −2% NAV (blocks new entries; exits still allowed) | `daily_loss_limit_pct` | auto-resets at start of next trading day (reset `portfolio.daily_pnl`) |
| Max positions | ≥ 20 open positions and new symbol | `max_positions` | close positions or raise the limit |
| Max gross exposure | new gross > 150% NAV | `max_gross_exposure` | trim other positions first |
| Max single instrument | projected position > 10% NAV | `max_single_position_pct` | send a smaller order |
| Max crypto allocation | new crypto gross > 30% NAV | `max_crypto_pct` | reduce crypto exposure |

### 2.2 Portfolio-health breakers (emit actions)

| Breaker | Trigger | Action | Severity |
|---------|---------|--------|----------|
| Drawdown 10% | `drawdown ≥ 10%` | `REDUCE_SIZE_50` | WARNING |
| Drawdown 15% | `drawdown ≥ 15%` | `REDUCE_SIZE_75` | CRITICAL |
| Drawdown 20% | `drawdown ≥ 20%` | `HALT_AND_FLATTEN` | EMERGENCY |
| Model stale (warn) | last retrain > 30 days | `REDUCE_SIZE_50` | WARNING |
| Model stale (halt) | last retrain > 60 days | `HALT` | CRITICAL |
| Connectivity | broker heartbeat gap > 60 s | `FLATTEN_ALL` | CRITICAL |
| Data quality | bar-rate z-score > 3σ | `REDUCE_SIZE_50` | WARNING |
| Correlation spike | portfolio pairwise correlation > 0.80 | `REDUCE_GROSS_50` | WARNING |
| Dead-man switch | no operator check-in for 24 h | `FLATTEN_ALL_AND_HALT` | EMERGENCY |

### 2.3 Resetting a breaker

| Action | Manual reset procedure |
|--------|------------------------|
| `REDUCE_SIZE_*` | Automatic — new orders size-adjust while the condition persists, return to full size when it clears |
| `FLATTEN_ALL` | Automatic — resumes after connectivity restored |
| `HALT`, `HALT_AND_FLATTEN` | **Manual**: confirm root cause fixed, then restart the paper/live runner |
| `FLATTEN_ALL_AND_HALT` (dead-man) | Operator must call `OperatorCheckin.checkin()` — on disk at `logs/operator_checkin.txt` — then restart runner |

---

## 3. TCA Interpretation

Every fill is analyzed post-hoc by `TCAAnalyzer.analyze_order`. Metrics:

| Metric | Meaning | When to investigate |
|--------|---------|---------------------|
| `slippage_bps` | Execution price vs arrival mid (signed so positive = cost) | equities > 5 bps sustained; crypto > 15 bps |
| `market_impact_bps` | Price drift during window — attributable to the order | > 50% of slippage → algorithm moving the market |
| `timing_cost_bps` | Drift **not** attributable to order (adverse selection) | Large negative → consider switching from passive to aggressive |
| `benchmark_vs_twap_bps` | Fill price vs TWAP of prices during window | Consistently +ve → algo lagging market rhythm |
| `benchmark_vs_vwap_bps` | Fill price vs volume-weighted benchmark | Consistently +ve → VWAP curve mis-predicted |
| `fill_rate` | filled / target quantity | < 0.90 on TWAP/VWAP → child orders too conservative |

**Typical benchmarks (per asset class)**

| Asset class | Expected slippage | Expected impact | Investigate at |
|-------------|-------------------|-----------------|----------------|
| US equities (large-cap) | 1–3 bps | 0.5–2 bps | > 10 bps |
| US equities (mid-cap) | 3–8 bps | 2–6 bps | > 20 bps |
| Crypto (major pairs) | 5–15 bps | 2–10 bps | > 30 bps |
| Futures | 1–2 bps | 0.5–2 bps | > 10 bps |

**Degradation detection.** `TCAAnalyzer.detect_execution_degradation`
compares the recent 20 fills to a rolling 500-fill baseline; raises a
warning if recent mean slippage exceeds baseline mean + 2σ. Wire this
into the daily report.

---

## 4. Paper Trading Protocol

Before promoting to live capital:

| Gate | Requirement |
|------|-------------|
| **Duration** | ≥ 8 weeks continuous paper (design-doc §13 Phase 5) |
| **Trade count** | ≥ 200 filled trades so the TCA sample is non-trivial |
| **Sharpe (realized, net of costs)** | ≥ 1.0 on the paper window |
| **Max drawdown** | < 15% during the paper window |
| **Meta-labeler win rate** | ≥ 55% on filled trades |
| **Position reconciliation** | Zero unresolved discrepancies for 7 consecutive days |
| **TCA drift** | No sustained execution degradation alert in the final 2 weeks |
| **Circuit breaker incidents** | No EMERGENCY-level triggers in the final 2 weeks |

**Transition to live.** When all gates pass:

1. Freeze the production model and data-pipeline version in MLflow.
2. Switch `broker: paper` → `broker: alpaca_live` (or IBKR) in
   `config/paper_trading.yaml`, with 10% of target capital.
3. Re-run the full test suite + smoke test against the live broker in
   a staging account.
4. Run for 2 weeks at 10% before scaling to target capital.

---

## 5. Monitoring Setup

### 5.1 Start the local monitoring stack

```bash
docker compose up -d timescaledb redis prometheus grafana
```

Services:

- TimescaleDB: `localhost:5432`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin / admin)

### 5.2 Expose the metrics endpoint

The running paper-trading process exposes Prometheus metrics via
`MetricsCollector.start_server(port=9091)`. The scrape config is in
`config/prometheus.yml` (targets `host.docker.internal:9091`).

### 5.3 Import the Grafana dashboard and alerts

Create a Grafana API key (Configuration → API keys → admin role), then:

```bash
python scripts/setup_grafana.py \
    --grafana-url http://localhost:3000 \
    --api-key <grafana-api-key>
```

This installs the Prometheus + TimescaleDB datasources, imports the main
dashboard (6 rows: Portfolio, Positions, Signals & Model, Execution,
Data Health, Risk), and pushes the 6 alert rules generated by
`generate_alerting_rules()`.

### 5.4 Configure Telegram alerts

Create a bot via `@BotFather` and get your chat ID (start the bot, then
`curl https://api.telegram.org/bot<TOKEN>/getUpdates`). Then either:

- Set env vars `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` and rerun
  `setup_grafana.py` with `--telegram-bot-token` / `--telegram-chat-id`.
- Or wire `TelegramChannel` into your bootstrap's `AlertManager`:

  ```python
  AlertManager(channel_map={
      AlertSeverity.INFO:      [LogChannel()],
      AlertSeverity.WARNING:   [LogChannel(), TelegramChannel(token, chat)],
      AlertSeverity.CRITICAL:  [LogChannel(), TelegramChannel(token, chat)],
      AlertSeverity.EMERGENCY: [LogChannel(), TelegramChannel(token, chat)],
  })
  ```

---

## 6. Daily Operations Runbook

Run **every trading day** (or every 24 h for 24×7 crypto):

1. **Operator check-in.** Run `OperatorCheckin.checkin()` (or a tiny
   CLI wrapper) — prevents the dead-man switch from firing.
2. **Reconciliation.** `DailyReconciliation.run()` compares internal
   positions to broker, runs TCA, checks drift, persists snapshot, and
   alerts on discrepancies.
3. **Review the daily report.** `generate_daily_report` produces a
   Markdown summary: NAV, daily/MTD/YTD returns, top/bottom positions,
   trade count, avg slippage, drift warnings, breaker activations,
   model age. Read it before market open.
4. **Retraining check.** `RetrainScheduler.should_retrain()` is True
   every 7 days once ≥ 100 new bars are available. If True,
   `scheduler.retrain()` runs the full pipeline and promotes only if
   purged-CV metric improves.
5. **Reset `daily_pnl`.** At session start, clear
   `portfolio.daily_pnl = 0.0` so the daily-loss circuit breaker resets.

Weekly:

- Review cumulative TCA by algorithm and asset class.
- Spot-check feature drift dashboard — any feature > 0.5 KL for > 2
  days warrants investigation.
- Back up TimescaleDB + MLflow to GCS.

---

## 7. Troubleshooting Guide

### 7.1 Data gap

**Symptom.** `data_gap_seconds{symbol=…}` > 300 s; bar-formation-rate
panel drops to zero.

**Resolution.**
1. Check the ingestion log (`logs/ingestion.log`) for WebSocket errors.
2. Verify broker/exchange status page.
3. Restart the ingestion process: `make run-equities` or `make run-crypto`.
4. If the gap exceeds 5× the expected bar interval, the Data Quality
   breaker fires automatically (`REDUCE_SIZE_50`) — no manual action.

### 7.2 Model stale

**Symptom.** `model_last_retrain_age_hours` > 720 (30 days).

**Resolution.**
1. `RetrainScheduler.get_retrain_status()` — confirm `new_bars_since_retrain`
   is large enough (default min 100).
2. Run `make retrain SYMBOL=<sym>` manually.
3. If retrain fails: check the alert for the error, inspect
   `logs/retrain.log`. Common causes: feature NaN inflation, insufficient
   label diversity after triple-barrier.

### 7.3 Position reconciliation mismatch

**Symptom.** `DailyReconciliation.run()` returns non-empty
`discrepancies`; `AlertManager` fires a CRITICAL.

**Resolution.**
1. Dump the diff: each entry is `{symbol, internal_signed_qty,
   broker_signed_qty, delta}`.
2. Typical causes:
   - Manual broker intervention (positions opened outside the system).
   - Missed fill event — check `orders` and `fills` tables for the symbol.
   - Corporate action (split / dividend) — reconcile against the
     broker's cash-adjusted ledger.
3. To sync: either
   - Adopt broker state: `portfolio.positions = await broker.get_positions()`.
   - Or reject broker state: file a broker ticket and keep internal
     ledger as source of truth.
4. Until resolved, force-halt the engine (`shutdown()`).

### 7.4 Circuit breaker in persistent `HALT` state

**Symptom.** Engine not placing orders even after data/connectivity
are fine.

**Resolution.**
1. Inspect `CircuitBreakerAction` log — which breaker is active?
2. Address the root cause: retrain (if model-stale), close stopped
   positions (if drawdown-halt), check operator check-in (if dead-man).
3. Restart the paper/live runner — the `HALT` action requires
   intentional manual restart by design.

### 7.5 Feature drift flood

**Symptom.** `feature_drift_kl` heatmap mostly red; drift detector
flags > 50% of features.

**Resolution.** Major regime change likely. `FeatureDriftDetector
.recommend_action` returns *"Critical — major regime shift. Reduce
position sizes. Immediate retrain."* Execute the retrain immediately
and consider a temporary manual risk-reduction (cut gross 50%) until
the new model is validated.

---

## References

- Design doc §10 (Execution Engine), §12 (Monitoring), §13 (Roadmap
  Phase 5).
- Johnson, *Algorithmic Trading and DMA* — VWAP/TWAP, market impact.
- Narang, *Inside the Black Box* — TCA framework.
- López de Prado, *AFML* Ch. 21 — execution layer.
