# Wang Trading Web App — API Contracts (v2)

Companion to [docs/web_app_design_v2.md](web_app_design_v2.md) §27. Concrete request/response shapes for every new v2 endpoint, plus updated DTO examples.

All examples assume base URL `/api/v1`. All timestamps are UTC ISO-8601. All probabilities are in `[0, 1]`. All bps values are signed basis points (negative = adverse to the operator).

## 0. Conventions

### 0.1 Standard Envelope

Every response is wrapped:

```json
{
  "as_of": "2026-05-15T18:45:21.193Z",
  "source": "trade_ideas_service",
  "staleness_seconds": 18.2,
  "source_freshness": {
    "bars": 2.1,
    "features": 4.7,
    "signals": 4.7,
    "onchain": 312.0,
    "sentiment": 84.5,
    "factor_model": 3600.0
  },
  "model_version": "meta_v1.7.2-2026-05-12",
  "regime": {
    "label": "trending_up",
    "probabilities": {
      "trending_up": 0.72,
      "trending_down": 0.08,
      "mean_reverting": 0.14,
      "high_volatility": 0.06
    },
    "as_of": "2026-05-15T18:44:00.000Z"
  },
  "warnings": ["sentiment source older than 60s"],
  "errors": [],
  "data": { /* endpoint-specific payload */ }
}
```

`source_freshness`, `model_version`, and `regime` are optional — present only when meaningful for the endpoint.

### 0.2 Error Format

HTTP status codes are honoured (`400`, `401`, `403`, `404`, `409`, `422`, `429`, `500`, `503`). The envelope still wraps the response when possible:

```json
{
  "as_of": "2026-05-15T18:45:21.193Z",
  "source": "scenario_service",
  "warnings": [],
  "errors": [
    {
      "code": "STALE_FACTOR_MODEL",
      "message": "factor model older than 24h; scenario results degraded",
      "field": null
    }
  ],
  "data": null
}
```

`errors[i].code` is a stable identifier. The frontend should switch on `code`, not `message`.

### 0.3 Pagination

For list endpoints:

```text
GET /api/v1/audit/entries?limit=200&cursor=eyJ0c...
```

Response:

```json
{
  "data": {
    "items": [ /* ... */ ],
    "next_cursor": "eyJ0c..."
  }
}
```

`next_cursor` is omitted when no further pages.

### 0.4 Auth & Roles

All endpoints require an authenticated session. Roles are sent as a header `X-Role: viewer|operator|live_operator|quant_admin|admin`. Mutation endpoints additionally require:

- `X-Idempotency-Key: <uuid>`
- `X-Confirmation-Phrase: <typed phrase>` for danger-gated actions

### 0.5 Time Parameters

Where endpoints accept a time window:

- `from` and `to` accept ISO-8601 or relative shorthand (`-1h`, `-7d`).
- `as_of` accepts ISO-8601 only; defaults to "now".

---

## 1. Trade Ideas (extended)

### 1.1 `GET /api/v1/trade-ideas`

Query params:

- `symbols`: comma-separated; optional
- `action`: comma-separated; optional (`BUY,SELL,WATCH,...`)
- `min_abs_weight`: float; optional
- `min_calibrated_prob`: float; optional
- `min_regime_fit`: float; optional
- `max_cost_bps`: float; optional
- `min_track_win_rate`: float; optional
- `diff_only`: bool; optional — return only rows that mutated since `since`
- `since`: ISO-8601; required if `diff_only=true`

Response `data`:

```json
{
  "idea_count": 23,
  "totals": {
    "buy": 9,
    "sell": 4,
    "watch": 7,
    "model_required": 2,
    "no_data": 1,
    "error": 0,
    "gross_target_weight": 0.184,
    "net_target_weight": 0.062
  },
  "ideas": [
    {
      "symbol": "AAPL",
      "action": "BUY",
      "target_weight": 0.012,
      "target_notional": 121443.0,
      "estimated_quantity": 612,
      "latest_price": 198.34,
      "latest_bar_at": "2026-05-15T18:44:00.000Z",
      "bar_type": "TIB",
      "bars_loaded": 500,
      "feature_rows": 500,
      "signal_count": 3,
      "top_signal_family": "ts_momentum",
      "top_signal_side": 1,
      "top_signal_confidence": 0.74,
      "avg_signal_confidence": 0.61,
      "meta_probability": 0.66,
      "calibrated_probability": 0.61,
      "regime": {
        "label": "trending_up",
        "probabilities": {
          "trending_up": 0.72,
          "trending_down": 0.08,
          "mean_reverting": 0.14,
          "high_volatility": 0.06
        },
        "as_of": "2026-05-15T18:44:00.000Z"
      },
      "regime_fit_score": 0.81,
      "bet_size": 0.012,
      "sizing_constraints_applied": ["kelly_cap", "atr_cap"],
      "strategy": "trend",
      "reason": "Long target 1.20% from trend; top signal ts_momentum, confidence 74%, meta 66%.",
      "expected_cost_bps": 6.0,
      "top_shap_feature": {
        "feature": "vol_term_structure",
        "value": 1.12,
        "contribution": 0.084,
        "abs_contribution": 0.084,
        "percentile": 0.93
      },
      "track_record_win_rate": 0.58,
      "track_record_n": 24,
      "stage_latency_seconds": {
        "data_fetch": 0.012,
        "feature_compute": 0.084,
        "signal_generation": 0.041,
        "meta_inference": 0.022,
        "sizing": 0.005,
        "target_generation": 0.018
      },
      "errors": []
    }
  ]
}
```

### 1.2 `GET /api/v1/trade-ideas/{symbol}`

Returns full `TradeIdeaDetail` for one symbol.

Response `data`:

```json
{
  "idea": { /* TradeIdea object above */ },
  "chain": [
    { "name": "bars", "status": "ok", "value": 500, "count": 500, "timestamp": "2026-05-15T18:44:00Z", "latency_seconds": 0.012 },
    { "name": "features", "status": "ok", "value": 500, "count": 500, "latency_seconds": 0.084 },
    { "name": "signals", "status": "ok", "value": 3, "count": 3, "latency_seconds": 0.041, "message": "ts_momentum, vrp, cs_momentum" },
    { "name": "regime", "status": "ok", "value": "trending_up", "latency_seconds": 0.004 },
    { "name": "model", "status": "ok", "value": 0.66, "latency_seconds": 0.022, "message": "meta_v1.7.2" },
    { "name": "calibration", "status": "ok", "value": 0.61, "latency_seconds": 0.001 },
    { "name": "sizing", "status": "ok", "value": 0.012, "latency_seconds": 0.005, "message": "kelly_cap, atr_cap binding" },
    { "name": "target", "status": "ok", "value": 0.012, "latency_seconds": 0.018 },
    { "name": "cost", "status": "ok", "value": 6.0, "message": "vwap algo, 14d rolling" }
  ],
  "signals": [
    { "timestamp": "2026-05-15T18:44:00Z", "family": "ts_momentum", "side": 1, "confidence": 0.74 },
    { "timestamp": "2026-05-15T18:44:00Z", "family": "vrp",        "side": 1, "confidence": 0.58 },
    { "timestamp": "2026-05-15T18:44:00Z", "family": "cs_momentum","side": 1, "confidence": 0.51 }
  ],
  "signal_metadata": {
    "ts_momentum": {
      "family": "ts_momentum",
      "lookbacks": [21, 63, 126],
      "weights": [0.5, 0.3, 0.2],
      "z_scores": { "z_21": 1.62, "z_63": 2.10, "z_126": 1.84 },
      "aggregate": 1.85
    },
    "vrp": {
      "family": "vrp",
      "iv": 0.225,
      "rv": 0.184,
      "vrp": 0.041,
      "vrp_percentile_rank": 0.78,
      "regime_modifier": { "ts_momentum": 1.1, "mean_reversion": 0.8 }
    },
    "cs_momentum": {
      "family": "cs_momentum",
      "decile_rank": 88,
      "lookback_return": 0.082,
      "z_score": 1.41,
      "skip_periods": 5
    }
  },
  "model": {
    "source": "mlflow_production",
    "version": "meta_v1.7.2-2026-05-12",
    "run_id": "8d8a2c4d83b6",
    "alias": "production",
    "trained_at": "2026-05-12T03:14:00Z",
    "n_training_events": 184213,
    "calibration": "isotonic",
    "calibration_age_days": 3,
    "feature_hash": "a91f...e3b"
  },
  "shap": [
    { "feature": "vol_term_structure", "value": 1.12, "contribution": 0.084, "abs_contribution": 0.084, "percentile": 0.93 },
    { "feature": "vpin",               "value": 0.42, "contribution": 0.061, "abs_contribution": 0.061, "percentile": 0.87 },
    { "feature": "ts_momentum_z_63",   "value": 2.10, "contribution": 0.058, "abs_contribution": 0.058, "percentile": 0.91 },
    { "feature": "amihud_lambda",      "value": -1.3, "contribution": -0.041,"abs_contribution": 0.041, "percentile": 0.18 },
    { "feature": "regime_prob_trend_up","value":0.72,"contribution": 0.038, "abs_contribution": 0.038, "percentile": 0.80 }
  ],
  "sizing": {
    "layers": [
      { "name": "afml",  "value": 0.018, "capped": false, "cap_reason": null },
      { "name": "kelly", "value": 0.014, "capped": true,  "cap_reason": "kelly_fraction=0.5" },
      { "name": "vol",   "value": 0.013, "capped": true,  "cap_reason": "vrp_haircut active" },
      { "name": "atr",   "value": 0.012, "capped": true,  "cap_reason": "atr_multiplier=2.0" },
      { "name": "final", "value": 0.012, "capped": false, "cap_reason": null }
    ],
    "constraints_applied": ["kelly_cap", "vrp_haircut", "atr_cap"],
    "side": 1,
    "final": 0.012
  },
  "features": {
    "rows": 500,
    "latest": {
      "vol_term_structure": 1.12,
      "rsi_14": 64.2,
      "bb_width": 0.041,
      "return_z": 1.18
    }
  },
  "microstructure": {
    "kyle_lambda": 2.4e-7,
    "amihud_lambda": -1.3,
    "roll_spread": 0.012,
    "vpin": 0.42,
    "hasbrouck_lambda": 3.1e-7,
    "order_flow_imbalance": 0.08,
    "trade_intensity": 1.4
  },
  "structural_breaks": {
    "cusum": 3.2,
    "sadf": 1.84,
    "gsadf": 2.01,
    "chow_p_value": 0.04
  },
  "onchain": null,
  "sentiment": {
    "score": 0.21,
    "momentum_24h": 0.08,
    "article_count_24h": 14
  },
  "bars": {
    "bar_type": "TIB",
    "n": 500,
    "latest": {
      "open": 197.81, "high": 198.92, "low": 197.55, "close": 198.34,
      "volume": 1284003, "vwap": 198.21, "tick_count": 4192,
      "buy_volume": 712001, "sell_volume": 572002,
      "imbalance": 0.109, "threshold": 0.085, "bar_duration_seconds": 312.0
    }
  },
  "cost_forecast": {
    "expected_total_bps": 6.0,
    "expected_slippage_bps": 4.1,
    "expected_market_impact_bps": 1.6,
    "expected_commission_bps": 0.3,
    "algo": "vwap",
    "window_days": 14,
    "n_observations": 41,
    "vs_twap_bps": 0.8,
    "vs_vwap_bps": 0.0
  },
  "track_record": {
    "symbol": "AAPL",
    "family": "ts_momentum",
    "trailing_90d": { "n": 24, "win_rate": 0.58, "avg_return": 0.0042, "median_holding_bars": 38 },
    "trailing_180d": { "n": 48, "win_rate": 0.55, "avg_return": 0.0037, "median_holding_bars": 41 },
    "all_time": { "n": 312, "win_rate": 0.53, "avg_return": 0.0029, "median_holding_bars": 44 }
  },
  "nl_explanation": "The system is calling BUY AAPL at target weight 1.20% (~$121k notional). Strongest signal is ts_momentum (long, confidence 0.74) — aggregate z-score 1.85 across 21/63/126-bar lookbacks. Meta probability 0.66, calibrated to 0.61. Regime is trending_up at 0.72 — favourable for momentum. Top SHAP contributor is vol_term_structure (+0.084). Sizing started at 1.80% (AFML), Kelly clipped to 1.40%, vol-adjustment (VRP haircut active) to 1.30%, ATR cap to 1.20%. Expected cost 6 bps. Trailing 90d win-rate on this family-symbol: 58% over 24 calls. No active alerts.",
  "related_alerts": [],
  "related_audit_entries": [
    { "entry_id": "ae_991240", "event_type": "META_LABEL_PREDICTED", "timestamp": "2026-05-15T18:44:00Z" }
  ]
}
```

---

## 2. Symbol-Level Detail Endpoints (new)

### 2.1 `GET /api/v1/symbols/{symbol}/microstructure`

Query: `window=200` (number of recent bars)

Response `data`:

```json
{
  "symbol": "AAPL",
  "window": 200,
  "latest": {
    "kyle_lambda": 2.4e-7,
    "amihud_lambda": -1.3,
    "roll_spread": 0.012,
    "vpin": 0.42,
    "hasbrouck_lambda": 3.1e-7,
    "order_flow_imbalance": 0.08,
    "trade_intensity": 1.4
  },
  "series": {
    "kyle_lambda": [[ /* ts */, /* value */ ]]
  }
}
```

### 2.2 `GET /api/v1/symbols/{symbol}/structural-breaks`

Response `data`:

```json
{
  "symbol": "AAPL",
  "latest": {
    "cusum": 3.2,
    "sadf": 1.84,
    "gsadf": 2.01,
    "chow_p_value": 0.04,
    "regime_break_detected": true,
    "as_of": "2026-05-15T18:44:00Z"
  }
}
```

### 2.3 `GET /api/v1/symbols/{symbol}/onchain`

For crypto assets only. Returns `404` for equities/futures.

```json
{
  "symbol": "BTC",
  "exchange_inflow_btc": 12421.0,
  "exchange_outflow_btc": 11200.0,
  "net_flow_btc": -1221.0,
  "whale_transactions": 84,
  "active_addresses": 921030,
  "stablecoin_supply_change_24h": 0.0082,
  "as_of": "2026-05-15T18:40:00Z"
}
```

### 2.4 `GET /api/v1/symbols/{symbol}/sentiment`

```json
{
  "symbol": "AAPL",
  "score": 0.21,
  "momentum_24h": 0.08,
  "momentum_7d": 0.12,
  "article_count_24h": 14,
  "article_count_7d": 84,
  "as_of": "2026-05-15T18:30:00Z"
}
```

### 2.5 `GET /api/v1/symbols/{symbol}/shap?event_ts=...`

`event_ts` optional; defaults to the latest meta-label prediction for the symbol.

```json
{
  "symbol": "AAPL",
  "event_ts": "2026-05-15T18:44:00Z",
  "model_version": "meta_v1.7.2-2026-05-12",
  "base_value": 0.41,
  "predicted_value": 0.66,
  "shap": [
    { "feature": "vol_term_structure", "value": 1.12, "contribution": 0.084, "abs_contribution": 0.084, "percentile": 0.93 }
  ]
}
```

### 2.6 `GET /api/v1/symbols/{symbol}/track-record`

Query: `window=90d|180d|365d|all`, `family=ts_momentum` (optional)

```json
{
  "symbol": "AAPL",
  "family": "ts_momentum",
  "window": "90d",
  "summary": { "n": 24, "win_rate": 0.58, "avg_return": 0.0042, "median_holding_bars": 38 },
  "equity_curve": [[ /* ts */, /* cum_pnl_pct */ ]],
  "calls": [
    {
      "timestamp": "2026-05-04T13:32:00Z",
      "action": "BUY",
      "target_weight": 0.011,
      "calibrated_prob": 0.62,
      "regime": "trending_up",
      "realised_return_pct": 0.0072,
      "holding_bars": 42,
      "agreed_with_outcome": true
    }
  ]
}
```

### 2.7 `GET /api/v1/symbols/{symbol}/news-overlay`

For Symbol Detail chart overlay.

```json
{
  "symbol": "AAPL",
  "from": "2026-05-08T00:00:00Z",
  "to":   "2026-05-15T18:45:00Z",
  "events": [
    {
      "ts": "2026-05-12T13:30:00Z",
      "kind": "earnings",
      "headline": "AAPL reports Q2 EPS beat",
      "article_count": 84,
      "sentiment_score": 0.31
    }
  ]
}
```

---

## 3. Portfolio (extended)

### 3.1 `GET /api/v1/portfolio/factor-decomposition`

```json
{
  "as_of": "2026-05-15T18:44:00Z",
  "factors": [
    { "name": "F1", "explained_variance_ratio": 0.42, "exposure": 0.18, "contribution_to_variance": 0.31 },
    { "name": "F2", "explained_variance_ratio": 0.21, "exposure": -0.04, "contribution_to_variance": 0.08 }
  ],
  "systematic_variance": 0.0094,
  "idiosyncratic_variance": 0.0031,
  "factor_loadings": {
    "AAPL": { "F1": 0.74, "F2": -0.12 }
  }
}
```

### 3.2 `POST /api/v1/portfolio/simulate`

Request body:

```json
{
  "apply": "current_targets",
  "filter": {
    "min_calibrated_prob": 0.55,
    "actions": ["BUY", "SELL"]
  }
}
```

Response:

```json
{
  "projected": {
    "gross_exposure": 0.31,
    "net_exposure": 0.11,
    "factor_exposures": { "F1": 0.21, "F2": -0.06 },
    "family_concentration": { "ts_momentum": 0.18, "vrp": 0.07 },
    "asset_class_concentration": { "equities": 0.24, "crypto": 0.06 }
  },
  "deltas": {
    "gross_exposure": 0.07,
    "net_exposure": 0.04
  },
  "breaker_headroom_after": {
    "max_gross_exposure": 0.19,
    "max_family_allocation_ts_momentum": 0.07,
    "daily_loss_pct": 0.014
  },
  "expected_cost_usd": 2412.0,
  "expected_cost_bps": 4.8,
  "estimated_duration_seconds": 1820,
  "affected_ideas": [
    { "symbol": "AAPL", "old_weight": 0.0, "new_weight": 0.012, "flipped": false, "expected_cost_bps": 6.0 }
  ],
  "warnings": []
}
```

---

## 4. Execution (extended)

### 4.1 `GET /api/v1/execution/cost-forecast`

Query: `symbol`, `side` (`-1|0|1`), `algo` (optional)

```json
{
  "symbol": "AAPL",
  "side": 1,
  "algo": "vwap",
  "window_days": 14,
  "n_observations": 41,
  "expected_total_bps": 6.0,
  "expected_slippage_bps": 4.1,
  "expected_market_impact_bps": 1.6,
  "expected_commission_bps": 0.3,
  "p25_total_bps": 3.4,
  "p75_total_bps": 9.1,
  "recommended_algo": "vwap"
}
```

---

## 5. Signals (extended)

### 5.1 `GET /api/v1/signals/family-regime-attribution`

Query: `window=180d`

```json
{
  "window": "180d",
  "matrix": [
    { "family": "ts_momentum",   "regime": "trending_up",     "n": 412, "win_rate": 0.61, "avg_return": 0.0048 },
    { "family": "ts_momentum",   "regime": "mean_reverting",  "n": 218, "win_rate": 0.47, "avg_return": -0.0012 },
    { "family": "mean_reversion","regime": "mean_reverting",  "n": 311, "win_rate": 0.58, "avg_return": 0.0034 }
  ]
}
```

### 5.2 `GET /api/v1/signals/family-shap-attribution`

```json
{
  "window": "30d",
  "families": [
    { "family": "ts_momentum",  "avg_shap_contribution": 0.062, "n_fires": 1241 },
    { "family": "vrp",          "avg_shap_contribution": 0.024, "n_fires": 884 }
  ]
}
```

---

## 6. Model (extended)

### 6.1 `GET /api/v1/model/calibration`

Query: `window=90d|180d`

```json
{
  "window": "90d",
  "buckets": [
    { "predicted_bucket_lo": 0.50, "predicted_bucket_hi": 0.55, "predicted_mean": 0.524, "observed_rate": 0.512, "count": 814 },
    { "predicted_bucket_lo": 0.55, "predicted_bucket_hi": 0.60, "predicted_mean": 0.572, "observed_rate": 0.561, "count": 612 }
  ],
  "brier_score": 0.218,
  "ece": 0.012,
  "as_of": "2026-05-15T18:00:00Z"
}
```

### 6.2 `GET /api/v1/model/regime`

```json
{
  "current": {
    "label": "trending_up",
    "probabilities": {
      "trending_up": 0.72, "trending_down": 0.08,
      "mean_reverting": 0.14, "high_volatility": 0.06
    },
    "as_of": "2026-05-15T18:44:00Z"
  },
  "sequence": [
    { "ts": "2026-05-15T18:00:00Z", "label": "trending_up",   "confidence": 0.71 },
    { "ts": "2026-05-15T17:00:00Z", "label": "trending_up",   "confidence": 0.66 }
  ],
  "training_metrics": {
    "best_val_loss": 0.412,
    "best_epoch": 17,
    "val_accuracy": 0.78,
    "train_accuracy": 0.83,
    "n_train": 41200,
    "n_val": 8240
  }
}
```

### 6.3 `GET /api/v1/model/rl-shadow`

```json
{
  "window_days": 30,
  "n_observations": 642,
  "hrp_metrics": { "total_return": 0.041, "sharpe": 1.14, "max_drawdown": -0.034, "n_trades": 218 },
  "rl_metrics":  { "total_return": 0.038, "sharpe": 1.02, "max_drawdown": -0.041, "n_trades": 264 },
  "paired_t_stat": 0.81,
  "p_value": 0.42,
  "rl_is_better": false,
  "significance": "insignificant",
  "promotion_eligibility": {
    "eligible": false,
    "reasons": ["months_span < 6", "no statistically significant outperformance"],
    "months_span": 4
  },
  "recent_decisions": [
    {
      "ts": "2026-05-15T18:00:00Z",
      "hrp_target": { "AAPL": 0.012, "NVDA": 0.018 },
      "rl_target":  { "AAPL": 0.010, "NVDA": 0.024 },
      "executed_target": { "AAPL": 0.012, "NVDA": 0.018 }
    }
  ]
}
```

### 6.4 `GET /api/v1/model/retrain-history`

```json
{
  "events": [
    {
      "ts": "2026-05-12T03:14:00Z",
      "type": "scheduled",
      "model_version": "meta_v1.7.2-2026-05-12",
      "outcome": "success",
      "gate_results": { "cpcv": true, "dsr": true, "pbo": true },
      "n_training_events": 184213
    },
    {
      "ts": "2026-04-30T03:14:00Z",
      "type": "drift_triggered",
      "model_version": "meta_v1.7.1-2026-04-30",
      "outcome": "success",
      "gate_results": { "cpcv": true, "dsr": true, "pbo": false },
      "n_training_events": 178104
    }
  ]
}
```

---

## 7. Backtests (extended)

### 7.1 `GET /api/v1/backtests/compare?a={run_id_a}&b={run_id_b}`

```json
{
  "a": { "run_id": "bt_2412", "label": "v1.7.2 baseline" },
  "b": { "run_id": "bt_2418", "label": "v1.7.2 + vrp tweak" },
  "headline_diff": {
    "sharpe":      { "a": 1.41, "b": 1.52, "delta":  0.11 },
    "max_drawdown":{ "a": -0.082, "b": -0.071, "delta":  0.011 },
    "turnover":    { "a": 0.81, "b": 0.94, "delta":  0.13 }
  },
  "gate_diff": {
    "cpcv": { "a": true, "b": true },
    "dsr":  { "a": true, "b": true },
    "pbo":  { "a": true, "b": false }
  },
  "monthly_returns_diff": {
    "2026-04": 0.004,
    "2026-03": -0.001
  }
}
```

---

## 8. Scenarios (new)

### 8.1 `GET /api/v1/scenarios/library`

```json
{
  "scenarios": [
    { "id": "spy_down_3", "label": "SPY -3%", "type": "parametric" },
    { "id": "btc_down_10","label": "BTC -10%","type": "parametric" },
    { "id": "vix_x15",    "label": "VIX x1.5","type": "vol_shock" },
    { "id": "corr_break", "label": "Cross-asset correlations -> 1","type": "correlation" },
    { "id": "hist_2020_03_16","label": "Replay 2020-03-16 shock","type": "historical" }
  ]
}
```

### 8.2 `POST /api/v1/scenarios/run`

Request:

```json
{
  "shocks": {
    "symbol_pct": { "SPY": -0.03 },
    "vol_multiplier": 1.2,
    "correlation_target": null,
    "factor_shift": { "F1": -0.02 },
    "liquidity_multiplier": 1.5
  },
  "apply_to": "current_targets"
}
```

Response:

```json
{
  "pnl_impact_usd": -41200.0,
  "drawdown_impact": -0.012,
  "factor_exposure_deltas": { "F1": -0.04, "F2": 0.01 },
  "breaker_headroom": {
    "max_daily_loss_pct_remaining": 0.008,
    "max_gross_exposure_remaining": 0.21
  },
  "affected_ideas": [
    { "symbol": "AAPL", "old_target": 0.012, "new_target": 0.008, "flipped": false },
    { "symbol": "TSLA", "old_target": 0.011, "new_target": -0.002, "flipped": true }
  ],
  "suggested_hedges": [
    { "long": "TLT", "short": "SPY", "ratio": 0.5 }
  ],
  "warnings": ["factor model freshness 2h — result still valid"]
}
```

---

## 9. Track Record (new, global)

### 9.1 `GET /api/v1/track-record`

Query: `family`, `symbol`, `regime`, `window`, `action`, `min_calibrated_prob`

```json
{
  "filter": { "family": "ts_momentum", "window": "180d" },
  "summary": {
    "n_calls": 1842,
    "win_rate": 0.56,
    "avg_return": 0.0033,
    "median_holding_bars": 41,
    "sharpe": 1.21
  },
  "equity_curve": [[ /* ts */, /* cum_pnl_pct */ ]],
  "hit_rate_heatmap": {
    "ts_momentum_trending_up": 0.62,
    "ts_momentum_mean_reverting": 0.47,
    "vrp_high_volatility": 0.58
  },
  "decay_vs_backtest": {
    "ts_momentum": { "live_wr": 0.56, "backtest_wr": 0.59, "delta": -0.03, "significant": false }
  }
}
```

---

## 10. Replay (new)

### 10.1 `GET /api/v1/replay?ts={iso_ts}&symbol={optional}`

```json
{
  "ts": "2026-05-04T13:32:00Z",
  "model_version": "meta_v1.7.1-2026-04-30",
  "regime": {
    "label": "trending_up",
    "probabilities": {
      "trending_up": 0.68, "trending_down": 0.10,
      "mean_reverting": 0.18, "high_volatility": 0.04
    },
    "as_of": "2026-05-04T13:30:00Z"
  },
  "ideas": [ /* TradeIdea[] as they were at ts */ ],
  "audit_chain_verified_to": "ae_982134",
  "warnings": []
}
```

---

## 11. Preflight (new)

### 11.1 `GET /api/v1/preflight`

```json
{
  "overall": "BLOCKED",
  "checks": [
    { "name": "operator_checkin", "status": "PASS", "evaluated_at": "2026-05-15T18:30:00Z" },
    { "name": "halt_sentinel_clear", "status": "PASS", "evaluated_at": "2026-05-15T18:30:00Z" },
    { "name": "preflight_disk", "status": "PASS", "evaluated_at": "2026-05-15T18:30:00Z" },
    { "name": "preflight_broker_health", "status": "FAIL", "evaluated_at": "2026-05-15T18:30:00Z",
      "reason": "Alpaca account margin headroom < 10%",
      "runbook": "docs/runbooks/broker_health.md" }
  ],
  "operator_sentinel": {
    "checkin_age_seconds": 240,
    "halt_active": false,
    "last_operator_action": { "ts": "2026-05-15T15:14:00Z", "action": "REVIEW_COMPLETE" }
  },
  "capital_deployment": {
    "phase": 2,
    "multiplier": 0.5,
    "days_in_phase": 9,
    "next_phase_eligible": false,
    "blockers": ["needs 5 more clean days"]
  },
  "infra_probes": {
    "database": "ok",
    "feature_store": "ok",
    "mlflow": "ok",
    "broker": "warning",
    "prometheus": "ok"
  },
  "reconciliation": {
    "max_abs_delta": 0,
    "symbols_out_of_sync": 0,
    "as_of": "2026-05-15T18:30:00Z"
  }
}
```

---

## 12. Monitoring (extended)

### 12.1 `GET /api/v1/monitoring/freshness-heatmap`

```json
{
  "as_of": "2026-05-15T18:45:00Z",
  "rows": ["AAPL", "MSFT", "NVDA", "BTC", "ETH"],
  "columns": ["bars", "features", "signals", "onchain", "sentiment", "factor_model"],
  "values": {
    "AAPL": { "bars": 2, "features": 5, "signals": 5, "onchain": null, "sentiment": 84, "factor_model": 3600 },
    "BTC":  { "bars": 1, "features": 3, "signals": 3, "onchain": 312, "sentiment": 64, "factor_model": 3600 }
  },
  "thresholds": {
    "bars":         { "ok": 30,  "warn": 120 },
    "features":     { "ok": 60,  "warn": 300 },
    "signals":      { "ok": 60,  "warn": 300 },
    "onchain":      { "ok": 900, "warn": 3600 },
    "sentiment":    { "ok": 600, "warn": 3600 },
    "factor_model": { "ok": 86400, "warn": 172800 }
  }
}
```

### 12.2 `GET /api/v1/monitoring/escalation-channels`

```json
{
  "channels": [
    { "id": "telegram_primary", "kind": "telegram", "configured": true, "last_delivery_at": "2026-05-15T17:14:00Z", "last_delivery_status": "ok" },
    { "id": "email_oncall",     "kind": "email",    "configured": false, "last_delivery_at": null }
  ]
}
```

---

## 13. Audit (extended)

### 13.1 `GET /api/v1/audit/event-timeline`

Query: `window=24h`, `bucket=15m`

```json
{
  "window": "24h",
  "bucket": "15m",
  "series": [
    {
      "ts_bucket": "2026-05-15T18:30:00Z",
      "counts": {
        "SIGNAL_GENERATED": 184,
        "META_LABEL_PREDICTED": 92,
        "BET_SIZED": 48,
        "ORDER_SUBMITTED": 6,
        "FILL_RECEIVED": 6
      }
    }
  ]
}
```

---

## 14. Streams (extended)

### 14.1 `GET /api/v1/stream/ops` (SSE)

Each event line:

```text
event: broker_heartbeat
data: {"ok": true, "latency_ms": 41, "ts": "2026-05-15T18:45:21Z"}

event: regime
data: {"label":"trending_up","confidence":0.72,"ts":"2026-05-15T18:45:00Z"}

event: drift_severity
data: {"n_features_drifted": 3, "worst": "vpin", "ts": "2026-05-15T18:45:00Z"}

event: breaker
data: {"name":"max_daily_loss","active":false}

event: alert
data: { /* AlertSummary */ }
```

### 14.2 `GET /api/v1/stream/diff` (SSE)

Each event represents a diff-rail entry:

```text
event: idea_new
data: {"symbol":"NVDA","action":"BUY","target_weight":0.014,"since":"2026-05-15T18:40:00Z"}

event: idea_flipped
data: {"symbol":"META","from":"BUY","to":"SELL","since":"2026-05-15T18:40:00Z"}

event: idea_weight_changed
data: {"symbol":"AAPL","old":0.010,"new":0.012,"delta_bps":20,"since":"2026-05-15T18:40:00Z"}

event: idea_error
data: {"symbol":"TSLA","kind":"new","error":"feature staleness"}

event: idea_cleared_error
data: {"symbol":"BABA","since":"2026-05-15T18:40:00Z"}
```

---

## 15. Mutation Endpoints (Phase 4)

All require `X-Idempotency-Key`. Danger-gated also require `X-Confirmation-Phrase`.

### 15.1 `POST /api/v1/preflight/check-in`

Request: empty body.
Response:

```json
{ "ok": true, "checkin_age_seconds": 0 }
```

### 15.2 `POST /api/v1/preflight/clear-halt`

Headers: `X-Confirmation-Phrase: CLEAR HALT`

Request body:

```json
{ "reason": "post-incident recovery, see runbook ABC" }
```

Response:

```json
{ "ok": true, "halt_active": false, "audit_entry_id": "ae_991240" }
```

### 15.3 `POST /api/v1/model/calibration/refit`

Roles: `quant_admin`, `admin`. Idempotent on key.

```json
{ "ok": true, "calibrator": "isotonic", "n_samples": 8421, "calibration_age_days": 0 }
```

### 15.4 `POST /api/v1/escalation/test`

Roles: `admin`.

```json
{ "channel_id": "telegram_primary" }
```

Response:

```json
{ "ok": true, "delivered": true, "latency_ms": 412 }
```

---

## 16. DTO Reference

The full TypeScript type set lives in `docs/web_app_design_v2.md` §27.4–27.10. This file shows concrete payloads; the v2 design doc shows the typed shapes.

When the frontend and backend diverge from these examples, the design doc is the source of truth. Update both in lockstep.
