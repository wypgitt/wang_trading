# `src/web` — Wang Trading BFF (scaffold)

Backend-for-frontend service for the v2 web app. See
[`docs/web_app_design_v2.md`](../../docs/web_app_design_v2.md) and
[`docs/api_contracts_v2.md`](../../docs/api_contracts_v2.md) for the
contract this code implements.

## Run

```bash
pip install fastapi uvicorn
uvicorn src.web.app:app --reload --port 8080
```

`fastapi` and `uvicorn` are not yet in `requirements.txt` — add them
when you commit to FastAPI as the BFF runtime.

Smoke check:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/api/v1/trade-ideas
```

## Layout

```text
src/web/
├── __init__.py
├── app.py                          FastAPI application + route mounts
├── envelope.py                     ApiEnvelope + RegimeSnapshot
├── dtos.py                         All Pydantic v2 DTOs in one file
├── README.md                       this file
├── routes/
│   ├── __init__.py
│   ├── overview.py                 GET /api/v1/overview          (stub)
│   ├── trade_ideas.py              GET /api/v1/trade-ideas       (wired)
│   ├── scenarios.py                /api/v1/scenarios/{library,run}
│   ├── replay.py                   GET /api/v1/replay
│   └── preflight.py                GET /api/v1/preflight         (stub)
└── services/
    ├── __init__.py
    ├── trade_ideas_service.py      bridges to src.ui.trade_ideas
    ├── regime_service.py           regime snapshot + family fit (stub)
    ├── scenario_service.py         shock engine (stub returns mock)
    ├── replay_service.py           audit-chain reconstruction (stub)
    └── preflight_service.py        go-live checklist (stub)
```

## What's wired

- `GET /healthz`
- `GET /api/v1/trade-ideas` — calls the existing
  `src.ui.trade_ideas.generate_trade_idea_report_sync` and adapts the
  legacy dict shape into the v2 `TradeIdea` DTO. Fields the v2 DTO adds
  beyond v1 (`regime`, `regime_fit_score`, `expected_cost_bps`,
  `top_shap_feature`, `track_record_*`) are populated as `null` until
  the supporting services land.

## What's stubbed (returns mock or 501)

- `GET /api/v1/overview`
- `GET /api/v1/trade-ideas/{symbol}` (501)
- `GET /api/v1/scenarios/library` (deterministic library list)
- `POST /api/v1/scenarios/run` (deterministic mock response)
- `GET /api/v1/replay` (empty snapshot)
- `GET /api/v1/preflight` (stub status)

## What's not yet scaffolded

The v2 design doc lists 16 services and ~40 endpoints. The scaffold
focuses on the highest-leverage Phase-1 surfaces. Services to add next
(in priority order):

1. `shap_service.py` — per-row SHAP from `src.ml_layer.feature_importance`.
2. `cost_forecast_service.py` — rolling-TCA-derived expected cost per symbol-side-algo.
3. `signals_service.py` — per-family metadata payloads + correlation + regime attribution.
4. `model_service.py` — model status, calibration, drift, retrain history.
5. `portfolio_service.py` — positions, targets, factor decomposition, **simulator**.
6. `execution_service.py` — orders, fills, TCA, reconciliation.
7. `track_record_service.py` — historical call performance.
8. `monitoring_service.py` — Prometheus snapshot + freshness heatmap + escalation panel.
9. `audit_service.py` — audit chain + event timeline.
10. `nl_explain_service.py` — natural-language paragraph per idea.
11. `diff_service.py` — diff-rail SSE source.
12. `freshness_service.py` — per-source per-symbol staleness vector.
13. `rl_shadow_service.py` — HRP vs RL paired stats.
14. `microstructure_service.py`, `onchain_service.py`, `sentiment_service.py` —
    feature-factory bridges for the Symbol Detail tabs.

## Conventions

- Every response wraps in `envelope(...)`. Routes never return naked dicts.
- Services never import FastAPI — they return DTOs so they can be tested
  without an HTTP layer.
- Lazy-import heavy `src.*` modules (pipeline, ML) inside service methods
  so the BFF imports cleanly in environments that lack the full stack.
- No secret-bearing field may appear in any response. See
  `docs/web_app_design_v2.md` §6.2.

## Auth

Not yet implemented. The design doc specifies role-based headers
(`X-Role`, `X-Idempotency-Key`, `X-Confirmation-Phrase` for mutations);
add a FastAPI dependency when wiring Phase 4 mutation endpoints.

## Tests

Not yet implemented. Suggested first tests:

- `tests/web/test_envelope.py` — envelope serialisation, exclude-none behaviour.
- `tests/web/test_trade_ideas_adapter.py` — legacy dict → v2 DTO mapping.
- `tests/web/test_routes_boot.py` — every router can be imported and the
  FastAPI app starts.
