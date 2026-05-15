# Wang Trading BFF — Backend Design & Development Plan

Status: design (v1)
Companion to [docs/web_app_design_v2.md](web_app_design_v2.md) (product/UX) and
[docs/api_contracts_v2.md](api_contracts_v2.md) (request/response shapes).
This document covers *how* the backend is built and shipped: topology,
service patterns, auth, observability, deployment, testing, and the
phase-by-phase development plan.

Audience: backend engineers, SRE, security review, anyone wiring the
React frontend against the BFF.

## Part I — Architecture

## 1. Goals & Non-Goals

### 1.1 Goals

- Provide a **read-mostly** HTTP/SSE API in front of the existing Python
  trading stack (`src.ui.trade_ideas`, `src.bet_sizing`, `src.ml_layer`,
  `src.execution`, `src.monitoring`, `src.backtesting`).
- Never duplicate trading logic. The BFF is a *thin* projection layer.
- Keep the existing local `src.ui.trade_ideas_app` working unchanged
  during migration; nothing forces a flag day.
- Stay independently testable: services are plain Python classes; the
  HTTP layer adds nothing the test layer can't reach.
- Operate safely without secrets in payloads, without unbounded queries,
  with explicit staleness, and with a deterministic envelope.
- Hit the v2 latency budgets in §17 even when the BFF runs on the same
  host as the trading process.

### 1.2 Non-Goals

- No business logic in the BFF — every meaningful computation happens
  in `src.*` modules.
- No GraphQL or ORM coupling. REST + Pydantic is enough.
- No multi-tenant SaaS — this is single-fund, single-tenant.
- No mobile-first work in Phase 1.
- No public exposure in Phase 1. Bind to localhost or VPN only.

## 2. System Topology

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ Operator host (single-tenant, single-process trading stack today)        │
│                                                                          │
│  ┌────────────────────────┐                                              │
│  │ React SPA (Vite build) │ ─── HTTPS ──> nginx reverse proxy            │
│  └────────────────────────┘                       │                      │
│                                                   │  /api/v1/*           │
│                                                   ▼                      │
│                                       ┌────────────────────────┐         │
│                                       │ uvicorn (FastAPI)      │         │
│                                       │  src.web.app:app       │         │
│                                       │  workers: 2 (uvloop)   │         │
│                                       └─────────┬──────────────┘         │
│                                                 │ in-process calls       │
│                                                 ▼                        │
│                                ┌──────────────────────────┐              │
│                                │ src.bootstrap pipeline   │              │
│                                │ src.* repositories       │              │
│                                └─────────┬────────────────┘              │
│              ┌──────────────────────────┼────────────────────┐           │
│              ▼                          ▼                    ▼           │
│       TimescaleDB              Parquet feature store   MLflow tracking   │
│       (bars/signals/audit)     (versioned features)    (model registry)  │
│                                                                          │
│  Prometheus scrapes /metrics on uvicorn AND on the trading process.      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 One Process or Two?

**Phase 1–2: one process.** The BFF imports the existing modules
directly. This keeps deployment simple and avoids serialising heavy
artefacts (feature frames, SHAP arrays) over IPC. The downside is that a
slow API call can block a hot pipeline thread. We accept that risk
because Phase 1 is read-only and we control the workload.

**Phase 4+: split if needed.** When mutation actions, broker connections,
and the SSE fan-out start competing with the trading loop, split the
BFF into its own process and talk to the trading process over a thin
local gRPC or Unix socket. Until profiling shows GIL or socket
contention, do not split.

### 2.2 Reverse Proxy

nginx terminates TLS, serves the static React bundle, and proxies
`/api/v1/*` to uvicorn. nginx also:

- enforces a 1 MB request body cap
- adds the `X-Request-Id` header when one isn't provided
- forwards `X-Forwarded-For` and `X-Forwarded-Proto`
- gzips JSON responses above 1 KB
- caches static assets with a hashed filename strategy

A `docker-compose` profile bundles nginx + uvicorn + Prometheus + Grafana
+ TimescaleDB for the dev environment. Production deployment uses
systemd units mirroring the compose profile.

### 2.3 Process Model & Concurrency

- **Workers**: 2 uvicorn workers in Phase 1. Each handles ~50 concurrent
  read requests. We do not autoscale workers based on load — the
  workload is bounded (the operator population is small).
- **Event loop**: `uvloop`. Required for SSE backpressure.
- **Blocking calls**: every call into `src.*` that does I/O (DB queries,
  MLflow, broker) must be wrapped in `asyncio.to_thread` or moved to a
  thread pool. Routes use `def` (sync) only when the entire call chain
  is non-blocking; otherwise `async def` + `to_thread`.
- **Long-running endpoints**: scenarios, replay, backtest compare. These
  may take seconds. They run on the FastAPI thread pool with a 30s
  timeout. Beyond 30s, the route returns `202 Accepted` with a job id
  and the client polls `/jobs/{id}` (Phase 3).

## 3. Service Layer Design

### 3.1 Layout

```text
src/web/
├── app.py                 FastAPI app, lifespan, route mounts
├── envelope.py            ApiEnvelope, RegimeSnapshot, error model
├── dtos.py                All Pydantic DTOs (split when > 600 lines)
├── deps.py                Dependency providers (DB, model registry, services)
├── auth.py                Role binding, idempotency check, audit hook
├── errors.py              Error code taxonomy, exception handlers
├── logging.py             structured logging config, request id, timing
├── metrics.py             Prometheus instrumentation for the BFF itself
├── cache.py               in-process LRU + TTL helpers
├── sse.py                 SSE broadcaster + backpressure
├── jobs.py                long-running job registry (Phase 3+)
├── routes/                APIRouter modules, one per domain
│   ├── overview.py
│   ├── trade_ideas.py
│   ├── symbols.py         microstructure, structural breaks, onchain, sentiment, shap, track-record, news overlay
│   ├── portfolio.py       summary, positions, targets, risk, factor decomposition, simulate
│   ├── execution.py       orders, fills, tca, reconciliation, cost forecast
│   ├── signals.py         summary, latest, correlation, family attribution
│   ├── model.py           status, calibration, drift, importance, regime, rl-shadow, retrain history
│   ├── backtests.py       runs, detail, compare
│   ├── scenarios.py       library, run, save
│   ├── replay.py
│   ├── preflight.py
│   ├── track_record.py
│   ├── monitoring.py      metrics snapshot, freshness heatmap, escalation channels
│   ├── alerts.py
│   ├── audit.py           entries, verify, event timeline
│   └── streams.py         /stream/ops, /stream/diff (SSE)
└── services/              business-logic-free service classes
    ├── trade_ideas_service.py
    ├── regime_service.py
    ├── shap_service.py
    ├── signals_service.py
    ├── model_service.py
    ├── portfolio_service.py
    ├── execution_service.py
    ├── monitoring_service.py
    ├── audit_service.py
    ├── backtest_service.py
    ├── scenario_service.py
    ├── replay_service.py
    ├── preflight_service.py
    ├── track_record_service.py
    ├── cost_forecast_service.py
    ├── freshness_service.py
    ├── rl_shadow_service.py
    ├── diff_service.py
    ├── escalation_service.py
    ├── microstructure_service.py
    ├── onchain_service.py
    ├── sentiment_service.py
    ├── calibration_service.py
    └── nl_explain_service.py
```

### 3.2 Service Contract

Every service class:

- Has no FastAPI imports.
- Takes its dependencies (DB manager, pipeline, registry) via the
  constructor, not via module-level globals.
- Returns Pydantic DTOs, never raw dicts or `pandas.DataFrame`.
- Logs at INFO when it starts a non-trivial call; logs at WARNING when
  it returns degraded data; raises domain exceptions (see §15) on
  failure.
- Is unit-tested with fakes for the `src.*` dependencies.

### 3.3 Dependency Injection

FastAPI's `Depends(...)` system carries services. A `src/web/deps.py`
module exposes provider functions:

```python
# src/web/deps.py (sketch)
from functools import lru_cache
from .services.trade_ideas_service import TradeIdeasService
from .services.regime_service import RegimeService

@lru_cache(maxsize=1)
def get_trade_ideas_service() -> TradeIdeasService:
    return TradeIdeasService(config_path="config/live_trading.yaml")

def get_regime_service() -> RegimeService:
    return RegimeService()

# in routes:
@router.get("")
def list_ideas(service: TradeIdeasService = Depends(get_trade_ideas_service)):
    return envelope(service.list_ideas().model_dump(mode="json"), source="trade_ideas_service")
```

This keeps services singleton-y for connection reuse but trivially
overridable in tests via `app.dependency_overrides`.

### 3.4 Lifespan & Resource Management

`src/web/app.py` uses FastAPI's `lifespan` context to:

- Bootstrap the database connection pool (TimescaleDB) once.
- Eagerly load the production model from MLflow at startup; cache its
  feature names, version, hash.
- Start the SSE broadcaster background task.
- Start the diff-rail computation background task.
- On shutdown: cancel background tasks, drain SSE connections, close
  the DB pool, flush logs.

Eager startup catches misconfigurations at boot instead of on the first
request. Failure during lifespan startup exits the process with a clear
error rather than serving 500s.

## 4. Data Access Patterns

### 4.1 TimescaleDB

- Connection pool: psycopg3 + `psycopg_pool.AsyncConnectionPool`,
  `min_size=2`, `max_size=10`. Pool is created in lifespan startup.
- All queries are parameterised. No string concatenation.
- All queries have an explicit `LIMIT`. The default is 200; the maximum
  is 5000 and only on the audit endpoint (cursor-paginated).
- Long-running aggregations (>500 ms) are precomputed nightly into
  continuous aggregates (Timescale) or materialised views, and served
  by the BFF directly. List of precomputes:

  | View | Refresh | Source |
  |---|---|---|
  | `mv_recent_signals_24h` | 1 min | `signals` table |
  | `mv_family_regime_attribution_180d` | 1 hour | `signals + meta_labels + labels` |
  | `mv_track_record_per_symbol_family` | 15 min | `positions_history + meta_labels` |
  | `mv_calibration_buckets_90d` | 15 min | `meta_labels + labels` |
  | `mv_drift_severity_24h` | 5 min | `features + baseline_features` |
  | `mv_audit_event_counts_24h` | 1 min | `audit_entries` |

### 4.2 Parquet Feature Store

- The feature store (under `data/feature_store/`) is read-only from the
  BFF. The BFF never writes Parquet.
- Reads go through `src.data_engine.storage.feature_store` so file
  layout changes don't ripple into the BFF.
- Point-in-time reads (`get_features_at(ts)`) must complete in <100 ms
  for the active universe. If a read exceeds that, log a warning with
  the symbol and `as_of` so the file layout can be tuned.

### 4.3 MLflow

- The BFF queries MLflow only via `src.ml_layer.model_registry`. Direct
  `mlflow` imports are forbidden in `src.web.*`.
- The production model is loaded once at startup and cached in the
  `ModelService` singleton. Cache key: `(stage, model_version)`.
- Model refresh is *event-driven* (the trading process announces a
  promotion via Prometheus + audit log) — not polled. Phase 4: listen
  for `PHASE_PROMOTED` audit events and reload the cached model.

### 4.4 Broker (Phase 4 only)

The BFF never opens broker sessions itself. Mutation routes (`halt`,
`flatten`, `verify-flat`) call into `src.execution.live_trading` which
owns the broker session. The BFF's role is auth, audit, idempotency,
and response shaping.

## 5. Caching Strategy

Three layers, in order of preference:

1. **DB-side (Timescale continuous aggregates / materialised views).**
   Best for aggregates over time windows. Refresh is bounded by the
   view's policy; the BFF reads instantly.
2. **In-process LRU + TTL (`src/web/cache.py`).** Best for small JSON
   payloads with high read volume: `/healthz`, `/preflight`,
   `/model/status`, `/scenarios/library`. Default TTL: 30 s.
3. **Pre-rendered response cache.** Reserved for the
   `/trade-ideas` and `/overview` endpoints. The trading process writes
   the latest `TradeIdeasResponse` JSON to a tmpfs path after each
   cycle; the BFF serves it directly when the cache file is fresh
   enough. This makes the hot path zero-compute for the BFF.

### 5.1 Cache Keys & Invalidation

| Endpoint | Cache | Key | TTL / Invalidation |
|---|---|---|---|
| `/healthz` | LRU | none | 5 s |
| `/preflight` | LRU | `role` | 10 s |
| `/model/status` | LRU | `()` | 60 s; busted on `PHASE_PROMOTED` |
| `/model/calibration` | LRU | `window` | 10 min |
| `/scenarios/library` | LRU | `()` | 1 h |
| `/trade-ideas` | tmpfs | `(filter_hash)` | 30 s or until cycle end |
| `/portfolio/factor-decomposition` | LRU | `()` | 5 min |
| `/track-record` | DB MV | `(family,symbol,regime,window)` | 15 min |
| `/scenarios/run` | LRU | `(scenario_hash, portfolio_hash, targets_hash)` | 60 s |
| `/replay?ts=` | LRU | `(ts, symbol)` | infinite (ts-bounded, immutable) |
| `/audit/entries` | none (cursor paginated) | — | — |

### 5.2 Cache Coherence

The cache is best-effort. Every cached response carries `as_of` from the
underlying data, *not* from the cache hit. The client must trust
`as_of`, not the wall clock of the request.

## 6. SSE & Real-Time

Two SSE channels (see [api_contracts_v2.md](api_contracts_v2.md) §14):

- `/api/v1/stream/ops` — broker heartbeat, regime, drift, alerts,
  breaker state.
- `/api/v1/stream/diff` — diff-rail events: new ideas, flipped sides,
  weight deltas, error transitions.

### 6.1 Broadcaster

`src/web/sse.py` exposes a `Broadcaster` singleton with `subscribe()`
and `publish(event, payload)`. Background tasks (registered in
lifespan) generate events:

- `ops_publisher` — wakes every 10 s, snapshots broker / regime / drift
  / breakers, diffs against last snapshot, publishes only on change.
- `diff_publisher` — listens for new `TradeIdeasResponse` payloads on a
  process-internal queue, computes the diff vs the previous payload,
  publishes individual chip events.

### 6.2 Backpressure & Disconnection

Each SSE connection holds an `asyncio.Queue(maxsize=50)`. If a slow
client fills the queue, the broadcaster drops the connection rather
than buffer unbounded. The client is expected to reconnect with the
`Last-Event-ID` header; the broadcaster replays at most 100 events from
a ring buffer.

### 6.3 Why SSE Over WebSockets

- Unidirectional fits Phase 1–3 (no client-initiated messages).
- HTTP/2 + nginx routes SSE without special config.
- Reconnection / replay semantics are baked into the protocol.
- Phase 4 mutation actions ride normal POST endpoints, not the stream.

If interactive features (e.g. live order entry) appear later, swap to
WebSockets at that point.

## 7. Background Jobs

| Job | Interval | Owner |
|---|---|---|
| ops snapshot publisher | 10 s | lifespan task |
| diff publisher | event-driven (queue) | lifespan task |
| trade idea cache writer | each cycle | trading process (not BFF) |
| materialised view refresh | per view | Timescale background worker |
| audit chain verification | 5 min | `audit_service` |
| freshness heatmap refresh | 30 s | lifespan task |

Long-running ad-hoc jobs (scenarios, replay, backtest compare beyond
30 s) use a tiny in-process registry (`src/web/jobs.py`):

```text
POST /api/v1/jobs           -> create  (returns id)
GET  /api/v1/jobs/{id}      -> status + result | progress
DELETE /api/v1/jobs/{id}    -> cancel
```

Persistent / cross-process jobs are out of scope for Phase 3. If needed
later, swap the in-process registry for Redis Streams or RQ.

## 8. Authentication & Authorization

### 8.1 Phase 1: VPN-only

Bind to `127.0.0.1` behind a Tailscale / WireGuard tunnel. No app-level
auth. The reverse proxy adds a default `X-Role: viewer` header. This is
the documented Phase 1 footprint.

### 8.2 Phase 2: SSO + role binding

OIDC against the operator's identity provider (Google Workspace by
default). On callback, mint a server-side session cookie keyed by user
+ role + expiry. Roles are sourced from a static `config/roles.yaml`
(checked into the repo, not into the model output) until a directory
group sync exists.

Roles (from [docs/web_app_design_v2.md](web_app_design_v2.md) §32):

```text
viewer | operator | live_operator | quant_admin | admin
```

### 8.3 Authorization

Each route declares its required role through a FastAPI dependency:

```python
@router.post("/api/v1/live/halt", dependencies=[Depends(require_role("live_operator"))])
def halt(...): ...
```

The dependency:

1. Reads the session cookie.
2. Resolves to a role from `roles.yaml`.
3. Compares to the declared minimum. Returns `403` on mismatch.
4. Emits an INFO log with `user, role, route, allowed`.

### 8.4 Idempotency

All mutation endpoints require `X-Idempotency-Key: <uuid>`. The BFF
stores `(key, response, expires_at)` in a small SQLite table on the
operator host (or Postgres if we go that route). Duplicate keys within
24 h return the original response without re-executing.

### 8.5 Confirmation Phrase

Danger-gated routes (`halt`, `flatten`, `clear-halt`, `promote`)
require `X-Confirmation-Phrase: <typed phrase>` matching a fixed
per-action string (e.g. `HALT NOW`). The frontend collects this in a
typed modal; the BFF rejects with `409 Conflict` on mismatch.

### 8.6 Audit Hook

Every mutation route writes a `OPERATOR_ACTION` audit entry through
`src.execution.audit_log` before invoking the underlying action. The
entry includes user, role, route, idempotency key, request body
fingerprint, and outcome. Audit failure is itself a `500` (we'd rather
fail loud than silently mutate).

## 9. Error Handling & Taxonomy

### 9.1 Error Codes

A flat enum in `src/web/errors.py`. The frontend switches on `code`,
never on `message`.

| Code | HTTP | Meaning |
|---|---|---|
| `BAD_REQUEST` | 400 | malformed input |
| `VALIDATION_FAILED` | 422 | Pydantic validation |
| `NOT_FOUND` | 404 | resource doesn't exist |
| `UNAUTHENTICATED` | 401 | missing session |
| `FORBIDDEN` | 403 | insufficient role |
| `CONFLICT` | 409 | idempotency / confirmation mismatch |
| `STALE_FACTOR_MODEL` | 200 | success with degraded scenario |
| `STALE_MODEL` | 200 | success with degraded probability |
| `MODEL_UNAVAILABLE` | 503 | no production model loaded |
| `BROKER_UNAVAILABLE` | 503 | broker disconnected |
| `DB_UNAVAILABLE` | 503 | TimescaleDB down |
| `TIMEOUT` | 504 | upstream timeout |
| `RATE_LIMITED` | 429 | per-user rate cap exceeded |
| `INTERNAL` | 500 | unhandled exception |

`STALE_*` returns are intentional 200s with a non-empty `errors` array:
the data is shippable, just degraded. The frontend renders a banner
based on the `code`.

### 9.2 Exception Handlers

`src/web/app.py` registers handlers for:

- `pydantic.ValidationError` → `422 VALIDATION_FAILED`
- Domain `DegradedResponse` → 200 + `STALE_*`
- Domain `BrokerUnavailable` → `503 BROKER_UNAVAILABLE`
- Generic `Exception` → `500 INTERNAL` with redacted message and a
  log entry tagged with `X-Request-Id`.

The generic handler must never include exception text or stack traces
in the response.

### 9.3 Timeouts

Every service call has an explicit timeout. Defaults:

- DB query: 5 s
- MLflow call: 10 s
- Broker call: 5 s
- Scenario engine: 30 s
- Replay reconstruction: 30 s

Exceeded timeouts raise `Timeout` and become `504`.

### 9.4 Retries

The BFF does *not* retry by default. The trading process owns retry
policy for upstream services; the BFF surfaces what it sees. If a
service flake is observed, fix it at the source, not by retrying in the
BFF.

## 10. Logging & Observability

### 10.1 Structured Logs

JSON to stdout. Captured by journald in prod, by `docker logs` in dev.
Every log line has:

```json
{
  "ts": "2026-05-15T18:45:21.193Z",
  "level": "INFO",
  "logger": "web.routes.trade_ideas",
  "request_id": "0b0e...",
  "user": "yp",
  "role": "operator",
  "route": "GET /api/v1/trade-ideas",
  "status": 200,
  "duration_ms": 84.2,
  "msg": "..."
}
```

Configured in `src/web/logging.py`. The middleware adds `request_id`,
`route`, `status`, and `duration_ms` automatically.

### 10.2 Tracing

OpenTelemetry SDK with the OTLP exporter, optional. Disabled by default;
enabled with `OTEL_EXPORTER_OTLP_ENDPOINT`. Spans for every route and
every service method that takes >50 ms.

### 10.3 BFF Self-Metrics

`src/web/metrics.py` exposes a Prometheus endpoint at `/metrics`
(distinct from the trading process's own `/metrics`). Counters and
histograms:

| Metric | Labels | Notes |
|---|---|---|
| `bff_requests_total` | `route, status` | counter |
| `bff_request_duration_seconds` | `route` | histogram, buckets `[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]` |
| `bff_sse_active_connections` | `stream` | gauge |
| `bff_sse_events_published_total` | `stream, event` | counter |
| `bff_cache_hits_total` | `cache` | counter |
| `bff_cache_misses_total` | `cache` | counter |
| `bff_upstream_errors_total` | `service, code` | counter |
| `bff_session_active` | — | gauge |

These feed Grafana dashboards (one BFF dashboard, separate from the
trading dashboard).

## 11. Performance Budgets

Per [docs/web_app_design_v2.md](web_app_design_v2.md) §30. Per-endpoint
p95 latency targets on the operator host (no network round-trip):

| Endpoint | p95 | Notes |
|---|---:|---|
| `GET /healthz` | 5 ms | LRU cached |
| `GET /api/v1/overview` | 80 ms | tmpfs cache |
| `GET /api/v1/trade-ideas` | 120 ms | tmpfs cache or DB scan |
| `GET /api/v1/trade-ideas/{symbol}` | 400 ms | full detail with SHAP |
| `GET /api/v1/portfolio/*` | 200 ms | |
| `POST /api/v1/portfolio/simulate` | 500 ms | factor recomputation |
| `GET /api/v1/execution/orders` | 200 ms | paginated |
| `GET /api/v1/audit/entries` | 200 ms | cursor paginated |
| `POST /api/v1/scenarios/run` | 5 s | budgeted; cached after first |
| `GET /api/v1/replay` | 5 s | budgeted; ts-immutable cache |
| `GET /api/v1/stream/ops` (publish) | 100 ms / event | SSE |
| `GET /api/v1/model/calibration` | 300 ms | DB MV-backed |

Each budget is enforced by a Prometheus alert that fires when the
30-minute rolling p95 exceeds 1.5× the target.

## 12. Pagination & Limits

- List endpoints accept `limit` (default 100, max 1000) and `cursor`.
- The cursor is opaque base64 of `(last_id, last_ts)` from the
  underlying table; the BFF does not expose row ids.
- A `next_cursor` field is omitted when the page is the last.
- Array responses that are inherently small (`/scenarios/library`,
  `/preflight`, `/freshness-heatmap`) are returned in full without
  pagination but capped at a documented maximum (50 / 80 / 200
  respectively).

## 13. Sensitive Data & Redaction

Hardcoded denylist of keys never serialised by any DTO or envelope:

```text
api_key, secret_key, signing_key, telegram_bot_token, mlflow_password,
database_password, broker_password, broker_account_id (when in
free-text), private_key, jwt_secret
```

`src/web/auth.py` adds a recursive filter that strips any key matching
`(^|_)(secret|password|token|key)($|_)` from any dict on its way out,
as a defence-in-depth on top of typed DTOs.

The full feature matrix (rows) is *never* returned via the BFF in
Phase 1. Only `latest` snapshots and selected SHAP top-N entries are.
A future research endpoint that exposes raw features will require
explicit role gating.

## 14. Rate Limiting & Abuse

Phase 1: none. Single-tenant, VPN-only.

Phase 2: per-session token bucket. Defaults:

- 600 requests / minute / session (read endpoints)
- 30 mutation requests / minute / session
- 10 SSE connections / session

Enforced by a FastAPI middleware backed by an in-process counter. If we
ever go multi-host, swap for Redis token buckets.

## 15. Security Headers & CORS

- HSTS, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: same-origin`, strict CSP for the React bundle
  (no unsafe-inline; styles from local; scripts from local).
- CORS: locked to the operator's frontend origin. No wildcards in
  production. Dev compose uses `http://localhost:5173`.
- Cookies: `Secure`, `HttpOnly`, `SameSite=Strict`, signed with a
  rotating server-side secret stored in the systemd unit's environment.

## 16. Configuration

- Settings live in `config/` YAML files (the same convention the
  trading process uses).
- The BFF reads `config/web.yaml` on startup. Schema:
  - `bind_host`, `bind_port`, `workers`
  - `database` (DSN, pool sizes)
  - `mlflow` (tracking URI, run id override)
  - `feature_store` (path, version)
  - `cache` (tmpfs path, TTLs)
  - `auth` (mode = `none | oidc`, provider config)
  - `cors_origins`
  - `roles_yaml`
  - `telemetry` (otlp endpoint optional, prom endpoint required)
  - `feature_flags` (which routes are mounted; default-on for v1, off
    for Phase 4 mutations until explicitly enabled)
- Secrets never live in `config/web.yaml`. They come from environment
  variables or a `.env` file mounted by systemd.

## Part II — Operations

## 17. Deployment Topology

### 17.1 Local Dev

```bash
make web-dev    # uvicorn --reload on :8080, React dev server on :5173
```

`scripts/dev_web.sh` brings up uvicorn with `--reload`, autocreates a
fresh `config/web.local.yaml`, and seeds the tmpfs cache.

### 17.2 Production (operator host)

systemd unit `wang-bff.service`:

- runs as user `wang`
- working dir `/opt/wang_trading`
- ExecStart `uvicorn src.web.app:app --host 127.0.0.1 --port 8080 --workers 2 --loop uvloop`
- Restart=on-failure
- Environment loaded from `/etc/wang-bff.env`
- Depends on `wang-trading.service` (the trading loop) and the
  `timescaledb.service`

nginx unit `wang-web.service` proxies `/api/v1/*` to `127.0.0.1:8080`
and serves the React bundle from `/opt/wang_trading/web/dist`.

### 17.3 Container Image

`Dockerfile.web` (multi-stage):

1. `node:20` builds the React bundle into `/dist`.
2. `python:3.12-slim` installs the Python stack and copies `/dist`
   into `/app/web/dist`.
3. Entrypoint runs `uvicorn`.

`docker-compose.yaml` adds a `bff` and `web` service for local
end-to-end testing.

## 18. Build & CI/CD

### 18.1 Build Pipeline (GitHub Actions)

1. `lint` (ruff)
2. `type-check` (mypy on `src/web/`)
3. `unit` (pytest on `tests/web/`)
4. `contract` (verify Pydantic DTOs match `docs/api_contracts_v2.md`
   examples)
5. `e2e` (boot uvicorn, hit the wired endpoints, assert envelope shape)
6. `frontend-lint`, `frontend-type-check`, `frontend-unit`,
   `frontend-build`
7. `docker-build` (only on main and tags)

### 18.2 Branch Protection

- All seven checks required to merge to `master`.
- Squash merges only.
- Tag releases as `web/vX.Y.Z` (independent of the trading stack
  release cadence).

### 18.3 Deploy

Phase 1: manual `make web-deploy` runs `rsync` + `systemctl restart`.
Phase 2: GitHub Actions deploys to a staging operator host on every
`master` push and to production on tag.

## 19. Testing Strategy

Pyramid (target ratio):

```
       e2e (UI + BFF)             10%
    contract / DTO tests          20%
  unit tests (services, deps)     70%
```

### 19.1 Unit (`tests/web/unit/`)

- One file per service, per route module.
- Services tested with fake repositories (no DB, no MLflow).
- Use `pytest-asyncio` for async services.

### 19.2 Contract (`tests/web/contract/`)

- For each example payload in [docs/api_contracts_v2.md](api_contracts_v2.md),
  assert the corresponding Pydantic DTO accepts it without modification
  and `model_dump(mode="json")` round-trips back to a superset.
- Catches doc/code drift early.

### 19.3 Integration (`tests/web/integration/`)

- Boot the FastAPI app with `TestClient` and a real (test) Timescale
  via `docker-compose -f docker-compose.test.yaml`.
- Hit each route, assert envelope shape, assert no secret-shaped keys
  appear in responses.

### 19.4 End-to-End (`tests/web/e2e/`)

- Playwright. Loads the React dev server hitting a TestClient-backed
  BFF.
- Critical paths: load Command Center; filter Trade Ideas to BUY; open
  drawer; verify SHAP renders; verify diff rail populates on second
  refresh; verify Cmd+K opens; verify Replay reconstructs.

### 19.5 Performance (`tests/web/perf/`)

- Locust scenarios for the five hot endpoints.
- Assert p95 stays within budget (§11) under 10 concurrent users.

## 20. Failure Modes & Degradation

| Upstream | Fail mode | BFF behaviour |
|---|---|---|
| TimescaleDB down | psycopg pool exhausted | `503 DB_UNAVAILABLE`; `/healthz` returns ok=false |
| MLflow down | timeout | `503 MODEL_UNAVAILABLE` on model endpoints; `/trade-ideas` returns 200 with `STALE_MODEL` warning and `meta_probability=null` |
| Broker disconnected | heartbeat fails | broker pill turns red; mutation endpoints `503 BROKER_UNAVAILABLE` |
| Feature store stale | freshness > threshold | `200` with `warnings` populated; Symbol Detail data tabs render with stale badges |
| Trading process down | tmpfs cache stale | `/trade-ideas` falls back to DB scan; warning in envelope |
| Factor model stale | `factor_risk` returns old fit | `200` with `STALE_FACTOR_MODEL` error code (not 4xx) |
| MLflow alias missing | no production model | `/model/status` returns `production_model_exists=false`; `/trade-ideas` action is `MODEL_REQUIRED` |
| Audit chain broken | signature mismatch | Audit page shows broken-from id; Replay disables forward past break |

In every case the operator sees a clear pill + banner. The system never
fakes data to hide an outage.

## 21. Migration From The Existing Local UI

The current `src.ui.trade_ideas_app` (stdlib HTTP server, single HTML
page) keeps working through the migration. The new BFF can run on a
different port (8080 vs 8765) so both serve simultaneously.

Migration steps:

1. Phase 1.0: BFF mounts only `/healthz` and `/api/v1/trade-ideas` and
   `/api/v1/overview`. React shell loads on `:5173` (dev) and hits the
   BFF. Stdlib UI continues on `:8765`.
2. Phase 1.5: React shell ships behind nginx. Stdlib UI is still
   available behind nginx on `:8766` for fallback.
3. Phase 2.x: Stdlib UI marked deprecated; redirect to React.
4. Phase 3.x: Stdlib UI removed. The two files at
   `src/ui/trade_ideas_app.py` and `src/ui/trade_ideas.py` shrink: the
   `_app.py` is deleted, `trade_ideas.py` survives as a pure report
   producer reused by the BFF.

No flag day. No big-bang rewrite. The frontend can move page-by-page.

## Part III — Development Plan

## 22. Sprint Plan

12-week plan, 2-week sprints. Each sprint ends with a demoable build.

### Sprint 1 (W1–W2): Foundations

- Decide on `fastapi`, `uvicorn`, `uvloop`, `psycopg[binary,pool]`,
  `httpx`, `python-jose` (Phase 2 only) — add to `requirements.txt`.
- Stand up `src/web/app.py` with lifespan, CORS, request-id middleware,
  exception handlers, logging config, Prometheus instrumentation.
- Stand up nginx + docker-compose dev profile.
- Wire `/healthz`, `/api/v1/overview`, `/api/v1/trade-ideas` (already
  scaffolded — just complete them).
- Write the `envelope`, `dtos`, `deps`, `errors` tests.
- Bootstrap the React app (`apps/web` directory): Vite, React 18,
  TypeScript, TanStack Query, design tokens from
  [docs/web_app_design_v2.md](web_app_design_v2.md) §8.

**Demo:** Load `http://localhost:5173`, see a working status bar +
Trade Ideas table powered by real BFF data.

### Sprint 2 (W3–W4): Command Center

- Implement `Diff Rail` SSE + `diff_service` (publisher).
- Implement `Edge-vs-size scatter`, `Action counts`, `Stage latency`,
  `Recent events` panels.
- Implement Cmd+K command palette skeleton.
- Wire `/api/v1/stream/ops` and `/api/v1/stream/diff`.
- Add the BFF Grafana dashboard (one row of panels).

**Demo:** Live Command Center matching [docs/wireframes/command_center.html](wireframes/command_center.html).

### Sprint 3 (W5–W6): Trade Ideas Drawer + Explainability

- Implement `TradeIdeaDetail` end-to-end:
  - `signal_metadata` from `signal_battery` (per-family typed payloads)
  - `shap_service` (per-row SHAP)
  - `sizing` waterfall with `constraints_applied`
  - `microstructure_service`, `structural_breaks` snapshot
  - `cost_forecast_service` (rolling TCA)
- Build the drawer with the 12 tabs from §13.4.
- NL explanation v1 (deterministic templated paragraph).

**Demo:** Click any idea, see SHAP, signals, sizing, microstructure,
expected cost, NL paragraph.

### Sprint 4 (W7–W8): Symbol Detail + Signals + Model & Features

- Symbol Detail page: price chart, news overlay, calibration history,
  microstructure, feature drift snapshot, per-symbol track record.
- Signals page with full per-family metadata viewer.
- Model & Features page with calibration plot, per-row SHAP browser,
  regime panel, RL shadow comparison, retrain timeline.

**Demo:** Navigate from a Trade Ideas row to Symbol Detail to Model &
Features and back via Cmd+K.

### Sprint 5 (W9–W10): Portfolio + Execution + Monitoring + Audit

- Portfolio & Risk including pre-trade simulator.
- Execution & TCA with v2 cost columns.
- Monitoring & Alerts with freshness heatmap + full Prometheus surface.
- Audit & Compliance with event timeline.

**Demo:** Full read-only operator cockpit. Stdlib UI marked deprecated.

### Sprint 6 (W11–W12): Replay, Scenarios, Track Record, Preflight

- Replay / Time Travel page.
- Scenarios & Stress Test page (read-only scenario engine).
- Track Record page.
- Preflight & Go-Live page (read-only).
- Backtests & Research with head-to-head compare.

**Demo:** All 15 pages live. Phase 1–3 of the design doc complete.

### Sprint 7+ (W13+): Phase 4 — Controlled Mutations

Out of scope for this 12-week plan. Phase 4 will need:

- OIDC sign-in
- Idempotency table
- Audit hook on every mutation
- Typed confirmation modals
- `live_operator` role flow (halt, verify-flat, flatten)
- `quant_admin` flow (retrain request, promote request)
- Calibration refit endpoint

## 23. Resourcing & Risk

### 23.1 Team Shape (lean)

- 1 backend engineer for the BFF and services.
- 1 frontend engineer for the React app and design system.
- 0.25 SRE / DevOps for deployment, monitoring, nginx, TLS.
- 0.25 quant engineer for SHAP, regime fit, track record service
  semantics — *not* full-time but as a code reviewer/oracle.

With a leaner team (1 fullstack), drop one full sprint of scope and
ship the same 12-week plan as 16 weeks.

### 23.2 Top Risks

1. **SHAP for ensemble models can be slow.** Mitigation: cache per
   (model_version, feature_hash) for 24 h; serve cached results.
2. **TimescaleDB schema drift between trading process and BFF.**
   Mitigation: never query raw tables from `src.web.*`; always go
   through `src.data_engine.storage`.
3. **Model registry calls in the hot path.** Mitigation: eager load at
   lifespan startup, event-driven refresh, never per-request.
4. **SSE in nginx.** Mitigation: explicit `proxy_buffering off` and
   `proxy_read_timeout 3600s` on the SSE locations.
5. **Pydantic v2 + FastAPI version drift.** Mitigation: pin both in
   `requirements.txt` and renovate on a schedule, not opportunistically.
6. **Auth scope creep.** Mitigation: stay VPN-only in Phase 1. Don't
   ship OIDC until the read-only experience is solid.

## 24. Decisions (resolved 2026-05-15)

1. **React app lives in `apps/web/`** in this repository. Same-repo PRs
   for full-stack changes; shared CI; design docs are already here.
2. **Production deploys via systemd; dev and staging via Docker
   Compose.** Matches the `wang-trading.service` pattern. nginx is also
   systemd-managed in prod.
3. **Roles config: `config/roles.example.yaml` committed,
   `config/roles.yaml` in `.gitignore`** and populated per-host. Schema
   documented in the example file.
4. **OpenTelemetry instrumented from day one, disabled by default.**
   Auto-instrumentation behind the `OTEL_EXPORTER_OTLP_ENDPOINT` env
   var. Zero overhead when unset; one-flip enable when needed.
5. **Trade idea cache: tmpfs primary, sync-regenerate fallback.**
   - Trading process writes `TradeIdeasResponse.model_dump_json()` to
     `/run/wang/trade_ideas.json` after each cycle (atomic
     `rename(2)`). File includes an `as_of` field that the BFF reads
     to compute staleness.
   - BFF reads the file on every request; if missing, older than
     `cache.trade_ideas_max_age_seconds` (default 90 s), or
     unparsable, falls back to `generate_trade_idea_report_sync` with
     a 30 s in-process LRU.
   - The post-cycle hook in the trading process is a Sprint-1 deliverable
     coordinated with `src.execution.daily_ops` (or wherever the cycle
     ends).
6. **Frontend uses plain React + design tokens.** No component
   library. Density requirements and the v2 wireframe show plain CSS
   is enough; no Mantine/Chakra/Tailwind dependency to govern.

## 24a. Follow-On Decisions

Locked from the §24 choices:

- **Python**: pin to the version the trading stack already uses
  (currently CPython 3.14 per the bytecode in `__pycache__`). The BFF
  must boot against the same interpreter.
- **Node**: 20 LTS for the frontend build.
- **Package manager**: `pnpm` for `apps/web/` (deterministic, fast,
  cheap disk).
- **TypeScript**: strict mode on.
- **React**: 18.x in Phase 1 (stable, broad library compatibility).
  Reassess React 19 at the start of Phase 2.
- **TanStack Query**: v5.
- **Vite**: 5.x.
- **Tmpfs path**: `/run/wang/` (tmpfs on Linux). Dev environment uses
  `${XDG_RUNTIME_DIR}/wang/` on macOS.

## 25. Acceptance Criteria for Phase 1

The BFF is Phase-1 production-ready when:

- All Phase-1 endpoints in §22 sprints 1–3 are wired and return real
  data with the envelope, freshness, and model-version metadata.
- p95 latencies meet §11 budgets under load.
- No secret-shaped keys appear in any response (verified by an
  integration test that fuzzes the response surface).
- Lifespan startup fails loudly on misconfigured DB / model registry /
  feature store.
- The React app loads, the Command Center renders matching the
  wireframe, and every page in §22 sprints 1–3 is reachable.
- Prometheus scrapes `/metrics` and Grafana shows the BFF dashboard.
- All unit, contract, and integration tests pass in CI.
- The existing local Trade Ideas UI continues to work unchanged on its
  port.

## 26. Design Summary

The BFF is a thin, opinionated read layer in front of a serious quant
stack. It never reimplements logic; it projects, normalises, redacts,
and tags. It boots eagerly, caches at three layers, fans out to React
via REST + SSE, and gates mutations behind role + idempotency +
confirmation. It runs on the operator host beside the trading process
in Phase 1 and can be split into its own process when load demands it.

The 12-week plan ships Phase 1–3 of the v2 product design and leaves
Phase 4 mutation flows for a follow-on. The risk and the answer are
the same: do not duplicate the quant stack inside the BFF. Wrap, don't
rewrite.
