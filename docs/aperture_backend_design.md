# Aperture — Backend Design (Canonical)

**The one design of record for the Aperture backend: a read-only FastAPI BFF over the `wang_trading` engine, serving the web + iOS clients through a single, honest API contract.**

Version 1.0 · 2026-06-19 · Owner: YW · Status: **canonical**

---

## 0. What this document is

This is the **single design of record** for the Aperture API + services layer. It is grounded in the **now-built frontend** (web + iOS), which pins down the exact endpoints, inputs, and outputs the backend must serve, and it is **honest about data** per [data_readiness.md](data_readiness.md): the BFF serves only what the engine actually produces, and every absent field is `null` with a *named* unlock condition — never a fabricated number.

It was produced by an understand → design → adversarial-review pipeline against the live tree, and it reconciles two earlier docs.

### Relationship to the earlier docs

| Doc | Role now |
|---|---|
| **This doc** | **Canonical backend design** — the 8-endpoint v1 contract, services, reliability, security, observability, deployment; right-sized for one operator |
| [backend_design.md](backend_design.md) | **Depth catalog** — its architecture rationale (topology, DI, caching, SSE, error taxonomy) is solid and folded in here; kept for the long-form reasoning |
| [api_contracts_v2.md](api_contracts_v2.md) | **Superseded for the *current* contract** — its ~40 speculative endpoints are re-anchored here to the 8 the frontend actually calls; kept as a field-level wishlist for future waves |
| [aperture_v1_design.md](aperture_v1_design.md) | The frontend counterpart — the client contract this backend serves |
| [data_readiness.md](data_readiness.md) | Ground truth for what is LIVE vs COMING, per wave |

### The shape of v1 (locked)

- **Read-only.** The engine owns *all* writes; the BFF projects, normalizes, redacts, tags, and degrades — and never fabricates. No mutations in v1 (the engine is PAPER).
- **8 endpoints** are the entire v1 API surface. The 8 COMING screens need **no endpoint** — the client renders their locked `ComingState` from `readiness.ts`.
- **One frozen `ApiEnvelope<T>`** identical across Python / TypeScript / iOS, carrying `request_id`.
- **Right-sized.** Single-operator / single-tenant: "reliable + scalable" means rock-solid graceful degradation and clean service boundaries — **not** web-scale microservices.

---

## Table of contents

1. [Foundations, the read-only invariant & API-style decision](#1-foundations-the-read-only-invariant--api-style-decision)
2. [API surface — the 8 v1 endpoints](#2-api-surface--the-8-v1-endpoints)
3. [The response envelope, errors, casing & versioning](#3-the-response-envelope-errors-casing--versioning)
4. [Service architecture & data access](#4-service-architecture--data-access)
5. [Reliability & graceful degradation](#5-reliability--graceful-degradation)
6. [Real-time & the "what changed" model](#6-real-time--the-what-changed-model)
7. [Security & authorization](#7-security--authorization)
8. [Observability](#8-observability)
9. [Scalability (right-sized) & deployment / DR](#9-scalability-right-sized--deployment--dr)
10. [Build roadmap & locked decisions](#10-build-roadmap--locked-decisions)


---

## 0.1 Precision notes (locked in review)

This design was adversarially reviewed against the live tree. Six precise semantics **control** wherever a section below reads loosely:

1. **The error-code enum is closed; this design *extends* it (it does not "keep it verbatim").** Today `ErrorCode` (`src/web/errors.py`) is a closed set — `BAD_REQUEST`, `VALIDATION_FAILED`, `NOT_FOUND`, `UNAUTHENTICATED`, `FORBIDDEN`, `CONFLICT`, `STALE_FACTOR_MODEL`, `STALE_MODEL`, `DB_UNAVAILABLE`, … This doc **adds** two snapshot-spine codes: **`STALE_SNAPSHOT`** (snapshot past the 90 s threshold → HTTP 200 + `errors[]`) and **`SNAPSHOT_UNAVAILABLE`** (no snapshot to read → degrade to last-good, else 503). New codes are additive; clients switch on `code`, never `message`.

2. **`MODEL_REQUIRED` is an idea `action` value, not an error code.** When no production model is loaded, each idea's `action` field is the string `"MODEL_REQUIRED"` and `metaProbability`/`calibratedProbability` are `null`; at the envelope level `model_version` stays `null` (which drives the client's global *Model* pill). There is no `errors[].code = MODEL_REQUIRED`.

3. **`exclude_none` is scoped to the inner `data` payload, not the whole envelope.** The honesty mechanism (absent → omitted → `null`) applies to **`data` fields**. Envelope metadata — `as_of`, `source`, `staleness_seconds`, `source_freshness`, `request_id` — is **always populated** and must never be dropped (the clients bind `staleness_seconds`/`source_freshness` as non-null; dropping them would break the freshness/trust signal). Implementation: always compute `staleness_seconds`, and apply `exclude_none=True` only when serializing the `data` DTO — not the envelope. *(Today `envelope.py:91` dumps the whole envelope with `exclude_none`; this is the one change that makes the honesty mechanism safe.)*

4. **`data` is nullable: `data: T | null`.** On a screen-level hard failure the BFF returns `data: null` + `errors[]`; on a *partial* failure it returns a structurally-valid `data` object with the absent inner fields `null`. The client `ApiEnvelope.data` is therefore `T | null` (a one-line widening of `envelope.ts`/the Swift model), and screens already tolerate per-field nulls via the honesty system.

5. **SSE is served by a dedicated single-process ASGI service on its own port** — *not* by "pinning a uvicorn worker." `Last-Event-ID` replay from an in-process ring buffer is only coherent in one process; the read-API runs ≥2 workers and does not stream. See §6.

6. **The BFF connects to TimescaleDB with a read-only role (`GRANT SELECT` only).** This turns the §1.2 "BFF never writes a row" invariant from a convention into a **database privilege boundary**. See §4.

> A handful of section examples use different illustrative values for the envelope `source` field (`"trade_ideas.json"`, `"trade_ideas_service"`, …). The convention is: `source` names the **producing service** (e.g. `overview_service`, `trade_ideas_service`), matching the code (`overview.py`, `trade_ideas.py`).

---

## 1. Foundations, the Read-Only Invariant & API-Style Decision

### 1.1 Product framing — a single-operator quant cockpit

Aperture is a **decision-support cockpit for one operator**, not a multi-tenant SaaS. There is exactly one person watching this surface, one trading engine behind it, and one host. That single fact governs every backend choice in this catalog: "reliable and scalable" here means **rock-solid graceful degradation and clean service boundaries**, *not* web-scale fan-out. We optimize for an operator who needs to trust what they see at a glance and reach for depth on demand — not for throughput.

The product is **read-first**. The web and iOS clients are identical and both *observe* the engine; they do not drive it. `deriveTrust` in the frontend hardcodes `mode='PAPER'`, and the engine is read-only/paper today — so the v1 backend advertises **no mutations** at all. No "place order," no "clear halt," no "refit calibration." The cockpit shows decisions the engine has already made and the evidence behind them. That posture is not a temporary limitation to apologize for; it is the design.

### 1.2 The one invariant — the engine owns all writes; the BFF never fabricates

Everything in this document serves a single load-bearing rule, and it is **already true in code**:

> **The trading engine owns ALL writes. The BFF is read-only: it projects, normalizes, redacts, tags, and degrades — but it never reruns business logic and never invents a number. When data is absent, it returns `null` plus a `warning`, never a placeholder-as-number.**

Concretely, the BFF must **never**:

- write a Parquet file or a database row,
- open a broker session,
- re-execute the signal/sizing pipeline to "fill in" a value,
- synthesize, interpolate, or default a field the engine did not produce.

The BFF is a **thin projection layer** over two upstream sources. Its only verbs are *shape* (DTO mapping, casing), *redact* (secret defense-in-depth), *tag* (freshness, staleness, regime, model version), and *degrade* (return `200` with non-empty `errors[]` rather than crash). When something is missing, the honest answer is `null` + a warning — and that is a *feature* the frontend is built to render, not an error to paper over.

This invariant is what makes the whole system trustworthy with one operator and no second pair of eyes: **if a number renders, the engine produced it.** There is no path by which the read layer can manufacture a plausible-but-fake figure.

### 1.3 The real data spine — two sources, and only two

The entire LIVE surface is backed by **exactly two upstream sources**. Memorize them, because any endpoint not traceable to one of these is either an aggregate of a snapshot or **blocked on net-new engine persistence — not BFF work.**

| Source | What it is | Producer | Consumed by |
|---|---|---|---|
| **`bars` hypertable** | TimescaleDB OHLCV history (the real bar microstructure — the honesty centerpiece of the symbol view) | engine ingestion → `src/data_engine/storage/database.py` | `/markets`, `/symbols/{symbol}` |
| **`trade_ideas.json`** | tmpfs snapshot of the latest cycle's full trade-idea report | `src/execution/trade_idea_publisher.py` | `/overview`, `/trade-ideas`, `/signals/*` (snapshot joins) |

The snapshot writer is deliberately crash-safe: it serializes to a `.tmp` sibling and `os.replace`-s it into place, so a reader **never observes a partial document**. Its on-disk shape is frozen as `{ "as_of": "<ISO-8601 UTC>", "report": <TradeIdeaReport.to_dict()> }`, and the publisher plus the BFF's `TmpfsTradeIdeasCache` are the *only* writer/reader of that format (`trade_idea_publisher.py:9-19`). Because this snapshot is **1 of 2 real sources**, the engine→BFF boundary across it is the single highest-leverage failure point in the system — §2 mandates a `schema_version` stamp and a cross-boundary contract test there.

Two consequences fall directly out of this spine:

1. **The bars table is already persisted but has no route yet.** `/markets` and `/symbols/{symbol}` are *net-new BFF routes over data that already exists* — the cheapest real wins available. They are LIVE the moment the route is mounted.
2. **The snapshot is overwritten every cycle.** It is a *current-state* document, not a history. Anything requiring a time series of *decisions* (track record, win rates, call history) is **blocked on net-new append-only engine persistence** and must stay in a COMING state — the BFF cannot reconstruct history the engine never kept.

### 1.4 The data-honesty contract — null, never placeholder

The frontend implements a **7-state honesty system** keyed to `readiness.ts`: every field is in exactly one of these states, and the backend's job is to put it in the *truthful* one — never to upgrade an absent field to a fabricated number.

| State | Meaning | Backend behavior |
|---|---|---|
| **Live** | engine produced a real value | serve the value |
| **Empty** | producer exists, genuinely zero rows this cycle | serve `[]`/empty + `as_of` (e.g. `/model` `metaProbHist` with no rows — Empty, *not* Coming) |
| **Stale** | value exists but past freshness threshold | serve value + `STALE_*` warning (HTTP 200) |
| **Coming** | no producer yet; blocked on a named engine gate | omit field → arrives `null`; client renders locked `ComingState` |
| **Model-required** | needs a registered production model | `action=MODEL_REQUIRED`, probabilities `null` |
| **Data-unavailable** | source unreachable right now | `null` + warning, degrade-but-ship |
| **Error** | request failed | enveloped `errors[]` + `request_id` |

The serialization mechanism that enforces this is one line: the envelope is dumped with **`exclude_none=True`** (`envelope.py:91`). Absent fields simply do not appear, arriving as `null`/missing on the client, which routes them straight into the honesty system. **The rule the backend must never break: do not synthesize a value for a field the engine does not produce.** A `null` that the client renders as a clean "Coming" panel is infinitely more valuable to a single operator betting real capital than a plausible fake.

The seven COMING fields hardwired to `null` in `trade_ideas_service.py:358-367` (`regime`, `regime_fit_score`, `sizing_constraints_applied`, `expected_cost_bps`, `top_shap_feature`, `track_record_win_rate`, `track_record_n`) are the canonical example: each is honestly absent and gated behind a *named* engine-persistence task, never invented.

### 1.5 API-style decision — REST/FastAPI + a frozen JSON envelope, SSE for streams

**Decision: REST over FastAPI, with `ApiEnvelope<T>` as the primary contract. SSE for the few real-time streams. No GraphQL. No gRPC on the public API.**

#### Primary contract — REST + the frozen envelope

Every response is an `ApiEnvelope<T>` (`envelope.py:50-60`), and this shape is **already identical across all three layers** — `src/web/envelope.py` ↔ `apps/aperture-web/src/data/envelope.ts` ↔ iOS:

```jsonc
{
  "data":              <T> | null,
  "as_of":             "2026-06-19T14:00:00Z",
  "source":            "trade_ideas.json",
  "staleness_seconds": 12.4,
  "source_freshness":  { "bars": 3.1, "snapshot": 12.4 },
  "model_version":     null,
  "regime":            null,
  "warnings":          ["no persisted portfolio"],
  "errors":            [{ "code": "STALE_SNAPSHOT", "message": "...", "field": null }],
  "request_id":        "req_9f1c..."
}
```

Clients **switch on `errors[].code`, never on `message`** — `STALE_*` codes are intentionally **HTTP 200 with a non-empty `errors[]`** (degrade-but-ship). The rationale for REST + this envelope is decisive:

- **The engine is Python.** FastAPI + Pydantic is the path of least impedance from the existing `src.*` codebase to an HTTP surface; services already return DTOs with zero FastAPI imports.
- **The envelope *is* the value.** The freshness/staleness/source/regime/model-version/warnings/errors metadata — the honesty contract — is the product. It rides on every response uniformly. That is something to *freeze*, not to abstract behind a query language.
- **One operator, low cardinality.** The whole v1 surface is **8 endpoints**. There is no combinatorial query space to justify a query layer; a fixed, well-shaped envelope per route is simpler to reason about, cache, and test.
- **The built frontend already binds it.** The web and iOS clients consume exactly this contract today; the frontend `source:` strings *are* the authoritative endpoint paths. Rewriting the contract to match the frontend is cheaper and more honest than rewriting two clients.

#### SSE for the few real-time streams

For the handful of genuinely real-time needs (live tape / cycle-tick push), use **Server-Sent Events over WebSockets**: it is unidirectional (matching read-first), rides plain HTTP, auto-reconnects, and needs no bidirectional protocol. (Note the honest caveat carried into §6: with 2 workers, per-worker replay buffers mean a reconnect can land on a worker that never saw the missed events — pin SSE to one worker or document the limitation; do not *claim* reliable replay we don't have.)

#### Explicitly rejected

- **GraphQL — rejected.** There is no client demanding flexible, ad-hoc queries; the surface is 8 fixed shapes for one operator. GraphQL would dissolve the envelope/honesty contract (the actual value) into a resolver graph, complicate per-field freshness/degradation semantics, and add a query planner, schema layer, and N+1 surface — all cost, no benefit at this cardinality.
- **gRPC on the public API — rejected.** Browsers don't speak it natively (gRPC-Web needs a proxy), it fights the human-readable JSON envelope, and the clients already bind REST. gRPC is the *right* tool for **exactly one future scenario**: an *internal* engine↔BFF process split over a Unix socket, **and only if profiling shows real GIL/event-loop contention** (§6). It is not a current build target. The "scalable" property we want — clean boundaries (zero FastAPI imports in services, DTO returns, DI singletons) — makes that future split *mechanical*; we earn it by design discipline now, not by building the transport preemptively.

---

## 2. API surface — the 8 v1 endpoints

This is **the** authoritative request/response contract for Aperture v1. The frontend (web + iOS, identical `source:` accessors) needs **exactly 8 endpoints** under `/api/v1`. Everything is wrapped in the frozen `ApiEnvelope<T>` (§3); responses serialize with `exclude_none=True`, so absent fields arrive `null`/missing and the client maps them to its honesty system — they are **never synthesized**.

Two rules govern every row below:

1. **The BFF projects two real sources only** — the `bars` hypertable (`src/data_engine/storage/database.py`) and the tmpfs `trade_ideas.json` snapshot (`src/execution/trade_idea_publisher.py`). A field not derivable from one of those is held `null` with a named unlock gate, not computed.
2. **Casing is camelCase on the wire** (Pydantic alias generator, single source — §3). This section names fields in camelCase; the Python DTOs use snake_case internally.

### Readiness tiers (used in the tables)

| Tier | Meaning |
|------|---------|
| **BUILT** | Router mounted, reads a real source, ships today (`overview`, `trade-ideas` per `app.py:203-209`). |
| **PARTIAL** | Mounted + real for some fields, others held `null`+warning (`/overview` portfolio block). |
| **PLANNED — net-new** | Route does not exist yet (`markets`, `symbols`, `signals`); data exists or is model-gated. Build on the named wave. |
| **STUB** | Exists, returns non-required placeholder; must not 500 the client (`/trade-ideas/{symbol}`). |

---

### 2.1 `GET /overview` — Home dashboard

| | |
|---|---|
| **Method / path** | `GET /api/v1/overview` |
| **Inputs** | none |
| **Caller** | Home / Overview screen (action ribbon + top-actionable list + latency strip) |
| **Tier / wave** | **BUILT (PARTIAL)** — Wave 1 |

**LIVE fields** (all from the snapshot): `actionCounts` (the six-bucket tally), `topActionable` (≤5 full TradeIdea rows, sorted by `|targetWeight|`), `stageLatencySeconds` (per-stage wall-clock summed from the snapshot).

**Held `null` / ComingState** (with a standing `warning` "no persisted portfolio"):

| Field | Unlock gate | Wave |
|---|---|---|
| `nav`, `dailyPnl`, `drawdown`, `grossExposure`, `netExposure`, `positionsCount` | net-new portfolio persistence (snapshot is overwritten each cycle — no position history) | 5 |
| `regime` | `RegimeDetector` has **zero runtime callers** — must be invoked + persisted | 6 |

```jsonc
// data:
{
  "actionCounts": { "BUY": 3, "SELL": 1, "WATCH": 8, "MODEL_REQUIRED": 2, "NO_DATA": 0, "ERROR": 0 },
  "topActionable": [ /* ≤5 TradeIdea (see §2.2) */ ],
  "stageLatencySeconds": { "ingest": 0.41, "features": 1.20, "signals": 0.33, "sizing": 0.08 },
  "nav": null, "dailyPnl": null, "drawdown": null,
  "grossExposure": null, "netExposure": null, "positionsCount": null,
  "regime": null
}
```

> `/overview` is the **reference implementation**: real where data exists, `null` + envelope `warning` where it doesn't, correct try/except degrade. Every other route mirrors its shape.

---

### 2.2 `GET /trade-ideas` — the table (highest-traffic surface)

| | |
|---|---|
| **Method / path** | `GET /api/v1/trade-ideas` |
| **Inputs (query)** | `symbols` (csv), `bar_limit=500`, `min_abs_weight=0.0025`, `allow_confidence_fallback=false` |
| **Caller** | Trade Ideas table + the row-hydrated detail drawer |
| **Tier / wave** | **BUILT** — Wave 1 |

Returns `TradeIdeasResponse { ideas: TradeIdea[], totals }`. ~20 LIVE fields per idea from the snapshot: `symbol, action, targetWeight, targetNotional, estimatedQuantity, latestPrice, latestBarAt, barType, barsLoaded, featureRows, signalCount, topSignalFamily, topSignalSide, topSignalConfidence, avgSignalConfidence, betSize, strategy, reason, stageLatencySeconds, errors[]`, plus the nested `signals[]`.

**Model-gated pair** — `metaProbability`, `calibratedProbability`. The split is **real** on the model path (`bootstrap.py:309` calls `predict_proba(return_raw=True)`); it collapses to equal only in paper-mode `ConfidenceMetaPipeline` fallback (`bootstrap.py:379`). Both are `null` until a registered MLflow **production model** is loaded → action surfaces as `MODEL_REQUIRED`. **Wave 2.**

**7 fields hardwired `null`/`[]`** — verified at `trade_ideas_service.py:358-367`. These are not "TODO maybe"; they are deliberately held because no producer exists:

| Field | Held as | Unlock gate (named in code) | Wave |
|---|---|---|---|
| `regime` | `null` | `RegimeDetector` wiring (zero callers) | 6 |
| `regimeFitScore` | `null` | SignalsService attribution (no producer) | 6 |
| `sizingConstraintsApplied` | `[]` | surface cascade `constraints_applied` through `ui/trade_ideas.py` | 4 |
| `expectedCostBps` | `null` | build `CostForecastService` | 5 |
| `topShapFeature` | `null` | call+persist `shap_importance` (`FeatureStore.save_features` has zero callers) | 4 |
| `trackRecordWinRate` | `null` | append-only call-history store (snapshot overwrites each cycle) | 5 |
| `trackRecordN` | `null` | same | 5 |

```jsonc
// data.ideas[0]:
{
  "symbol": "AAPL", "action": "BUY", "targetWeight": 0.034, "targetNotional": 34000,
  "estimatedQuantity": 180, "latestPrice": 188.4, "barType": "dollar", "barsLoaded": 500,
  "signalCount": 4, "topSignalFamily": "momentum", "topSignalConfidence": 0.71,
  "betSize": 0.34, "strategy": "trend", "reason": "...",
  "metaProbability": null, "calibratedProbability": null,   // model-gated → Wave 2
  "regime": null, "regimeFitScore": null,                   // 7 hardwired-null fields
  "sizingConstraintsApplied": [], "expectedCostBps": null,
  "topShapFeature": null, "trackRecordWinRate": null, "trackRecordN": null,
  "signals": [ { "family": "...", "side": "LONG", "confidence": 0.71 } ],
  "stageLatencySeconds": { "...": 0.0 }, "errors": []
}
```

> **Fix on adoption:** the route currently does `TradeIdeasService()` per-request (bypassing `deps.py` singletons → defeats the 30s regenerate-debounce) and raises raw `HTTPException(500, detail=str(exc))` at `trade_ideas.py:38` (leaks exception text, bypasses the envelope). Route through DI; replace with a typed `ApiException` → enveloped `errors[]`. Wire the computed-but-dropped `_last_staleness_seconds` into the envelope + a `STALE_*` warning.

---

### 2.3 `GET /markets` — Markets browser

| | |
|---|---|
| **Method / path** | `GET /api/v1/markets` |
| **Inputs** | none (filter/sort are client-side) |
| **Caller** | Markets list/grid |
| **Tier / wave** | **PLANNED — net-new** (data exists; router not mounted) — Wave 1 |

Returns `Sym[]`, joined: bar microstructure from the `bars` hypertable + `hasIdea` from the snapshot. Per row: `symbol, name, type, price, change{1d,1w,1m,ytd}, spark/line/candles (OHLCV), volume, hasIdea, bar (BarMicro)`.

**Held / dropped:**

- `marketCap` — **drop from the type entirely.** No producer; never serve.
- Requires a static `symbol → {name, assetClass}` ref map (bars carry only `symbol` + `bar_type`). Ship it with this route; unknown symbol → ticker-as-name + default tint.

---

### 2.4 `GET /symbols/{symbol}` — Symbol detail (the honesty centerpiece)

| | |
|---|---|
| **Method / path** | `GET /api/v1/symbols/{symbol}` |
| **Inputs (path)** | `symbol` |
| **Caller** | Symbol Detail screen |
| **Tier / wave** | **PLANNED — net-new** — Wave 1 (idea/model fields read Wave 2) |

Returns `SymbolDetail { sym: Sym, idea: TradeIdea | null }`. **LIVE:** candles, change strip, **real bar microstructure** (the deliberate honesty showcase — we show what the engine actually computed, not a synthetic OHLC), plus `action/reason/strategy/targetWeight` when an idea exists.

**Held / ComingState:** feature-factory features (GARCH/RSI/VPIN/Kyle-λ), per-symbol SHAP, `regimeFit`, expected cost, track-record. Each is `null` until its named gate (same gates as §2.2). Model-gated reads (idea probabilities) follow Wave 2.

---

### 2.5 `GET /signals/families` — Strategy families overview

| | |
|---|---|
| **Method / path** | `GET /api/v1/signals/families` |
| **Inputs** | none |
| **Caller** | Signals / Strategy Families screen |
| **Tier / wave** | **PLANNED — net-new** — Wave 1 (live counts Wave 2) |

Returns `Strategy[]` (all 10): `id, name, category, source, thesis, params, assetClasses` (static, LIVE) + an active-idea count from the snapshot. The client derives 4 active / 6 inactive from `FAMILY_READINESS` (6 of 10 families are dead-on-arrival — no caller supplies panel/pair/exchange/futures context).

**Never serve real** — delete from the type, do not synthesize: `sharpe, winRate, trades, contributionPct, pnlYtd, allocation, equityCurve, regimeFit, avgHoldBars`. These need per-strategy attribution + persisted P&L (Wave 5).

---

### 2.6 `GET /signals/family-{id}` — One family's ideas this cycle

| | |
|---|---|
| **Method / path** | `GET /api/v1/signals/family-{id}` |
| **Inputs (path)** | `id` |
| **Caller** | Family drill-down |
| **Tier / wave** | **PLANNED — net-new** — Wave 1 (ideas Wave 2) |

Returns `{ strategy: Strategy, ideas: TradeIdea[] }` — this family's ideas this cycle, LIVE from the snapshot. **Must be reachable for dormant families** (return `status` + `reason`, not a 404). Per-strategy Sharpe/win/PnL/equity → ComingState (Wave 5–6).

---

### 2.7 `GET /model` — Model card

| | |
|---|---|
| **Method / path** | `GET /api/v1/model` |
| **Inputs** | none |
| **Caller** | Model screen |
| **Tier / wave** | **PLANNED — net-new** — Wave 2 |

Returns `MODEL`: `version, trainedAt, lastRetrainHours, runId, type, cvScore, trainAcc, trainingEvents` (from `ModelRegistry.get_production_model()` + MLflow), `metaProbHist` (calibrated-prob buckets — render **Empty**, not Coming, when zero rows), `gates{cpcv,dsr,pbo}` (render neutral "not run" — the retrain gate is now real at `retrain_pipeline.py:_run_gate`, but verify DSR/CPCV/PBO are actually written to MLflow before showing verdicts), `retrainTimeline[]` (MLflow `search_runs`).

**No production model → the whole screen is Empty, `modelVersion=null`.** **Never serve real** — delete from the type: `auc, brier, ece, calibration[], featureImportance[], drift[], rlShadow`. (Drift is doubly dead: `DriftDetector.set_baseline` has zero callers → emits hardcoded 1.0.)

---

### 2.8 `GET /trade-ideas/{symbol}` — Detail (intentionally not required)

| | |
|---|---|
| **Method / path** | `GET /api/v1/trade-ideas/{symbol}` |
| **Tier / wave** | **STUB (501)** — verified `trade_ideas.py:51` — resolve later |

The detail drawer **hydrates from the list row** (§2.2) — no detail round-trip in v1. This endpoint stays a non-required 501. The one hard requirement: it **must not 500 the client**, and the client must resolve detail from the list, not depend on this path. Wire it only when `TradeIdeaDetail` (signals + SHAP + sizing waterfall + microstructure + cost + track-record) actually has producers.

---

### 2.9 The 8 COMING screens need NO endpoint

**Portfolio, Execution, Backtests, Scenarios, Track Record, Monitoring, Preflight, Replay** require **zero v1 endpoints.** The client renders locked `ComingState` panels entirely from `readiness.ts`. Do **not** mount routes for them in v1 — there is no source to back them (no position history, no execution writes, no backtest-run persistence, no audit rows, and the pipeline `MetricsCollector` uses a disjoint `CollectorRegistry` at `metrics.py:44` that nothing scrapes). Mounting a route here would force exactly the synthesize-a-number anti-pattern this design forbids.

> Note: `scenarios`, `replay`, and `preflight` routers *are* currently mounted (`app.py:203-209`) as MOCK/STUB. They are not part of the 8-endpoint v1 contract; treat them as non-shipping scaffolds (and `/preflight` is the cheapest future real win — it can wrap `src.execution.preflight` + infra_probe with no persistence).

---

### 2.10 Build order — gated to readiness waves

| Wave | Endpoints | Why it unlocks |
|------|-----------|----------------|
| **Wave 1** | #1 `/overview`, #2 `/trade-ideas` (**done**); #3 `/markets`, #4 `/symbols/{symbol}` (**net-new on `bars` + snapshot — data exists, just build the route**) | Both real sources already persist. |
| **Wave 2** | #7 `/model`, #5 `/signals/families`, #6 `/signals/family-{id}` — **model-gated reads** | Need a registered MLflow **production model**; absent one → `action=MODEL_REQUIRED` + null probabilities. |
| **Later** | #8 `/trade-ideas/{symbol}` | Only when `TradeIdeaDetail` has producers. |
| **Per-field, not per-route** | the 7 hardwired-null fields (§2.2), portfolio block (§2.1), regime | Each stays `null`/ComingState until **its named engine-persistence gate** lands (Wave 4–6). |

The invariant throughout: **a field ships LIVE only when an engine source produces it; otherwise it is `null` + a typed warning. Never a placeholder-as-number.**

---

## 3. The response envelope, errors, casing & versioning

Every BFF response — without exception, including errors, stale data, and empty results — is wrapped in `ApiEnvelope<T>`. The envelope is the trust contract: the frontend's persistent trust state binds to its metadata, not to the payload. This is the one part of the system that is already *fully consistent* across all three layers, so the design posture here is **freeze, don't redesign** — pin it with tests and close the two open seams (casing of the inner DTO; versioning of the tmpfs payload).

### 3.1 The frozen `ApiEnvelope<T>`

The envelope is **identical** across `src/web/envelope.py` (`ApiEnvelope[T]`, lines 50–60), `apps/aperture-web/src/data/envelope.ts` (`ApiEnvelope<T>`, lines 9–20), and the iOS client. `request_id` is present in all three (it was just added — `envelope.py:59` `Field(default_factory=_new_request_id)`, `envelope.ts:19`, and consumed by `deriveTrust` at `envelope.ts:68`).

```jsonc
// The frozen wire shape. Inner `data` is the only endpoint-specific part.
{
  "data": T | null,                 // the DTO; null on hard error / not-yet-available
  "as_of": "2026-06-19T14:03:11Z",  // ISO-8601 UTC — when the underlying snapshot was produced
  "source": "tmpfs:trade_ideas",    // which of the two real spines (or "bff") served this
  "staleness_seconds": 4.0,         // age of `as_of` vs now; UI flips to `stale` past threshold
  "source_freshness": {             // per-input freshness for the freshness SLO panel
    "bars": 2.1, "snapshot": 4.0
  },
  "model_version": "meta_v1.7.2",   // null ⟹ no production model ⟹ UI shows MODEL_REQUIRED
  "regime": null,                   // null today — RegimeDetector has zero runtime callers
  "warnings": ["no persisted portfolio"],  // human-readable, non-blocking degradation notes
  "errors": [],                     // structured; see §3.2. Non-empty + HTTP 200 = degraded-but-shipped
  "request_id": "req_9f3a1c20bb7e44d1"     // copyable ref; logged under the same key server-side
}
```

| Field | Type | Meaning | Today |
|---|---|---|---|
| `data` | `T \| null` | The DTO. `null` when absent or hard-errored. | LIVE |
| `as_of` | ISO-8601 UTC | Producer timestamp of the underlying snapshot — **not** response time. | LIVE |
| `source` | string | Provenance tag (`tmpfs:trade_ideas`, `bars`, `bff`). | LIVE |
| `staleness_seconds` | float? | Age of `as_of`. Drives the UI `stale` state. | **Computed but not wired** — see §3.5 |
| `source_freshness` | `{str: float}`? | Per-input freshness for the freshness SLO. | PARTIAL |
| `model_version` | string? | Production model id; `null` gates all model-derived fields. | LIVE |
| `regime` | `RegimeSnapshot?` | `{label, probabilities{4}, as_of}`. | COMING (no runtime caller) |
| `warnings[]` | string[] | Non-blocking degradation notes. | LIVE |
| `errors[]` | `EnvelopeError[]` | Structured, code-switched. See §3.2. | LIVE |
| `request_id` | string | Per-response trace ref. | LIVE |

**Serialization note (load-bearing):** `envelope()` returns `model_dump(mode="json", exclude_none=True)` (`envelope.py:91`). `exclude_none=True` is the mechanism behind the frontend's 7-state honesty system — an absent field arrives **missing**, never as a fabricated `0`. Do not relax this to `exclude_none=False`; doing so would silently turn "not yet available" into a hard zero on the wire.

### 3.2 Error taxonomy — switch on `code`, never `message`

Errors are structured, not prose. `EnvelopeError = {code, message, field?}` (`envelope.py:44–47` `ApiError`; `envelope.ts:22–26`). The frontend switches on the **stable `code`**; `message` is human-facing telemetry only and may change freely.

The taxonomy already exists and is production-grade (`errors.py`): a closed `ErrorCode` enum (`errors.py:33–49`), one typed `ApiException` subclass per code, a status map (`_HTTP_STATUS`, `errors.py:53–68`), and registered handlers. **Keep it verbatim.**

| `ErrorCode` | HTTP | Semantics |
|---|---|---|
| `BAD_REQUEST` | 400 | Malformed request. |
| `VALIDATION_FAILED` | 422 | Field-level — handler surfaces `field` path but **never the offending value** (`errors.py:207–232`). |
| `NOT_FOUND` | 404 | Unknown symbol / resource. |
| `UNAUTHENTICATED` / `FORBIDDEN` | 401 / 403 | Reserved (Phase-1 has no app-layer authz yet). |
| `CONFLICT` | 409 | Reserved for future mutations. |
| **`STALE_MODEL` / `STALE_FACTOR_MODEL`** | **200** | **Degrade-but-ship.** Payload is valid; `errors[]` flags the stale source. |
| `MODEL_UNAVAILABLE` | 503 | No production model where one is required. |
| `BROKER_UNAVAILABLE` / `DB_UNAVAILABLE` | 503 | Upstream down. |
| `TIMEOUT` | 504 | Upstream timed out. |
| `RATE_LIMITED` | 429 | Reserved. |
| `INTERNAL` | 500 | Catch-all. Handler **never leaks exception text or stack** — only `request_id` (`errors.py:235–260`). |

**The `STALE_*`-as-200 rule is the heart of the contract.** A degraded but usable payload ships with HTTP 200 *and* a non-empty `errors[]` (`_HTTP_STATUS` maps both stale codes to 200; `DegradedResponse`, `errors.py:116–138`). The client renders the data **and** the trust banner — it never has to choose between showing stale data and showing an error, because it gets both. A client that treats any non-empty `errors[]` as a failure is wrong; it must inspect `code`.

```python
# Route pattern — degrade-but-ship. Raise the typed exception; the handler envelopes it.
if snapshot_age > STALE_THRESHOLD:
    raise DegradedResponse(
        "serving snapshot older than 90s",
        code=ErrorCode.STALE_MODEL,
        data=projected_dto,           # payload still ships
    )
```

**Two bugs this section mandates fixing (from §4 of the brief):**

1. **No raw `HTTPException` in routes.** `/trade-ideas` and `/replay` currently raise `HTTPException(500, detail=str(exc))` (`trade_ideas.py:38`), which leaks exception text and bypasses the envelope entirely. **Replace every raw raise with a typed `ApiException` subclass** so the registered handler (`errors.py:192–204`) produces an enveloped `errors[]` with a stable code. The `Exception` catch-all exists precisely so an unconverted error still degrades safely — but routes must not rely on it.
2. **`/trade-ideas/{symbol}` (501 stub) must never 500 the client.** The drawer hydrates from the list row (no detail round-trip in v1). The endpoint stays non-required, but if hit it must return a clean enveloped `NOT_FOUND`/`501`-style response, not an unhandled exception.

### 3.3 Casing — Pydantic camelCase alias generator on the BFF DTOs (single source of truth)

This is the one undecided seam and a go-live blocker. **Decision: the BFF emits camelCase for inner-DTO fields via a Pydantic alias generator. One place, server-side. The frontend's existing camelCase binding is the contract — we conform the BFF to it, not the reverse.**

The mismatch is real and verified:
- The frontend binds **camelCase DTO fields**: `api.ts:65` sorts on `b.targetWeight`; `Sym`/`TradeIdea` use `metaProbability`, `stageLatency`, etc.
- The BFF DTO emits **snake_case**: `trade_ideas_service.py:343` constructs `target_weight=...`, `:302` `gross_target_weight=...`. `model_dump` serializes those names verbatim.

So today `data.targetWeight` would be `undefined` against a real BFF response. Two options were on the table; we pick the alias generator because it is one change in one place and keeps the frontend untouched:

```python
# Base for all inner DTOs (TradeIdea, Sym, TradeIdeasResponse, …). NOT the envelope.
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

class BffModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,   # target_weight -> targetWeight on output
        populate_by_name=True,      # internal code keeps writing snake_case
    )
    # serialize with by_alias=True (set once in the envelope/route serializer)
```

**Critical scoping rule — the envelope keys stay snake_case; only the inner `data` DTO goes camelCase.** Confirmed by both sides: `envelope.ts` reads `as_of`, `staleness_seconds`, `source_freshness`, `model_version`, `request_id` — all snake. The alias generator therefore applies to the **DTO base class only** (`TradeIdea`, `Sym`, `TradeIdeasResponse`, `Strategy`, `Model`, …), never to `ApiEnvelope`. Two casing conventions on one wire is intentional and correct here: the envelope is a Python↔TS↔Swift shared shape frozen in snake; the DTOs mirror the frontend's TS view-model in camel.

The rejected alternative — a snake→camel adapter in `data/api.ts` — pushes the contract into the client, duplicates field knowledge, and gives every future endpoint a place to drift. Single source of truth wins.

### 3.4 Versioning the tmpfs `TradeIdeasResponse` snapshot

The tmpfs `trade_ideas.json` is **one of only two real data spines** (`trade_idea_publisher.py`), so the engine→BFF boundary is the system's highest-leverage failure point. Today the payload is `{"as_of": ..., "report": <report.to_dict()>}` (`trade_idea_publisher.py:137–140`) — it carries a timestamp but **no schema version**. A field rename or shape change on the engine side fails silently or with a confusing parse error in the BFF.

**Add `schema_version`** (the `as_of` already exists; surface it into the envelope's `as_of` rather than mtime):

```python
# trade_idea_publisher.py — publish_once()
SNAPSHOT_SCHEMA_VERSION = 1   # bump on any breaking shape change to report.to_dict()

payload = {
    "schema_version": SNAPSHOT_SCHEMA_VERSION,
    "as_of": datetime.now(timezone.utc).isoformat(),
    "report": report.to_dict() if hasattr(report, "to_dict") else report,
}
```

The BFF's `TmpfsTradeIdeasCache` reads `schema_version` and, on mismatch, raises `DegradedResponse(code=STALE_MODEL)` (ship the last-known-good projection with a warning) rather than crashing or — worse — projecting a misaligned payload. The publisher and cache are the only writer/reader of this format (`trade_idea_publisher.py:16–18`), so the version lives in exactly one producer and one consumer.

### 3.5 Wire `staleness_seconds` (currently dropped)

`staleness_seconds` is a first-class trust field that the frontend already binds (`deriveTrust`, `envelope.ts:67` flips `stale` past `STALE_THRESHOLD_SECONDS`), but the BFF **computes it and throws it away** — `TradeIdeasService._last_staleness_seconds` is never read into the envelope (brief §4). Wire it: compute staleness as `now − snapshot.as_of` (the embedded producer timestamp, **not** file mtime — mtime lies after an atomic `os.replace`), pass it to `envelope(staleness_seconds=...)`, and when it exceeds the threshold also emit a `STALE_*` entry in `errors[]` so the degrade-but-ship path (§3.2) lights up consistently.

### 3.6 Mandated contract tests (two, both cheap, both high-leverage)

These are not optional — they are how "frozen" stays true without a code reviewer manually diffing three languages.

1. **Cross-language envelope field-parity test.** Assert the field set (names + nullability) of `ApiEnvelope` is identical across `envelope.py`, `envelope.ts`, and the iOS model, **including `request_id`**. Implement as a single canonical field manifest checked against each language's introspected/parsed model (Pydantic `model_fields`; a tiny TS/Swift reflection or a checked-in golden JSON). Fails CI the moment any layer adds, renames, or drops a field. This is the test that keeps the one consistent contract consistent.

2. **Engine→BFF snapshot contract test.** Feed a fixture `trade_ideas.json` (with `schema_version` + `as_of`) through `TmpfsTradeIdeasCache` → `TradeIdeasService` and assert: (a) the projected DTO matches the frozen `TradeIdeasResponse` shape, (b) camelCase aliases are emitted (`targetWeight` present, `target_weight` absent on the wire), (c) the 7 v2 fields are `null`/absent not fabricated (`trade_ideas_service.py:358–367`), and (d) a bumped/unknown `schema_version` degrades to `STALE_*`-200 rather than throwing. This guards the most fragile real boundary in the system.

### 3.7 Pagination, limits & versioning — right-sized for single-operator

URI versioning is settled: **`/api/v1`** prefixes the whole surface. Bump the path only on a breaking envelope or contract change; additive DTO fields (gated `null` today) are non-breaking by construction thanks to `exclude_none`.

Pagination is deliberately minimal — this is single-operator, single-tenant, and the payloads are small and bounded by design:

- `/overview` `top_actionable` is **capped at ≤5** server-side; `/trade-ideas` is filtered by `min_abs_weight` (default `0.0025`) and `symbols`, not paged. No cursor needed.
- Endpoints that *could* grow (`/markets`, `/trade-ideas` with a wide symbol set) take a simple bounded `limit` with a hard server-side ceiling and a `warnings[]` note when truncated. **Do not** build cursor pagination, page tokens, or `Link`-header machinery — there is no client that needs it and no dataset large enough to justify it. That is exactly the kind of web-scale over-engineering this product should avoid.
- Inputs are validated (`bar_limit`, `min_abs_weight`); out-of-range values yield `VALIDATION_FAILED` (422) with the `field` path but never the rejected value (`errors.py:207–232`).

**Net:** the envelope and error taxonomy are already correct and consistent — freeze them with the two parity/contract tests. The only net-new work in this section is mechanical and small: the camelCase alias generator on DTOs (one base class), `schema_version` on the tmpfs payload (one constant, one reader check), wiring `staleness_seconds` (read a field that's already computed), and deleting the two raw `HTTPException(500)` raises in favor of typed enveloped errors.

---

## 4. Service architecture & data access

This section defines the layer between the FastAPI routes and the engine's two real data sources. The shape is already mostly correct in the tree — `deps.py` singletons, `LRUTTLCache`, the typed error taxonomy, the staleness-vs-`as_of` reader all exist and are production-grade. What's wrong is that the routes don't *use* the machinery that's sitting right next to them. The design below codifies the boundary and fixes the three live bugs that defeat it.

### 4.1 The layered boundary

```
┌─ Engine (writes) ────────────┐        ┌─ BFF (read-only) ──────────────────────────┐
│ trading process              │ bars   │  routes/  ──Depends()──►  deps.py singletons │
│  • bars  → TimescaleDB       │───────►│  (zero src.* logic)         │ (lru_cache)     │
│  • trade_ideas.json → tmpfs  │ tmpfs  │   envelope() ◄── DTOs ◄── services/           │
│    (write→fsync→os.replace)  │───────►│                          (zero FastAPI imports)│
└──────────────────────────────┘        │   data access: bars / tmpfs / MLflow / refmap │
                                         └────────────────────────────────────────────────┘
```

Three rules, each load-bearing and each already partially true:

1. **Wrap, don't rewrite.** A service is a plain Python class that bridges to existing `src.*` code (`TradeIdeasService` bridges `src.ui.trade_ideas`, `trade_ideas_service.py:262`). It never reruns business logic it can read from a snapshot.
2. **Services have zero FastAPI imports and return DTOs, not DataFrames.** `database.get_bars()` returns a `pd.DataFrame` (`database.py:314`); the service is the seam that converts it to a Pydantic DTO (`Sym`, `BarMicro`). No DataFrame, `Engine`, broker session, or `Depends` symbol crosses into a route. This is the one property that makes the future process-split (§6) mechanical rather than a rewrite — so it is a hard rule, not a style preference.
3. **The engine owns all writes; the BFF only projects / normalizes / redacts / tags / degrades.** No service opens a write transaction, writes Parquet, or opens a broker session. When a source is absent the service returns `null` + a warning — never a synthesized number (§4.7).

### 4.2 Dependency injection: one singleton per service, overridable in tests

`deps.py` already implements the correct pattern: each service is a `@lru_cache(maxsize=1)` provider, so the first request lazily constructs a process-wide singleton and every subsequent request reuses it (`deps.py:29-51`). Tests swap implementations via `app.dependency_overrides` (which wraps the `lru_cache`, leaving the cache untouched) and `reset_service_singletons()` clears state between cases (`deps.py:54-61`).

The singleton lifetime is **not** a micro-optimization — it is what makes the snapshot machinery work at all. `TradeIdeasService` holds two pieces of per-instance state: the `TmpfsTradeIdeasCache` reader and the 30-second `_TtlLru` regenerate-debounce (`trade_ideas_service.py:193-197`). If the service is reconstructed per-request, both are reborn empty every time. **This is bug #1 (§4.6).**

Rule: **routes obtain every service through `Depends(get_*_service)`. A route may never call a service constructor directly.** Adding a new route means adding a provider to `deps.py` first.

```python
@router.get("")
def list_trade_ideas(
    svc: TradeIdeasService = Depends(get_trade_ideas_service),
) -> dict:
    ...
```

### 4.3 Lifespan: eager-load and fail-loud

The current lifespan is an honest stub — it logs `"db_pool": "stub (deferred to later sprint)"` and yields (`app.py:54-75`). That is fine while only the two snapshot-backed routes are mounted, because `TmpfsTradeIdeasCache` degrades gracefully on a missing file. It stops being fine the moment the bars-backed routes (`/markets`, `/symbols`) land, because a route that discovers a dead DB pool *per request* turns a config error into a runtime 500 storm.

Design: on startup, eagerly construct the shared resources and **fail loud** — a missing DB URL or unreachable Timescale should crash the boot, not the first request.

```python
@asynccontextmanager
async def _lifespan(app: FastAPI):
    setup_logging(...)
    app.state.db = DatabaseManager(require_env("WANG_DB_URL"))   # raises → boot fails
    app.state.db.engine.connect().close()                        # prove reachability now
    app.state.registry = ModelRegistry()                          # MLflow client (lazy queries)
    app.state.refmap = load_instrument_reference()                # §4.5, static, must parse
    try:
        yield
    finally:
        app.state.db.engine.dispose()
```

Three deliberate restraints, single-operator-appropriate:
- **MLflow is *not* a boot blocker.** The registry client constructs eagerly but the model query is lazy; MLflow being unreachable degrades `/model` to `model_version=null`, it does not stop the BFF (mirrors the degraded-mode contract: "MLflow/S3 unreachable → don't block").
- **The tmpfs snapshot is *not* a boot blocker.** Missing snapshot → `/trade-ideas` and `/overview` degrade with a warning; it must never trigger a pipeline run at boot.
- **The DB pool *is* a boot blocker** once bars-backed routes mount — readiness depends on it, so prove it once at startup.

This also feeds the liveness-vs-readiness split (§6): `app.state` holding live handles is what `/healthz` (readiness) inspects to report snapshot age, model-loaded, and DB reachability.

### 4.4 Data access — the two real sources, plus two read-only references

| Source | Owner | Access pattern | Notes |
|---|---|---|---|
| `bars` hypertable | engine writes | `database.get_bars(symbol, bar_type, start, end, limit)` (`database.py:289`) | Backs `/markets`, `/symbols`. Parameterized via SQLAlchemy `:bind` params + **mandatory `LIMIT`** — both already enforced (`database.py:300-312`). |
| tmpfs `trade_ideas.json` | engine writes (atomic) | `TmpfsTradeIdeasCache.read()` | Backs `/trade-ideas`, `/overview`. Staleness measured vs embedded `as_of`, not file mtime (§4.4.2). |
| MLflow registry | engine trains | `ModelRegistry.get_production_model()` + `search_runs` | Backs `/model`. Read-only; degrades to null. |
| Instrument-reference map | static config | in-memory dict (§4.5) | Needed by `/markets`, `/symbols` for `name` + `asset_class`. |

#### 4.4.1 The bars hypertable — query hygiene already correct, one addition

`get_bars` is parameterized and `LIMIT`-bounded. Two design notes for the new routes that consume it:

- **The `LIMIT` is mandatory and bounded.** `/markets` and `/symbols` must pass an explicit, capped `limit` (e.g. `bar_limit ≤ 1000`); never expose an unbounded fetch. The default `limit=10_000` (`database.py:295`) is a safety net, not a contract.
- **Cursor pagination, not offset.** A 7-year hypertable makes `OFFSET` pathological. If a future endpoint needs to page bars, page on the `timestamp` column (`WHERE timestamp < :cursor ORDER BY timestamp DESC LIMIT :n`) — Timescale's time-partitioning makes this a chunk-local scan. For the v1 surface (spark/candle strips of recent bars) a single bounded `get_bars` call suffices; **do not build pagination preemptively.**

#### 4.4.2 The tmpfs snapshot read — staleness vs `as_of`, debounced regenerate

The reader is the model the whole BFF should follow. It:
- reads the atomic file the publisher writes via write→`fsync`→`os.replace` (`trade_idea_publisher.py:142-153`), so it never sees a torn write;
- computes `staleness = now − payload.as_of` (`trade_ideas_service.py:115`) — **embedded `as_of`, never file mtime**, so clock drift between the publisher host and BFF host stays *visible* in metrics instead of being silently masked by a fresh mtime;
- returns `None` (→ caller degrades) on any of: missing file, `OSError`, unparsable JSON, wrong shape, missing `as_of`/`report`, or `staleness > 90s` (`trade_ideas_service.py:83-130`). Every failure mode is a clean, logged degrade.

On a miss/stale read, the service falls back to a synchronous regenerate **debounced by a 30s `_TtlLru`** keyed on the filter args (`trade_ideas_service.py:229-248`), so a thundering herd of concurrent misses triggers at most one pipeline run per 30s window. (The brief flags the sync-regenerate fallback itself as a reliability hazard — see §6/§7; the debounce is the current mitigation, the eventual fix is "stale snapshot + warning, never regenerate in a web worker.")

**Required hardening at the boundary (§2 of the brief):** the publisher payload is `{"as_of", "report"}` today. Add **`schema_version`** to the JSON and a cross-boundary contract test, because this file is one of only two real sources and a silent shape drift between engine and BFF is the single highest-leverage failure point. The reader already rejects shape mismatches; `schema_version` lets it reject *semantic* drift explicitly rather than via a downstream adapter exception.

### 4.5 The instrument-reference map (`symbol → {name, asset_class}`)

Bars carry only `symbol` + `bar_type` (`database.py:246-267`) — there is no instrument name or asset class anywhere the engine persists. `/markets` and `/symbols` need both (for display name and asset-class tint). This is a **static, read-only config artifact** the BFF owns, not engine data:

```python
# config/instruments.yaml  →  loaded once at lifespan into app.state.refmap
BTCUSDT: { name: "Bitcoin / USDT",  asset_class: crypto }
AAPL:    { name: "Apple Inc.",      asset_class: equity }
```

Resolution rule (honest degradation): **unknown symbol → `name = symbol`, `asset_class = "unknown"` (default tint)**. A symbol with bars but no ref entry must still render — it just shows its ticker as its name. Ship the loader with the `/markets` route; keep the file tiny and hand-maintained (single-operator — no instrument-master service).

### 4.6 Caching: the bounded thread-safe `LRUTTLCache`

`LRUTTLCache` (`cache.py:33`) is the one cache primitive: bounded (`maxsize`, oldest evicted on insert), per-entry TTL, thread-safe under a single `threading.Lock` (`cache.py:43-91`). The lock is justified — the critical section is a sub-µs `OrderedDict` update, so contention is negligible at single-operator load. **No Redis. No diskcache.** A second cache tier here would be pure over-engineering (§6).

Two distinct caching jobs, kept separate:

| Cache | Instance | Key | TTL | Invalidation |
|---|---|---|---|---|
| **Snapshot regenerate-debounce** | `_TtlLru` inside `TradeIdeasService` | `(symbols, bar_limit, min_abs_weight, allow_fallback)` | 30s | TTL-only; `reset_service_singletons()` in tests |
| **Read-through service results** (e.g. `/markets`, `/model`) | named `LRUTTLCache` via `@cached(cache_name=...)` | `(qualname, args, kwargs)` | 15–30s | TTL-only; `wrapper.cache_clear()` in tests |

TTL guidance: tie the cache TTL to the underlying source's natural cadence — never longer than the snapshot's 90s staleness ceiling, and shorter (15–30s) for anything fronting the bars table on a hot path. **Invalidation is TTL-based on purpose**: there is no write path in the BFF to trigger explicit invalidation, and inventing a pub/sub invalidation channel for a read-only single-tenant cockpit would be over-engineering. The only explicit-clear path is the test hook.

### 4.7 The BFF bugs to fix (brief §4)

Three bugs, all in the request→service→envelope path, all defeating machinery that already exists.

**Bug 1 — Routes instantiate services per-request, bypassing the singletons.**
`routes/trade_ideas.py:29` does `service = TradeIdeasService()` and `routes/overview.py:44` does `TradeIdeasService().list_ideas()` — both construct a *fresh* service per request. Consequences: a fresh `TmpfsTradeIdeasCache` + fresh empty `_TtlLru` every call, which **defeats the 30s regenerate-debounce** (every concurrent miss can now fire its own pipeline run) and makes `dependency_overrides` **inert** (tests can't swap the service because the route never asks the DI graph for it).
*Fix:* route through `Depends(get_trade_ideas_service)`. The provider already exists in `deps.py`; the routes just aren't calling it.

```python
# before:  service = TradeIdeasService()
# after:
def list_trade_ideas(svc: TradeIdeasService = Depends(get_trade_ideas_service)) -> dict:
    response = svc.list_ideas(...)
```

**Bug 2 — Raw `HTTPException(500, str(exc))` leaks exception text and bypasses the envelope.**
`routes/trade_ideas.py:38` raises `HTTPException(status_code=500, detail=str(exc))` (and `/replay` does the same). This leaks raw exception text — potentially a DSN or internal path — straight to the client, and returns a bare FastAPI body instead of an `ApiEnvelope`. The typed taxonomy that solves this **already exists**: `errors.py` defines `ApiException` subclasses, a generic handler that *never* includes exception text (`errors.py:235-260`), and the `STALE_*`-as-200 contract.
*Fix:* let unexpected errors propagate to the registered `_generic_exception_handler` (enveloped, `code=INTERNAL`, request-id only — no text), and raise typed subclasses (`DbUnavailable`, `DegradedResponse`, …) for known conditions. Delete the `try/except → HTTPException(500, str(exc))` blocks entirely.

```python
# before:
try:
    response = service.list_ideas(...)
except Exception as exc:
    raise HTTPException(status_code=500, detail=str(exc)) from exc   # leaks text

# after: no catch — the generic handler envelopes it without leaking;
# raise typed exceptions only for conditions you want to specialise:
response = svc.list_ideas(...)        # DbUnavailable / DegradedResponse raised inside
```

**Bug 3 — `_last_staleness_seconds` is computed but never wired into the envelope.**
`TradeIdeasService` records `_last_staleness_seconds` on every call (`trade_ideas_service.py:201, 226, 230`) precisely so the route can surface it — the docstring says so (`:198-200`). But the route calls `envelope(response.model_dump(...), source=...)` and **never reads it back** (`trade_ideas.py:40-43`). The envelope's `staleness_seconds` field (`envelope.py:53`) ships as null on every fresh-snapshot response.
*Fix:* read it after the call and pass it through; emit a `STALE_*` warning over threshold.

```python
response = svc.list_ideas(...)
staleness = svc._last_staleness_seconds          # promote to a public accessor
warnings = []
if staleness is not None and staleness > STALE_WARN_SECONDS:
    warnings.append(f"trade ideas snapshot stale: {staleness:.0f}s")
return envelope(
    response.model_dump(mode="json"),
    source="trade_ideas_service",
    staleness_seconds=staleness,
    warnings=warnings,
)
```
(`staleness is None` means the response came from the sync-regenerate fallback — surface that as its own warning, not as "fresh".)

### 4.8 Over-engineering to avoid (so the boundary stays thin)

This is single-operator / single-tenant. The service layer earns the word "scalable" by having **clean boundaries** (zero FastAPI imports in services, DTO returns, DI singletons) — *not* by adding infrastructure. Explicitly out of scope for v1: a second cache tier (Redis/diskcache) in front of `LRUTTLCache`; a connection-pool proxy beyond SQLAlchemy's built-in pool (`database.py:189`); an instrument-master service (the static map is enough); pub/sub cache invalidation (TTL is enough); and any per-request service construction "for isolation" (the singletons *are* the isolation, and reconstructing them is bug #1). The boundary is designed so a future process-split over a Unix socket is mechanical — but that split is deferred until profiling shows real contention, not built now.

---

## 5. Reliability & graceful degradation

The reliability bar for Aperture is not uptime-nines or web-scale throughput — it is a single operator, two uvicorn workers, two real data sources. The bar is: **degrade, don't 500; one field's failure never blanks a screen; the web layer never reruns the trading engine.** Everything here serves that, and explicitly stops short of distributed-systems machinery the single-tenant deployment doesn't need.

### 5.1 The contract: degrade-don't-500, end-to-end

The `ApiEnvelope<T>` (§2) is the load-bearing mechanism, not just a wrapper. Every degradation maps to envelope fields rather than an HTTP failure code. The rule is **one HTTP 200 carries both the data that succeeded and the structured record of what didn't.**

| Failure | HTTP | `data` | Envelope signal | Client render |
|---|---|---|---|---|
| Snapshot fresh, all fields present | 200 | full | — | LIVE |
| Snapshot stale (> threshold vs embedded `as_of`) | **200** | full | `errors[]: STALE_SNAPSHOT` | data + stale banner |
| One sub-field absent (e.g. `regime`, portfolio) | 200 | partial, field `null` | `warnings[]` | that tile → ComingState; rest LIVE |
| Source-of-truth (bars / snapshot) unreachable | 200 | `null` | `errors[]: DATA_UNAVAILABLE` | screen-level Empty/Unavailable |
| Unexpected exception in a route | **200** (enveloped) | `null` | `errors[]: INTERNAL` (no leak) | DataUnavailable |
| Genuinely malformed request | 4xx | — | — | client bug, not degradation |

This is already the design intent (`STALE_*` = 200-with-errors, `backend_design.md` §9.1/§20) and `/overview` already implements it correctly — portfolio fields null + standing warning while action counts render LIVE. **`/overview` is the reference implementation for every other route.** The honesty rule (§0) makes this non-negotiable: an absent field is `null` + `warning`, never a synthesized number.

Two existing bugs violate the contract and are P0 for the redesign:

- **Raw error leak.** `/trade-ideas` and `/replay` raise `HTTPException(500, detail=str(exc))` (`src/web/routes/trade_ideas.py:38`). This bypasses the envelope and leaks exception text. Replace with typed `ApiException` subclasses → enveloped `errors[]` with a stable `code`, never raw `str(exc)`. The generic handler in `errors.py` already does this correctly; these two routes simply route around it.
- **Staleness computed but dropped.** `TradeIdeasService._last_staleness_seconds` is computed and never read into the envelope. Wire it into `staleness_seconds` and emit `STALE_*` over threshold — otherwise stale data ships silently as if fresh. *(The client switches on `errors[].code`, never `message` — keep codes stable; messages are free to change.)*

**Invariant for every route handler:** the handler's own `try/except` degrades sub-fields to null; only request-shape errors reach a 4xx. A screen is blanked only when *both* real sources are gone, and even then it returns 200 with `data: null` + `DATA_UNAVAILABLE`, so the client renders a deliberate Unavailable state rather than a network error.

### 5.2 Enforce the `asyncio.to_thread` rule (currently a latent event-loop stall)

Every blocking `src.*` call (`database.get_bars()`, snapshot reads, MLflow registry lookups) is synchronous and GIL-bound. On a single process with 2 fixed workers, an un-wrapped blocking call inside an `async def` route **stalls the entire event loop** for that worker — every concurrent request on it hangs, including `/healthz`. The `asyncio.to_thread` rule (`backend_design.md` §2.3) exists for exactly this, but it is **unenforced** — nothing stops a future route from calling `src.*` inline.

**Action: make it a CI/lint gate, not a convention.** A small AST check (custom flake8/ruff plugin or a focused AST script in CI) that fails the build when, inside any `src/web/routes/*.py` coroutine, a call resolves to a `src.*` symbol *not* wrapped in `await asyncio.to_thread(...)` (or an explicit `# blocking-ok:` escape hatch for genuinely-cheap calls). This is the cheapest durable fix for the single most likely production stall and it scales with the codebase for free.

This is *right-sized*: the answer is not async DB drivers or a process pool — it is "push the ~handful of blocking calls off the loop and let CI keep it that way."

### 5.3 Kill the synchronous full-pipeline tmpfs fallback

The most serious reliability defect in the current BFF: on snapshot miss/stale, `TradeIdeasService` **runs the live trading pipeline synchronously inside the web worker** (the "regenerate" path, debounced 30s; see the module docstring at `src/web/services/trade_ideas_service.py:1-14`). This drags a heavy, GIL-bound, business-logic pipeline onto the **120ms hot path**, holds a worker hostage for the run, and violates the §0 invariant that **the BFF never reruns engine logic.** With 2 workers, two coincident misses can wedge the entire web layer.

**Replace with "stale snapshot + warning." A missing snapshot must never trigger a pipeline run.** New read semantics for the tmpfs source:

```
read snapshot:
  fresh (age ≤ T_stale)      → 200, data, no warning
  stale (age > T_stale)      → 200, data, errors[]: STALE_SNAPSHOT (age, as_of)
  missing / unparsable       → 200, data: null, errors[]: SNAPSHOT_UNAVAILABLE
  NEVER: regenerate / run pipeline
```

Freshness is measured against the snapshot's **embedded `as_of`**, not file mtime (mtime lies after an atomic replace). The engine — and only the engine — produces snapshots, via a **post-cycle writer hook treated as a hard dependency** (§5.5). This deletes the entire debounce/LRU-regenerate machinery, which is also the thing the DI bug below silently defeats anyway.

> **Removes an over-engineering trap, not capability.** The fallback *felt* like resilience ("always have an answer") but its failure mode is worse than the problem it solved. A stale answer with an honest banner is strictly better than a wedged worker.

### 5.4 Route through `deps.py` singletons (DI bypass defeats every safeguard)

Routes currently construct `TradeIdeasService()` per-request, which builds a **fresh `TmpfsTradeIdeasCache` + LRU on every call.** This defeats any cross-request debounce/cache and makes `dependency_overrides` inert (so tests can't substitute fakes). Route through `deps.py` `lru_cache` singletons via `Depends`. Once §5.3 lands the regenerate-debounce is gone, but the singleton discipline still matters: it's what makes the snapshot reader, freshness clock, and (later) instrument-ref map single, swappable, testable instances.

### 5.5 Schema-version the tmpfs boundary + the writer hook as a hard dependency

The engine→tmpfs→BFF boundary is **one of only two real data sources** and therefore the highest-leverage silent-failure point: an engine-side shape change breaks every read with no compile-time signal.

- Add **`schema_version`** and **`as_of`** to the `TradeIdeasResponse` JSON payload. The reader asserts a known `schema_version`; unknown major → treat as `SNAPSHOT_UNAVAILABLE` + log loudly (never best-effort-parse a payload you don't understand).
- A **cross-boundary contract test** (engine writer ↔ BFF reader against a golden fixture) that fails CI on drift — the analogue of the §2 envelope parity test.
- The **post-cycle writer hook is a hard dependency**: the deploy must verify the engine writes the snapshot atomically (write→fsync→`os.replace`) every cycle. If the hook isn't firing, that's an engine-availability incident surfaced via the freshness SLO (§5.6) — *not* something the BFF papers over by regenerating.
- **Durability:** the snapshot and the `.live_crash` flag must live on **durable disk, not tmpfs** — tmpfs-backed recovery state does not survive reboot, which would silently lose the crash signal exactly when it's needed.

### 5.6 Health: liveness vs readiness, a stall watchdog, and a freshness SLO

Current health is **crash-only**: `/healthz` answers as long as the process is alive. This misses the dominant failure mode of a 24/7 ingest+serve system — **wedged-but-alive** (process up, event loop stalled, ingestion frozen, snapshot hours stale). Split the single endpoint:

| Endpoint | Question | 200 when | Used by |
|---|---|---|---|
| `/livez` | Is the process responsive? | event loop services the request promptly | systemd / restart policy |
| `/readyz` | Should it receive traffic? | sources reachable + dependencies eager-loaded (lifespan) | nginx upstream gate |

Add two layers beyond crash-detection:

1. **Stall watchdog.** Use systemd `WatchdogSec` + periodic `sd_notify(WATCHDOG=1)` emitted from a task that proves the loop is *turning* (so a stalled loop fails to ping → systemd restarts). This catches the §5.2 stall even when `to_thread` discipline slips.
2. **Freshness SLO** — a structured readiness payload, not a boolean. These are *observations the BFF can honestly make from what it can see*; it does not invent SLOs for capabilities the engine doesn't expose.

```json
{
  "status": "degraded",
  "checks": {
    "last_bar_age_seconds": 41,
    "snapshot_age_seconds": 73,
    "model_loaded": true,
    "mv_staleness_seconds": null   // null when no MV scraped — honest, not 0
  }
}
```

`last_bar_age` and `snapshot_age` over threshold drive the same `STALE_*` signals the read paths emit, so health and per-request degradation tell **one consistent story.** `model_loaded:false` is *expected* in Wave 1 (degraded, not broken) and is what turns model-gated fields into `MODEL_REQUIRED` rather than errors. `mv_staleness` is reported `null`, not `0`, until a registry is actually scraped — fabricating a healthy `0` would itself violate §0.

### 5.7 Degraded-mode contracts (explicit, fail-safe-by-direction)

Each dependency has a **named, opposite-direction** failure posture. "Reliable" here means these are decided in the design, not improvised in an incident:

| Dependency | On failure | Rationale |
|---|---|---|
| **DB write** (engine path) | **HALT TRADING** | A position the system can't durably record is a correctness/safety hazard. Fail *closed*. |
| **Cache loss** (LRU/in-proc) | **Degrade to source-of-truth** (bars + snapshot), don't crash | Cache is an optimization; losing it costs latency, not correctness. Fail *open*. |
| **MLflow / S3 unreachable** | **Don't block trading**; model-gated fields → `MODEL_REQUIRED`/null | The model is an enrichment, not the spine. A model-registry outage must never stop the engine or blank `/overview`. Fail *open*. |
| **Snapshot missing/stale** (read path) | 200 + `STALE_*`/`SNAPSHOT_UNAVAILABLE`, **never regenerate** | §5.3. |

The asymmetry is the whole point: **writes fail closed (halt), reads/enrichments fail open (degrade).** A single global "retry everything" or "fail everything" policy would be wrong in both directions.

### 5.8 Honest limitation: per-worker SSE replay & in-process job registry

This must be stated plainly rather than papered over. With 2 uvicorn workers:

- The **SSE ring-buffer replay** is per-worker. A client that reconnects and lands on a *different* worker than the one that buffered the missed events gets **silent gaps** — the replay guarantee in `backend_design.md` §6 does not hold across workers.
- The **in-process job registry** is per-worker and in-memory: jobs are invisible to the other worker and **vanish on restart.**

Three honest options, in cost order — **none is "build a distributed event bus":**

1. **Document the limitation** and treat SSE replay as best-effort (acceptable for a single operator who can refresh).
2. **Pin SSE + jobs to a single dedicated worker/process** so replay and the registry are authoritative for that stream. Cheapest path to a *real* guarantee.
3. Only if profiling demands it, move the ring buffer + registry to a small shared store. **Do not reach for Redis/RQ preemptively** — that's the §6 over-engineering the brief explicitly forbids.

The redesign should pick (1) or (2) and **say so in the contract.** Claiming reliable cross-worker replay without one of these would be dishonest.

### 5.9 Explicitly NOT in scope (avoid over-engineering)

Reliability for a single-operator system is graceful degradation + clean boundaries, **not** redundancy infrastructure. The single-process / 2-worker stance is correct. We deliberately do **not** build: multi-node failover, Kubernetes/health-mesh probes, Redis-backed shared state for SSE/jobs (unless profiling forces it), circuit-breaker libraries, or distributed tracing. The §5.7 degraded-mode contracts plus the §5.6 watchdog give us the resilience that matters at this scale; everything heavier is cost without a tenant to justify it.

---

## 6. Real-time & the "what changed" model

Aperture is a monitoring surface for a single operator watching one engine cycle through its work. The real-time requirement is therefore narrow and honest: **push trust/freshness/health transitions as they happen, and surface "what changed" between cycles** — nothing more. There is no collaborative editing, no client-initiated stream traffic, no fan-out to many tenants. The design that follows is sized to exactly that.

The load-bearing constraint that shapes this entire section is in `src/execution/trade_idea_publisher.py:117-153`: each cycle the publisher writes a **fresh** snapshot (`as_of` + full report) to a `.tmp` sibling and `os.replace`-s it into place. **The previous snapshot is gone.** There is no call-history store. Any "what changed" feature that the *server* computes would need to diff against state the server no longer has. This is why the honest v1 diff lives on the client, and the server-streamed event-level diff is **DEFERRED**, not merely unbuilt.

### 6.1 Transport: SSE, not WebSockets

We use Server-Sent Events over two `GET` endpoints. The justification is concrete, not stylistic:

| Property | Why it fits Aperture |
|---|---|
| **Unidirectional** | The client never sends stream messages. Every mutation (when mutations land at all — today the engine is read-only/PAPER) rides a normal `POST`, not the stream. WebSockets' bidirectional channel is dead weight here. |
| **Plain HTTP** | SSE is `text/event-stream` over HTTP/1.1 or HTTP/2. nginx proxies it with `proxy_buffering off;` and nothing else — no `Upgrade`/`Connection` handshake to special-case, no WS-aware proxy config. |
| **Auto-reconnect + replay in the protocol** | The browser `EventSource` reconnects automatically and re-sends `Last-Event-ID`. We get reconnection and at-least-once replay semantics from the standard, not from bespoke client code. |
| **Degradation story** | A dropped SSE connection is invisible to correctness: the page still polls the 8 REST endpoints. The stream is an *accelerator*, never a source of truth. |

If genuinely interactive features ever appear (live order entry), revisit WebSockets at that point. Not before.

```
GET /api/v1/stream/ops    → trust / freshness / health transitions
GET /api/v1/stream/diff   → the diff rail ("what changed" chips)
```

Both emit standard SSE frames with monotonic `id:` for `Last-Event-ID` replay:

```
id: 4817
event: ops
data: {"kind":"freshness","snapshot_age_seconds":42,"model_loaded":true,"as_of":"2026-06-19T14:03:11Z"}
```

### 6.2 `/stream/ops` — trust, freshness, health (LIVE-capable)

This channel carries the small set of facts the operator needs to know *the moment they change*, all derivable from data the engine already persists or the BFF already computes:

- **Freshness** — snapshot age (now − embedded `as_of`, the same computation already in `trade_ideas_service.py:115`), and whether it has crossed the `STALE_*` threshold. This is the highest-value event: it's the difference between "the engine is keeping up" and "the engine has wedged."
- **Trust** — mode (`PAPER`, hardcoded today), and any envelope-level `errors[]` / `warnings[]` transitioning on or off.
- **Health** — model-loaded yes/no, and (once the registries are unified — see §observability) last-bar timestamp staleness and broker heartbeat.

The honest implementation is a lightweight publisher task registered in the FastAPI lifespan that wakes on a fixed cadence (10 s), recomputes this trust/freshness/health tuple, diffs it against the last emitted tuple, and **publishes only on change**. This is a self-diff of a tiny fixed-shape struct that the server fully owns each tick — it does **not** depend on call history, so it is honest to ship. Until last-bar/heartbeat persistence lands, `ops` carries only the freshness + trust subset; the health fields are simply absent (the client's honesty system renders them as ComingState), never faked.

### 6.3 `/stream/diff` — the diff rail, and the two-tier honesty split

This is where the "what changed" model must be precise about what is real.

**Tier 1 — client-side two-snapshot diff (SHIPPED, the honest v1).** The frontend already computes "what changed" entirely on the client, and this is the correct v1. From `apps/aperture-web/src/app/overview/page.tsx:40-60`, on each new snapshot the client diffs the current `top_actionable` against the previous one held in `localStorage`, keyed by symbol, and emits chips:

| Chip | Rule (verbatim from shipped code) | Scope |
|---|---|---|
| `new` | symbol present now, absent in previous snapshot | PRODUCED |
| `flipped` | action changed **and** it is a true reversal (`BUY→SELL` or `SELL→BUY`) | PRODUCED |
| `up` / `down` | `\|Δ target_weight\| ≥ 0.0025` (25 bps), displayed as `±N bps` | PRODUCED |

Three properties make this honest and worth keeping as the v1:

1. **Scoped to PRODUCED fields only** — `action`, `target_weight`, `top_signal_side`. It never diffs a field the engine doesn't produce (no regime flips, no cost deltas, no track-record changes), so it cannot invent a change in a phantom field.
2. **Cold-launch is honestly empty** — no previous snapshot ⇒ no chips (`overview/page.tsx:46`), not a wall of spurious "new" chips.
3. **Keyed to `as_of`** — a new cycle is a new `as_of` is a fresh strip; dismissal is persisted per `as_of` (`overview/page.tsx:80`) so it survives navigation but reappears on the next genuine cycle.

The role of `/stream/diff` **in v1** is therefore *not* to compute the diff — it is to be the **wake signal**: it emits a single lightweight `cycle` event when a new `as_of` is published, so the client recomputes its two-snapshot diff immediately instead of waiting for the next poll. This keeps the server stateless about history while making the rail feel live.

```
id: 5102
event: diff
data: {"kind":"cycle","as_of":"2026-06-19T14:04:00Z","schema_version":2}
```

**Tier 2 — server-streamed event-level diff (DEFERRED, do not build in v1).** A richer rail — per-idea chip events computed *server-side*, including transitions the client can't see (a symbol that appeared and vanished *between* the client's polls, or per-cycle attribution deltas) — requires an **append-only call-history store**. The current snapshot is overwritten every publish (`trade_idea_publisher.py:153`), so the server has nothing to diff against. This is the same persistence gate that blocks `track_record_*` (the IdeaDrawer already states this honestly: *"the snapshot is overwritten each publish"*, `apps/aperture-web/src/components/ideas/IdeaDrawer.tsx:226`). **Do not stand up a server-side `diff_publisher` that fabricates event-level diffs from a single live snapshot** — it would either silently miss inter-poll transitions or invent them. It unlocks only when the call-history store lands (same wave as track-record). Until then the existing backend_design §6.1 `diff_publisher` description is aspirational and should be marked DEFERRED in the consolidated doc.

### 6.4 Backpressure & disconnection

Per connection, one bounded `asyncio.Queue`:

- **Bounded queue, drop-oldest.** Each subscriber holds a queue with a small `maxsize` (50). A single operator with one or two tabs will never fill it; the bound exists to cap a wedged-client's memory, not to handle load. On overflow we **drop the oldest event and continue** rather than drop the connection — for a freshness/trust rail the *newest* state is what matters, and a stale-but-alive stream beats a forced reconnect storm. (This is a deliberate change from backend_design §6.2's "drop the connection on overflow," which is the wrong tradeoff for a state-snapshot rail.)
- **Ring-buffer replay via `Last-Event-ID`.** The broadcaster keeps a small ring buffer (≈100 events) of recent frames with monotonic ids. On reconnect, the client's `EventSource` sends `Last-Event-ID`; we replay everything after it that's still in the buffer. Anything evicted is covered by the fact that the client also holds REST as source-of-truth — a gap in the stream is a *latency* event, never a *correctness* event.

### 6.5 The honest multi-worker caveat (do not over-claim)

The deployment runs **2 uvicorn workers** (a fixed, deliberate single-host stance — §scalability). The ring buffer and subscriber set are **per-process**. This produces one real limitation that the design must state plainly rather than paper over:

> On reconnect, `EventSource` may land on a *different* worker than the one that served the original stream. That worker's ring buffer never saw the missed events, so `Last-Event-ID` replay returns nothing for that gap. **Cross-worker replay is not reliable.**

There are exactly three honest postures. We pick (1) for v1 and document (3):

1. **Pin SSE to a single worker/process (RECOMMENDED for v1).** Route `/api/v1/stream/*` to one dedicated worker (nginx `upstream` with a single SSE backend, or a dedicated single-process SSE service on its own port). Replay becomes reliable because every connection — original and reconnect — hits the same in-process ring buffer. This is trivial on a single host and matches the single-operator reality: the SSE fan-out is one viewer, not a crowd.
2. **Move the ring buffer to a shared store (Redis Streams / Postgres `LISTEN`/`NOTIFY`).** Reliable cross-worker replay, but it introduces a shared dependency and an eviction/ordering story we do not need for one viewer. **This is over-engineering for v1 — do not build it.**
3. **Document the limitation and lean on REST.** If we keep the buffer per-process and *don't* pin, we must not claim reliable replay. The client treats SSE as best-effort and reconciles against the REST endpoints on every reconnect. Acceptable, but strictly worse than (1) for no saved effort.

**Decision:** pin `/stream/*` to a single worker (posture 1); never advertise cross-worker replay we don't have. The same reasoning applies to any in-process job registry — keep it on the pinned/single process, or accept that jobs vanish on restart and say so.

### 6.6 Deliberately NOT built

To keep the rail right-sized for a single operator:

- **No WebSockets**, no bidirectional channel (§6.1).
- **No server-side event-level diff** until the call-history store lands (§6.3, Tier 2).
- **No shared-store SSE fan-out** (Redis Streams, message bus) — per-process buffer on a pinned worker is sufficient (§6.5).
- **No guaranteed/exactly-once delivery** — SSE is at-least-once-with-gaps by design; REST is the source of truth and closes any gap on the next poll.
- **No stream-carried business data the REST layer doesn't also serve** — the stream only *signals*; it never becomes a second, divergent data path. This preserves the §0 invariant: the BFF projects, it does not fabricate, on the stream as much as on REST.

---

## 7. Security & authorization

Aperture runs Phase-1 **VPN-only, single-operator, with no app-layer authn**. That posture is intentional and correct for one trusted operator behind a private network — but "no authn" is not the same as "no security boundary." This section closes the holes that bind-address alone leaves open, redacts secrets in depth, and pre-specifies the authz + mutation machinery so that when writes eventually land they land *closed*, not open. Everything here is sized for one tenant: no OAuth provider, no session store, no user database, no web-scale rate limiting.

The non-negotiable backdrop: **the engine is read-only / PAPER today.** `deriveTrust` hardcodes `mode: 'PAPER'` (`apps/aperture-web/src/data/envelope.ts:62`, with the comment *"engine is read-only; live_orders_sent=0"*) and the deployed order path sends zero orders. Every mutation control in §7.5 is therefore **specified but dormant** — do not advertise, mount, or document a mutation endpoint while the surface is read-only (§3, `api_contracts_v2.md` over-spec). Build the lock before you build the door.

---

### 7.1 Threat model (single-operator / single-tenant)

| In scope (Phase 1) | Explicitly out of scope |
|---|---|
| A process on the host (or anything that crosses the VPN to uvicorn) **forging the role header** | Multi-user account takeover / session theft (no sessions exist) |
| **Secret leakage** through logs, error envelopes, or free-text values | DDoS / volumetric abuse (VPN-only, one client) |
| **CORS** allowing a browser on any origin to read operator data | CSRF on mutations (no mutations live; revisit in §7.5) |
| **Accidental** destructive writes once mutations land (fat-finger, double-submit) | Web-scale rate limiting, WAF, bot defence |

The single realistic adversary in Phase 1 is **"anything that can open a TCP connection to uvicorn"** — a misconfigured proxy, a sidecar, a compromised co-located process, or a future second service. The defence is to stop trusting the network edge as the *only* control. Bind-address is a layer, not the layer.

---

### 7.2 The forged-role hole — sign the role at the trust boundary

The architecture (`src/web/README.md:104`, `src/web/logging.py` role contextvar) anticipates an `X-Role` header (`viewer` / `operator` / `admin`) injected by nginx after VPN/mTLS termination. **The hole: `X-Role` is an unsigned, client-settable forwarded header.** Anything that reaches uvicorn directly — bypassing nginx — can send `X-Role: admin` and be believed. Today the BFF doesn't even parse it (no role read in the `app.py:101` middleware), so the *header is inert* — but the moment role-gating is wired (§7.4), an unsigned header is a privilege-escalation primitive. Bind-address does not save you here: the threat is a process that is *already inside* the bind.

Two acceptable resolutions; **pick exactly one and make it the only path that sets the role contextvar:**

**Option A — HMAC the role between nginx and uvicorn (recommended).** nginx is the only entity holding the shared secret; uvicorn rejects any role assertion not signed by it.

```nginx
# nginx, after VPN + (optional) mTLS client-cert maps to a role
set $role "operator";
proxy_set_header X-Role         $role;
proxy_set_header X-Role-Ts      $msec;
# HMAC-SHA256(role + "." + ts) keyed with APERTURE_ROLE_HMAC_KEY
proxy_set_header X-Role-Sig      $computed_sig;   # via njs / lua
```

```python
# src/web/auth.py — verify_role_header(): the ONLY writer of the role contextvar
def resolve_role(headers) -> Role:
    raw, ts, sig = headers.get("X-Role"), headers.get("X-Role-Ts"), headers.get("X-Role-Sig")
    if not (raw and ts and sig):
        return Role.VIEWER                       # fail-closed to least privilege
    expected = hmac.new(ROLE_HMAC_KEY, f"{raw}.{ts}".encode(), "sha256").hexdigest()
    if not hmac.compare_digest(expected, sig):   # constant-time
        log.warning("role_sig_invalid", extra={"claimed_role": raw})
        return Role.VIEWER                        # forged → viewer, never the claimed role
    if abs(now_ms() - int(ts)) > 30_000:         # replay window: 30s, clock-skew aware
        return Role.VIEWER
    return Role(raw)
```

**Option B — resolve roles *inside* uvicorn**, ignoring any inbound role header entirely. Map the **mTLS client-certificate CN** (or VPN identity nginx exposes as a header nginx itself controls and the client cannot forge end-to-end) to a role via a static dict in `config/web.yaml`. Simpler if mTLS is already terminated; no shared HMAC secret to rotate.

**Decision:** Option A unless mTLS client certs are already deployed, in which case Option B is less moving-parts. Either way the invariant is: **an unsigned/unverifiable role claim resolves to `viewer` (least privilege), never to the claimed value, and never to a default of `operator`.** Strip any inbound `X-Role*` headers at nginx so only nginx-minted ones survive.

*Over-engineering to avoid:* no JWT, no OIDC, no token introspection service. One HMAC key (or one cert→role map) in `config/web.yaml`, rotated by editing a file and reloading nginx.

---

### 7.3 CORS — drop the wildcard, drive from config

Today (`src/web/app.py:87-92`):

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

`allow_origins=["*"]` lets a page on *any* origin issue credentialed reads of operator data from a browser that can reach the VPN'd host. For a single-operator cockpit the legitimate origin set is tiny and known. **Drive it from `config/web.yaml`, per-environment, fail-closed:**

```yaml
# config/web.yaml
web:
  cors:
    allowed_origins:            # dev override may add http://localhost:4180
      - "https://aperture.internal"
    allow_methods: ["GET", "OPTIONS"]    # GET-only until mutations land (§7.5)
    allow_headers: ["X-Request-Id"]
    expose_headers: ["X-Request-Id"]
```

```python
cfg = load_web_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors.allowed_origins,   # never ["*"] outside dev
    allow_methods=cfg.cors.allow_methods,     # add POST/DELETE only when §7.5 ships
    allow_headers=cfg.cors.allow_headers,
)
```

Rules: **dev** may include `localhost`; **prod must never be `["*"]`** (assert this at startup and fail loud if the env is prod and origins contains `*`). Keep `allow_methods` to `GET`/`OPTIONS` while read-only — advertising `POST`/`DELETE` (as `app.py:90` does today) signals a mutation surface that doesn't exist. Widen it in lockstep with §7.5, not before.

---

### 7.4 Phased authorization roadmap

Phase 1 ships **one role gate that is a no-op in practice** (everything is a viewer-safe read) but the *plumbing* is in place so Phase 2 is a config change, not a refactor.

| Role | Phase 1 (read-only) | Phase 2 (writes land) |
|---|---|---|
| `viewer` | Full read of all 8 endpoints (§1). This is the default and the only role the operator needs today. | Unchanged: all reads, **zero mutations**. |
| `operator` | Identical to viewer (no extra surface exists). | Gated mutations (§7.5): check-in, clear-halt, refit — with confirmation + idempotency. |
| `admin` | Identical to viewer. | Operator powers **+** role/secret rotation, escalation-test, audit export. |

Implementation: a single `require_role(min: Role)` FastAPI dependency reading the **verified** role contextvar (§7.2). In Phase 1 every route is `Depends(require_role(Role.VIEWER))` — cheap, uniform, and it forces the verified-role path to be exercised in prod *before* it guards anything dangerous. The role lands in the JSON log line (`logging.py` already reserves the `role` contextvar) so every request is attributable to a role even with one human operator.

*Over-engineering to avoid:* no per-endpoint ACL matrix, no policy engine (OPA/Casbin), no permission strings. Three roles, one `min`-level check, ordered `viewer < operator < admin`.

---

### 7.5 Gated mutations — specified now, dormant until writes land

**Do not mount these in v1.** The engine is read-only and `deriveTrust` hardcodes `mode='PAPER'`. This subsection exists so the controls are *designed* before the first write endpoint is tempting to ship without them. A mutation is admissible only when **all five gates pass**; any gate that cannot be evaluated **fails closed**.

```
POST /api/v1/<mutation>
  Headers: X-Role(verified) · X-Idempotency-Key · X-Confirmation-Phrase
  ──► [1] role check    ──► [2] confirmation phrase ──► [3] idempotency
       (operator+)            (exact match, typed)        (fail-CLOSED)
  ──► [4] execute (engine owns the write) ──► [5] audit hook (HMAC-chained)
```

**[1] Role check.** `require_role(Role.OPERATOR)` on the verified role only (§7.2). A forged/unsigned role is `viewer` → 403.

**[2] `X-Confirmation-Phrase`.** The client must echo a server-specified exact phrase (e.g. the symbol + action, or `CLEAR-HALT`) for destructive/irreversible actions. Mismatch → `422 CONFIRMATION_REQUIRED` in the envelope (`errors[]`, switch on `code` per §2). This is the fat-finger guard, not a security control — but it's the cheapest one and the operator will thank it.

**[3] `X-Idempotency-Key` — fail-CLOSED, with eviction and a 2-worker story.** Double-submits (retries, reconnects, the operator clicking twice) must not double-execute an order or double-clear a halt.

- **Store:** SQLite table `idempotency(key TEXT PRIMARY KEY, request_hash TEXT, response_json TEXT, created_at INTEGER)` on **durable disk, not tmpfs** (§6 DR: recovery state must survive reboot). One small table; no Redis (explicitly rejected as over-engineering in §6).
- **Fail-closed posture:** if the store is **unavailable or unwritable, reject the mutation** (`503 IDEMPOTENCY_UNAVAILABLE`) rather than executing un-deduplicated. A trading write you can't dedupe is worse than a write you didn't make. This is the single most important word in §7.5: *closed*.
- **2-worker concurrency:** with 2 uvicorn workers two retries can race. Use the DB as the lock: `INSERT … ON CONFLICT(key) DO NOTHING` and treat *rows-affected = 0* as "another worker owns this key" → return the stored response if present, else `409 IN_FLIGHT`. SQLite's single-writer lock + a short `busy_timeout` makes this correct without a separate lock service. Also bind the key to a **request-body hash**: same key + different body → `422` (a key is a promise about *one* request, not a reusable token).
- **Eviction:** keys expire after **24h** (the realistic retry/replay horizon for one operator). Evict lazily on write (`DELETE WHERE created_at < now-24h`) plus a daily cron sweep — no background thread, no TTL daemon.

**[4] Execute.** The route still does **not** write — it calls the engine's mutation surface; the engine owns the write (the §0 invariant survives into the mutation era). The BFF projects the result into an envelope.

**[5] Audit hook.** Every mutation appends to the HMAC-chained audit log. Note the engine gap (§5): `ComplianceAuditLogger` is *never instantiated* → 0 audit rows today. **Wiring the audit writer is a hard prerequisite for the first mutation** — an unaudited write must be impossible, so the audit append failing also **fails the mutation closed**. The HMAC chain depends on monotonic correct time, which is exactly why NTP/chrony is a preflight blocker (§6).

**Lifting PAPER mode** (`deriveTrust` `mode='PAPER'` → real `mode` from `live_orders_sent`/engine trust) is the *trigger* that turns this subsection on. Until that flips, the honest move is: don't ship the endpoints, don't widen CORS methods, don't advertise the headers (`api_contracts_v2.md` currently does — delete them from the "current contract" per §3).

---

### 7.6 Secret redaction — defence-in-depth on values, not just key-names

The current redaction (`docs/backend_design.md` §13, `src/web/auth.py`) is a **recursive filter on key *names*** matching `(^|_)(secret|password|token|key)($|_)`, plus a hardcoded denylist (`api_key`, `database_password`, `broker_password`, `signing_key`, `jwt_secret`, …). That correctly stops a `{"database_password": "..."}` field from ever serializing. **It does not catch a secret living in a *value* under an innocent key** — the classic leak being a DSN, broker URL, or token embedded in a free-text **error message**:

```json
{ "errors": [{ "code": "INTERNAL",
    "message": "could not connect: postgresql://aperture:hunter2@db.internal:5432/bars" }] }
```

The key is `message` (allow-listed); the *value* carries the credential. Add a **value-scrubbing pass** as a second layer, applied to every string on its way out — especially the generic-exception path (`errors.py`, which the brief notes already refuses to leak stack/exception text — extend it to scrub the message body it *does* emit):

- **Connection-string / URL credentials:** redact the `user:pass@` userinfo in any `scheme://…` (Postgres, Redis, AMQP, MLflow, S3 presigned). `postgresql://aperture:***@db.internal/bars`.
- **High-entropy / known-shape tokens** in free text: `sk-…`, JWT triplets (`xxx.yyy.zzz`), AWS-key shapes, bearer headers.
- **Belt-and-braces:** the existing key-name filter stays; the value scrubber is *additive*. Apply both at the single envelope-serialization choke point so no route can bypass it.

This is genuinely defence-in-depth: typed DTOs (the structure can't carry secret-named fields) → key-name denylist (catches dicts that slip through) → **value scrubber** (catches secrets that rode in as data inside an allowed field). For a single-operator system the threat isn't an attacker scraping the API — it's the operator's own log aggregator / screen-share / bug report quietly exfiltrating a DSN. Cheap, no dependencies, high payoff.

---

### 7.7 What we deliberately do *not* build (Phase 1)

Per the single-tenant mandate (§6 "scalability — deliberately do not build"), the following are **out of scope** and adding them is the over-engineering to avoid:

- **No auth provider / OIDC / session store / user DB** — there is one operator behind a VPN.
- **No rate limiting / token bucket / WAF** — VPN-only, one client; Redis-backed limiting is explicitly rejected.
- **No CSRF tokens** — no mutations are live; revisit *with* §7.5, not before.
- **No secrets-manager integration (Vault/KMS)** — secrets stay in `config/` + env, redacted on the way out; the value scrubber (§7.6) is the realistic mitigation, not a vault.
- **No per-request signing of *read* responses** — integrity inside the VPN is provided by TLS to nginx; signing reads is ceremony without a threat.

The bar for adding any of these is a *demonstrated* threat (a second user, an exposed origin, a real abuse signal), not a checklist. "Secure" here means: the role can't be forged, secrets can't leak, origins are pinned, and the day writes land they land closed and audited.

---

### 7.8 Phase-1 security checklist (lock these)

1. **Sign or internally-resolve the role** (§7.2) — unsigned `X-Role` resolves to `viewer`; strip inbound role headers at nginx. *Do this before any role gate guards anything.*
2. **CORS from `config/web.yaml`**, never `["*"]` in prod, `GET`/`OPTIONS` only while read-only (§7.3); assert-fail on prod + wildcard.
3. **Role plumbing in place** (`require_role`, role in JSON logs) even though every route is viewer-safe today (§7.4).
4. **Value-scrubbing redaction** layered onto the existing key-name denylist, at the envelope choke point (§7.6).
5. **Mutation gates specified, endpoints unmounted** (§7.5); fail-closed idempotency (durable SQLite, 24h eviction, DB-as-lock for 2 workers), confirmation phrase, audit-append-or-abort. Flip on only when PAPER mode lifts.

---

## 8. Observability

Observability for a single-operator system has one job: when something is wrong at 3am, the operator can answer *what broke, when, and is the data I'm looking at real* — without attaching a debugger. We already have two of the three pillars at production grade (structured logs, request correlation). The third (metrics) is built but **wired to the wrong registry** — the single highest-leverage fix in this section. Everything downstream of metrics (latency budgets, alerts) is blocked until that's fixed, so we sequence accordingly and refuse to pin alerts on numbers that aren't being scraped yet.

The guiding discipline: **alert on what is real (freshness, model-loaded, breaker state), not on what is aspirational (per-endpoint p95 before the histogram is even scraped).**

### 8.1 Structured Logging — LIVE, keep as-is

The BFF emits one JSON object per log line to stdout (`src/web/logging.py`), captured by journald in prod and `docker logs` in dev. This is done and correct — do not rebuild it.

Shape (`JSONFormatter`, `logging.py:124-171`):

```json
{"ts":"2026-06-19T14:22:01.481Z","level":"INFO","logger":"web.routes.trade_ideas",
 "msg":"snapshot stale, regenerating","request_id":"a1b2c3d4","role":"viewer",
 "route":"GET /api/v1/trade-ideas","status":200,"duration_ms":84.2,
 "staleness_seconds":97.3}
```

Three properties worth preserving explicitly because they're easy to regress:

- **Exceptions live only in the log, never in the response.** Tracebacks go into a single `"exc"` string field (`logging.py:166-167`); the envelope's `errors[]` carries only a stable `code` (§2). This is the log/response boundary that the error taxonomy depends on — a raw `HTTPException(500, detail=str(exc))` (the known bug at `trade_ideas.py:38`) violates it from *both* sides and must be replaced with typed enveloped errors.
- **`extra=` fields are promoted to top-level keys** (`logging.py:158-162`), so a service can attach `staleness_seconds`, `snapshot_age`, `symbol` to a record and it becomes queryable without a schema change. Use this instead of string-interpolating values into `msg`.
- **uvicorn/fastapi access logs are re-routed through the same JSON handler** (`logging.py:199-204`) — one log format, not two. Keep that.

**One redaction gap to close (cross-ref §7 Security):** the secret denylist matches on *key names*, not values. A DSN or token embedded in a free-text `msg` or an `extra` value (e.g. a DB error string) will be logged in clear. Redact structured fields on the way into the formatter, not just exception text. This is a logging-layer fix, applied here, even though the threat is catalogued under Security.

### 8.2 Request Correlation — LIVE, this is our tracing for v1

`request_id` is the correlation id across all three surfaces (envelope.py ↔ envelope.ts ↔ iOS, per §2) **and** the log context. It's propagated via a `contextvars.ContextVar` (`logging.py:29`), bound by the request middleware on entry and reset on exit (`bind_request_context` / `reset_request_context`, `logging.py:37-84`). Because it's a contextvar, every log line emitted anywhere in the async call stack for a request automatically carries its `request_id` — no threading of the id through call signatures.

The operator workflow this enables, end to end:

```
client sees errors:[{code:"STALE_SNAPSHOT"}] + request_id "a1b2c3d4"
   → grep request_id "a1b2c3d4" in journald
   → full ordered trace of that request: route, staleness, which upstream degraded, exc
```

**Decision: `request_id` IS the v1 distributed-tracing story. Do not adopt OpenTelemetry now.** With a single process and two workers, OTel spans buy almost nothing over a correlated structured log and cost a collector, exporter config, and span plumbing through every `src.*` call. Defer until/unless we split the engine and BFF into separate processes over a socket (§6 — itself deferred until profiling justifies it). When that day comes, `request_id` is already the natural trace-id to carry across the boundary, so nothing here is throwaway.

The contextvar carries `user` and `role` too (`logging.py:30-34`) — useful for an audit trail once the role header is signed (§7), but today `role` is an unsigned forwarded header, so treat it as *informational in logs, not authoritative*.

### 8.3 Metrics — THE CRITICAL FIX (two registries, one is scraped)

There are two Prometheus registries and **they do not overlap**. Verified:

| Producer | Registry | Scraped by `/metrics`? |
|---|---|---|
| BFF self-metrics (`bff_requests_total`, `bff_request_duration_seconds`, cache, SSE, upstream errors) | global `REGISTRY` | **Yes** — `generate_latest(REGISTRY)` at `src/web/metrics.py:154` |
| Pipeline `MetricsCollector` (NAV, exposure, orders, slippage, signals, model staleness, drift, breaker triggers) | its own `CollectorRegistry()`, the default at `src/monitoring/metrics.py:44` | **No** — nothing serves this registry |

The BFF `/metrics` ASGI endpoint serves *only* the global registry. The pipeline's `MetricsCollector` constructs a fresh, private `CollectorRegistry` whenever no registry is passed in — and nothing passes one in, and nothing scrapes it. **Every business-and-trading metric the engine records is written to a registry that no scraper can see.** The Monitoring page in the frontend is correctly held as `ComingState` partly *because* of this (§5) — there's no metric backend to feed it.

This is load-bearing: **§8.4 latency budgets and §8.5 alerts are meaningless until this is fixed**, because half the signals don't reach Prometheus.

**The fix (pick one; recommend A):**

- **A — Unify on the global registry.** Pass the global `REGISTRY` into `MetricsCollector(registry=REGISTRY)` at the engine's metrics-init site. The collector already accepts an injected registry (`metrics.py:39-44`); we're just supplying it. One `/metrics` endpoint then exposes both `bff_*` and `wang_trading_*` series. Simplest, fewest moving parts, single scrape target. **In a 2-worker BFF + separate engine process, this only works if the engine process *also* exposes the unified registry over its own `/metrics`** — see the multi-process note below.
- **B — Scrape both.** Stand up the engine's metrics on its own port via `start_http_server` (already imported, `metrics.py:19`) and add a second Prometheus scrape target. More config, but cleaner process isolation and the honest model if engine and BFF are genuinely separate processes.

**Multi-process reality check (do not skip):** the BFF runs **2 uvicorn workers**, and the engine is a **separate process** from the BFF. Prometheus client metrics are per-process. That means:

- The two BFF workers each have their own copy of `bff_*` counters; a single `/metrics` scrape hits one worker at random and undercounts. For a single-operator deploy this is **acceptable** — document it, don't reach for `prometheus_client` multiprocess mode (a gateway dir, more failure surface) unless a real discrepancy bites. *Right-sizing over correctness-theater.*
- The engine's `wang_trading_*` metrics live in the *engine* process, not the BFF. So "pass `REGISTRY` into `MetricsCollector`" (option A) makes them visible *on the engine's own `/metrics`*, not on the BFF's. **Conclusion: option B (separate scrape target per process) is the structurally honest choice**; option A is the right move *within each process* (kill the private default registry) but does not by itself put engine metrics on the BFF endpoint. Net recommendation: **eliminate the private `CollectorRegistry()` default (use the process-global `REGISTRY`), and give the engine process its own `/metrics` exposition + scrape target.**

#### Per-service `/metrics` exposition contract (version-controlled with the deploy doc)

| Service | Port / path | Registry contents | Scrape interval |
|---|---|---|---|
| BFF (uvicorn, 2 workers) | `:<bff>/metrics` | `bff_requests_total`, `bff_request_duration_seconds`, `bff_cache_*`, `bff_sse_*`, `bff_upstream_errors_total` | 15s |
| Engine (pipeline process) | `:<engine>/metrics` | `wang_trading_*` (NAV, exposure, orders, slippage, signals, model_staleness, drift, breaker) | 15s |

The exposition contract, the scrape interval, and the AlertManager rules **live in version control next to `deploy.sh` / the deploy doc** — not hand-edited on the host. Route-label cardinality is already bounded by template normalization (`_normalise_route`, `metrics.py:122-135`: `/api/v1/trade-ideas/{symbol}`, never per-symbol) — preserve that; it's the difference between a healthy metrics surface and a cardinality explosion.

### 8.4 Latency budgets — define the SLI now, pin the threshold only after §8.3

The histogram exists (`bff_request_duration_seconds`, buckets `0.01…10.0s`, `metrics.py:81-86`) and the middleware records every request with status and route label (`metrics.py:175-191`). So the **SLI is ready**; what's not ready is calling a number an SLO.

**Rule: do not commit a p95 budget per endpoint until the metric has been scraped through a representative run and we've seen the actual distribution.** Pinning "`/trade-ideas` p95 < 120ms" today would be fiction — and worse, the 120ms hot path is currently undermined by the synchronous full-pipeline tmpfs fallback (§6) and by per-request `TradeIdeasService()` construction defeating the 30s regenerate-debounce (§4). **Fix those first, then measure, then pin.** A budget on top of a known-broken hot path just generates noise.

Provisional budgets to *validate against measured data* once §8.3 lands (snapshot-backed reads should be fast; the pipeline-regenerate path is the tail):

| Endpoint class | Backing | Provisional p95 (to confirm) |
|---|---|---|
| `/overview`, `/markets`, `/symbols/{symbol}` | bars + snapshot, cached | ~120 ms |
| `/trade-ideas` (cache/snapshot hit) | tmpfs snapshot | ~120 ms |
| `/trade-ideas` (debounced regenerate) | pipeline run | tail — exclude from the read-path SLO, alert separately if it dominates |

### 8.5 Alerting discipline — alert on freshness, not on un-wired latency

This is the part most likely to be over-engineered. The instinct is a wall of per-endpoint latency alerts. **Resist it.** Per-endpoint latency alerts before §8.3 is fixed would fire on missing data, not on real degradation. Instead, alert on the **freshness SLO**, which is *real today* and is exactly what a single operator needs to trust the screen in front of them.

**Tier 1 — freshness SLO (alert on these now; they're grounded in real signals):**

| Alert | Signal source | Why it's real |
|---|---|---|
| **Snapshot staleness** | `staleness_seconds` vs the snapshot's embedded `as_of` (not mtime) — `TradeIdeasService._last_staleness_seconds`, must be wired into the envelope (§4 bug) | The snapshot is 1 of 2 real data sources; >90s stale means the cycle wedged |
| **Last-bar staleness** | max `as_of` of the `bars` hypertable | Catches *ingestion wedged while process alive* — the crash-only-health blind spot (§6) |
| **Model-loaded** | `ModelRegistry.get_production_model()` present | Distinguishes "no model → `MODEL_REQUIRED`" (expected, not an alert) from "model vanished" (alert) |
| **Circuit-breaker state** | `wang_trading_orders_rejected_total` rate / breaker gauge (once §8.3 scraped) | Breaker tripping is the single most important trading-safety signal |

**Tier 2 — pin after §8.3 + a measured baseline:** per-endpoint p95 (§8.4), cache hit-rate floor (`bff_cache_*`), `bff_upstream_errors_total` rate by `service`.

**Explicitly NOT alerting (over-engineering to avoid):** SLO error-budget burn-rate math, anomaly-detection on latency, multi-window multi-burn-rate alerting, paging escalation tiers. One operator, one pager. A freshness alert and a breaker-state alert delivered to a single channel is the *complete* correct alerting surface for v1.

**Health endpoint split (cross-ref §6):** today health is crash-only — the process can be alive while ingestion is wedged. Split `/healthz` into **liveness** (process responds) vs **readiness** (snapshot age within budget AND model-loaded AND no MV staleness), and surface snapshot age + model-loaded as the readiness payload. The freshness SLO above and the readiness probe are the same signals viewed two ways: the probe gates traffic, the alert pages the operator.

### 8.6 Sequencing (so nothing is built on un-scraped data)

1. **Unify / expose the registries** (§8.3) — eliminate the private `CollectorRegistry()` default; give the engine its own `/metrics` + scrape target. *Blocks everything below.*
2. **Wire `staleness_seconds` into the envelope** (§4 bug) and emit `STALE_*` over threshold — turns the freshness SLI on.
3. **Ship Tier-1 freshness alerts** (§8.5) — these need only steps 1–2, not measured latency.
4. **Close the value-redaction gap** in logging (§8.1).
5. **Measure, then pin** provisional latency budgets (§8.4) and Tier-2 alerts — only after a representative run with metrics actually flowing.

Steps 1–4 are real work with real payoff today. Step 5 is deliberately last and deliberately gated on data, not on a calendar.

---

## 9. Scalability (right-sized) & deployment / DR

This is a **single-operator, single-tenant** system. There is one human reading one BFF backed by two data sources (the `bars` hypertable and one tmpfs `trade_ideas.json` snapshot). "Scalable" here does **not** mean web-scale; it means *clean enough boundaries that a future split is mechanical*, plus *graceful degradation under partial failure*. The most serious infrastructure gap in this whole document is not throughput — it is the **single-host blast radius** (§9.3). Read this section as: spend zero effort on horizontal scale, spend real effort on DR.

### 9.1 Scalability — the explicit DO-NOT-BUILD list

The §2.1 stance (one process, **2 uvicorn workers**, `lru_cache` singletons behind `Depends`) is already correct. Do not regress it into a distributed system. The following are explicitly **out of scope for v1 and the foreseeable future** — building any of them is over-engineering against a single-operator load profile:

| Do **not** build | Why it's wrong here | What we do instead |
|---|---|---|
| Kubernetes / container orchestration | One host, one process. Orchestrating one pod is pure overhead. | systemd unit (`config/systemd/wang-live-trading.service`) + a second unit for the BFF. |
| Service mesh (Envoy/Istio, mTLS sidecars) | Nothing to mesh — services are in-process Python objects. | In-process DI; sign the role header at the nginx→uvicorn hop (§Security). |
| Multi-node / horizontal scale-out | One reader. There is no second node to balance to. | Vertical headroom on the single host. |
| Redis token-bucket rate limiting | One authenticated operator cannot self-DoS. | None. Bind-address + signed role is the only gate. |
| Redis / RQ / Celery job queues | The one async job (snapshot regenerate) is debounced by a 30s in-process LRU. | Keep the in-process debounce; **route it through the `deps.py` singleton** (today routes do `TradeIdeasService()` per-request, which defeats it). |
| Microservice split (engine/BFF/auth as separate deployables) | Adds network hops and serialization to a system with zero scale pressure. | Keep the monolith; preserve boundaries (below). |
| Autoscaling (HPA, scale-to-zero) | Load is flat and human-paced. | Fixed 2 workers. |
| Per-worker shared infra for SSE/jobs | v1 ships no SSE and no mutations. | Defer; if SSE lands, pin it to a single worker or document the replay limitation (§Reliability). |

**"Scalable" = clean boundaries, not capacity.** The one architectural rule that *keeps* a future split mechanical is already enforced and must stay enforced:

- **Zero `fastapi` imports in `src/web/services/*` and `src/*`.** Services take primitives, return DTOs (never DataFrames), and know nothing about HTTP.
- **DTO returns + DI via `Depends` + `lru_cache` singletons**, overridable through `dependency_overrides` in tests.

Because of this, a future engine↔BFF process split would be a *transport swap* (in-process call → gRPC/Unix-socket), not a redesign. **Defer that split until profiling shows real event-loop contention** — not preemptively. There is currently no evidence of contention; the real event-loop risk is a *blocking call*, addressed by the `asyncio.to_thread` lint gate in §Reliability, not by splitting processes.

### 9.2 Performance budgets (right-sized)

Budgets exist to catch regressions, not to chase milliseconds. **Do not pin per-endpoint p95 alerts until the two Prometheus registries are unified** (the pipeline `MetricsCollector` uses its own `CollectorRegistry` at `src/monitoring/metrics.py:44`, disjoint from the global `REGISTRY` the BFF `/metrics` serves) — alerting on a metric nothing populates is worse than no alert.

| Path | Target | Backed by |
|---|---|---|
| Cached read (envelope from warm `LRUTTLCache`) | **< 5 ms** | in-process cache hit, no I/O |
| `/overview`, `/trade-ideas` snapshot hot path (warm snapshot, no regenerate) | **< 120 ms** | tmpfs read + projection only |
| `/markets`, `/symbols/{symbol}` (Wave 1, net-new) | **< 120 ms** typical | `database.get_bars()` with mandatory `LIMIT` + cursor pagination |
| Snapshot regenerate (cold/stale miss) | off the hot path | **must not** run the synchronous full pipeline (see below) |

**Two budget-protecting rules (both are correctness, not optimization):**

1. **Kill the synchronous full-pipeline tmpfs fallback** (the old `§24.5` behavior). Running the GIL-bound pipeline inside a web worker blows the 120 ms budget and stalls both workers. A **missing or stale snapshot returns the last-known snapshot + a `STALE_*` warning** (HTTP 200, non-empty `errors[]`), never a pipeline run. A truly absent snapshot returns `null` data + warning. The engine — not the BFF — owns regeneration.
2. **Wire `staleness_seconds`.** `TradeIdeasService._last_staleness_seconds` is computed today but never read into the envelope. Surface it and emit a `STALE_*` code past threshold so a slow/wedged engine is *visible* rather than silently serving old numbers under budget.

### 9.3 Deployment & disaster recovery — the real work

The single-host blast radius is the highest-severity infra gap in this design. Everything below is "keep it simple" — no backup product, no HA cluster — but the restore **must be rehearsed**, because an unrehearsed backup is a guess.

#### 9.3.1 Process & container model

Deployment is `scripts/deploy.sh` → rsync source into `/opt/wang_trading`, build venv, seed config (no-overwrite), install the systemd unit. The unit (`config/systemd/wang-live-trading.service`) is already hardened: `Type=simple`, `ProtectSystem=strict`, `ProtectHome=true`, `NoNewPrivileges=true`, narrow `ReadWritePaths`, `ConditionPathExists=!.live_halt` (never auto-starts after a halt), `ExecStartPre` runs preflight, `Restart=on-failure` with `StartLimitBurst=3`. **Keep all of it.** The BFF gets its own analogous unit (separate `ExecStart`, same hardening), so a BFF crash never touches the trading process and vice-versa.

#### 9.3.2 Schema migrations — versioned, idempotent, *before* restart (gap)

`deploy.sh` today has **no migration step** (verified: steps 1–8 cover user, dirs, venv, config, systemd, supervisor, logrotate — none touch the DB schema). For a 7-year hypertable this is a latent foot-gun. Add a numbered-SQL or Alembic migration runner that runs **idempotently in `deploy.sh`, before the service restart**:

```bash
# deploy.sh, new step BEFORE any service (re)start:
echo "==> Applying DB migrations (idempotent)"
sudo -u "${SERVICE_USER}" "${INSTALL_DIR}/venv/bin/python" -m src.data_engine.migrate --up
#   - each migration guarded (IF NOT EXISTS / version table); re-running is a no-op
#   - non-zero exit ABORTS the deploy; the old process keeps running on the old schema
```

Migrations must be backward-compatible for one release (no-flag-day / expand-then-contract), so a rollback to the previous binary still reads the new schema.

#### 9.3.3 Offsite backups + a TESTED restore (the serious gap)

Recovery state today is **30-day local pickles on the same disk as everything else** (`src/execution/disaster_recovery.py`: `SNAPSHOT_DIR = Path("logs/snapshots")`, `SnapshotManager(retention_days=30, interval_seconds=300)`, `snapshot_*.pkl`). The `bars` hypertable lives on the same host. **A single-disk failure loses both the database and the recent recovery state.** Fix with the simplest thing that survives losing the host — cron + `rclone`, no backup appliance:

```cron
# Daily logical dump + off-host copy (single operator, keep it dumb-simple)
15 4 * * *  pg_dump --format=custom wang_trading \
              > /opt/wang_trading/backups/wang_$(date +\%F).dump && \
            rclone copy /opt/wang_trading/backups        remote:wang-backups/db && \
            rclone copy /opt/wang_trading/logs/snapshots remote:wang-backups/snapshots
```

- For a 7-year hypertable, prefer **continuous archiving (WAL-G / `pg_receivewal`)** over a nightly full dump once the dump time grows uncomfortable — but ship the dump first; don't block on the fancier option.
- The `.pkl` snapshots and DB dumps both go off-host **on a different failure domain** (S3/B2/another machine).
- **Rehearse the restore on a clean box on a schedule** (e.g. monthly): pull the dump, restore into a scratch DB, verify row counts and the latest `bars` timestamp, and `StateSnapshot.load(..., verify=True)` a recent snapshot (it is checksummed — `compute_checksum` — so tampering/corruption fails loudly). A restore you have never run is not a backup.

#### 9.3.4 Snapshot & crash-flag durability — durable disk, not tmpfs (gap to confirm)

The trade-idea snapshot is intentionally on **tmpfs** (`/run/wang/trade_ideas.json`, atomic `write→fsync→os.replace` per `trade_idea_publisher.py`). That is correct for the *projection* — it's a derived artifact the engine rewrites each cycle and losing it on reboot is fine (the engine regenerates it).

What must **not** live on tmpfs is **recovery state**: the `SnapshotManager` pickles and the `.live_crash` / `.live_halt` flags. Today `disaster_recovery.py` uses relative `Path(".live_crash")` / `Path(".live_halt")` and `logs/snapshots`, which resolve under the systemd `WorkingDirectory=/opt/wang_trading` (durable disk) — **good, but the unit sets `PrivateTmp=true`**, so any code that drops recovery state under `/tmp` would land in a per-invocation tmpfs that vanishes on restart and defeats crash detection. **Action:** pin recovery-state paths to absolute durable-disk locations under `/opt/wang_trading/logs` (in `ReadWritePaths`), and add a deploy-time assertion that `SNAPSHOT_DIR` and the crash/halt flags are not on a `tmpfs` mount. Snapshot age itself becomes a freshness SLO signal (§Reliability).

#### 9.3.5 NTP / chrony — a hard preflight blocker

The system relies on **HMAC-chained timestamps and a time-ordered audit chain**. Clock drift is not a nuisance here — it is a *correctness* risk that can invalidate the chain. Add a chrony/NTP sync check to `src.execution.preflight` as a **blocker** (not a warning): if the host clock is unsynchronized or drift exceeds a tight bound, preflight fails and `ExecStartPre` refuses to start the service. This costs a few lines and closes a silent-corruption class.

#### 9.3.6 Degraded-mode contracts (make them explicit and testable)

State the failure posture for each dependency so behavior under partial failure is a *contract*, not an accident:

| Dependency fails | Required behavior |
|---|---|
| **DB write** (engine cannot persist bars/state) | **Halt trading** (fail-closed). Correctness over availability. |
| **In-process / cache loss** | **Degrade to source of truth** (re-read snapshot / DB), never crash. Cache is an optimization. |
| **Snapshot stale or missing** (BFF) | Serve last-known + `STALE_*` warning (200); never trigger a pipeline run. |
| **MLflow / S3 / model registry unreachable** | **Do not block trading.** Model-gated fields return `MODEL_REQUIRED` + null; trading and the LIVE endpoints carry on. |
| **Mutations** (when they eventually land) | **Fail-closed** if the idempotency store is unavailable; do not silently proceed. |

#### 9.3.7 Runbooks

Operational procedures are operator-facing and live in `docs/deployment.md` (referenced by the systemd unit's `Documentation=`) and the `deploy.sh` post-install instructions block — **emergency stop** (`systemctl stop`), **emergency flatten** (`live_trading --emergency-flatten`), **verify-flat**, preflight, and the restore drill above. This section is the *design*; the step-by-step is the runbook. Keep them cross-linked so the on-call (you) is never reconstructing the restore sequence during an incident.

---

## 10. Build roadmap & locked decisions

The build plan is **wave-gated against engine persistence**, not against frontend appetite. The rule is mechanical: a route ships in the wave where its data already exists on the **single real data spine** — the `bars` hypertable (`src/data_engine/storage/database.py`) and the tmpfs `trade_ideas.json` snapshot (`src/execution/trade_idea_publisher.py`). Anything backed by neither stays a client-side `ComingState` panel until its named net-new persistence gate lands. We never mount an endpoint that synthesizes a number for a field the engine doesn't produce.

### 10.1 Wave 1 — build now (all data already persists)

Wave 1 is fully unblocked: every output below is a projection of `bars` or the snapshot. No engine change is required to ship it.

#### 10.1.1 The two net-new routes

| Route | Backed by | Net-new BFF work |
|-------|-----------|------------------|
| `GET /markets` (#3) | `database.get_bars()` per symbol + snapshot join for `hasIdea` + static instrument-ref map | New `markets` router (not mounted today, `app.py:203-209`) producing `Sym[]`: symbol, name, type, price, `change{1d,1w,1m,ytd}`, spark/line/candles (OHLCV), volume, `bar` (BarMicro). **Drop `marketCap` from the type — no producer.** |
| `GET /symbols/{symbol}` (#4) | `database.get_bars(symbol)` + snapshot lookup for the matching `TradeIdea` | New `symbols` router producing `SymbolDetail{sym, idea\|null}`. LIVE centerpiece = **real bar microstructure** from the hypertable. Feature-factory features, per-symbol SHAP, regime_fit, cost, track-record all render `ComingState` — not served. |

Both routes depend on a net-new **static instrument-ref map** `symbol → {name, asset_class}`, because `bars` carries only `symbol + bar_type`. Ship the map *with* the markets route; an unknown symbol degrades to ticker-as-name + default tint, never an error.

#### 10.1.2 BFF bug fixes (block correctness of routes already mounted)

These are not features — they are defects in the two REAL routes (`/overview`, `/trade-ideas`) and must land in Wave 1 alongside the new routes.

| Fix | Defect (verified) | Action |
|-----|-------------------|--------|
| **Route through `deps.py` singletons** | Routes do `TradeIdeasService()` per-request → fresh `TmpfsTradeIdeasCache` + LRU each call → defeats the 30s regenerate-debounce and makes `dependency_overrides` inert | Inject via `Depends` against the `deps.py` singleton providers |
| **Typed enveloped errors** | `/trade-ideas` and `/replay` raise raw `HTTPException(500, detail=str(exc))` (`trade_ideas.py:38`) — leaks exception text, bypasses the envelope | Replace with typed `ApiException` subclasses → enveloped `errors[]` switched on `code` |
| **Wire staleness** | `TradeIdeasService._last_staleness_seconds` is computed but never read into the envelope | Thread it into `staleness_seconds`; emit a `STALE_*` warning (HTTP 200) over threshold |

#### 10.1.3 The cheap real wins (data exists or is one caller change away)

- **Wire `/preflight` to the real engine.** Replace the `overall=UNKNOWN` single-stub check with `src.execution.preflight` + `infra_probe`. Runnable now, requires **no persistence** — the result is computed live, not stored.
- **Thread `db_manager` into `predict_proba`.** The live cycle calls `predict_proba` *without* `db_manager`, so `insert_meta_label` never fires and no `meta_labels` rows are written. This is a **one-line caller change** that starts a real `meta_labels` series — the seed the `/model` `metaProbHist` will later read.

### 10.2 Wave 2 — model-gated reads (need a registered MLflow production model)

Wave 2 unlocks the moment a production model is registered in `ModelRegistry`. Until then these routes return `action=MODEL_REQUIRED` and null probabilities — they do not 404 and do not fabricate.

| Route | LIVE output when model present | Gated / deleted |
|-------|-------------------------------|-----------------|
| `GET /model` (#7) | version, trainedAt, lastRetrainHours, runId, type, cvScore, trainAcc, trainingEvents (`ModelRegistry.get_production_model()` + MLflow); `metaProbHist` (calibrated_prob buckets — render **Empty** not Coming when no rows); `gates{cpcv,dsr,pbo}` → neutral "not run"; `retrainTimeline[]` (MLflow `search_runs`). No model → whole screen Empty, `model_version=null`. | **Delete, never serve real:** auc, brier, ece, calibration[], featureImportance[], drift[], rlShadow |
| `GET /signals/families` (#5) | `Strategy[]` (all 10): id/name/category/source/thesis/params/assetClasses (static LIVE) + active idea count from snapshot. Client derives 4 active / 6 inactive from `FAMILY_READINESS`. | **Delete, never serve real:** sharpe, winRate, trades, contributionPct, pnlYtd, allocation, equityCurve, regimeFit, avgHoldBars |
| `GET /signals/family-{id}` (#6) | `{strategy, ideas[]}` — this family's ideas this cycle (LIVE from snapshot); reachable for dormant families with status + reason | per-strategy Sharpe/win/PnL/equity → `ComingState` |

#### 10.2.1 The meta/calibrated probability pair is a REAL split — expose it

The brief corrects the older audit: the meta-vs-calibrated split is **already real on the model path**. `bootstrap.py:309` calls `self.model.predict_proba(X, return_raw=True)` and unpacks `(meta_prob, cal_prob)` when the model returns a 2-tuple. Critically, these two fields are **already passed through** (`trade_ideas_service.py:356-357`) — they are *not* in the hardwired-null block at `:358-367`.

The contract therefore exposes both as a **genuine pair**, with this discipline:

```
when a production model is loaded AND returns the (raw, calibrated) tuple:
    meta_probability   = proba[0][0]   # raw meta-label score
    calibrated_probability = proba[1][0]   # post-calibration
when paper-mode ConfidenceMetaPipeline fallback OR plain/pyfunc model:
    meta_probability == calibrated_probability   # collapse-to-equal (bootstrap.py:316-317)
when no production model:
    both null, action = MODEL_REQUIRED
```

Gate the pair behind `MODEL_REQUIRED`; collapse-to-equal in paper mode is honest (it reflects the engine genuinely not producing a distinct raw score), not a synthesized number.

### 10.3 COMING domains → exact engine persistence gate & wave

Every COMING screen renders a locked `ComingState` panel client-side from `readiness.ts`. It needs **no endpoint in v1**. Each unlocks only when its named net-new *engine persistence* lands — BFF work alone cannot move any of these.

| Domain / field | Net-new engine persistence gate | Wave |
|----------------|----------------------------------|------|
| **Portfolio** (`nav, daily_pnl, drawdown, gross/net_exposure, positions_count`) | A persisted portfolio / positions store (none exists; `/overview` returns null + standing warning today) | 5 |
| **Execution** | Order-routing writes `ExecutionStorage` (the deployed path sends 0 orders → 0 rows) | 5 |
| **Backtests** | Backtest-run persistence (no run history written) | 5 |
| **Scenarios** | `factor_risk` wired into `ScenarioService` (not even imported today; `/scenarios/run` is a self-identified MOCK) | 5 |
| **Track Record** (`track_record_win_rate`, `track_record_n`) | Append-only call-history store — the snapshot is overwritten every cycle, so history must be appended elsewhere | 5–6 |
| **Monitoring** | Unify the two Prometheus registries first: `MetricsCollector` uses its own `CollectorRegistry` (`metrics.py:44`); BFF `/metrics` serves the disjoint global `REGISTRY` → nothing scrapes the pipeline. No Monitoring page can be backed until they're unified | 5 |
| **Replay** | Audit-chain writer — `ComplianceAuditLogger` is **never instantiated** → 0 audit rows (`/replay` is an empty-snapshot STUB) | 5–6 |
| **`regime`** (overview + per-idea) | `RegimeDetector` has **zero runtime callers** — invoke it in the cycle and persist its output | 6 |
| **`expected_cost_bps` / cost** | Build `CostForecastService` (no producer) | 5–6 |
| **`top_shap_feature` / SHAP** | Call + persist `shap_importance` — `FeatureStore.save_features` has **zero callers** | 4–6 |
| **drift fields** | `DriftDetector.set_baseline` has **zero callers** → drift is inert (hardcoded 1.0, empty baseline). Wire `set_baseline` from the retrain/training path first | 4–6 |
| **`regime_fit_score`** | `SignalsService` attribution (no producer) | 5–6 |
| **`sizing_constraints_applied`** | Surface the cascade `constraints_applied` through `ui/trade_ideas.py` (currently hardwired `[]` at `trade_ideas_service.py:361`) | 4–5 |

`GET /trade-ideas/{symbol}` (#8) stays a **501 stub** and is **non-required**: the drawer hydrates from the list row, so there is no detail round-trip in v1. It must never 500 the client.

### 10.4 THE LOCKED DECISIONS

These are non-negotiable for the design phase. They restate the brief's §7 as build directives.

1. **The 8 endpoints in §1 are the authoritative contract.** Rewrite `api_contracts_v2.md` to match — tier every path BUILT / PARTIAL / PLANNED by wave, add the missing `/overview` section, and delete the ~37 speculative endpoints from any "current contract" framing. The frontend `source:` strings *are* the contract; we rewrite the doc to the frontend, not the frontend to the doc.

2. **Casing: a Pydantic camelCase alias generator on the BFF DTOs.** The frontend binds camelCase (`targetWeight`, `stageLatency`, `metaProbability`); `model_dump` emits snake_case. Fix it in **one place** (the alias generator) — not with a per-field adapter scattered in `data/api.ts`.

3. **Freeze the envelope.** `ApiEnvelope<T>` is locked as a pair across `envelope.py ↔ envelope.ts ↔ iOS`. Add a **cross-language field-parity test** (including `request_id`) and stamp **`schema_version` + `as_of`** onto the tmpfs `TradeIdeasResponse` JSON with a contract test across the engine→BFF boundary — it is one of only two real data sources and the highest-leverage failure point.

4. **DI, errors, staleness.** Route through `deps.py` singletons; replace every raw `HTTPException(500)` with typed enveloped errors switched on `code`; wire `staleness_seconds` into the envelope and emit `STALE_*` (HTTP 200) over threshold.

5. **Close the Phase-1 auth hole even VPN-only.** Sign/HMAC the `X-Role` header between nginx and uvicorn (or resolve roles inside uvicorn) — an unsigned forwarded header lets anything reaching uvicorn claim any role. **Fail-closed** on mutations when they land (closed if the idempotency store is unavailable). **Tighten CORS** from `allow_origins=['*']` via `config/web.yaml` for non-dev.

6. **Unify the two Prometheus registries before pinning any budget or alert.** Pass the global `REGISTRY` into `MetricsCollector` (or scrape both). Do **not** pin per-endpoint p95 alerts until metrics are actually wired — alerting on an unscraped registry is alerting on silence.

7. **Build wave-gated; never synthesize an absent field.** Wave 1 = #3, #4 (net-new on `bars`) plus the #1/#2 fixes and cheap real wins → Wave 2 = #5, #6, #7 (model-gated reads). Everything else stays `ComingState` until its named engine persistence gate lands. When data is absent the answer is always `null` + a `warning` — never a placeholder-as-number.

> **Over-engineering to explicitly avoid (single-operator / single-tenant):** no Kubernetes, no service mesh, no multi-node, no Redis token-bucket rate limiting, no Redis/RQ job queues, no microservice split, no autoscaling. The single-process / 2-worker stance is correct. "Scalable" here means clean service boundaries (zero FastAPI imports in services, DTO returns, DI) so a *future* split is mechanical — not a current build target. Defer any process-split over gRPC/Unix-socket until profiling shows real contention, not preemptively.

---

## 11. Data flow & runtime services (engine → API)

The 8-endpoint surface above is the *read* contract. This section is the *runtime*: how the engine's output reaches the BFF, and the processes that keep it flowing. Built + verified 2026-06-20. Right-sized for one host — **one atomic file + one DB, no message bus / Redis / queue**.

### 11.1 Three data planes

```
                 ┌─────────────────────────── engine (writers) ──────────────────────────┐
  Plane A ideas: live cycle (run_cycle) ──post-cycle hook──▶ trade_ideas.json (tmpfs, atomic)
  Plane B bars:  data_ingestion runner ──insert_bars──────▶ TimescaleDB `bars` hypertable
  Plane C model: retrain scheduler ──promote `production`──▶ MLflow registry
                 └────────────────────────────────────────────────────────────────────────┘
                                            │ read-only, never writes
                 ┌──────────────────────────▼ BFF (src/web) ─────────────────────────────┐
  Plane A read:  TradeIdeasService.read_snapshot  → /overview /trade-ideas  (sub-ms, tmpfs)
  Plane B read:  BarsGateway (parameterized+LIMIT) → /markets /symbols       (~ms)
  Plane C read:  ModelService (singleton registry + 60s TTL cache) → /model  (<5ms warm)
                 └──────────────────────────┬────────────────────────────────────────────┘
                                            ▼  ApiEnvelope<T>  →  frontend (live or mock)
```

### 11.2 Plane A — the trade-ideas bridge (primary + fallback)

The snapshot is the BFF's primary read; keeping it fresh is the bridge.

- **Primary — in-cycle post-cycle hook.** `PaperTradingPipeline.run_cycle` (step 12) builds a `TradeIdeaReport` from the artifacts it *already computed this tick* (`build_report_from_cycle` reuses the same `_idea_from_artifacts` assembly the read-only rehearsal uses) and hands it to a guarded `report_sink` = `TradeIdeaPublisher.publish_report`. Trade ideas are computed **exactly once**, in the live cycle. The write is atomic (`write → fsync → os.replace`); the hook is fully `try/except`-guarded so a publish failure can never affect trading. Wired in `live_trading.main` (opt out `--no-publish-trade-ideas`).
- **Fallback — standalone publisher daemon** (`config/supervisor/trade_idea_publisher.conf`, `autostart=false`). Re-runs the read-only pipeline every 60s and writes the same file. For paper-only viewing when no live cycle is running. Must NOT run alongside the live cycle (double compute).

The writer and reader agree on **one path** — `WANG_TRADE_IDEAS_PATH=/run/wang/trade_ideas.json` — set in the live-trading unit/conf and `bff.conf`. On reboot tmpfs is empty; `tmpfiles.d` recreates `/run/wang` and the next cycle republishes within one period (the BFF `/readyz` 503s honestly until then — no durable-disk mirror, the 90s self-heal is sufficient).

### 11.3 Runtime services (supervisor)

`scripts/deploy.sh` installs these (engine daemons + the bridge + the API):

| Service | Role | autostart |
|---|---|---|
| `wang_live_trading` | the cycle — trades **and** publishes the snapshot (hook) | manual (post-preflight) |
| `wang_data_ingestion` | Alpaca feed → `bars` hypertable (Plane B) | yes |
| `wang_retrain_scheduler` | promote MLflow `production` model (Plane C) | yes |
| `wang_bff` | read-only API (`uvicorn`, localhost, read-only DB role) | yes |
| `wang_trade_idea_publisher` | snapshot **fallback** (paper-only) | **no** |

The systemd live-trading unit is sandboxed (`ProtectSystem=strict`); `/run/wang` is in `ReadWritePaths` so the cycle can write the snapshot.

### 11.4 Freshness SLO & low-latency

- **Health split:** `/livez` (process up, always 200) · `/healthz` (200 + the freshness vector + envelope `source_freshness`) · `/readyz` (**503** when the snapshot is absent or stale > 90s — the bridge isn't delivering). Prometheus gauges `bff_snapshot_age_seconds` / `bff_last_bar_age_seconds` / `bff_model_loaded` (`-1` = unavailable, so a dead feed flat-lines and alerts).
- **Latency budget held:** snapshot read **< 2 ms** (tmpfs is RAM); bars query **< 20 ms** (hypertable + bounded `LIMIT`, pooled connection); model card **< 5 ms warm** — the MLflow registry is built **once** and the result TTL-cached (`LRUTTLCache`), eliminating the ~1 s per-request round-trip.
