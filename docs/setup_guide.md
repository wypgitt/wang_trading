# Setup Guide — wang_trading

A complete, copy/paste-friendly walkthrough for standing this repo up from
scratch: local dev → paper trading → production live trading.

This document is **prescriptive**: it tells you exactly what to install,
what to copy, what to set, and what to run, in order. For deeper design
background, see [architecture_overview.md](architecture_overview.md); for
the go-live checklist, see [go_live_checklist.md](go_live_checklist.md).

---

## Contents

1. [What this repo is](#1-what-this-repo-is)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Clone and install Python dependencies](#3-step-1--clone-and-install-python-dependencies)
4. [Step 2 — Start infrastructure (TimescaleDB, Redis, Prometheus, Grafana)](#4-step-2--start-infrastructure-timescaledb-redis-prometheus-grafana)
5. [Step 3 — Copy and edit config files](#5-step-3--copy-and-edit-config-files)
6. [Step 4 — Environment variables](#6-step-4--environment-variables)
7. [Step 5 — API keys and secrets](#7-step-5--api-keys-and-secrets)
8. [Step 6 — Initialise the database](#8-step-6--initialise-the-database)
9. [Step 7 — Verify the install (smoke tests)](#9-step-7--verify-the-install-smoke-tests)
10. [Step 8 — Use the system locally](#10-step-8--use-the-system-locally)
11. [Step 9 — Monitoring (Prometheus + Grafana)](#11-step-9--monitoring-prometheus--grafana)
12. [Step 10 — Production deployment (Linux host)](#12-step-10--production-deployment-linux-host)
13. [Step 11 — Going live](#13-step-11--going-live)
14. [Day-to-day operations](#14-day-to-day-operations)
15. [Troubleshooting](#15-troubleshooting)
16. [Appendix A — Environment variables reference](#16-appendix-a--environment-variables-reference)
17. [Appendix B — Makefile targets](#17-appendix-b--makefile-targets)
18. [Appendix C — File and directory layout](#18-appendix-c--file-and-directory-layout)
19. [Appendix D — Where to look next](#19-appendix-d--where-to-look-next)

---

## 1. What this repo is

`wang_trading` is a multi-strategy, multi-asset quantitative trading
engine. It is organised in six phases; each phase is a self-contained
subsystem under `src/`:

| Phase | Subsystem | Directory |
|---|---|---|
| 1 | Data foundation (bars, ingestion, storage, validation) | `src/data_engine/` |
| 2 | Feature factory + signal battery | `src/feature_factory/`, `src/signal_battery/` |
| 3 | Labeling + ML (meta-labeler, regime, RL) + bet sizing | `src/labeling/`, `src/ml_layer/`, `src/bet_sizing/` |
| 4 | Backtesting + portfolio construction | `src/backtesting/`, `src/portfolio/` |
| 5 | Execution + paper trading + monitoring | `src/execution/`, `src/monitoring/` |
| 6 | Live capital, RL portfolio agent, production hardening | (additions across all of the above) |

You can run the pipeline at three levels of risk:

- **Dev / research** — no brokers, no real money; mostly for writing
  features/signals, training models, running backtests.
- **Paper trading** — full pipeline with a simulated broker
  (`PaperBroker`) or a broker's paper endpoint (Alpaca paper, Binance
  testnet, IBKR paper account).
- **Live trading** — real capital. Gated by explicit env-var switches
  (`WANG_ALLOW_LIVE_*`), a systemd preflight, an operator check-in
  sentinel, and a 4-phase capital ramp.

---

## 2. Prerequisites

### Operating system

- **Local dev:** macOS 12+ or Linux (Ubuntu 22.04 LTS tested). Windows
  is not supported — use WSL2 if you are on Windows.
- **Production:** Ubuntu 22.04 LTS on a host with ≥ 8 CPU cores, ≥ 32 GB
  RAM, ≥ 500 GB SSD. See [deployment.md](deployment.md) for hardware sizing.

### Software you must have before starting

| Tool | Version | How to install |
|---|---|---|
| Python | 3.11+ | `brew install python@3.11` (macOS) / `apt-get install python3.11 python3.11-venv python3-dev` (Ubuntu) |
| Docker + Docker Compose v2 | latest stable | [docker.com/get-started](https://docs.docker.com/get-docker/) |
| git | any modern | `brew install git` / `apt-get install git` |
| make | GNU make 3.81+ | pre-installed on macOS/Linux |
| Build tools (Linux) | gcc, libpq-dev, libssl-dev, libomp-dev | `sudo apt-get install build-essential libpq-dev libssl-dev libomp-dev` |

### Check prerequisites

```bash
python3 --version        # Must print 3.11.x or later
docker --version
docker compose version   # Must be the v2 plugin (note: "docker compose", not "docker-compose")
git --version
make --version
```

### Services you'll run locally via Docker

These are **not** prerequisites — the repo starts them for you via
`docker-compose.yaml`:

- TimescaleDB (PostgreSQL 16 + TimescaleDB extension) on port 5432
- Redis 7 on port 6379
- Prometheus on port 9090
- Grafana on port 3000

### External accounts (only needed for the features you use)

| Provider | What it's for | Required when |
|---|---|---|
| [Alpaca](https://alpaca.markets) | US equities data + brokerage | You want real equities data or to paper/live-trade equities |
| [Polygon.io](https://polygon.io) | Tick-level US equities data (optional) | You want tick-level supplements beyond Alpaca |
| [Binance](https://www.binance.com/) / [Coinbase](https://www.coinbase.com/) / Kraken / Bybit | Crypto data + brokerage (via CCXT) | You want crypto data or to trade crypto |
| [IBKR](https://www.interactivebrokers.com/) | Futures brokerage | You want to trade futures |
| [Telegram bot](https://core.telegram.org/bots) | Alerts | You want alert notifications (optional) |
| [MLflow server](https://mlflow.org/) | Model registry | Optional — can also run locally |

**Start with Alpaca paper.** It is free, works for equities data and
paper trading, and is enough to exercise the whole pipeline end-to-end.

---

## 3. Step 1 — Clone and install Python dependencies

```bash
# Clone (if you haven't already)
git clone <your-fork-or-this-repo-url> wang_trading
cd wang_trading

# Create and activate a virtualenv — do NOT skip this
python3 -m venv venv
source venv/bin/activate

# Install dependencies + scaffold config/logs/data dirs in one go
make setup
```

`make setup` runs:

```
pip install -r requirements.txt
cp -n config/settings.example.yaml config/settings.yaml
mkdir -p logs data/features
```

It's idempotent — safe to re-run. It will **not** overwrite an existing
`config/settings.yaml`.

### On macOS you may need this one extra env var

Some ML packages (LightGBM, XGBoost) ship with OpenMP and collide with
NumPy's bundled OpenMP on macOS. Set this in your shell profile
(`~/.zshrc` or `~/.bash_profile`):

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

The test suite and systemd unit already set this automatically; you only
need it in your interactive shell.

---

## 4. Step 2 — Start infrastructure (TimescaleDB, Redis, Prometheus, Grafana)

Everything you need for local dev is declared in `docker-compose.yaml`:

```bash
# Start all four services in the background
make db-up              # wraps: docker compose up -d
# or just the data layer:
docker compose up -d timescaledb redis

# Verify
docker compose ps
```

You should see four healthy containers: `quant-timescaledb`,
`quant-redis`, `quant-prometheus`, `quant-grafana`.

### Default local credentials (dev only)

These are the dev defaults baked into `docker-compose.yaml`. They are
safe for local use and **must be overridden for production**:

| Service | User / password | Port |
|---|---|---|
| TimescaleDB | `quant` / `password` | 5432 |
| Grafana admin | `admin` / `admin` | 3000 |

### Stopping services

```bash
make db-down            # docker compose down — keeps data volumes
docker compose down -v  # also wipes the data volumes (DESTRUCTIVE)
```

---

## 5. Step 3 — Copy and edit config files

The repo ships five `.example` files under `config/`. You copy each to
the non-example filename and edit. None of the non-example files are
committed (they live in `.gitignore`), so your keys stay private.

| Example file | Copy to | When |
|---|---|---|
| `config/settings.example.yaml` | `config/settings.yaml` | **Always** — auto-copied by `make setup` |
| `config/paper_trading.example.yaml` | `config/paper_trading.yaml` | Before paper trading |
| `config/live_trading.example.yaml` | `config/live_trading.yaml` | Before live trading |
| `config/futures_contracts.example.yaml` | `config/futures_contracts.yaml` | Only if trading IBKR futures |
| `config/prometheus.yml` | (leave as-is) | — |

### `config/settings.yaml` — master config

Holds the universe, data sources, database connection, bar parameters,
and feature-store location. The shipped example uses `${ENV_VAR}` syntax
so secrets stay in env vars rather than this file. Minimum edits:

```yaml
database:
  host: "localhost"         # or the Postgres host
  port: 5432
  name: "quantsystem"
  user: "quant"
  password: "${DB_PASSWORD}"  # keep as env var, or replace with literal

data_sources:
  alpaca:
    api_key: "${ALPACA_API_KEY}"
    secret_key: "${ALPACA_SECRET_KEY}"
    base_url: "https://paper-api.alpaca.markets"   # PAPER by default
    feed: "iex"              # "iex" is free; "sip" is paid
```

For a full walkthrough of every section in this file, see the
annotations in `config/settings.example.yaml` itself.

### `config/paper_trading.yaml` — paper-trading runtime

Used by `python -m src.execution.paper_trading`. Minimum edits:

- `asset_class`: `equities | crypto | futures`
- `symbols`: the tickers you want to trade
- `broker.initial_cash`: starting virtual capital
- `data.api_key`/`data.api_secret`: set to `env:ALPACA_API_KEY` etc. to pull from env vars

### `config/live_trading.yaml` — live-trading runtime

Used by `python -m src.execution.live_trading`. **Do not edit this until
you have paper-traded successfully.** See
[§13 — Going live](#13-step-11--going-live).

### `config/prometheus.yml`

Used by the Dockerised Prometheus to scrape the trading engine's
metrics port (default 9091). Leave it alone unless you change the
metrics port.

---

## 6. Step 4 — Environment variables

The repo reads env vars in three situations:

1. **Credentials referenced from YAML via `${VAR}` interpolation.**
   `src/config/settings.py:_interpolate_env_vars()` expands these at
   load time.
2. **Switches read directly by Python code** — e.g. the live-trading
   safety gates.
3. **Runtime knobs** set by systemd/supervisor units.

### Minimum set for local dev (paper trading with Alpaca)

Put these in your shell profile or a `.env` you `source` before running:

```bash
# Alpaca (paper)
export ALPACA_API_KEY="PK..."
export ALPACA_SECRET_KEY="..."

# Database
export DB_PASSWORD="password"        # matches docker-compose.yaml default

# macOS only — OpenMP duplicate-library guard
export KMP_DUPLICATE_LIB_OK="TRUE"
```

### Additional env vars by use case

- **Trading crypto:** `BINANCE_API_KEY`, `BINANCE_SECRET_KEY`
- **Polygon tick data:** `POLYGON_API_KEY`
- **Telegram alerts:** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- **GCS-backed feature store:** `GCS_FEATURE_BUCKET` + `GOOGLE_APPLICATION_CREDENTIALS`
- **File-based secrets backend:** `WANG_SECRETS_BACKEND=file`, `WANG_MASTER_KEY=<fernet key>`
- **Custom config path:** `QUANT_CONFIG=/path/to/settings.yaml`

### The three live-trading safety gates

These are the switches that let real money move. They are **separate on
purpose** so you can go live on one asset class at a time.

| Variable | Enables | Read by |
|---|---|---|
| `WANG_ALLOW_LIVE_TRADING` | Alpaca equities live orders | `src/execution/broker_adapter.py:AlpacaBrokerAdapter` |
| `WANG_ALLOW_LIVE_CRYPTO` | CCXT crypto live orders | `src/execution/broker_adapter.py:CCXTBrokerAdapter` |
| `WANG_ALLOW_LIVE_FUTURES` | IBKR futures live orders | `src/execution/ibkr_adapter.py:IBKRBrokerAdapter` |

**Any value other than empty string enables the gate.** Leave them unset
for paper trading.

See [Appendix A](#16-appendix-a--environment-variables-reference) for
the complete reference.

---

## 7. Step 5 — API keys and secrets

There are two ways to provide API keys to the running system:

### Option A — env vars (good for local dev)

Set `ALPACA_API_KEY` etc. in your shell. The YAML files read them via
`${VAR}` interpolation. This is what `make setup` + the defaults in
`config/settings.example.yaml` expect.

### Option B — the SecretsManager (recommended for production)

`src/config/secrets.py` implements a pluggable secrets backend chosen by
the `WANG_SECRETS_BACKEND` env var:

| Backend value | Where secrets live | When to use |
|---|---|---|
| `env` *(default)* | Process env vars (`WANG_SECRET_<NAME>`) | Dev |
| `file` | Fernet-encrypted `config/secrets.enc` | Single-host prod |
| `aws` | AWS Secrets Manager | AWS prod |
| `gcp` | GCP Secret Manager | GCP prod |

#### Bootstrapping the encrypted-file backend

```bash
# 1. Generate a master key (keep this somewhere safe — losing it
#    means you cannot decrypt your secrets)
python scripts/setup_secrets.py --generate-master-key
# → prints a Fernet key like  gAAAAABh...

# 2. Export the master key + switch the backend
export WANG_MASTER_KEY="<the key from step 1>"
export WANG_SECRETS_BACKEND="file"

# 3. Seed the secrets interactively (you'll be prompted for each)
python scripts/setup_secrets.py
```

The script seeds these keys (skip any you don't need by pressing Enter):

```
alpaca.api_key
alpaca.secret_key
binance.api_key
binance.secret_key
coinbase.api_key
coinbase.secret_key
coinbase.passphrase
ibkr.account_id
telegram.bot_token
telegram.chat_id
```

They are Fernet-encrypted and written to `config/secrets.enc` (which is
git-ignored). Broker adapters (`AlpacaBrokerAdapter`, `CCXTBrokerAdapter`,
`IBKRBrokerAdapter`) read from the active backend at startup and fall
back to `config/settings.yaml` if a key is missing.

### Where to get keys (quick links)

- **Alpaca paper keys:** log in → Paper Trading → *View API Keys* → generate
- **Binance testnet:** [testnet.binance.vision](https://testnet.binance.vision/) → Generate HMAC_SHA256 Key
- **Polygon:** free tier at [polygon.io](https://polygon.io/) → dashboard → API keys
- **Telegram bot:** chat with [@BotFather](https://t.me/BotFather) → `/newbot` → receive token; then send a message to your bot and fetch `chat_id` from `https://api.telegram.org/bot<TOKEN>/getUpdates`

**Never commit keys.** The `.gitignore` excludes `config/settings.yaml`,
`config/live_trading.yaml`, `config/paper_trading.yaml`, and
`config/secrets.enc` — but double-check before you push.

---

## 8. Step 6 — Initialise the database

Once Docker is up and `DB_PASSWORD` is exported:

```bash
make db-setup          # == python scripts/setup_db.py
```

This connects to TimescaleDB and creates the full schema:

- **Hypertables** (time-partitioned): `raw_ticks`, `bars`, `features`, `signals`
- **Regular tables:** `cusum_events`, `orders`, `fills`, `tca_results`,
  `portfolio_snapshots`, `model_predictions`, `audit_log`

It's safe to re-run — `CREATE TABLE IF NOT EXISTS`. If you need to wipe
and start over:

```bash
make db-reset          # DESTRUCTIVE: drops the five main hypertables
```

There are **no Alembic-style migrations** — the schema is declared in
code (`src/data_engine/storage/database.py`). If you pull a newer commit
that changes the schema, run `make db-setup` again; additive changes
apply cleanly. For destructive changes, see the release notes.

---

## 9. Step 7 — Verify the install (smoke tests)

Before you try to ingest data or trade, run the two built-in smoke tests:

```bash
# 1. End-to-end Phase 5 smoke — features → signals → meta → sizing
#    → optimizer → execution → monitoring. Uses synthetic data, no
#    external services. Finishes in < 10 s.
make smoke-test

# 2. Production-hardening smoke — preflight, deployment controller,
#    audit log, snapshots, recovery, alerts. Finishes in < 10 s.
python scripts/production_smoke_test.py

# 3. Design-doc conformance audit (needs ≥ 95 % to exit 0)
make audit-conformance
```

If all three pass, your install is good.

### Run the unit tests

```bash
make test               # fast — excludes integration
make test-integration   # slower — full Phase 2/3/4 pipelines
make test-cov           # with coverage report
```

Expect **> 1000 unit tests and ≈ 28 integration tests** to pass.

---

## 10. Step 8 — Use the system locally

### Ingest real data

```bash
# Real-time streaming (runs until Ctrl+C)
make run-equities       # Alpaca equities
make run-crypto         # Binance / Coinbase crypto

# Historical backfill
make backfill SYMBOL=AAPL DAYS=30
```

Ingested bars land in the `bars` hypertable. Check:

```bash
docker exec -it quant-timescaledb psql -U quant -d quantsystem \
  -c "SELECT symbol, count(*) FROM bars GROUP BY symbol;"
```

### Validate data quality

```bash
make validate SYMBOL=AAPL
```

Reports gaps, outliers, and primary-vs-secondary construction
disagreements.

### Backtest + validation gates

```python
from src.backtesting.gate_orchestrator import StrategyGate
from src.config import get_settings

gate = StrategyGate(settings=get_settings())
result = gate.quick_validate(
    symbols=["AAPL"], start="2023-01-01", end="2023-12-31"
)
print(result)
```

This runs walk-forward, CPCV (45 paths), Deflated Sharpe, and PBO. See
[phase4_backtesting.md](phase4_backtesting.md) for knobs.

### Train / retrain a model

```bash
# Use the saved-best hyperparameters
make retrain SYMBOL=AAPL

# Run Optuna hyperparameter search (slow)
make retrain-tune SYMBOL=AAPL N_TRIALS=50

# Whole universe
make retrain-all TUNE=1
```

The trained meta-labeler is logged to MLflow and, if validation gates
pass, gets the `production` alias — which is what the live pipeline
loads.

### Paper trading

```bash
cp config/paper_trading.example.yaml config/paper_trading.yaml
# edit symbols, initial_cash, broker adapter, etc.
python -m src.execution.paper_trading --config config/paper_trading.yaml
```

This is the full pipeline end-to-end, but every fill is simulated. It
writes to the `orders`, `fills`, `tca_results`, and `portfolio_snapshots`
tables and exposes Prometheus metrics on port 9091.

---

## 11. Step 9 — Monitoring (Prometheus + Grafana)

Prometheus and Grafana came up as part of `make db-up`. To wire up
dashboards:

```bash
# 1. Log into Grafana at http://localhost:3000 (admin / admin) and
#    create an API key: Configuration → API Keys → Add (Admin role)

# 2. Provision the "Trading" dashboard + alerting rules
python scripts/setup_grafana.py \
    --grafana-url http://localhost:3000 \
    --api-key <your-grafana-api-key>
```

This creates a 6-row, 17+ panel dashboard covering NAV, Sharpe,
drawdown, win rate, orders, signals, feature drift, model staleness,
circuit-breaker triggers, and more.

### Metrics endpoint

Whatever pipeline you're running (`paper_trading`, `live_trading`)
exposes `http://localhost:9091/metrics` in Prometheus format. The
scrape job `wang_trading` in `config/prometheus.yml` pulls it every
15 s via the special `host.docker.internal` host.

### Alert channels

Configure Telegram in `config/paper_trading.yaml` /
`config/live_trading.yaml`:

```yaml
monitoring:
  telegram:
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"
```

Use `TelegramAlertManager.send_test()` to verify the channel before you
rely on it.

---

## 12. Step 10 — Production deployment (Linux host)

Everything above is enough for paper trading. Production adds three
things: systemd units, logrotate, and supervised auxiliary services.

The one-shot deploy script does the work:

```bash
# SSH into the target Linux host, then:
sudo ./scripts/deploy.sh --repo /path/to/wang_trading
```

What the script does (see `scripts/deploy.sh` for detail):

1. Creates a `wang` system user with no login shell.
2. Creates `/opt/wang_trading` (code) and `/var/log/wang_trading` (logs),
   both owned by `wang:wang`.
3. Rsyncs the repo into `/opt/wang_trading`, excluding `.git`, venvs, caches.
4. Builds a fresh venv at `/opt/wang_trading/venv` and installs `requirements.txt`.
5. Copies `.example` configs to their live names (no overwrite).
6. Installs:
   - systemd unit `/etc/systemd/system/wang-live-trading.service`
   - supervisor configs in `/etc/supervisor/conf.d/` *(if supervisor is installed)*
   - logrotate policy `/etc/logrotate.d/wang_trading`
7. Runs `systemctl daemon-reload`.

### Supervised services (started automatically by supervisor)

| Service | What it runs |
|---|---|
| `wang_data_equities` | `python -m src.data_engine.ingestion.runner --asset-class equities` |
| `wang_data_crypto` | `python -m src.data_engine.ingestion.runner --asset-class crypto` |
| `wang_monitoring` | Prometheus metrics server for non-trading processes |
| `wang_retrain_scheduler` | Weekly retraining, PBO, drift re-baseline |

### Live trading — systemd only, manual start

`wang-live-trading.service` is installed but **not** enabled to start at
boot. It is the only service that moves real money; you have to start
it explicitly after every boot, every code push, every halt.

### Post-deploy edits

On the host, before the first live start:

1. Fill in `/opt/wang_trading/config/live_trading.yaml` — symbols,
   broker section, deployment plan, circuit-breaker thresholds.
2. Create `/opt/wang_trading/config/live_trading.env` with the safety
   gates. Leave gates commented until the very moment you go live:
   ```ini
   # Uncomment ONE line at a time, asset class by asset class
   # WANG_ALLOW_LIVE_TRADING=yes
   # WANG_ALLOW_LIVE_CRYPTO=yes
   # WANG_ALLOW_LIVE_FUTURES=yes
   PYTHONPATH=/opt/wang_trading
   KMP_DUPLICATE_LIB_OK=TRUE
   ```
3. Seed production secrets via the file/AWS/GCP backend:
   ```bash
   sudo -u wang WANG_SECRETS_BACKEND=file WANG_MASTER_KEY=... \
       /opt/wang_trading/venv/bin/python /opt/wang_trading/scripts/setup_secrets.py
   ```

See [deployment.md](deployment.md) for the full operator-focused
deploy runbook, and [runbooks/deployment.md](runbooks/deployment.md) for
the ships-and-rollbacks procedure.

---

## 13. Step 11 — Going live

**Read [docs/go_live_checklist.md](go_live_checklist.md) first.** It
has 17 items that must all be ticked before you touch the start button.
Summary of the mechanical steps:

```bash
# On the production host, as an operator:

# 1. Prove a human is present
sudo -u wang touch /opt/wang_trading/.operator_checkin

# 2. Run the preflight — 18 blocker checks (brokers, model registry,
#    paper-trading proof window, infra, risk limits, secrets, …).
#    Exits 0 only if every check passes.
cd /opt/wang_trading
sudo -u wang venv/bin/python -m src.execution.preflight --full-check

# 3. If (and only if) preflight exits 0, start
sudo systemctl start wang-live-trading

# 4. Watch it
sudo journalctl -u wang-live-trading -f
```

### Stopping and recovering

```bash
# Graceful stop (writes a HALT sentinel; next start refuses until
# an operator clears it):
make live-stop
sudo systemctl stop wang-live-trading

# Emergency — cancel every open order, flatten every position:
make live-flatten

# Crashed / unclean exit — reconcile with broker from last snapshot:
make recover
```

### The 4-phase capital ramp

Live trading starts in **phase 1 — pilot ($5K)** by default. The
`CapitalDeploymentController` monitors paper-vs-live divergence for two
weeks before allowing promotion to phase 2 ($15K), then phase 3 ($50K),
then phase 4 (full). Each phase has an auto-halt on drawdown. See
[runbooks/capital_management.md](runbooks/capital_management.md).

---

## 14. Day-to-day operations

Once live, the operator follows [runbooks/daily_operations.md](runbooks/daily_operations.md).
The key touchpoints:

| Cadence | Runbook | What to do |
|---|---|---|
| Pre-open | daily_operations.md | Touch operator check-in, confirm broker heartbeat, review overnight alerts |
| Intraday | (watch Grafana) | Watch NAV, drawdown, breaker count, data-quality panel |
| End of day | daily_operations.md | Run daily reconciliation report; archive logs |
| Weekly | model_operations.md | Review retrain results; decide on RL shadow promotion |
| Every incident | incident_response.md | SEV-based triage; HALT first, investigate second |
| Monthly | compliance.md | Export audit log; verify HMAC chain; tax summary |

---

## 15. Troubleshooting

### `make setup` fails on `pip install`

- Upgrade pip first: `pip install --upgrade pip setuptools wheel`.
- On Ubuntu, ensure `build-essential`, `libpq-dev`, `libssl-dev`,
  `libomp-dev`, `python3-dev` are installed.
- On Apple Silicon, some wheels need Rosetta: `arch -arm64 pip install ...`.

### `scripts/setup_db.py` fails to connect

- `docker compose ps` — is `quant-timescaledb` healthy?
- `docker compose logs timescaledb` — any startup error?
- Is `DB_PASSWORD` set and does it match `docker-compose.yaml`?
- Try `psql -h localhost -U quant -d quantsystem` — works?

### Ingestion runs but no bars appear

- Market closed? Alpaca IEX only streams during RTH unless you upgrade
  to SIP.
- Check `logs/ingestion.log` for API errors (bad key, scope, rate limit).
- Confirm you have ≥ `min_ticks_per_bar` ticks — by default 10.

### `make smoke-test` passes but paper trading fails to start

- The paper-trading pipeline needs a trained model with `production`
  alias in MLflow. Run `make retrain SYMBOL=<sym>` for each symbol in
  your `paper_trading.yaml`.
- Check the Prometheus port isn't already in use: `lsof -i :9091`.

### Preflight fails before live start

- Read the failure reason — preflight prints each of the 18 checks.
- Common blockers: paper-trading proof window too short; model stale
  (> 7 days); operator check-in expired (> 1 h old); secrets backend
  not configured; circuit-breaker config too loose.

### systemd service flaps (starts then immediately exits)

- `journalctl -u wang-live-trading -e` for the last 100 lines.
- `ExecStartPre` runs the preflight — a preflight failure prevents
  `ExecStart` from ever running.
- Check `/opt/wang_trading/config/live_trading.env` exists and contains
  the gates you expect.

---

## 16. Appendix A — Environment variables reference

| Variable | Required? | Purpose | Read by |
|---|---|---|---|
| `ALPACA_API_KEY` | For equities | Alpaca REST + streaming key | `config/settings.yaml` via `${...}` |
| `ALPACA_SECRET_KEY` | For equities | Alpaca secret | `config/settings.yaml` via `${...}` |
| `POLYGON_API_KEY` | Optional | Polygon tick data | `config/settings.yaml` via `${...}` |
| `BINANCE_API_KEY` | For crypto | Binance (or testnet) key | `config/settings.yaml` via `${...}` |
| `BINANCE_SECRET_KEY` | For crypto | Binance secret | `config/settings.yaml` via `${...}` |
| `DB_PASSWORD` | Yes | TimescaleDB password | `config/settings.yaml` via `${...}` |
| `DATABASE_URL` | Optional | Full DB URL override (used by paper config) | `config/paper_trading.yaml` |
| `GCS_FEATURE_BUCKET` | Optional | GCS bucket for feature store (if `feature_store.backend: gcs`) | `config/settings.yaml` via `${...}` |
| `TELEGRAM_BOT_TOKEN` | Optional | Telegram bot for alerts | `config/*.yaml` via `${...}` |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat | `config/*.yaml` via `${...}` |
| `QUANT_CONFIG` | Optional | Alternate path to settings YAML | `src/config/settings.py` |
| `WANG_SECRETS_BACKEND` | Optional | Secrets backend: `env` (default), `file`, `aws`, `gcp` | `src/config/secrets.py` |
| `WANG_MASTER_KEY` | If `backend=file` | Fernet master key | `src/config/secrets.py` |
| `WANG_ALLOW_LIVE_TRADING` | To go live (equities) | Gates Alpaca live orders | `src/execution/broker_adapter.py` |
| `WANG_ALLOW_LIVE_CRYPTO` | To go live (crypto) | Gates CCXT live orders | `src/execution/broker_adapter.py` |
| `WANG_ALLOW_LIVE_FUTURES` | To go live (futures) | Gates IBKR live orders | `src/execution/ibkr_adapter.py` |
| `KMP_DUPLICATE_LIB_OK` | macOS only | OpenMP duplicate-lib guard (set to `TRUE`) | Runtime loader |
| `PYTHONPATH` | Set by systemd | Module search path (`/opt/wang_trading`) | systemd unit / supervisor |

---

## 17. Appendix B — Makefile targets

| Target | What it does |
|---|---|
| `make help` | Print every target with its docstring |
| `make setup` | `pip install -r requirements.txt` + scaffold config/logs dirs |
| `make db-up` / `make db-down` | Start / stop the Docker stack |
| `make db-setup` | Create the TimescaleDB schema |
| `make db-reset` | **Destructive.** Drop + recreate the main hypertables |
| `make run-equities` / `make run-crypto` | Real-time ingestion runners |
| `make backfill SYMBOL=AAPL DAYS=30` | Historical data backfill |
| `make validate SYMBOL=AAPL` | Data-quality validation for a symbol |
| `make test` | Unit tests (fast; excludes integration) |
| `make test-integration` | End-to-end Phase 2/3/4 integration tests |
| `make test-bars` | Bar-constructor tests only |
| `make test-cov` | Tests with coverage |
| `make smoke-test` | Phase 5 end-to-end smoke (<10 s, no external services) |
| `make preflight` | 18-blocker preflight check for live trading |
| `make audit-conformance` | Design-doc conformance audit (must be ≥ 95%) |
| `make live-start` / `make live-stop` | Start / halt-sentinel the live pipeline |
| `make live-flatten` | Emergency: cancel orders + close positions |
| `make recover` | Disaster recovery from last state snapshot |
| `make retrain SYMBOL=AAPL` | Retrain meta-labeler with saved hyperparams |
| `make retrain-tune SYMBOL=AAPL N_TRIALS=50` | Retrain + Optuna TPE tuning |
| `make retrain-all TUNE=1` | Retrain the whole configured universe |
| `make bench` / `make bench-backtest` | Phase 2 / Phase 4 micro-benchmarks |
| `make clean` | Remove `__pycache__`, `*.pyc`, `.pytest_cache`, `.coverage` |

---

## 18. Appendix C — File and directory layout

```
wang_trading/
├── Makefile                       # canonical task runner
├── README.md
├── requirements.txt               # Python dependencies
├── docker-compose.yaml            # TimescaleDB, Redis, Prometheus, Grafana
├── pytest.ini                     # pytest config (markers: integration, db)
│
├── config/
│   ├── settings.example.yaml      # → settings.yaml (required, git-ignored)
│   ├── paper_trading.example.yaml # → paper_trading.yaml (before paper trading)
│   ├── live_trading.example.yaml  # → live_trading.yaml (before live)
│   ├── futures_contracts.example.yaml  # IBKR contract specs
│   ├── prometheus.yml             # Prometheus scrape config
│   ├── supervisor/                # four supervisord configs (prod)
│   └── systemd/
│       └── wang-live-trading.service
│
├── scripts/
│   ├── setup_db.py                # create DB schema
│   ├── setup_secrets.py           # seed Fernet-encrypted secret vault
│   ├── setup_grafana.py           # provision dashboards + alerts
│   ├── deploy.sh                  # production install (run as root)
│   ├── smoke_test.py              # Phase 5 end-to-end smoke
│   ├── production_smoke_test.py   # Phase 6 subsystems smoke
│   ├── retrain_model.py           # meta-labeler retrain + tune
│   ├── replay_prediction.py       # debug a past prediction
│   └── design_doc_audit.py        # §1–§8 conformance audit
│
├── src/                           # Python package (import as `src.*`)
│   ├── config/                    # settings + secrets loaders
│   ├── data_engine/               # Phase 1
│   ├── feature_factory/           # Phase 2 features
│   ├── signal_battery/            # Phase 2 signals
│   ├── labeling/                  # Phase 3 labeling
│   ├── ml_layer/                  # Phase 3 ML
│   ├── bet_sizing/                # Phase 3 sizing
│   ├── backtesting/               # Phase 4 backtesting + gates
│   ├── portfolio/                 # Phase 4 portfolio construction
│   ├── execution/                 # Phase 5+6 execution, paper, live
│   └── monitoring/                # Phase 5+6 Prometheus, alerting, drift
│
├── tests/                         # > 1000 unit tests, ~28 integration tests
│   ├── benchmarks/
│   └── test_*.py
│
└── docs/
    ├── setup_guide.md             # ← you are here
    ├── architecture_overview.md   # system design, data flow, lifecycle
    ├── deployment.md              # operator-facing deploy runbook
    ├── go_live_checklist.md       # 17-item pre-live sign-off
    ├── phase2_features.md
    ├── phase2_signals.md
    ├── phase3_ml_pipeline.md
    ├── phase4_backtesting.md
    ├── phase4_portfolio.md
    ├── phase5_execution.md
    └── runbooks/
        ├── README.md              # runbook index
        ├── daily_operations.md
        ├── incident_response.md
        ├── deployment.md          # ships + rollbacks
        ├── model_operations.md
        ├── capital_management.md
        └── compliance.md
```

After `make setup` you will also see these (all git-ignored):

```
venv/
logs/
data/features/
config/settings.yaml
config/paper_trading.yaml     # only if you copied it
config/live_trading.yaml      # only if you copied it
config/secrets.enc            # only if using file backend
```

---

## 19. Appendix D — Where to look next

| If you want to… | Read |
|---|---|
| Understand the 6-phase design at a glance | [architecture_overview.md](architecture_overview.md) |
| Ship a code change to production | [runbooks/deployment.md](runbooks/deployment.md) |
| Run production day-to-day | [runbooks/daily_operations.md](runbooks/daily_operations.md) |
| Triage an incident | [runbooks/incident_response.md](runbooks/incident_response.md) |
| Add or retrain a model | [phase3_ml_pipeline.md](phase3_ml_pipeline.md), [runbooks/model_operations.md](runbooks/model_operations.md) |
| Adjust risk / capital phases | [runbooks/capital_management.md](runbooks/capital_management.md) |
| Export audit logs / verify HMAC chain | [runbooks/compliance.md](runbooks/compliance.md) |
| Go from paper to live | [go_live_checklist.md](go_live_checklist.md) |
| Understand Phase N in depth | `docs/phaseN_*.md` |

---

**Last updated:** 2026-04-17. If anything in this guide contradicts the
code, the code wins — open an issue or PR against this document.
