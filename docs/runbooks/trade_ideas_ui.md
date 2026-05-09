# Trade ideas UI

This UI is a local, read-only browser view for daily trade ideas. It is not
Django. It is a small Python standard-library HTTP server that serves one
HTML/CSS/JavaScript page and a JSON API.

The UI reuses the production bootstrap in paper-rehearsal mode:

```text
live_trading.yaml
  -> live bootstrap
  -> database bars
  -> feature assembler
  -> signal battery
  -> production MLflow meta model
  -> bet sizing
  -> target weights
  -> UI table
```

It does not call `OrderManager.run_cycle`, broker `submit_order`, or any live
order-routing path.

## Prerequisites

Run from the project environment with the normal dependencies installed:

```bash
cd /opt/wang_trading
source venv/bin/activate
pip install -r requirements.txt
```

For local development, use the repo directory instead:

```bash
cd /Users/yingpengwang/wang_trading
```

The UI needs the same runtime inputs as the live stack:

- `config/live_trading.yaml`
- reachable database from `storage.database_url`
- recent bars in the `bars` table for the configured symbols
- production MLflow model if you want exact live target generation

## Start the UI

From the repo root:

```bash
make trade-ideas-ui
```

Or run the script directly:

```bash
python3 scripts/trade_ideas_ui.py \
    --config config/live_trading.yaml \
    --host 127.0.0.1 \
    --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## What the table shows

Each row is one symbol-level trade idea:

- `Action`: `BUY`, `SELL`, `WATCH`, `MODEL_REQUIRED`, `NO_DATA`, or `ERROR`
- `Target`: target portfolio weight from the sizing and target optimizer
- `Notional`: target weight multiplied by current NAV
- `Est qty`: estimated quantity using the latest close
- `Meta`: production meta-label probability
- `Confidence`: top current signal confidence
- `Signals`: number of current signal rows for that symbol
- `Top family`: signal family with the strongest latest signal
- `Reason`: short explanation of why the row is actionable or not

The UI filters signals to the latest signal timestamp before meta inference,
sizing, and target generation. This keeps old historical signals from being
summed into the current daily idea.

## Controls

- `Symbols`: optional comma-separated override such as `AAPL,MSFT,NVDA`
- `Bars`: number of recent bars to load per symbol
- `Min weight`: minimum absolute target weight required for `BUY` or `SELL`
- `Paper fallback`: use signal confidence as a paper-only meta probability
  when no production MLflow model is loaded

Use `Paper fallback` only for research or UI smoke checks. For actual daily
operator review, leave it off so missing production model state is visible as
`MODEL_REQUIRED`.

## One-shot JSON report

To print the same data as JSON and exit:

```bash
python3 scripts/trade_ideas_ui.py \
    --config config/live_trading.yaml \
    --symbols AAPL,MSFT,NVDA \
    --bar-limit 500 \
    --once
```

The browser API endpoint is:

```text
GET /api/ideas?symbols=AAPL,MSFT&bar_limit=500&min_abs_weight=0.0025
```

Health check:

```text
GET /healthz
```

## Troubleshooting

If the page shows `No module named 'pandas'`, you started it outside the
project Python environment. Activate the venv and install requirements.

If rows show `NO_DATA`, confirm bars exist for the configured `bar_type`:

```sql
SELECT symbol, bar_type, MAX(timestamp), COUNT(*)
FROM bars
GROUP BY symbol, bar_type
ORDER BY symbol, bar_type;
```

If rows show `MODEL_REQUIRED`, the production MLflow model was not loaded.
Run the MLflow/model preflight checks before trusting daily target generation.

If rows show `ERROR`, read the `Reason` column first. It usually names the
failed stage: database fetch, feature assembly, signal generation, meta
inference, sizing, or target generation.

If port `8765` is already in use:

```bash
python3 scripts/trade_ideas_ui.py \
    --config config/live_trading.yaml \
    --port 8766
```

## Safety notes

This UI is a decision-support surface, not an execution surface. A displayed
`BUY` or `SELL` row means the current pipeline target is non-zero; it is not an
instruction to bypass preflight, broker checks, liquidity checks, position
limits, or human review.
