.PHONY: help setup test test-integration bench run-equities run-crypto backfill db-up db-down db-setup clean broker-lifecycle-paper local-readiness-rehearsal

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ──

setup: ## Install dependencies and set up environment
	pip install -r requirements.txt
	cp -n config/settings.example.yaml config/settings.yaml 2>/dev/null || true
	mkdir -p logs data/features
	@echo "\n✓ Setup complete. Edit config/settings.yaml with your API keys."

# ── Database ──

db-up: ## Start TimescaleDB and Redis (Docker)
	docker compose up -d
	@echo "Waiting for database..."
	@sleep 3
	@echo "✓ Database ready"

db-down: ## Stop TimescaleDB and Redis
	docker compose down

db-setup: ## Create database schema
	python scripts/setup_db.py

db-reset: ## Reset database (DESTRUCTIVE)
	python scripts/setup_db.py --reset

# ── Ingestion ──

run-equities: ## Start real-time equities ingestion
	python -m src.data_engine.ingestion.runner --asset-class equities

run-crypto: ## Start real-time crypto ingestion
	python -m src.data_engine.ingestion.runner --asset-class crypto

backfill: ## Backfill historical data (usage: make backfill SYMBOL=AAPL DAYS=30)
	python -m src.data_engine.ingestion.runner --backfill --symbol $(SYMBOL) --days $(DAYS)

# ── Validation ──

validate: ## Validate bars (usage: make validate SYMBOL=AAPL)
	python -m src.data_engine.validation.runner --symbol $(SYMBOL) --compare-all

# ── Testing ──

test: ## Run unit tests (excludes @pytest.mark.integration)
	pytest tests/ -v

preflight: ## Run pre-flight checks before live trading (P6.05)
	python -m src.execution.preflight --full-check --config config/live_trading.yaml

audit-conformance: ## Check design-doc conformance (C5). Exit 0 if >= 95%.
	python scripts/design_doc_audit.py

live-start: ## Start the live trading pipeline (P6.07) — requires preflight clean
	python -m src.execution.live_trading --config config/live_trading.yaml

live-halt: ## Operator kill switch: write the HALT sentinel
	python -m src.execution.live_trading --config config/live_trading.yaml --halt --halt-reason operator_halt

live-stop: live-halt ## Backwards-compatible alias for live-halt

live-flatten: ## Operator kill switch: cancel orders, flatten positions, and halt
	python -m src.execution.live_trading --config config/live_trading.yaml --flatten

live-verify-flat: ## Verify broker/internal book is flat after a halt or flatten
	python -m src.execution.live_trading --config config/live_trading.yaml --verify-flat

recover: ## Run disaster recovery against the last snapshot (P6.13)
	python -m src.execution.disaster_recovery --recover

# ── Retraining (Phase 3) ──

retrain: ## Retrain meta-labeler with saved params (usage: make retrain SYMBOL=AAPL)
	python scripts/retrain_now.py --config config/retrain.yaml --symbol $(SYMBOL) --trigger manual

retrain-tune: ## Research-only retrain + Optuna tuning (usage: make retrain-tune SYMBOL=AAPL [N_TRIALS=50] [TIMEOUT=600])
	python scripts/retrain_model.py --symbol $(SYMBOL) --tune \
		--n-trials $(or $(N_TRIALS),50) \
		--timeout $(or $(TIMEOUT),600)

retrain-all: ## Retrain the full configured universe (usage: make retrain-all [TUNE=1])
	python scripts/retrain_now.py --config config/retrain.yaml --all --trigger manual

test-bars: ## Run bar constructor tests only
	pytest tests/test_bar_constructors.py -v

test-integration: ## Run end-to-end Phase 2 integration test
	pytest tests/ -v -m integration -o addopts=""

smoke-test: ## Run end-to-end Phase 5 smoke test (no external services)
	python3 scripts/smoke_test.py

smoke-production-bootstrap: ## Run validated DB-backed bootstrap smoke without live orders
	python3 scripts/production_bootstrap_smoke.py --config config/live_trading.yaml

shadow-replay: ## Replay recent bars through live target generation in shadow mode
	python3 scripts/shadow_replay.py --config config/live_trading.yaml --output logs/shadow_replay_report.md

morning-divergence: ## Generate an operator-readable paper/live divergence report from return CSVs
	python -m src.execution.daily_ops --paper-live-divergence \
		--paper-returns-csv logs/paper_returns.csv \
		--live-returns-csv logs/live_returns.csv

broker-lifecycle-paper: ## Exercise paper broker heartbeat/account/quote/order/cancel/flatten lifecycle
	python3 scripts/broker_lifecycle_check.py --config config/live_trading.yaml --output logs/broker_lifecycle_paper.json

local-readiness-rehearsal: ## Build local MLflow/DB/probe env and run shadow replay + preflight
	python3 scripts/local_readiness_rehearsal.py

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

bench: ## Run Phase 2 performance benchmarks
	python3 tests/benchmarks/bench_features.py

bench-backtest: ## Run Phase 4 backtesting / portfolio benchmarks
	python3 tests/benchmarks/bench_backtesting.py

# ── Utilities ──

clean: ## Remove generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
