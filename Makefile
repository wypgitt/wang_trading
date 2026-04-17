.PHONY: help setup test test-integration bench run-equities run-crypto backfill db-up db-down db-setup clean

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

# ── Retraining (Phase 3) ──

retrain: ## Retrain meta-labeler with saved params (usage: make retrain SYMBOL=AAPL)
	python scripts/retrain_model.py --symbol $(SYMBOL) --use-best-params

retrain-tune: ## Retrain meta-labeler and tune hyperparameters (usage: make retrain-tune SYMBOL=AAPL [N_TRIALS=50] [TIMEOUT=600])
	python scripts/retrain_model.py --symbol $(SYMBOL) --tune \
		--n-trials $(or $(N_TRIALS),50) \
		--timeout $(or $(TIMEOUT),600)

retrain-all: ## Retrain the full configured universe (usage: make retrain-all [TUNE=1])
	python scripts/retrain_model.py --all-symbols \
		$(if $(filter 1,$(TUNE)),--tune --n-trials $(or $(N_TRIALS),50) --timeout $(or $(TIMEOUT),3600),--use-best-params)

test-bars: ## Run bar constructor tests only
	pytest tests/test_bar_constructors.py -v

test-integration: ## Run end-to-end Phase 2 integration test
	pytest tests/ -v -m integration -o addopts=""

smoke-test: ## Run end-to-end Phase 5 smoke test (no external services)
	python3 scripts/smoke_test.py

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
