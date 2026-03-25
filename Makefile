.PHONY: help setup test run-equities run-crypto backfill db-up db-down db-setup clean

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

test: ## Run all tests
	pytest tests/ -v

test-bars: ## Run bar constructor tests only
	pytest tests/test_bar_constructors.py -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Utilities ──

clean: ## Remove generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
