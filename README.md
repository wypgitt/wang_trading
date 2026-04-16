# Unified Quantitative Trading System

A multi-strategy, multi-asset quantitative trading engine integrating research from
López de Prado (AFML), Chan, Jansen, Clenow, Narang, Johnson, Isichenko, and Sinclair.

## Architecture

```
src/
├── data_engine/       # Phase 1: Bars, ingestion, storage
│   ├── bars/          # Bar constructors (tick, volume, dollar, TIB, VIB)
│   ├── ingestion/     # WebSocket/REST data collection
│   │   └── adapters/  # Exchange-specific adapters (Alpaca, Binance, IBKR)
│   ├── storage/       # TimescaleDB interface, feature store
│   └── validation/    # Bar quality validation
├── feature_factory/   # Phase 2: FFD, entropy, microstructure, GARCH, NLP
├── signal_battery/    # Phase 2: Momentum, mean-rev, trend, stat-arb, carry, vol
├── labeling/          # Phase 3: Triple-barrier, meta-labeling, sample weights
├── ml_layer/          # Phase 3: XGBoost meta-labeler, LSTM regime, RL agent
├── bet_sizing/        # Phase 3: AFML sizing, Kelly, GARCH adjustment
├── portfolio/         # Phase 4: HRP, factor risk, risk parity
├── backtesting/       # Phase 4: CPCV, deflated Sharpe, PBO
├── execution/         # Phase 5: VWAP/TWAP, market impact, TCA
└── monitoring/        # Phase 5: Grafana dashboards, alerting
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit config
cp config/settings.example.yaml config/settings.yaml
# Add your API keys (Alpaca, Polygon, etc.)

# 3. Set up database
python scripts/setup_db.py

# 4. Run ingestion (equities)
python -m src.data_engine.ingestion.runner --asset-class equities

# 5. Validate bars
python -m src.data_engine.validation.runner --symbol AAPL --bar-type tib
```

## Testing

```bash
make test               # unit tests (excludes integration)
make test-integration   # end-to-end Phase 2 pipeline test
make bench              # Phase 2 performance benchmarks
```

## Configuration

All settings are in `config/settings.yaml`. API keys should be set via
environment variables or the config file (never committed to git).

## Phase Status

- [x] **Phase 1: Data Foundation** (complete)
  - TIB/VIB/dollar/volume/tick bars; ETF trick; CUSUM filter; TimescaleDB storage
- [x] **Phase 2: Feature Factory + Signal Battery** (complete)
  - Features: FFD, structural breaks, entropy, microstructure, GARCH, NLP sentiment, on-chain, autoencoder, classical
  - Signals: 10 generators across the 7 canonical families plus a unified orchestrator
  - 326 unit tests + 1 end-to-end integration test, all green
  - Benchmarks: all 7 components under target (`make bench`)
  - See: [docs/phase2_features.md](docs/phase2_features.md) · [docs/phase2_signals.md](docs/phase2_signals.md)
- [ ] Phase 3: Labeling + Core ML
- [ ] Phase 4: Backtesting + Portfolio
- [ ] Phase 5: Execution + Paper Trading
- [ ] Phase 6: Live Capital + RL Agent
