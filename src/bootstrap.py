"""Production bootstrap wiring for the trading stack.

This module is the single place where CLI entry points turn YAML/env config
into concrete runtime objects. Strategy modules stay independent; bootstrap
only connects already-defined components.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.backtesting.gate_orchestrator import StrategyGate
from src.backtesting.transaction_costs import (
    CRYPTO_COSTS,
    EQUITIES_COSTS,
    FUTURES_COSTS,
    TransactionCostModel,
)
from src.bet_sizing.cascade import BetSizingCascade
from src.config import Settings, load_settings, validate_runtime_config
from src.data_engine.storage.database import DatabaseManager
from src.data_engine.storage.feature_store import FeatureStore
from src.execution.broker_adapter import PaperBrokerAdapter
from src.execution.broker_factory import BrokerFactory
from src.execution.capital_deployment import CapitalDeploymentController
from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.disaster_recovery import RecoveryManager, SnapshotManager
from src.execution.live_trading import LiveTradingPipeline
from src.execution.models import PortfolioState
from src.execution.order_manager import OrderManager
from src.execution.paper_trading import PaperTradingPipeline, PipelineConfig
from src.execution.preflight import PreflightChecker
from src.execution.retrain_scheduler import RetrainScheduler
from src.feature_factory.assembler import FeatureAssembler
from src.labeling.meta_labeler_pipeline import MetaLabelingPipeline
from src.ml_layer.meta_labeler import MetaLabeler
from src.ml_layer.retrain_pipeline import RetrainPipeline
from src.monitoring.alerting import (
    AlertManager,
    AlertSeverity,
    LogChannel,
    TelegramChannel,
)
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector
from src.signal_battery.orchestrator import create_default_battery

log = logging.getLogger(__name__)


# Runtime context records

@dataclass
class PipelineContext:
    pipeline: PaperTradingPipeline
    symbols: list[str]
    settings: Settings
    runtime_config: dict[str, Any]


@dataclass
class LivePipelineContext(PipelineContext):
    pipeline: LiveTradingPipeline
    broker_factory: BrokerFactory
    preflight_checker: PreflightChecker
    deployment_controller: CapitalDeploymentController


@dataclass
class RetrainContext:
    pipeline: RetrainPipeline
    data_loader: Callable[[str], tuple[pd.Series, pd.DataFrame]]
    universe: list[str]
    scheduler: RetrainScheduler | None = None
    runtime_config: dict[str, Any] | None = None

    def __iter__(self):
        yield self.pipeline
        yield self.data_loader
        yield self.universe


@dataclass
class ReplayContext:
    db: DatabaseManager
    model_registry: Any
    feature_store: FeatureStore

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


# Config loading

def load_runtime_config(
    config_path: str | Path | None = None,
    *,
    default_name: str = "paper_trading",
) -> dict[str, Any]:
    """Load runtime YAML and expand `env:KEY` leaves."""
    path = _resolve_config_path(config_path, default_name)
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"runtime config must be a mapping: {path}")
    validate_runtime_config(data, default_name, source=str(path))
    return _resolve_env_values(data)


def _resolve_config_path(
    config_path: str | Path | None,
    default_name: str,
) -> Path | None:
    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    else:
        candidates.extend([
            Path("config") / f"{default_name}.yaml",
            Path("config") / f"{default_name}.example.yaml",
        ])
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_env_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_env_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_values(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("env:"):
        return os.environ.get(obj[4:], "")
    return obj


def _settings(settings_path: str | Path | None = None) -> Settings:
    return load_settings(str(settings_path)) if settings_path else load_settings()


def _symbols(
    asset_class: str,
    runtime_config: dict[str, Any],
    settings: Settings,
) -> list[str]:
    configured = runtime_config.get("symbols")
    if configured:
        return list(configured)
    instruments = getattr(settings.instruments, asset_class, {}) or {}
    if asset_class == "equities":
        return list(
            instruments.get("test_symbols")
            or instruments.get("custom_symbols")
            or ["AAPL"]
        )
    return list(instruments.get("symbols") or [])


def _database_url(runtime_config: dict[str, Any], settings: Settings) -> str:
    return (
        ((runtime_config.get("storage") or {}).get("database_url"))
        or settings.database.url
    )


def _preflight_risk_config(runtime_config: dict[str, Any]) -> dict[str, Any]:
    risk = dict(runtime_config.get("risk") or {})
    breakers = runtime_config.get("circuit_breakers") or {}
    if breakers:
        risk.setdefault("circuit_breakers", breakers)
        risk.setdefault(
            "max_single_position",
            breakers.get("max_single_position_pct"),
        )
        risk.setdefault(
            "max_daily_loss",
            breakers.get("daily_loss_limit_pct"),
        )
    if runtime_config.get("operator_checkin_path"):
        risk.setdefault(
            "dead_mans_switch_path",
            runtime_config["operator_checkin_path"],
        )
    return {k: v for k, v in risk.items() if v is not None}


# Shared component builders

def build_alert_manager(config: dict[str, Any] | None = None) -> AlertManager:
    monitoring = (config or {}).get("monitoring", {})
    channels = {severity: [LogChannel()] for severity in AlertSeverity}
    telegram = monitoring.get("telegram", {}) if isinstance(monitoring, dict) else {}
    if telegram.get("enabled"):
        bot_token = telegram.get("bot_token", "")
        chat_id = telegram.get("chat_id", "")
        if bot_token and chat_id:
            tg = TelegramChannel(bot_token=bot_token, chat_id=chat_id)
            channels = {severity: [LogChannel(), tg] for severity in AlertSeverity}
    return AlertManager(channel_map=channels)


def build_circuit_breakers(config: dict[str, Any] | None = None) -> CircuitBreakerManager:
    return CircuitBreakerManager(**((config or {}).get("circuit_breakers") or {}))


def build_cost_model(config: dict[str, Any] | None = None) -> TransactionCostModel:
    costs = (config or {}).get("costs") or {}
    return TransactionCostModel(
        equities_config=costs.get("equities") or EQUITIES_COSTS,
        crypto_config=costs.get("crypto") or CRYPTO_COSTS,
        futures_config=costs.get("futures") or FUTURES_COSTS,
    )


def build_model_registry(config: dict[str, Any] | None = None) -> Any | None:
    ml = (config or {}).get("mlflow") or (config or {}).get("model_registry") or {}
    try:
        from src.ml_layer.model_registry import ModelRegistry
        return ModelRegistry(
            tracking_uri=ml.get("tracking_uri") or "sqlite:///mlflow.db",
            experiment_name=ml.get("experiment_name") or "meta-labeler",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("model registry unavailable: %s", exc)
        return None


def _asset_class_map(symbols: list[str], asset_class: str) -> dict[str, str]:
    return {symbol: asset_class for symbol in symbols}


def _price_feed_from_db(
    db: DatabaseManager | None,
    bar_type: str,
    fallback: float = 100.0,
) -> Callable[[str], float]:
    def _feed(symbol: str) -> float:
        if db is None:
            return fallback
        try:
            bars = db.get_bars(symbol, bar_type, limit=1)
            if not bars.empty and "close" in bars:
                return float(bars["close"].iloc[-1])
        except Exception:
            return fallback
        return fallback

    return _feed


def _load_production_model(registry: Any | None) -> Any | None:
    if registry is None:
        return None
    info_fn = getattr(registry, "get_production_model", None)
    load_fn = getattr(registry, "load_model", None)
    if not callable(info_fn) or not callable(load_fn):
        return None
    try:
        info = info_fn()
    except Exception as exc:  # noqa: BLE001
        log.warning("production model lookup failed: %s", exc)
        return None
    if not info:
        return None
    run_id = info.get("run_id") if isinstance(info, dict) else getattr(info, "run_id", None)
    if not run_id:
        return None
    try:
        return load_fn(run_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("production model load failed for %s: %s", run_id, exc)
        return None


class ModelMetaPipeline:
    """Live inference adapter: signals + feature rows -> meta probabilities."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.feature_names = list(getattr(model, "feature_names_", []) or [])

    def predict(self, features: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        if signals is None or signals.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for _, signal in signals.iterrows():
            timestamp = pd.Timestamp(signal.get("timestamp"))
            feature_row = self._feature_row(features, timestamp)
            if feature_row is None:
                continue
            X = self._live_frame(feature_row, signal, timestamp)
            try:
                # Prefer the (raw, calibrated) split when the model exposes
                # it (MetaLabeler.predict_proba(return_raw=True)); fall back
                # to a single calibrated value for plain/pyfunc models.
                try:
                    proba = self.model.predict_proba(X, return_raw=True)
                except TypeError:
                    proba = self.model.predict_proba(X)
                if isinstance(proba, tuple) and len(proba) == 2:
                    meta_prob = float(proba[0][0])
                    cal_prob = float(proba[1][0])
                else:
                    cal_prob = float(proba[0])
                    meta_prob = cal_prob
            except Exception as exc:  # noqa: BLE001
                log.warning("meta-label prediction failed: %s", exc)
                continue
            row = signal.to_dict()
            row["meta_prob"] = meta_prob
            row["calibrated_prob"] = cal_prob
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _feature_row(features: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
        if features.empty:
            return None
        frame = features
        if not isinstance(frame.index, pd.DatetimeIndex):
            return frame.iloc[-1]
        if timestamp.tzinfo is None and frame.index.tz is not None:
            timestamp = timestamp.tz_localize(frame.index.tz)
        elif timestamp.tzinfo is not None and frame.index.tz is None:
            timestamp = timestamp.tz_convert(None)
        elif timestamp.tzinfo is not None and frame.index.tz is not None:
            timestamp = timestamp.tz_convert(frame.index.tz)
        if timestamp in frame.index:
            row = frame.loc[timestamp]
            return row.iloc[0] if isinstance(row, pd.DataFrame) else row
        prior = frame.loc[frame.index <= timestamp]
        return None if prior.empty else prior.iloc[-1]

    def _live_frame(
        self,
        feature_row: pd.Series,
        signal: pd.Series,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        row = pd.Series(feature_row, dtype=float).copy()
        row["signal_side"] = int(signal.get("side", 0))
        row["signal_confidence"] = float(signal.get("confidence", 0.0))
        family = str(signal.get("family", ""))
        for name in self.feature_names:
            if name.startswith("signal_family_"):
                row[name] = 1.0 if name == f"signal_family_{family}" else 0.0
        if self.feature_names:
            for name in self.feature_names:
                if name not in row:
                    row[name] = 0.0
            frame = row.to_frame().T[self.feature_names]
        else:
            frame = row.to_frame().T
        frame.index = pd.DatetimeIndex([timestamp], name="event_timestamp")
        return frame.astype(float)


class ConfidenceMetaPipeline:
    """Paper-mode fallback that treats signal confidence as meta probability."""

    def predict(self, features: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        if signals is None or signals.empty:
            return pd.DataFrame()
        out = signals.copy()
        conf = out.get("confidence", pd.Series(0.0, index=out.index)).astype(float)
        out["meta_prob"] = (0.5 + 0.5 * conf.clip(lower=0.0, upper=1.0)).clip(0.0, 1.0)
        out["calibrated_prob"] = out["meta_prob"]
        return out


class CascadeBetSizingAdapter:
    """Adapter from meta-label rows to BetSizingCascade batch output."""

    def __init__(
        self,
        portfolio: PortfolioState,
        *,
        asset_class: str,
        asset_class_map: dict[str, str] | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.asset_class = asset_class
        self.asset_class_map = asset_class_map or {}
        self.cascade = BetSizingCascade()

    def compute(self, meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if meta is None or meta.empty:
            return pd.DataFrame()
        signals = meta.copy()
        if "prob" not in signals.columns:
            signals["prob"] = signals.get("calibrated_prob", signals.get("meta_prob", 0.5))
        if "asset_class" not in signals.columns:
            signals["asset_class"] = signals["symbol"].map(
                lambda s: self.asset_class_map.get(str(s), self.asset_class)
            )
        current_positions = {
            symbol: {
                "size": abs(pos.market_value) / max(self.portfolio.nav, 1.0),
                "family": pos.signal_family,
                "asset_class": self.asset_class_map.get(symbol, self.asset_class),
            }
            for symbol, pos in self.portfolio.positions.items()
        }
        return self.cascade.compute_position_sizes_batch(
            signals,
            features,
            {
                "nav": self.portfolio.nav,
                "current_positions": current_positions,
            },
        )


class DirectTargetOptimizer:
    """Convert signed final bet sizes directly into target portfolio weights."""

    def __init__(
        self,
        *,
        max_single_position: float = 0.10,
        max_gross_exposure: float = 1.50,
    ) -> None:
        self.max_single_position = float(max_single_position)
        self.max_gross_exposure = float(max_gross_exposure)

    def compute_target_portfolio(self, **kwargs) -> pd.DataFrame:
        bets = kwargs.get("bet_sizes")
        if bets is None or bets.empty:
            return pd.DataFrame(columns=["symbol", "target_weight", "strategy"])
        weight_col = "final_size" if "final_size" in bets.columns else "size"
        if weight_col not in bets.columns:
            return pd.DataFrame(columns=["symbol", "target_weight", "strategy"])
        work = bets.assign(target_weight=bets[weight_col].astype(float))
        if "family" in work.columns:
            grouped = (
                work.groupby("symbol", as_index=False)
                .agg({"target_weight": "sum", "family": "last"})
            )
            grouped = grouped.rename(columns={"family": "strategy"})
        else:
            grouped = (
                work.groupby("symbol", as_index=False)
                .agg({"target_weight": "sum"})
            )
            grouped["strategy"] = "meta_label"
        grouped["target_weight"] = grouped["target_weight"].clip(
            lower=-self.max_single_position,
            upper=self.max_single_position,
        )
        gross = float(grouped["target_weight"].abs().sum())
        if gross > self.max_gross_exposure and gross > 0:
            grouped["target_weight"] *= self.max_gross_exposure / gross
        return grouped[["symbol", "target_weight", "strategy"]]


def _runtime_target_optimizer(runtime: dict[str, Any]) -> DirectTargetOptimizer:
    risk = runtime.get("circuit_breakers") or runtime.get("risk") or {}
    return DirectTargetOptimizer(
        max_single_position=float(
            risk.get("max_single_position_pct", risk.get("max_single_position", 0.10))
        ),
        max_gross_exposure=float(risk.get("max_gross_exposure", 1.50)),
    )


# Paper / live pipeline builders

def build_paper_trading_pipeline(
    config_path: str | Path | None = None,
    *,
    settings_path: str | Path | None = None,
) -> PipelineContext:
    runtime = load_runtime_config(config_path, default_name="paper_trading")
    settings = _settings(settings_path)
    asset_class = runtime.get("asset_class", "equities")
    symbols = _symbols(asset_class, runtime, settings)

    db = DatabaseManager(_database_url(runtime, settings))
    bar_type = (runtime.get("bars") or {}).get(
        "type",
        getattr(getattr(settings.bars, asset_class), "primary_type", "tib"),
    )

    broker_cfg = runtime.get("broker") or {}
    broker = PaperBrokerAdapter(
        initial_cash=float(broker_cfg.get("initial_cash", 100_000.0)),
        slippage_bps=float(broker_cfg.get("slippage_bps", 2.0)),
        fill_delay_ms=int(broker_cfg.get("fill_delay_ms", 100)),
        price_feed=_price_feed_from_db(db, bar_type),
    )
    portfolio = PortfolioState(cash=float(broker_cfg.get("initial_cash", 100_000.0)))
    order_manager = OrderManager(
        broker,
        build_circuit_breakers(runtime),
        build_cost_model(runtime),
        portfolio,
        asset_class_map=_asset_class_map(symbols, asset_class),
    )
    asset_map = _asset_class_map(symbols, asset_class)
    registry = build_model_registry(runtime)
    model = _load_production_model(registry)
    meta_pipeline = (
        ModelMetaPipeline(model)
        if model is not None
        else ConfidenceMetaPipeline()
        if bool(runtime.get("allow_confidence_meta_fallback", True))
        else None
    )

    pipeline = PaperTradingPipeline(
        data_adapter=DatabaseBarDataAdapter(db, bar_type=bar_type),
        bar_constructors={},
        feature_assembler=FeatureAssembler(runtime.get("features")),
        signal_battery=create_default_battery(runtime.get("signals")),
        meta_pipeline=meta_pipeline,
        meta_labeler=None,
        bet_sizing=CascadeBetSizingAdapter(
            portfolio, asset_class=asset_class, asset_class_map=asset_map,
        ),
        portfolio_optimizer=_runtime_target_optimizer(runtime),
        order_manager=order_manager,
        metrics=MetricsCollector(),
        alert_manager=build_alert_manager(runtime),
        drift_detector=FeatureDriftDetector(),
        config=PipelineConfig(**(runtime.get("pipeline") or {})),
        db_manager=db,
    )
    return PipelineContext(pipeline, symbols, settings, runtime)


def build_live_trading_pipeline(
    config_path: str | Path | None = None,
    *,
    settings_path: str | Path | None = None,
    paper_rehearsal: bool = False,
    rehearsal_record_path: str | Path | None = None,
) -> LivePipelineContext:
    runtime = load_runtime_config(config_path, default_name="live_trading")
    rehearsal_cfg = dict(runtime.get("paper_rehearsal") or {})
    rehearsal_enabled = paper_rehearsal or bool(rehearsal_cfg.get("enabled", False))
    if rehearsal_enabled:
        runtime = dict(runtime)
        runtime["dry_run"] = True
        runtime["broker"] = dict(runtime.get("broker") or {})
        runtime["broker"]["adapter"] = "paper"
    settings = _settings(settings_path)
    asset_class = runtime.get("asset_class", "equities")
    symbols = _symbols(asset_class, runtime, settings)
    if not symbols:
        raise RuntimeError("live trading requires at least one configured symbol")

    db = DatabaseManager(_database_url(runtime, settings))
    bar_cfg = getattr(settings.bars, asset_class)
    bar_type = (
        ((runtime.get("bars") or {}).get("type"))
        or runtime.get("bar_type")
        or getattr(bar_cfg, "primary_type", "tib")
    )

    broker_factory = BrokerFactory(runtime)
    broker = broker_factory.get_broker(symbols[0], asset_class=asset_class)
    rehearsal_recorder = None
    if rehearsal_enabled:
        from src.execution.rehearsal import (
            OrderRehearsalRecorder,
            RecordingBrokerAdapter,
        )

        path = (
            rehearsal_record_path
            or rehearsal_cfg.get("record_path")
            or "logs/paper_production_rehearsal_orders.jsonl"
        )
        rehearsal_recorder = OrderRehearsalRecorder(path)
        broker = RecordingBrokerAdapter(broker, rehearsal_recorder)
    portfolio = PortfolioState(cash=float(runtime.get("initial_cash", 100_000.0)))
    asset_map = _asset_class_map(symbols, asset_class)
    order_manager = OrderManager(
        broker,
        build_circuit_breakers(runtime),
        build_cost_model(runtime),
        portfolio,
        asset_class_map=asset_map,
    )

    alert_manager = build_alert_manager(runtime)
    deployment = CapitalDeploymentController(
        portfolio=portfolio,
        metrics=None,
        asset_class=asset_class,
        alert_manager=alert_manager,
    )
    metrics = MetricsCollector()
    preflight = build_preflight_checker(
        config_path=config_path,
        settings_path=settings_path,
        broker_factory=broker_factory,
        portfolio=portfolio,
        metrics=metrics,
    )
    snapshot_manager = SnapshotManager(
        directory=Path((runtime.get("snapshots") or {}).get("directory", "logs/snapshots"))
    )
    recovery = RecoveryManager(
        snapshot_manager=snapshot_manager,
        halt_file=runtime.get("halt_file", ".live_halt"),
        crash_file=runtime.get("crash_file", ".live_crash"),
        alert_manager=alert_manager,
    )
    registry = build_model_registry(runtime)
    model = _load_production_model(registry)
    meta_pipeline = ModelMetaPipeline(model) if model is not None else None

    pipeline = LiveTradingPipeline(
        broker_factory=broker_factory,
        preflight_checker=preflight,
        deployment_controller=deployment,
        halt_file=runtime.get("halt_file", ".live_halt"),
        operator_checkin_path=runtime.get("operator_checkin_path", ".operator_checkin"),
        operator_checkin_max_age_h=float(runtime.get("operator_checkin_max_age_h", 1.0)),
        compliance_log_path=runtime.get(
            "compliance_log_path", "logs/live_trading_compliance.log"
        ),
        recovery_manager=recovery,
        paper_rehearsal=rehearsal_enabled,
        rehearsal_recorder=rehearsal_recorder,
        managed_symbols=symbols,
        data_adapter=DatabaseBarDataAdapter(db, bar_type=bar_type),
        bar_constructors={},
        feature_assembler=FeatureAssembler(runtime.get("features")),
        signal_battery=create_default_battery(runtime.get("signals")),
        meta_pipeline=meta_pipeline,
        meta_labeler=None,
        bet_sizing=CascadeBetSizingAdapter(
            portfolio, asset_class=asset_class, asset_class_map=asset_map,
        ),
        portfolio_optimizer=_runtime_target_optimizer(runtime),
        order_manager=order_manager,
        metrics=metrics,
        alert_manager=alert_manager,
        drift_detector=FeatureDriftDetector(),
        config=PipelineConfig(**(runtime.get("pipeline") or {})),
        db_manager=db,
    )
    return LivePipelineContext(
        pipeline=pipeline,
        symbols=symbols,
        settings=settings,
        runtime_config=runtime,
        broker_factory=broker_factory,
        preflight_checker=preflight,
        deployment_controller=deployment,
    )


def build_preflight_checker(
    config_path: str | Path | None = None,
    *,
    settings_path: str | Path | None = None,
    broker_factory: BrokerFactory | None = None,
    portfolio: PortfolioState | None = None,
    metrics: Any | None = None,
) -> PreflightChecker:
    runtime = load_runtime_config(config_path, default_name="live_trading")
    settings = _settings(settings_path)
    asset_class = runtime.get("asset_class", "equities")
    symbols = _symbols(asset_class, runtime, settings)
    factory = broker_factory or BrokerFactory(runtime)
    for symbol in symbols:
        try:
            factory.get_broker(symbol, asset_class=asset_class)
        except Exception as exc:  # noqa: BLE001
            log.warning("broker bootstrap failed for %s: %s", symbol, exc)

    risk_cfg = _preflight_risk_config(runtime)
    operator_cfg = runtime.get("operator") or {}
    alert_manager = build_alert_manager(runtime)
    infra_probe = _build_infra_probe(runtime, settings, alert_manager)
    return PreflightChecker(
        broker_factory=factory,
        portfolio=portfolio,
        model_registry=build_model_registry(runtime),
        metrics=metrics,
        paper_stats=runtime.get("paper_stats") or {},
        infra=runtime.get("infra") or {},
        infra_probe=infra_probe,
        risk_config=risk_cfg,
        operator_config=operator_cfg,
        config=runtime.get("preflight") or None,
    )


def _build_infra_probe(
    runtime: dict[str, Any],
    settings: Settings,
    alert_manager: AlertManager,
) -> Any | None:
    try:
        from src.execution.infra_probe import InfrastructureProbe
    except Exception as exc:  # noqa: BLE001
        log.warning("infra probe unavailable: %s", exc)
        return None

    infra = runtime.get("infra") or {}
    monitoring = runtime.get("monitoring") or {}
    ml = runtime.get("mlflow") or runtime.get("model_registry") or {}
    feature_store_cfg = runtime.get("feature_store") or {}
    storage_path = (
        feature_store_cfg.get("local_path")
        or getattr(settings.feature_store, "local_path", None)
    )
    return InfrastructureProbe(
        db_url=_database_url(runtime, settings),
        mlflow_tracking_uri=ml.get("tracking_uri") or "sqlite:///mlflow.db",
        prometheus_url=infra.get("prometheus_url") or monitoring.get("prometheus_url"),
        grafana_url=infra.get("grafana_url") or monitoring.get("grafana_url"),
        feature_store_path=infra.get("feature_store_path") or storage_path,
        db_disk_path=infra.get("db_disk_path"),
        alert_manager=alert_manager,
        alert_ping_enabled=bool(infra.get("alert_ping_enabled", True)),
        timeout_s=float(infra.get("probe_timeout_s", 3.0)),
    )


# Retrain / replay builders

def build_retrain_pipeline(
    config_path: str | Path | None = None,
    *,
    settings_path: str | Path | None = None,
) -> RetrainContext:
    runtime = load_runtime_config(config_path, default_name="retrain")
    settings = _settings(settings_path)
    db = DatabaseManager(_database_url(runtime, settings))
    asset_class = runtime.get("asset_class", "equities")
    symbols = _symbols(asset_class, runtime, settings)
    bar_type = runtime.get(
        "bar_type",
        getattr(getattr(settings.bars, asset_class), "primary_type", "tib"),
    )

    meta = MetaLabelingPipeline(**(runtime.get("meta_labeling") or {}))
    registry = build_model_registry(runtime)
    if registry is None:
        raise RuntimeError("model registry unavailable; cannot bootstrap retrain")

    model_type = (runtime.get("model") or {}).get("type", "lightgbm")

    def trainer(X: pd.DataFrame, y: pd.Series, sw: pd.Series) -> tuple[Any, float]:
        model = MetaLabeler(model_type=model_type, calibrate=True)
        model.fit(X, y, sample_weight=sw)
        return model, 0.0

    def scorer(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
        if hasattr(model, "predict"):
            pred = model.predict(X)
            return float((pred == y.to_numpy()).mean())
        return float("-inf")

    pipeline = RetrainPipeline(
        meta_labeling_pipeline=meta,
        feature_assembler=FeatureAssembler(runtime.get("features")),
        signal_battery=create_default_battery(runtime.get("signals")),
        gate=StrategyGate(),
        registry=registry,
        cost_model=build_cost_model(runtime),
        alert_manager=build_alert_manager(runtime),
        trainer=trainer,
        scorer=scorer,
        min_improvement_pct=float(runtime.get("min_improvement_pct", 0.05)),
    )

    def data_loader(symbol: str) -> tuple[pd.Series, pd.DataFrame]:
        bars = db.get_bars(
            symbol,
            bar_type,
            limit=int(runtime.get("limit", 100_000)),
        )
        if bars.empty:
            raise RuntimeError(f"no bars for {symbol}/{bar_type}")
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.set_index("timestamp").sort_index()
        return bars["close"], bars

    scheduler = RetrainScheduler(
        retrain_interval_days=int(runtime.get("retrain_interval_days", 7)),
        min_new_bars=int(runtime.get("min_new_bars", 100)),
        model_registry=registry,
        alert_manager=build_alert_manager(runtime),
        retrain_pipeline=pipeline,
    )
    return RetrainContext(
        pipeline,
        data_loader,
        symbols,
        scheduler=scheduler,
        runtime_config=runtime,
    )


def build_replay_context(
    config_path: str | Path | None = None,
    *,
    settings_path: str | Path | None = None,
) -> ReplayContext:
    runtime = load_runtime_config(config_path, default_name="live_trading")
    settings = _settings(settings_path)
    registry = build_model_registry(runtime)
    if registry is None:
        raise RuntimeError("model registry unavailable")
    return ReplayContext(
        db=DatabaseManager(_database_url(runtime, settings)),
        model_registry=registry,
        feature_store=FeatureStore(settings.feature_store.local_path),
    )


# Small data adapter

class DatabaseBarDataAdapter:
    """Read recent bars from TimescaleDB through the existing DatabaseManager."""

    def __init__(
        self,
        db: DatabaseManager,
        *,
        bar_type: str = "tib",
        limit: int = 500,
    ) -> None:
        self.db = db
        self.bar_type = bar_type
        self.limit = int(limit)

    async def get_bars(self, symbol: str) -> list[dict[str, Any]]:
        df = await asyncio.to_thread(
            self.db.get_bars,
            symbol,
            self.bar_type,
            None,
            None,
            self.limit,
        )
        if df.empty:
            return []
        return df.to_dict("records")


# Runner helpers used by CLIs/services

async def run_paper_trading(config_path: str | Path | None = None) -> None:
    ctx = build_paper_trading_pipeline(config_path)
    await ctx.pipeline.run(ctx.symbols)


async def run_live_trading(config_path: str | Path | None = None) -> None:
    ctx = build_live_trading_pipeline(config_path)
    await ctx.pipeline.startup_sequence()
    await ctx.pipeline.run(ctx.symbols)


async def run_retrain_scheduler(config_path: str | Path | None = None) -> None:
    ctx = build_retrain_pipeline(config_path)
    scheduler = ctx.scheduler
    if scheduler is None:
        raise RuntimeError("scheduler bootstrap failed")
    interval_s = int((ctx.runtime_config or {}).get("scheduler_interval_s", 3600))
    while True:
        now = datetime.now(timezone.utc)
        log.info("retrain scheduler heartbeat at %s", now.isoformat())
        for symbol in ctx.universe:
            close, bars = ctx.data_loader(symbol)
            await scheduler.retrain_via_pipeline(symbol, close, bars)
        await asyncio.sleep(interval_s)
