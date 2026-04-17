"""Paper Trading Pipeline (Phase 5 / P5.12).

Top-level orchestrator that connects every subsystem — data → features →
signals → ML → sizing → portfolio → execution → monitoring — into one loop.

The pipeline is intentionally loosely coupled: each stage is driven through
a minimal duck-typed protocol, so tests can inject fakes for any component.
The production CLI wires in the real implementations.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.execution.circuit_breakers import CircuitBreakerManager
from src.execution.models import OrderStatus, PortfolioState
from src.execution.order_manager import OrderManager
from src.monitoring.alerting import AlertManager, AlertSeverity
from src.monitoring.drift_detector import FeatureDriftDetector
from src.monitoring.metrics import MetricsCollector

log = logging.getLogger(__name__)


# ── Component protocols (duck-typed) ───────────────────────────────────

class _DataAdapter(Protocol):
    async def get_bars(self, symbol: str) -> list[Any]: ...


class _BarConstructor(Protocol):
    def update(self, tick_or_bar: Any) -> Any | None: ...


class _FeatureAssembler(Protocol):
    def compute(self, bars: pd.DataFrame) -> pd.DataFrame: ...


class _SignalBattery(Protocol):
    def generate(self, features: pd.DataFrame) -> pd.DataFrame: ...


class _MetaPipeline(Protocol):
    def predict(self, features: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame: ...


class _BetSizing(Protocol):
    def compute(self, meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame: ...


class _PortfolioOptimizer(Protocol):
    def compute_target_portfolio(self, **kwargs) -> pd.DataFrame: ...


# ── Pipeline ──────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    drift_check_every: int = 100       # bars
    snapshot_every: int = 50            # bars
    max_cycles: int | None = None
    sleep_seconds: float = 1.0
    regime: str | None = None
    nav_target: float = 1.0
    positions_history_every: int = 0   # 0 = disabled; >0 persists every N cycles


class PaperTradingPipeline:
    def __init__(
        self,
        *,
        data_adapter: _DataAdapter | None,
        bar_constructors: dict[str, _BarConstructor],
        feature_assembler: _FeatureAssembler | None,
        signal_battery: _SignalBattery | None,
        meta_pipeline: _MetaPipeline | None,
        meta_labeler: Any | None,
        bet_sizing: _BetSizing | None,
        portfolio_optimizer: _PortfolioOptimizer | None,
        order_manager: OrderManager,
        metrics: MetricsCollector,
        alert_manager: AlertManager,
        drift_detector: FeatureDriftDetector,
        config: dict | PipelineConfig | None = None,
        db_manager: Any | None = None,
        retrain_pipeline: Any | None = None,
        drift_retrain_threshold: float = 0.50,
        drift_warning_threshold: float = 0.20,
        drift_retrain_cooldown_hours: float = 24.0,
    ) -> None:
        self.data_adapter = data_adapter
        self.bar_constructors = bar_constructors
        self.feature_assembler = feature_assembler
        self.signal_battery = signal_battery
        self.meta_pipeline = meta_pipeline
        self.meta_labeler = meta_labeler
        self.bet_sizing = bet_sizing
        self.portfolio_optimizer = portfolio_optimizer
        self.order_manager = order_manager
        self.metrics = metrics
        self.alert_manager = alert_manager
        self.drift_detector = drift_detector

        if isinstance(config, dict):
            import dataclasses as _dc
            valid = {f.name for f in _dc.fields(PipelineConfig)}
            config = PipelineConfig(
                **{k: v for k, v in config.items() if k in valid}
            )
        self.config: PipelineConfig = config or PipelineConfig()

        self.db_manager = db_manager
        self.retrain_pipeline = retrain_pipeline
        self.drift_retrain_threshold = float(drift_retrain_threshold)
        self.drift_warning_threshold = float(drift_warning_threshold)
        self.drift_retrain_cooldown_hours = float(drift_retrain_cooldown_hours)
        self._last_drift_retrain: datetime | None = None
        self._drift_size_haircut_until: datetime | None = None
        self.running = False
        self.cycle_count = 0
        self.start_time = datetime.now(timezone.utc)
        self.start_nav = self.order_manager.portfolio.nav
        self._nav_history: list[tuple[datetime, float]] = []
        self._orders_placed_count = 0
        self._orders_filled_count = 0

    # ── Core cycle ─────────────────────────────────────────────────────

    async def run_cycle(
        self,
        *,
        bars: dict[str, Any] | None = None,
        features: pd.DataFrame | None = None,
        prices: dict[str, float] | None = None,
    ) -> dict:
        """Execute one full cycle. Inputs may be supplied directly (tests)
        or pulled from `data_adapter` (production)."""
        self.cycle_count += 1
        pf = self.order_manager.portfolio

        if prices is None and bars is not None:
            prices = {sym: float(getattr(bar, "close", 0.0)) for sym, bar in bars.items()}
        prices = prices or {}

        # 1-4. Features → signals → meta-labeler → bet sizes
        signals = None
        meta = None
        bet_sizes = None
        if features is not None:
            if self.signal_battery is not None:
                signals = self.signal_battery.generate(features)
                if signals is not None and not signals.empty:
                    for family in signals.get("family", pd.Series(dtype=str)).unique():
                        self.metrics.record_signal(str(family))
            if self.meta_pipeline is not None and signals is not None:
                meta = self.meta_pipeline.predict(features, signals)
                if meta is not None and "meta_prob" in meta.columns:
                    for p in meta["meta_prob"]:
                        if np.isfinite(p):
                            self.metrics.record_meta_label_prob(float(p))
                    await self._persist_meta_labels(meta, features)
            if self.bet_sizing is not None and meta is not None:
                bet_sizes = self.bet_sizing.compute(meta, features)

        # 5. Portfolio optimizer
        target = None
        if self.portfolio_optimizer is not None and meta is not None:
            try:
                target = self.portfolio_optimizer.compute_target_portfolio(
                    strategy_returns={},
                    current_signals=signals if signals is not None else pd.DataFrame(),
                    bet_sizes=bet_sizes if bet_sizes is not None else pd.DataFrame(),
                    regime=self.config.regime,
                    nav=pf.nav,
                )
            except Exception as exc:
                log.debug("Portfolio optimizer skipped: %s", exc)
                target = None

        # 6-7. Execute via OrderManager (handles exits + rebalance + reconcile)
        summary = await self.order_manager.run_cycle(prices, target=target)

        # 8. Update metrics
        self.metrics.update_portfolio(pf)
        self._nav_history.append((datetime.now(timezone.utc), pf.nav))
        filled = [
            o for o in summary.get("rebalance_orders", []) + summary.get("exits", [])
            if o.status == OrderStatus.FILLED
        ]
        rejected = [
            o for o in summary.get("rebalance_orders", []) + summary.get("exits", [])
            if o.status == OrderStatus.REJECTED
        ]
        self._orders_placed_count += len(summary.get("rebalance_orders", []))
        self._orders_filled_count += len(filled)
        if filled:
            self.metrics.record_order_filled(len(filled))
        if rejected:
            self.metrics.record_order_rejected(len(rejected))

        # 9. Feature drift (sampled)
        if (
            features is not None
            and self.cycle_count % max(1, self.config.drift_check_every) == 0
        ):
            drifted = self.drift_detector.get_drifted_features(features)
            for feat in drifted:
                self.metrics.record_feature_drift(feat, 1.0)
            total_features = max(1, int(features.shape[1]) if hasattr(features, "shape") else 1)
            drift_pct = len(drifted) / total_features if total_features else 0.0
            await self._handle_drift(drifted, drift_pct)

        # 10. Circuit breakers (already run by OrderManager.run_cycle; alert on any)
        for breaker in summary.get("breakers", []):
            self.metrics.record_circuit_breaker(breaker.action)
            await self.alert_manager.send_alert(
                self.alert_manager.alert_circuit_breaker(breaker),
            )

        # 10b. Periodic positions_history snapshot
        every = getattr(self.config, "positions_history_every", 0) or 0
        if every > 0 and self.db_manager is not None and (self.cycle_count % every == 0):
            try:
                await self.db_manager.insert_positions_snapshot(
                    datetime.now(timezone.utc), pf.positions,
                )
            except Exception as exc:
                log.warning("positions_history snapshot failed: %s", exc)

        # 11. Log cycle summary
        log.debug(
            "cycle=%d nav=%.2f dd=%.2%% orders=%d exits=%d",
            self.cycle_count, pf.nav, pf.drawdown,
            len(summary.get("rebalance_orders", [])),
            len(summary.get("exits", [])),
        )

        return {
            "cycle": self.cycle_count,
            "nav": pf.nav,
            "drawdown": pf.drawdown,
            "orders": summary.get("rebalance_orders", []),
            "exits": summary.get("exits", []),
            "breakers": summary.get("breakers", []),
            "reconciliation": summary.get("reconciliation", []),
        }

    # ── Meta-label persistence (C4) ────────────────────────────────────

    async def _persist_meta_labels(
        self, meta: pd.DataFrame, features: pd.DataFrame | None,
    ) -> None:
        """Write one row per meta-label prediction to the meta_labels table.

        Any failure is swallowed — a DB hiccup must not break the live cycle.
        """
        if self.db_manager is None or meta is None or len(meta) == 0:
            return
        try:
            from src.feature_factory.assembler import compute_feature_hash
            model_version = str(getattr(self, "model_version", "") or "")
            now = datetime.now(timezone.utc)
            for i, row in meta.reset_index(drop=True).iterrows():
                symbol = str(row.get("symbol", "") or "")
                family = str(row.get("family") or row.get("signal_family") or "")
                prob = float(row.get("meta_prob", 0.0))
                cal = row.get("calibrated_prob")
                cal = float(cal) if cal is not None and np.isfinite(cal) else None
                timestamp = row.get("timestamp") or now
                fhash: str | None = None
                if features is not None and i < len(features):
                    try:
                        fhash = compute_feature_hash(features.iloc[i])
                    except Exception:
                        fhash = None
                await self.db_manager.insert_meta_label(
                    timestamp=timestamp, symbol=symbol, signal_family=family,
                    meta_prob=prob, calibrated_prob=cal,
                    model_version=model_version, feature_hash=fhash,
                )
        except Exception as exc:  # pragma: no cover - best-effort only
            log.warning("meta_labels persistence failed: %s", exc)

    # ── Drift → retrain plumbing (C3) ──────────────────────────────────

    async def _handle_drift(self, drifted: list[str], drift_pct: float) -> None:
        if not drifted:
            return
        now = datetime.now(timezone.utc)
        if drift_pct >= self.drift_retrain_threshold:
            log.warning(
                "severe drift: %.1f%% of features (%d of them); triggering emergency retrain",
                drift_pct * 100, len(drifted),
            )
            # Haircut bet sizes to 50% for 24 hours
            self._drift_size_haircut_until = now + timedelta(hours=24)
            await self.alert_manager.send_alert(
                self.alert_manager.alert_feature_drift(drifted[0], kl=drift_pct)
            )
            if self.retrain_pipeline is not None:
                cooldown_ok = (
                    self._last_drift_retrain is None
                    or (now - self._last_drift_retrain)
                       >= timedelta(hours=self.drift_retrain_cooldown_hours)
                )
                if cooldown_ok:
                    self._last_drift_retrain = now
                    try:
                        asyncio.create_task(self._dispatch_emergency_retrain(drifted, drift_pct))
                    except RuntimeError as exc:  # pragma: no cover - no running loop
                        log.warning("could not schedule retrain: %s", exc)
                else:
                    log.info("drift retrain skipped — within cooldown window")
        elif drift_pct >= self.drift_warning_threshold:
            await self.alert_manager.send_alert(
                self.alert_manager.alert_feature_drift(drifted[0], kl=drift_pct)
            )
        else:
            log.debug("mild drift: %.1f%% (%d features)", drift_pct * 100, len(drifted))

    async def _dispatch_emergency_retrain(
        self, drifted: list[str], drift_pct: float,
    ) -> None:
        """Fire-and-forget wrapper so run_cycle is never blocked by retraining."""
        try:
            await self.retrain_pipeline.emergency_retrain(
                symbol="",
                reason=f"drift_pct={drift_pct:.2f} features={drifted[:5]}",
                close=pd.Series(dtype=float),
                bars=pd.DataFrame(),
            )
        except Exception as exc:  # pragma: no cover - best-effort
            log.exception("emergency retrain failed: %s", exc)

    @property
    def drift_size_multiplier(self) -> float:
        if self._drift_size_haircut_until is None:
            return 1.0
        if datetime.now(timezone.utc) >= self._drift_size_haircut_until:
            self._drift_size_haircut_until = None
            return 1.0
        return 0.5

    # ── Main loop ──────────────────────────────────────────────────────

    async def run(self, symbols: list[str]) -> None:
        self.running = True
        max_cycles = self.config.max_cycles
        try:
            while self.running:
                bars = await self._fetch_next_bars(symbols)
                if not bars:
                    await asyncio.sleep(self.config.sleep_seconds)
                    continue
                features = await self._compute_features(bars)
                await self.run_cycle(bars=bars, features=features)
                if max_cycles is not None and self.cycle_count >= max_cycles:
                    break
                await asyncio.sleep(self.config.sleep_seconds)
        finally:
            await self.shutdown()

    async def _fetch_next_bars(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch the latest bar per symbol from the data adapter."""
        if self.data_adapter is None:
            return {}
        bars: dict[str, Any] = {}
        for sym in symbols:
            try:
                feed = await self.data_adapter.get_bars(sym)
                if feed:
                    bars[sym] = feed[-1]
            except Exception as exc:
                log.warning("data adapter failed for %s: %s", sym, exc)
        return bars

    async def _compute_features(self, bars: dict[str, Any]) -> pd.DataFrame | None:
        if self.feature_assembler is None or not bars:
            return None
        try:
            frame = pd.DataFrame.from_dict(
                {sym: bar.__dict__ if hasattr(bar, "__dict__") else bar
                 for sym, bar in bars.items()},
                orient="index",
            )
            return self.feature_assembler.compute(frame)
        except Exception as exc:
            log.debug("feature compute failed: %s", exc)
            return None

    # ── Shutdown ───────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        self.running = False
        pf = self.order_manager.portfolio
        for order in list(pf.open_orders):
            try:
                await self.order_manager.broker.cancel_order(order.order_id)
            except Exception as exc:
                log.warning("cancel failed for %s: %s", order.order_id, exc)
        log.info(
            "Paper trading shutdown: nav=%.2f orders_filled=%d cycles=%d",
            pf.nav, self._orders_filled_count, self.cycle_count,
        )
        await self.alert_manager.send_alert(
            self.alert_manager.alert_execution_failure(
                type("O", (), {"order_id": "shutdown", "symbol": "SYSTEM"})(),
                error="pipeline shutdown",
            ) if False else _shutdown_alert(pf, self._orders_filled_count,
                                            self.cycle_count),
        )
        self._save_snapshot()

    def _save_snapshot(self) -> None:
        """Hook for subclasses/integrations to persist the snapshot.

        Default is a no-op; `ExecutionStorage` integration can subclass.
        """
        pf = self.order_manager.portfolio
        log.info("snapshot nav=%.2f cash=%.2f positions=%d",
                 pf.nav, pf.cash, pf.position_count)

    # ── Performance summary ────────────────────────────────────────────

    def get_performance_summary(self) -> dict:
        pf = self.order_manager.portfolio
        now = datetime.now(timezone.utc)
        days = max((now - self.start_time).total_seconds() / 86400.0, 1 / 86400.0)

        ret = (pf.nav - self.start_nav) / self.start_nav if self.start_nav > 0 else 0.0
        sharpe = self._annualized_sharpe()
        return {
            "nav": pf.nav,
            "start_nav": self.start_nav,
            "return_pct": ret,
            "sharpe": sharpe,
            "drawdown": pf.drawdown,
            "peak_nav": pf.peak_nav,
            "days_running": days,
            "cycles": self.cycle_count,
            "orders_placed": self._orders_placed_count,
            "orders_filled": self._orders_filled_count,
        }

    def _annualized_sharpe(self) -> float:
        if len(self._nav_history) < 3:
            return 0.0
        navs = np.array([v for _, v in self._nav_history], dtype=float)
        rets = np.diff(navs) / navs[:-1]
        rets = rets[np.isfinite(rets)]
        if len(rets) < 2 or rets.std(ddof=1) == 0:
            return 0.0
        # Per-cycle Sharpe × sqrt(cycles/year). We don't know the cycle
        # period, so approximate with hourly cycles (252 * 24 * 60 = 362880
        # per year) and leave interpretation to caller.
        return float(rets.mean() / rets.std(ddof=1) * math.sqrt(252 * 24))


def _shutdown_alert(pf, orders_filled: int, cycles: int):
    # Small helper so the shutdown message is a proper Alert object.
    from src.monitoring.alerting import Alert
    return Alert(
        severity=AlertSeverity.INFO,
        title="Paper Trading Shutdown",
        message=f"NAV ${pf.nav:,.0f} · {orders_filled} fills · {cycles} cycles",
        source="paper_trading",
        metadata={"nav": pf.nav, "filled": orders_filled, "cycles": cycles},
    )


# ── CLI ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="paper_trading")
    p.add_argument("--asset-class", choices=["equities", "crypto", "futures"],
                   default="equities")
    p.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL",
                   help="Comma-separated symbol list")
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config path")
    p.add_argument("--max-cycles", type=int, default=None)
    return p.parse_args()


def main() -> int:  # pragma: no cover — CLI glue only
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info("Paper trading CLI — asset_class=%s symbols=%s",
             args.asset_class, args.symbols)
    log.warning(
        "The production paper-trading runner wires in the real adapters; "
        "use the PaperTradingPipeline factory from a project-level bootstrap "
        "rather than this bare CLI."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
