"""Prometheus metrics (Phase 5 / P5.08).

Design doc §12.1 dashboard panels: NAV/PnL, exposure, positions, order
counts, slippage, bar formation rate, feature drift, model staleness,
circuit breaker triggers, data gaps. All exposed over an HTTP endpoint for
Grafana scraping.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from src.execution.models import PortfolioState


# Default slippage-bps histogram buckets (0.1bps → 500bps).
_SLIPPAGE_BUCKETS: tuple[float, ...] = (
    0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
)
# Meta-label probability buckets (0.0–1.0).
_PROB_BUCKETS: tuple[float, ...] = tuple(i / 10 for i in range(11))


class MetricsCollector:
    """Prometheus metrics for portfolio, orders, signals, features."""

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        namespace: str = "wang_trading",
    ) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        ns = namespace
        reg = self.registry

        # ── Portfolio gauges ──
        self.portfolio_nav = Gauge(
            f"{ns}_portfolio_nav", "Current portfolio NAV ($)", registry=reg,
        )
        self.portfolio_drawdown = Gauge(
            f"{ns}_portfolio_drawdown", "Current drawdown from peak", registry=reg,
        )
        self.portfolio_daily_pnl = Gauge(
            f"{ns}_portfolio_daily_pnl", "Today's P&L ($)", registry=reg,
        )
        self.portfolio_gross_exposure = Gauge(
            f"{ns}_portfolio_gross_exposure",
            "Gross exposure ($)", registry=reg,
        )
        self.portfolio_net_exposure = Gauge(
            f"{ns}_portfolio_net_exposure", "Net exposure ($)", registry=reg,
        )
        self.positions_count = Gauge(
            f"{ns}_positions_count", "Open position count", registry=reg,
        )

        # ── Order counters ──
        self.orders_submitted = Counter(
            f"{ns}_orders_submitted_total",
            "Orders submitted lifetime", registry=reg,
        )
        self.orders_filled = Counter(
            f"{ns}_orders_filled_total",
            "Orders fully filled lifetime", registry=reg,
        )
        self.orders_rejected = Counter(
            f"{ns}_orders_rejected_total",
            "Orders rejected by circuit breakers", registry=reg,
        )

        # ── Signals / ML ──
        self.signals = Counter(
            f"{ns}_signal_count",
            "Signals generated per family", ["family"], registry=reg,
        )
        self.meta_label_prob = Histogram(
            f"{ns}_meta_label_prob",
            "Meta-labeler probability distribution",
            buckets=_PROB_BUCKETS, registry=reg,
        )

        # ── Execution quality ──
        self.execution_slippage_bps = Histogram(
            f"{ns}_execution_slippage_bps",
            "Execution slippage distribution (bps)",
            buckets=_SLIPPAGE_BUCKETS, registry=reg,
        )

        # ── Data pipeline ──
        self.bar_formation_rate = Gauge(
            f"{ns}_bar_formation_rate",
            "Bars per hour per symbol", ["symbol"], registry=reg,
        )
        self.data_gap_seconds = Gauge(
            f"{ns}_data_gap_seconds",
            "Seconds since last tick per symbol", ["symbol"], registry=reg,
        )

        # ── Features ──
        self.feature_drift_kl = Gauge(
            f"{ns}_feature_drift_kl",
            "KL divergence vs training distribution", ["feature"], registry=reg,
        )

        # ── Model ──
        self.model_last_retrain_age_hours = Gauge(
            f"{ns}_model_last_retrain_age_hours",
            "Hours since last model retrain", registry=reg,
        )

        # ── Circuit breakers ──
        self.circuit_breaker_triggers = Counter(
            f"{ns}_circuit_breaker_triggers",
            "Circuit breaker triggers", ["breaker_type"], registry=reg,
        )

        self._server_started = False

    # ── Update APIs ────────────────────────────────────────────────────

    def update_portfolio(self, state: PortfolioState) -> None:
        self.portfolio_nav.set(state.nav)
        self.portfolio_drawdown.set(state.drawdown)
        self.portfolio_daily_pnl.set(state.daily_pnl)
        self.portfolio_gross_exposure.set(state.gross_exposure)
        self.portfolio_net_exposure.set(state.net_exposure)
        self.positions_count.set(state.position_count)

    def record_order_submitted(self, n: int = 1) -> None:
        self.orders_submitted.inc(n)

    def record_order_filled(self, n: int = 1) -> None:
        self.orders_filled.inc(n)

    def record_order_rejected(self, n: int = 1) -> None:
        self.orders_rejected.inc(n)

    def record_signal(self, family: str, count: int = 1) -> None:
        self.signals.labels(family=family).inc(count)

    def record_meta_label_prob(self, prob: float) -> None:
        self.meta_label_prob.observe(prob)

    def record_fill(self, slippage_bps: float) -> None:
        self.execution_slippage_bps.observe(abs(slippage_bps))

    def record_feature_drift(self, feature: str, kl_divergence: float) -> None:
        self.feature_drift_kl.labels(feature=feature).set(kl_divergence)

    def record_bar_rate(self, symbol: str, bars_per_hour: float) -> None:
        self.bar_formation_rate.labels(symbol=symbol).set(bars_per_hour)

    def record_data_gap(self, symbol: str, seconds: float) -> None:
        self.data_gap_seconds.labels(symbol=symbol).set(seconds)

    def record_circuit_breaker(self, breaker_type: str, n: int = 1) -> None:
        self.circuit_breaker_triggers.labels(breaker_type=breaker_type).inc(n)

    def update_model_age(self, last_retrain: datetime) -> None:
        now = datetime.now(timezone.utc)
        hours = (now - last_retrain).total_seconds() / 3600.0
        self.model_last_retrain_age_hours.set(hours)

    # ── HTTP server ────────────────────────────────────────────────────

    def start_server(self, port: int = 9090, addr: str = "0.0.0.0") -> None:
        """Expose /metrics on the given port. Idempotent per process."""
        if self._server_started:
            return
        start_http_server(port, addr=addr, registry=self.registry)
        self._server_started = True

    # ── Snapshot ───────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, float]:
        """Return a dict of all sample values (useful for tests + debug)."""
        out: dict[str, float] = {}
        for metric in self.registry.collect():
            for sample in metric.samples:
                if sample.labels:
                    key = f"{sample.name}{{{','.join(f'{k}={v}' for k, v in sorted(sample.labels.items()))}}}"
                else:
                    key = sample.name
                out[key] = sample.value
        return out
