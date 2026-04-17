"""
Walk-forward backtester (design-doc §9.4).

Given per-symbol close prices, signals, meta-labeled probabilities, and
bet sizes, simulates sequential deployment bar-by-bar. Positions are opened
after a configurable execution delay, exited via triple-barrier rules, and
all-in transaction costs are subtracted at entry and exit.

``run()`` is the pure simulator — no model retraining. ``run_expanding_window()``
wraps it with a periodic retrain loop that mimics how the strategy would
actually be run live: fit, forecast forward, step, refit, etc.

Equity is tracked mark-to-market every bar; realised costs and unrealised P&L
on still-open positions both flow through the NAV series, so the drawdown
curve reflects what an operator would have experienced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.backtesting.transaction_costs import CostEstimate, TransactionCostModel


@dataclass
class BacktestTrade:
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    symbol: str
    side: int
    entry_price: float
    exit_price: float
    size: float
    signal_family: str
    gross_pnl: float
    costs: CostEstimate
    net_pnl: float
    holding_period_bars: int
    meta_label_prob: float
    return_pct: float


@dataclass
class BacktestResult:
    trades: list[BacktestTrade]
    equity_curve: pd.Series
    returns: pd.Series
    drawdown_curve: pd.Series
    metrics: dict = field(default_factory=dict)


# ── metrics ─────────────────────────────────────────────────────────────────


def _infer_periods_per_year(index: pd.Index) -> int:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        median_seconds = float(
            np.median(np.diff(index.view("int64"))) / 1e9
        )
        if median_seconds <= 0:
            return 252
        # seconds/year ≈ 365.25 · 86400
        return max(1, int(round(365.25 * 86400.0 / median_seconds)))
    return 252


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[BacktestTrade],
    risk_free_rate: float = 0.04,
) -> dict:
    """All standard performance metrics from an equity curve + trade list."""

    if len(equity_curve) < 2:
        return {"total_trades": len(trades)}

    periods_per_year = _infer_periods_per_year(equity_curve.index)
    returns = equity_curve.pct_change().dropna()

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    n_periods = len(returns)
    annualized_return = (
        (1.0 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1.0
        if n_periods > 0
        else 0.0
    )
    annualized_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))

    excess_per_period = returns - risk_free_rate / periods_per_year
    sharpe = (
        float(excess_per_period.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))
        if returns.std(ddof=0) > 0
        else 0.0
    )

    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = (
        (annualized_return - risk_free_rate) / downside_vol
        if downside_vol > 0
        else 0.0
    )

    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    max_drawdown = float(drawdown.min())

    # drawdown duration = longest run of non-zero drawdown (in bars)
    in_dd = (drawdown < 0).astype(int)
    max_dd_duration = 0
    cur = 0
    for v in in_dd.values:
        if v:
            cur += 1
            max_dd_duration = max(max_dd_duration, cur)
        else:
            cur = 0

    calmar = (
        annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    )
    recovery_factor = (
        (equity_curve.iloc[-1] - equity_curve.iloc[0]) / abs(
            max_drawdown * equity_curve.iloc[0]
        )
        if max_drawdown < 0
        else 0.0
    )

    # trade-level stats
    if trades:
        pnls = np.array([t.net_pnl for t in trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = float(len(wins) / len(pnls))
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        )
        avg_trade = float(pnls.mean())
        avg_holding = float(
            np.mean([t.holding_period_bars for t in trades])
        )
        total_costs = float(sum(t.costs.total_cost for t in trades))
        total_gross = float(sum(t.gross_pnl for t in trades))
        total_notional = float(
            sum(abs(t.size) * t.entry_price for t in trades)
        )
        avg_nav = float(equity_curve.mean())
        turnover = total_notional / avg_nav if avg_nav > 0 else 0.0
        cost_drag_bps = (
            total_costs / abs(total_gross) * 10_000.0
            if total_gross != 0
            else 0.0
        )
    else:
        win_rate = profit_factor = avg_trade = avg_holding = 0.0
        turnover = cost_drag_bps = 0.0

    # period-length in years for "trades_per_month"
    if isinstance(equity_curve.index, pd.DatetimeIndex) and len(equity_curve) > 1:
        span_days = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86400.0
        months = span_days / 30.4375
        trades_per_month = len(trades) / months if months > 0 else 0.0
    else:
        trades_per_month = 0.0

    skewness = float(returns.skew()) if n_periods > 2 else 0.0
    kurt = float(returns.kurt()) if n_periods > 3 else 0.0
    if len(returns) >= 20:
        tail_hi = float(np.quantile(returns, 0.95))
        tail_lo = float(np.quantile(returns, 0.05))
        tail_ratio = tail_hi / abs(tail_lo) if tail_lo < 0 else 0.0
    else:
        tail_ratio = 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_bars": int(max_dd_duration),
        "recovery_factor": float(recovery_factor),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_trade": float(avg_trade),
        "avg_holding_period_bars": float(avg_holding),
        "total_trades": len(trades),
        "trades_per_month": float(trades_per_month),
        "turnover": float(turnover),
        "cost_drag_bps": float(cost_drag_bps),
        "skewness": skewness,
        "kurtosis": kurt,
        "tail_ratio": float(tail_ratio),
        "periods_per_year": periods_per_year,
    }


# ── engine ──────────────────────────────────────────────────────────────────


def _as_aligned(obj, index, columns, default=0.0):
    """Coerce None/DataFrame/Series → DataFrame aligned to (index, columns)."""

    if obj is None:
        return pd.DataFrame(default, index=index, columns=columns, dtype=float)
    if isinstance(obj, pd.Series):
        obj = pd.DataFrame({columns[0]: obj}) if len(columns) == 1 else obj.to_frame()
    return obj.reindex(index=index, columns=columns).astype(float)


class WalkForwardBacktester:
    """Bar-by-bar simulator with triple-barrier exits and realistic costs."""

    def __init__(
        self,
        cost_model: TransactionCostModel,
        initial_capital: float = 100_000.0,
        execution_delay_bars: int = 1,
        max_positions: int = 20,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0,
        max_holding_period: int = 20,
        asset_class: str = "equities",
        signal_family: str = "composite",
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if execution_delay_bars < 0:
            raise ValueError("execution_delay_bars must be non-negative")
        if max_positions < 1:
            raise ValueError("max_positions must be >= 1")

        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.execution_delay_bars = execution_delay_bars
        self.max_positions = max_positions
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self.max_holding_period = max_holding_period
        self.asset_class = asset_class
        self.signal_family = signal_family

    # ── core simulator ────────────────────────────────────────────────

    def run(
        self,
        close: pd.DataFrame,
        signals_df: pd.DataFrame,
        meta_probs: pd.DataFrame | pd.Series | None = None,
        bet_sizes: pd.DataFrame | pd.Series | None = None,
        adv: pd.DataFrame | None = None,
        volatility: pd.DataFrame | None = None,
    ) -> BacktestResult:
        if close.empty:
            raise ValueError("close is empty")

        index = close.index
        symbols = list(close.columns)

        signals = signals_df.reindex(index=index, columns=symbols).fillna(0)
        probs = _as_aligned(meta_probs, index, symbols, default=1.0).fillna(1.0)
        sizes = _as_aligned(bet_sizes, index, symbols, default=1.0).fillna(0.0)
        adv_df = _as_aligned(adv, index, symbols, default=1e6)
        vol_df = _as_aligned(volatility, index, symbols, default=0.02).clip(lower=1e-6)

        n_bars = len(index)
        delay = self.execution_delay_bars

        # All signal → entry bar decisions, indexed by symbol
        trades: list[BacktestTrade] = []
        open_positions: dict[str, dict] = {}
        nav_series = np.full(n_bars, self.initial_capital, dtype=float)
        cash = self.initial_capital

        for t in range(n_bars):
            ts = index[t]

            # ── 1. exit any open positions whose barriers are hit at bar t ──
            to_close: list[str] = []
            for sym, pos in open_positions.items():
                if t < pos["entry_bar"]:
                    continue
                px = close.iloc[t][sym]
                if np.isnan(px):
                    continue
                hit_upper = px >= pos["upper_barrier"]
                hit_lower = px <= pos["lower_barrier"]
                hit_time = t >= pos["expiry_bar"]
                if hit_upper or hit_lower or hit_time:
                    to_close.append(sym)

            for sym in to_close:
                pos = open_positions.pop(sym)
                exit_px = float(close.iloc[t][sym])
                self._close_position(
                    pos=pos,
                    exit_ts=ts,
                    exit_bar=t,
                    exit_price=exit_px,
                    trades=trades,
                )
                # reconcile cash: refund exit proceeds - exit cost.
                # Entry cost was already deducted at entry; combined.total_cost
                # already includes it, so we subtract only the exit leg here.
                trade = trades[-1]
                exit_leg_cost = trade.costs.total_cost - pos["entry_cost"].total_cost
                cash += trade.gross_pnl - exit_leg_cost

            # ── 2. open new positions from signals at bar t-delay ──
            if t - delay >= 0:
                src_bar = t - delay
                src_ts = index[src_bar]
                for sym in symbols:
                    side = int(signals.iloc[src_bar][sym])
                    if side == 0:
                        continue
                    if sym in open_positions:
                        continue
                    if len(open_positions) >= self.max_positions:
                        break
                    bet = float(sizes.iloc[src_bar][sym])
                    if bet == 0.0:
                        continue
                    prob = float(probs.iloc[src_bar][sym])
                    entry_px = float(close.iloc[t][sym])
                    if np.isnan(entry_px) or entry_px <= 0:
                        continue
                    vol = float(vol_df.iloc[src_bar][sym])
                    adv_sym = float(adv_df.iloc[src_bar][sym])

                    # bet size is signed fraction of current NAV
                    nav_est = cash + self._mtm_open(open_positions, t, close)
                    notional = abs(bet) * nav_est
                    qty = notional / entry_px
                    if qty <= 0:
                        continue

                    entry_cost = self.cost_model.estimate(
                        order_size=qty,
                        price=entry_px,
                        adv=adv_sym,
                        volatility=vol,
                        asset_class=self.asset_class,
                    )
                    cash -= entry_cost.total_cost

                    upper = entry_px * (1 + self.upper_multiplier * vol)
                    lower = entry_px * (1 - self.lower_multiplier * vol)
                    expiry_bar = min(t + self.max_holding_period, n_bars - 1)

                    open_positions[sym] = {
                        "symbol": sym,
                        "side": side,
                        "entry_bar": t,
                        "entry_ts": ts,
                        "entry_signal_ts": src_ts,
                        "entry_price": entry_px,
                        "qty": qty,
                        "bet_fraction": bet,
                        "prob": prob,
                        "upper_barrier": upper,
                        "lower_barrier": lower,
                        "expiry_bar": expiry_bar,
                        "vol": vol,
                        "adv": adv_sym,
                        "entry_cost": entry_cost,
                    }

            # ── 3. mark-to-market NAV ──
            nav_series[t] = cash + self._mtm_open(open_positions, t, close)

        # Force-close any still-open at the final bar (already marked, but
        # convert unrealized → realized so the trades list is complete)
        final_bar = n_bars - 1
        final_ts = index[final_bar]
        for sym in list(open_positions.keys()):
            pos = open_positions.pop(sym)
            exit_px = float(close.iloc[final_bar][sym])
            if np.isnan(exit_px):
                continue
            self._close_position(
                pos=pos,
                exit_ts=final_ts,
                exit_bar=final_bar,
                exit_price=exit_px,
                trades=trades,
            )
            trade = trades[-1]
            cash += trade.gross_pnl - (trade.costs.total_cost - pos["entry_cost"].total_cost)
        nav_series[final_bar] = cash  # all flat

        equity = pd.Series(nav_series, index=index, name="equity")
        returns = equity.pct_change().fillna(0.0)
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        metrics = compute_metrics(equity, trades)

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            returns=returns,
            drawdown_curve=drawdown,
            metrics=metrics,
        )

    # ── expanding window orchestrator ──────────────────────────────────

    def run_expanding_window(
        self,
        bars_df: pd.DataFrame,
        features_df: pd.DataFrame,
        signal_battery: Callable,
        meta_labeling_pipeline: Callable,
        meta_labeler,
        bet_sizing_cascade: Callable,
        retrain_interval: int = 252,
        initial_train_size: int = 504,
    ) -> BacktestResult:
        """Expanding-window walk-forward: retrain periodically, forecast forward.

        Each ``retrain_interval`` bars the model is refit on all data up to
        the current boundary; the forward slice uses those fresh predictions.
        The callables let this method stay backend-agnostic — any signal-
        battery / labeling / cascade implementation that exposes the expected
        signatures plugs in.
        """

        n = len(bars_df)
        if n < initial_train_size:
            raise ValueError(
                f"need at least {initial_train_size} bars, got {n}"
            )

        all_signals: list[pd.DataFrame] = []
        all_probs: list[pd.DataFrame] = []
        all_sizes: list[pd.DataFrame] = []

        boundary = initial_train_size
        while boundary < n:
            window_end = min(boundary + retrain_interval, n)
            train_bars = bars_df.iloc[:boundary]
            train_feats = features_df.iloc[:boundary]
            fwd_bars = bars_df.iloc[boundary:window_end]
            fwd_feats = features_df.iloc[boundary:window_end]

            training_pack = meta_labeling_pipeline(train_bars, train_feats, signal_battery)
            meta_labeler.fit(
                training_pack["X"],
                training_pack["y"],
                sample_weight=training_pack.get("sample_weight"),
            )

            fwd_signals = signal_battery(fwd_bars, fwd_feats)
            fwd_probs_df = meta_labeler.predict_proba(fwd_feats)
            fwd_sizes_df = bet_sizing_cascade(fwd_signals, fwd_probs_df, fwd_bars)

            all_signals.append(fwd_signals)
            all_probs.append(fwd_probs_df)
            all_sizes.append(fwd_sizes_df)

            boundary = window_end

        signals_df = pd.concat(all_signals)
        probs_df = pd.concat(all_probs)
        sizes_df = pd.concat(all_sizes)

        close = bars_df[signals_df.columns].loc[signals_df.index]
        return self.run(
            close=close,
            signals_df=signals_df,
            meta_probs=probs_df,
            bet_sizes=sizes_df,
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _close_position(
        self,
        pos: dict,
        exit_ts: pd.Timestamp,
        exit_bar: int,
        exit_price: float,
        trades: list[BacktestTrade],
    ) -> None:
        side = pos["side"]
        qty = pos["qty"]
        entry_px = pos["entry_price"]
        gross = side * (exit_price - entry_px) * qty
        exit_cost = self.cost_model.estimate(
            order_size=qty,
            price=exit_price,
            adv=pos["adv"],
            volatility=pos["vol"],
            asset_class=self.asset_class,
        )
        total_costs = pos["entry_cost"].total_cost + exit_cost.total_cost
        combined = CostEstimate(
            commission=pos["entry_cost"].commission + exit_cost.commission,
            spread_cost=pos["entry_cost"].spread_cost + exit_cost.spread_cost,
            slippage=pos["entry_cost"].slippage + exit_cost.slippage,
            market_impact=pos["entry_cost"].market_impact + exit_cost.market_impact,
            total_cost=total_costs,
            cost_bps=(
                total_costs / (qty * 0.5 * (entry_px + exit_price)) * 10_000.0
                if qty > 0
                else 0.0
            ),
        )
        net = gross - total_costs
        return_pct = side * (exit_price / entry_px - 1.0)

        trades.append(
            BacktestTrade(
                entry_timestamp=pos["entry_ts"],
                exit_timestamp=exit_ts,
                symbol=pos["symbol"],
                side=side,
                entry_price=entry_px,
                exit_price=exit_price,
                size=pos["bet_fraction"],
                signal_family=self.signal_family,
                gross_pnl=float(gross),
                costs=combined,
                net_pnl=float(net),
                holding_period_bars=int(exit_bar - pos["entry_bar"]),
                meta_label_prob=float(pos["prob"]),
                return_pct=float(return_pct),
            )
        )

    def _mtm_open(
        self,
        open_positions: dict,
        t: int,
        close: pd.DataFrame,
    ) -> float:
        """Unrealized P&L across all open positions at bar t."""

        if not open_positions:
            return 0.0
        mtm = 0.0
        for pos in open_positions.values():
            px = close.iloc[t][pos["symbol"]]
            if np.isnan(px):
                continue
            mtm += pos["side"] * (float(px) - pos["entry_price"]) * pos["qty"]
        return mtm
