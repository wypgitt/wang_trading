"""Gymnasium trading environment for RL training (P6.08).

The environment exposes one decision per symbol per step. An action is a
position tier in ``[0, tier_count]`` — 0 means flat, higher tiers map to
larger long exposures. The reward combines a rolling Sharpe signal with a
drawdown penalty and a turnover cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

log = logging.getLogger(__name__)


# ── Environment ───────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    initial_capital: float = 100_000.0
    max_steps: int | None = None
    position_tier_count: int = 5
    max_position_pct: float = 0.20
    sharpe_window: int = 20
    drawdown_limit: float = 0.30
    drawdown_penalty_lambda: float = 10.0
    drawdown_threshold: float = 0.05
    turnover_penalty_mu: float = 0.1
    default_cost_bps: float = 5.0


class TradingEnv(gym.Env):
    """Multi-symbol trading environment (long-only, tiered sizing)."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        bars: dict[str, pd.DataFrame] | pd.DataFrame,
        features: pd.DataFrame,
        signals: pd.DataFrame | None = None,
        meta_probs: pd.DataFrame | None = None,
        regime_probs: pd.DataFrame | None = None,
        cost_model: Any | None = None,
        initial_capital: float = 100_000.0,
        max_steps: int | None = None,
        position_tier_count: int = 5,
        max_position_pct: float = 0.20,
    ) -> None:
        super().__init__()

        # Normalise bars → dict[symbol, DataFrame]
        if isinstance(bars, pd.DataFrame):
            bars = {"SYM": bars}
        if not bars:
            raise ValueError("bars must contain at least one symbol")
        self.symbols: list[str] = list(bars.keys())
        self.n_symbols = len(self.symbols)

        # Build a 2D close-price matrix [T, n_symbols]
        closes = []
        for sym in self.symbols:
            df = bars[sym]
            if "close" not in df.columns:
                raise ValueError(f"bars[{sym!r}] must have a 'close' column")
            closes.append(df["close"].to_numpy(dtype=float))
        lengths = {len(c) for c in closes}
        if len(lengths) > 1:
            raise ValueError("all symbol bar histories must be the same length")
        self._closes = np.vstack(closes).T  # shape (T, n_symbols)
        self.T = self._closes.shape[0]
        if self.T < 2:
            raise ValueError("need at least 2 bars")

        self.features = features.reset_index(drop=True)
        if len(self.features) != self.T:
            raise ValueError("features must have the same length as bars")
        self.n_features = self.features.shape[1]

        self.meta_probs = self._ensure_symbol_frame(meta_probs)
        self.regime_probs = (
            regime_probs.reset_index(drop=True)
            if regime_probs is not None
            else pd.DataFrame(np.zeros((self.T, 1)))
        )
        self.n_regimes = self.regime_probs.shape[1]
        self.signals = signals  # not in observation but kept for render

        # Config
        self.config = EnvConfig(
            initial_capital=initial_capital,
            max_steps=max_steps,
            position_tier_count=position_tier_count,
            max_position_pct=max_position_pct,
        )
        self.cost_model = cost_model

        # Action & observation spaces
        tier_choices = position_tier_count + 1  # 0 = flat
        self.action_space = spaces.MultiDiscrete([tier_choices] * self.n_symbols)
        self.observation_space = spaces.Dict({
            "features": spaces.Box(low=-np.inf, high=np.inf,
                                   shape=(self.n_features,), dtype=np.float32),
            "positions": spaces.Box(low=-1.0, high=1.0,
                                    shape=(self.n_symbols,), dtype=np.float32),
            "meta_probs": spaces.Box(low=0.0, high=1.0,
                                     shape=(self.n_symbols,), dtype=np.float32),
            "regime_probs": spaces.Box(low=0.0, high=1.0,
                                       shape=(self.n_regimes,), dtype=np.float32),
            "portfolio_stats": spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(5,), dtype=np.float32),
            "calendar": spaces.Box(low=-np.inf, high=np.inf,
                                   shape=(2,), dtype=np.float32),
        })

        # Runtime state (populated by reset)
        self._step_idx = 0
        self._weights = np.zeros(self.n_symbols, dtype=float)
        self._nav = float(initial_capital)
        self._peak_nav = float(initial_capital)
        self._returns_hist: list[float] = []
        self._nav_hist: list[float] = [self._nav]

    # ── Helpers ───────────────────────────────────────────────────────

    def _ensure_symbol_frame(self, frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None:
            return pd.DataFrame(np.full((self.T, self.n_symbols), 0.5),
                                columns=self.symbols)
        f = frame.reset_index(drop=True)
        # Add missing symbol columns
        for sym in self.symbols:
            if sym not in f.columns:
                f[sym] = 0.5
        return f[self.symbols]

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=int)
        if a.shape != (self.n_symbols,):
            raise ValueError(f"action shape {a.shape} != ({self.n_symbols},)")
        if np.any(a < 0) or np.any(a > self.config.position_tier_count):
            raise ValueError(
                f"action out of range [0, {self.config.position_tier_count}]"
            )
        tiers = a.astype(float) / float(self.config.position_tier_count)
        return tiers * self.config.max_position_pct

    def _cost_bps(self, symbol: str) -> float:
        if self.cost_model is None:
            return self.config.default_cost_bps
        for name in ("estimate_bps", "bps_for"):
            fn = getattr(self.cost_model, name, None)
            if callable(fn):
                try:
                    return float(fn(symbol))
                except Exception:
                    continue
        return self.config.default_cost_bps

    def _build_observation(self) -> dict[str, np.ndarray]:
        t = min(self._step_idx, self.T - 1)
        feats = self.features.iloc[t].to_numpy(dtype=np.float32)
        regime = self.regime_probs.iloc[t].to_numpy(dtype=np.float32)
        meta = self.meta_probs.iloc[t].to_numpy(dtype=np.float32)
        gross = float(np.sum(np.abs(self._weights)))
        net = float(np.sum(self._weights))
        dd = 0.0 if self._peak_nav <= 0 else (self._peak_nav - self._nav) / self._peak_nav
        daily_pnl = self._returns_hist[-1] if self._returns_hist else 0.0
        stats = np.array([
            self._nav / self.config.initial_capital, gross, net, dd, daily_pnl,
        ], dtype=np.float32)
        # Calendar: normalized step position + sin of step for seasonality
        calendar = np.array([
            float(t) / max(1, self.T),
            float(np.sin(2 * np.pi * t / max(1, self.T))),
        ], dtype=np.float32)
        return {
            "features": np.nan_to_num(feats),
            "positions": self._weights.astype(np.float32),
            "meta_probs": np.clip(meta, 0.0, 1.0),
            "regime_probs": np.clip(regime, 0.0, 1.0),
            "portfolio_stats": stats,
            "calendar": calendar,
        }

    def _rolling_sharpe(self) -> float:
        window = self.config.sharpe_window
        if len(self._returns_hist) < 2:
            return 0.0
        tail = np.asarray(self._returns_hist[-window:], dtype=float)
        mean = tail.mean()
        std = tail.std(ddof=1)
        if std <= 0:
            return 0.0
        return float(mean / std)

    # ── gym API ───────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_idx = 0
        self._weights = np.zeros(self.n_symbols, dtype=float)
        self._nav = float(self.config.initial_capital)
        self._peak_nav = self._nav
        self._returns_hist = []
        self._nav_hist = [self._nav]
        return self._build_observation(), {}

    def step(self, action):
        new_weights = self._action_to_weights(action)
        turnover = float(np.sum(np.abs(new_weights - self._weights)))

        # Commission / turnover cost (as fraction of NAV)
        cost_bps = np.mean([self._cost_bps(s) for s in self.symbols]) if self.symbols else 0.0
        commission_cost = turnover * (cost_bps / 10_000.0)

        # Advance one step: returns from t → t+1
        t = self._step_idx
        next_t = min(t + 1, self.T - 1)
        price_t = self._closes[t]
        price_next = self._closes[next_t]
        raw_returns = np.divide(
            price_next - price_t, price_t,
            out=np.zeros_like(price_t, dtype=float), where=price_t != 0,
        )
        pnl_pct = float(np.dot(new_weights, raw_returns)) - commission_cost

        prev_nav = self._nav
        self._nav = prev_nav * (1.0 + pnl_pct)
        self._returns_hist.append(pnl_pct)
        self._nav_hist.append(self._nav)
        if self._nav > self._peak_nav:
            self._peak_nav = self._nav
        drawdown = (self._peak_nav - self._nav) / self._peak_nav if self._peak_nav > 0 else 0.0

        self._weights = new_weights
        self._step_idx = next_t

        # Reward
        sharpe_reward = self._rolling_sharpe()
        dd_penalty = 0.0
        if drawdown > self.config.drawdown_threshold:
            dd_penalty = (
                self.config.drawdown_penalty_lambda
                * (drawdown - self.config.drawdown_threshold) ** 2
            )
        turnover_penalty = self.config.turnover_penalty_mu * turnover
        reward = float(sharpe_reward - dd_penalty - turnover_penalty)
        if not np.isfinite(reward):
            reward = 0.0

        terminated = bool(drawdown > self.config.drawdown_limit)
        max_steps = self.config.max_steps
        truncated = bool(
            self._step_idx >= self.T - 1
            or (max_steps is not None and self._step_idx >= max_steps)
        )

        info = {
            "nav": self._nav,
            "drawdown": drawdown,
            "turnover": turnover,
            "sharpe": sharpe_reward,
            "commission_cost": commission_cost,
            "pnl_pct": pnl_pct,
        }
        return self._build_observation(), reward, terminated, truncated, info

    def render(self):  # pragma: no cover - textual only
        print(
            f"step={self._step_idx} nav={self._nav:.2f} "
            f"weights={self._weights.round(3)} peak={self._peak_nav:.2f}"
        )


# ── Factory ───────────────────────────────────────────────────────────────

def create_env_from_historical(
    symbols: Sequence[str],
    start: datetime | str,
    end: datetime | str,
    data_loader: Callable[[str, Any, Any], pd.DataFrame],
    *,
    feature_loader: Callable[..., pd.DataFrame] | None = None,
    **env_kwargs: Any,
) -> TradingEnv:
    """Build a ``TradingEnv`` by loading OHLCV data + features via callables.

    ``data_loader(symbol, start, end)`` must return a DataFrame with a
    ``close`` column indexed by timestamp. All symbols must cover the same
    index. ``feature_loader`` is optional; when omitted, a flat zero-feature
    frame is constructed.
    """
    bars: dict[str, pd.DataFrame] = {}
    ref_index: pd.Index | None = None
    for sym in symbols:
        df = data_loader(sym, start, end)
        if ref_index is None:
            ref_index = df.index
        bars[sym] = df
    if ref_index is None:
        raise ValueError("no data loaded")
    features = (
        feature_loader(symbols, start, end) if feature_loader is not None
        else pd.DataFrame({"const": np.ones(len(ref_index))})
    )
    return TradingEnv(bars=bars, features=features, **env_kwargs)


# ── Validator ─────────────────────────────────────────────────────────────

class EnvValidator:
    """Quick sanity checks for an initialized ``TradingEnv``."""

    @staticmethod
    def validate_reset(env: TradingEnv) -> list[str]:
        errors: list[str] = []
        obs, info = env.reset()
        for key, arr in obs.items():
            if not np.all(np.isfinite(arr)):
                errors.append(f"reset: non-finite values in obs[{key!r}]")
        return errors

    @staticmethod
    def validate_action(env: TradingEnv, action: np.ndarray) -> list[str]:
        errors: list[str] = []
        if not env.action_space.contains(np.asarray(action, dtype=np.int64)):
            errors.append("action not in action_space")
        return errors

    @staticmethod
    def validate_step(env: TradingEnv, n_steps: int = 10) -> list[str]:
        errors: list[str] = []
        env.reset()
        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            if not np.isfinite(reward):
                errors.append(f"non-finite reward: {reward}")
            for key, arr in obs.items():
                if not np.all(np.isfinite(arr)):
                    errors.append(f"step: non-finite in obs[{key!r}]")
                    break
            if term or trunc:
                break
        return errors

    @classmethod
    def full_check(cls, env: TradingEnv) -> dict[str, list[str]]:
        return {
            "reset": cls.validate_reset(env),
            "step": cls.validate_step(env),
        }
