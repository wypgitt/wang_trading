"""
Statistical Arbitrage / Pairs Trading (Chan — Algorithmic Trading)

Builds cointegration-based pairs trades with a Kalman-filtered hedge ratio
that adapts over time. Design doc §4.3.

Pipeline:
    1. Scan for cointegrated pairs (Engle-Granger, rolling).
    2. Filter by spread half-life (too fast or too slow = untradeable).
    3. For each surviving pair, track the hedge ratio with a Kalman filter
       so the spread stays stationary as the relationship drifts.
    4. Generate z-score-based entry/exit signals on the spread.

Also exposes ``johansen_cointegration`` for multi-asset baskets (useful
for ETF-vs-components pairs and sector-neutral constructions).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.signal_battery.base_signal import BaseSignalGenerator, Signal
from src.signal_battery.mean_reversion import compute_ou_halflife


# ---------------------------------------------------------------------------
# Cointegration scanners
# ---------------------------------------------------------------------------

def find_cointegrated_pairs(
    prices_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
) -> list[tuple[str, str, float]]:
    """
    Engle-Granger test for every pair of columns in ``prices_df``.

    Args:
        prices_df:         Wide-format DataFrame, one column per symbol.
        p_value_threshold: Maximum Engle-Granger p-value to keep.

    Returns:
        List of (sym_a, sym_b, pvalue) tuples with pvalue < threshold,
        sorted ascending by pvalue (strongest cointegration first).
    """
    if prices_df.shape[1] < 2:
        return []
    cols = list(prices_df.columns)
    results: list[tuple[str, str, float]] = []
    for a, b in combinations(cols, 2):
        ya = prices_df[a].dropna()
        yb = prices_df[b].dropna()
        common = ya.index.intersection(yb.index)
        if len(common) < 30:
            continue
        try:
            _, pval, _ = coint(ya.loc[common].values, yb.loc[common].values)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug(f"coint failed for {a}/{b}: {exc}")
            continue
        if np.isfinite(pval) and pval < p_value_threshold:
            results.append((a, b, float(pval)))
    results.sort(key=lambda t: t[2])
    return results


def johansen_cointegration(
    prices_df: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> dict:
    """
    Johansen cointegration test for a group of assets.

    Args:
        prices_df: Wide-format DataFrame (columns = assets).
        det_order: Deterministic trend term (see statsmodels docs).
        k_ar_diff: Number of lagged differences.

    Returns:
        dict with keys:
            eigenvalues       : array of Johansen eigenvalues (descending)
            eigenvectors      : array of cointegrating vectors (columns)
            trace_stats       : trace statistics (array)
            critical_values   : matrix of 10/5/1% critical values (rows align
                                with trace_stats)
            n_cointegrating   : integer # of cointegrating relationships at
                                the 5% level
    """
    data = prices_df.dropna().to_numpy(dtype=float)
    if data.shape[0] < 50 or data.shape[1] < 2:
        raise ValueError("need >= 2 assets and >= 50 observations for Johansen")

    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    trace = np.asarray(result.lr1, dtype=float)
    cv = np.asarray(result.cvt, dtype=float)  # rows: tested r, cols: 10/5/1%
    # Count rejections at the 5% level (column index 1).
    n_coint = int(np.sum(trace > cv[:, 1]))
    return {
        "eigenvalues": np.asarray(result.eig, dtype=float),
        "eigenvectors": np.asarray(result.evec, dtype=float),
        "trace_stats": trace,
        "critical_values": cv,
        "n_cointegrating": n_coint,
    }


# ---------------------------------------------------------------------------
# Kalman-filtered hedge ratio
# ---------------------------------------------------------------------------

class KalmanFilterHedgeRatio:
    """
    Dynamic hedge ratio via Kalman filter.

    State: [hedge_ratio, intercept], random-walk transition.
    Observation: y_t = hedge_ratio_t * x_t + intercept_t + noise.

    The transition covariance ``delta`` controls how fast the hedge ratio is
    allowed to drift (larger → more responsive, noisier). ``obs_cov`` is
    the observation-noise variance.

    Manual implementation (no pykalman dependency at runtime): the model is
    small enough that two matrix products per step are faster than the
    general-purpose library.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        obs_cov: float = 1.0,
        init_hedge: float = 1.0,
        init_intercept: float = 0.0,
        init_state_cov: float = 1.0,
    ) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive")
        if obs_cov <= 0:
            raise ValueError("obs_cov must be positive")

        # State vector x = [hedge_ratio; intercept].
        self.state = np.array([init_hedge, init_intercept], dtype=float)
        # State covariance P (2x2).
        self.P = init_state_cov * np.eye(2)
        # Transition: random walk → Q = delta/(1-delta) * I (Chan §3).
        self.Q = (delta / max(1.0 - delta, 1e-12)) * np.eye(2)
        self.R = float(obs_cov)

    def update(self, y: float, x: float) -> float:
        """
        Feed a new (y, x) pair, return the updated hedge-ratio estimate.

        Args:
            y: dependent variable observation.
            x: independent variable observation.

        Returns:
            Current best estimate of the hedge ratio (state[0]).
        """
        # Predict — state doesn't change (F = I), but covariance grows by Q.
        self.P = self.P + self.Q

        # Observation row H = [x, 1] so y = H·state + noise. Use 1-D arrays
        # throughout — cleaner than juggling (1,2)/(2,) shapes.
        H = np.array([x, 1.0])
        y_pred = float(H @ self.state)
        innov = y - y_pred
        # Innovation variance S = H·P·Hᵀ + R
        S = float(H @ self.P @ H) + self.R
        if S <= 0 or not np.isfinite(S):
            return float(self.state[0])
        # Kalman gain K = P·Hᵀ / S, shape (2,)
        K = (self.P @ H) / S
        # State update: x += K·innov
        self.state = self.state + K * innov
        # Covariance update: P -= K·(H·P)  (Joseph form would be numerically
        # safer; for a 2×2 system the plain form is fine).
        self.P = self.P - np.outer(K, H @ self.P)
        return float(self.state[0])

    def get_spread(self, y: pd.Series, x: pd.Series) -> pd.Series:
        """
        Run the filter over ``y`` / ``x`` and return the spread series.

        The spread at time t is y_t − hedge_ratio_t * x_t (the intercept is
        absorbed into the spread level). ``y`` and ``x`` are inner-joined on
        their indices.
        """
        df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        spreads = np.empty(len(df))
        hedge_ratios = np.empty(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            hr = self.update(row["y"], row["x"])
            hedge_ratios[i] = hr
            spreads[i] = row["y"] - hr * row["x"]
        return pd.Series(spreads, index=df.index, name="spread")


# ---------------------------------------------------------------------------
# StatArbSignal
# ---------------------------------------------------------------------------

class StatArbSignal(BaseSignalGenerator):
    """
    Pairs-trade signal on a cointegrated (y, x).

    The ``generate`` method takes a two-column bars DataFrame via keyword
    args ``y_series`` and ``x_series`` — each a close-price pandas Series.
    It is independent of the Bar dataclass shape.

    Flow:
      1. Fit Kalman hedge ratio → dynamic spread.
      2. Compute O-U half-life of the spread. If the half-life is outside
         ``(min_halflife, max_halflife)`` no signals are emitted.
      3. Roll a z-score of the spread with window = half-life.
      4. For each bar:
           z  >  entry → short y, long x (side = -1 for y)
           z  < -entry → long y, short x (side = +1 for y)
           |z| < exit  → flatten (side = 0)
           otherwise   → hold (no signal).
      5. Each Signal's metadata includes pair symbols, hedge ratio, spread
         z-score, spread half-life.
    """

    REQUIRED_COLUMNS = ()  # uses kwargs instead of a single bars DataFrame
    DEFAULT_PARAMS = {
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
        "min_halflife": 1.0,
        "max_halflife": 100.0,
        # Reject pairs whose spread isn't stationary. The Kalman filter is
        # flexible enough to make almost any pair's spread look mean-reverting
        # by half-life alone, so we double-check with ADF.
        "adf_pvalue": 0.05,
        # Very small delta keeps the hedge ratio near-static so the Kalman
        # spread reflects the true cointegration residual rather than being
        # absorbed into ratio drift. Chan (Algorithmic Trading §3) uses a
        # similar magnitude. Tests may override for faster adaptation.
        "kalman_delta": 1e-5,
        "kalman_obs_cov": 1.0,
    }

    def __init__(
        self,
        name: str = "stat_arb",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        if self.params["entry_threshold"] <= self.params["exit_threshold"]:
            raise ValueError("entry_threshold must exceed exit_threshold")
        if self.params["min_halflife"] >= self.params["max_halflife"]:
            raise ValueError("min_halflife must be < max_halflife")

    # ------------------------------------------------------------------ API --

    def generate(
        self,
        bars: pd.DataFrame | None = None,
        *,
        y_series: pd.Series | None = None,
        x_series: pd.Series | None = None,
        y_symbol: str = "Y",
        x_symbol: str = "X",
        **kwargs: Any,
    ) -> list[Signal]:
        if y_series is None or x_series is None:
            raise ValueError("StatArbSignal requires `y_series` and `x_series`")
        if y_series.empty or x_series.empty:
            raise ValueError("input series must be non-empty")

        # One Kalman pass → hedge-ratio series + spread.
        spread, hr_series = self._run_kalman(y_series, x_series)
        if spread.empty:
            return []

        # Half-life of the spread — also serves as the z-score window.
        hl, adf_p = compute_ou_halflife(spread)
        if not np.isfinite(hl) or not (
            self.params["min_halflife"] < hl < self.params["max_halflife"]
        ):
            return []
        # Reject pairs whose spread is not actually stationary (e.g. two
        # independent random walks whose Kalman-filtered residual happens
        # to have a plausible-looking half-life).
        if adf_p > float(self.params["adf_pvalue"]):
            return []

        window = max(2, int(round(hl)))
        # Use PRIOR-window statistics (shifted by 1) so a fresh spike is not
        # diluted by being included in its own reference mean/std. This is
        # the standard mean-reversion construction; it also avoids subtle
        # lookahead bleed between a signal and its own denominator.
        prior = spread.shift(1)
        rolling_mean = prior.rolling(window=window, min_periods=window).mean()
        rolling_std = prior.rolling(window=window, min_periods=window).std(ddof=0)
        z = (spread - rolling_mean) / rolling_std.replace(0.0, np.nan)

        entry = float(self.params["entry_threshold"])
        exit_ = float(self.params["exit_threshold"])
        signals: list[Signal] = []
        for t, zt in z.items():
            if not np.isfinite(zt):
                continue
            abs_z = abs(float(zt))
            if abs_z > entry:
                # Spread > 0 means y over-priced relative to x → short y, long x.
                side = -int(np.sign(zt))
                confidence = float(min(abs_z / 4.0, 1.0))
                event = "entry"
            elif abs_z < exit_:
                side = 0
                confidence = 0.0
                event = "exit"
            else:
                continue

            signals.append(
                Signal(
                    timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                    symbol=y_symbol,
                    family=self.name,
                    side=side,
                    confidence=confidence,
                    metadata={
                        "pair": (y_symbol, x_symbol),
                        "hedge_ratio": float(hr_series.get(t, np.nan)),
                        "spread": float(spread.loc[t]),
                        "z_score": float(zt),
                        "half_life": float(hl),
                        "adf_pvalue": float(adf_p),
                        "event": event,
                    },
                )
            )
        return signals

    # ------------------------------------------------------------------ helpers --

    def _run_kalman(
        self, y: pd.Series, x: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Single Kalman pass producing both the spread and per-bar hedge ratio.

        Returns:
            (spread, hedge_ratio_series) aligned on the inner index of y/x.
        """
        kf = KalmanFilterHedgeRatio(
            delta=float(self.params["kalman_delta"]),
            obs_cov=float(self.params["kalman_obs_cov"]),
        )
        df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        n = len(df)
        hr = np.empty(n)
        spread_vals = np.empty(n)
        for i, (_, row) in enumerate(df.iterrows()):
            ratio = kf.update(float(row["y"]), float(row["x"]))
            hr[i] = ratio
            spread_vals[i] = float(row["y"]) - ratio * float(row["x"])
        return (
            pd.Series(spread_vals, index=df.index, name="spread"),
            pd.Series(hr, index=df.index, name="hedge_ratio"),
        )


# ---------------------------------------------------------------------------
# scan_for_pairs
# ---------------------------------------------------------------------------

def scan_for_pairs(
    prices_dict: dict[str, pd.Series],
    max_pairs: int = 30,
    lookback: int = 252,
    p_value_threshold: float = 0.05,
    min_halflife: float = 1.0,
    max_halflife: float = 100.0,
) -> list[tuple[str, str]]:
    """
    Identify tradeable cointegrated pairs.

    For each pair:
      1. Test Engle-Granger cointegration on the trailing ``lookback`` bars.
      2. Compute the Kalman-filtered spread half-life.
      3. Accept pairs with p < threshold and half-life in (min, max).

    Returns the top ``max_pairs`` by cointegration strength.
    """
    if not prices_dict:
        return []

    # Truncate to the trailing ``lookback`` bars on the intersection index.
    prices_df = pd.DataFrame(prices_dict).dropna()
    if len(prices_df) > lookback:
        prices_df = prices_df.iloc[-lookback:]
    if len(prices_df) < 30:
        return []

    raw_pairs = find_cointegrated_pairs(prices_df, p_value_threshold=p_value_threshold)

    tradeable: list[tuple[str, str, float, float]] = []  # (a, b, p, hl)
    for a, b, pval in raw_pairs:
        kf = KalmanFilterHedgeRatio()
        spread = kf.get_spread(prices_df[a], prices_df[b])
        hl, _ = compute_ou_halflife(spread)
        if np.isfinite(hl) and min_halflife < hl < max_halflife:
            tradeable.append((a, b, pval, hl))

    tradeable.sort(key=lambda t: t[2])  # ascending p-value
    return [(a, b) for a, b, _, _ in tradeable[:max_pairs]]
