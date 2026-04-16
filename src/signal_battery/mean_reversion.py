"""
Mean Reversion Signals (Chan — Quantitative Trading / Algorithmic Trading)

Implements an Ornstein-Uhlenbeck-based mean reversion signal:

    1. Fit an O-U process to the price series to estimate the half-life
       of mean reversion.
    2. Require the series to be stationary (ADF p < 0.05) and the half-life
       to be within a tradeable band.
    3. Compute a rolling z-score of the price with window = half-life.
    4. Enter in the direction opposite to the deviation when |z| > entry;
       exit (side = 0) when |z| < exit.

Also exposes ``compute_bollinger_zscore`` as a simpler alternative for
comparison / ensembling.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.signal_battery.base_signal import BaseSignalGenerator, Signal


# ---------------------------------------------------------------------------
# O-U half-life
# ---------------------------------------------------------------------------

def compute_ou_halflife(series: pd.Series) -> tuple[float, float]:
    """
    Estimate the Ornstein-Uhlenbeck half-life of mean reversion.

    Fits the regression  delta_y_t = lambda * y_{t-1} + epsilon  and reports
    ``half_life = -log(2) / lambda``. If lambda is non-negative (no mean
    reversion), returns (inf, adf_pvalue).

    Also runs an Augmented Dickey-Fuller test on the series so the caller
    can filter out non-stationary inputs in a single pass.

    Args:
        series: Price (or spread) series. NaNs are dropped.

    Returns:
        (half_life_bars, adf_pvalue). ``half_life_bars`` is +inf if the
        regression slope is >= 0 or the estimate is numerically degenerate.
    """
    y = series.dropna().astype(float).to_numpy()
    if len(y) < 10:
        return float("inf"), 1.0

    y_lag = y[:-1]
    dy = np.diff(y)

    # Guard against constant input (var(y_lag) = 0).
    if np.std(y_lag) == 0.0:
        return float("inf"), 1.0

    # Slope of dy on y_lag via closed-form OLS (faster than polyfit).
    y_lag_mean = y_lag.mean()
    dy_mean = dy.mean()
    num = np.sum((y_lag - y_lag_mean) * (dy - dy_mean))
    den = np.sum((y_lag - y_lag_mean) ** 2)
    if den == 0.0:
        return float("inf"), 1.0
    lam = num / den

    try:
        adf_pvalue = float(adfuller(y, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        adf_pvalue = 1.0

    if lam >= 0 or not math.isfinite(lam):
        return float("inf"), adf_pvalue
    half_life = -math.log(2.0) / lam
    if not math.isfinite(half_life) or half_life <= 0:
        return float("inf"), adf_pvalue
    return float(half_life), adf_pvalue


# ---------------------------------------------------------------------------
# Bollinger-band z-score (simple alternative)
# ---------------------------------------------------------------------------

def compute_bollinger_zscore(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """
    Bollinger z-score: (close - SMA) / (num_std * rolling_std).

    At ±1 the price touches the Bollinger bands. At 0 the price sits on the
    SMA. With ``num_std=1`` this reduces to the plain rolling z-score.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    if num_std <= 0:
        raise ValueError("num_std must be positive")
    sma = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    denom = (num_std * sd).replace(0.0, np.nan)
    return ((close - sma) / denom).rename(f"bb_z_{window}")


# ---------------------------------------------------------------------------
# Mean reversion signal
# ---------------------------------------------------------------------------

class MeanReversionSignal(BaseSignalGenerator):
    """
    O-U-based mean reversion signal.

    Workflow per ``generate`` call (one instrument at a time):
      1. Fit O-U half-life on the full supplied series. If the series is
         non-stationary (ADF p > ``adf_pvalue``) or the half-life falls
         outside [min_halflife, max_halflife], return an empty list.
      2. Round the half-life to an integer window W.
      3. Compute rolling z-score (price − SMA_W) / rolling_std_W.
      4. For each bar (from bar W onward):
           |z| > entry_threshold  → emit (side = -sign(z),
                                          confidence = min(|z|/4, 1)).
           |z| < exit_threshold   → emit (side = 0, confidence = 0).
           otherwise              → skip (hold the current position).

    Parameters (via ``params``):
        entry_threshold: |z| cutoff for entry signals (default 2.0).
        exit_threshold:  |z| cutoff for exit signals (default 0.5).
        min_halflife:    Minimum tradeable half-life in bars (default 1).
        max_halflife:    Maximum tradeable half-life in bars (default 100).
        adf_pvalue:      Maximum ADF p-value to accept the series as
                         stationary (default 0.05).
    """

    REQUIRED_COLUMNS = ("close",)
    DEFAULT_PARAMS = {
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
        "min_halflife": 1.0,
        "max_halflife": 100.0,
        "adf_pvalue": 0.05,
    }

    def __init__(
        self,
        name: str = "mean_reversion",
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
        symbol: str = "UNKNOWN",
        **kwargs: Any,
    ) -> list[Signal]:
        if bars is None:
            raise ValueError(f"{self.name}: bars DataFrame is required")
        self.validate_input(bars)
        close = bars["close"].astype(float)

        half_life, adf_p = compute_ou_halflife(close)
        if (
            not math.isfinite(half_life)
            or half_life < self.params["min_halflife"]
            or half_life > self.params["max_halflife"]
            or adf_p > self.params["adf_pvalue"]
        ):
            return []

        window = max(2, int(round(half_life)))
        sma = close.rolling(window=window, min_periods=window).mean()
        sd = close.rolling(window=window, min_periods=window).std(ddof=0)
        z = (close - sma) / sd.replace(0.0, np.nan)

        entry = float(self.params["entry_threshold"])
        exit_ = float(self.params["exit_threshold"])

        signals: list[Signal] = []
        for t, zt in z.items():
            if not np.isfinite(zt):
                continue
            abs_z = abs(float(zt))
            if abs_z > entry:
                # Mean reversion: fade the deviation.
                side = -int(np.sign(zt))
                confidence = float(min(abs_z / 4.0, 1.0))
                signals.append(
                    Signal(
                        timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                        symbol=symbol,
                        family=self.name,
                        side=side,
                        confidence=confidence,
                        metadata={
                            "half_life": float(half_life),
                            "window": int(window),
                            "z_score": float(zt),
                            "adf_pvalue": float(adf_p),
                            "event": "entry",
                        },
                    )
                )
            elif abs_z < exit_:
                signals.append(
                    Signal(
                        timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                        symbol=symbol,
                        family=self.name,
                        side=0,
                        confidence=0.0,
                        metadata={
                            "half_life": float(half_life),
                            "window": int(window),
                            "z_score": float(zt),
                            "adf_pvalue": float(adf_p),
                            "event": "exit",
                        },
                    )
                )
            # else: "hold" — no signal emitted.
        return signals
