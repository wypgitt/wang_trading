"""
Momentum Signals (design doc §4.1)

Two flavors:
    - TimeSeriesMomentumSignal:   per-asset, multi-lookback. Long if the
                                  weighted-average lookback return is
                                  positive, short if negative.
    - CrossSectionalMomentumSignal: panel of symbols → top/bottom decile
                                    long/short with a 1-month skip to avoid
                                    short-term reversal.

Source: Jegadeesh & Titman (1993), AQR, Chan. See AFML Ch. 17 for the
sign-of-return definition and Narang §Alpha Model for the "many modest
signals combined" thesis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.signal_battery.base_signal import BaseSignalGenerator, Signal


# ---------------------------------------------------------------------------
# Time-series momentum
# ---------------------------------------------------------------------------

class TimeSeriesMomentumSignal(BaseSignalGenerator):
    """
    Per-asset time-series momentum across multiple lookbacks.

    Pipeline:
      1. For each lookback L_i, compute r_i = close[t] / close[t - L_i] - 1.
      2. Normalise each r_i to a z-score using the asset's rolling history
         of non-overlapping L_i-returns (trailing window of ``history_window``).
      3. Aggregate across lookbacks with configurable weights (equal by
         default), clipping each normalised contribution to [-1, 1].
      4. Emit a Signal with side = sign(aggregate), confidence = |aggregate|.

    Parameters (via ``params``):
        lookbacks:       Sequence of lookback lengths in bars.
                         Default [21, 63, 126, 252] (≈ 1, 3, 6, 12 months).
        weights:         Optional parallel weights. Defaults to equal.
        history_window:  Rolling-history size used to standardise each r_i.
                         Default 252 bars.
        min_history:     Minimum bars required before emitting signals.
                         Default max(lookbacks) + history_window.
    """

    REQUIRED_COLUMNS = ("close",)
    DEFAULT_LOOKBACKS = (21, 63, 126, 252)

    def __init__(self, name: str = "ts_momentum", params: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, params=params)
        lookbacks = tuple(self.params.get("lookbacks", self.DEFAULT_LOOKBACKS))
        if not lookbacks or any(l < 1 for l in lookbacks):
            raise ValueError("lookbacks must be a non-empty sequence of positive ints")
        self._lookbacks = lookbacks

        weights = self.params.get("weights")
        if weights is None:
            self._weights = np.ones(len(lookbacks)) / len(lookbacks)
        else:
            w = np.asarray(weights, dtype=float)
            if len(w) != len(lookbacks) or (w < 0).any() or w.sum() == 0:
                raise ValueError("weights must be non-negative and same length as lookbacks")
            self._weights = w / w.sum()

        self._history_window = int(self.params.get("history_window", 252))
        if self._history_window < 10:
            raise ValueError("history_window must be >= 10")

        self._min_history = int(
            self.params.get("min_history", max(self._lookbacks) + self._history_window)
        )

    # ------------------------------------------------------------------ API --

    def generate(
        self,
        bars: pd.DataFrame | None = None,
        symbol: str = "UNKNOWN",
        **kwargs: Any,
    ) -> list[Signal]:
        """
        Emit one Signal per bar from ``_min_history`` onward.

        Args:
            bars:   Bars DataFrame with a ``close`` column and datetime index.
            symbol: Instrument ticker stamped on every Signal.

        Returns:
            List of Signal objects (possibly empty if the history is short).
        """
        if bars is None:
            raise ValueError(f"{self.name}: bars DataFrame is required")
        self.validate_input(bars)
        close = bars["close"].astype(float)
        if len(close) < self._min_history:
            return []

        # Per-lookback Sharpe-like score: L-bar return normalized by the
        # expected L-bar-return volatility (daily vol × sqrt(L)). This gives
        # a persistent uptrend a consistently positive score — unlike
        # subtracting a rolling mean, which would centre the trend on itself.
        daily_ret = close.pct_change(periods=1)
        daily_vol = daily_ret.rolling(
            window=self._history_window, min_periods=self._history_window,
        ).std(ddof=0)

        lookback_frames: list[pd.Series] = []
        for L in self._lookbacks:
            r_L = close.pct_change(periods=L)
            L_vol = daily_vol * np.sqrt(L)
            z = r_L / L_vol.replace(0.0, np.nan)
            # Clip Sharpe-like scores to ±3 (extreme) and rescale to ~[-1, 1].
            lookback_frames.append(z.clip(-3.0, 3.0) / 3.0)

        z_matrix = pd.concat(lookback_frames, axis=1, sort=False)
        z_matrix.columns = [f"z_{L}" for L in self._lookbacks]

        signals: list[Signal] = []
        for t, row in z_matrix.iterrows():
            if row.isna().any():
                continue
            # Weighted aggregate across lookbacks.
            aggregate = float(np.dot(row.to_numpy(dtype=float), self._weights))
            # Clip to [-1, 1] (already close, but guards against float drift).
            aggregate = float(np.clip(aggregate, -1.0, 1.0))
            side = int(np.sign(aggregate))
            confidence = float(abs(aggregate))
            # Zero-magnitude aggregate → neutral signal is still meaningful.
            if side == 0:
                confidence = 0.0

            signals.append(
                Signal(
                    timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                    symbol=symbol,
                    family=self.name,
                    side=side,
                    confidence=confidence,
                    metadata={
                        "lookbacks": list(self._lookbacks),
                        "weights": self._weights.tolist(),
                        "z_scores": {f"z_{L}": float(row[f"z_{L}"]) for L in self._lookbacks},
                        "aggregate": aggregate,
                    },
                )
            )
        return signals


# ---------------------------------------------------------------------------
# Cross-sectional momentum
# ---------------------------------------------------------------------------

class CrossSectionalMomentumSignal(BaseSignalGenerator):
    """
    Panel-level 12-month momentum with a 1-month skip.

    For a given rebalance timestamp t:
      - momentum = close[t - skip] / close[t - lookback] - 1
        (skip-month ≡ exclude the most recent month to avoid short-term
        reversal).
      - Rank symbols by momentum, compute percentile rank in [0, 1].
      - Top decile (rank >= 0.9) → long (+1); bottom decile (<= 0.1) → short.
      - Middle → neutral (0).
      - Confidence = |rank - 0.5| * 2 (highest for extreme ranks).

    Requires >= ``min_universe_size`` symbols (default 20) at the rebalance
    time. Symbols with insufficient history are dropped with a warning
    before the ranking.

    Parameters (via ``params``):
        lookback_bars:       Total lookback. Default 252 (≈ 12 months).
        skip_bars:           Most-recent bars to exclude. Default 21 (≈ 1m).
        top_decile:          Long threshold (quantile). Default 0.9.
        bottom_decile:       Short threshold (quantile). Default 0.1.
        min_universe_size:   Minimum symbols needed to rank.
    """

    REQUIRED_COLUMNS = ("close",)

    def __init__(
        self,
        name: str = "cs_momentum",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        self._lookback = int(self.params.get("lookback_bars", 252))
        self._skip = int(self.params.get("skip_bars", 21))
        self._top = float(self.params.get("top_decile", 0.9))
        self._bottom = float(self.params.get("bottom_decile", 0.1))
        self._min_universe = int(self.params.get("min_universe_size", 20))

        if self._lookback <= self._skip:
            raise ValueError("lookback_bars must exceed skip_bars")
        if not (0.5 < self._top < 1.0):
            raise ValueError("top_decile must be in (0.5, 1.0)")
        if not (0.0 < self._bottom < 0.5):
            raise ValueError("bottom_decile must be in (0.0, 0.5)")
        if self._min_universe < 2:
            raise ValueError("min_universe_size must be >= 2")

    # ------------------------------------------------------------------ API --

    def generate(
        self,
        bars: pd.DataFrame | None = None,
        *,
        panel: dict[str, pd.DataFrame] | None = None,
        timestamp=None,
        **kwargs: Any,
    ) -> list[Signal]:
        """
        Emit long/short signals at a single rebalance timestamp.

        Unlike TimeSeriesMomentumSignal, CS momentum is inherently a
        cross-sectional snapshot; call once per rebalance date.

        Args:
            bars:      Ignored (present for interface symmetry). Use
                       ``panel`` instead.
            panel:     Dict mapping symbol → bars DataFrame (with ``close``).
            timestamp: Rebalance time. If None, the latest shared timestamp
                       across the panel is used.
        """
        if not panel:
            raise ValueError(
                "CrossSectionalMomentumSignal requires a `panel` dict of "
                "{symbol: bars_df}"
            )
        if len(panel) < self._min_universe:
            raise ValueError(
                f"universe too small: {len(panel)} < "
                f"{self._min_universe} required symbols"
            )

        # Validate each DataFrame has the right column.
        for sym, df in panel.items():
            if "close" not in df.columns:
                raise ValueError(f"panel[{sym!r}] missing `close` column")

        # Build an aligned close matrix.
        closes = pd.DataFrame(
            {sym: df["close"].astype(float) for sym, df in panel.items()}
        ).sort_index()

        if timestamp is None:
            timestamp = closes.index[-1]
        else:
            if timestamp not in closes.index:
                # Use the most recent bar on/before the requested timestamp.
                locs = closes.index[closes.index <= timestamp]
                if len(locs) == 0:
                    raise ValueError(
                        f"no bars at or before timestamp={timestamp}"
                    )
                timestamp = locs[-1]

        # Per-symbol momentum with the skip-month.
        t_end_idx = closes.index.get_loc(timestamp)
        if t_end_idx < self._lookback:
            raise ValueError(
                f"not enough history before {timestamp} to compute "
                f"{self._lookback}-bar momentum"
            )
        end_pos = t_end_idx - self._skip
        start_pos = t_end_idx - self._lookback
        if end_pos <= start_pos:
            raise ValueError("lookback/skip configuration leaves no measurement window")

        start_prices = closes.iloc[start_pos]
        end_prices = closes.iloc[end_pos]

        # Drop symbols with any missing price in the window.
        momentum = (end_prices / start_prices - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        if len(momentum) < self._min_universe:
            raise ValueError(
                f"after dropping missing-data symbols, universe is "
                f"{len(momentum)} < {self._min_universe}"
            )

        # Percentile rank in [0, 1]. pandas.rank(pct=True) handles ties well.
        ranks = momentum.rank(pct=True)

        signals: list[Signal] = []
        for sym, rank in ranks.items():
            if rank >= self._top:
                side = 1
            elif rank <= self._bottom:
                side = -1
            else:
                side = 0
            confidence = float(min(1.0, max(0.0, abs(rank - 0.5) * 2.0)))
            if side == 0:
                confidence = confidence * 0.0  # neutral positions carry zero conviction
            signals.append(
                Signal(
                    timestamp=timestamp.to_pydatetime()
                    if hasattr(timestamp, "to_pydatetime")
                    else timestamp,
                    symbol=str(sym),
                    family=self.name,
                    side=side,
                    confidence=confidence,
                    metadata={
                        "momentum_12m_skip_1m": float(momentum[sym]),
                        "percentile_rank": float(rank),
                        "lookback_bars": self._lookback,
                        "skip_bars": self._skip,
                    },
                )
            )
        return signals
