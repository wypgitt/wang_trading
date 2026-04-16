"""
Volatility Signals (Sinclair — Volatility Trading)

Two pieces:
    - VolatilityRiskPremiumSignal  : IV vs RV gap → short-vol / long-vol
                                     regime signal, plus a regime_modifier
                                     meta-field used by the meta-labeler to
                                     amplify/mute momentum or mean-reversion
                                     signals.
    - VolRegimeClassifier          : rolling-percentile vol regime labels
                                     (low / normal / high) + transition stats.

Design doc §4.7.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.signal_battery.base_signal import BaseSignalGenerator, Signal


# ---------------------------------------------------------------------------
# VolatilityRiskPremiumSignal
# ---------------------------------------------------------------------------

class VolatilityRiskPremiumSignal(BaseSignalGenerator):
    """
    VRP = IV − RV. Rolling-percentile-rank regime classifier.

    Input: bars DataFrame with ``iv`` (implied vol) and ``rv`` (realized vol)
    columns, both expressed in the same units (e.g. annualized fractions).

    Signal:
        rank > high_percentile  →  side = -1  (short vol: sell premium)
        rank < low_percentile   →  side = +1  (long vol: buy protection)
        otherwise               →  no signal emitted

    Confidence is the distance from the 50th percentile, rescaled to
    [0, 1]: 100th percentile → 1.0, 50th → 0.0.

    Metadata includes ``regime_modifier``, a dict of family→multiplier that
    the meta-labeler / bet sizer can apply to other signal families:
        high VRP → {"momentum": 1.2, "mean_reversion": 1.0}
        low VRP  → {"momentum": 1.0, "mean_reversion": 1.2}
        neutral  → all 1.0

    Parameters (via ``params``):
        vrp_lookback:    Rolling window for percentile rank (default 30).
        high_percentile: Upper cutoff for short-vol regime (default 75).
        low_percentile:  Lower cutoff for long-vol regime (default 25).
        boost:           Multiplier for the "favored" family (default 1.2).
    """

    REQUIRED_COLUMNS = ("iv", "rv")
    DEFAULT_PARAMS = {
        "vrp_lookback": 30,
        "high_percentile": 75.0,
        "low_percentile": 25.0,
        "boost": 1.2,
    }

    def __init__(
        self,
        name: str = "vrp",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)

        if int(self.params["vrp_lookback"]) < 5:
            raise ValueError("vrp_lookback must be >= 5")
        if not (0 < self.params["low_percentile"] < 50 < self.params["high_percentile"] < 100):
            raise ValueError(
                "need 0 < low_percentile < 50 < high_percentile < 100"
            )
        if float(self.params["boost"]) <= 1.0:
            raise ValueError("boost must be > 1.0 to be meaningful")

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

        iv = bars["iv"].astype(float)
        rv = bars["rv"].astype(float)
        vrp = (iv - rv).rename("vrp")

        lookback = int(self.params["vrp_lookback"])
        hi = float(self.params["high_percentile"])
        lo = float(self.params["low_percentile"])
        boost = float(self.params["boost"])

        # Rolling percentile rank: for each bar, what fraction of the trailing
        # ``lookback`` VRP values is <= the current value? Reported as a percent.
        def _pct_rank(window: np.ndarray) -> float:
            latest = window[-1]
            if not np.isfinite(latest):
                return np.nan
            prior = window[np.isfinite(window)]
            if len(prior) < 2:
                return np.nan
            return float((prior <= latest).sum() / len(prior) * 100.0)

        pct_rank = vrp.rolling(window=lookback, min_periods=lookback).apply(
            _pct_rank, raw=True
        )

        signals: list[Signal] = []
        for t in vrp.index:
            v = vrp.loc[t]
            r = pct_rank.loc[t]
            if not (np.isfinite(v) and np.isfinite(r)):
                continue

            if r > hi:
                side = -1  # short vol
                confidence = float(np.clip((r - 50.0) / 50.0, 0.0, 1.0))
                regime = "high_vrp"
                modifier = {"momentum": boost, "mean_reversion": 1.0}
            elif r < lo:
                side = 1   # long vol
                confidence = float(np.clip((50.0 - r) / 50.0, 0.0, 1.0))
                regime = "low_vrp"
                modifier = {"momentum": 1.0, "mean_reversion": boost}
            else:
                # Neutral: emit no signal; the family's contribution for this
                # bar is simply absent (implicitly "hold").
                continue

            signals.append(
                Signal(
                    timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                    symbol=symbol,
                    family=self.name,
                    side=int(side),
                    confidence=confidence,
                    metadata={
                        "vrp": float(v),
                        "vrp_percentile": float(r),
                        "regime": regime,
                        "regime_modifier": modifier,
                    },
                )
            )
        return signals


# ---------------------------------------------------------------------------
# VolRegimeClassifier
# ---------------------------------------------------------------------------

class VolRegimeClassifier:
    """
    Rolling-percentile regime labels from a GARCH conditional vol series.

    classify(garch_vol) → DataFrame with columns:
        regime         : {'low_vol', 'normal_vol', 'high_vol'}
        percentile     : rolling percentile rank of vol (0..100)
        transition     : bool — True if regime changed from prior bar

    Plus ``transition_probabilities()`` which returns a 3×3 DataFrame of
    empirically observed transition frequencies between regimes.

    Parameters:
        window:          Rolling window for percentile ranking (default 60).
        low_percentile:  Cutoff for the 'low_vol' bucket (default 25).
        high_percentile: Cutoff for the 'high_vol' bucket (default 75).
    """

    LABELS = ("low_vol", "normal_vol", "high_vol")

    def __init__(
        self,
        window: int = 60,
        low_percentile: float = 25.0,
        high_percentile: float = 75.0,
    ) -> None:
        if window < 5:
            raise ValueError("window must be >= 5")
        if not (0 < low_percentile < high_percentile < 100):
            raise ValueError(
                "need 0 < low_percentile < high_percentile < 100"
            )
        self.window = int(window)
        self.low = float(low_percentile)
        self.high = float(high_percentile)
        self._last_result: pd.DataFrame | None = None

    # ------------------------------------------------------------------ API --

    def classify(self, garch_vol: pd.Series) -> pd.DataFrame:
        """Label each bar with its vol-regime bucket."""
        if not isinstance(garch_vol, pd.Series):
            raise ValueError("garch_vol must be a pandas Series")

        def _pct_rank(window: np.ndarray) -> float:
            latest = window[-1]
            if not np.isfinite(latest):
                return np.nan
            prior = window[np.isfinite(window)]
            if len(prior) < 2:
                return np.nan
            return float((prior <= latest).sum() / len(prior) * 100.0)

        pct = garch_vol.rolling(
            window=self.window, min_periods=self.window
        ).apply(_pct_rank, raw=True)

        regimes = pd.Series(None, index=garch_vol.index, dtype="object")
        regimes = regimes.mask(pct <= self.low, "low_vol")
        regimes = regimes.mask((pct > self.low) & (pct < self.high), "normal_vol")
        regimes = regimes.mask(pct >= self.high, "high_vol")

        # Build transition mask elementwise so NaN-vs-label comparisons don't
        # return ambiguous NA booleans.
        reg_vals = regimes.to_numpy(dtype=object)
        prev = np.roll(reg_vals, 1)
        transitions = np.array(
            [
                bool(curr is not None and prev_v is not None and curr != prev_v)
                for curr, prev_v in zip(reg_vals, prev)
            ],
            dtype=bool,
        )
        transitions[0] = False

        out = pd.DataFrame(
            {"regime": regimes, "percentile": pct, "transition": transitions},
            index=garch_vol.index,
        )
        self._last_result = out
        return out

    def transition_probabilities(self) -> pd.DataFrame:
        """
        Empirical transition matrix from the most recent ``classify`` call.

        Returns a 3×3 DataFrame indexed by "from" regime, columns "to"
        regime, values are frequencies (rows sum to ~1.0). Returns an
        all-NaN frame if no classification has been run yet.
        """
        if self._last_result is None:
            raise RuntimeError("call classify() first")

        regimes = self._last_result["regime"].dropna()
        if len(regimes) < 2:
            return pd.DataFrame(
                np.nan, index=list(self.LABELS), columns=list(self.LABELS),
            )

        from_r = regimes.iloc[:-1].to_numpy()
        to_r = regimes.iloc[1:].to_numpy()
        labels = list(self.LABELS)
        mtx = pd.DataFrame(0.0, index=labels, columns=labels)
        for f, t in zip(from_r, to_r):
            if f in labels and t in labels:
                mtx.loc[f, t] += 1.0

        # Normalise each row to a probability.
        row_sums = mtx.sum(axis=1)
        non_zero = row_sums > 0
        mtx.loc[non_zero, :] = mtx.loc[non_zero, :].div(
            row_sums[non_zero], axis=0
        )
        mtx.loc[~non_zero, :] = np.nan
        return mtx
