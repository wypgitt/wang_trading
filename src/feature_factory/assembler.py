"""
Feature Assembler

Orchestrates the full Phase-2 feature pipeline: runs FFD, structural-break,
entropy, microstructure, GARCH, sentiment, on-chain, and classical
indicator blocks; aligns everything on a common index; and enforces
stationarity on any remaining non-stationary column by applying FFD
post-hoc.

The assembler does NOT run the autoencoder — latent features are computed
separately after the matrix is assembled (and, optionally, after training).

Expected ``bars_df`` columns: the Bar dataclass fields
    close, volume, dollar_volume, buy_volume, sell_volume,
    tick_count, bar_duration_seconds
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_factory.entropy import compute_entropy_features
from src.feature_factory.fractional_diff import find_min_d, frac_diff_ffd
from src.feature_factory.microstructure import compute_microstructure_features
from src.feature_factory.structural_breaks import compute_structural_break_features
from src.feature_factory.volatility import compute_volatility_features


# ---------------------------------------------------------------------------
# Classical indicators
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Exponential (Wilder) smoothing.
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.rename(f"rsi_{window}")


def _bollinger_width(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger-band width as (upper - lower) / middle."""
    mid = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    width = (2.0 * num_std * sd) / mid.replace(0.0, np.nan)
    return width.rename(f"bbw_{window}")


def _rolling_return_zscore(close: pd.Series, window: int) -> pd.Series:
    """Z-score of close-to-close return normalised by rolling return vol."""
    ret = close.pct_change()
    mean = ret.rolling(window=window, min_periods=window).mean()
    sd = ret.rolling(window=window, min_periods=window).std(ddof=0)
    z = (ret - mean) / sd.replace(0.0, np.nan)
    return z.rename(f"ret_z_{window}")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "ffd": {
        "columns": ["close", "volume", "dollar_volume"],
        "p_value": 0.05,
        "max_d": 1.0,
    },
    "structural_breaks": {
        "window": 50,
        "min_window_sadf": 20,
        "min_period_chow": 30,
    },
    "entropy": {
        "window": 100,
    },
    "microstructure": {
        "window": 50,
    },
    "volatility": {
        "window": 252,
        "refit_interval": 25,
        "short_window": 5,
        "long_window": 30,
        "vvol_window": 30,
    },
    "classical": {
        "rsi_window": 14,
        "bbw_window": 20,
        "ret_z_windows": [5, 10, 20],
    },
    "stationarity": {
        # Post-hoc FFD only for very non-stationary features.
        "p_value": 0.05,
        "ffd_max_d": 1.0,
        "ffd_threshold": 1e-3,
    },
}


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class FeatureAssembler:
    """
    Compose the full Phase-2 feature matrix from a bars DataFrame.

    Usage:
        assembler = FeatureAssembler()
        X = assembler.assemble(bars_df, sentiment_scores=..., onchain_features=...)
        print(assembler.get_feature_names())
        print(assembler.get_optimal_d_values())
    """

    REQUIRED_BAR_COLS = (
        "close", "volume", "dollar_volume",
        "buy_volume", "sell_volume",
        "tick_count", "bar_duration_seconds",
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Shallow-merge over the default so callers can override a subset.
        merged: dict[str, Any] = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
        if config:
            for key, val in config.items():
                if isinstance(val, dict) and key in merged:
                    merged[key].update(val)
                else:
                    merged[key] = val
        self.config = merged
        self._optimal_d: dict[str, float] = {}
        self._feature_names: list[str] = []

    # -- public API --

    def assemble(
        self,
        bars_df: pd.DataFrame,
        implied_vol: pd.Series | None = None,
        sentiment_scores: pd.DataFrame | None = None,
        onchain_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compose the full Phase-2 feature matrix.

        Runs every feature block (FFD, structural breaks, entropy,
        microstructure, GARCH, sentiment, on-chain, classical), aligns
        all blocks on a common index, drops warmup NaN rows, and runs a
        post-hoc stationarisation pass. Returns the aligned, finite,
        stationary feature matrix.

        Args:
            bars_df:         Input bars DataFrame (requires the Bar
                             dataclass fields).
            implied_vol:     Optional IV series (enables rv_iv_spread).
            sentiment_scores: Optional pre-computed sentiment feature block.
            onchain_features: Optional pre-computed on-chain feature block.

        Returns:
            pd.DataFrame indexed like ``bars_df`` (minus warmup) with one
            row per bar and one column per feature.
        """
        missing = [c for c in self.REQUIRED_BAR_COLS if c not in bars_df.columns]
        if missing:
            raise KeyError(f"bars_df missing columns: {missing}")

        close = bars_df["close"]

        blocks: list[pd.DataFrame] = []

        # 1) FFD price / volume blocks (keeps memory while enforcing stationarity)
        blocks.append(self._ffd_block(bars_df))

        # 2) Structural-break block
        sb_cfg = self.config["structural_breaks"]
        blocks.append(
            compute_structural_break_features(
                close,
                window=sb_cfg["window"],
                min_window_sadf=sb_cfg["min_window_sadf"],
                min_period_chow=sb_cfg["min_period_chow"],
            )
        )

        # 3) Entropy block
        ent_cfg = self.config["entropy"]
        blocks.append(
            compute_entropy_features(close, window=ent_cfg["window"])
        )

        # 4) Microstructure block
        ms_cfg = self.config["microstructure"]
        blocks.append(
            compute_microstructure_features(bars_df, window=ms_cfg["window"])
        )

        # 5) GARCH volatility block
        vol_cfg = self.config["volatility"]
        blocks.append(
            compute_volatility_features(
                close,
                implied_vol=implied_vol,
                window=vol_cfg["window"],
                refit_interval=vol_cfg["refit_interval"],
                short_window=vol_cfg["short_window"],
                long_window=vol_cfg["long_window"],
                vvol_window=vol_cfg["vvol_window"],
            )
        )

        # 6) Classical indicators
        blocks.append(self._classical_block(close))

        # 7) Sentiment / on-chain — optional, joined on the bar index.
        if sentiment_scores is not None:
            blocks.append(
                sentiment_scores.reindex(bars_df.index)
            )
        if onchain_features is not None:
            blocks.append(
                onchain_features.reindex(bars_df.index, method="ffill")
            )

        # Align on intersection; dropna across the full matrix to drop warmup.
        matrix = pd.concat(blocks, axis=1)
        matrix = matrix.dropna(how="any")

        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("feature matrix contains non-finite values")

        # 8) Enforce stationarity on any feature that still fails ADF.
        matrix = self._stationarise(matrix)

        self._feature_names = list(matrix.columns)
        return matrix

    def get_feature_names(self) -> list[str]:
        """Return the column names produced by the most recent ``assemble``."""
        return list(self._feature_names)

    def get_optimal_d_values(self) -> dict[str, float]:
        """Return the optimal FFD d value chosen per column during ``assemble``."""
        return dict(self._optimal_d)

    # -- private helpers --

    def _ffd_block(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """Compute FFD-differenced series for the configured columns."""
        cfg = self.config["ffd"]
        cols: list[pd.Series] = []
        for col in cfg["columns"]:
            if col not in bars_df.columns:
                logger.warning(f"FFD: column {col!r} not in bars_df; skipping")
                continue
            series = bars_df[col].astype(float).dropna()
            if series.empty:
                continue
            d_star = find_min_d(
                series, d_range=(0.0, cfg["max_d"]), p_value=cfg["p_value"]
            )
            self._optimal_d[col] = float(d_star)
            diffed = frac_diff_ffd(series, d=d_star)
            cols.append(diffed.rename(f"ffd_{col}"))
        if not cols:
            return pd.DataFrame(index=bars_df.index)
        return pd.concat(cols, axis=1, sort=False).reindex(bars_df.index)

    def _classical_block(self, close: pd.Series) -> pd.DataFrame:
        cfg = self.config["classical"]
        series = [
            _rsi(close, window=cfg["rsi_window"]),
            _bollinger_width(close, window=cfg["bbw_window"]),
        ]
        for w in cfg["ret_z_windows"]:
            series.append(_rolling_return_zscore(close, window=w))
        return pd.concat(series, axis=1)

    def _stationarise(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Post-hoc stationarity enforcement in two passes.

        Pass 1  For every column whose ADF p-value >= ``p_crit``, apply FFD
                with the smallest d that brings it below threshold. If FFD
                still doesn't help, apply plain first differences.
        Trim    Drop rows with any remaining NaN (introduced by differencing).
        Pass 2  Re-run ADF on the trimmed matrix. Any column still
                non-stationary gets differenced one more time and accepted.

        The two passes are necessary because ADF p-values depend on the
        exact slice of data: a differenced column that passed ADF on its
        500-row post-FFD clean slice might fail after the final ``dropna``
        removes its warmup head. Verifying *after* the trim catches this.
        """
        from statsmodels.tsa.stattools import adfuller

        cfg = self.config["stationarity"]
        threshold = cfg["ffd_threshold"]
        p_crit = cfg["p_value"]

        def _pval(vals: np.ndarray) -> float:
            if len(vals) < 20 or np.std(vals) == 0 or not np.isfinite(vals).all():
                return 1.0
            try:
                return float(adfuller(vals, autolag="AIC")[1])
            except Exception:  # noqa: BLE001
                return 1.0

        # -------- Pass 1: FFD / diff on clearly non-stationary columns -----
        result = matrix.copy()
        for col in matrix.columns:
            values = matrix[col].to_numpy(dtype=float)
            if not np.isfinite(values).all() or values.std() == 0:
                continue
            if _pval(values) < p_crit:
                continue

            d_star = find_min_d(
                matrix[col],
                d_range=(0.05, cfg["ffd_max_d"]),
                p_value=p_crit,
                threshold=threshold,
            )
            diffed = frac_diff_ffd(matrix[col], d=d_star, threshold=threshold)
            diffed_aligned = diffed.reindex(matrix.index)
            self._optimal_d[col] = float(d_star)

            diffed_clean = diffed_aligned.dropna().to_numpy(dtype=float)
            if _pval(diffed_clean) < p_crit:
                result[col] = diffed_aligned
                continue

            # FFD didn't help → first differences.
            first_diff = matrix[col].diff()
            if _pval(first_diff.dropna().to_numpy(dtype=float)) < p_crit:
                result[col] = first_diff
                self._optimal_d[col] = 1.0
                continue

            # Keep the FFD attempt; Pass 2 will try once more after trim.
            result[col] = diffed_aligned

        result = result.dropna(how="any")

        # -------- Pass 2: re-verify on the trimmed matrix -----------------
        for col in list(result.columns):
            vals = result[col].to_numpy(dtype=float)
            if np.std(vals) == 0:
                continue
            if _pval(vals) < p_crit:
                continue
            # One final plain diff.
            diffed = result[col].diff()
            clean = diffed.dropna().to_numpy(dtype=float)
            if _pval(clean) < p_crit:
                result[col] = diffed
                self._optimal_d[col] = 1.0
            else:
                logger.warning(
                    f"assembler: could not stationarise {col!r}; leaving as-is"
                )

        result = result.dropna(how="any")
        return result
