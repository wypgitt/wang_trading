"""
Microstructural Features (AFML Ch. 19 + Johnson)

Rolling estimators of liquidity, informed-trading probability, and price
impact from bar-level data. These features feed the ML meta-labeler
(Section 5 of the design doc) and also modulate bet sizing via VPIN.

Features:
    - kyle_lambda           : price impact per unit of signed volume (OLS slope)
    - amihud_lambda         : |return| / dollar_volume illiquidity
    - roll_spread           : bid-ask spread estimate from return autocovariance
    - vpin                  : Volume-Synchronized Probability of Informed Trading
    - hasbrouck_lambda      : VAR-based permanent price impact (slow; opt-in)
    - order_flow_imbalance  : rolling net signed volume
    - trade_intensity       : trades per second
    - compute_microstructure_features : convenience aggregator

All rolling functions accept pandas Series/DataFrames aligned on the same
index; they return NaN for the warmup period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Kyle's lambda (OLS slope of dp on signed volume)
# ---------------------------------------------------------------------------

def kyle_lambda(
    close: pd.Series,
    volume: pd.Series,
    signed_volume: pd.Series,
    window: int = 50,
) -> pd.Series:
    """
    Rolling Kyle's Lambda: slope of price change on signed order flow.

    Kyle (1985) models mid-price impact as dp_t = lambda * q_t + eps_t, where
    q_t is signed order flow (buy volume minus sell volume or signed trade
    size). Lambda is the illiquidity (or "depth inverse"): higher lambda
    means the same signed volume moves price more.

    Estimated by rolling OLS:
        lambda_t = cov_window(dp, sv) / var_window(sv)

    Args:
        close:         Close price series.
        volume:        Total per-bar volume (unused in the formula but kept
                       in the signature for API symmetry / future variants).
        signed_volume: Per-bar signed volume (buy_vol - sell_vol or similar).
        window:        Rolling window length.

    Returns:
        pd.Series: Kyle's lambda per bar. NaN for the first ``window`` bars.
    """
    del volume  # accepted for API symmetry; not used in the OLS slope
    if window < 3:
        raise ValueError("window must be >= 3")

    dp = close.diff()
    sv = signed_volume.astype(float)

    # Rolling covariance and variance, then take the ratio.
    cov = dp.rolling(window=window, min_periods=window).cov(sv)
    var = sv.rolling(window=window, min_periods=window).var()
    lam = cov / var.replace(0.0, np.nan)
    lam = lam.rename("kyle_lambda")
    return lam


# ---------------------------------------------------------------------------
# Amihud's illiquidity measure
# ---------------------------------------------------------------------------

def amihud_lambda(
    close: pd.Series,
    dollar_volume: pd.Series,
    window: int = 50,
) -> pd.Series:
    """
    Rolling Amihud (2002) illiquidity:

        ILLIQ_t = mean over window of ( |return_i| / dollar_volume_i )

    Higher values indicate a less liquid market (large return per dollar
    traded). Bars with zero dollar_volume contribute NaN and are dropped
    from the window average.

    Args:
        close:         Close price series.
        dollar_volume: Per-bar dollar volume (price * volume).
        window:        Rolling window length.

    Returns:
        pd.Series: Amihud illiquidity per bar.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    ret = close.pct_change()
    dv = dollar_volume.replace(0.0, np.nan)
    ratio = ret.abs() / dv
    out = ratio.rolling(window=window, min_periods=window).mean()
    return out.rename("amihud_lambda")


# ---------------------------------------------------------------------------
# Roll spread estimate
# ---------------------------------------------------------------------------

def roll_spread(close: pd.Series, window: int = 50) -> pd.Series:
    """
    Roll (1984) effective-spread estimate from return autocovariance.

    Roll shows that bid-ask bounce induces a negative first-order
    autocovariance in transaction-price changes:

        cov(dp_t, dp_{t-1}) = -(spread / 2)^2

    So:
        spread = 2 * sqrt(-cov(dp_t, dp_{t-1}))  if cov < 0,
                 0                                 otherwise.

    The returned feature is the spread expressed as a fraction of the mid
    price (using the current close as a proxy).

    Args:
        close:  Close price series.
        window: Rolling window length.

    Returns:
        pd.Series: Roll spread estimate (fractional).
    """
    if window < 3:
        raise ValueError("window must be >= 3")

    dp = close.diff()
    dp_lag = dp.shift(1)
    # Rolling covariance of (dp, dp_{t-1}).
    cov = dp.rolling(window=window, min_periods=window).cov(dp_lag)
    # Clip at 0 — only negative covariance yields a bid-ask bounce estimate.
    neg = (-cov).clip(lower=0.0)
    spread_price = 2.0 * np.sqrt(neg)
    fractional = spread_price / close.replace(0.0, np.nan)
    return fractional.rename("roll_spread")


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------

def vpin(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    total_volume: pd.Series,
    window: int = 50,
) -> pd.Series:
    """
    Volume-Synchronized Probability of Informed Trading (Easley et al. 2012).

    VPIN_t = mean over the trailing ``window`` *volume bars* of
             |buy_volume_i - sell_volume_i| / total_volume_i.

    Values lie in [0, 1]; elevated VPIN precedes volatility spikes and is
    commonly used as a size throttle. Must be computed on volume or dollar
    bars (not clock-time bars) for the "volume-synchronized" property to hold.

    Args:
        buy_volume:   Per-bar buy volume.
        sell_volume:  Per-bar sell volume.
        total_volume: Per-bar total volume (should equal buy + sell per bar).
        window:       Number of bars in the rolling average.

    Returns:
        pd.Series: VPIN per bar.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    tv = total_volume.replace(0.0, np.nan)
    per_bar = (buy_volume - sell_volume).abs() / tv
    out = per_bar.rolling(window=window, min_periods=window).mean()
    return out.rename("vpin")


# ---------------------------------------------------------------------------
# Hasbrouck's lambda (VAR impulse response)
# ---------------------------------------------------------------------------

def hasbrouck_lambda(
    close: pd.Series,
    signed_volume: pd.Series,
    window: int = 50,
    lags: int = 5,
) -> pd.Series:
    """
    Rolling Hasbrouck (1991) permanent price impact from a bivariate VAR.

    Fits a VAR(``lags``) on [dp_t, signed_volume_t] over each trailing
    window and returns the cumulative response of dp to a unit shock in
    signed volume at horizon ``lags``. Higher values indicate a larger
    permanent (informed) price impact per unit of order flow.

    This is slow — it fits one statsmodels VAR per bar. It is opt-in via
    ``compute_microstructure_features(include_hasbrouck=True)``.

    Args:
        close:         Close price series.
        signed_volume: Per-bar signed volume.
        window:        Rolling window length (>= 4 * lags is recommended).
        lags:          VAR order and impulse-response horizon.

    Returns:
        pd.Series: Hasbrouck's lambda per bar. NaN on estimation failure.
    """
    from statsmodels.tsa.api import VAR

    if window < 4 * lags:
        raise ValueError(f"window ({window}) should be >= 4*lags ({4 * lags})")

    dp = close.diff()
    sv = signed_volume.astype(float)
    data = pd.concat({"dp": dp, "sv": sv}, axis=1).dropna()
    n = len(close)
    out = np.full(n, np.nan)

    # Map back from the concat'd index to the original series index.
    orig_index = close.index
    position = {idx: i for i, idx in enumerate(orig_index)}

    for end in range(window, len(data) + 1):
        block = data.iloc[end - window : end]
        if block["sv"].std() == 0 or block["dp"].std() == 0:
            continue
        try:
            model = VAR(block)
            fitted = model.fit(lags)
            irf = fitted.irf(periods=lags)
            # Cumulative IRF: response of dp (variable 0) to shock in sv
            # (variable 1), summed over the horizon.
            cum = irf.cum_effects
            lam = float(cum[lags, 0, 1])
        except (ValueError, np.linalg.LinAlgError, Exception) as exc:
            logger.debug(f"VAR fit failed at end={end}: {exc}")
            continue
        t_idx = block.index[-1]
        if t_idx in position:
            out[position[t_idx]] = lam

    return pd.Series(out, index=orig_index, name="hasbrouck_lambda")


# ---------------------------------------------------------------------------
# Order flow imbalance
# ---------------------------------------------------------------------------

def order_flow_imbalance(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Rolling signed-volume ratio:

        OFI_t = sum(buy_vol - sell_vol) / sum(buy_vol + sell_vol)

    over the trailing ``window`` bars. OFI is in [-1, 1]: +1 means every
    share traded in the window was a buy, -1 means all sells, 0 is balanced.

    Args:
        buy_volume:  Per-bar buy volume.
        sell_volume: Per-bar sell volume.
        window:      Rolling window length.

    Returns:
        pd.Series: Order flow imbalance per bar.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    signed = (buy_volume - sell_volume).rolling(
        window=window, min_periods=window
    ).sum()
    total = (buy_volume + sell_volume).rolling(
        window=window, min_periods=window
    ).sum()
    out = signed / total.replace(0.0, np.nan)
    return out.rename("order_flow_imbalance")


# ---------------------------------------------------------------------------
# Trade intensity
# ---------------------------------------------------------------------------

def trade_intensity(
    tick_count: pd.Series,
    bar_duration: pd.Series,
) -> pd.Series:
    """
    Trades per second.

        intensity_t = tick_count_t / bar_duration_seconds_t

    Args:
        tick_count:   Number of trades per bar.
        bar_duration: Bar duration in seconds.

    Returns:
        pd.Series: Trades per second.
    """
    duration = bar_duration.replace(0.0, np.nan)
    return (tick_count / duration).rename("trade_intensity")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

_REQUIRED_COLS = (
    "close",
    "volume",
    "dollar_volume",
    "buy_volume",
    "sell_volume",
    "tick_count",
    "bar_duration_seconds",
)


def compute_microstructure_features(
    bars_df: pd.DataFrame,
    window: int = 50,
    include_hasbrouck: bool = False,
    hasbrouck_lags: int = 5,
) -> pd.DataFrame:
    """
    Compute the full microstructure feature block from a bars DataFrame.

    Expected columns on ``bars_df``:
        close, volume, dollar_volume, buy_volume, sell_volume,
        tick_count, bar_duration_seconds
    (matches the dataclass fields produced by ``src.data_engine.models.Bar``).

    Args:
        bars_df:           Input bar DataFrame.
        window:            Rolling window used by all features.
        include_hasbrouck: Whether to compute Hasbrouck's lambda (slow).
        hasbrouck_lags:    VAR lag order for Hasbrouck's lambda.

    Returns:
        pd.DataFrame indexed like ``bars_df`` with columns kyle_lambda,
        amihud_lambda, roll_spread, vpin, order_flow_imbalance,
        trade_intensity (+ hasbrouck_lambda when requested).
    """
    missing = [c for c in _REQUIRED_COLS if c not in bars_df.columns]
    if missing:
        raise KeyError(f"bars_df missing columns: {missing}")

    close = bars_df["close"]
    buy = bars_df["buy_volume"]
    sell = bars_df["sell_volume"]
    signed = buy - sell
    total = bars_df["volume"]

    logger.debug(
        f"microstructure: computing features on {len(bars_df)} bars "
        f"(window={window}, hasbrouck={'on' if include_hasbrouck else 'off'})"
    )

    cols: dict[str, pd.Series] = {
        "kyle_lambda": kyle_lambda(close, total, signed, window=window),
        "amihud_lambda": amihud_lambda(close, bars_df["dollar_volume"], window=window),
        "roll_spread": roll_spread(close, window=window),
        "vpin": vpin(buy, sell, total, window=window),
        "order_flow_imbalance": order_flow_imbalance(buy, sell, window=window),
        "trade_intensity": trade_intensity(
            bars_df["tick_count"], bars_df["bar_duration_seconds"]
        ),
    }
    if include_hasbrouck:
        cols["hasbrouck_lambda"] = hasbrouck_lambda(
            close, signed, window=window, lags=hasbrouck_lags
        )

    return pd.DataFrame(cols, index=bars_df.index)
