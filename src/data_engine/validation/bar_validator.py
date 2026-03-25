"""
Bar Validation (Phase 1 Checkpoint)

Statistical validation of bar quality to verify that information-driven
bars have the expected properties from AFML Ch. 2:

1. Returns closer to normal distribution than time bars
2. Lower serial correlation in returns
3. More stable variance of returns across bars
4. Reasonable bar formation rate

Run this after generating bars to validate before proceeding to Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


@dataclass
class BarValidationReport:
    """Results of bar quality validation."""
    symbol: str
    bar_type: str
    n_bars: int

    # Distribution tests
    returns_mean: float
    returns_std: float
    returns_skewness: float
    returns_kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float

    # Serial correlation
    autocorr_lag1: float
    autocorr_lag5: float
    ljung_box_stat: float
    ljung_box_pvalue: float

    # Variance stability
    variance_of_bar_returns_std: float  # std of rolling variance
    variance_ratio: float  # VR test statistic

    # Bar formation
    avg_ticks_per_bar: float
    avg_bar_duration_seconds: float
    bar_duration_cv: float  # coefficient of variation

    @property
    def is_healthy(self) -> bool:
        """Quick health check: are the bars reasonable?"""
        return (
            self.n_bars >= 100
            and self.returns_std > 0
            and abs(self.returns_skewness) < 5.0
            and self.returns_kurtosis < 50.0
            and self.avg_ticks_per_bar >= 5
        )

    def summary(self) -> str:
        """Human-readable summary."""
        status = "HEALTHY" if self.is_healthy else "WARNING"
        lines = [
            f"═══ Bar Validation Report: {self.symbol} ({self.bar_type}) ═══",
            f"Status: {status}",
            f"Bars: {self.n_bars}",
            f"",
            f"── Returns Distribution ──",
            f"  Mean:      {self.returns_mean:.6f}",
            f"  Std:       {self.returns_std:.6f}",
            f"  Skewness:  {self.returns_skewness:.4f}",
            f"  Kurtosis:  {self.returns_kurtosis:.4f}",
            f"  JB stat:   {self.jarque_bera_stat:.2f} (p={self.jarque_bera_pvalue:.4f})",
            f"",
            f"── Serial Correlation ──",
            f"  Autocorr(1): {self.autocorr_lag1:.4f}",
            f"  Autocorr(5): {self.autocorr_lag5:.4f}",
            f"  Ljung-Box:   {self.ljung_box_stat:.2f} (p={self.ljung_box_pvalue:.4f})",
            f"",
            f"── Variance Stability ──",
            f"  Rolling var std: {self.variance_of_bar_returns_std:.6f}",
            f"  Variance ratio:  {self.variance_ratio:.4f}",
            f"",
            f"── Bar Formation ──",
            f"  Avg ticks/bar:     {self.avg_ticks_per_bar:.1f}",
            f"  Avg duration (s):  {self.avg_bar_duration_seconds:.1f}",
            f"  Duration CV:       {self.bar_duration_cv:.4f}",
        ]
        return "\n".join(lines)


def validate_bars(
    bars_df: pd.DataFrame,
    symbol: str,
    bar_type: str,
    rolling_window: int = 50,
) -> BarValidationReport:
    """
    Run full validation suite on a bars DataFrame.

    Args:
        bars_df:        DataFrame with bar data (must have 'close', 'tick_count', etc.)
        symbol:         Instrument symbol
        bar_type:       Bar type string
        rolling_window: Window for rolling variance test

    Returns:
        BarValidationReport with all test results
    """
    if len(bars_df) < 10:
        logger.warning(f"Too few bars ({len(bars_df)}) for validation")
        return BarValidationReport(
            symbol=symbol, bar_type=bar_type, n_bars=len(bars_df),
            returns_mean=0, returns_std=0, returns_skewness=0,
            returns_kurtosis=0, jarque_bera_stat=0, jarque_bera_pvalue=1,
            autocorr_lag1=0, autocorr_lag5=0, ljung_box_stat=0,
            ljung_box_pvalue=1, variance_of_bar_returns_std=0,
            variance_ratio=0, avg_ticks_per_bar=0,
            avg_bar_duration_seconds=0, bar_duration_cv=0,
        )

    # Compute returns
    closes = bars_df["close"].values.astype(float)
    returns = np.diff(np.log(closes[closes > 0]))
    returns = returns[np.isfinite(returns)]

    if len(returns) < 10:
        logger.warning("Too few valid returns for validation")
        returns = np.zeros(10)

    # Distribution tests
    ret_mean = float(np.mean(returns))
    ret_std = float(np.std(returns))
    ret_skew = float(stats.skew(returns))
    ret_kurt = float(stats.kurtosis(returns))
    jb_stat, jb_pval = stats.jarque_bera(returns)

    # Serial correlation
    ret_series = pd.Series(returns)
    ac1 = float(ret_series.autocorr(lag=1)) if len(ret_series) > 1 else 0.0
    ac5 = float(ret_series.autocorr(lag=5)) if len(ret_series) > 5 else 0.0

    # Ljung-Box test for serial correlation
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
        lb_stat = float(lb_result["lb_stat"].iloc[0])
        lb_pval = float(lb_result["lb_pvalue"].iloc[0])
    except (ImportError, Exception):
        lb_stat, lb_pval = 0.0, 1.0

    # Variance stability (rolling variance of returns)
    if len(returns) > rolling_window:
        rolling_var = pd.Series(returns).rolling(rolling_window).var().dropna()
        var_std = float(rolling_var.std())
    else:
        var_std = ret_std

    # Variance ratio (VR test: VR(q) should be ~1 for random walk)
    q = min(10, len(returns) // 4)
    if q > 1 and len(returns) > q:
        var_1 = np.var(returns)
        agg_returns = np.array([
            np.sum(returns[i:i+q]) for i in range(0, len(returns) - q + 1, q)
        ])
        var_q = np.var(agg_returns) if len(agg_returns) > 1 else var_1
        vr = var_q / (q * var_1) if var_1 > 0 else 1.0
    else:
        vr = 1.0

    # Bar formation stats
    tick_counts = bars_df.get("tick_count", pd.Series([0]))
    avg_ticks = float(tick_counts.mean())

    if "bar_duration_secs" in bars_df.columns:
        durations = bars_df["bar_duration_secs"].values
    elif "bar_duration_seconds" in bars_df.columns:
        durations = bars_df["bar_duration_seconds"].values
    else:
        durations = np.zeros(len(bars_df))

    avg_duration = float(np.mean(durations))
    dur_cv = float(np.std(durations) / avg_duration) if avg_duration > 0 else 0.0

    report = BarValidationReport(
        symbol=symbol,
        bar_type=bar_type,
        n_bars=len(bars_df),
        returns_mean=ret_mean,
        returns_std=ret_std,
        returns_skewness=ret_skew,
        returns_kurtosis=ret_kurt,
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pvalue=float(jb_pval),
        autocorr_lag1=ac1,
        autocorr_lag5=ac5,
        ljung_box_stat=lb_stat,
        ljung_box_pvalue=lb_pval,
        variance_of_bar_returns_std=var_std,
        variance_ratio=vr,
        avg_ticks_per_bar=avg_ticks,
        avg_bar_duration_seconds=avg_duration,
        bar_duration_cv=dur_cv,
    )

    logger.info(f"Validation complete for {symbol}/{bar_type}: {'HEALTHY' if report.is_healthy else 'WARNING'}")
    return report


def compare_bar_types(
    reports: list[BarValidationReport],
) -> pd.DataFrame:
    """
    Compare validation reports across bar types.

    Useful for selecting the best bar type for an instrument.
    AFML predicts that information-driven bars (TIB/VIB) should
    show lower autocorrelation and more normal returns than
    time/tick/volume bars.
    """
    rows = []
    for r in reports:
        rows.append({
            "bar_type": r.bar_type,
            "n_bars": r.n_bars,
            "returns_std": r.returns_std,
            "|skewness|": abs(r.returns_skewness),
            "excess_kurtosis": r.returns_kurtosis,
            "|autocorr_1|": abs(r.autocorr_lag1),
            "jb_pvalue": r.jarque_bera_pvalue,
            "variance_ratio": r.variance_ratio,
            "avg_ticks": r.avg_ticks_per_bar,
            "healthy": r.is_healthy,
        })

    df = pd.DataFrame(rows)
    logger.info(f"\n{df.to_string(index=False)}")
    return df
