"""
On-Chain Crypto Features (Glassnode + emerging research)

Exchange-flow, whale-activity, network-health, and stablecoin-supply
features for BTC/ETH and other tracked assets.

All HTTP traffic in GlassnodeClient goes through ``_http_get`` so tests can
monkeypatch network access; the feature functions themselves are pure and
take pandas Series as input.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Glassnode REST client
# ---------------------------------------------------------------------------

class GlassnodeClient:
    """
    Thin client for Glassnode's time-series metrics API.

    The Glassnode endpoint pattern is
    ``/v1/metrics/{category}/{metric}?a=BTC&s=<start>&u=<end>&i=24h&api_key=...``
    and returns a JSON array of ``{"t": unix_seconds, "v": value}`` records
    (or ``{"t", "o": {...}}`` for multi-value metrics).

    Rate limits are handled with exponential backoff on 429 responses.
    Network and API errors log a warning and return an empty Series/DataFrame
    rather than raising — the feature pipeline degrades gracefully when a
    metric is unavailable.
    """

    BASE_URL = "https://api.glassnode.com/v1/metrics"
    FLOW_METRICS = {
        "inflow": "transactions/transfers_volume_to_exchanges_sum",
        "outflow": "transactions/transfers_volume_from_exchanges_sum",
    }

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    # -- public --

    def get_metric(
        self,
        asset: str,
        metric: str,
        start: datetime,
        end: datetime,
        resolution: str = "24h",
    ) -> pd.Series:
        """
        Fetch a single Glassnode metric as a time-indexed Series.

        Returns an empty Series on API or network failure.
        """
        if end < start:
            raise ValueError("end must be >= start")

        url = self._build_url(metric, asset, start, end, resolution)
        payload = self._get_json(url)
        if not payload:
            return pd.Series(dtype=float, name=metric)

        times: list[pd.Timestamp] = []
        values: list[float] = []
        for row in payload:
            try:
                ts = pd.Timestamp(int(row["t"]), unit="s", tz="UTC")
                # Single-value metrics use "v"; multi-value use "o".
                if "v" in row and row["v"] is not None:
                    val = float(row["v"])
                elif "o" in row and isinstance(row["o"], dict):
                    # Fall back to first numeric field in "o".
                    val = float(next(iter(row["o"].values())))
                else:
                    continue
            except (KeyError, TypeError, ValueError, StopIteration):
                continue
            times.append(ts)
            values.append(val)

        return pd.Series(values, index=pd.DatetimeIndex(times), name=metric)

    def get_exchange_flows(
        self,
        asset: str,
        start: datetime,
        end: datetime,
        resolution: str = "24h",
    ) -> pd.DataFrame:
        """
        Fetch exchange inflow and outflow volume for ``asset``.

        Returns a DataFrame indexed by timestamp with columns
        ``inflow``, ``outflow``, ``netflow`` (= outflow - inflow). If either
        side can't be fetched, that column is NaN.
        """
        inflow = self.get_metric(
            asset, self.FLOW_METRICS["inflow"], start, end, resolution,
        ).rename("inflow")
        outflow = self.get_metric(
            asset, self.FLOW_METRICS["outflow"], start, end, resolution,
        ).rename("outflow")

        df = pd.concat([inflow, outflow], axis=1).sort_index()
        df["netflow"] = df["outflow"] - df["inflow"]
        return df

    # -- internals --

    def _build_url(
        self,
        metric: str,
        asset: str,
        start: datetime,
        end: datetime,
        resolution: str,
    ) -> str:
        params = {
            "a": asset.upper(),
            "s": int(start.astimezone(timezone.utc).timestamp()),
            "u": int(end.astimezone(timezone.utc).timestamp()),
            "i": resolution,
            "api_key": self.api_key,
        }
        return f"{self.BASE_URL}/{metric}?{urllib.parse.urlencode(params)}"

    def _get_json(self, url: str) -> list | None:
        """GET + JSON parse with retry on 429, warn-and-return-None on error."""
        delay = self.backoff_seconds
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._http_get(url)
            except urllib.error.HTTPError as exc:  # rate limit or auth error
                if exc.code == 429 and attempt < self.max_retries:
                    logger.warning(
                        f"Glassnode 429 rate-limited; retry {attempt}/"
                        f"{self.max_retries} after {delay:.1f}s"
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                logger.warning(f"Glassnode HTTP error: {exc}")
                return None
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Glassnode request failed: {exc}")
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(f"Glassnode JSON parse failed: {exc}")
                return None
        return None

    def _http_get(self, url: str, timeout: float = 30.0) -> str:
        """HTTP GET. Extracted for easy monkeypatching in tests."""
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def exchange_flow_features(
    inflow: pd.Series,
    outflow: pd.Series,
    window: int = 7,
) -> pd.DataFrame:
    """
    Derive features from exchange inflow / outflow series.

    Columns returned:
        net_flow         : outflow - inflow  (positive = accumulation)
        flow_ratio       : inflow / outflow  (> 1 → net selling pressure)
        net_flow_zscore  : rolling z-score of net_flow over ``window`` bars

    Args:
        inflow:  Coins moved onto exchanges per bar.
        outflow: Coins moved off exchanges per bar.
        window:  Rolling window for the z-score.

    Returns:
        pd.DataFrame indexed to match the input (aligned inner index).
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    df = pd.concat(
        [inflow.rename("inflow"), outflow.rename("outflow")], axis=1,
    )
    net = df["outflow"] - df["inflow"]
    # Avoid divide-by-zero in the ratio.
    ratio = df["inflow"] / df["outflow"].replace(0.0, np.nan)

    rolling_mean = net.rolling(window=window, min_periods=window).mean()
    rolling_std = net.rolling(window=window, min_periods=window).std(ddof=0)
    zscore = (net - rolling_mean) / rolling_std.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "net_flow": net,
            "flow_ratio": ratio,
            "net_flow_zscore": zscore,
        },
        index=df.index,
    )


def whale_activity(
    transfer_volume: pd.Series,
    threshold_percentile: float = 95,
    window: int = 14,
) -> pd.DataFrame:
    """
    Rolling whale-activity features from per-bar transfer volume.

    A bar is flagged as "whale" when its transfer volume exceeds the
    ``threshold_percentile`` quantile of the trailing ``window`` bars.

    Columns:
        whale_tx_count      : count of whale bars within the window
        whale_volume_ratio  : sum(whale volume) / sum(total volume) in window

    Args:
        transfer_volume:      Per-bar transfer volume (non-negative).
        threshold_percentile: Percentile in [0, 100] defining the whale cutoff.
        window:               Rolling window length.

    Returns:
        pd.DataFrame indexed like ``transfer_volume``.
    """
    if not (0 < threshold_percentile < 100):
        raise ValueError("threshold_percentile must be in (0, 100)")
    if window < 2:
        raise ValueError("window must be >= 2")

    q = threshold_percentile / 100.0
    tv = transfer_volume.astype(float)

    # For each bar, compute the rolling quantile and flag bars above it.
    threshold = tv.rolling(window=window, min_periods=window).quantile(q)
    is_whale = (tv >= threshold).astype(float)
    whale_volume = tv * is_whale

    whale_count = is_whale.rolling(window=window, min_periods=window).sum()
    whale_volume_sum = whale_volume.rolling(window=window, min_periods=window).sum()
    total_volume = tv.rolling(window=window, min_periods=window).sum()
    ratio = whale_volume_sum / total_volume.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "whale_tx_count": whale_count,
            "whale_volume_ratio": ratio,
        },
        index=tv.index,
    )


def network_health(
    active_addresses: pd.Series,
    price: pd.Series,
    window: int = 30,
) -> pd.DataFrame:
    """
    Network health features derived from active-address / price series.

    Columns:
        nvm_ratio              : price / active_addresses^2 (Metcalfe-normalized
                                 Network Value to Metcalfe ratio; low = under-
                                 valued relative to network size)
        addr_price_divergence  : rolling Pearson correlation of active
                                 addresses and price over ``window`` bars
                                 (low/negative → divergence → mean-reversion)

    Args:
        active_addresses: Active-address count per bar.
        price:            Price series aligned to ``active_addresses``.
        window:           Rolling window for the correlation.

    Returns:
        pd.DataFrame indexed to match the inputs.
    """
    if window < 3:
        raise ValueError("window must be >= 3")

    addr = active_addresses.astype(float)
    p = price.astype(float)
    # Align on the intersection — Glassnode data may be sparser than bars.
    df = pd.concat([addr.rename("addr"), p.rename("price")], axis=1)

    addr_sq = df["addr"] ** 2
    nvm = df["price"] / addr_sq.replace(0.0, np.nan)

    corr = df["addr"].rolling(window=window, min_periods=window).corr(df["price"])

    return pd.DataFrame(
        {
            "nvm_ratio": nvm,
            "addr_price_divergence": corr,
        },
        index=df.index,
    )


def stablecoin_supply_ratio(
    stablecoin_mcap: pd.Series,
    btc_mcap: pd.Series,
) -> pd.Series:
    """
    Stablecoin Supply Ratio: stablecoin market cap / BTC market cap.

    High SSR → lots of sidelined stablecoins (dry powder). Low SSR → capital
    is already deployed into BTC / risk assets.
    """
    btc = btc_mcap.replace(0.0, np.nan)
    return (stablecoin_mcap / btc).rename("ssr")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_onchain_features(
    asset: str,
    client: GlassnodeClient,
    start: datetime,
    end: datetime,
    resolution: str = "24h",
    flow_window: int = 7,
    whale_window: int = 14,
    whale_percentile: float = 95,
    health_window: int = 30,
) -> pd.DataFrame:
    """
    Fetch Glassnode metrics for ``asset`` and compute all on-chain features.

    Pulls exchange flows, active addresses, transfer volume, and price from
    the Glassnode client and concatenates the resulting feature blocks on
    the intersection of their timestamps.

    Args:
        asset:            Symbol (e.g. "BTC", "ETH").
        client:           GlassnodeClient (or mock with matching interface).
        start, end:       Time range in UTC.
        resolution:       Sampling resolution passed to the API.
        flow_window:      Window for exchange_flow_features.
        whale_window:     Window for whale_activity.
        whale_percentile: Percentile cutoff for whale detection.
        health_window:    Window for network_health correlation.

    Returns:
        pd.DataFrame with all available feature columns. Columns whose
        underlying metric returned no data are omitted.
    """
    flows = client.get_exchange_flows(asset, start, end, resolution)
    active = client.get_metric(
        asset, "addresses/active_count", start, end, resolution,
    )
    transfer_vol = client.get_metric(
        asset, "transactions/transfers_volume_sum", start, end, resolution,
    )
    price = client.get_metric(asset, "market/price_usd_close", start, end, resolution)

    blocks: list[pd.DataFrame] = []

    if not flows.empty and {"inflow", "outflow"}.issubset(flows.columns):
        blocks.append(
            exchange_flow_features(
                flows["inflow"], flows["outflow"], window=flow_window,
            )
        )

    if not transfer_vol.empty:
        blocks.append(
            whale_activity(
                transfer_vol,
                threshold_percentile=whale_percentile,
                window=whale_window,
            )
        )

    if not active.empty and not price.empty:
        blocks.append(
            network_health(active, price, window=health_window)
        )

    if not blocks:
        logger.warning(f"on-chain: no metrics available for {asset}")
        return pd.DataFrame()

    result = pd.concat(blocks, axis=1)
    # Ensure no duplicate timestamps (can happen if two metrics straddle a
    # resolution boundary) and sort chronologically.
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result
