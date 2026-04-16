"""Tests for on-chain crypto features (Glassnode)."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.feature_factory.onchain import (
    GlassnodeClient,
    compute_onchain_features,
    exchange_flow_features,
    network_health,
    stablecoin_supply_ratio,
    whale_activity,
)


# ---------------------------------------------------------------------------
# GlassnodeClient — HTTP mocked via monkeypatched _http_get
# ---------------------------------------------------------------------------

def _glassnode_payload(points: list[tuple[int, float]]) -> str:
    """Build a fake Glassnode response: list of {t, v} records."""
    return json.dumps([{"t": t, "v": v} for t, v in points])


class TestGlassnodeClient:
    def test_get_metric_parses_response(self, monkeypatch):
        client = GlassnodeClient(api_key="dummy")
        pts = [(1700000000, 1.5), (1700086400, 2.5)]
        monkeypatch.setattr(
            client, "_http_get", lambda url, timeout=30.0: _glassnode_payload(pts)
        )
        s = client.get_metric(
            "BTC", "addresses/active_count",
            datetime(2023, 11, 14, tzinfo=timezone.utc),
            datetime(2023, 11, 16, tzinfo=timezone.utc),
        )
        assert len(s) == 2
        np.testing.assert_allclose(s.values, [1.5, 2.5])
        assert str(s.index.tz) == "UTC"

    def test_get_metric_on_http_error_returns_empty(self, monkeypatch):
        client = GlassnodeClient(api_key="dummy")

        def boom(url, timeout=30.0):
            raise ConnectionError("down")

        monkeypatch.setattr(client, "_http_get", boom)
        s = client.get_metric(
            "BTC", "addresses/active_count",
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 1, 2, tzinfo=timezone.utc),
        )
        assert s.empty

    def test_get_metric_bad_json_returns_empty(self, monkeypatch):
        client = GlassnodeClient(api_key="dummy")
        monkeypatch.setattr(client, "_http_get", lambda url, timeout=30.0: "not json")
        s = client.get_metric(
            "BTC", "addresses/active_count",
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 1, 2, tzinfo=timezone.utc),
        )
        assert s.empty

    def test_get_exchange_flows_builds_frame(self, monkeypatch):
        client = GlassnodeClient(api_key="dummy")
        inflow_pts = [(1700000000, 100.0), (1700086400, 200.0)]
        outflow_pts = [(1700000000, 150.0), (1700086400, 50.0)]

        def fake_http(url, timeout=30.0):
            if "to_exchanges" in url:
                return _glassnode_payload(inflow_pts)
            if "from_exchanges" in url:
                return _glassnode_payload(outflow_pts)
            return json.dumps([])

        monkeypatch.setattr(client, "_http_get", fake_http)
        df = client.get_exchange_flows(
            "BTC",
            datetime(2023, 11, 14, tzinfo=timezone.utc),
            datetime(2023, 11, 16, tzinfo=timezone.utc),
        )
        assert set(df.columns) == {"inflow", "outflow", "netflow"}
        np.testing.assert_allclose(df["netflow"].values, [50.0, -150.0])

    def test_invalid_date_range(self):
        client = GlassnodeClient(api_key="dummy")
        with pytest.raises(ValueError):
            client.get_metric(
                "BTC", "addresses/active_count",
                datetime(2023, 2, 1, tzinfo=timezone.utc),
                datetime(2023, 1, 1, tzinfo=timezone.utc),
            )


# ---------------------------------------------------------------------------
# exchange_flow_features
# ---------------------------------------------------------------------------

class TestExchangeFlowFeatures:
    def test_net_flow_sign(self):
        """net_flow = outflow - inflow; positive when outflow dominates."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        inflow = pd.Series([50, 100, 200, 10, 30], index=idx, dtype=float)
        outflow = pd.Series([100, 80, 150, 120, 50], index=idx, dtype=float)
        out = exchange_flow_features(inflow, outflow, window=3)
        np.testing.assert_allclose(out["net_flow"].values, [50, -20, -50, 110, 20])

    def test_flow_ratio(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        inflow = pd.Series([100, 50, 200], index=idx, dtype=float)
        outflow = pd.Series([50, 100, 100], index=idx, dtype=float)
        out = exchange_flow_features(inflow, outflow, window=2)
        np.testing.assert_allclose(out["flow_ratio"].values, [2.0, 0.5, 2.0])

    def test_zscore_warmup(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        # Use varying net_flow so the rolling std is strictly positive
        # (constant net_flow would give std=0 and NaN z-scores by design).
        inflow = pd.Series([10, 40, 25, 60, 15], index=idx, dtype=float)
        outflow = pd.Series([20, 30, 50, 20, 70], index=idx, dtype=float)
        out = exchange_flow_features(inflow, outflow, window=3)
        assert out["net_flow_zscore"].iloc[:2].isna().all()
        assert out["net_flow_zscore"].iloc[2:].notna().all()

    def test_zero_outflow_does_not_blow_up(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        inflow = pd.Series([10, 20, 30], index=idx, dtype=float)
        outflow = pd.Series([0, 0, 0], index=idx, dtype=float)
        out = exchange_flow_features(inflow, outflow, window=2)
        # flow_ratio should be NaN (divide-by-zero), not inf.
        assert out["flow_ratio"].isna().all()
        # net_flow still computable.
        np.testing.assert_allclose(out["net_flow"].values, [-10, -20, -30])

    def test_invalid_window(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            exchange_flow_features(s, s, window=1)


# ---------------------------------------------------------------------------
# whale_activity
# ---------------------------------------------------------------------------

class TestWhaleActivity:
    def test_identifies_above_threshold(self):
        """A spike should be flagged as whale activity."""
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        # Mostly small volumes with a single huge spike at position 15.
        vol = pd.Series(np.full(20, 100.0), index=idx)
        vol.iloc[15] = 10_000.0

        out = whale_activity(vol, threshold_percentile=90, window=10)
        # Before the spike, within-window quantile is 100 → flagged? No — the
        # quantile of constant values matches that value, so rows at/above it
        # include all bars. The "volume_ratio" for a constant region is 1.0.
        # After the spike enters the window, its outsized magnitude should
        # dominate whale_volume_ratio.
        post = out["whale_volume_ratio"].iloc[15:19].dropna()
        assert post.max() > 0.9  # spike dominates the window volume

    def test_output_shape_and_warmup(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        vol = pd.Series(np.arange(1, 31, dtype=float), index=idx)
        out = whale_activity(vol, threshold_percentile=95, window=10)
        assert set(out.columns) == {"whale_tx_count", "whale_volume_ratio"}
        assert out.iloc[:9].isna().all().all()

    def test_whale_ratio_bounded(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        vol = pd.Series(rng.uniform(0, 100, size=50), index=idx)
        out = whale_activity(vol, threshold_percentile=80, window=14)
        r = out["whale_volume_ratio"].dropna()
        assert (r >= 0).all() and (r <= 1.0 + 1e-9).all()

    def test_invalid_percentile(self):
        s = pd.Series([1.0] * 20)
        with pytest.raises(ValueError):
            whale_activity(s, threshold_percentile=0)
        with pytest.raises(ValueError):
            whale_activity(s, threshold_percentile=100)


# ---------------------------------------------------------------------------
# network_health
# ---------------------------------------------------------------------------

class TestNetworkHealth:
    def test_nvm_decreases_when_addresses_grow_faster_than_price(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        # Price doubles over the window; active addresses quadruple.
        price = pd.Series(np.linspace(100.0, 200.0, 50), index=idx)
        addr = pd.Series(np.linspace(1000.0, 4000.0, 50), index=idx)
        out = network_health(addr, price, window=10)
        # NVM = price / addr^2 shrinks as addr grows faster than price.
        assert out["nvm_ratio"].iloc[-1] < out["nvm_ratio"].iloc[0]

    def test_divergence_correlation_positive_when_aligned(self):
        idx = pd.date_range("2024-01-01", periods=60, freq="D")
        # Price and addresses co-move → correlation near +1.
        price = pd.Series(np.linspace(100.0, 200.0, 60), index=idx)
        addr = pd.Series(np.linspace(1000.0, 2000.0, 60), index=idx)
        out = network_health(addr, price, window=30)
        assert out["addr_price_divergence"].iloc[-1] > 0.99

    def test_invalid_window(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            network_health(s, s, window=2)


# ---------------------------------------------------------------------------
# stablecoin_supply_ratio
# ---------------------------------------------------------------------------

class TestStablecoinSupplyRatio:
    def test_simple_division(self):
        stable = pd.Series([100.0, 200.0, 300.0])
        btc = pd.Series([1000.0, 500.0, 600.0])
        ssr = stablecoin_supply_ratio(stable, btc)
        np.testing.assert_allclose(ssr.values, [0.1, 0.4, 0.5])
        assert ssr.name == "ssr"

    def test_zero_btc_is_nan(self):
        stable = pd.Series([100.0, 200.0])
        btc = pd.Series([0.0, 500.0])
        ssr = stablecoin_supply_ratio(stable, btc)
        assert np.isnan(ssr.iloc[0])
        np.testing.assert_allclose(ssr.iloc[1], 0.4)


# ---------------------------------------------------------------------------
# compute_onchain_features — end-to-end with stub client
# ---------------------------------------------------------------------------

class _StubGlassnode:
    """
    In-memory stub for GlassnodeClient used by the aggregator test. Stores
    canned Series keyed by metric name and returns them from ``get_metric``;
    ``get_exchange_flows`` composes the inflow/outflow pair.
    """

    def __init__(
        self,
        metrics: dict[str, pd.Series],
    ) -> None:
        self.metrics = metrics

    def get_metric(self, asset, metric, start, end, resolution="24h"):  # noqa: ARG002
        return self.metrics.get(metric, pd.Series(dtype=float, name=metric))

    def get_exchange_flows(self, asset, start, end, resolution="24h"):  # noqa: ARG002
        inflow = self.metrics.get(
            GlassnodeClient.FLOW_METRICS["inflow"],
            pd.Series(dtype=float),
        ).rename("inflow")
        outflow = self.metrics.get(
            GlassnodeClient.FLOW_METRICS["outflow"],
            pd.Series(dtype=float),
        ).rename("outflow")
        df = pd.concat([inflow, outflow], axis=1).sort_index()
        df["netflow"] = df["outflow"] - df["inflow"]
        return df


class TestComputeOnchainFeatures:
    def _make_client(self, n: int = 40) -> _StubGlassnode:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        return _StubGlassnode({
            GlassnodeClient.FLOW_METRICS["inflow"]: pd.Series(
                rng.uniform(100, 500, size=n), index=idx
            ),
            GlassnodeClient.FLOW_METRICS["outflow"]: pd.Series(
                rng.uniform(100, 500, size=n), index=idx
            ),
            "addresses/active_count": pd.Series(
                rng.uniform(500_000, 1_000_000, size=n), index=idx
            ),
            "transactions/transfers_volume_sum": pd.Series(
                rng.uniform(1e4, 1e6, size=n), index=idx
            ),
            "market/price_usd_close": pd.Series(
                30_000 + rng.normal(0, 500, size=n).cumsum(), index=idx
            ),
        })

    def test_full_feature_set_columns(self):
        client = self._make_client(40)
        df = compute_onchain_features(
            "BTC", client,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 2, 10, tzinfo=timezone.utc),
            flow_window=5, whale_window=10, health_window=10,
        )
        expected = {
            "net_flow", "flow_ratio", "net_flow_zscore",
            "whale_tx_count", "whale_volume_ratio",
            "nvm_ratio", "addr_price_divergence",
        }
        assert expected.issubset(set(df.columns))
        assert not df.empty

    def test_missing_metrics_degrades_gracefully(self):
        client = _StubGlassnode({})  # no data at all
        df = compute_onchain_features(
            "BTC", client,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        assert df.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
