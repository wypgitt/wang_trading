"""Tests for cross-exchange arbitrage signals."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.signal_battery.cross_exchange_arb import (
    CrossExchangeArbSignal,
    MultiExchangePriceTracker,
)


# ---------------------------------------------------------------------------
# CrossExchangeArbSignal — bar-level detector
# ---------------------------------------------------------------------------

class TestCrossExchangeArbSignal:
    def _bars(self, rows: list[dict]) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=len(rows), freq="min")
        return pd.DataFrame(rows, index=idx)

    def test_signal_fires_when_spread_exceeds_fees(self):
        # binance = 50000, coinbase = 50200 → spread = 40 bps. With default
        # min_spread_bps=10 + fee_estimate_bps=20 → threshold=30 bps → fire.
        bars = self._bars([
            {"binance": 50_000.0, "coinbase": 50_200.0},
        ])
        gen = CrossExchangeArbSignal()
        sigs = gen.generate(bars, symbol="BTC")
        assert len(sigs) == 1
        s = sigs[0]
        assert s.side == 1
        assert s.metadata["buy_exchange"] == "binance"
        assert s.metadata["sell_exchange"] == "coinbase"
        # 40 bps spread − 20 bps fees = 20 bps profit.
        assert abs(s.metadata["estimated_profit_bps"] - 20.0) < 0.1

    def test_no_signal_when_spread_below_threshold(self):
        # 15 bps spread < 10 + 20 = 30 bps threshold → no signal.
        bars = self._bars([
            {"binance": 50_000.0, "coinbase": 50_075.0},
        ])
        gen = CrossExchangeArbSignal()
        sigs = gen.generate(bars, symbol="BTC")
        assert sigs == []

    def test_no_signal_when_spread_negative(self):
        # All exchanges at same price → spread is zero, no signal.
        bars = self._bars([
            {"binance": 50_000.0, "coinbase": 50_000.0},
        ])
        gen = CrossExchangeArbSignal()
        assert gen.generate(bars, symbol="BTC") == []

    def test_confidence_scales_with_spread(self):
        # Three rows with increasing spreads: 40 bps, 100 bps, 200 bps.
        # Net profit (minus 20 bps fees): 20, 80, 180 bps → confidence
        # = 0.2, 0.8, 1.0 (clipped).
        bars = self._bars([
            {"binance": 50_000.0, "coinbase": 50_200.0},   # 40 bps
            {"binance": 50_000.0, "coinbase": 50_500.0},   # 100 bps
            {"binance": 50_000.0, "coinbase": 51_000.0},   # 200 bps
        ])
        sigs = CrossExchangeArbSignal().generate(bars, symbol="BTC")
        assert len(sigs) == 3
        confidences = [s.confidence for s in sigs]
        assert confidences[0] < confidences[1] < confidences[2]
        np.testing.assert_allclose(confidences[2], 1.0, atol=1e-9)

    def test_three_venues_picks_extremes(self):
        # kraken=49900, binance=50000, coinbase=50500 → buy=kraken, sell=coinbase.
        bars = self._bars([
            {"binance": 50_000.0, "coinbase": 50_500.0, "kraken": 49_900.0},
        ])
        sigs = CrossExchangeArbSignal().generate(bars, symbol="BTC")
        assert len(sigs) == 1
        assert sigs[0].metadata["buy_exchange"] == "kraken"
        assert sigs[0].metadata["sell_exchange"] == "coinbase"

    def test_handles_missing_values_gracefully(self):
        bars = self._bars([
            {"binance": 50_000.0, "coinbase": np.nan},  # only one valid price
            {"binance": 50_000.0, "coinbase": 50_300.0},  # 60 bps spread, valid
        ])
        sigs = CrossExchangeArbSignal().generate(bars, symbol="BTC")
        assert len(sigs) == 1
        # Only the second bar had two valid prices.
        assert sigs[0].timestamp == bars.index[1].to_pydatetime()

    def test_requires_at_least_two_columns(self):
        bars = pd.DataFrame({"binance": [50_000.0]})
        with pytest.raises(ValueError, match="at least 2 exchange"):
            CrossExchangeArbSignal().generate(bars)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            CrossExchangeArbSignal(params={"min_spread_bps": -1})
        with pytest.raises(ValueError):
            CrossExchangeArbSignal(params={"fee_estimate_bps": -0.5})


# ---------------------------------------------------------------------------
# MultiExchangePriceTracker
# ---------------------------------------------------------------------------

class TestMultiExchangePriceTracker:
    def test_snapshot_includes_all_fresh_exchanges(self):
        t0 = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        tr = MultiExchangePriceTracker(stale_after=5.0)
        tr.update("binance", "BTC", bid=49_900, ask=49_910, timestamp=t0)
        tr.update("coinbase", "BTC", bid=49_920, ask=49_930, timestamp=t0)

        snap = tr.get_snapshot("BTC", now=t0)
        assert set(snap.keys()) == {"binance", "coinbase"}
        assert snap["binance"]["bid"] == 49_900
        assert snap["binance"]["ask"] == 49_910
        assert snap["binance"]["mid"] == 49_905

    def test_stale_quotes_excluded(self):
        t0 = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        t_now = datetime(2024, 3, 1, 12, 0, 10, tzinfo=timezone.utc)  # +10s
        tr = MultiExchangePriceTracker(stale_after=5.0)
        tr.update("binance", "BTC", 49_900, 49_910, t0)
        tr.update("coinbase", "BTC", 49_920, 49_930, t_now)
        snap = tr.get_snapshot("BTC", now=t_now)
        # binance quote is 10s old > 5s → excluded; coinbase fresh.
        assert "binance" not in snap
        assert "coinbase" in snap

    def test_arb_opportunity_when_bid_above_ask_cross_venue(self):
        t0 = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        tr = MultiExchangePriceTracker()
        # Kraken has a bid (49_990) above Binance's ask (49_910) → arb!
        tr.update("binance", "BTC", bid=49_900, ask=49_910, timestamp=t0)
        tr.update("kraken", "BTC", bid=49_990, ask=50_000, timestamp=t0)
        opps = tr.get_arb_opportunities("BTC", min_spread_bps=1.0, now=t0)
        assert len(opps) == 1
        o = opps[0]
        assert o["buy_exchange"] == "binance"    # lowest ask
        assert o["sell_exchange"] == "kraken"    # highest bid
        # Spread: (49990 - 49910)/49910 * 10000 ≈ 16 bps.
        assert 15.5 < o["spread_bps"] < 16.5

    def test_no_arb_when_spread_below_threshold(self):
        t0 = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        tr = MultiExchangePriceTracker()
        tr.update("binance", "BTC", 49_900, 49_910, t0)
        tr.update("kraken", "BTC", 49_912, 49_920, t0)  # ~0.4 bp cross
        opps = tr.get_arb_opportunities("BTC", min_spread_bps=10.0, now=t0)
        assert opps == []

    def test_update_rejects_invalid_inputs(self):
        tr = MultiExchangePriceTracker()
        t = datetime(2024, 3, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            tr.update("binance", "BTC", bid=-1, ask=1, timestamp=t)
        with pytest.raises(ValueError):
            tr.update("binance", "BTC", bid=100, ask=50, timestamp=t)  # crossed

    def test_naive_timestamp_treated_as_utc(self):
        """A naive datetime should be interpreted as UTC, not crash."""
        tr = MultiExchangePriceTracker(stale_after=5.0)
        t_naive = datetime(2024, 3, 1, 12, 0, 0)  # no tzinfo
        tr.update("binance", "BTC", 100.0, 101.0, t_naive)
        now_utc = datetime(2024, 3, 1, 12, 0, 1, tzinfo=timezone.utc)
        snap = tr.get_snapshot("BTC", now=now_utc)
        assert "binance" in snap

    def test_invalid_stale_after(self):
        with pytest.raises(ValueError):
            MultiExchangePriceTracker(stale_after=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
