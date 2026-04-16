"""Tests for carry / funding-rate signals."""

import numpy as np
import pandas as pd
import pytest

from src.signal_battery.carry import (
    FundingRateArbSignal,
    FuturesCarrySignal,
    annualize_funding_rate,
)


# ---------------------------------------------------------------------------
# annualize_funding_rate
# ---------------------------------------------------------------------------

class TestAnnualizeFundingRate:
    def test_binance_8h_rate_of_1bp(self):
        """
        A funding of 0.01% (1 bp) every 8 hours ≈ 11% annualized.
        Exact: (1.0001)^(3*365) - 1 ≈ 0.1167.
        """
        out = annualize_funding_rate(0.0001, payments_per_day=3)
        assert abs(out - 0.1167) < 0.01

    def test_zero_rate_is_zero(self):
        assert annualize_funding_rate(0.0) == 0.0

    def test_negative_rate(self):
        out = annualize_funding_rate(-0.0001, payments_per_day=3)
        # Slightly negative compounded rate ≈ -10.4%
        assert -0.12 < out < -0.08

    def test_invalid_payments_per_day(self):
        with pytest.raises(ValueError):
            annualize_funding_rate(0.0001, payments_per_day=0)


# ---------------------------------------------------------------------------
# FuturesCarrySignal
# ---------------------------------------------------------------------------

class TestFuturesCarrySignal:
    def _bars(self, front: list[float], back: list[float], days: int = 30) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=len(front), freq="D")
        return pd.DataFrame(
            {"front_price": front, "back_price": back, "days_to_expiry": [days] * len(front)},
            index=idx,
        )

    def test_long_in_backwardation(self):
        """front > back → positive carry → long."""
        # 100 bars where front is consistently above back (backwardation).
        front = np.full(100, 105.0)
        back = np.full(100, 100.0)
        bars = self._bars(front, back, days=30)
        gen = FuturesCarrySignal()
        sigs = gen.generate(bars, symbol="CL")
        assert len(sigs) > 0
        # All should be long with positive carry.
        assert all(s.side == 1 for s in sigs)
        # Annualized carry = 5/105 * 365/30 ≈ 0.579 → metadata should reflect.
        assert all(s.metadata["carry"] > 0 for s in sigs)

    def test_short_in_contango(self):
        """front < back → negative carry → short."""
        front = np.full(100, 100.0)
        back = np.full(100, 105.0)
        bars = self._bars(front, back, days=30)
        gen = FuturesCarrySignal()
        sigs = gen.generate(bars, symbol="CL")
        assert len(sigs) > 0
        assert all(s.side == -1 for s in sigs)
        assert all(s.metadata["carry"] < 0 for s in sigs)

    def test_annualization_toggle(self):
        front = np.full(50, 105.0)
        back = np.full(50, 100.0)
        bars = self._bars(front, back, days=30)
        ann = FuturesCarrySignal(params={"annualize": True}).generate(bars)
        raw = FuturesCarrySignal(params={"annualize": False}).generate(bars)
        # Annualized carry should be ~12x the raw carry for a 30-day contract
        # (365/30 ≈ 12.17).
        ratio = ann[0].metadata["carry"] / raw[0].metadata["carry"]
        np.testing.assert_allclose(ratio, 365.0 / 30.0, rtol=1e-9)

    def test_default_days_when_column_missing(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        bars = pd.DataFrame(
            {"front_price": np.full(50, 105.0), "back_price": np.full(50, 100.0)},
            index=idx,
        )
        gen = FuturesCarrySignal(params={"default_days_to_expiry": 60})
        sigs = gen.generate(bars)
        assert sigs[0].metadata["days_to_expiry"] == 60.0

    def test_confidence_in_unit_interval(self):
        rng = np.random.default_rng(0)
        n = 300
        front = 100.0 + rng.normal(0.0, 2.0, size=n)
        back = 100.0 + rng.normal(0.0, 2.0, size=n)
        bars = self._bars(list(front), list(back), days=30)
        gen = FuturesCarrySignal()
        sigs = gen.generate(bars)
        assert all(0.0 <= s.confidence <= 1.0 for s in sigs)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            FuturesCarrySignal(params={"default_days_to_expiry": 0})
        with pytest.raises(ValueError):
            FuturesCarrySignal(params={"confidence_window": 1})


# ---------------------------------------------------------------------------
# FundingRateArbSignal
# ---------------------------------------------------------------------------

class TestFundingRateArbSignal:
    def test_entry_and_exit_transitions(self):
        """
        Build a funding-rate series that crosses up through entry and
        later falls below exit. Expect exactly one entry and one exit.
        """
        # ~0.04% per 8h ≈ 57% annualized → above entry_threshold=0.10
        high_rate = 0.0004
        # 0.001% per 8h ≈ 1.1% annualized → below exit_threshold=0.02
        low_rate = 0.00001
        rates = (
            [0.0] * 10
            + [high_rate] * 20  # strong yield → should trigger entry
            + [low_rate] * 20   # funding collapses → should trigger exit
        )
        idx = pd.date_range("2024-01-01", periods=len(rates), freq="8h")
        bars = pd.DataFrame({"funding_rate": rates}, index=idx)

        gen = FundingRateArbSignal()
        sigs = gen.generate(bars, symbol="BTC-PERP")
        events = [(s.metadata["event"], s.side) for s in sigs]
        assert events.count(("entry", 1)) == 1
        assert events.count(("exit", 0)) == 1
        # Entry should come before exit.
        entry_idx = events.index(("entry", 1))
        exit_idx = events.index(("exit", 0))
        assert entry_idx < exit_idx

    def test_no_signal_when_funding_stays_moderate(self):
        """Funding between exit and entry thresholds should emit nothing."""
        # ~10%/yr funding — above exit (2%) but below entry (10%) band.
        mid_rate = 0.00008  # annualized ≈ 9.2%
        rates = [mid_rate] * 50
        idx = pd.date_range("2024-01-01", periods=50, freq="8h")
        bars = pd.DataFrame({"funding_rate": rates}, index=idx)
        gen = FundingRateArbSignal()
        sigs = gen.generate(bars)
        assert sigs == []

    def test_annualized_input_bypasses_conversion(self):
        rates = [0.20, 0.20, 0.01]  # already annualized
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        bars = pd.DataFrame({"funding_rate": rates}, index=idx)
        gen = FundingRateArbSignal(params={"annualized": True})
        sigs = gen.generate(bars, symbol="ETH-PERP")
        # 0.20 > entry → entry; then 0.01 < exit → exit.
        events = [(s.metadata["event"], s.side) for s in sigs]
        assert events == [("entry", 1), ("exit", 0)]

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            FundingRateArbSignal(params={"entry_threshold": 0.02, "exit_threshold": 0.10})
        with pytest.raises(ValueError):
            FundingRateArbSignal(params={"payments_per_day": 0})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
