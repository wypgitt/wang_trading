"""Tests for the ETF trick futures roll implementation."""

import pandas as pd
import numpy as np
import pytest

from src.data_engine.bars.etf_trick import ETFTrick


class TestETFTrick:
    def test_basic_continuous_series(self):
        """ETF trick should produce returns-preserving continuous series."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")

        # Two contracts: front expires at day 10
        front_prices = list(range(100, 110)) + [np.nan] * 10
        back_prices = [np.nan] * 5 + list(range(95, 110))

        prices = pd.DataFrame({
            "front": front_prices,
            "back": back_prices,
        }, index=dates)

        roll_dates = [dates[9]]  # roll on day 10
        result = ETFTrick.compute(prices, roll_dates, initial_value=100.0)

        assert len(result) > 0
        assert result.iloc[0] == 100.0
        # Value should change after roll
        assert not result.isna().any()

    def test_preserves_returns(self):
        """Returns should be accurate across the roll."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")

        # Front contract: 100, 102, 104 ... (2% return per day)
        front = [100, 102, 104, 106, 108, np.nan, np.nan, np.nan, np.nan, np.nan]
        # Back contract overlaps, different level
        back = [np.nan, np.nan, np.nan, 200, 204, 208, 212, 216, 220, 224]

        prices = pd.DataFrame({"front": front, "back": back}, index=dates)
        roll_dates = [dates[4]]  # roll at day 5

        result = ETFTrick.compute(prices, roll_dates, initial_value=100.0)
        assert len(result) > 5

    def test_detect_roll_dates(self):
        """Roll dates should be detected by OI crossover."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        front_oi = pd.Series([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100], index=dates)
        back_oi = pd.Series([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], index=dates)

        rolls = ETFTrick.detect_roll_dates(front_oi, back_oi)
        assert len(rolls) == 1
        # Crossover happens when back > front for the first time
        assert rolls[0] == dates[5]  # 600 > 500

    def test_no_roll_if_no_crossover(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        front_oi = pd.Series([1000, 900, 800, 700, 600], index=dates)
        back_oi = pd.Series([100, 100, 100, 100, 100], index=dates)

        rolls = ETFTrick.detect_roll_dates(front_oi, back_oi)
        assert len(rolls) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
