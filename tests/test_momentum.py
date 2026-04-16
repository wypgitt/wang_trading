"""Tests for the momentum signal generators."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.signal_battery.base_signal import BaseSignalGenerator, Signal
from src.signal_battery.momentum import (
    CrossSectionalMomentumSignal,
    TimeSeriesMomentumSignal,
)


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

class TestSignal:
    def test_serialization_to_dict(self):
        sig = Signal(
            timestamp=datetime(2024, 3, 10, 12, 0),
            symbol="AAPL",
            family="ts_momentum",
            side=1,
            confidence=0.75,
            metadata={"lookback": 252},
        )
        d = sig.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["side"] == 1
        assert d["confidence"] == 0.75
        assert d["family"] == "ts_momentum"
        # timestamp serialised to ISO 8601
        assert isinstance(d["timestamp"], str)
        assert "2024-03-10" in d["timestamp"]
        assert d["metadata"] == {"lookback": 252}

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime(2024, 1, 1), symbol="X",
                family="f", side=2, confidence=0.5,
            )

    def test_confidence_bounds_enforced(self):
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime(2024, 1, 1), symbol="X",
                family="f", side=1, confidence=1.5,
            )
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime(2024, 1, 1), symbol="X",
                family="f", side=1, confidence=-0.1,
            )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TestBaseSignalGenerator:
    def test_validate_input_missing_column(self):
        gen = TimeSeriesMomentumSignal()
        bad = pd.DataFrame({"open": [1.0, 2.0]})
        with pytest.raises(ValueError, match="missing columns"):
            gen.validate_input(bad)

    def test_validate_input_empty(self):
        gen = TimeSeriesMomentumSignal()
        with pytest.raises(ValueError, match="empty"):
            gen.validate_input(pd.DataFrame({"close": []}))


# ---------------------------------------------------------------------------
# Time-series momentum
# ---------------------------------------------------------------------------

def _trending_bars(n: int, drift: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": close}, index=idx)


class TestTimeSeriesMomentum:
    def test_long_signal_for_uptrend(self):
        bars = _trending_bars(n=600, drift=0.002, seed=0)  # ~50 bps/day up
        gen = TimeSeriesMomentumSignal(
            params={"lookbacks": [21, 63, 126], "history_window": 126}
        )
        sigs = gen.generate(bars, symbol="UP")
        assert len(sigs) > 0
        # Final signal on a persistent uptrend must be long.
        assert sigs[-1].side == 1
        # And at least 80% of emitted signals should be long.
        fraction_long = sum(1 for s in sigs if s.side == 1) / len(sigs)
        assert fraction_long > 0.8

    def test_short_signal_for_downtrend(self):
        bars = _trending_bars(n=600, drift=-0.002, seed=1)
        gen = TimeSeriesMomentumSignal(
            params={"lookbacks": [21, 63, 126], "history_window": 126}
        )
        sigs = gen.generate(bars, symbol="DOWN")
        assert len(sigs) > 0
        assert sigs[-1].side == -1
        fraction_short = sum(1 for s in sigs if s.side == -1) / len(sigs)
        assert fraction_short > 0.8

    def test_confidence_higher_for_stronger_trend(self):
        weak = _trending_bars(n=600, drift=0.0005, seed=2)
        strong = _trending_bars(n=600, drift=0.004, seed=2)
        gen = TimeSeriesMomentumSignal(
            params={"lookbacks": [21, 63, 126], "history_window": 126}
        )
        c_weak = np.mean(
            [s.confidence for s in gen.generate(weak, symbol="W")]
        )
        c_strong = np.mean(
            [s.confidence for s in gen.generate(strong, symbol="S")]
        )
        assert c_strong > c_weak

    def test_returns_empty_on_insufficient_history(self):
        bars = _trending_bars(n=50, drift=0.001, seed=0)
        gen = TimeSeriesMomentumSignal(
            params={"lookbacks": [21, 63, 126], "history_window": 126}
        )
        assert gen.generate(bars, symbol="SHORT") == []

    def test_metadata_fields_populated(self):
        bars = _trending_bars(n=500, drift=0.002)
        gen = TimeSeriesMomentumSignal(
            params={"lookbacks": [21, 63], "history_window": 100}
        )
        sigs = gen.generate(bars, symbol="ABC")
        assert len(sigs) > 0
        md = sigs[-1].metadata
        assert md["lookbacks"] == [21, 63]
        assert "aggregate" in md
        assert "z_21" in md["z_scores"]

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            TimeSeriesMomentumSignal(params={"lookbacks": []})
        with pytest.raises(ValueError):
            TimeSeriesMomentumSignal(params={"lookbacks": [21, 63], "weights": [1, -1]})
        with pytest.raises(ValueError):
            TimeSeriesMomentumSignal(params={"history_window": 5})


# ---------------------------------------------------------------------------
# Cross-sectional momentum
# ---------------------------------------------------------------------------

def _make_panel(
    n_symbols: int = 40, n_bars: int = 300, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Build a panel: 8 uptrends, 8 downtrends, rest flat noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_symbols):
        if i < 8:
            drift = 0.002
        elif i < 16:
            drift = -0.002
        else:
            drift = 0.0
        returns = rng.normal(loc=drift, scale=0.01, size=n_bars)
        close = 100.0 * np.exp(np.cumsum(returns))
        panel[f"S{i:02d}"] = pd.DataFrame({"close": close}, index=idx)
    return panel


class TestCrossSectionalMomentum:
    def test_ranks_winners_long_and_losers_short(self):
        # 40 symbols → top decile = 4 longs, bottom decile = 4 shorts.
        panel = _make_panel(n_symbols=40, n_bars=300)
        gen = CrossSectionalMomentumSignal(
            params={"lookback_bars": 250, "skip_bars": 20, "min_universe_size": 20}
        )
        sigs = gen.generate(panel=panel)
        by_symbol = {s.symbol: s for s in sigs}

        # S00..S07 are the uptrend cohort; at least 3 of them should land in
        # the top-4 decile. S08..S15 are downtrends; same for bottom.
        winners_long = sum(1 for s in range(8) if by_symbol[f"S{s:02d}"].side == 1)
        losers_short = sum(1 for s in range(8, 16) if by_symbol[f"S{s:02d}"].side == -1)
        assert winners_long >= 3
        assert losers_short >= 3

    def test_raises_on_small_universe(self):
        panel = _make_panel(n_symbols=10, n_bars=300)
        gen = CrossSectionalMomentumSignal()
        with pytest.raises(ValueError, match="universe too small"):
            gen.generate(panel=panel)

    def test_skip_month_excludes_recent_month(self):
        """
        Build a symbol whose last 20 bars rally strongly but whose prior 12
        months are flat; with skip=20 the signal should NOT place it in the
        top decile.
        """
        n_symbols = 40
        n_bars = 300
        idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")

        rng = np.random.default_rng(42)
        panel: dict[str, pd.DataFrame] = {}
        # Background: flat symbols.
        for i in range(n_symbols):
            drift = rng.normal(0.0, 0.0005)
            returns = rng.normal(loc=drift, scale=0.01, size=n_bars)
            panel[f"S{i:02d}"] = pd.DataFrame(
                {"close": 100.0 * np.exp(np.cumsum(returns))}, index=idx,
            )
        # Inject a target symbol: flat for the 12 months prior to the
        # 20-bar skip window, and only the last 20 bars explode upward.
        base = np.concatenate(
            [np.zeros(n_bars - 20), np.full(20, 0.05)]  # last 20 bars: +5% each
        )
        panel["TARGET"] = pd.DataFrame(
            {"close": 100.0 * np.exp(np.cumsum(base))}, index=idx
        )

        gen = CrossSectionalMomentumSignal(
            params={"lookback_bars": 250, "skip_bars": 20, "min_universe_size": 20}
        )
        sigs = {s.symbol: s for s in gen.generate(panel=panel)}
        # Because we skip the last 20 bars (the rally), TARGET's measured
        # 12m-skip-1m momentum is ~0, so it should NOT land in the top decile.
        assert sigs["TARGET"].side != 1

    def test_rejects_missing_panel(self):
        gen = CrossSectionalMomentumSignal()
        with pytest.raises(ValueError, match="panel"):
            gen.generate()

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            CrossSectionalMomentumSignal(params={"top_decile": 0.2})
        with pytest.raises(ValueError):
            CrossSectionalMomentumSignal(params={"bottom_decile": 0.8})
        with pytest.raises(ValueError):
            CrossSectionalMomentumSignal(
                params={"lookback_bars": 20, "skip_bars": 30}
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
