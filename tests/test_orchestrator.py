"""Tests for the Signal Battery orchestrator."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.signal_battery.base_signal import BaseSignalGenerator, Signal
from src.signal_battery.orchestrator import SignalBattery, create_default_battery


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _AlwaysLong(BaseSignalGenerator):
    """Deterministic generator that emits one long signal per bar."""

    REQUIRED_COLUMNS = ("close",)

    def __init__(self, name: str = "always_long"):
        super().__init__(name=name)

    def generate(self, bars, symbol="UNKNOWN", **_):
        self.validate_input(bars)
        return [
            Signal(
                timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                symbol=symbol,
                family=self.name,
                side=1,
                confidence=0.5,
                metadata={"k": "v"},
            )
            for t in bars.index
        ]


class _AlwaysShort(BaseSignalGenerator):
    REQUIRED_COLUMNS = ("close",)

    def __init__(self, name: str = "always_short"):
        super().__init__(name=name)

    def generate(self, bars, symbol="UNKNOWN", **_):
        self.validate_input(bars)
        return [
            Signal(
                timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                symbol=symbol,
                family=self.name,
                side=-1,
                confidence=0.3,
                metadata={},
            )
            for t in bars.index
        ]


class _Crashy(BaseSignalGenerator):
    REQUIRED_COLUMNS = ("close",)

    def __init__(self, name: str = "crashy"):
        super().__init__(name=name)

    def generate(self, bars, symbol="UNKNOWN", **_):
        raise RuntimeError("boom")


def _bars(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": np.arange(100, 100 + n, dtype=float)}, index=idx)


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------

class TestSignalBattery:
    def test_runs_all_registered_generators(self):
        b = SignalBattery()
        b.register(_AlwaysLong())
        b.register(_AlwaysShort())

        bars = _bars(n=10)
        df = b.generate_all(bars, event_timestamps=None, symbol="AAPL")
        # 10 bars × 2 families = 20 signals.
        assert len(df) == 20
        assert set(df["family"].unique()) == {"always_long", "always_short"}

    def test_output_schema_matches_spec(self):
        b = SignalBattery()
        b.register(_AlwaysLong())
        df = b.generate_all(_bars(n=5), event_timestamps=None, symbol="AAPL")
        # Required base columns are in order, metadata columns are prefixed.
        base = ["timestamp", "symbol", "family", "side", "confidence"]
        assert list(df.columns)[:5] == base
        assert any(c.startswith("meta_") for c in df.columns)
        assert df["side"].unique().tolist() == [1]

    def test_events_filter_restricts_output(self):
        b = SignalBattery()
        b.register(_AlwaysLong())
        bars = _bars(n=10)
        events = bars.index[[2, 5, 8]]  # 3 events
        df = b.generate_all(bars, event_timestamps=events, symbol="AAPL")
        assert len(df) == 3

    def test_empty_events_returns_empty_dataframe(self):
        """An explicit empty event list filters every signal out."""
        b = SignalBattery()
        b.register(_AlwaysLong())
        df = b.generate_all(
            _bars(n=5), event_timestamps=pd.DatetimeIndex([]), symbol="AAPL"
        )
        assert df.empty
        # Schema is preserved even when empty.
        assert list(df.columns) == [
            "timestamp", "symbol", "family", "side", "confidence",
        ]

    def test_no_generators_returns_empty(self):
        b = SignalBattery()
        df = b.generate_all(_bars(n=5), event_timestamps=None)
        assert df.empty
        # Even when empty the schema is present for downstream consumers.
        assert list(df.columns) == [
            "timestamp", "symbol", "family", "side", "confidence",
        ]

    def test_crashing_generator_is_skipped(self):
        b = SignalBattery()
        b.register(_AlwaysLong())
        b.register(_Crashy())
        df = b.generate_all(_bars(n=5), event_timestamps=None, symbol="AAPL")
        # Crashy's signals are dropped, always_long's survive.
        assert len(df) == 5
        assert "crashy" not in df["family"].unique()

    def test_conflicting_signals_all_preserved(self):
        """Same bar + same symbol, long AND short family → both kept."""
        b = SignalBattery()
        b.register(_AlwaysLong())
        b.register(_AlwaysShort())
        bars = _bars(n=3)
        df = b.generate_all(bars, event_timestamps=None, symbol="AAPL")
        # At every bar, exactly one long and one short signal.
        per_bar = df.groupby("timestamp")["side"].agg(set)
        for sides in per_bar:
            assert sides == {1, -1}

    def test_signal_stats(self):
        b = SignalBattery()
        b.register(_AlwaysLong())
        b.register(_AlwaysShort())
        df = b.generate_all(_bars(n=10), event_timestamps=None, symbol="AAPL")
        stats = b.get_signal_stats(df)
        assert set(stats.keys()) == {"always_long", "always_short"}
        al = stats["always_long"]
        assert al["count"] == 10
        assert al["long_ratio"] == 1.0
        assert al["short_ratio"] == 0.0
        assert abs(al["avg_confidence"] - 0.5) < 1e-9
        short = stats["always_short"]
        assert short["short_ratio"] == 1.0

    def test_get_signal_stats_empty(self):
        b = SignalBattery()
        assert b.get_signal_stats(None) == {}
        assert b.get_signal_stats(pd.DataFrame()) == {}

    def test_get_active_families(self):
        b = SignalBattery()
        b.register(_AlwaysLong())
        b.register(_AlwaysShort())
        assert b.get_active_families() == ["always_long", "always_short"]

    def test_register_invalid_kind(self):
        b = SignalBattery()
        with pytest.raises(ValueError):
            b.register(_AlwaysLong(), kind="invalid")

    def test_bars_extra_requires_context_key(self):
        b = SignalBattery()
        with pytest.raises(ValueError):
            b.register(_AlwaysLong(), kind="bars_extra")


# ---------------------------------------------------------------------------
# Default battery factory
# ---------------------------------------------------------------------------

class TestCreateDefaultBattery:
    def test_registers_all_families(self):
        battery = create_default_battery()
        families = battery.get_active_families()
        # All 10 generators from the 7 family groups should be registered.
        expected = {
            "ts_momentum",
            "cs_momentum",
            "mean_reversion",
            "stat_arb",
            "ma_crossover",
            "donchian_breakout",
            "futures_carry",
            "funding_rate_arb",
            "cross_exchange_arb",
            "vrp",
        }
        assert expected.issubset(set(families))

    def test_config_overrides_propagate(self):
        """A param passed via the config dict should reach the generator."""
        battery = create_default_battery(
            config={"ts_momentum": {"lookbacks": [5, 10]}}
        )
        # Find the ts_momentum registration and confirm its params.
        ts = next(
            r for r in battery._registry if r.name == "ts_momentum"
        ).generator
        assert ts._lookbacks == (5, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
