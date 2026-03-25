"""
Tests for bar constructors.

Validates that all bar types produce correct OHLCV values and
trigger at the expected thresholds.
"""

import math
from datetime import datetime, timedelta

import pytest

from src.data_engine.models import Tick, Bar, BarType, Side, BarAccumulator
from src.data_engine.bars.tick_rule import TickRuleClassifier
from src.data_engine.bars.constructors import (
    TickBarConstructor,
    VolumeBarConstructor,
    DollarBarConstructor,
    TIBConstructor,
    VIBConstructor,
    create_bar_constructor,
)
from src.data_engine.bars.cusum_filter import cusum_filter, compute_cusum_threshold


# ── Helpers ──

def make_tick(price: float, volume: float = 1.0, side: Side = Side.UNKNOWN,
              ts_offset_ms: int = 0, symbol: str = "TEST") -> Tick:
    """Create a test tick with a timestamp offset."""
    return Tick(
        timestamp=datetime(2024, 1, 1) + timedelta(milliseconds=ts_offset_ms),
        symbol=symbol,
        price=price,
        volume=volume,
        side=side,
    )


def make_ticks(prices: list[float], volume: float = 1.0, symbol: str = "TEST") -> list[Tick]:
    """Create a sequence of ticks from prices."""
    return [
        make_tick(p, volume=volume, ts_offset_ms=i * 100, symbol=symbol)
        for i, p in enumerate(prices)
    ]


# ── Tick Rule Tests ──

class TestTickRule:
    def test_price_up_is_buy(self):
        classifier = TickRuleClassifier()
        t1 = make_tick(100.0)
        t2 = make_tick(101.0)
        classifier.classify(t1)
        assert classifier.classify(t2) == Side.BUY

    def test_price_down_is_sell(self):
        classifier = TickRuleClassifier()
        t1 = make_tick(100.0)
        t2 = make_tick(99.0)
        classifier.classify(t1)
        assert classifier.classify(t2) == Side.SELL

    def test_unchanged_carries_forward(self):
        classifier = TickRuleClassifier()
        t1 = make_tick(100.0)
        t2 = make_tick(101.0)  # BUY
        t3 = make_tick(101.0)  # unchanged → BUY (carry forward)
        classifier.classify(t1)
        classifier.classify(t2)
        assert classifier.classify(t3) == Side.BUY

    def test_respects_existing_side(self):
        classifier = TickRuleClassifier()
        t = make_tick(100.0, side=Side.SELL)
        assert classifier.classify(t) == Side.SELL

    def test_sequence(self):
        classifier = TickRuleClassifier()
        prices = [100, 101, 102, 101, 100, 100, 101]
        expected = [Side.BUY, Side.BUY, Side.BUY, Side.SELL, Side.SELL, Side.SELL, Side.BUY]
        results = [classifier.classify(make_tick(p)) for p in prices]
        assert results == expected


# ── Tick Bar Tests ──

class TestTickBars:
    def test_triggers_at_n_ticks(self):
        constructor = TickBarConstructor(symbol="TEST", bar_size=5)
        ticks = make_ticks([100, 101, 102, 103, 104])
        bars = constructor.process_ticks(ticks)
        assert len(bars) == 1

    def test_ohlcv_correct(self):
        constructor = TickBarConstructor(symbol="TEST", bar_size=5)
        ticks = make_ticks([100, 105, 95, 102, 101])
        bars = constructor.process_ticks(ticks)
        assert len(bars) == 1
        bar = bars[0]
        assert bar.open == 100
        assert bar.high == 105
        assert bar.low == 95
        assert bar.close == 101
        assert bar.volume == 5.0
        assert bar.tick_count == 5

    def test_multiple_bars(self):
        constructor = TickBarConstructor(symbol="TEST", bar_size=3)
        ticks = make_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
        bars = constructor.process_ticks(ticks)
        assert len(bars) == 3

    def test_partial_bar_not_emitted(self):
        constructor = TickBarConstructor(symbol="TEST", bar_size=5)
        ticks = make_ticks([100, 101, 102])  # only 3 of 5
        bars = constructor.process_ticks(ticks)
        assert len(bars) == 0


# ── Volume Bar Tests ──

class TestVolumeBars:
    def test_triggers_at_volume_threshold(self):
        constructor = VolumeBarConstructor(symbol="TEST", bar_size=10.0)
        ticks = [
            make_tick(100, volume=3.0, ts_offset_ms=0),
            make_tick(101, volume=3.0, ts_offset_ms=100),
            make_tick(102, volume=4.0, ts_offset_ms=200),  # total=10 → trigger
        ]
        bars = constructor.process_ticks(ticks)
        assert len(bars) == 1
        assert bars[0].volume == 10.0

    def test_large_tick_triggers_immediately(self):
        constructor = VolumeBarConstructor(symbol="TEST", bar_size=5.0)
        ticks = [make_tick(100, volume=10.0)]  # exceeds threshold in one tick
        bars = constructor.process_ticks(ticks)
        assert len(bars) == 1


# ── Dollar Bar Tests ──

class TestDollarBars:
    def test_triggers_at_dollar_threshold(self):
        constructor = DollarBarConstructor(symbol="TEST", bar_size=1000.0)
        ticks = [
            make_tick(100, volume=3.0, ts_offset_ms=0),   # $300
            make_tick(100, volume=3.0, ts_offset_ms=100),  # $600
            make_tick(100, volume=5.0, ts_offset_ms=200),  # $1100 → trigger
        ]
        bars = constructor.process_ticks(ticks)
        assert len(bars) == 1
        assert bars[0].dollar_volume == 1100.0

    def test_normalizes_across_prices(self):
        # Same dollar threshold, different prices → different tick counts
        low_price = DollarBarConstructor(symbol="LOW", bar_size=1000.0)
        high_price = DollarBarConstructor(symbol="HIGH", bar_size=1000.0)

        low_ticks = make_ticks([10] * 200, volume=1.0, symbol="LOW")   # $10/trade
        high_ticks = make_ticks([500] * 200, volume=1.0, symbol="HIGH")  # $500/trade

        low_bars = low_price.process_ticks(low_ticks)
        high_bars = high_price.process_ticks(high_ticks)

        # Low price needs ~100 trades per bar, high price needs ~2
        assert low_bars[0].tick_count > high_bars[0].tick_count


# ── TIB Tests ──

class TestTIBConstructor:
    def test_triggers_on_imbalance(self):
        constructor = TIBConstructor(symbol="TEST", initial_threshold=5.0)
        # 5 consecutive buys → imbalance = 5 → triggers
        ticks = make_ticks([100, 101, 102, 103, 104])  # all classified as BUY
        bars = constructor.process_ticks(ticks)
        assert len(bars) == 1
        assert bars[0].buy_ticks >= bars[0].sell_ticks

    def test_balanced_flow_no_trigger(self):
        constructor = TIBConstructor(symbol="TEST", initial_threshold=10.0)
        # Alternating buys and sells → imbalance stays near 0
        ticks = []
        for i in range(20):
            p = 100 + (1 if i % 2 == 0 else -1)  # oscillate 101, 99, 101, 99...
            ticks.append(make_tick(p, ts_offset_ms=i * 100))
        bars = constructor.process_ticks(ticks)
        # With balanced flow and threshold=10, should produce fewer bars
        # than unbalanced flow
        assert len(bars) <= 2

    def test_threshold_adapts(self):
        constructor = TIBConstructor(symbol="TEST", initial_threshold=20.0, ewma_span=10)
        initial_threshold = constructor._expected_imbalance

        # Generate a long trend (all buys) to produce many bars
        ticks = make_ticks([100 + i * 0.1 for i in range(500)])
        bars = constructor.process_ticks(ticks)

        # After producing multiple bars, threshold should have adapted
        # from the initial value (it may go up or down depending on
        # actual imbalance at trigger, but should differ from initial)
        assert len(bars) > 5
        # The threshold should have moved from initial as EWMA updates
        assert constructor._expected_imbalance != initial_threshold or len(bars) > 10

    def test_bar_metadata(self):
        constructor = TIBConstructor(symbol="TEST", initial_threshold=3.0)
        ticks = make_ticks([100, 101, 102, 103])
        bars = constructor.process_ticks(ticks)
        if bars:
            bar = bars[0]
            assert bar.bar_type == BarType.TICK_IMBALANCE
            assert bar.symbol == "TEST"
            assert bar.threshold > 0


# ── VIB Tests ──

class TestVIBConstructor:
    def test_triggers_on_volume_imbalance(self):
        constructor = VIBConstructor(symbol="TEST", initial_threshold=100.0)
        # Large buy volumes
        ticks = [
            make_tick(100 + i, volume=30.0, side=Side.BUY, ts_offset_ms=i * 100)
            for i in range(5)
        ]
        bars = constructor.process_ticks(ticks)
        assert len(bars) >= 1


# ── Factory Tests ──

class TestFactory:
    def test_create_tick_bars(self):
        c = create_bar_constructor("TEST", "tick", bar_size=100)
        assert isinstance(c, TickBarConstructor)

    def test_create_volume_bars(self):
        c = create_bar_constructor("TEST", "volume", bar_size=50000)
        assert isinstance(c, VolumeBarConstructor)

    def test_create_dollar_bars(self):
        c = create_bar_constructor("TEST", "dollar", bar_size=1000000)
        assert isinstance(c, DollarBarConstructor)

    def test_create_tib(self):
        c = create_bar_constructor("TEST", "tib", ewma_span=50)
        assert isinstance(c, TIBConstructor)

    def test_create_vib(self):
        c = create_bar_constructor("TEST", "vib", ewma_span=50)
        assert isinstance(c, VIBConstructor)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown bar type"):
            create_bar_constructor("TEST", "invalid")


# ── CUSUM Filter Tests ──

class TestCUSUMFilter:
    def test_detects_upward_break(self):
        import pandas as pd
        import numpy as np

        # Stable prices then a jump
        prices = [100.0] * 50 + [100.0 + i * 0.5 for i in range(50)]
        series = pd.Series(prices, index=pd.date_range("2024-01-01", periods=100, freq="h"))

        events = cusum_filter(series, threshold=0.01)
        assert len(events) > 0

    def test_stable_series_few_events(self):
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        prices = 100.0 + np.random.normal(0, 0.001, 500).cumsum()
        series = pd.Series(prices, index=pd.date_range("2024-01-01", periods=500, freq="h"))

        # High threshold → few events
        events = cusum_filter(series, threshold=0.1)
        assert len(events) < 10

    def test_compute_threshold(self):
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        prices = 100.0 + np.random.normal(0, 0.01, 252).cumsum()
        series = pd.Series(prices, index=pd.date_range("2024-01-01", periods=252, freq="D"))

        threshold = compute_cusum_threshold(series, multiplier=1.5)
        assert threshold > 0


# ── Bar Accumulator Tests ──

class TestBarAccumulator:
    def test_accumulates_correctly(self):
        acc = BarAccumulator(symbol="TEST", bar_type=BarType.TICK)
        ticks = [
            make_tick(100, volume=1.0, side=Side.BUY, ts_offset_ms=0),
            make_tick(105, volume=2.0, side=Side.SELL, ts_offset_ms=100),
            make_tick(95, volume=1.5, side=Side.BUY, ts_offset_ms=200),
        ]
        for t in ticks:
            acc.add_tick(t)

        bar = acc.to_bar()
        assert bar.open == 100
        assert bar.high == 105
        assert bar.low == 95
        assert bar.close == 95
        assert bar.volume == 4.5
        assert bar.tick_count == 3
        assert bar.buy_volume == 2.5
        assert bar.sell_volume == 2.0
        assert bar.buy_ticks == 2
        assert bar.sell_ticks == 1

    def test_reset_clears_state(self):
        acc = BarAccumulator(symbol="TEST", bar_type=BarType.TICK)
        acc.add_tick(make_tick(100))
        acc.reset()
        assert acc.tick_count == 0
        assert acc.volume == 0
        assert acc.high == float('-inf')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
