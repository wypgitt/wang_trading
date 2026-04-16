"""Tests for AFML Ch. 4 sample weights (uniqueness, bootstrap, decay)."""

import time

import numpy as np
import pandas as pd
import pytest

from src.labeling.sample_weights import (
    compute_sample_weights,
    get_average_uniqueness,
    get_num_concurrent_events,
    get_return_attribution_weights,
    get_time_decay_weights,
    sequential_bootstrap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(n: int, freq: str = "1min") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq=freq)


def _events(close_index: pd.DatetimeIndex,
            ranges: list[tuple[int, int]]) -> tuple[pd.DatetimeIndex,
                                                    pd.DatetimeIndex]:
    starts = pd.DatetimeIndex([close_index[s] for s, _ in ranges])
    ends = pd.DatetimeIndex([close_index[e] for _, e in ranges])
    return starts, ends


# ---------------------------------------------------------------------------
# get_num_concurrent_events
# ---------------------------------------------------------------------------

class TestConcurrentEvents:
    def test_non_overlapping_count_one(self):
        idx = _make_index(100)
        starts, ends = _events(idx, [(0, 20), (30, 50), (60, 80)])
        concurrent = get_num_concurrent_events(idx, starts, ends)
        # Between labelled ranges the count is 0; inside each range it is 1.
        assert (concurrent.iloc[0:21] == 1).all()
        assert (concurrent.iloc[21:30] == 0).all()
        assert (concurrent.iloc[30:51] == 1).all()
        assert (concurrent.iloc[60:81] == 1).all()

    def test_fully_overlapping_count_two(self):
        idx = _make_index(50)
        starts, ends = _events(idx, [(10, 30), (10, 30)])
        concurrent = get_num_concurrent_events(idx, starts, ends)
        assert (concurrent.iloc[10:31] == 2).all()
        assert (concurrent.iloc[:10] == 0).all()
        assert (concurrent.iloc[31:] == 0).all()

    def test_partial_overlap_transitions(self):
        idx = _make_index(30)
        starts, ends = _events(idx, [(5, 15), (10, 20)])
        concurrent = get_num_concurrent_events(idx, starts, ends)
        assert (concurrent.iloc[5:10] == 1).all()
        assert (concurrent.iloc[10:16] == 2).all()
        assert (concurrent.iloc[16:21] == 1).all()

    def test_empty_events(self):
        idx = _make_index(10)
        starts = pd.DatetimeIndex([])
        ends = pd.DatetimeIndex([])
        out = get_num_concurrent_events(idx, starts, ends)
        assert (out == 0).all()
        assert out.index.equals(idx)

    def test_mismatched_lengths_raise(self):
        idx = _make_index(10)
        with pytest.raises(ValueError):
            get_num_concurrent_events(
                idx, pd.DatetimeIndex([idx[0]]), pd.DatetimeIndex([]),
            )


# ---------------------------------------------------------------------------
# get_average_uniqueness
# ---------------------------------------------------------------------------

class TestAverageUniqueness:
    def test_non_overlapping_is_one(self):
        idx = _make_index(100)
        starts, ends = _events(idx, [(0, 20), (30, 50), (60, 80)])
        u = get_average_uniqueness(starts, ends, idx)
        assert np.allclose(u.to_numpy(), 1.0)

    def test_fully_overlapping_is_half(self):
        idx = _make_index(50)
        starts, ends = _events(idx, [(10, 30), (10, 30)])
        u = get_average_uniqueness(starts, ends, idx)
        assert np.allclose(u.to_numpy(), 0.5)

    def test_three_mutual_overlap_is_third(self):
        idx = _make_index(50)
        starts, ends = _events(idx, [(10, 30), (10, 30), (10, 30)])
        u = get_average_uniqueness(starts, ends, idx)
        assert np.allclose(u.to_numpy(), 1.0 / 3.0)

    def test_partial_overlap_between_half_and_one(self):
        # Event A: [5,15]; Event B: [10,20]. A is solo on [5,9] (5 bars),
        # shared on [10,15] (6 bars). uniqueness_A = (5*1 + 6*0.5) / 11.
        idx = _make_index(30)
        starts, ends = _events(idx, [(5, 15), (10, 20)])
        u = get_average_uniqueness(starts, ends, idx)
        expected = (5 * 1.0 + 6 * 0.5) / 11
        assert np.isclose(u.iloc[0], expected)
        assert np.isclose(u.iloc[1], expected)  # symmetric


# ---------------------------------------------------------------------------
# sequential_bootstrap
# ---------------------------------------------------------------------------

class TestSequentialBootstrap:
    def test_fallback_path_distributes_draws(self):
        # 50 events, uniform uniqueness → draws should cover many indices.
        u = pd.Series(np.ones(50), index=_make_index(50))
        draws = sequential_bootstrap(u, n_samples=200, random_state=0)
        assert draws.shape == (200,)
        assert draws.min() >= 0 and draws.max() < 50
        # Should hit most indices — not degenerate to a single one.
        assert len(np.unique(draws)) > 25

    def test_reproducible(self):
        u = pd.Series(
            np.linspace(0.2, 1.0, 40), index=_make_index(40),
        )
        a = sequential_bootstrap(u, n_samples=100, random_state=123)
        b = sequential_bootstrap(u, n_samples=100, random_state=123)
        assert np.array_equal(a, b)

    def test_full_algorithm_prefers_non_overlapping(self):
        """
        Two events overlap fully; a third is isolated. The isolated event
        should be drawn noticeably more often than either overlap-twin.
        """
        idx = _make_index(100)
        # Events: two overlap fully on [0, 20]; one solo on [60, 80].
        starts, ends = _events(idx, [(0, 20), (0, 20), (60, 80)])
        u = get_average_uniqueness(starts, ends, idx)
        draws = sequential_bootstrap(
            u, event_starts=starts, event_ends=ends, close_index=idx,
            n_samples=3000, random_state=0,
        )
        counts = np.bincount(draws, minlength=3)
        # The solo event should dominate the two overlapping ones.
        assert counts[2] > counts[0]
        assert counts[2] > counts[1]

    def test_full_algorithm_returns_valid_indices(self):
        idx = _make_index(200)
        starts, ends = _events(
            idx, [(10, 30), (40, 60), (20, 45), (100, 120), (110, 130)],
        )
        u = get_average_uniqueness(starts, ends, idx)
        draws = sequential_bootstrap(
            u, event_starts=starts, event_ends=ends, close_index=idx,
            n_samples=50, random_state=7,
        )
        assert draws.shape == (50,)
        assert draws.min() >= 0
        assert draws.max() < 5

    def test_empty_uniqueness_returns_empty(self):
        u = pd.Series([], dtype=float)
        out = sequential_bootstrap(u, random_state=1)
        assert out.shape == (0,)


# ---------------------------------------------------------------------------
# get_return_attribution_weights
# ---------------------------------------------------------------------------

class TestReturnAttribution:
    def test_bigger_return_gets_bigger_weight(self):
        # Build a series where event A spans a 5% jump and event B spans a
        # 1% drift. Attribution weight on A must exceed B.
        idx = _make_index(40)
        prices = np.full(40, 100.0)
        # First 10 bars flat, next 10 jump by 5%, next 10 drift 1%, last flat.
        prices[10:20] = np.linspace(100.0, 105.0, 10)
        prices[20:30] = np.linspace(105.0, 106.05, 10)  # +1%
        close = pd.Series(prices, index=idx)
        returns = close.pct_change().fillna(0.0)

        starts = pd.DatetimeIndex([idx[10], idx[20]])
        ends = pd.DatetimeIndex([idx[19], idx[29]])
        w = get_return_attribution_weights(returns, starts, ends, idx)
        assert w.iloc[0] > w.iloc[1]
        assert np.isclose(w.sum(), 2.0)  # Normalized to len(events)=2

    def test_sum_equals_n_events(self):
        idx = _make_index(100)
        close = pd.Series(100.0 + np.arange(100) * 0.5, index=idx)
        returns = close.pct_change().fillna(0.0)
        starts, ends = _events(idx, [(0, 20), (30, 50), (60, 80)])
        w = get_return_attribution_weights(returns, starts, ends, idx)
        assert np.isclose(w.sum(), 3.0)

    def test_zero_returns_returns_uniform_weights(self):
        idx = _make_index(50)
        close = pd.Series(100.0, index=idx)  # flat — zero returns
        returns = close.pct_change().fillna(0.0)
        starts, ends = _events(idx, [(0, 10), (20, 30)])
        w = get_return_attribution_weights(returns, starts, ends, idx)
        # Degenerate fallback: all weights 1.0.
        assert np.allclose(w.to_numpy(), 1.0)


# ---------------------------------------------------------------------------
# get_time_decay_weights
# ---------------------------------------------------------------------------

class TestTimeDecay:
    def test_newest_heavier_linear(self):
        # Linear decay, oldest=0.1, newest=1.0 → weights monotonic in time.
        idx = _make_index(5)
        u = pd.Series(1.0, index=idx)
        w = get_time_decay_weights(
            u, oldest_weight=0.1, newest_weight=1.0, decay_type="linear",
        )
        arr = w.loc[idx].to_numpy()
        # Strictly increasing.
        assert np.all(np.diff(arr) > 0)
        # End-points bracket the configuration.
        assert w.iloc[0] < 1.0
        assert np.isclose(w.iloc[-1], 1.0)

    def test_no_decay_when_endpoints_equal(self):
        idx = _make_index(10)
        u = pd.Series(1.0, index=idx)
        w = get_time_decay_weights(
            u, oldest_weight=1.0, newest_weight=1.0, decay_type="linear",
        )
        assert np.allclose(w.to_numpy(), 1.0)

    def test_exponential_decay(self):
        idx = _make_index(5)
        u = pd.Series(1.0, index=idx)
        w = get_time_decay_weights(
            u, oldest_weight=0.1, newest_weight=1.0, decay_type="exponential",
        )
        arr = w.loc[idx].to_numpy()
        # Exponential interpolation in log-space.
        log_arr = np.log(arr)
        diffs = np.diff(log_arr)
        # Equal log-spacing → equal diffs.
        assert np.allclose(diffs, diffs[0])
        assert np.isclose(arr[0], 0.1)
        assert np.isclose(arr[-1], 1.0)

    def test_rejects_unknown_decay(self):
        u = pd.Series(1.0, index=_make_index(3))
        with pytest.raises(ValueError):
            get_time_decay_weights(u, decay_type="polynomial")

    def test_exponential_rejects_nonpositive(self):
        u = pd.Series(1.0, index=_make_index(3))
        with pytest.raises(ValueError):
            get_time_decay_weights(u, oldest_weight=0.0, decay_type="exponential")


# ---------------------------------------------------------------------------
# compute_sample_weights (integration)
# ---------------------------------------------------------------------------

class TestComputeSampleWeights:
    def _labels_df(self, close: pd.Series,
                   ranges: list[tuple[int, int]]) -> pd.DataFrame:
        idx = close.index
        starts = pd.DatetimeIndex([idx[s] for s, _ in ranges])
        exits = pd.DatetimeIndex([idx[e] for _, e in ranges])
        return pd.DataFrame(
            {
                "return": np.zeros(len(ranges)),
                "label": np.ones(len(ranges), dtype=int),
                "barrier_touched": ["upper"] * len(ranges),
                "exit_timestamp": exits,
                "holding_period": [e - s for s, e in ranges],
            },
            index=starts,
        )

    def test_output_sums_to_n_labels(self):
        close = pd.Series(
            100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, 200)),
            index=_make_index(200),
        )
        labels = self._labels_df(close, [(10, 40), (50, 80), (100, 150)])
        w = compute_sample_weights(labels, close, time_decay=0.5)
        assert np.isclose(w.sum(), len(labels))
        assert (w >= 0).all()

    def test_missing_exit_timestamp_raises(self):
        close = pd.Series(100.0, index=_make_index(10))
        bad = pd.DataFrame({"return": [0.01]}, index=[close.index[0]])
        with pytest.raises(ValueError):
            compute_sample_weights(bad, close)

    def test_empty_labels_returns_empty(self):
        close = pd.Series(100.0, index=_make_index(10))
        labels = pd.DataFrame(
            columns=["return", "exit_timestamp"],
            index=pd.DatetimeIndex([]),
        )
        # Ensure exit_timestamp dtype is datetime-like even when empty.
        labels["exit_timestamp"] = pd.to_datetime(labels["exit_timestamp"])
        w = compute_sample_weights(labels, close)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_1000_labels_under_5s(self):
        """End-to-end pipeline should handle 1000 events in < 5 seconds."""
        n_bars = 5000
        n_events = 1000
        rng = np.random.default_rng(0)
        close = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bars))),
            index=_make_index(n_bars),
        )
        # Random events with durations in [10, 60] bars.
        start_positions = np.sort(rng.integers(0, n_bars - 60, size=n_events))
        durations = rng.integers(10, 60, size=n_events)
        end_positions = np.minimum(start_positions + durations, n_bars - 1)

        starts = close.index[start_positions]
        ends = close.index[end_positions]

        t0 = time.perf_counter()
        concurrent = get_num_concurrent_events(close.index, starts, ends)
        uniqueness = get_average_uniqueness(starts, ends, close.index)
        returns = close.pct_change().fillna(0.0)
        ret_attr = get_return_attribution_weights(
            returns, starts, ends, close.index,
        )
        decay = get_time_decay_weights(uniqueness, oldest_weight=0.5)
        draws = sequential_bootstrap(
            uniqueness, starts, ends, close.index,
            n_samples=n_events, random_state=0,
        )
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"pipeline took {elapsed:.2f}s"
        assert len(concurrent) == n_bars
        assert len(uniqueness) == n_events
        assert len(ret_attr) == n_events
        assert len(decay) == n_events
        assert len(draws) == n_events
