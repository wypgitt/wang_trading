"""Tests for triple-barrier labeling (AFML Ch. 3)."""

import numpy as np
import pandas as pd
import pytest

from src.labeling.triple_barrier import (
    apply_triple_barrier,
    get_daily_volatility,
    get_meta_labels,
    get_vertical_barriers,
    make_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(n: int, freq: str = "1min") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq=freq)


def _linear_prices(n: int, slope: float, start: float = 100.0) -> pd.Series:
    return pd.Series(start + slope * np.arange(n), index=_make_index(n))


def _gbm_prices(n: int, mu: float, sigma: float, seed: int,
                start: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    increments = rng.normal(mu, sigma, size=n)
    return pd.Series(start * np.exp(np.cumsum(increments)), index=_make_index(n))


# ---------------------------------------------------------------------------
# get_daily_volatility
# ---------------------------------------------------------------------------

class TestGetDailyVolatility:
    def test_shape_and_index(self):
        close = _gbm_prices(200, 0.0, 0.01, seed=0)
        vol = get_daily_volatility(close, span=50)
        assert len(vol) == len(close)
        assert vol.index.equals(close.index)
        assert vol.name == "daily_vol"

    def test_first_value_is_nan(self):
        close = _linear_prices(50, slope=0.1)
        vol = get_daily_volatility(close)
        assert np.isnan(vol.iloc[0])

    def test_rises_in_high_vol_regime(self):
        calm = _gbm_prices(300, 0.0, 0.001, seed=1)
        noisy = _gbm_prices(300, 0.0, 0.02, seed=2) * calm.iloc[-1] / 100.0
        concat = pd.Series(
            np.concatenate([calm.to_numpy(), noisy.to_numpy()]),
            index=_make_index(600),
        )
        vol = get_daily_volatility(concat, span=50)
        # Allow EWM to warm up into the noisy regime.
        assert vol.iloc[500:].mean() > vol.iloc[100:300].mean()

    def test_rejects_bad_span(self):
        with pytest.raises(ValueError):
            get_daily_volatility(_linear_prices(20, 0.1), span=1)

    def test_rejects_non_series(self):
        with pytest.raises(ValueError):
            get_daily_volatility(np.arange(10.0))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_vertical_barriers
# ---------------------------------------------------------------------------

class TestGetVerticalBarriers:
    def test_expiry_is_n_bars_ahead(self):
        close = _linear_prices(100, slope=0.0)
        events = close.index[[10, 25, 50]]
        out = get_vertical_barriers(events, close, max_holding_period=15)
        assert out.loc[events[0]] == close.index[25]
        assert out.loc[events[1]] == close.index[40]
        assert out.loc[events[2]] == close.index[65]

    def test_clamps_to_last_timestamp(self):
        close = _linear_prices(100, slope=0.0)
        events = close.index[[95, 99]]
        out = get_vertical_barriers(events, close, max_holding_period=20)
        assert out.loc[events[0]] == close.index[-1]
        assert out.loc[events[1]] == close.index[-1]

    def test_rejects_bad_max_holding(self):
        close = _linear_prices(10, slope=0.0)
        with pytest.raises(ValueError):
            get_vertical_barriers(close.index, close, max_holding_period=0)

    def test_rejects_empty_close(self):
        close = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError):
            get_vertical_barriers(
                pd.DatetimeIndex([]), close, max_holding_period=5,
            )


# ---------------------------------------------------------------------------
# apply_triple_barrier — directional cases
# ---------------------------------------------------------------------------

class TestApplyTripleBarrierLong:
    """side = +1 (long)."""

    def _events(self, close: pd.Series, vol: float,
                horizon: int = 50) -> pd.DataFrame:
        ev_time = close.index[0]
        return pd.DataFrame(
            {
                "side": [1],
                "volatility": [vol],
                "vertical_barrier": [close.index[horizon]],
            },
            index=pd.DatetimeIndex([ev_time]),
        )

    def test_upper_hit_on_uptrend(self):
        # Strong, monotonic uptrend guaranteed to hit the upper barrier first.
        close = _linear_prices(100, slope=1.0, start=100.0)
        events = self._events(close, vol=0.05, horizon=50)
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=2.0, lower_multiplier=2.0)
        row = out.iloc[0]
        assert row["label"] == 1
        assert row["barrier_touched"] == "upper"
        assert row["return"] > 0
        assert row["holding_period"] > 0

    def test_lower_hit_on_downtrend(self):
        close = _linear_prices(100, slope=-1.0, start=100.0)
        events = self._events(close, vol=0.05, horizon=50)
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=2.0, lower_multiplier=2.0)
        row = out.iloc[0]
        assert row["label"] == -1
        assert row["barrier_touched"] == "lower"
        # Return is side-adjusted: long in falling market → negative.
        assert row["return"] < 0

    def test_vertical_when_flat(self):
        # Flat price → neither horizontal barrier is ever touched.
        close = pd.Series(100.0, index=_make_index(100))
        events = self._events(close, vol=0.01, horizon=20)
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=2.0, lower_multiplier=2.0)
        row = out.iloc[0]
        assert row["barrier_touched"] == "vertical"
        # Flat return → label falls back to 0 (neither profitable nor not).
        assert row["label"] == 0
        assert row["holding_period"] == 20

    def test_vertical_label_sign_matches_return(self):
        # Drift small enough that horizontals are not hit, but price still
        # ends above entry — label should be +1 (profitable long).
        close = _linear_prices(50, slope=0.02, start=100.0)  # 1.0% total drift
        events = self._events(close, vol=0.05, horizon=40)
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=5.0, lower_multiplier=5.0)
        row = out.iloc[0]
        assert row["barrier_touched"] == "vertical"
        assert row["label"] == 1
        assert row["return"] > 0


class TestApplyTripleBarrierShort:
    """side = -1 (short): profit is downward."""

    def _events(self, close: pd.Series, vol: float,
                horizon: int = 50) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "side": [-1],
                "volatility": [vol],
                "vertical_barrier": [close.index[horizon]],
            },
            index=pd.DatetimeIndex([close.index[0]]),
        )

    def test_price_drop_yields_plus_one(self):
        close = _linear_prices(60, slope=-1.0, start=100.0)
        events = self._events(close, vol=0.05, horizon=50)
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=2.0, lower_multiplier=2.0)
        row = out.iloc[0]
        assert row["label"] == 1  # short profits from a drop
        assert row["return"] > 0

    def test_price_rise_yields_minus_one(self):
        close = _linear_prices(60, slope=+1.0, start=100.0)
        events = self._events(close, vol=0.05, horizon=50)
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=2.0, lower_multiplier=2.0)
        row = out.iloc[0]
        assert row["label"] == -1  # short stopped out on a rally
        assert row["return"] < 0


# ---------------------------------------------------------------------------
# Asymmetric multipliers
# ---------------------------------------------------------------------------

class TestAsymmetricBarriers:
    def test_wide_upper_tight_lower(self):
        """
        upper_mult=3, lower_mult=1, vol=0.05 → physical upper = entry*1.15,
        physical lower = entry*0.95. A gentle downtrend should trip the
        tight stop long before the wide take-profit could possibly fire.
        """
        close = _linear_prices(60, slope=-0.1, start=100.0)  # 95.1 at t=49
        events = pd.DataFrame(
            {
                "side": [1],
                "volatility": [0.05],
                "vertical_barrier": [close.index[50]],
            },
            index=pd.DatetimeIndex([close.index[0]]),
        )
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=3.0, lower_multiplier=1.0)
        row = out.iloc[0]
        assert row["barrier_touched"] == "lower"
        assert row["label"] == -1

    def test_tight_upper_wide_lower(self):
        # Mirror: gentle uptrend trips the tight upper (mult=1) before the
        # wide lower (mult=3) ever could.
        close = _linear_prices(60, slope=+0.1, start=100.0)
        events = pd.DataFrame(
            {
                "side": [1],
                "volatility": [0.05],
                "vertical_barrier": [close.index[50]],
            },
            index=pd.DatetimeIndex([close.index[0]]),
        )
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=1.0, lower_multiplier=3.0)
        row = out.iloc[0]
        assert row["barrier_touched"] == "upper"
        assert row["label"] == 1


# ---------------------------------------------------------------------------
# Holding period & edge-of-series behaviour
# ---------------------------------------------------------------------------

class TestHoldingPeriod:
    def test_matches_bars_between_event_and_exit(self):
        # Spike at bar 10; upper barrier is hit there and nowhere else.
        n = 50
        prices = np.full(n, 100.0)
        prices[10] = 120.0  # 20% jump — well past any reasonable upper barrier
        close = pd.Series(prices, index=_make_index(n))
        events = pd.DataFrame(
            {
                "side": [1],
                "volatility": [0.05],
                "vertical_barrier": [close.index[30]],
            },
            index=pd.DatetimeIndex([close.index[0]]),
        )
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=2.0, lower_multiplier=2.0)
        row = out.iloc[0]
        assert row["barrier_touched"] == "upper"
        assert row["holding_period"] == 10
        assert row["exit_timestamp"] == close.index[10]

    def test_vertical_edge_of_series(self):
        # Event near the end of the series → vertical_barrier clamps to last
        # bar and the label still computes without crashing.
        close = _linear_prices(50, slope=0.0, start=100.0)
        last_idx = close.index[-1]
        vertical = get_vertical_barriers(
            pd.DatetimeIndex([close.index[48]]), close, max_holding_period=20,
        )
        events = pd.DataFrame(
            {
                "side": [1],
                "volatility": [0.01],
                "vertical_barrier": [vertical.iloc[0]],
            },
            index=pd.DatetimeIndex([close.index[48]]),
        )
        out = apply_triple_barrier(close, events,
                                   upper_multiplier=2.0, lower_multiplier=2.0)
        row = out.iloc[0]
        assert row["exit_timestamp"] == last_idx
        assert row["barrier_touched"] == "vertical"
        assert row["holding_period"] == 1  # only one bar from 48 to 49


# ---------------------------------------------------------------------------
# make_labels + get_meta_labels on realistic price path
# ---------------------------------------------------------------------------

class TestMakeLabelsGBM:
    def test_produces_valid_labels_on_gbm(self):
        # GBM with positive drift: we should get a mix of +1, -1, and some
        # vertical exits. All labels should be in {-1, 0, 1}.
        close = _gbm_prices(500, mu=0.0005, sigma=0.01, seed=7)
        vol = get_daily_volatility(close, span=50).bfill()
        events = close.index[50:450:20]  # 20 sparse events, well inside bounds
        sides = pd.Series(1, index=events)
        out = make_labels(close, events, sides, vol,
                          max_holding_period=40,
                          upper_mult=2.0, lower_mult=2.0)
        assert len(out) == len(events)
        assert set(out["label"].unique()).issubset({-1, 0, 1})
        assert set(out["barrier_touched"].unique()).issubset(
            {"upper", "lower", "vertical"}
        )
        # Returns and holding periods should be self-consistent.
        assert (out["holding_period"] >= 0).all()
        assert (out["holding_period"] <= 40).all()

    def test_meta_labels_are_binary(self):
        close = _gbm_prices(500, mu=0.0005, sigma=0.01, seed=11)
        vol = get_daily_volatility(close, span=50).bfill()
        events = close.index[50:450:20]
        sides = pd.Series(
            np.where(np.arange(len(events)) % 2 == 0, 1, -1),
            index=events,
        )
        meta = get_meta_labels(close, events, sides, vol,
                               max_holding_period=40)
        assert meta.name == "meta_label"
        assert set(meta.unique()).issubset({0, 1})
        assert len(meta) == len(events)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_apply_rejects_missing_columns(self):
        close = _linear_prices(10, 0.1)
        bad = pd.DataFrame({"side": [1]}, index=[close.index[0]])
        with pytest.raises(ValueError):
            apply_triple_barrier(close, bad)

    def test_apply_rejects_bad_multipliers(self):
        close = _linear_prices(10, 0.1)
        events = pd.DataFrame(
            {"side": [1], "volatility": [0.01],
             "vertical_barrier": [close.index[5]]},
            index=pd.DatetimeIndex([close.index[0]]),
        )
        with pytest.raises(ValueError):
            apply_triple_barrier(close, events, upper_multiplier=0.0)

    def test_make_labels_rejects_unaligned_sides(self):
        close = _linear_prices(60, 0.0)
        vol = get_daily_volatility(close).bfill()
        events = close.index[[10, 20, 30]]
        # sides missing an entry for events[1]
        sides = pd.Series([1, 1], index=close.index[[10, 30]])
        with pytest.raises(ValueError):
            make_labels(close, events, sides, vol, max_holding_period=10)

    def test_apply_skips_invalid_events(self):
        # Side = 0 and vol <= 0 events are skipped without crashing.
        close = _linear_prices(30, slope=0.1)
        events = pd.DataFrame(
            {
                "side": [0, 1, 1],
                "volatility": [0.01, -0.01, 0.01],
                "vertical_barrier": [close.index[10]] * 3,
            },
            index=pd.DatetimeIndex(close.index[[0, 1, 2]]),
        )
        out = apply_triple_barrier(close, events)
        # Only the third event is valid.
        assert len(out) == 1
        assert out.index[0] == close.index[2]
