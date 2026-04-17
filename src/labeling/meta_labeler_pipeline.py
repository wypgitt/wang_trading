"""
Meta-Labeling Pipeline (AFML §6.2 + design-doc §6)

Glues the Signal Battery, Labeling Engine, and Feature Factory into the
single transformation that feeds Tier 1 (LightGBM meta-labeler) training:

    (close, signals_df, features_df) → (X, y, sample_weights)

Flow:

    1. Drop neutral (side=0) signals.
    2. Compute daily volatility (EWM std of returns) once per close series.
    3. For each signal family:
         - look up the family's asymmetric barrier multipliers
         - run triple-barrier labeling on that family's events
    4. Concatenate the per-family label frames. Different families that fire
       at the same bar remain separate events, matching the design-doc rule
       that "each signal is evaluated independently".
    5. Compute sample weights: uniqueness × return_attribution × time decay.
    6. Align the feature matrix to event timestamps (backward fill — latest
       feature row at or before each event).
    7. Append signal metadata as model features: one-hot-encoded family,
       ``signal_side``, ``signal_confidence``.
    8. Collapse the label to a binary meta-target (1 = profitable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from src.labeling.sample_weights import (
    get_average_uniqueness,
    get_return_attribution_weights,
)
from src.labeling.triple_barrier import (
    get_daily_volatility,
    make_labels,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    from src.signal_battery.base_signal import Signal


# ---------------------------------------------------------------------------
# Per-family barrier presets
# ---------------------------------------------------------------------------

def configure_barrier_asymmetry(signal_family: str) -> tuple[float, float]:
    """
    Return ``(upper_mult, lower_mult)`` tuned for the signal family's profile.

    The asymmetry shapes the expected win/loss distribution of the label:
    mean-reversion bets favour many small wins (tight take-profit, wider
    stop), momentum bets favour rare large wins (wide take-profit, tight
    stop), carry lets positions run (very wide take-profit, tight stop).
    Lookup is substring-based and case-insensitive so it works for the
    actual family strings emitted by the Signal Battery (``ts_momentum``,
    ``mean_reversion``, ``stat_arb``, ``ma_crossover``, ``donchian``,
    ``futures_carry``, ``funding_arb``, ``cross_exchange_arb``, ...).

    Returns:
        (upper_mult, lower_mult)
    """
    fam = signal_family.lower()
    # Carry comes first: "funding_arb" also contains "arb" but we want the
    # carry profile, not the arb profile.
    if "carry" in fam or "funding" in fam:
        return (3.0, 1.0)
    if "reversion" in fam:
        return (1.0, 1.5)
    if (
        "momentum" in fam
        or "trend" in fam
        or "crossover" in fam
        or "donchian" in fam
        or "breakout" in fam
    ):
        return (2.5, 1.0)
    if "arb" in fam:
        return (1.5, 1.5)
    return (2.0, 2.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MetaLabelingPipeline:
    """
    End-to-end transformer from (close, signals, features) to (X, y, w).

    Attributes
    ----------
    feature_columns_ : list[str]
        Populated by :meth:`prepare_training_data`. The exact column order
        the trained model expects — use it to align live inputs.
    family_columns_ : list[str]
        Names of the one-hot family dummy columns that were emitted during
        training (e.g. ``signal_family_ts_momentum``). Live inference uses
        this list to produce a consistent one-hot vector.
    """

    def __init__(
        self,
        upper_barrier_mult: float = 2.0,
        lower_barrier_mult: float = 2.0,
        max_holding_period: int = 50,
        vol_span: int = 100,
        time_decay: float = 1.0,
    ) -> None:
        if upper_barrier_mult <= 0 or lower_barrier_mult <= 0:
            raise ValueError("barrier multipliers must be > 0")
        if max_holding_period < 1:
            raise ValueError("max_holding_period must be >= 1")
        if vol_span < 2:
            raise ValueError("vol_span must be >= 2")
        if time_decay < 0.0 or time_decay > 1.0:
            raise ValueError("time_decay must be in [0, 1]")

        self.upper_barrier_mult = upper_barrier_mult
        self.lower_barrier_mult = lower_barrier_mult
        self.max_holding_period = max_holding_period
        self.vol_span = vol_span
        self.time_decay = time_decay

        self.feature_columns_: list[str] = []
        self.family_columns_: list[str] = []

    # ------------------------------------------------------------------ API --

    def prepare_training_data(
        self,
        close: pd.Series,
        signals_df: pd.DataFrame,
        features_df: pd.DataFrame,
        *,
        db_manager: object | None = None,
        symbol: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Build ``(X, y, sample_weights)`` for training the meta-labeler.

        Args:
            close:        Close price series indexed by bar timestamp.
            signals_df:   Signal Battery output (columns: ``timestamp``,
                          ``symbol``, ``family``, ``side``, ``confidence``).
                          Neutral (side=0) signals are dropped.
            features_df:  Feature matrix indexed by bar timestamp (the output
                          of :class:`src.feature_factory.assembler.FeatureAssembler`).

        Returns:
            X, y, sample_weights — row count equal. Rows are one per signal
            event. The event timestamp is the Series/DataFrame index (may
            contain duplicates when multiple families fire on the same bar).
        """
        self._validate_inputs(close, signals_df, features_df)

        if len(signals_df) == 0:
            return self._empty_output(features_df)

        directional = signals_df.loc[signals_df["side"].astype(int) != 0].copy()
        if len(directional) == 0:
            return self._empty_output(features_df)
        directional["timestamp"] = pd.to_datetime(directional["timestamp"])

        vol = get_daily_volatility(close, span=self.vol_span)

        # -- per-family labeling ------------------------------------------
        per_family_frames: list[pd.DataFrame] = []
        for family, group in directional.groupby("family"):
            group = group.drop_duplicates(subset=["timestamp"])
            events = pd.DatetimeIndex(group["timestamp"])
            if len(events) == 0:
                continue
            sides = pd.Series(group["side"].astype(int).to_numpy(), index=events)

            upper_mult, lower_mult = configure_barrier_asymmetry(str(family))
            try:
                labels = make_labels(
                    close, events, sides, vol,
                    max_holding_period=self.max_holding_period,
                    upper_mult=upper_mult,
                    lower_mult=lower_mult,
                )
            except ValueError as exc:
                logger.warning(
                    f"MetaLabelingPipeline: family {family!r} labeling failed: {exc}"
                )
                continue
            if len(labels) == 0:
                continue

            labels = labels.reset_index()  # event_timestamp becomes a column
            # Preserve the ORIGINAL family string (configure_barrier_asymmetry
            # is lossy — two different families that share a barrier profile
            # must remain distinguishable as features).
            labels["family"] = family
            conf_lookup = group.set_index("timestamp")["confidence"].astype(float)
            side_lookup = group.set_index("timestamp")["side"].astype(int)
            ts = labels["event_timestamp"]
            labels["confidence"] = conf_lookup.reindex(ts).to_numpy()
            labels["side"] = side_lookup.reindex(ts).to_numpy()
            per_family_frames.append(labels)

        if not per_family_frames:
            return self._empty_output(features_df)

        combined = pd.concat(per_family_frames, ignore_index=True)

        # -- sample weights -----------------------------------------------
        event_starts = pd.DatetimeIndex(combined["event_timestamp"])
        event_ends = pd.DatetimeIndex(combined["exit_timestamp"])
        close_index = pd.DatetimeIndex(close.index)

        uniqueness = get_average_uniqueness(
            event_starts, event_ends, close_index,
        ).to_numpy()
        returns = close.pct_change().fillna(0.0)
        ret_attr = get_return_attribution_weights(
            returns, event_starts, event_ends, close_index,
        ).to_numpy()
        decay = self._rank_time_decay(combined["event_timestamp"].to_numpy())

        raw_weights = uniqueness * ret_attr * decay

        # -- feature alignment --------------------------------------------
        feat_index_arr = features_df.index.to_numpy()
        event_ts_arr = combined["event_timestamp"].to_numpy()
        # Backward fill: take the most recent feature row at or before each event.
        feat_positions = np.searchsorted(feat_index_arr, event_ts_arr, side="right") - 1
        valid = (feat_positions >= 0) & (feat_positions < len(feat_index_arr))
        if not valid.any():
            logger.warning(
                "MetaLabelingPipeline: no events overlap features_df range; "
                "returning empty output"
            )
            return self._empty_output(features_df)

        combined = combined.loc[valid].reset_index(drop=True)
        feat_positions = feat_positions[valid]
        raw_weights = raw_weights[valid]

        feature_rows = features_df.iloc[feat_positions].reset_index(drop=True)

        # -- signal-metadata features -------------------------------------
        family_dummies = pd.get_dummies(
            combined["family"], prefix="signal_family", dtype=float,
        )
        self.family_columns_ = list(family_dummies.columns)

        signal_side = combined["side"].astype(int).rename("signal_side")
        signal_confidence = combined["confidence"].astype(float).rename(
            "signal_confidence"
        )

        X = pd.concat(
            [feature_rows, family_dummies,
             signal_side.to_frame(), signal_confidence.to_frame()],
            axis=1,
        )
        X.index = pd.DatetimeIndex(
            combined["event_timestamp"], name="event_timestamp",
        )
        self.feature_columns_ = list(X.columns)

        # Binary meta-label (1 = profitable).
        y = (combined["label"] > 0).astype(int)
        y.index = X.index
        y.name = "meta_label"

        # Re-normalize weights after the valid-event filter.
        total = raw_weights.sum()
        if total > 0:
            weights = raw_weights * (len(raw_weights) / total)
        else:
            weights = np.ones_like(raw_weights)
        sample_weights = pd.Series(weights, index=X.index, name="sample_weight")

        if db_manager is not None and symbol is not None:
            self._persist_labels(combined, weights, db_manager, symbol)

        return X, y, sample_weights

    @staticmethod
    def _persist_labels(
        combined: pd.DataFrame, weights: np.ndarray,
        db_manager: object, symbol: str,
    ) -> None:
        """Best-effort async insert of the labels frame into TimescaleDB."""
        import asyncio as _asyncio
        labels_df = combined.copy()
        labels_df["symbol"] = symbol
        labels_df["sample_weight"] = weights
        for col in ("volatility", "upper_barrier", "lower_barrier",
                    "barrier_touched", "return_pct", "holding_period_bars",
                    "vertical_barrier"):
            if col not in labels_df.columns:
                labels_df[col] = None
        try:
            coro = db_manager.insert_labels(labels_df)  # type: ignore[attr-defined]
            loop = _asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning(f"persist_labels failed: {exc}")

    def prepare_live_features(
        self,
        features_at_event: pd.Series,
        signal: "Signal",
    ) -> pd.DataFrame:
        """
        Build a 1-row feature frame for live inference.

        Uses ``family_columns_`` / ``feature_columns_`` captured during the
        most recent :meth:`prepare_training_data` call so the output column
        order matches what the trained model expects. If the pipeline has
        not been fitted, the returned frame still contains the signal-
        metadata columns but no family dummies beyond the current signal's.

        Args:
            features_at_event: 1-D Series of feature values (e.g. a single
                               row from ``features_df``).
            signal:            A :class:`Signal` from the Signal Battery.

        Returns:
            DataFrame with exactly one row, columns aligned with the
            training feature layout when one is available.
        """
        row = pd.Series(features_at_event, dtype=float).copy()
        # Signal metadata.
        row["signal_side"] = int(signal.side)
        row["signal_confidence"] = float(signal.confidence)

        # Family one-hot: respect the vocabulary seen during training.
        if self.family_columns_:
            for col in self.family_columns_:
                row[col] = 0.0
            current_col = f"signal_family_{signal.family}"
            if current_col in self.family_columns_:
                row[current_col] = 1.0
            else:
                logger.warning(
                    f"prepare_live_features: family {signal.family!r} was not "
                    "seen during training; one-hot vector will be all zeros"
                )
        else:
            row[f"signal_family_{signal.family}"] = 1.0

        frame = row.to_frame().T
        frame.index = pd.DatetimeIndex([signal.timestamp], name="event_timestamp")
        if self.feature_columns_:
            # Ensure column order matches training; missing columns default to 0.
            for col in self.feature_columns_:
                if col not in frame.columns:
                    frame[col] = 0.0
            frame = frame[self.feature_columns_]
        return frame.astype(float)

    # ------------------------------------------------------------------ helpers --

    @staticmethod
    def _validate_inputs(
        close: pd.Series,
        signals_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> None:
        if not isinstance(close, pd.Series):
            raise ValueError("close must be a pandas Series")
        if not isinstance(signals_df, pd.DataFrame):
            raise ValueError("signals_df must be a pandas DataFrame")
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("features_df must be a pandas DataFrame")
        required_signal_cols = {"timestamp", "family", "side", "confidence"}
        missing = required_signal_cols - set(signals_df.columns)
        if missing:
            raise ValueError(
                f"signals_df missing columns {sorted(missing)}"
            )

    def _empty_output(
        self, features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        empty_idx = pd.DatetimeIndex([], name="event_timestamp")
        feature_cols = list(features_df.columns) + ["signal_side", "signal_confidence"]
        X = pd.DataFrame(columns=feature_cols, index=empty_idx)
        y = pd.Series([], dtype=int, index=empty_idx, name="meta_label")
        w = pd.Series([], dtype=float, index=empty_idx, name="sample_weight")
        return X, y, w

    def _rank_time_decay(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Rank-based linear decay in [time_decay, 1.0].

        Stable against duplicate timestamps (important here: two families
        that fire on the same bar produce two rows with identical event
        timestamps; :func:`get_time_decay_weights` can't disambiguate them
        via the index but positional ranks can).
        """
        n = len(timestamps)
        if n == 0:
            return np.empty(0, dtype=float)
        if n == 1:
            return np.array([1.0])
        order = np.argsort(timestamps, kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n) / (n - 1)
        return self.time_decay + (1.0 - self.time_decay) * ranks
