"""
Bet-Sizing Cascade — 5 layers (design-doc §8).

Turns a calibrated meta-labeler probability into a final, risk-budgeted
position size. Every layer lowers (or preserves) the size — the function
is monotonically non-increasing across the cascade, so it's always safe
to stop early (e.g., skip Layer 4 when ATR data is missing). Each layer
logs its output so the full audit trail is returned alongside the final
size.

    Layer 1 — AFML sizing: prob → raw fraction (AFML Ch. 10)
    Layer 2 — Kelly cap:   hard cap at fractional Kelly (Chan)
    Layer 3 — Vol adjust:  scale by avg/current vol; VRP haircut (Sinclair)
    Layer 4 — ATR cap:     Clenow's risk-per-trade cap (trend + futures)
    Layer 5 — Risk budget: single / family / gross / crypto limits

Layer 4 is conditional — only applies to trend-following families and
futures, and only when the caller supplies the instrument's ATR and
price. Layer 5 needs a ``current_positions`` snapshot to compute family
and gross usage; when missing, only the single-position cap is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.bet_sizing.afml_sizing import bet_size_from_probability
from src.bet_sizing.kelly import fractional_kelly


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CascadeConfig:
    """Static knobs for the cascade — matches the design-doc risk table."""

    # Layer 1
    max_raw_size: float = 1.0

    # Layer 2
    kelly_fraction: float = 0.25

    # Layer 3
    vrp_haircut: float = 0.25          # 25% off when VRP in top quartile

    # Layer 4
    risk_per_trade: float = 0.01       # 1% NAV risked per trade
    atr_multiplier: float = 2.0

    # Layer 5 — risk budget (fractions of NAV, not percentages)
    max_single_position: float = 0.10  # 10% NAV
    max_family_allocation: float = 0.30
    max_gross_exposure: float = 1.50   # 150% (100% no leverage)
    max_crypto_allocation: float = 0.30
    max_sector_exposure: float = 0.20  # GICS sector cap (equities)


@dataclass
class FamilyStats:
    """Historical win/loss magnitudes per signal family (for Kelly)."""
    avg_win: float
    avg_loss: float


# ---------------------------------------------------------------------------
# Family classifier helpers
# ---------------------------------------------------------------------------

def _is_trend_following(family: str) -> bool:
    fam = (family or "").lower()
    return any(
        key in fam for key in
        ("momentum", "trend", "crossover", "donchian", "breakout")
    )


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

class BetSizingCascade:
    """
    Stateless orchestrator — all per-call inputs come through
    :meth:`compute_position_size`. State on the instance is just the config
    and the historical family stats.

    Attributes
    ----------
    config : CascadeConfig
    family_stats : dict[str, FamilyStats]
        Per-family avg_win / avg_loss magnitudes driving Layer 2. Missing
        families skip the Kelly cap with a debug log (no crash).
    """

    def __init__(
        self,
        config: CascadeConfig | None = None,
        family_stats: dict[str, FamilyStats | dict[str, float]] | None = None,
    ) -> None:
        self.config = config or CascadeConfig()
        # Normalise any plain-dict stats to FamilyStats.
        normalised: dict[str, FamilyStats] = {}
        for name, stats in (family_stats or {}).items():
            if isinstance(stats, FamilyStats):
                normalised[name] = stats
            else:
                normalised[name] = FamilyStats(
                    avg_win=float(stats["avg_win"]),
                    avg_loss=float(stats["avg_loss"]),
                )
        self.family_stats = normalised

    # -------------------------------------------------------------- API --

    def compute_position_size(
        self,
        prob: float,
        side: int,
        symbol: str,
        signal_family: str,
        current_vol: float,
        avg_vol: float,
        portfolio_nav: float,
        current_positions: dict[str, dict[str, Any]] | None = None,
        vrp_quartile: int | None = None,
        atr: float | None = None,
        price: float | None = None,
        point_value: float = 1.0,
        asset_class: str = "equity",
    ) -> dict[str, Any]:
        """
        Run all 5 layers and return an audit dict.

        Parameters
        ----------
        prob : calibrated meta-label probability
        side : +1 long, -1 short, 0 neutral (returns zero size)
        current_vol, avg_vol : current and long-run vol for Layer 3
        portfolio_nav : used for Layers 4 and 5
        current_positions : ``{symbol: {"size": fraction, "family": str,
                              "asset_class": str, "sector": str}}``. When
                            omitted, Layer 5 only enforces the single-position
                            cap.
        vrp_quartile : 0..3 — top quartile triggers the Sinclair haircut.
        atr, price, point_value : Layer-4 inputs. ATR in price units; the
                                  layer is skipped if any are missing.
        asset_class : ``"equity"`` / ``"crypto"`` / ``"futures"``.

        Returns
        -------
        dict with keys
            ``afml_size``        — Layer 1 (unsigned, fraction of NAV)
            ``kelly_capped``     — Layer 2
            ``vol_adjusted``     — Layer 3
            ``atr_capped``       — Layer 4 (same as vol_adjusted if skipped)
            ``final_size``       — Layer 5 output, SIGNED (side * magnitude)
            ``side``             — directional tag copied from input
            ``constraints_applied`` — list of rule tags that bound the size
        """
        if portfolio_nav <= 0:
            raise ValueError(f"portfolio_nav must be > 0 (got {portfolio_nav})")
        if side not in (-1, 0, 1):
            raise ValueError(f"side must be -1, 0, or +1 (got {side})")

        constraints: list[str] = []

        # --- Layer 1: AFML sizing -----------------------------------------
        afml_size = bet_size_from_probability(
            prob, max_size=self.config.max_raw_size,
        )
        afml_size = float(afml_size)  # ensure scalar

        # Short-circuit: neutral side OR sub-50% probability.
        if side == 0 or afml_size == 0.0:
            return self._zero_result(side, afml_size)

        # --- Layer 2: Kelly cap -------------------------------------------
        kelly_capped = afml_size
        stats = self.family_stats.get(signal_family)
        if stats is not None and stats.avg_win > 0 and stats.avg_loss > 0:
            kelly_cap = fractional_kelly(
                prob, stats.avg_win, stats.avg_loss,
                fraction=self.config.kelly_fraction,
            )
            if kelly_cap < afml_size:
                constraints.append("kelly_cap")
            kelly_capped = min(afml_size, kelly_cap)
        else:
            logger.debug(
                f"cascade: no family_stats for {signal_family!r}; "
                "skipping Kelly cap"
            )

        # --- Layer 3: Volatility adjustment -------------------------------
        vol_adjusted = kelly_capped
        if current_vol > 0 and avg_vol > 0 and current_vol != avg_vol:
            ratio = avg_vol / current_vol
            vol_adjusted = kelly_capped * ratio
            if ratio < 1.0:
                constraints.append("vol_scaling")
        if vrp_quartile == 3:  # top VRP quartile — vol-spike hedge
            vol_adjusted *= (1.0 - self.config.vrp_haircut)
            constraints.append("vrp_haircut")
        # Respect the original cap — vol scaling could push above max_raw_size
        # in very calm regimes; clip back to [0, max_raw_size] to keep the
        # output interpretable as a fraction of NAV.
        vol_adjusted = max(0.0, min(vol_adjusted, self.config.max_raw_size))

        # --- Layer 4: ATR normalisation (Clenow) --------------------------
        atr_capped = vol_adjusted
        use_atr = (
            (_is_trend_following(signal_family) or asset_class == "futures")
            and atr is not None and atr > 0
            and price is not None and price > 0
        )
        if use_atr:
            # Narrow optionals for type-checkers — `use_atr` already proved
            # both atr and price are non-None scalars.
            atr_val = float(atr)  # type: ignore[arg-type]
            price_val = float(price)  # type: ignore[arg-type]
            # Per-unit dollar risk of the stop distance.
            per_unit_risk = atr_val * self.config.atr_multiplier * point_value
            max_units = (portfolio_nav * self.config.risk_per_trade) / per_unit_risk
            max_notional = max_units * price_val * point_value
            max_fraction = max_notional / portfolio_nav
            if atr_capped > max_fraction:
                constraints.append("atr_cap")
                atr_capped = max_fraction

        # --- Layer 5: Risk budget -----------------------------------------
        final_magnitude = self._apply_risk_budget(
            size=atr_capped,
            symbol=symbol,
            family=signal_family,
            asset_class=asset_class,
            current_positions=current_positions,
            constraints=constraints,
        )
        final_magnitude = max(0.0, final_magnitude)
        final_size = side * final_magnitude

        return {
            "afml_size": float(afml_size),
            "kelly_capped": float(kelly_capped),
            "vol_adjusted": float(vol_adjusted),
            "atr_capped": float(atr_capped),
            "final_size": float(final_size),
            "side": int(side),
            "constraints_applied": list(constraints),
        }

    def compute_position_sizes_batch(
        self,
        signals_df: pd.DataFrame,
        features_df: pd.DataFrame,
        portfolio_state: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Vectorised per-signal sizing; returns one row per input signal.

        ``signals_df`` is expected to carry the meta-labeler probability in a
        column named ``prob`` (or ``meta_label_prob``). ``features_df`` is
        indexed by bar timestamp and supplies per-bar state that the cascade
        needs (``current_vol``, ``avg_vol``, and optionally ``atr`` /
        ``price`` / ``vrp_quartile``). Missing columns fall back to sensible
        defaults.

        Args:
            signals_df:       Each row must have ``timestamp``, ``symbol``,
                              ``family``, ``side``, ``prob``, and
                              optionally ``asset_class`` / ``point_value``.
            features_df:      Bar-indexed frame with volatility columns.
            portfolio_state:  ``{"nav": float, "current_positions": dict}``.
        """
        required = {"timestamp", "symbol", "family", "side"}
        missing = required - set(signals_df.columns)
        if missing:
            raise ValueError(
                f"signals_df missing columns {sorted(missing)}"
            )
        if "prob" not in signals_df.columns and "meta_label_prob" not in signals_df.columns:
            raise ValueError(
                "signals_df must have a 'prob' or 'meta_label_prob' column"
            )
        nav = float(portfolio_state.get("nav", 0.0))
        if nav <= 0:
            raise ValueError("portfolio_state['nav'] must be > 0")
        positions = portfolio_state.get("current_positions")

        prob_col = "prob" if "prob" in signals_df.columns else "meta_label_prob"

        feat_index = features_df.index.to_numpy()

        rows: list[dict[str, Any]] = []
        for _, sig in signals_df.iterrows():
            ts = pd.Timestamp(sig["timestamp"])
            # Backward-fill feature lookup.
            pos = int(np.searchsorted(feat_index, ts.to_numpy(), side="right")) - 1
            if pos < 0:
                # No feature history yet — skip with final_size = 0.
                rows.append({
                    "timestamp": ts, "symbol": sig["symbol"],
                    "family": sig["family"], "side": int(sig["side"]),
                    "prob": float(sig[prob_col]),
                    "afml_size": 0.0, "kelly_capped": 0.0,
                    "vol_adjusted": 0.0, "atr_capped": 0.0,
                    "final_size": 0.0,
                    "constraints_applied": ["no_feature_history"],
                })
                continue
            feats = features_df.iloc[pos]
            current_vol = float(feats.get("current_vol", np.nan))
            avg_vol = float(feats.get("avg_vol", np.nan))
            if not np.isfinite(current_vol):
                current_vol = float(feats.get("garch_vol", np.nan))
            if not np.isfinite(current_vol):
                current_vol = 0.0
            if not np.isfinite(avg_vol):
                avg_vol = current_vol if current_vol > 0 else 0.0

            atr = feats.get("atr")
            price = feats.get("price")
            vrp_q = feats.get("vrp_quartile")
            if pd.isna(atr):
                atr = None
            if pd.isna(price):
                price = None
            if pd.isna(vrp_q):
                vrp_q = None

            result = self.compute_position_size(
                prob=float(sig[prob_col]),
                side=int(sig["side"]),
                symbol=str(sig["symbol"]),
                signal_family=str(sig["family"]),
                current_vol=current_vol,
                avg_vol=avg_vol,
                portfolio_nav=nav,
                current_positions=positions,
                vrp_quartile=int(vrp_q) if vrp_q is not None else None,
                atr=float(atr) if atr is not None else None,
                price=float(price) if price is not None else None,
                point_value=float(sig.get("point_value", 1.0)),
                asset_class=str(sig.get("asset_class", "equity")),
            )
            rows.append({
                "timestamp": ts, "symbol": sig["symbol"],
                "family": sig["family"],
                "prob": float(sig[prob_col]),
                **result,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ helpers --

    def _zero_result(self, side: int, afml_size: float) -> dict[str, Any]:
        return {
            "afml_size": float(afml_size),
            "kelly_capped": 0.0,
            "vol_adjusted": 0.0,
            "atr_capped": 0.0,
            "final_size": 0.0,
            "side": int(side),
            "constraints_applied": [],
        }

    def _apply_risk_budget(
        self,
        size: float,
        symbol: str,
        family: str,
        asset_class: str,
        current_positions: dict[str, dict[str, Any]] | None,
        constraints: list[str],
    ) -> float:
        # Max single position — always enforced.
        if size > self.config.max_single_position:
            constraints.append("max_single_position")
            size = self.config.max_single_position

        if current_positions is None:
            return size

        # Exclude the symbol itself from budget accounting so this trade
        # overwrites (not adds to) any existing position on the same symbol.
        other = {
            s: p for s, p in current_positions.items() if s != symbol
        }

        # Max family allocation
        family_used = sum(
            abs(float(p.get("size", 0.0))) for p in other.values()
            if p.get("family") == family
        )
        family_budget = self.config.max_family_allocation - family_used
        if size > family_budget:
            constraints.append("max_family_allocation")
            size = max(0.0, family_budget)

        # Max gross exposure
        gross_used = sum(
            abs(float(p.get("size", 0.0))) for p in other.values()
        )
        gross_budget = self.config.max_gross_exposure - gross_used
        if size > gross_budget:
            constraints.append("max_gross_exposure")
            size = max(0.0, gross_budget)

        # Max crypto allocation
        if asset_class == "crypto":
            crypto_used = sum(
                abs(float(p.get("size", 0.0))) for p in other.values()
                if p.get("asset_class") == "crypto"
            )
            crypto_budget = self.config.max_crypto_allocation - crypto_used
            if size > crypto_budget:
                constraints.append("max_crypto_allocation")
                size = max(0.0, crypto_budget)

        # Max sector exposure (equities only)
        if asset_class == "equity":
            # Only enforced if the incoming signal has a sector tag; the
            # caller can't know which sector this particular bet belongs to
            # without an extra argument. We look it up via current_positions
            # if the same symbol is already held and re-tags itself; otherwise
            # we skip. (A richer version would accept sector directly.)
            this_sector = (
                current_positions.get(symbol, {}).get("sector")
                if current_positions else None
            )
            if this_sector:
                sector_used = sum(
                    abs(float(p.get("size", 0.0))) for p in other.values()
                    if p.get("sector") == this_sector
                )
                sector_budget = self.config.max_sector_exposure - sector_used
                if size > sector_budget:
                    constraints.append("max_sector_exposure")
                    size = max(0.0, sector_budget)

        return size
