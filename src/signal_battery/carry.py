"""
Carry / Funding Rate Signals (Clenow + crypto industry practice)

Two strategies:
    - FuturesCarrySignal      : roll yield from front/back contract spread
    - FundingRateArbSignal    : crypto perp funding rate arbitrage
                                (delta-neutral: long spot + short perp)

Plus ``annualize_funding_rate`` helper.

Design doc §4.5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.signal_battery.base_signal import BaseSignalGenerator, Signal


# ---------------------------------------------------------------------------
# Funding-rate helpers
# ---------------------------------------------------------------------------

def annualize_funding_rate(rate: float, payments_per_day: int = 3) -> float:
    """
    Convert a per-payment funding rate into an annualized compound rate.

        (1 + rate)^(payments_per_day * 365) - 1

    Binance / Bybit / most CEX perps pay funding every 8 hours (3×/day),
    which is the default.

    Args:
        rate:             Funding rate per payment (e.g. 0.0001 = 1 bp).
        payments_per_day: Funding cycles per day (3 for 8h cadence).
    """
    if payments_per_day <= 0:
        raise ValueError("payments_per_day must be positive")
    n = payments_per_day * 365
    return float((1.0 + rate) ** n - 1.0)


# ---------------------------------------------------------------------------
# Futures carry
# ---------------------------------------------------------------------------

class FuturesCarrySignal(BaseSignalGenerator):
    """
    Roll-yield carry for a single futures series.

    Expects a bars DataFrame with columns:
        front_price:    Near (front-month) contract price per bar.
        back_price:     Next (back-month) contract price per bar.
        days_to_expiry: Calendar days between the two contracts (used to
                        annualize). Column is optional; if missing, the
                        ``default_days_to_expiry`` param is used.

    Carry formula:
        carry = (front_price - back_price) / front_price * (365 / days_between)

    If ``annualize`` is False, the (365 / days_between) factor is dropped.

    Side:
        carry > 0  (backwardation)  → long  (+1) : front rich, roll gain.
        carry < 0  (contango)       → short (-1) : front cheap, roll drag.

    Confidence = |carry| / rolling_max(|carry|, window=``confidence_window``),
    clipped to [0, 1].

    Parameters (via ``params``):
        annualize:                 Scale to annual rate (default True).
        default_days_to_expiry:    Fallback days_between when missing (30).
        confidence_window:         Lookback for confidence normalization (252).
    """

    REQUIRED_COLUMNS = ("front_price", "back_price")
    DEFAULT_PARAMS = {
        "annualize": True,
        "default_days_to_expiry": 30,
        "confidence_window": 252,
    }

    def __init__(
        self,
        name: str = "futures_carry",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)

        if int(self.params["default_days_to_expiry"]) <= 0:
            raise ValueError("default_days_to_expiry must be positive")
        if int(self.params["confidence_window"]) < 2:
            raise ValueError("confidence_window must be >= 2")

    # ------------------------------------------------------------------ API --

    def generate(
        self,
        bars: pd.DataFrame | None = None,
        symbol: str = "UNKNOWN",
        **kwargs: Any,
    ) -> list[Signal]:
        if bars is None:
            raise ValueError(f"{self.name}: bars DataFrame is required")
        self.validate_input(bars)

        front = bars["front_price"].astype(float)
        back = bars["back_price"].astype(float)
        if "days_to_expiry" in bars.columns:
            days = bars["days_to_expiry"].astype(float)
        else:
            days = pd.Series(
                float(self.params["default_days_to_expiry"]),
                index=bars.index,
            )

        # Avoid divide-by-zero on front_price and days.
        front_safe = front.replace(0.0, np.nan)
        days_safe = days.replace(0.0, np.nan)

        base_carry = (front - back) / front_safe
        if self.params["annualize"]:
            carry = base_carry * (365.0 / days_safe)
        else:
            carry = base_carry

        # Rolling max |carry| for confidence normalization.
        abs_carry = carry.abs()
        rolling_max = abs_carry.rolling(
            window=int(self.params["confidence_window"]), min_periods=1,
        ).max().replace(0.0, np.nan)

        signals: list[Signal] = []
        for t in carry.index:
            c = carry.loc[t]
            if not np.isfinite(c) or c == 0.0:
                continue
            side = int(np.sign(c))
            rmax = rolling_max.loc[t]
            if np.isfinite(rmax) and rmax > 0:
                confidence = float(min(abs(c) / rmax, 1.0))
            else:
                confidence = 0.0
            signals.append(
                Signal(
                    timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                    symbol=symbol,
                    family=self.name,
                    side=side,
                    confidence=confidence,
                    metadata={
                        "front_price": float(front.loc[t]),
                        "back_price": float(back.loc[t]),
                        "days_to_expiry": float(days.loc[t]),
                        "carry": float(c),
                        "annualized": bool(self.params["annualize"]),
                    },
                )
            )
        return signals


# ---------------------------------------------------------------------------
# Funding rate arbitrage
# ---------------------------------------------------------------------------

class FundingRateArbSignal(BaseSignalGenerator):
    """
    Delta-neutral crypto funding rate arbitrage.

    The trade: long spot + short perpetual. The position earns the perp's
    funding rate while being market-neutral. ``side`` is always +1 when
    active (we're always "long" the yield); exits emit side=0.

    Expects a bars DataFrame with a single ``funding_rate`` column (per-
    payment rate, e.g. 0.0001 for 1 bp). Set ``annualized`` param True if
    your data is already annualized.

    Entry:  annualized_funding  >  entry_threshold  → side = +1
    Exit:   annualized_funding  <  exit_threshold   → side =  0
    Otherwise: no signal emitted (hold current state).

    Confidence = clip(annualized_funding / 0.5, 0, 1).

    Parameters (via ``params``):
        entry_threshold:  Annualized entry threshold (default 0.10 = 10%).
        exit_threshold:   Annualized exit threshold (default 0.02 = 2%).
        payments_per_day: Funding cycles per day (default 3 for 8h cadence).
        annualized:       If True, input ``funding_rate`` is already
                          annualized and no conversion is applied. Default False.
    """

    REQUIRED_COLUMNS = ("funding_rate",)
    DEFAULT_PARAMS = {
        "entry_threshold": 0.10,
        "exit_threshold": 0.02,
        "payments_per_day": 3,
        "annualized": False,
    }

    def __init__(
        self,
        name: str = "funding_rate_arb",
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)

        if self.params["entry_threshold"] <= self.params["exit_threshold"]:
            raise ValueError("entry_threshold must exceed exit_threshold")
        if int(self.params["payments_per_day"]) <= 0:
            raise ValueError("payments_per_day must be positive")

    # ------------------------------------------------------------------ API --

    def generate(
        self,
        bars: pd.DataFrame | None = None,
        symbol: str = "UNKNOWN",
        **kwargs: Any,
    ) -> list[Signal]:
        if bars is None:
            raise ValueError(f"{self.name}: bars DataFrame is required")
        self.validate_input(bars)

        rates = bars["funding_rate"].astype(float)
        entry = float(self.params["entry_threshold"])
        exit_ = float(self.params["exit_threshold"])
        annualized = bool(self.params["annualized"])
        ppd = int(self.params["payments_per_day"])

        signals: list[Signal] = []
        current_side = 0  # track state so we only emit transitions

        for t in rates.index:
            raw = float(rates.loc[t])
            if not np.isfinite(raw):
                continue
            ann = raw if annualized else annualize_funding_rate(raw, payments_per_day=ppd)

            event: str | None = None
            side = current_side
            if current_side == 0 and ann > entry:
                event = "entry"
                side = 1
            elif current_side == 1 and ann < exit_:
                event = "exit"
                side = 0

            if event is None:
                continue

            confidence = float(np.clip(ann / 0.5, 0.0, 1.0)) if side == 1 else 0.0
            signals.append(
                Signal(
                    timestamp=t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                    symbol=symbol,
                    family=self.name,
                    side=int(side),
                    confidence=confidence,
                    metadata={
                        "event": event,
                        "raw_funding_rate": raw,
                        "annualized_funding": ann,
                        "entry_threshold": entry,
                        "exit_threshold": exit_,
                    },
                )
            )
            current_side = int(side)

        return signals
