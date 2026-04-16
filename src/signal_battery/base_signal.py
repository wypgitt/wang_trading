"""
Signal Battery — base types.

Every primary alpha model emits a stream of ``Signal`` objects consumed by
the Labeling Engine (triple-barrier) and the ML meta-labeler. Generators
subclass ``BaseSignalGenerator`` and implement ``generate(bars)``.

The Signal is deliberately simple and family-agnostic: ``side`` and
``confidence`` are what the Bet Sizing layer cares about; everything
strategy-specific goes in ``metadata``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Signal:
    """
    One alpha-model output.

    Fields:
        timestamp:  Bar close time at which the signal is valid.
        symbol:     Instrument ticker.
        family:     Short tag for the generating signal family (e.g.
                    ``"ts_momentum"``, ``"mean_reversion"``).
        side:       +1 long, -1 short, 0 neutral.
        confidence: Signal strength in [0, 1]; passed through to bet sizing.
        metadata:   Free-form context (lookbacks, z-scores, hedge ratios, etc.).
    """

    timestamp: datetime
    symbol: str
    family: str
    side: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in (-1, 0, 1):
            raise ValueError(f"side must be -1, 0 or +1 (got {self.side})")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1] (got {self.confidence})"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (datetime serialised to ISO 8601)."""
        d = asdict(self)
        if isinstance(d["timestamp"], datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        return d


# ---------------------------------------------------------------------------
# Abstract base generator
# ---------------------------------------------------------------------------

class BaseSignalGenerator(ABC):
    """
    Abstract parent for all signal generators.

    Subclasses must:
      - set ``REQUIRED_COLUMNS`` (class attribute) to the minimum set of bar
        columns they need;
      - implement ``generate(bars, **kwargs) -> list[Signal]``.

    Callers can invoke ``validate_input(bars)`` to guard against shape
    errors before running the (potentially expensive) generator.
    """

    REQUIRED_COLUMNS: tuple[str, ...] = ("close",)

    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.params: dict[str, Any] = dict(params) if params else {}

    @abstractmethod
    def generate(
        self,
        bars: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> list[Signal]:
        """Produce signals from a bars DataFrame (or kwarg-supplied inputs).

        Most subclasses consume ``bars`` directly. Panel- and pair-style
        generators (cross-sectional momentum, stat-arb) take their input
        via keyword arguments and ignore ``bars``; the base's signature
        therefore makes it optional so orchestrator dispatch stays
        type-clean.
        """

    def validate_input(self, bars: pd.DataFrame) -> bool:
        """
        Verify ``bars`` has the required columns and a usable index.

        Returns True when valid. Raises ValueError on malformed input.
        """
        if not isinstance(bars, pd.DataFrame):
            raise ValueError("bars must be a pandas DataFrame")
        missing = [c for c in self.REQUIRED_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(
                f"{self.name}: bars missing columns {missing}"
            )
        if bars.empty:
            raise ValueError(f"{self.name}: bars DataFrame is empty")
        return True

    def __repr__(self) -> str:  # pragma: no cover — trivial
        return f"{type(self).__name__}(name={self.name!r}, params={self.params!r})"
