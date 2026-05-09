#!/usr/bin/env python3
"""Run the local read-only trade ideas UI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ui.trade_ideas_app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
