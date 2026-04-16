"""Shared test fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Work around the OpenMP-duplicate-libomp.dylib conflict between numpy/scipy
# (Accelerate/openblas) and PyTorch on macOS. Set before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Intentionally do NOT import torch here — pulling torch into the conftest
# loads its bundled libomp first, which then crashes LightGBM's Dataset
# constructor in later tests on macOS. Modules that need torch (e.g.
# regime_detector) cap its thread pool at their own import site.
