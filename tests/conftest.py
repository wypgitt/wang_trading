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
