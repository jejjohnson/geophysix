"""The kinds of array a function must accept, and what a test needs to run.

The kinds, and the check that they agree, are pipekit-array's
(`pipekit_array.testing`). JAX, xarray, dask and torch are each optional: a kind whose
library is missing is left out, so the job without JAX and xarray runs the rest. torch
is in no dependency group, because it is large; one CI job installs it.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from pipekit_array.testing import ArrayKind, array_kinds, assert_backends_agree


__all__ = [
    "HAS_JAX",
    "KINDS",
    "ArrayKind",
    "assert_backends_agree",
    "needs_jax",
    "needs_torch",
    "outputs",
]

HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

needs_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")

KINDS = array_kinds()


def outputs(result: Any) -> tuple[Any, ...]:
    """The arrays a function returned, as a tuple."""
    return result if isinstance(result, tuple) else (result,)
