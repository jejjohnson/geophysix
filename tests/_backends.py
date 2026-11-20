"""Backend helpers for tests: the JAX marker and the NumPy / JAX parity check.

JAX is optional, so every JAX test is marked ``needs_jax`` and skips when it
is missing; the job that installs geophysix without JAX runs the rest.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest


__all__ = [
    "HAS_JAX",
    "assert_numpy_jax_agree",
    "needs_jax",
]

HAS_JAX = importlib.util.find_spec("jax") is not None

needs_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")


def assert_numpy_jax_agree(
    fn: Callable[..., Any], *args: np.ndarray, rtol: float = 1e-12
) -> None:
    """Check a function's backends and dtypes.

    Always checks that NumPy in gives NumPy out and that float32 in gives
    float32 out. When JAX is importable, also checks that ``fn`` under
    ``jax.jit`` on JAX arrays gives the same numbers as on NumPy, to ``rtol``,
    and that JAX float32 in gives float32 out. Without JAX only the NumPy
    checks run.

    Args:
        fn: The function under test, taking arrays and returning one array.
        *args: Float64 NumPy arrays to call it with.
        rtol: Relative tolerance of the NumPy and JAX comparison.

    Raises:
        AssertionError: If a check fails.
    """
    expected = fn(*args)
    assert isinstance(expected, np.ndarray), "NumPy in must give NumPy out"

    args32 = [np.asarray(a, dtype=np.float32) for a in args]
    out32 = fn(*args32)
    assert isinstance(out32, np.ndarray), "NumPy in must give NumPy out"
    assert out32.dtype == np.float32, "float32 in must give float32 out"

    if not HAS_JAX:
        return
    import jax
    import jax.numpy as jnp

    jitted = jax.jit(fn)
    got = jitted(*(jnp.asarray(a) for a in args))
    assert isinstance(got, jax.Array), "JAX in must give JAX out"
    np.testing.assert_allclose(np.asarray(got), expected, rtol=rtol)

    got32 = jitted(*(jnp.asarray(a) for a in args32))
    assert got32.dtype == jnp.float32, "JAX float32 in must give float32 out"
