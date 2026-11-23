"""The kinds of array a function must accept, and the check that they agree.

JAX, xarray, dask and torch are each optional here: a kind whose library is missing is
left out, so the job without JAX and xarray runs the rest. torch is in no dependency
group, because it is large; one CI job installs it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from collections.abc import Callable
from typing import Any

import array_api_strict as strict
import numpy as np
import pytest


__all__ = [
    "HAS_DASK",
    "HAS_JAX",
    "HAS_TORCH",
    "HAS_XARRAY",
    "KINDS",
    "ArrayKind",
    "assert_backends_agree",
    "needs_jax",
    "needs_torch",
    "needs_xarray",
    "outputs",
]

HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_DASK = importlib.util.find_spec("dask") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_XARRAY = importlib.util.find_spec("xarray") is not None

needs_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
needs_xarray = pytest.mark.skipif(not HAS_XARRAY, reason="xarray is not installed")


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayKind:
    """One kind of array: how to make it, what comes back, and how to read it."""

    name: str
    make: Callable[[np.ndarray], Any]
    type: Any
    to_numpy: Callable[[Any], np.ndarray] = np.asarray

    def read(self, array: Any) -> np.ndarray:
        """The array as NumPy, after checking that it is of this kind."""
        assert isinstance(array, self.type), (
            f"{self.name} in must give {self.name} out, got {type(array).__name__}"
        )
        return self.to_numpy(array)


def _dims(array: np.ndarray) -> tuple[str, ...]:
    """Dimension names from the right, so that arrays broadcast as NumPy does."""
    return tuple(f"dim{array.ndim - axis}" for axis in range(array.ndim))


def _second_device() -> ArrayKind:
    """`array_api_strict` on a device that is not the default one."""
    device = strict.Device("device1")

    def make(array: np.ndarray) -> Any:
        return strict.asarray(array, device=device)

    def to_numpy(array: Any) -> np.ndarray:
        assert array.device == device, f"the result is on {array.device}"
        return np.asarray(array.to_device(strict.Device("CPU_DEVICE")))

    return ArrayKind("second device", make, type(strict.asarray(0.0)), to_numpy)


def _kinds() -> list[ArrayKind]:
    strict_type = type(strict.asarray(0.0))
    kinds = [
        ArrayKind("numpy", np.asarray, (np.ndarray, np.generic)),
        ArrayKind("strict", strict.asarray, strict_type),
        _second_device(),
    ]
    if HAS_JAX:
        import jax
        import jax.numpy as jnp

        kinds.append(ArrayKind("jax", jnp.asarray, jax.Array))
        kinds.append(ArrayKind("jax.jit", jnp.asarray, jax.Array))
    if HAS_DASK:
        import dask.array as da

        kinds.append(ArrayKind("dask", da.from_array, da.Array))
    if HAS_TORCH:
        import torch

        kinds.append(
            ArrayKind("torch", torch.tensor, torch.Tensor, lambda t: t.detach().numpy())
        )
    if HAS_XARRAY:
        import xarray as xr

        def labelled(array: np.ndarray) -> Any:
            return xr.DataArray(array, dims=_dims(array))

        kinds.append(ArrayKind("xarray", labelled, xr.DataArray))
    return kinds


KINDS = _kinds()


def outputs(result: Any) -> tuple[Any, ...]:
    """The arrays a function returned, as a tuple."""
    return result if isinstance(result, tuple) else (result,)


def assert_backends_agree(
    fn: Callable[..., Any],
    *args: Any,
    rtol: float = 1e-12,
    atol: float = 0.0,
    rtol32: float = 1e-5,
    atol32: float = 0.0,
    kinds: list[ArrayKind] | None = None,
) -> None:
    """Check that every array kind gives NumPy's numbers, kind, dtype and device.

    For each kind: the same kind comes back, float64 values agree with NumPy's
    to ``rtol`` and ``atol``, and float32 in gives float32 out, which agrees
    with NumPy's float64 to ``rtol32`` and ``atol32``.

    Args:
        fn: The function under test; it returns an array or a tuple of them.
        *args: Float64 NumPy arrays to call it with.
        rtol: Relative tolerance in float64.
        atol: Absolute tolerance in float64, for results that cross zero.
        rtol32: Relative tolerance of float32 against float64.
        atol32: Absolute tolerance of float32 against float64.
        kinds: The kinds to check; every one in `KINDS` by default.

    Raises:
        AssertionError: If a check fails; the message names the kind.
    """
    arrays = [np.asarray(arg, dtype=np.float64) for arg in args]
    expected = outputs(fn(*arrays))
    checks = ((np.float64, rtol, atol), (np.float32, rtol32, atol32))
    for kind in KINDS if kinds is None else kinds:
        variant = fn
        if kind.name == "jax.jit":
            import jax

            variant = jax.jit(fn)
        for dtype, relative, absolute in checks:
            inputs = [kind.make(array.astype(dtype)) for array in arrays]
            got = outputs(variant(*inputs))
            label = f"{kind.name} {np.dtype(dtype).name}"
            for array, want in zip(got, expected, strict=True):
                values = kind.read(array)
                assert values.dtype == dtype, f"{label}: {values.dtype} came back"
                np.testing.assert_allclose(
                    values, want, rtol=relative, atol=absolute, err_msg=label
                )
