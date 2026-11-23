"""The array namespace, dtype and device of a call."""

from __future__ import annotations

import subprocess
import sys
import types

import array_api_compat.numpy as numpy_namespace
import array_api_strict as strict
import numpy as np
import pytest
from _backends import needs_jax
from hypothesis import given, strategies as st

from geophysix import BackendError, GeophysixError
from geophysix._src._arrays import floating_inputs, like
from geophysix._src.backend import array_namespace, device_of, is_host, place
from geophysix.earth import geodetic_to_ecef
from geophysix.rotation import coriolis


SECOND = strict.Device("device1")


class Unknown:
    """An array-like of no known library, as a pandas Series is."""

    shape = (2,)
    ndim = 1
    dtype = np.dtype(np.float64)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.zeros(2)


def affine(values, slope, offset=0.0):
    """The toy function: sin(slope · values) + offset, through the backend."""
    xp, (values, slope, offset) = floating_inputs(values, slope, offset)
    return xp.sin(slope * values) + offset


# --- the namespace ---


@pytest.mark.parametrize(
    "value", [1, 1.0, True, None, [1.0], (1.0, 2.0), np.zeros(2), np.float64(1.0)]
)
def test_host_values_alone_resolve_to_numpy(value: object) -> None:
    assert is_host(value)
    assert array_namespace(value) is numpy_namespace


def test_no_arguments_resolve_to_numpy() -> None:
    assert array_namespace() is numpy_namespace


def test_numpy_joins_the_other_library() -> None:
    assert array_namespace(np.zeros(2), strict.asarray([1.0])) is strict
    assert array_namespace(strict.asarray([1.0]), 2.0, None) is strict


@needs_jax
def test_jax_wins_over_numpy_and_meets_no_third_library() -> None:
    import jax
    import jax.numpy as jnp

    assert array_namespace(np.zeros(2), jnp.zeros(2)) is jnp
    with pytest.raises(BackendError, match=r"array_api_strict, jax\.numpy"):
        array_namespace(jnp.zeros(2), strict.asarray([1.0]))

    seen = []

    def probe(x):
        seen.append(array_namespace(x))
        return x

    jax.jit(probe)(jnp.ones(2))
    assert seen == [jnp]


@pytest.mark.parametrize("value", [Unknown(), object(), "text", {"a": 1}])
def test_an_unknown_array_raises_and_is_never_numpy(value: object) -> None:
    with pytest.raises(BackendError, match="not an array of a known library"):
        array_namespace(np.zeros(2), value)
    with pytest.raises(BackendError, match="not an array of a known library"):
        coriolis(value)


# --- the dtype ---


def test_float32_stays_float32_next_to_numbers_and_integers() -> None:
    x = np.zeros(2, dtype=np.float32)
    _, arrays = floating_inputs(x, 2, 1.5, True, np.asarray([1, 2]), np.asarray(True))
    assert [a.dtype for a in arrays] == [np.float32] * 6


def test_float32_and_float64_give_float64() -> None:
    _, arrays = floating_inputs(np.float32([1.0]), np.float64([1.0]))
    assert [a.dtype for a in arrays] == [np.float64] * 2


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint16, np.int64, np.bool_])
def test_integers_become_the_default_float(dtype: type) -> None:
    _, (x,) = floating_inputs(np.asarray([0, 1], dtype=dtype))
    assert x.dtype == np.float64


def test_numbers_alone_give_the_default_float() -> None:
    _, (x, y) = floating_inputs(1, 2.5)
    assert x.dtype == y.dtype == np.float64
    assert x.shape == ()


def test_a_numpy_scalar_carries_its_dtype() -> None:
    _, (x, y) = floating_inputs(np.float32([1.0]), np.float64(1.0))
    assert x.dtype == y.dtype == np.float64


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1j]),
        3j,
        np.asarray(["2020-01-01"], dtype="datetime64[D]"),
        np.asarray(["a"]),
        np.asarray([object()]),
    ],
)
def test_values_that_are_not_real_numbers_raise(value: object) -> None:
    with pytest.raises(GeophysixError, match="real numbers"):
        floating_inputs(value, 1.0)


def test_params_choose_the_namespace_and_never_the_dtype() -> None:
    record = strict.asarray(1.0, dtype=strict.float64)
    xp, (x,) = floating_inputs(np.float32([1.0]), params=(record, 2.0))
    assert xp is strict
    assert x.dtype == strict.float32


def test_like_takes_the_dtype_of_the_reference() -> None:
    assert like(0.1, np.float32([1.0]), np).dtype == np.float32
    assert like(0.1, strict.asarray([1.0]), strict).dtype == strict.float64


@given(
    st.lists(st.floats(-10.0, 10.0), min_size=1, max_size=5),
    st.sampled_from([np.float32, np.float64]),
)
def test_every_input_comes_back_in_one_dtype(values: list[float], dtype: type) -> None:
    x = np.asarray(values, dtype=dtype)
    _, arrays = floating_inputs(x, 1, x[:1], values[0])
    assert {a.dtype for a in arrays} == {np.dtype(dtype)}
    np.testing.assert_array_equal(arrays[0], x)


# --- the device ---


def test_host_values_join_the_device_of_the_array() -> None:
    x = strict.asarray([1.0, 2.0], device=SECOND)
    _, arrays = floating_inputs(1.0, np.ones(2), x, [3.0])
    assert [a.device for a in arrays] == [SECOND] * 4
    assert like(0.1, x, strict).device == SECOND
    assert affine(x, 2.0, np.ones(2)).device == SECOND


def test_a_record_field_can_choose_the_device() -> None:
    field = strict.asarray(2.0, device=SECOND)
    _, (x,) = floating_inputs([1.0, 2.0], params=(field,))
    assert x.device == SECOND


def test_a_host_value_has_no_device() -> None:
    assert device_of(np.zeros(2), numpy_namespace) is None
    assert device_of(1.0, strict) is None


def test_public_functions_stay_on_the_second_device() -> None:
    x = strict.asarray([10.0, 20.0], device=SECOND)
    assert coriolis(x).device == SECOND
    assert [a.device for a in geodetic_to_ecef(x, x)] == [SECOND] * 3


# --- arrays of the call's library are cast, never rebuilt ---


class Recorder(types.ModuleType):
    """A namespace that fails when ``asarray`` is called on one of its arrays."""

    def __getattr__(self, name: str) -> object:
        return getattr(strict, name)

    def asarray(self, value, **kwargs):
        assert is_host(value), "asarray was called on an array of the library"
        return strict.asarray(value, **kwargs)


def test_an_array_of_the_library_is_never_rebuilt(monkeypatch) -> None:
    # torch.asarray detaches its input in torch 2.5 and 2.10
    recorder = Recorder("recorder")
    for module in ("geophysix._src.backend", "geophysix._src._arrays"):
        monkeypatch.setattr(f"{module}.array_namespace", lambda *values: recorder)
    x = strict.asarray([1.0, 2.0])
    field = strict.asarray(2.0)
    xp, (y, z) = floating_inputs(x, 1.5, params=(field,))
    like(field, y, xp)
    assert y is x
    assert z.dtype == x.dtype


def test_place_returns_the_array_itself_when_nothing_changes() -> None:
    x = strict.asarray([1.0])
    assert place(x, strict, strict.float64, None) is x
    assert place(x, strict, strict.float32, None).dtype == strict.float32


# --- the standard and JAX ---


def test_the_toy_function_runs_on_the_standard_only() -> None:
    x = strict.asarray([0.1, 0.2])
    out = affine(x, strict.asarray(2, dtype=strict.int32), 1.0)
    assert out.dtype == strict.float64
    expected = np.sin(2.0 * np.asarray([0.1, 0.2])) + 1.0
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-15)


@needs_jax
def test_transforms_go_through_the_backend() -> None:
    import jax
    import jax.numpy as jnp

    x = jnp.asarray([0.1, 0.2, 0.3])
    eager = jnp.stack([affine(value, 2.0) for value in x])
    np.testing.assert_allclose(jax.jit(affine)(x, 2.0), eager, rtol=1e-15)
    np.testing.assert_allclose(jax.vmap(affine, (0, None))(x, 2.0), eager, rtol=1e-15)
    slope = jax.grad(lambda value: affine(value, 2.0))(0.1)
    np.testing.assert_allclose(slope, 2.0 * np.cos(0.2), rtol=1e-15)


@needs_jax
def test_jax_float32_stays_float32() -> None:
    import jax.numpy as jnp

    assert affine(jnp.float32([1.0]), 2.0, 1).dtype == jnp.float32


@needs_jax
def test_a_traced_record_field_selects_jax() -> None:
    import jax

    def scaled(field):
        xp, (x,) = floating_inputs(np.asarray([1.0, 2.0]), params=(field,))
        return xp.sum(like(field, x, xp) ** 2 * x)

    step = 1e-6
    central = (scaled(2.0 + step) - scaled(2.0 - step)) / (2.0 * step)
    np.testing.assert_allclose(jax.grad(scaled)(2.0), central, rtol=1e-8)


def test_importing_the_package_imports_neither_jax_nor_xarray() -> None:
    code = (
        "import sys, geophysix.thermo, geophysix.earth, numpy as np;"
        "geophysix.earth.meridional_radius(np.zeros(2));"
        "sys.exit(bool({'jax', 'xarray'} & set(sys.modules)))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
