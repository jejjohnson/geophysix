"""geophysix on the array backend of `pipekit_array.functions`, which tests itself."""

from __future__ import annotations

import subprocess
import sys

import array_api_strict as strict
import numpy as np
import pytest
from _backends import needs_jax
from pipekit_array.functions import floating_inputs, like

from geophysix import BackendError
from geophysix.earth import geodetic_to_ecef, sphere
from geophysix.rotation import coriolis


SECOND = strict.Device("device1")


class Unknown:
    """An array-like of no known library, as a pandas Series is."""

    shape = (2,)
    ndim = 1
    dtype = np.dtype(np.float64)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.zeros(2)


@pytest.mark.parametrize("value", [Unknown(), object(), "text", {"a": 1}])
def test_an_unknown_array_raises_and_is_never_numpy(value: object) -> None:
    with pytest.raises(BackendError, match="not an array of a known library"):
        coriolis(value)


@needs_jax
def test_two_libraries_in_one_call_raise() -> None:
    import jax.numpy as jnp

    with pytest.raises(BackendError, match=r"array_api_strict, jax\.numpy"):
        geodetic_to_ecef(jnp.zeros(2), strict.zeros(2))


def test_values_that_are_not_real_numbers_raise() -> None:
    with pytest.raises(ValueError, match="real numbers"):
        coriolis(np.asarray([1j]))


def test_public_functions_stay_on_the_second_device() -> None:
    x = strict.asarray([10.0, 20.0], device=SECOND)
    assert coriolis(x).device == SECOND
    assert [a.device for a in geodetic_to_ecef(x, x)] == [SECOND] * 3


def test_a_record_field_can_choose_the_device() -> None:
    rate = strict.asarray(7e-5, device=SECOND)
    assert coriolis([1.0, 2.0], earth=sphere(6.4e6, rate)).device == SECOND


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
