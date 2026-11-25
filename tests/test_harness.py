"""Smoke test of the harness: one toy function passes all four checks.

The toy function is f(x) = 2 sin x, written the way every geophysix function
is: it computes in the namespace of its inputs. The four checks
are the house style's: an oracle, a property, a gradient and backend parity. That
the last one fails where it should is tested in pipekit-array, which has it.
"""

from __future__ import annotations

import numpy as np
import pytest
from _backends import HAS_JAX, assert_backends_agree, needs_jax
from _oracles import ecef_from_geodetic, geod_inverse, gsw_module
from hypothesis import given, strategies as st
from pipekit_array.arrays import floating_inputs, like
from pipekit_array.kernels import elementwise


@elementwise()
def toy(x):
    """Return 2 sin x in the namespace of ``x``."""
    xp, (x,) = floating_inputs(x)
    return like(2.0, x, xp) * xp.sin(x)


def test_oracle_matches_the_closed_form() -> None:
    x = np.linspace(-3.0, 3.0, 25)
    np.testing.assert_allclose(toy(x), 2.0 * np.sin(x), rtol=1e-14)


@given(st.floats(min_value=-10.0, max_value=10.0))
def test_toy_is_odd(value: float) -> None:
    x = np.asarray(value)
    assert toy(-x) == -toy(x)


@needs_jax
def test_gradient_matches_central_differences() -> None:
    import jax
    import jax.numpy as jnp

    x = jnp.linspace(-2.0, 2.0, 9)
    step = 1e-6
    central = (toy(x + step) - toy(x - step)) / (2.0 * step)
    grad = jax.vmap(jax.grad(toy))(x)
    np.testing.assert_allclose(np.asarray(grad), np.asarray(central), rtol=1e-6)


def test_every_array_kind_agrees() -> None:
    assert_backends_agree(toy, np.linspace(-3.0, 3.0, 25), labelled=True)


def test_has_jax_matches_the_environment() -> None:
    import importlib.util

    assert (importlib.util.find_spec("jax") is not None) == HAS_JAX


def test_pyproj_ecef_oracle_on_the_equator() -> None:
    x, y, z = ecef_from_geodetic(0.0, 0.0, 0.0)
    assert x == pytest.approx(6378137.0, rel=1e-12)
    assert y == pytest.approx(0.0, abs=1e-6)
    assert z == pytest.approx(0.0, abs=1e-6)


def test_pyproj_geodesic_oracle_on_a_meridian() -> None:
    distance = geod_inverse(0.0, 0.0, 0.0, 1.0)
    assert distance == pytest.approx(110574.389, abs=1e-2)


def test_gsw_oracle_returns_a_sea_water_density() -> None:
    density = gsw_module().rho(35.0, 10.0, 0.0)
    assert 1025.0 < float(density) < 1028.0
