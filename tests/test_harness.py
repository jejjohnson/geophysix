"""Smoke test of the harness: one toy function passes all four checks.

The toy function is f(x) = 2 sin x, written the way every geophysix function
is: it computes in the namespace of its inputs. The four checks
are the house style's: an oracle, a property, a gradient and backend parity.
"""

from __future__ import annotations

import numpy as np
import pytest
from _backends import HAS_JAX, KINDS, assert_backends_agree, needs_jax
from _oracles import ecef_from_geodetic, geod_inverse, gsw_module
from hypothesis import given, strategies as st

from geophysix._src._arrays import floating_inputs, like
from geophysix._src.backend import array_namespace


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
    assert_backends_agree(toy, np.linspace(-3.0, 3.0, 25))


def test_the_check_catches_numpy_coming_back() -> None:
    def leaves(x):
        return np.sin(np.asarray(x))

    plain = [kind for kind in KINDS if kind.name == "strict"]
    with pytest.raises(AssertionError, match="strict in must give strict out"):
        assert_backends_agree(leaves, np.ones(3), kinds=plain)


def test_the_check_catches_a_promoted_float32() -> None:
    def promotes(x):
        return np.asarray(x, dtype=np.float64)

    with pytest.raises(AssertionError, match="numpy float32: float64 came back"):
        assert_backends_agree(promotes, np.ones(3), kinds=KINDS[:1])


def test_the_check_catches_a_constant_on_the_default_device() -> None:
    def misplaces(x):
        return x * array_namespace(x).asarray(2.0, dtype=x.dtype)

    second = [kind for kind in KINDS if kind.name == "second device"]
    with pytest.raises(ValueError, match="two different devices"):
        assert_backends_agree(misplaces, np.ones(3), kinds=second)


def test_the_check_catches_a_wrong_value() -> None:
    def disagrees(x):
        return x if isinstance(x, np.ndarray) else 2.0 * x

    with pytest.raises(AssertionError, match="strict float64"):
        assert_backends_agree(disagrees, np.ones(3))


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
