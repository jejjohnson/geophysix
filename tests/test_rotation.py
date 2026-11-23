"""The Coriolis parameter f, its gradient β, and the β-plane."""

from __future__ import annotations

import math

import numpy as np
from _backends import needs_jax
from hypothesis import given, strategies as st

from geophysix.earth import WGS84, meridional_radius, sphere
from geophysix.rotation import beta, beta_plane, coriolis


LAT = np.linspace(-89.0, 89.0, 37)
# gridax's convention: R = 6,371,200 m and Ω = 2π / 86,164 s
GRIDAX_EARTH = sphere(6_371_200.0, rotation_rate=2.0 * math.pi / 86_164.0)


def test_closed_forms() -> None:
    omega = WGS84.rotation_rate
    np.testing.assert_allclose(
        coriolis(LAT), 2.0 * omega * np.sin(np.deg2rad(LAT)), rtol=1e-15
    )
    want = 2.0 * omega * np.cos(np.deg2rad(LAT)) / meridional_radius(LAT)
    np.testing.assert_allclose(beta(LAT), want, rtol=1e-14)


def test_reproduces_gridax_at_its_constants() -> None:
    # G2-gridax: gridax's coriolis_param and beta_param at one latitude
    omega, radius = GRIDAX_EARTH.rotation_rate, GRIDAX_EARTH.semi_major
    phi = math.radians(40.0)
    assert math.isclose(
        float(coriolis(40.0, earth=GRIDAX_EARTH)),
        2.0 * omega * math.sin(phi),
        rel_tol=1e-15,
    )
    assert math.isclose(
        float(beta(40.0, earth=GRIDAX_EARTH)),
        2.0 * omega * math.cos(phi) / radius,
        rel_tol=1e-15,
    )
    # gridax's coriolis_fn defaults are this pair, to the 4 digits they are written with
    assert math.isclose(
        float(coriolis(40.0, earth=GRIDAX_EARTH)), 9.375e-5, rel_tol=1e-3
    )
    assert math.isclose(float(beta(40.0, earth=GRIDAX_EARTH)), 1.754e-11, rel_tol=1e-3)


@given(st.floats(-90.0, 90.0))
def test_coriolis_is_odd_and_beta_even(lat: float) -> None:
    assert math.isclose(float(coriolis(-lat)), -float(coriolis(lat)), abs_tol=1e-20)
    assert math.isclose(
        float(beta(-lat)), float(beta(lat)), rel_tol=1e-15, abs_tol=1e-30
    )


def test_poles_and_equator() -> None:
    assert float(coriolis(0.0)) == 0.0
    assert math.isclose(float(coriolis(90.0)), 2.0 * WGS84.rotation_rate, rel_tol=1e-15)
    assert abs(float(beta(90.0))) < 1e-25


def test_beta_plane_is_linear() -> None:
    y = np.linspace(-1e6, 1e6, 5)
    f = beta_plane(y, f0=1e-4, beta=2e-11, y0=2e5)
    np.testing.assert_allclose(f, 1e-4 + 2e-11 * (y - 2e5), rtol=1e-15)


def test_radians() -> None:
    np.testing.assert_allclose(
        coriolis(np.deg2rad(LAT), degrees=False), coriolis(LAT), rtol=1e-15
    )
    np.testing.assert_allclose(
        beta(np.deg2rad(LAT), degrees=False), beta(LAT), rtol=1e-15
    )


@needs_jax
def test_gradient_of_f_is_its_derivative() -> None:
    import jax
    import jax.numpy as jnp

    # ∂f/∂φ = 2 Ω cos φ, in radians
    lat = jnp.deg2rad(jnp.asarray(LAT))
    grad = jax.vmap(jax.grad(lambda p: coriolis(p, degrees=False)))(lat)
    np.testing.assert_allclose(
        grad, 2.0 * WGS84.rotation_rate * np.cos(np.asarray(lat)), rtol=1e-14
    )


@needs_jax
def test_a_traced_rotation_rate() -> None:
    import jax

    grad = jax.grad(
        lambda omega: coriolis(30.0, earth=sphere(6.4e6, rotation_rate=omega))
    )(7e-5)
    assert math.isclose(float(grad), 2.0 * math.sin(math.radians(30.0)), rel_tol=1e-14)
