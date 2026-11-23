"""The horizontal wind: speed, bearing and components."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _backends import needs_jax
from hypothesis import given, settings, strategies as st

from geophysix.exceptions import GeophysixError
from geophysix.wind import wind_components, wind_direction, wind_speed


RNG = np.random.default_rng(5)
U = RNG.normal(0.0, 8.0, 64)
V = RNG.normal(0.0, 8.0, 64)


def test_cardinal_winds() -> None:
    # a westerly blows from 270°, a northerly from 0°, a southerly from 180°
    u = np.array([5.0, 0.0, 0.0, -5.0])
    v = np.array([0.0, -5.0, 5.0, 0.0])
    np.testing.assert_allclose(wind_direction(u, v), [270.0, 0.0, 180.0, 90.0])
    np.testing.assert_allclose(
        wind_direction(u, v, convention="to"), [90.0, 180.0, 0.0, 270.0]
    )
    np.testing.assert_allclose(wind_speed(u, v), 5.0)


def test_bearings_lie_in_one_turn() -> None:
    for convention in ("from", "to"):
        deg = wind_direction(U, V, convention=convention)
        rad = wind_direction(U, V, convention=convention, degrees=False)
        assert bool(np.all((deg >= 0.0) & (deg < 360.0)))
        assert bool(np.all((rad >= 0.0) & (rad < 2.0 * math.pi)))
        np.testing.assert_allclose(np.deg2rad(deg), rad, rtol=1e-13)


@given(
    st.floats(-50.0, 50.0).filter(lambda x: abs(x) > 1e-3),
    st.floats(-50.0, 50.0),
    st.sampled_from(["from", "to"]),
    st.booleans(),
)
@settings(max_examples=100, deadline=None)
def test_round_trip(u: float, v: float, convention: str, degrees: bool) -> None:
    speed = wind_speed(u, v)
    direction = wind_direction(u, v, convention=convention, degrees=degrees)
    u2, v2 = wind_components(speed, direction, convention=convention, degrees=degrees)
    np.testing.assert_allclose([u2, v2], [u, v], rtol=1e-12, atol=1e-12)


def test_calm_air() -> None:
    assert float(wind_speed(0.0, 0.0)) == 0.0
    assert float(wind_direction(0.0, 0.0)) == 270.0


def test_matches_xrops_and_plumax_bit_for_bit() -> None:
    # xrops atm/_src/wind.py and plumax _src/wind.py
    np.testing.assert_array_equal(wind_speed(U, V), (U**2 + V**2) ** 0.5)
    from_xrops = np.mod(270.0 - np.degrees(np.arctan2(V, U)), 360.0)
    np.testing.assert_array_equal(wind_direction(U, V), from_xrops)
    np.testing.assert_array_equal(
        wind_direction(U, V, convention="to"), np.mod(from_xrops + 180.0, 360.0)
    )
    speed, bearing = np.abs(U), np.mod(V * 40.0, 360.0)
    u, v = wind_components(speed, bearing)
    np.testing.assert_array_equal(u, -speed * np.sin(np.deg2rad(bearing)))
    np.testing.assert_array_equal(v, -speed * np.cos(np.deg2rad(bearing)))
    u, v = wind_components(speed, bearing, convention="to")
    np.testing.assert_array_equal(u, speed * np.sin(np.radians(bearing)))
    np.testing.assert_array_equal(v, speed * np.cos(np.radians(bearing)))


@pytest.mark.parametrize("fn", [wind_direction, wind_components])
def test_bad_convention(fn) -> None:
    with pytest.raises(GeophysixError, match="convention must be 'from' or 'to'"):
        fn(1.0, 1.0, convention="towards")


@needs_jax
def test_gradients_are_finite_at_calm() -> None:
    import jax

    for fn in (wind_speed, wind_direction):
        grads = jax.grad(fn, argnums=(0, 1))(0.0, 0.0)
        assert all(float(g) == 0.0 for g in grads)


@needs_jax
def test_jit_and_vmap_match_eager() -> None:
    import jax
    import jax.numpy as jnp

    u, v = jnp.asarray(U[:16]), jnp.asarray(V[:16])
    for fn in (wind_speed, wind_direction):
        eager = fn(u, v)
        # XLA fuses the bearing's arithmetic, so it rounds a few ulp apart
        np.testing.assert_allclose(jax.jit(fn)(u, v), eager, rtol=1e-14)
        np.testing.assert_allclose(jax.vmap(fn)(u, v), eager, rtol=1e-14)
        assert bool(jnp.isfinite(jax.vmap(jax.grad(fn))(u, v)).all())
    eager = jnp.stack(wind_components(jnp.abs(u), v))
    jitted = jnp.stack(jax.jit(wind_components)(jnp.abs(u), v))
    np.testing.assert_allclose(jitted, eager, rtol=1e-14)
