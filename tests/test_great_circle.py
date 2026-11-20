"""Great-circle distance and initial azimuth on a sphere.

The oracle is pyproj's ``Geod`` on a sphere (flattening 0), which solves the
geodesic problem with Karney's method rather than the haversine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _backends import assert_numpy_jax_agree, needs_jax
from hypothesis import given, settings, strategies as st

from geophysix.earth import (
    SPHERE,
    WGS84,
    great_circle_distance,
    initial_azimuth,
    sphere,
)
from geophysix.exceptions import EarthError


RNG = np.random.default_rng(1)
LON1, LON2 = RNG.uniform(-180, 180, (2, 300))
LAT1, LAT2 = RNG.uniform(-89.9, 89.9, (2, 300))
RADIUS = 6_371_000.0


def test_distance_matches_pyproj() -> None:
    pyproj = pytest.importorskip("pyproj")
    az, _, dist = pyproj.Geod(a=RADIUS, f=0.0).inv(LON1, LAT1, LON2, LAT2)
    got = great_circle_distance(LON1, LAT1, LON2, LAT2, earth=sphere(RADIUS))
    # 1e-12 of the half circumference, i.e. 20 µm
    np.testing.assert_allclose(got, dist, rtol=0, atol=1e-12 * math.pi * RADIUS)
    np.testing.assert_allclose(
        initial_azimuth(LON1, LAT1, LON2, LAT2), az, rtol=0, atol=1e-9
    )


def test_default_sphere_is_the_mean_radius() -> None:
    assert math.isclose(
        float(great_circle_distance(0.0, 0.0, 180.0, 0.0)),
        math.pi * SPHERE.semi_major,
        rel_tol=1e-15,
    )


def test_coincident_and_antipodal_points() -> None:
    assert float(great_circle_distance(12.0, -40.0, 12.0, -40.0)) == 0.0
    antipode = float(
        great_circle_distance(12.0, -40.0, -168.0, 40.0, earth=sphere(1.0))
    )
    assert math.isclose(antipode, math.pi, rel_tol=1e-15)


@settings(max_examples=100, deadline=None)
@given(
    lon1=st.floats(-180, 180),
    lat1=st.floats(-90, 90),
    lon2=st.floats(-180, 180),
    lat2=st.floats(-90, 90),
)
def test_symmetric_and_bounded(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> None:
    forward = float(great_circle_distance(lon1, lat1, lon2, lat2, earth=sphere(1.0)))
    backward = float(great_circle_distance(lon2, lat2, lon1, lat1, earth=sphere(1.0)))
    assert math.isclose(forward, backward, rel_tol=1e-12, abs_tol=1e-15)
    assert 0.0 <= forward <= math.pi + 1e-15


def test_ellipsoid_is_refused() -> None:
    with pytest.raises(EarthError, match="sphere"):
        great_circle_distance(0.0, 0.0, 1.0, 1.0, earth=WGS84)


def test_cardinal_azimuths() -> None:
    np.testing.assert_allclose(
        initial_azimuth(
            0.0,
            0.0,
            np.array([0.0, 10.0, 0.0, -10.0]),
            np.array([10.0, 0.0, -10.0, 0.0]),
        ),
        [0.0, 90.0, 180.0, -90.0],
        atol=1e-12,
    )


def test_radians() -> None:
    args = [np.deg2rad(v) for v in (LON1, LAT1, LON2, LAT2)]
    np.testing.assert_allclose(
        great_circle_distance(*args, degrees=False),
        great_circle_distance(LON1, LAT1, LON2, LAT2),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        initial_azimuth(*args, degrees=False),
        np.deg2rad(initial_azimuth(LON1, LAT1, LON2, LAT2)),
        atol=1e-14,
    )


# --- backends and transforms ---


@pytest.mark.parametrize("fn", [great_circle_distance, initial_azimuth])
def test_numpy_and_jax_agree(fn) -> None:
    assert_numpy_jax_agree(fn, LON1[:8], LAT1[:8], LON2[:8], LAT2[:8], rtol=1e-12)


@needs_jax
def test_gradient_is_finite_at_coincident_and_antipodal_points() -> None:
    import jax
    import jax.numpy as jnp

    def dist(p):
        return great_circle_distance(p[0], p[1], p[2], p[3], earth=sphere(1.0))

    for p in ([10.0, 20.0, 10.0, 20.0], [10.0, 20.0, -170.0, -20.0]):
        grad = jax.grad(dist)(jnp.asarray(p))
        assert bool(jnp.isfinite(grad).all())


@needs_jax
def test_gradient_matches_central_differences() -> None:
    import jax
    import jax.numpy as jnp

    def dist(p):
        return great_circle_distance(p[0], p[1], p[2], p[3], earth=sphere(1.0))

    p = jnp.asarray([10.0, 20.0, 40.0, -15.0])
    grad = jax.grad(dist)(p)
    step = 1e-6
    fd = jnp.stack(
        [
            (dist(p.at[i].add(step)) - dist(p.at[i].add(-step))) / (2 * step)
            for i in range(4)
        ]
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-7)


def test_float32_stays_float32() -> None:
    lat = np.float32([10.0, 20.0])
    assert great_circle_distance(lat, lat, lat[::-1], lat[::-1]).dtype == np.float32
    assert initial_azimuth(lat, lat, lat[::-1], lat[::-1]).dtype == np.float32
