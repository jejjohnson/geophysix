"""Geodetic ↔ ECEF coordinates and the radii of curvature.

Gates G1-pyproj and G1-roundtrip of the Wave 1 epic (#9): the forward map
matches pyproj's EPSG:4979 → EPSG:4978 transformer to 1e-12 relative in
float64, and geodetic → ECEF → geodetic returns its input to 1e-9 ° and
1e-4 m within ±10 km of the ellipsoid.
"""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_numpy_jax_agree, needs_jax
from _oracles import ecef_from_geodetic, geod_inverse, wgs84_transformer
from hypothesis import given, settings, strategies as st

from geophysix._src.backend import array_namespace
from geophysix.earth import (
    SPHERE,
    WGS84,
    ecef_to_geodetic,
    geodetic_to_ecef,
    meridian_degree_length,
    meridional_radius,
    parallel_degree_length,
    parallel_radius,
    prime_vertical_radius,
    sphere,
)


RNG = np.random.default_rng(0)
LON = RNG.uniform(-180.0, 180.0, 200)
LAT = RNG.uniform(-90.0, 90.0, 200)
HEIGHT = RNG.uniform(-1_000.0, 10_000.0, 200)


def _stack(fn):
    """Wrap a three-output function so it returns one stacked array."""

    def wrapped(*args):
        xp = array_namespace(*args)
        return xp.stack(fn(*args))

    return wrapped


# --- oracle: pyproj ---


def test_forward_matches_pyproj() -> None:
    want = ecef_from_geodetic(LON, LAT, HEIGHT)
    got = geodetic_to_ecef(LON, LAT, HEIGHT)
    # 1e-12 relative to the Earth's radius: well under a micrometre
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=0, atol=1e-12 * WGS84.semi_major)


@pytest.mark.parametrize("height", [-500.0, 0.0, 10_000.0])
def test_inverse_matches_pyproj(height: float) -> None:
    from pyproj.enums import TransformDirection

    x, y, z = ecef_from_geodetic(LON, LAT, np.full_like(LON, height))
    lon, lat, h = wgs84_transformer().transform(
        x, y, z, direction=TransformDirection.INVERSE
    )
    got_lon, got_lat, got_h = ecef_to_geodetic(x, y, z)
    # near the surface PROJ's inverse is accurate to well under 1e-9 °
    np.testing.assert_allclose(got_lat, lat, rtol=0, atol=1e-9)
    np.testing.assert_allclose(got_lon, lon, rtol=0, atol=1e-9)
    np.testing.assert_allclose(got_h, h, rtol=0, atol=1e-4)


@pytest.mark.parametrize("height", [705_000.0, 35_786_000.0])
def test_inverse_at_orbital_heights(height: float) -> None:
    # PROJ's own inverse loses ~1e-7 ° at geostationary height, so the check
    # here is the round trip through the forward map verified against pyproj
    lat = np.linspace(-89.0, 89.0, 401)
    got_lon, got_lat, got_h = ecef_to_geodetic(*geodetic_to_ecef(30.0, lat, height))
    np.testing.assert_allclose(got_lat, lat, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got_lon, 30.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got_h, height, rtol=0, atol=1e-6)


# --- properties ---


@settings(max_examples=200, deadline=None)
@given(
    lon=st.floats(-180.0, 180.0),
    lat=st.floats(-90.0, 90.0),
    height=st.floats(-10_000.0, 10_000.0),
)
def test_round_trip(lon: float, lat: float, height: float) -> None:
    got_lon, got_lat, got_h = ecef_to_geodetic(*geodetic_to_ecef(lon, lat, height))
    assert abs(float(got_lat) - lat) < 1e-9
    assert abs(float(got_h) - height) < 1e-4
    if abs(lat) < 90.0 - 1e-9:
        # longitude is undefined on the polar axis
        dlon = (float(got_lon) - lon + 180.0) % 360.0 - 180.0
        assert abs(dlon) < 1e-9


def test_poles_and_equator() -> None:
    x, _, z = geodetic_to_ecef(np.array([0.0, 0.0, 0.0]), np.array([0.0, 90.0, -90.0]))
    np.testing.assert_allclose(x, [WGS84.semi_major, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(z, [0.0, WGS84.semi_minor, -WGS84.semi_minor], atol=1e-6)
    lon, lat, h = ecef_to_geodetic(0.0, 0.0, WGS84.semi_minor + 100.0)
    assert (float(lon), float(lat)) == (0.0, 90.0)
    assert abs(float(h) - 100.0) < 1e-6


def test_radii_of_curvature() -> None:
    # at the equator N = a and M = a (1 − e²); at the pole N = M = a² / b
    a, b, e2 = WGS84.semi_major, WGS84.semi_minor, WGS84.eccentricity_squared
    np.testing.assert_allclose(prime_vertical_radius(0.0), a, rtol=1e-15)
    np.testing.assert_allclose(meridional_radius(0.0), a * (1.0 - e2), rtol=1e-15)
    np.testing.assert_allclose(prime_vertical_radius(90.0), a**2 / b, rtol=1e-14)
    np.testing.assert_allclose(meridional_radius(90.0), a**2 / b, rtol=1e-14)


def test_sphere_radii_are_the_radius() -> None:
    body = sphere(6_371_000.0)
    np.testing.assert_allclose(prime_vertical_radius(LAT, earth=body), 6_371_000.0)
    np.testing.assert_allclose(meridional_radius(LAT, earth=body), 6_371_000.0)


def test_radians() -> None:
    deg = geodetic_to_ecef(LON, LAT, HEIGHT)
    rad = geodetic_to_ecef(np.deg2rad(LON), np.deg2rad(LAT), HEIGHT, degrees=False)
    np.testing.assert_allclose(rad, deg, rtol=1e-14)
    _, lat, _ = ecef_to_geodetic(*deg, degrees=False)
    np.testing.assert_allclose(lat, np.deg2rad(LAT), atol=1e-11)


# --- the parallel's radius and the length of a degree ---


def test_degree_lengths_reference_values() -> None:
    # computed from WGS84 and SPHERE: a (1 − e²) π/180, a / √(1 − e²) π/180, a π/180
    np.testing.assert_allclose(meridian_degree_length(0.0), 110_574.27582, rtol=1e-9)
    np.testing.assert_allclose(meridian_degree_length(90.0), 111_693.97956, rtol=1e-9)
    np.testing.assert_allclose(parallel_degree_length(0.0), 111_319.49079, rtol=1e-9)
    np.testing.assert_allclose(parallel_radius(0.0), WGS84.semi_major, rtol=1e-15)


@pytest.mark.parametrize("lat", [0.0, 45.0, 80.0])
def test_degree_lengths_match_pyproj(lat: float) -> None:
    # a 0.001° step: the geodesic and the arc agree far below 1e-8 relative
    step = 1e-3
    along_meridian = geod_inverse(0.0, lat - step / 2, 0.0, lat + step / 2) / step
    along_parallel = geod_inverse(0.0, lat, step, lat) / step
    np.testing.assert_allclose(meridian_degree_length(lat), along_meridian, rtol=1e-8)
    np.testing.assert_allclose(parallel_degree_length(lat), along_parallel, rtol=1e-8)


@given(st.floats(-90.0, 90.0))
@settings(max_examples=50, deadline=None)
def test_degree_lengths_on_a_sphere(lat: float) -> None:
    per_degree = SPHERE.semi_major * np.pi / 180.0
    np.testing.assert_allclose(
        meridian_degree_length(lat, earth=SPHERE), per_degree, rtol=1e-15
    )
    np.testing.assert_allclose(
        parallel_degree_length(lat, earth=SPHERE),
        per_degree * np.cos(np.deg2rad(lat)),
        rtol=1e-12,
        atol=1e-9,
    )


def test_meridian_length_grows_towards_the_poles() -> None:
    lengths = meridian_degree_length(np.linspace(0.0, 90.0, 91))
    assert bool(np.all(np.diff(lengths) > 0.0))


def test_parallel_radius_at_the_poles() -> None:
    np.testing.assert_allclose(parallel_radius([90.0, -90.0]), 0.0, atol=1e-9)


def test_degree_lengths_in_radians() -> None:
    for fn in (parallel_radius, meridian_degree_length, parallel_degree_length):
        np.testing.assert_allclose(
            fn(np.deg2rad(LAT), degrees=False), fn(LAT), rtol=1e-14, atol=1e-9
        )


# --- backends and transforms ---


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (_stack(geodetic_to_ecef), (LON[:8], LAT[:8], HEIGHT[:8])),
        (prime_vertical_radius, (LAT[:8],)),
        (meridional_radius, (LAT[:8],)),
        (parallel_radius, (LAT[:8],)),
        (meridian_degree_length, (LAT[:8],)),
        (parallel_degree_length, (LAT[:8],)),
    ],
)
def test_numpy_and_jax_agree(fn, args) -> None:
    assert_numpy_jax_agree(fn, *(np.asarray(a) for a in args), rtol=1e-12)


def test_inverse_numpy_and_jax_agree() -> None:
    # the angles to 1e-12 relative; h is a difference of two 6.4e6 m terms,
    # so NumPy and XLA round it differently by about 1e-10 m
    xyz = [np.asarray(v) for v in ecef_from_geodetic(LON[:8], LAT[:8], HEIGHT[:8])]
    angles = _stack(lambda x, y, z: ecef_to_geodetic(x, y, z)[:2])
    assert_numpy_jax_agree(angles, *xyz, rtol=1e-12)
    height = _stack(lambda x, y, z: ecef_to_geodetic(x, y, z)[2:])
    assert_numpy_jax_agree(height, *(1.0 * v for v in xyz), rtol=1e-8)


@needs_jax
def test_gradients_match_central_differences() -> None:
    import jax
    import jax.numpy as jnp

    def lat_of(xyz):
        return ecef_to_geodetic(*xyz)[1]

    for lat in (10.0, 60.0, 89.999):
        xyz = jnp.stack(geodetic_to_ecef(25.0, lat, 500.0))
        grad = jax.grad(lat_of)(xyz)
        # a central difference with a 1 cm step, relative to the 6.4e6 m scale
        step = 1e-2
        fd = jnp.stack(
            [
                (lat_of(xyz.at[i].add(step)) - lat_of(xyz.at[i].add(-step)))
                / (2 * step)
                for i in range(3)
            ]
        )
        # the floor is the difference's round-off, relative to the largest entry
        np.testing.assert_allclose(
            grad, fd, rtol=1e-5, atol=1e-6 * float(abs(grad).max())
        )


@needs_jax
def test_gradient_is_finite_on_the_polar_axis() -> None:
    import jax
    import jax.numpy as jnp

    xyz = jnp.array([0.0, 0.0, WGS84.semi_minor + 100.0])
    grads = jax.jacrev(lambda v: jnp.stack(ecef_to_geodetic(*v)))(xyz)
    assert bool(jnp.isfinite(grads).all())


@needs_jax
def test_jit_and_vmap_match_eager() -> None:
    import jax
    import jax.numpy as jnp

    lon, lat, h = (jnp.asarray(v[:16]) for v in (LON, LAT, HEIGHT))
    eager = jnp.stack(geodetic_to_ecef(lon, lat, h))
    jitted = jnp.stack(jax.jit(geodetic_to_ecef)(lon, lat, h))
    mapped = jnp.stack(jax.vmap(geodetic_to_ecef)(lon, lat, h))
    np.testing.assert_allclose(jitted, eager, rtol=1e-15)
    np.testing.assert_allclose(mapped, eager, rtol=1e-15)


def test_float32_stays_float32() -> None:
    lat = np.float32([10.0, 20.0])
    out = geodetic_to_ecef(np.float32([1.0, 2.0]), lat, 0.0)
    assert all(v.dtype == np.float32 for v in out)
    assert prime_vertical_radius(lat).dtype == np.float32


@needs_jax
def test_degree_lengths_under_transforms() -> None:
    import jax
    import jax.numpy as jnp

    lat = jnp.asarray(LAT[:16])
    for fn in (parallel_radius, meridian_degree_length, parallel_degree_length):
        eager = fn(lat)
        np.testing.assert_allclose(jax.jit(fn)(lat), eager, rtol=1e-15)
        np.testing.assert_allclose(jax.vmap(fn)(lat), eager, rtol=1e-15)
        grads = jax.vmap(jax.grad(fn))(lat)
        assert bool(jnp.isfinite(grads).all())
