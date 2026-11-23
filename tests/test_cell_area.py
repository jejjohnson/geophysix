"""The area of latitude-longitude cells on the ellipsoid and on the sphere.

Gate G1-globe of the Wave 1 epic (#9): the cells of a whole-globe grid sum to
4πR² on a sphere and to 4πR_q² on WGS84 (R_q the authalic radius).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _backends import needs_jax
from hypothesis import given, settings, strategies as st

from geophysix.earth import (
    GRS80,
    WGS84,
    Ellipsoid,
    cell_area,
    meridional_radius,
    prime_vertical_radius,
    sphere,
)


def _authalic_radius_squared(body: Ellipsoid) -> float:
    """R_q² = (b² / 2) q_p, q_p = 1 / (1 − e²) + ln((1 + e) / (1 − e)) / (2e)."""
    e2 = body.eccentricity_squared
    e = math.sqrt(e2)
    q_p = 1.0 / (1.0 - e2) + math.log((1.0 + e) / (1.0 - e)) / (2.0 * e)
    return 0.5 * body.semi_minor**2 * q_p


# --- G1-globe ---


@pytest.mark.parametrize("radius", [1.0, 6_371_000.0])
def test_sphere_globe(radius: float) -> None:
    area = cell_area(
        np.linspace(-90, 90, 91), np.linspace(-180, 180, 145), earth=sphere(radius)
    )
    assert math.isclose(float(area.sum()), 4.0 * math.pi * radius**2, rel_tol=1e-12)


@pytest.mark.parametrize("body", [WGS84, GRS80])
def test_ellipsoid_globe(body: Ellipsoid) -> None:
    area = cell_area(np.linspace(-90, 90, 181), np.linspace(0, 360, 73), earth=body)
    want = 4.0 * math.pi * _authalic_radius_squared(body)
    assert math.isclose(float(area.sum()), want, rel_tol=1e-12)


# --- oracle: quadrature of the area element ---


@pytest.mark.parametrize(
    ("lat1", "lat2"), [(-89.0, -60.0), (0.0, 1.0), (44.5, 45.5), (80.0, 90.0)]
)
def test_matches_quadrature_of_the_area_element(lat1: float, lat2: float) -> None:
    # dA = M(φ) N(φ) cos φ dφ dλ, by 40-point Gauss-Legendre in φ
    nodes, weights = np.polynomial.legendre.leggauss(40)
    phi = 0.5 * (lat2 - lat1) * nodes + 0.5 * (lat2 + lat1)
    jac = 0.5 * math.radians(lat2 - lat1)
    integrand = (
        meridional_radius(phi) * prime_vertical_radius(phi) * np.cos(np.deg2rad(phi))
    )
    dlam = math.radians(2.5)
    want = jac * float(weights @ integrand) * dlam
    got = float(cell_area(np.array([lat1, lat2]), np.array([10.0, 12.5]))[0, 0])
    assert math.isclose(got, want, rel_tol=1e-12)


def test_one_degree_cell_on_the_sphere() -> None:
    # A = R² Δλ (sin φ₂ − sin φ₁)
    radius = 6_371_000.0
    got = float(
        cell_area(np.array([30.0, 31.0]), np.array([0.0, 1.0]), earth=sphere(radius))[
            0, 0
        ]
    )
    want = (
        radius**2
        * math.radians(1.0)
        * (math.sin(math.radians(31.0)) - math.sin(math.radians(30.0)))
    )
    assert math.isclose(got, want, rel_tol=1e-13)


# --- properties ---


@settings(max_examples=100, deadline=None)
@given(
    lat1=st.floats(-90.0, 89.0),
    dlat=st.floats(0.01, 1.0),
    split=st.floats(0.1, 0.9),
    lon1=st.floats(-180.0, 179.0),
    dlon=st.floats(0.01, 1.0),
)
def test_positive_and_additive(
    lat1: float, dlat: float, split: float, lon1: float, dlon: float
) -> None:
    lat2 = min(lat1 + dlat, 90.0)
    mid = lat1 + split * (lat2 - lat1)
    whole = cell_area(np.array([lat1, lat2]), np.array([lon1, lon1 + dlon]))
    parts = cell_area(
        np.array([lat1, mid, lat2]), np.array([lon1, lon1 + split * dlon, lon1 + dlon])
    )
    assert float(whole[0, 0]) > 0.0
    assert math.isclose(float(parts.sum()), float(whole[0, 0]), rel_tol=1e-10)


def test_edge_order_does_not_matter() -> None:
    lat, lon = np.linspace(-30, 30, 7), np.linspace(0, 10, 5)
    np.testing.assert_allclose(
        cell_area(lat[::-1], lon[::-1]), cell_area(lat, lon)[::-1, ::-1], rtol=1e-15
    )


def test_radians() -> None:
    lat, lon = np.linspace(-30, 30, 7), np.linspace(0, 10, 5)
    rad = cell_area(np.deg2rad(lat), np.deg2rad(lon), degrees=False)
    np.testing.assert_allclose(rad, cell_area(lat, lon), rtol=1e-14)


# --- backends and transforms ---


@needs_jax
def test_gradient_with_respect_to_flattening_is_finite_at_the_sphere() -> None:
    import jax
    import jax.numpy as jnp

    lat, lon = jnp.linspace(-60.0, 60.0, 5), jnp.linspace(0.0, 10.0, 3)

    def total(f):
        return cell_area(lat, lon, earth=Ellipsoid(6_371_000.0, f)).sum()

    for f in (0.0, WGS84.flattening):
        grad = jax.grad(total)(jnp.asarray(f))
        assert bool(jnp.isfinite(grad))
        # a central difference with step 1e-7 in f (from f = 0, a one-sided one)
        step = 1e-7
        fd = (
            (total(f + step) - total(f)) / step
            if f == 0.0
            else (total(f + step) - total(f - step)) / (2 * step)
        )
        np.testing.assert_allclose(grad, fd, rtol=1e-5)


def test_float32_stays_float32() -> None:
    area = cell_area(np.float32([0.0, 1.0]), np.float32([0.0, 1.0]))
    assert area.dtype == np.float32
