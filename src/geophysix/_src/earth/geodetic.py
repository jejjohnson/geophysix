"""Geodetic and Earth-centred (ECEF) coordinates, and the radii of curvature."""

from __future__ import annotations

import math
from typing import Any

from geophysix._src._arrays import floating_inputs, like
from geophysix._src.earth.records import WGS84, Ellipsoid


_DEG = math.pi / 180.0


def prime_vertical_radius(
    lat: Any, *, earth: Ellipsoid = WGS84, degrees: bool = True
) -> Any:
    """The prime-vertical radius of curvature N(φ).

    N(φ) = a / √(1 − e² sin²φ)

    Args:
        lat: Geodetic latitude φ, degrees (radians with ``degrees=False``).
        earth: The body; WGS84 by default.
        degrees: Whether ``lat`` is in degrees.

    Returns:
        N, m, in ``lat``'s shape, namespace and dtype.

    Examples:
        >>> import numpy as np
        >>> from geophysix.earth import WGS84, prime_vertical_radius
        >>> bool(np.isclose(prime_vertical_radius(0.0), WGS84.semi_major))
        True
        >>> prime_vertical_radius(np.zeros(3, dtype=np.float32)).dtype
        dtype('float32')
    """
    xp, (lat,) = floating_inputs(lat, params=(earth.semi_major, earth.flattening))
    phi = lat * like(_DEG, lat, xp) if degrees else lat
    a = like(earth.semi_major, lat, xp)
    e2 = like(earth.eccentricity_squared, lat, xp)
    # N(φ) = a / √(1 − e² sin²φ)
    return a / xp.sqrt(1.0 - e2 * xp.sin(phi) ** 2)


def meridional_radius(
    lat: Any, *, earth: Ellipsoid = WGS84, degrees: bool = True
) -> Any:
    """The meridional radius of curvature M(φ).

    M(φ) = a (1 − e²) / (1 − e² sin²φ)^(3/2)

    Args:
        lat: Geodetic latitude φ, degrees (radians with ``degrees=False``).
        earth: The body; WGS84 by default.
        degrees: Whether ``lat`` is in degrees.

    Returns:
        M, m, in ``lat``'s shape, namespace and dtype.

    Examples:
        >>> import numpy as np
        >>> from geophysix.earth import meridional_radius, sphere
        >>> bool(np.isclose(meridional_radius(45.0, earth=sphere(1.0)), 1.0))
        True
    """
    xp, (lat,) = floating_inputs(lat, params=(earth.semi_major, earth.flattening))
    phi = lat * like(_DEG, lat, xp) if degrees else lat
    a = like(earth.semi_major, lat, xp)
    e2 = like(earth.eccentricity_squared, lat, xp)
    # M(φ) = a (1 − e²) / (1 − e² sin²φ)^(3/2)
    return a * (1.0 - e2) / (1.0 - e2 * xp.sin(phi) ** 2) ** 1.5


def geodetic_to_ecef(
    lon: Any,
    lat: Any,
    height: Any = 0.0,
    *,
    earth: Ellipsoid = WGS84,
    degrees: bool = True,
) -> tuple[Any, Any, Any]:
    """Earth-centred, Earth-fixed coordinates of a geodetic position.

    (X, Y, Z) = ((N + h) cos φ cos λ, (N + h) cos φ sin λ, (N (1 − e²) + h) sin φ)

    with N(φ) the prime-vertical radius of curvature.

    Args:
        lon: Longitude λ, degrees (radians with ``degrees=False``).
        lat: Geodetic latitude φ, degrees (radians with ``degrees=False``).
        height: Height h above the ellipsoid, m.
        earth: The body; WGS84 by default.
        degrees: Whether ``lon`` and ``lat`` are in degrees.

    Returns:
        X, Y and Z, m, broadcast to one shape. In float32 an ECEF coordinate
        of 6.4 × 10⁶ m has a spacing of 0.5 m, so sub-metre work needs float64.

    Examples:
        >>> import numpy as np
        >>> from geophysix.earth import WGS84, geodetic_to_ecef
        >>> x, y, z = geodetic_to_ecef(0.0, 0.0)
        >>> bool(np.allclose([x, y, z], [WGS84.semi_major, 0.0, 0.0]))
        True
        >>> x, y, z = geodetic_to_ecef(0.0, 90.0)
        >>> bool(np.isclose(z, WGS84.semi_minor))
        True
    """
    xp, (lon, lat, height) = floating_inputs(
        lon, lat, height, params=(earth.semi_major, earth.flattening)
    )
    deg = like(_DEG, lat, xp)
    lam, phi = (lon * deg, lat * deg) if degrees else (lon, lat)
    a = like(earth.semi_major, lat, xp)
    e2 = like(earth.eccentricity_squared, lat, xp)
    sin_phi, cos_phi = xp.sin(phi), xp.cos(phi)
    # N(φ) = a / √(1 − e² sin²φ)
    n = a / xp.sqrt(1.0 - e2 * sin_phi**2)
    # X, Y = (N + h) cos φ (cos λ, sin λ);  Z = (N (1 − e²) + h) sin φ
    x = (n + height) * cos_phi * xp.cos(lam)
    y = (n + height) * cos_phi * xp.sin(lam)
    z = (n * (1.0 - e2) + height) * sin_phi
    return x, y, z


def ecef_to_geodetic(
    x: Any,
    y: Any,
    z: Any,
    *,
    earth: Ellipsoid = WGS84,
    degrees: bool = True,
) -> tuple[Any, Any, Any]:
    """Geodetic position of an Earth-centred, Earth-fixed point.

    λ = atan2(Y, X), and φ from Bowring's method, two fixed steps:

    φ = atan2(Z + e'² b sin³β, p − e² a cos³β), p = √(X² + Y²), e'² = e² / (1 − e²)

    Algorithm:
        1. λ = atan2(Y, X) and p = √(X² + Y²).
        2. Parametric latitude β₀ = atan2(Z, (1 − f) p).
        3. φ from β, then β = atan2((1 − f) sin φ, cos φ); twice.
        4. h = p cos φ + Z sin φ − a √(1 − e² sin²φ), which holds at every
           latitude, the poles included.

    One step is accurate to well under a millimetre within ±10 km of the
    ellipsoid; the second covers orbital heights.

    Args:
        x: X, m.
        y: Y, m.
        z: Z, m.
        earth: The body; WGS84 by default.
        degrees: Whether to return longitude and latitude in degrees.

    Returns:
        Longitude λ, geodetic latitude φ and height h (m), broadcast to one
        shape; on the polar axis the longitude is 0.

    References:
        Bowring, B. R. (1976), Transformation from spatial to geographical
        coordinates, Survey Review 23 (181), 323-327.

    Examples:
        >>> import numpy as np
        >>> from geophysix.earth import ecef_to_geodetic, geodetic_to_ecef
        >>> lon, lat, h = ecef_to_geodetic(*geodetic_to_ecef(12.5, -33.0, 850.0))
        >>> bool(np.allclose([lon, lat, h], [12.5, -33.0, 850.0]))
        True
        >>> lon, lat, h = ecef_to_geodetic(0.0, 0.0, 6_400_000.0)
        >>> bool(np.allclose([lon, lat], [0.0, 90.0]))
        True
    """
    xp, (x, y, z) = floating_inputs(
        x, y, z, params=(earth.semi_major, earth.flattening)
    )
    a = like(earth.semi_major, x, xp)
    f = like(earth.flattening, x, xp)
    # e² = f (2 − f), b = a (1 − f), e'² = e² / (1 − e²)
    e2 = f * (2.0 - f)
    b = a * (1.0 - f)
    ep2 = e2 / (1.0 - e2)

    # --- longitude and distance from the axis ---
    p2 = x**2 + y**2
    off_axis = p2 > 0.0
    # double where: on the axis the unused branch would give a NaN gradient
    p = xp.where(off_axis, xp.sqrt(xp.where(off_axis, p2, 1.0)), 0.0)
    x_safe = xp.where(off_axis, x, 1.0)
    y_safe = xp.where(off_axis, y, 0.0)
    lam = xp.where(off_axis, xp.atan2(y_safe, x_safe), 0.0)

    # --- latitude, two Bowring steps ---
    beta = xp.atan2(z, (1.0 - f) * p)
    for _ in range(2):
        # φ = atan2(Z + e'² b sin³β, p − e² a cos³β)
        phi = xp.atan2(z + ep2 * b * xp.sin(beta) ** 3, p - e2 * a * xp.cos(beta) ** 3)
        # tan β = (1 − f) tan φ
        beta = xp.atan2((1.0 - f) * xp.sin(phi), xp.cos(phi))

    # --- height ---
    sin_phi = xp.sin(phi)
    # h = p cos φ + Z sin φ − a √(1 − e² sin²φ)
    h = p * xp.cos(phi) + z * sin_phi - a * xp.sqrt(1.0 - e2 * sin_phi**2)
    if degrees:
        deg = like(_DEG, x, xp)
        return lam / deg, phi / deg, h
    return lam, phi, h
