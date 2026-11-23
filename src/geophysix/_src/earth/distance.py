"""Great-circle distance and initial azimuth on a sphere."""

from __future__ import annotations

import math
from typing import Any

from pipekit_array.functions import elementwise, floating_inputs, like

from geophysix._src._records import is_concrete
from geophysix._src.earth.records import SPHERE, Ellipsoid
from geophysix.exceptions import EarthError


_DEG = math.pi / 180.0


@elementwise()
def great_circle_distance(
    lon1: Any,
    lat1: Any,
    lon2: Any,
    lat2: Any,
    *,
    earth: Ellipsoid = SPHERE,
    degrees: bool = True,
) -> Any:
    """The great-circle distance between two points on a sphere of radius R.

    d = R atan2(√(n_x² + n_y²), cos φ₁ cos φ₂ cos Δλ + sin φ₁ sin φ₂)

    with n_x = cos φ₂ sin Δλ and n_y = cos φ₁ sin φ₂ − sin φ₁ cos φ₂ cos Δλ:
    the spherical case of Vincenty's formula, well conditioned from
    coincident to antipodal points, where the haversine loses about
    1e-8 rad.

    Args:
        lon1: Longitude λ₁ of the first point, degrees (radians with
            ``degrees=False``).
        lat1: Latitude φ₁ of the first point.
        lon2: Longitude λ₂ of the second point.
        lat2: Latitude φ₂ of the second point.
        earth: A sphere (flattening 0); the IUGG mean-radius sphere by default.
        degrees: Whether the angles are in degrees.

    Returns:
        d, m, broadcast over the inputs, in their namespace and dtype.

    References:
        Vincenty, T. (1975), Direct and inverse solutions of geodesics on the
        ellipsoid with application of nested equations, Survey Review 23 (176),
        88-93.

    Raises:
        EarthError: If ``earth`` is not a sphere; ellipsoidal geodesics are
            not provided (#20).

    Examples:
        >>> import math
        >>> from geophysix.earth import great_circle_distance, sphere
        >>> math.isclose(
        ...     float(great_circle_distance(0, 0, 90, 0, earth=sphere(1.0))),
        ...     math.pi / 2,
        ...     rel_tol=1e-15,
        ... )
        True
        >>> float(great_circle_distance(10.0, 20.0, 10.0, 20.0))
        0.0
    """
    if is_concrete(earth.flattening) and earth.flattening != 0.0:
        msg = (
            "great_circle_distance needs a sphere (flattening 0), got flattening"
            f" {earth.flattening}; ellipsoidal geodesics are not provided yet"
        )
        raise EarthError(msg)
    xp, (lon1, lat1, lon2, lat2) = floating_inputs(
        lon1, lat1, lon2, lat2, params=(earth.semi_major,)
    )
    if degrees:
        deg = like(_DEG, lat1, xp)
        lon1, lat1, lon2, lat2 = lon1 * deg, lat1 * deg, lon2 * deg, lat2 * deg
    radius = like(earth.semi_major, lat1, xp)
    dlam = lon2 - lon1
    sin1, cos1 = xp.sin(lat1), xp.cos(lat1)
    sin2, cos2 = xp.sin(lat2), xp.cos(lat2)
    # n_x = cos φ₂ sin Δλ;  n_y = cos φ₁ sin φ₂ − sin φ₁ cos φ₂ cos Δλ
    n_x = cos2 * xp.sin(dlam)
    n_y = cos1 * sin2 - sin1 * cos2 * xp.cos(dlam)
    # cos σ = sin φ₁ sin φ₂ + cos φ₁ cos φ₂ cos Δλ
    cos_sigma = sin1 * sin2 + cos1 * cos2 * xp.cos(dlam)
    n2 = n_x**2 + n_y**2
    # double where: √· has an infinite slope at 0 (coincident and antipodal points)
    apart = n2 > 0.0
    sin_sigma = xp.where(apart, xp.sqrt(xp.where(apart, n2, 1.0)), 0.0)
    # d = R σ, σ = atan2(sin σ, cos σ)
    return radius * xp.atan2(sin_sigma, cos_sigma)


@elementwise()
def initial_azimuth(
    lon1: Any, lat1: Any, lon2: Any, lat2: Any, *, degrees: bool = True
) -> Any:
    """The initial bearing of the great circle from the first point to the second.

    α₁ = atan2(sin Δλ cos φ₂, cos φ₁ sin φ₂ − sin φ₁ cos φ₂ cos Δλ)

    measured clockwise from north, in (−180°, 180°]. It does not depend on
    the sphere's radius.

    Args:
        lon1: Longitude λ₁ of the first point, degrees (radians with
            ``degrees=False``).
        lat1: Latitude φ₁ of the first point.
        lon2: Longitude λ₂ of the second point.
        lat2: Latitude φ₂ of the second point.
        degrees: Whether the angles, in and out, are in degrees.

    Returns:
        α₁, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> import numpy as np
        >>> from geophysix.earth import initial_azimuth
        >>> bool(np.isclose(initial_azimuth(0.0, 0.0, 10.0, 0.0), 90.0))
        True
        >>> bool(np.isclose(initial_azimuth(0.0, 0.0, 0.0, 10.0), 0.0))
        True
    """
    xp, (lon1, lat1, lon2, lat2) = floating_inputs(lon1, lat1, lon2, lat2)
    deg = like(_DEG, lat1, xp)
    if degrees:
        lon1, lat1, lon2, lat2 = lon1 * deg, lat1 * deg, lon2 * deg, lat2 * deg
    dlam = lon2 - lon1
    # α₁ = atan2(sin Δλ cos φ₂, cos φ₁ sin φ₂ − sin φ₁ cos φ₂ cos Δλ)
    alpha = xp.atan2(
        xp.sin(dlam) * xp.cos(lat2),
        xp.cos(lat1) * xp.sin(lat2) - xp.sin(lat1) * xp.cos(lat2) * xp.cos(dlam),
    )
    return alpha / deg if degrees else alpha
