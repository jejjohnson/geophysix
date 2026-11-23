"""The Coriolis parameter, its meridional gradient, and the β-plane."""

from __future__ import annotations

import math
from typing import Any

from pipekit_array.functions import elementwise, floating_inputs, like

from geophysix._src.earth.records import WGS84, Ellipsoid


_DEG = math.pi / 180.0


@elementwise()
def coriolis(lat: Any, *, earth: Ellipsoid = WGS84, degrees: bool = True) -> Any:
    """The Coriolis parameter at each latitude.

    f = 2 Ω sin φ

    Args:
        lat: Latitude φ, degrees (radians with ``degrees=False``).
        earth: The body, whose ``rotation_rate`` is Ω; WGS84 by default.
        degrees: Whether ``lat`` is in degrees.

    Returns:
        f, s⁻¹, in ``lat``'s shape, namespace and dtype.

    Examples:
        >>> import numpy as np
        >>> from geophysix.earth import WGS84
        >>> from geophysix.rotation import coriolis
        >>> bool(np.isclose(coriolis(90.0), 2.0 * WGS84.rotation_rate))
        True
        >>> float(coriolis(0.0))
        0.0
    """
    xp, (lat,) = floating_inputs(lat, params=(earth.rotation_rate,))
    phi = lat * like(_DEG, lat, xp) if degrees else lat
    omega = like(earth.rotation_rate, lat, xp)
    # f = 2 Ω sin φ
    return 2.0 * omega * xp.sin(phi)


@elementwise()
def beta(lat: Any, *, earth: Ellipsoid = WGS84, degrees: bool = True) -> Any:
    """The meridional gradient of the Coriolis parameter.

    β = ∂f/∂y = 2 Ω cos φ / M(φ),  M(φ) = a (1 − e²) / (1 − e² sin²φ)^(3/2)

    with y the northward distance, so dy = M(φ) dφ; on a sphere M = R and
    β = 2 Ω cos φ / R.

    Args:
        lat: Latitude φ, degrees (radians with ``degrees=False``).
        earth: The body; WGS84 by default.
        degrees: Whether ``lat`` is in degrees.

    Returns:
        β, m⁻¹ s⁻¹, in ``lat``'s shape, namespace and dtype.

    Examples:
        >>> import numpy as np
        >>> from geophysix.earth import sphere
        >>> from geophysix.rotation import beta
        >>> body = sphere(6.4e6, rotation_rate=1e-4)
        >>> bool(np.isclose(beta(0.0, earth=body), 2e-4 / 6.4e6))
        True
    """
    xp, (lat,) = floating_inputs(
        lat, params=(earth.semi_major, earth.flattening, earth.rotation_rate)
    )
    phi = lat * like(_DEG, lat, xp) if degrees else lat
    omega = like(earth.rotation_rate, lat, xp)
    a = like(earth.semi_major, lat, xp)
    e2 = like(earth.eccentricity_squared, lat, xp)
    # M(φ) = a (1 − e²) / (1 − e² sin²φ)^(3/2)
    meridional = a * (1.0 - e2) / (1.0 - e2 * xp.sin(phi) ** 2) ** 1.5
    # β = 2 Ω cos φ / M(φ)
    return 2.0 * omega * xp.cos(phi) / meridional


@elementwise()
def beta_plane(y: Any, *, f0: Any, beta: Any, y0: Any) -> Any:
    """The Coriolis parameter on a β-plane.

    f(y) = f₀ + β (y − y₀)

    The reference ``y0`` has no default: models disagree on it (the mean or
    the midpoint of the domain), and the choice moves f on an uneven grid.

    Args:
        y: Northward coordinate y, m.
        f0: Coriolis parameter f₀ at ``y0``, s⁻¹.
        beta: Its gradient β, m⁻¹ s⁻¹.
        y0: Reference coordinate y₀, m.

    Returns:
        f(y), s⁻¹, in ``y``'s shape, namespace and dtype.

    Examples:
        >>> import numpy as np
        >>> from geophysix.rotation import beta_plane
        >>> f = beta_plane(np.array([0.0, 1e5]), f0=1e-4, beta=2e-11, y0=0.0)
        >>> bool(np.allclose(f, [1.0e-4, 1.02e-4]))
        True
    """
    xp, (y,) = floating_inputs(y, params=(f0, beta, y0))
    # f = f₀ + β (y − y₀)
    return like(f0, y, xp) + like(beta, y, xp) * (y - like(y0, y, xp))
