"""The horizontal wind's speed, its bearing and its components."""

from __future__ import annotations

import math
from typing import Any, Literal

from pipekit_array.arrays import floating_inputs, like
from pipekit_array.kernels import elementwise

from geophysix.exceptions import GeophysixError


Convention = Literal["from", "to"]

_DEG = math.pi / 180.0
_RAD = 180.0 / math.pi


def _check_convention(convention: str) -> None:
    if convention not in ("from", "to"):
        msg = f"convention must be 'from' or 'to', got {convention!r}"
        raise GeophysixError(msg)


@elementwise()
def wind_speed(u: Any, v: Any) -> Any:
    """The horizontal wind speed.

    |V| = √(u² + v²)

    Calm air (u = v = 0) has speed 0 and a gradient of 0, not ``nan``.

    Args:
        u: Eastward wind u, m s⁻¹.
        v: Northward wind v, m s⁻¹.

    Returns:
        |V|, m s⁻¹, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> from geophysix.wind import wind_speed
        >>> float(wind_speed(3.0, 4.0))
        5.0
    """
    xp, (u, v) = floating_inputs(u, v)
    # s² = u² + v²
    squared = u**2 + v**2
    calm = squared == 0.0
    # double where: the square root never sees 0, so its gradient stays finite
    safe = xp.where(calm, xp.ones_like(squared), squared)
    # |V| = √(u² + v²)
    return xp.where(calm, xp.zeros_like(squared), safe**0.5)


@elementwise()
def wind_direction(
    u: Any, v: Any, *, convention: Convention = "from", degrees: bool = True
) -> Any:
    """The wind's bearing, clockwise from north.

    θ_from = (270° − atan2(v, u) 180°/π) mod 360°,  θ_to = (θ_from + 180°) mod 360°

    ``"from"`` is the meteorological convention (CF ``wind_from_direction``):
    the bearing the wind blows from, so a westerly is 270°. ``"to"`` (CF
    ``wind_to_direction``) is the bearing it blows towards. Calm air has no
    direction; it returns 270° under ``"from"``, the value atan2(0, 0) = 0
    gives, with a gradient of 0.

    Args:
        u: Eastward wind u, m s⁻¹.
        v: Northward wind v, m s⁻¹.
        convention: ``"from"`` or ``"to"``.
        degrees: Whether to return degrees on [0°, 360°) (radians on [0, 2π)
            with ``degrees=False``).

    Returns:
        θ, degrees (or radians), broadcast over the inputs, in their namespace
        and dtype.

    Raises:
        GeophysixError: If ``convention`` is not ``"from"`` or ``"to"``.

    Examples:
        >>> from geophysix.wind import wind_direction
        >>> float(wind_direction(5.0, 0.0))
        270.0
        >>> float(wind_direction(0.0, -5.0))
        0.0
        >>> float(wind_direction(5.0, 0.0, convention="to"))
        90.0
    """
    _check_convention(convention)
    xp, (u, v) = floating_inputs(u, v)
    calm = (u == 0.0) & (v == 0.0)
    # double where: atan2 never sees (0, 0), so its gradient stays finite
    safe_u = xp.where(calm, xp.ones_like(u), u)
    safe_v = xp.where(calm, xp.zeros_like(v), v)
    angle = xp.atan2(safe_v, safe_u)
    if degrees:
        full_turn, three_quarters, half = 360.0, 270.0, 180.0
        # atan2 in degrees
        angle = angle * like(_RAD, angle, xp)
    else:
        full_turn, three_quarters, half = 2.0 * math.pi, 1.5 * math.pi, math.pi
    # θ_from = (270° − atan2(v, u)) mod 360°
    bearing = xp.remainder(three_quarters - angle, full_turn)
    if convention == "to":
        # θ_to = (θ_from + 180°) mod 360°
        bearing = xp.remainder(bearing + half, full_turn)
    return bearing


@elementwise(n_outputs=2)
def wind_components(
    speed: Any,
    direction: Any,
    *,
    convention: Convention = "from",
    degrees: bool = True,
) -> tuple[Any, Any]:
    """The eastward and northward wind from a speed and a bearing.

    u = −|V| sin θ_from,  v = −|V| cos θ_from

    the inverse of `wind_speed` and `wind_direction`. Under ``"to"`` the
    signs flip: u = |V| sin θ_to, v = |V| cos θ_to.

    Args:
        speed: Wind speed |V|, m s⁻¹.
        direction: Bearing θ clockwise from north, degrees (radians with
            ``degrees=False``).
        convention: Whether ``direction`` is a ``"from"`` or a ``"to"`` bearing.
        degrees: Whether ``direction`` is in degrees.

    Returns:
        (u, v), m s⁻¹, broadcast over the inputs, in their namespace and dtype.

    Raises:
        GeophysixError: If ``convention`` is not ``"from"`` or ``"to"``.

    Examples:
        >>> import numpy as np
        >>> from geophysix.wind import wind_components
        >>> u, v = wind_components(5.0, 270.0)
        >>> bool(np.isclose(u, 5.0)), bool(np.isclose(v, 0.0, atol=1e-12))
        (True, True)
    """
    _check_convention(convention)
    xp, (speed, direction) = floating_inputs(speed, direction)
    theta = direction * like(_DEG, direction, xp) if degrees else direction
    sign = like(-1.0 if convention == "from" else 1.0, speed, xp)
    # u = −|V| sin θ,  v = −|V| cos θ  (signs flip under "to")
    return sign * speed * xp.sin(theta), sign * speed * xp.cos(theta)
