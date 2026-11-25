"""The Earth's figure as a frozen record: an ellipsoid of revolution or a sphere."""

from __future__ import annotations

import dataclasses

from pipekit_array.arrays import register_record

from geophysix._src._records import is_concrete
from geophysix.exceptions import EarthError


#: WGS84's rotation rate Ω, s⁻¹.
WGS84_ROTATION_RATE = 7.292115e-5


@dataclasses.dataclass(frozen=True, slots=True)
class Ellipsoid:
    """An ellipsoid of revolution: semi-major axis a, flattening f, rotation Ω.

    e² = f (2 − f), b = a (1 − f), R₁ = (2a + b) / 3

    A sphere is the case f = 0 (see :func:`sphere`), so every function that
    takes an ``Ellipsoid`` works on both.

    Attributes:
        semi_major: Semi-major (equatorial) axis a, m; > 0.
        flattening: Flattening f = (a − b) / a, in [0, 1).
        rotation_rate: Rotation rate Ω, s⁻¹ (WGS84's by default).

    Raises:
        EarthError: If a ≤ 0, or f is outside [0, 1).

    Examples:
        >>> import math
        >>> from geophysix.earth import WGS84, Ellipsoid
        >>> math.isclose(WGS84.semi_minor, 6_356_752.314245, rel_tol=1e-12)
        True
        >>> math.isclose(WGS84.mean_radius, 6_371_008.7714, rel_tol=1e-11)
        True
        >>> Ellipsoid(6.4e6, 1.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        geophysix.exceptions.EarthError: flattening must be in [0, 1), got 1.5
    """

    semi_major: float
    flattening: float
    rotation_rate: float = WGS84_ROTATION_RATE

    def __post_init__(self) -> None:
        a, f = self.semi_major, self.flattening
        # tracers and JAX's placeholder leaves are checked when they are numbers
        if is_concrete(a) and not a > 0.0:
            msg = f"semi_major must be > 0, got {a}"
            raise EarthError(msg)
        if is_concrete(f) and not 0.0 <= f < 1.0:
            msg = f"flattening must be in [0, 1), got {f}"
            raise EarthError(msg)

    @property
    def eccentricity_squared(self) -> float:
        """The first eccentricity squared, e² = f (2 − f)."""
        return self.flattening * (2.0 - self.flattening)

    @property
    def semi_minor(self) -> float:
        """The semi-minor (polar) axis, b = a (1 − f), m."""
        return self.semi_major * (1.0 - self.flattening)

    @property
    def mean_radius(self) -> float:
        """The IUGG mean radius, R₁ = (2a + b) / 3, m."""
        return (2.0 * self.semi_major + self.semi_minor) / 3.0


register_record(Ellipsoid)


def sphere(radius: float, rotation_rate: float = WGS84_ROTATION_RATE) -> Ellipsoid:
    """A sphere of radius R: the ellipsoid with f = 0.

    Args:
        radius: Radius R, m.
        rotation_rate: Rotation rate Ω, s⁻¹ (WGS84's by default).

    Returns:
        The ellipsoid with semi-major axis R and flattening 0.

    Examples:
        >>> from geophysix.earth import sphere
        >>> sphere(6_371_000.0)
        Ellipsoid(semi_major=6371000.0, flattening=0.0, rotation_rate=7.292115e-05)
    """
    return Ellipsoid(semi_major=radius, flattening=0.0, rotation_rate=rotation_rate)


#: The World Geodetic System 1984 ellipsoid.
WGS84 = Ellipsoid(semi_major=6_378_137.0, flattening=1.0 / 298.257223563)
#: The Geodetic Reference System 1980 ellipsoid.
GRS80 = Ellipsoid(semi_major=6_378_137.0, flattening=1.0 / 298.257222101)
#: The default sphere: WGS84's IUGG mean radius R₁ = 6,371,008.77 m.
SPHERE = sphere(WGS84.mean_radius)
