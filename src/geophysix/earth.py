"""The Earth's figure: ellipsoids and spheres, as frozen records.

``WGS84`` and ``GRS80`` are the reference ellipsoids; ``SPHERE`` is the
sphere of WGS84's IUGG mean radius R₁ = 6,371,008.77 m, and ``sphere(R)``
builds another. A function that needs the Earth takes the record as a keyword
argument, ``earth=``, with one of these as its default.

On them: geodetic ↔ Earth-centred (ECEF) coordinates and the radii of
curvature N(φ) and M(φ).
"""

from __future__ import annotations

from geophysix._src.earth.geodetic import (
    ecef_to_geodetic,
    geodetic_to_ecef,
    meridional_radius,
    prime_vertical_radius,
)
from geophysix._src.earth.records import GRS80, SPHERE, WGS84, Ellipsoid, sphere


__all__ = [
    "Ellipsoid",
    "GRS80",
    "SPHERE",
    "WGS84",
    "ecef_to_geodetic",
    "geodetic_to_ecef",
    "meridional_radius",
    "prime_vertical_radius",
    "sphere",
]
