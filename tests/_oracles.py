"""Reference oracles for tests: pyproj and gsw, each skipped when missing.

Every helper calls ``pytest.importorskip`` itself, so a test that uses an
oracle skips, rather than errors, in an environment without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


__all__ = [
    "ecef_from_geodetic",
    "geod_inverse",
    "gsw_module",
    "wgs84_geod",
    "wgs84_transformer",
]


def wgs84_transformer() -> Any:
    """Return the pyproj transformer from WGS 84 geodetic to geocentric.

    Returns:
        A ``pyproj.Transformer`` from EPSG:4979 (longitude, latitude, height)
        to EPSG:4978 (ECEF x, y, z in metres), with ``always_xy=True``.
    """
    pyproj = pytest.importorskip("pyproj")
    return pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def wgs84_geod() -> Any:
    """Return the pyproj geodesic calculator on the WGS 84 ellipsoid.

    Returns:
        A ``pyproj.Geod`` with ``ellps="WGS84"``.
    """
    pyproj = pytest.importorskip("pyproj")
    return pyproj.Geod(ellps="WGS84")


def ecef_from_geodetic(
    lon: Any, lat: Any, height: Any = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic coordinates to ECEF with pyproj.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        height: Height above the ellipsoid in metres.

    Returns:
        The ECEF coordinates ``(x, y, z)`` in metres.
    """
    x, y, z = wgs84_transformer().transform(lon, lat, height)
    return np.asarray(x), np.asarray(y), np.asarray(z)


def geod_inverse(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> np.ndarray:
    """Return the WGS 84 geodesic distance between two points with pyproj.

    Args:
        lon1: Longitude of the first point in degrees.
        lat1: Latitude of the first point in degrees.
        lon2: Longitude of the second point in degrees.
        lat2: Latitude of the second point in degrees.

    Returns:
        The distance in metres.
    """
    _, _, distance = wgs84_geod().inv(lon1, lat1, lon2, lat2)
    return np.asarray(distance)


def gsw_module() -> Any:
    """Return the ``gsw`` module (TEOS-10), skipping when it is missing.

    Returns:
        The imported ``gsw`` module.
    """
    return pytest.importorskip("gsw")
