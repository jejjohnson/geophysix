"""Exception hierarchy for `geophysix`.

Every error raised by this package derives from
[`GeophysixError`][geophysix.GeophysixError], so a caller can trap the whole
package with a single ``except`` clause while still being able to discriminate
between failure modes when it matters.

The base class is a `ValueError`, so code written against the standard library
keeps working without importing anything from this package.
"""

from __future__ import annotations


__all__ = [
    "EarthError",
    "GeophysixError",
    "RangeError",
    "UnitError",
]


class GeophysixError(ValueError):
    """Base class for every exception raised by ``geophysix``.

    Examples:
        >>> from geophysix import GeophysixError, RangeError
        >>> issubclass(RangeError, GeophysixError)
        True
        >>> issubclass(GeophysixError, ValueError)
        True
    """


class EarthError(GeophysixError):
    """Raised for a bad body, such as an unknown ellipsoid or sphere name."""


class UnitError(GeophysixError):
    """Raised for an unknown unit."""


class RangeError(GeophysixError):
    """Raised for a latitude out of range."""
