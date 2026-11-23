"""Exception hierarchy for `geophysix`.

Every error raised by this package derives from
[`GeophysixError`][geophysix.GeophysixError], so a caller can trap the whole
package with a single ``except`` clause while still being able to discriminate
between failure modes when it matters.

The exception is the array backend, which is pipekit-array's: an input that is
no array of one known library raises pipekit's `BackendError`, and an input
that is not real, or an unlabelled array next to a labelled one, raises a plain
`ValueError`.

Both are a `ValueError`, as the base class is, so code written against the
standard library keeps working without importing anything from this package.
"""

from __future__ import annotations

from pipekit_array.functions import BackendError


__all__ = [
    "BackendError",
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
