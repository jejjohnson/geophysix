"""The area of latitude-longitude cells on the ellipsoid and on the sphere."""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from typing import Any

from geophysix._src._arrays import floating_inputs, like
from geophysix._src.earth.records import WGS84, Ellipsoid
from geophysix._src.labelled import apply_along, is_dataarray
from geophysix.exceptions import GeophysixError


_DEG = math.pi / 180.0
# below this e² sin²φ, artanh(e s) / (e s) is its series 1 + w/3 + w²/5
_SERIES_BELOW = 1e-8


def _authalic_q(sin_phi: Any, e2: Any, xp: Any) -> Any:
    """The authalic function q(φ) = s / (1 − e² s²) + artanh(e s) / e, s = sin φ.

    Written through w = e² s², so it is smooth in e² and has the sphere's
    limit q = 2s at e = 0, gradient included.
    """
    w = e2 * sin_phi**2
    small = w < _SERIES_BELOW
    # double where: √w and artanh(√w) / √w are evaluated only away from 0
    w_safe = xp.where(small, 0.5, w)
    root = xp.sqrt(w_safe)
    exact = xp.atanh(root) / root
    # artanh(z) / z = 1 + z²/3 + z⁴/5 + …
    series = 1.0 + w / 3.0 + w**2 / 5.0
    ratio = xp.where(small, series, exact)
    # q = s / (1 − e² s²) + s · artanh(e s) / (e s)
    return sin_phi / (1.0 - w) + sin_phi * ratio


def cell_area(
    lat_edges: Any,
    lon_edges: Any,
    *,
    dims: tuple[str, str] | None = None,
    earth: Ellipsoid = WGS84,
    degrees: bool = True,
) -> Any:
    """The area of each cell of a latitude-longitude grid on the ellipsoid.

    Aᵢⱼ = (b² Δλⱼ / 2) |q(φᵢ₊₁) − q(φᵢ)|,  q(φ) = s / (1 − e² s²) + artanh(e s) / e

    with s = sin φ, b the semi-minor axis and e the eccentricity. On a
    sphere (e = 0) this is Aᵢⱼ = R² Δλⱼ |sin φᵢ₊₁ − sin φᵢ|.

    Algorithm:
        1. Evaluate q at each edge latitude (Ny + 1 values).
        2. Difference along latitude and along longitude.
        3. Multiply by b² / 2; the outer product gives the grid.

    Args:
        lat_edges: The Ny + 1 latitude edges φ, degrees (radians with
            ``degrees=False``), monotonic.
        lon_edges: The Nx + 1 longitude edges λ, monotonic.
        dims: The names of the result's latitude and longitude dimensions;
            needed with DataArrays, and only there. An edge dimension has one
            entry more than the cells, so its name is not reused.
        earth: The body; WGS84 by default.
        degrees: Whether the edges are in degrees.

    Returns:
        The areas, m², of shape (Ny, Nx), in the edges' namespace and dtype.
        With DataArrays a DataArray on ``dims``.

    Raises:
        GeophysixError: If DataArrays come without ``dims``, or ``dims``
            without DataArrays; if the edges are not two 1-D DataArrays with
            different dimensions.

    References:
        Snyder, J. P. (1987), Map Projections: A Working Manual, USGS
        Professional Paper 1395, chapter 3 (the authalic latitude).

    Examples:
        >>> import math
        >>> import numpy as np
        >>> from geophysix.earth import cell_area, sphere
        >>> area = cell_area(
        ...     np.linspace(-90, 90, 19), np.linspace(0, 360, 37), earth=sphere(1.0)
        ... )
        >>> area.shape
        (18, 36)
        >>> math.isclose(float(area.sum()), 4.0 * math.pi, rel_tol=1e-12)
        True
    """
    edges = {"lat_edges": lat_edges, "lon_edges": lon_edges}
    if any(map(is_dataarray, edges.values())):
        return _labelled_cell_area(edges, dims, earth=earth, degrees=degrees)
    if dims is not None:
        msg = f"dims={dims!r} names dimensions of DataArrays; plain arrays have none"
        raise GeophysixError(msg)
    xp, (lat_edges, lon_edges) = floating_inputs(
        lat_edges, lon_edges, params=(earth.semi_major, earth.flattening)
    )
    deg = like(_DEG, lat_edges, xp)
    phi, lam = (lat_edges * deg, lon_edges * deg) if degrees else (lat_edges, lon_edges)
    f = like(earth.flattening, lat_edges, xp)
    # b = a (1 − f), e² = f (2 − f)
    b = like(earth.semi_major, lat_edges, xp) * (1.0 - f)
    e2 = f * (2.0 - f)
    q = _authalic_q(xp.sin(phi), e2, xp)
    dq = xp.abs(q[1:] - q[:-1])
    dlam = xp.abs(lam[1:] - lam[:-1])
    # Aᵢⱼ = (b² / 2) Δqᵢ Δλⱼ
    return 0.5 * b**2 * dq[:, None] * dlam[None, :]


def _labelled_cell_area(
    edges: Mapping[str, Any], dims: tuple[str, str] | None, **options: Any
) -> Any:
    """`cell_area` of two 1-D DataArrays, on the dimensions ``dims``."""
    if dims is None or isinstance(dims, str) or len(set(dims)) != 2:
        msg = (
            "DataArrays need 'dims', two different names for the result's "
            f"latitude and longitude, got {dims!r}"
        )
        raise GeophysixError(msg)
    core_dims = {}
    for name, value in edges.items():
        if is_dataarray(value):
            if value.ndim != 1:
                msg = f"'{name}' must have one dimension, got {value.dims}"
                raise GeophysixError(msg)
            core_dims[name] = value.dims
    if len(set(core_dims.values())) < len(core_dims):
        msg = f"the edges share the dimension {core_dims['lat_edges'][0]!r}"
        raise GeophysixError(msg)
    return apply_along(
        functools.partial(cell_area, **options),
        edges,
        {name: core_dims.get(name, ()) for name in edges},
        output_dims=tuple(dims),
    )
