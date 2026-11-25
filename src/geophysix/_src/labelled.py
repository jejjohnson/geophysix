"""Functions that change the length of a named dimension, on DataArrays.

Elementwise functions go through ``pipekit_array.kernels.elementwise``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from typing import Any

from pipekit_array.arrays import is_dataarray

from geophysix.exceptions import GeophysixError


__all__ = [
    "apply_along",
]


def apply_along(
    function: Callable[..., Any],
    arguments: Mapping[str, Any],
    core_dims: Mapping[str, tuple[str, ...]],
    *,
    output_dims: tuple[str, ...],
) -> Any:
    """Call a function that changes the length of named dimensions.

    The inputs must have equal coordinates where they share a dimension. The
    coordinates along the core dimensions are dropped from the result, because
    they no longer have its length.

    Args:
        function: Takes the arguments' data as keywords, each with its core
            dimensions last.
        arguments: The DataArrays, by argument name.
        core_dims: The core dimensions of each argument.
        output_dims: The core dimensions of the result, which come last.

    Returns:
        A DataArray without a name or attributes.

    Raises:
        GeophysixError: If an argument is not a DataArray, or lacks one of its
            core dimensions.

    Examples:
        >>> apply_along(abs, {"x": 1.0}, {"x": ("level",)}, output_dims=("level",))
        Traceback (most recent call last):
        geophysix.exceptions.GeophysixError: 'x' must be a DataArray too
    """
    for name, value in arguments.items():
        if not is_dataarray(value):
            msg = f"'{name}' must be a DataArray too"
            raise GeophysixError(msg)
        missing = [dim for dim in core_dims[name] if dim not in value.dims]
        if missing:
            msg = f"'{name}' has no dimension {missing[0]!r}, only {value.dims}"
            raise GeophysixError(msg)
    xarray = sys.modules["xarray"]
    # apply_ufunc does not align the dimensions it excludes
    arrays = xarray.align(*arguments.values(), join="exact", copy=False)

    def on_data(*data: Any) -> Any:
        return function(**dict(zip(arguments, data, strict=True)))

    result = xarray.apply_ufunc(
        on_data,
        *arrays,
        input_core_dims=[core_dims[name] for name in arguments],
        output_core_dims=[output_dims],
        exclude_dims={dim for dims in core_dims.values() for dim in dims},
        dask="allowed",
        # for the coordinates: False would drop their attributes too
        keep_attrs="drop_conflicts",
    )
    result = result.rename(None)
    result.attrs = {}
    return result
