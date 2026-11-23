"""Labelled arrays: a DataArray in gives a DataArray out, aligned by dimension name."""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from geophysix._src.backend import array_namespace
from geophysix.exceptions import GeophysixError


__all__ = [
    "apply_along",
    "elementwise",
    "is_dataarray",
]


def is_dataarray(value: Any) -> bool:
    """Say whether a value is an xarray DataArray, without importing xarray.

    A DataArray can only exist once xarray is imported.

    Examples:
        >>> is_dataarray(1.0)
        False
    """
    xarray = sys.modules.get("xarray")
    return xarray is not None and isinstance(value, xarray.DataArray)


def elementwise[**P, R](
    *, n_outputs: int = 1
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Let an elementwise function take and return DataArrays.

    Algorithm:
        1. With no DataArray among the arguments, call the function as it is.
        2. Otherwise hand every DataArray to ``xarray.apply_ufunc``, which
           aligns them by dimension name and calls the function on their data.
           The other arguments are passed through.
        3. Broadcast every result to the dimensions of the inputs together.
        4. Return DataArrays with the inputs' coordinates, and without a name
           or attributes, because the quantity and its unit have changed.

    An array without dimension names cannot be aligned with one that has
    them, so it is refused, unless it has no dimensions.

    Args:
        n_outputs: How many arrays the function returns; above 1, a tuple.

    Returns:
        The decorator.

    Raises:
        GeophysixError: (the decorated function) if a DataArray meets an
            array with dimensions but no names.

    Examples:
        >>> import numpy as np
        >>> @elementwise()
        ... def twice(x):
        ...     return 2.0 * x
        >>> twice(np.asarray([1.0, 2.0])).tolist()
        [2.0, 4.0]
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not any(map(is_dataarray, (*args, *kwargs.values()))):
                return function(*args, **kwargs)
            return _apply(function, signature.bind(*args, **kwargs), n_outputs)

        return wrapper

    return decorate


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
        keep_attrs=False,
    )
    return result.rename(None)


def _apply(
    function: Callable[..., Any], bound: inspect.BoundArguments, n_outputs: int
) -> Any:
    """Call ``function`` through ``xarray.apply_ufunc`` on its labelled arguments."""
    xarray = sys.modules["xarray"]
    labelled = [name for name, v in bound.arguments.items() if is_dataarray(v)]
    for name, value in bound.arguments.items():
        if name not in labelled and _has_dimensions(value):
            msg = (
                f"'{name}' has dimensions but no names, and cannot be aligned with "
                f"the DataArray '{labelled[0]}'; make it a DataArray too"
            )
            raise GeophysixError(msg)

    def on_data(*data: Any) -> Any:
        bound.arguments.update(zip(labelled, data, strict=True))
        shape = np.broadcast_shapes(*(array.shape for array in data))
        values = function(*bound.args, **bound.kwargs)
        if n_outputs == 1:
            return _broadcast(values, shape)
        return tuple(_broadcast(value, shape) for value in values)

    result = xarray.apply_ufunc(
        on_data,
        *(bound.arguments[name] for name in labelled),
        output_core_dims=[()] * n_outputs,
        dask="allowed",
        keep_attrs=False,
    )
    if n_outputs == 1:
        return result.rename(None)
    return tuple(array.rename(None) for array in result)


def _broadcast(array: Any, shape: tuple[int, ...]) -> Any:
    """``array`` in the shape of the inputs together, as ``apply_ufunc`` expects.

    An output may not depend on every input: z of `geodetic_to_ecef` does not
    depend on the longitude.
    """
    if array.shape == shape:
        return array
    return array_namespace(array).broadcast_to(array, shape)


def _has_dimensions(value: Any) -> bool:
    """Whether a value is an array or a sequence, rather than a number or an option."""
    if isinstance(value, (list, tuple)):
        return True
    return getattr(value, "ndim", 0) > 0
