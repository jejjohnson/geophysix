"""Shared array plumbing: one namespace and one floating dtype per call."""

from __future__ import annotations

from typing import Any

from geophysix._src.backend import array_namespace


def floating_inputs(
    *values: Any, params: tuple[Any, ...] = ()
) -> tuple[Any, list[Any]]:
    """The namespace of a call and its array inputs in one floating dtype.

    The dtype is that of the array inputs (float32 stays float32); Python
    numbers among ``values`` adopt it, and integer arrays become the
    namespace's default float. ``params`` (record fields) take part in
    choosing the namespace, so a traced JAX scalar in a record selects JAX,
    but they never decide the dtype.

    Args:
        *values: Arrays or Python numbers.
        params: Record fields, Python floats or JAX scalars.

    Returns:
        The namespace, and ``values`` as arrays of one floating dtype.
    """
    xp = array_namespace(*values, *params)
    typed = [v for v in values if hasattr(v, "dtype")]
    if typed:
        dtype = xp.result_type(*(xp.asarray(v) for v in typed))
        if not xp.isdtype(dtype, "real floating"):
            dtype = xp.asarray(0.0).dtype
    else:
        dtype = xp.asarray(0.0).dtype
    return xp, [xp.asarray(v, dtype=dtype) for v in values]


def like(value: Any, ref: Any, xp: Any) -> Any:
    """``value`` (a number or a JAX scalar) as an array in ``ref``'s dtype."""
    return xp.asarray(value, dtype=ref.dtype)
