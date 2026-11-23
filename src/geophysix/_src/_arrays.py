"""Shared array plumbing: one namespace, one floating dtype and one device per call."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from geophysix._src.backend import array_namespace, device_of, is_host, place
from geophysix.exceptions import GeophysixError


__all__ = [
    "floating_inputs",
    "like",
]


def floating_inputs(
    *values: Any, params: tuple[Any, ...] = ()
) -> tuple[ModuleType, list[Any]]:
    """The namespace of a call and its array inputs in one floating dtype.

    The dtype is the common one of the floating arrays (float32 stays
    float32). Python numbers and integer or boolean arrays adopt it; with no
    floating array, everything takes the namespace's default float. Values on
    the host join the device of the first array that is not.

    Args:
        *values: The array inputs: arrays, sequences or Python numbers.
        params: Record fields, Python floats or JAX scalars. They help to
            choose the namespace and the device, so a traced JAX scalar in a
            record selects JAX, and never decide the dtype.

    Returns:
        The namespace, and ``values`` as arrays of one floating dtype.

    Raises:
        BackendError: As `array_namespace`.
        GeophysixError: If an input is complex, a date or not numeric.

    Examples:
        >>> import numpy as np
        >>> xp, (x, y) = floating_inputs(np.zeros(2, dtype=np.float32), 1.5)
        >>> x.dtype, y.dtype
        (dtype('float32'), dtype('float32'))
        >>> xp, (counts,) = floating_inputs(np.asarray([1, 2], dtype=np.uint16))
        >>> counts.dtype
        dtype('float64')
    """
    xp = array_namespace(*values, *params)
    devices = (device_of(value, xp) for value in (*values, *params))
    device = next((found for found in devices if found is not None), None)
    floating = []
    for value in values:
        if _is_number(value):
            continue
        dtype = xp.asarray(value).dtype if is_host(value) else value.dtype
        if xp.isdtype(dtype, "real floating"):
            floating.append(dtype)
        elif not xp.isdtype(dtype, ("bool", "integral")):
            msg = f"inputs must be real numbers, got dtype {dtype}"
            raise GeophysixError(msg)
    dtype = xp.result_type(*floating) if floating else xp.asarray(0.0).dtype
    return xp, [place(value, xp, dtype, device) for value in values]


def like(value: Any, ref: Any, xp: ModuleType) -> Any:
    """``value`` (a number or a 0-d array) in ``ref``'s dtype, on its device.

    Examples:
        >>> import numpy as np
        >>> like(0.1, np.zeros(2, dtype=np.float32), np).dtype
        dtype('float32')
    """
    return place(value, xp, ref.dtype, device_of(ref, xp))


def _is_number(value: Any) -> bool:
    """A Python real number, which adopts the dtype of the arrays.

    A NumPy scalar carries its own dtype, though ``np.float64`` is a float.
    """
    return isinstance(value, (bool, int, float)) and not isinstance(value, np.generic)
