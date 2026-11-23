"""Array dispatch: the namespace, and the device, that a call computes on.

A function computes in the namespace of its inputs: NumPy in gives NumPy out,
JAX in gives JAX out, and the same for every array-API library that
`array_api_compat` knows. Only functions of the array API standard are called
on a namespace. No array library but NumPy is imported here.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import array_api_compat
import array_api_compat.numpy as numpy_namespace
import numpy as np

from geophysix.exceptions import BackendError


__all__ = [
    "array_namespace",
    "device_of",
    "is_host",
    "place",
]

# Values that live on the host and join whatever library the others are from.
_HOST_TYPES = (bool, int, float, complex, np.ndarray, np.generic, list, tuple)


def is_host(value: Any) -> bool:
    """Say whether a value is ``None``, a Python number, a sequence or NumPy's.

    Examples:
        >>> is_host(1.0), is_host(None), is_host([1.0, 2.0])
        (True, True, True)
        >>> is_host(object())
        False
    """
    return value is None or isinstance(value, _HOST_TYPES)


def array_namespace(*values: Any) -> ModuleType:
    """The array namespace that a computation on ``values`` runs in.

    Python numbers, ``None``, sequences and NumPy arrays join the library of
    the other arrays; on their own they resolve to NumPy.

    Args:
        *values: Arguments of the function being dispatched.

    Returns:
        The namespace, such as ``array_api_compat.numpy`` or ``jax.numpy``.

    Raises:
        BackendError: If a value is not an array of a known library, or if two
            libraries other than NumPy meet in one call.

    Examples:
        >>> import numpy as np
        >>> array_namespace(np.zeros(3), 2.0, None).__name__
        'array_api_compat.numpy'
        >>> array_namespace("text")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        geophysix.exceptions.BackendError: str is not an array of a known library
    """
    found: dict[str, ModuleType] = {}
    for value in values:
        if is_host(value):
            continue
        try:
            namespace = array_api_compat.array_namespace(value)
        except TypeError as err:
            msg = f"{type(value).__name__} is not an array of a known library"
            raise BackendError(msg) from err
        found[namespace.__name__] = namespace
    if len(found) > 1:
        msg = f"arrays of several libraries in one call: {', '.join(sorted(found))}"
        raise BackendError(msg)
    return next(iter(found.values()), numpy_namespace)


def device_of(value: Any, xp: ModuleType) -> Any:
    """The device that new arrays join ``value`` on, or ``None`` for the default.

    A JAX tracer has no device, and an uncommitted JAX scalar joins an array
    on any device, so JAX is left to place its own arrays.

    Args:
        value: An array, or a value on the host.
        xp: The namespace of the call.

    Returns:
        The array's device; ``None`` for a host value and in JAX.

    Examples:
        >>> import numpy as np
        >>> device_of(np.zeros(2), np) is None
        True
    """
    if is_host(value) or array_api_compat.is_jax_namespace(xp):
        return None
    return array_api_compat.device(value)


def place(value: Any, xp: ModuleType, dtype: Any, device: Any) -> Any:
    """``value`` as an array of ``dtype``: cast if it is one already, else built.

    Args:
        value: An array of the namespace ``xp``, or a value on the host.
        xp: The namespace of the call.
        dtype: The dtype of the result.
        device: Where a host value goes, from `device_of`. An array stays on
            its own device.

    Returns:
        The array; ``value`` itself if nothing changes.

    Examples:
        >>> import numpy as np
        >>> place(0.1, np, np.float32, None).dtype
        dtype('float32')
    """
    if not is_host(value):
        # asarray would detach a torch tensor from its graph (torch 2.5, 2.10)
        return xp.astype(value, dtype, copy=False)
    if device is None:
        return xp.asarray(value, dtype=dtype)
    return xp.asarray(value, dtype=dtype, device=device)
