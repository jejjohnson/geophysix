"""Array-namespace dispatch for the NumPy and JAX backends.

A function computes in the namespace of its first array argument: NumPy in
gives NumPy out, a JAX array gives a JAX array. JAX is never imported at
module level; it is looked up only when a value that came from JAX is seen.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np


try:
    _compat: Any = importlib.import_module("array_api_compat")
except ImportError:
    _compat = None


__all__ = [
    "array_namespace",
    "is_jax",
]

# Values that always resolve to NumPy: Python scalars, None and NumPy objects.
_NUMPY_TYPES = (bool, int, float, complex, np.ndarray, np.generic)


def array_namespace(*values: Any) -> ModuleType:
    """Return the array namespace that a computation on ``values`` runs in.

    Python scalars, ``None`` and NumPy arrays resolve to NumPy. When
    ``array_api_compat`` is importable, it decides for every other array.
    Otherwise the first value with an ``__array_namespace__`` method whose
    namespace is not NumPy's wins, and a JAX array or tracer resolves to
    ``jax.numpy`` even if it lacks that method.

    Args:
        *values: Arguments of the function being dispatched; any mix of
            arrays, Python scalars and ``None``.

    Returns:
        The array namespace, such as ``numpy`` or ``jax.numpy``.

    Examples:
        >>> import numpy as np
        >>> array_namespace(1.0, None) is np
        True
        >>> array_namespace(np.zeros(3), 2.0) is np
        True
    """
    for value in values:
        if value is None or isinstance(value, _NUMPY_TYPES):
            continue
        if _compat is not None:
            try:
                return _compat.array_namespace(value)
            except TypeError:
                continue
        get_namespace = getattr(value, "__array_namespace__", None)
        if get_namespace is not None:
            namespace = get_namespace()
            if namespace is not np:
                return namespace
        elif type(value).__module__.split(".")[0] in ("jax", "jaxlib"):
            return importlib.import_module("jax.numpy")
    return np


def is_jax(xp: ModuleType) -> bool:
    """Say whether an array namespace is JAX's.

    Args:
        xp: An array namespace, as returned by `array_namespace`.

    Returns:
        ``True`` for ``jax.numpy`` and its submodules, ``False`` otherwise.

    Examples:
        >>> import numpy as np
        >>> is_jax(np)
        False
    """
    return xp.__name__.split(".")[0] == "jax"
