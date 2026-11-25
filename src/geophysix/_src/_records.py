"""Which fields of a record can be checked eagerly.

A record is a frozen dataclass of floats, registered as a JAX pytree by
`pipekit_array.records.register_record`. Under JAX a float field compiles in as
a constant; a JAX scalar placed in a field is traced, which is how a radius or a
molar mass is fitted.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np


def is_concrete(value: Any) -> bool:
    """Whether ``value`` is a plain number, so it can be checked eagerly.

    JAX tracers, and the placeholder leaves JAX substitutes while it maps over
    a pytree, are not concrete: a record built from them skips its checks.
    """
    if isinstance(value, numbers.Real):
        return True
    return isinstance(value, np.ndarray | np.generic) and np.ndim(value) == 0
