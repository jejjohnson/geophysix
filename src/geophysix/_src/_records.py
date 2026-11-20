"""Frozen records of physical constants, registered as JAX pytrees when JAX exists.

A record is a frozen dataclass of floats. Under JAX a float field compiles in
as a constant; a JAX scalar placed in a field is traced, which is how a
radius or a molar mass is fitted. Registration happens only when JAX is
importable, so ``import geophysix`` never imports JAX itself.
"""

from __future__ import annotations

import dataclasses
import numbers
from typing import Any, cast

import numpy as np


def register_record[T: type](cls: T, *, meta_fields: tuple[str, ...] = ()) -> T:
    """Register the frozen dataclass ``cls`` as a JAX pytree, if JAX is importable.

    Args:
        cls: A frozen dataclass.
        meta_fields: Fields that are static metadata (names, labels); every
            other field is a pytree leaf.

    Returns:
        ``cls`` itself, registered or not.
    """
    try:
        import jax
    # the NumPy-only install registers nothing
    except ImportError:
        return cls
    names = [f.name for f in dataclasses.fields(cast("Any", cls))]
    data = [n for n in names if n not in meta_fields]
    jax.tree_util.register_dataclass(
        cls, data_fields=data, meta_fields=list(meta_fields)
    )
    return cls


def is_concrete(value: Any) -> bool:
    """Whether ``value`` is a plain number, so it can be checked eagerly.

    JAX tracers, and the placeholder leaves JAX substitutes while it maps over
    a pytree, are not concrete: a record built from them skips its checks.
    """
    if isinstance(value, numbers.Real):
        return True
    return isinstance(value, np.ndarray | np.generic) and np.ndim(value) == 0
