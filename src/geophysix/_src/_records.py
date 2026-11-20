"""Frozen records of physical constants, registered as JAX pytrees once JAX loads.

A record is a frozen dataclass of floats. Under JAX a float field compiles in
as a constant; a JAX scalar placed in a field is traced, which is how a
radius or a molar mass is fitted.

``import geophysix`` never imports JAX, even when it is installed: libraries
that promise a JAX-free import (geoarray, geoops, xrops) import geophysix. So
a record is registered at once if JAX is already loaded, and otherwise the
moment something else imports JAX, by a finder on ``sys.meta_path`` that
runs the registrations after the ``jax`` package finishes loading.
"""

from __future__ import annotations

import dataclasses
import importlib.abc
import importlib.machinery
import importlib.util
import numbers
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np


# records waiting for JAX: (class, meta_fields)
_PENDING: list[tuple[type, tuple[str, ...]]] = []


def _register(jax: ModuleType, cls: type, meta_fields: tuple[str, ...]) -> None:
    """Register one frozen dataclass with ``jax.tree_util``."""
    names = [f.name for f in dataclasses.fields(cast("Any", cls))]
    data = [n for n in names if n not in meta_fields]
    jax.tree_util.register_dataclass(
        cls, data_fields=data, meta_fields=list(meta_fields)
    )


def _register_pending(jax: ModuleType) -> None:
    """Register every record that was defined before JAX was loaded."""
    while _PENDING:
        cls, meta_fields = _PENDING.pop(0)
        _register(jax, cls, meta_fields)


class _AfterJaxLoads(importlib.abc.Loader):
    """Wrap the ``jax`` loader so the pending records register once it has run."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        _register_pending(module)


class _JaxImportHook(importlib.abc.MetaPathFinder):
    """Finds nothing itself; when ``jax`` is imported, wraps its loader."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname != "jax":
            return None
        # let the other finders locate jax, then remove this hook for good
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec("jax")
        if spec is not None and spec.loader is not None:
            spec.loader = _AfterJaxLoads(spec.loader)
        return spec


def register_record[T: type](cls: T, *, meta_fields: tuple[str, ...] = ()) -> T:
    """Register the frozen dataclass ``cls`` as a JAX pytree, now or once JAX loads.

    Never imports JAX: with JAX already in ``sys.modules`` the record is
    registered at once; otherwise it is registered when something first
    imports ``jax``. Without JAX installed it is simply never registered.

    Args:
        cls: A frozen dataclass.
        meta_fields: Fields that are static metadata (names, labels); every
            other field is a pytree leaf.

    Returns:
        ``cls`` itself.
    """
    jax = sys.modules.get("jax")
    if jax is not None and hasattr(jax, "tree_util"):
        _register(jax, cls, meta_fields)
        return cls
    _PENDING.append((cls, meta_fields))
    if not any(isinstance(finder, _JaxImportHook) for finder in sys.meta_path):
        sys.meta_path.insert(0, _JaxImportHook())
    return cls


def is_concrete(value: Any) -> bool:
    """Whether ``value`` is a plain number, so it can be checked eagerly.

    JAX tracers, and the placeholder leaves JAX substitutes while it maps over
    a pytree, are not concrete: a record built from them skips its checks.
    """
    if isinstance(value, numbers.Real):
        return True
    return isinstance(value, np.ndarray | np.generic) and np.ndim(value) == 0
