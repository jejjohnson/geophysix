"""Tests for the private array-namespace dispatch."""

from __future__ import annotations

import types

import numpy as np
import pytest
from _backends import needs_jax

from geophysix._src import backend
from geophysix._src.backend import array_namespace, is_jax


class _Namespaced:
    """A stand-in array from another namespace."""

    def __init__(self, namespace: types.ModuleType) -> None:
        self._namespace = namespace

    def __array_namespace__(self) -> types.ModuleType:
        return self._namespace


OTHER = types.ModuleType("other_xp")


@pytest.mark.parametrize("value", [1, 1.0, True, 1j, None, np.zeros(3), np.float64(1)])
def test_scalars_none_and_numpy_resolve_to_numpy(value: object) -> None:
    assert array_namespace(value) is np


def test_no_arguments_resolve_to_numpy() -> None:
    assert array_namespace() is np


def test_numpy_array_namespace_method_does_not_win() -> None:
    assert array_namespace(_Namespaced(np), 1.0) is np


def test_the_first_non_numpy_namespace_wins() -> None:
    other = types.ModuleType("second_xp")
    assert array_namespace(np.zeros(2), _Namespaced(OTHER), _Namespaced(other)) is OTHER


def test_an_object_without_a_namespace_is_ignored() -> None:
    assert array_namespace(object(), "text") is np


def test_array_api_compat_decides_when_it_is_importable(monkeypatch) -> None:
    class Compat:
        @staticmethod
        def array_namespace(value: object) -> types.ModuleType:
            if value == "bad":
                raise TypeError("not an array")
            return OTHER

    monkeypatch.setattr(backend, "_compat", Compat)
    assert array_namespace("bad", 1.0) is np
    assert array_namespace(np.zeros(2), object()) is OTHER


def test_is_jax_is_false_for_numpy_and_other_modules() -> None:
    assert not is_jax(np)
    assert not is_jax(OTHER)


@needs_jax
def test_jax_arrays_and_tracers_resolve_to_jax_numpy() -> None:
    import jax
    import jax.numpy as jnp

    assert array_namespace(jnp.ones(3)) is jnp
    assert array_namespace(1.0, np.zeros(2), jnp.ones(3)) is jnp
    assert is_jax(jnp)

    seen = []

    def probe(x):
        seen.append(array_namespace(x))
        return x

    jax.jit(probe)(jnp.ones(2))
    assert seen == [jnp]


@needs_jax
def test_the_import_of_jax_is_lazy() -> None:
    import subprocess
    import sys

    code = (
        "import sys, geophysix, geophysix._src.backend as b, numpy as np;"
        "b.array_namespace(np.zeros(1)); sys.exit('jax' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
