"""Every public function on every array kind, and torch gradients against JAX's."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from _backends import (
    KINDS,
    ArrayKind,
    assert_backends_agree,
    needs_jax,
    needs_torch,
    outputs,
)
from _calls import CALLS, CHANGES_A_DIMENSION, MODULES, TAKES_NO_ARRAYS, Call

from geophysix import BackendError


CALL_IDS = [call.name for call in CALLS]


def test_every_public_function_has_a_call() -> None:
    public = {
        f"{module}.{name}"
        for module, contents in MODULES.items()
        for name in contents.__all__
        if inspect.isfunction(getattr(contents, name))
    }
    called = {call.name.split(" ")[0] for call in CALLS}
    assert called == public - TAKES_NO_ARRAYS, "add a Call to tests/_calls.py"


@pytest.mark.parametrize("kind", KINDS, ids=[kind.name for kind in KINDS])
@pytest.mark.parametrize("call", CALLS, ids=CALL_IDS)
def test_the_kind_comes_back(call: Call, kind: ArrayKind) -> None:
    if kind.name == "xarray" and call.name in CHANGES_A_DIMENSION:
        with pytest.raises(BackendError, match="DataArray"):
            call.function(*map(kind.make, call.args))
        return
    assert_backends_agree(
        call.function,
        *call.args,
        rtol=call.rtol,
        atol=call.atol,
        rtol32=call.rtol32,
        atol32=call.atol32,
        kinds=[kind],
    )


@needs_torch
@pytest.mark.parametrize("call", CALLS, ids=CALL_IDS)
def test_torch_keeps_the_gradient(call: Call) -> None:
    import torch

    tensors = [torch.tensor(arg, requires_grad=True) for arg in call.args]
    results = outputs(call.function(*tensors))
    assert all(result.requires_grad for result in results)
    # torch 2.10's asarray switched this off on the caller's own tensor
    assert all(tensor.requires_grad for tensor in tensors)


@needs_jax
@needs_torch
@pytest.mark.parametrize("call", CALLS, ids=CALL_IDS)
def test_torch_gradients_are_jax_gradients(call: Call) -> None:
    import jax
    import jax.numpy as jnp
    import torch

    tensors = [torch.tensor(arg, requires_grad=True) for arg in call.args]
    sum(result.sum() for result in outputs(call.function(*tensors))).backward()

    def total(*args):
        return sum(jnp.sum(result) for result in outputs(call.function(*args)))

    grads = jax.grad(total, argnums=tuple(range(len(call.args))))(*call.args)
    for tensor, grad in zip(tensors, grads, strict=True):
        want = np.asarray(grad)
        scale = np.max(np.abs(want), initial=0.0)
        np.testing.assert_allclose(
            tensor.grad.numpy(), want, rtol=1e-12, atol=1e-12 * scale
        )
