"""The linear equation of state of sea water, buoyancy and reduced gravity.

Gate G3-gridax of the Wave 3 epic (#11): at the same defaults the six
functions give gridax-jax's numbers bit for bit in float64. geophysix cannot
import gridax, so gridax-jax's expressions (``gridax_jax/_src/operators/eos.py``
at jejjohnson/finitevolX ``c3501f8``) are written out here verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from _backends import assert_numpy_jax_agree, needs_jax

from geophysix.constants import Constants
from geophysix.exceptions import GeophysixError
from geophysix.seawater import (
    LinearEOS,
    buoyancy,
    linear_density,
    linear_density_anomaly,
    linear_drho_dS,
    linear_drho_dT,
    reduced_gravity,
)


RNG = np.random.default_rng(3)
T = RNG.uniform(-2.0, 30.0, 50)
S = RNG.uniform(30.0, 38.0, 50)
RHO = 1020.0 + 10.0 * RNG.random(50)
RHO_0, ALPHA, BETA, T_REF, S_REF, GRAVITY = 1025.0, 2e-4, 7e-4, 10.0, 35.0, 9.80665


def test_reproduces_gridax_bit_for_bit() -> None:
    assert np.array_equal(
        linear_density(T, S), RHO_0 * (1.0 - ALPHA * (T - T_REF) + BETA * (S - S_REF))
    )
    assert np.array_equal(
        linear_density_anomaly(T, S),
        RHO_0 * (-ALPHA * (T - T_REF) + BETA * (S - S_REF)),
    )
    assert linear_drho_dT() == -RHO_0 * ALPHA
    assert linear_drho_dS() == RHO_0 * BETA
    assert np.array_equal(buoyancy(RHO), -GRAVITY * (RHO - RHO_0) / RHO_0)
    assert np.array_equal(
        reduced_gravity(RHO[:-1], RHO[1:]), GRAVITY * (RHO[1:] - RHO[:-1]) / RHO_0
    )


def test_reference_state() -> None:
    assert float(linear_density(10.0, 35.0)) == 1025.0
    assert float(linear_density_anomaly(10.0, 35.0)) == 0.0
    # warmer is lighter, saltier is heavier
    assert (
        float(linear_density(11.0, 35.0)) < 1025.0 < float(linear_density(10.0, 36.0))
    )


def test_anomaly_is_density_minus_reference() -> None:
    np.testing.assert_allclose(
        linear_density_anomaly(T, S),
        linear_density(T, S) - 1025.0,
        rtol=1e-10,
        atol=1e-12,
    )


def test_buoyancy_and_reduced_gravity_follow_the_records() -> None:
    eos, constants = (
        LinearEOS(reference_density=1000.0),
        Constants(standard_gravity=10.0),
    )
    assert float(buoyancy(990.0, eos=eos, constants=constants)) == pytest.approx(
        0.1, rel=1e-15
    )
    assert float(
        reduced_gravity(1000.0, 1001.0, eos=eos, constants=constants)
    ) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_density": 0.0},
        {"thermal_expansion": -1e-4},
        {"haline_contraction": -1e-4},
    ],
)
def test_invalid_eos_raises(kwargs) -> None:
    with pytest.raises(GeophysixError):
        LinearEOS(**kwargs)


def test_numpy_and_jax_agree() -> None:
    assert_numpy_jax_agree(linear_density, T, S, rtol=1e-15)
    assert_numpy_jax_agree(linear_density_anomaly, T, S, rtol=1e-13)
    assert_numpy_jax_agree(buoyancy, RHO, rtol=1e-12)
    assert_numpy_jax_agree(reduced_gravity, RHO[:-1], RHO[1:], rtol=1e-12)


@needs_jax
def test_gradients_are_the_derivatives() -> None:
    import jax

    d_t, d_s = jax.grad(lambda t, s: linear_density(t, s), argnums=(0, 1))(12.0, 34.0)
    assert float(d_t) == pytest.approx(linear_drho_dT(), rel=1e-15)
    assert float(d_s) == pytest.approx(linear_drho_dS(), rel=1e-15)
    # a traced ρ₀ is differentiable too
    grad = jax.grad(lambda rho0: linear_density(12.0, 34.0, eos=LinearEOS(rho0)))(
        1025.0
    )
    assert float(grad) == pytest.approx(1.0 - 2e-4 * 2.0 - 7e-4, rel=1e-14)
