"""The thermodynamics of air."""

from __future__ import annotations

import math

import numpy as np
from _backends import assert_numpy_jax_agree

from geophysix.constants import Constants
from geophysix.thermo import number_density


def test_loschmidt_constant() -> None:
    # CODATA 2018: n₀ = 2.686780111e25 m⁻³ at 273.15 K and 101.325 kPa
    assert math.isclose(
        float(number_density(101_325.0, 273.15)), 2.686780111e25, rel_tol=1e-9
    )


def test_scales_with_pressure_and_inverse_temperature() -> None:
    n = number_density(np.array([5e4, 1e5]), np.array([250.0, 250.0]))
    assert math.isclose(float(n[1] / n[0]), 2.0, rel_tol=1e-15)
    assert float(number_density(1.0, 1.0, constants=Constants(boltzmann=0.5))) == 2.0


def test_numpy_and_jax_agree() -> None:
    p = np.linspace(1e3, 1e5, 7)
    t = np.linspace(200.0, 300.0, 7)
    assert_numpy_jax_agree(number_density, p, t, rtol=1e-14)
