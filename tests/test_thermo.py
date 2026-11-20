"""The thermodynamics of air."""

from __future__ import annotations

import math

import numpy as np
from _backends import assert_numpy_jax_agree

from geophysix.constants import H2O, STANDARD, Constants
from geophysix.thermo import (
    dry_air_density,
    exner,
    hypsometric_thickness,
    number_density,
    potential_temperature,
    virtual_temperature,
)


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


# --- dry-air thermodynamics ---


def test_dry_air_density() -> None:
    # ρ = p / (R_d T); the ISA sea-level density is 1.225 kg m⁻³
    assert math.isclose(float(dry_air_density(101_325.0, 288.15)), 1.225, rel_tol=1e-3)
    assert math.isclose(
        float(dry_air_density(1e5, 300.0)),
        1e5 / (STANDARD.dry_air_gas_constant * 300.0),
        rel_tol=1e-15,
    )


def test_virtual_temperature() -> None:
    assert float(virtual_temperature(300.0, 0.0)) == 300.0
    # 1/ε − 1 = M_dry / M_H2O − 1 ≈ 0.608
    excess = STANDARD.dry_air_molar_mass / H2O.molar_mass - 1.0
    assert math.isclose(excess, 0.608, rel_tol=1e-3)
    assert math.isclose(
        float(virtual_temperature(300.0, 0.01)),
        300.0 * (1.0 + excess * 0.01),
        rel_tol=1e-15,
    )


def test_potential_temperature_and_exner() -> None:
    p = np.linspace(2e4, 1.05e5, 9)
    t = np.linspace(220.0, 300.0, 9)
    assert float(potential_temperature(288.0, 1e5)) == 288.0
    assert float(exner(1e5)) == 1.0
    # θ Π = T
    np.testing.assert_allclose(potential_temperature(t, p) * exner(p), t, rtol=1e-14)
    # κ ≈ 2/7 for dry air
    assert math.isclose(STANDARD.kappa, 2.0 / 7.0, rel_tol=1e-3)


def test_hypsometric_thickness() -> None:
    # isothermal layers: Δz = (R_d T / g) ln(p₁ / p₂), exactly
    p = np.array([1e5, 8e4, 5e4, 2e4])
    dz = hypsometric_thickness(p, np.full(4, 260.0))
    scale = STANDARD.dry_air_gas_constant * 260.0 / STANDARD.standard_gravity
    np.testing.assert_allclose(dz, scale * np.log(p[:-1] / p[1:]), rtol=1e-15)
    assert (dz > 0.0).all()
    # warmer air makes a thicker layer; top-down ordering gives negative thickness
    assert float(hypsometric_thickness(p[:2], np.array([280.0, 280.0]))[0]) > float(
        dz[0]
    )
    np.testing.assert_allclose(
        hypsometric_thickness(p[::-1], np.full(4, 260.0)), -dz[::-1], rtol=1e-15
    )
    # leading axes are batch axes
    assert hypsometric_thickness(np.stack([p, p]), np.full((2, 4), 260.0)).shape == (
        2,
        3,
    )


def test_thermo_numpy_and_jax_agree() -> None:
    p = np.linspace(2e4, 1.05e5, 9)
    t = np.linspace(220.0, 300.0, 9)
    assert_numpy_jax_agree(dry_air_density, p, t, rtol=1e-14)
    assert_numpy_jax_agree(
        virtual_temperature, t, np.linspace(0.0, 0.02, 9), rtol=1e-14
    )
    assert_numpy_jax_agree(potential_temperature, t, p, rtol=1e-13)
    assert_numpy_jax_agree(exner, p, rtol=1e-13)
    assert_numpy_jax_agree(hypsometric_thickness, p[::-1], t[::-1], rtol=1e-13)
