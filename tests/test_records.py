"""The Constants, Gas and Ellipsoid records: values, invariants and pytrees."""

from __future__ import annotations

import math

import pytest
from _backends import needs_jax

from geophysix.constants import CH4, CO, CO2, H2O, N2O, STANDARD, Constants, Gas
from geophysix.earth import GRS80, SPHERE, WGS84, Ellipsoid, sphere
from geophysix.exceptions import EarthError, GeophysixError


# --- G0-codata: the exact CODATA 2018 constants, compared exactly ---


def test_codata_exact_constants() -> None:
    assert STANDARD.boltzmann == 1.380649e-23
    assert STANDARD.avogadro == 6.02214076e23
    assert STANDARD.planck == 6.62607015e-34
    assert STANDARD.speed_of_light == 299_792_458.0


def test_conventional_constants() -> None:
    assert STANDARD.standard_gravity == 9.80665
    assert STANDARD.standard_pressure == 101_325.0


# --- G0-derived: derived values are properties of their parts ---


def test_gas_constant_matches_codata() -> None:
    # CODATA 2018: R = 8.314462618... J mol⁻¹ K⁻¹ (exact, as k_B N_A)
    assert math.isclose(STANDARD.gas_constant, 8.314462618, rel_tol=1e-10)
    assert math.isclose(
        STANDARD.gas_constant, STANDARD.boltzmann * STANDARD.avogadro, rel_tol=1e-15
    )


def test_dry_air_values() -> None:
    r_d = STANDARD.gas_constant / STANDARD.dry_air_molar_mass
    assert math.isclose(STANDARD.dry_air_gas_constant, r_d, rel_tol=1e-15)
    # R_d ≈ 287.05 J kg⁻¹ K⁻¹ and κ ≈ 2/7 for dry air
    assert math.isclose(STANDARD.dry_air_gas_constant, 287.05, rel_tol=1e-4)
    assert math.isclose(STANDARD.kappa, r_d / 1004.5, rel_tol=1e-15)


def test_derived_values_follow_a_custom_record() -> None:
    custom = Constants(boltzmann=1.0, avogadro=2.0)
    assert custom.gas_constant == 2.0


def test_records_are_frozen() -> None:
    with pytest.raises(AttributeError):
        STANDARD.standard_gravity = 9.81  # type: ignore[misc]
    with pytest.raises(AttributeError):
        WGS84.semi_major = 1.0  # type: ignore[misc]


# --- gases ---


@pytest.mark.parametrize(
    ("gas", "molar_mass"),
    [
        (CH4, 0.0160425),
        (CO2, 0.0440095),
        (H2O, 0.01801528),
        (N2O, 0.0440128),
        (CO, 0.0280101),
    ],
)
def test_gas_molar_masses(gas: Gas, molar_mass: float) -> None:
    assert gas.molar_mass == molar_mass


@pytest.mark.parametrize("molar_mass", [0.0, -1.0])
def test_gas_rejects_non_positive_mass(molar_mass: float) -> None:
    with pytest.raises(GeophysixError, match="molar_mass"):
        Gas("X", molar_mass)


# --- ellipsoids and spheres ---


def test_wgs84_published_values() -> None:
    # NGA (2014), WGS84: b = 6,356,752.314245 m; R₁ = (2a + b) / 3 = 6,371,008.7714 m
    # published to 1 µm, so 1e-12 relative
    assert math.isclose(WGS84.semi_minor, 6_356_752.314245, rel_tol=1e-12)
    assert math.isclose(WGS84.mean_radius, 6_371_008.7714, rel_tol=1e-11)
    assert math.isclose(WGS84.eccentricity_squared, 6.69437999014e-3, rel_tol=1e-11)
    assert WGS84.rotation_rate == 7.292115e-5


def test_grs80_differs_from_wgs84_only_in_flattening() -> None:
    assert GRS80.semi_major == WGS84.semi_major
    assert GRS80.flattening != WGS84.flattening
    # the two semi-minor axes differ by about 0.1 mm
    assert abs(GRS80.semi_minor - WGS84.semi_minor) < 2e-4


def test_sphere_is_the_zero_flattening_ellipsoid() -> None:
    body = sphere(6_371_000.0)
    assert body == Ellipsoid(6_371_000.0, 0.0)
    assert body.semi_minor == body.mean_radius == 6_371_000.0
    assert body.eccentricity_squared == 0.0
    assert SPHERE.semi_major == WGS84.mean_radius


@pytest.mark.parametrize(("a", "f"), [(0.0, 0.0), (-1.0, 0.0), (1.0, -0.1), (1.0, 1.0)])
def test_invalid_bodies_raise(a: float, f: float) -> None:
    with pytest.raises(EarthError):
        Ellipsoid(a, f)


def test_earth_error_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        Ellipsoid(1.0, 2.0)


# --- under JAX: pytrees, and a traced field ---


@needs_jax
def test_records_are_pytrees() -> None:
    import jax

    leaves, tree = jax.tree_util.tree_flatten(WGS84)
    assert leaves == [WGS84.semi_major, WGS84.flattening, WGS84.rotation_rate]
    assert jax.tree_util.tree_unflatten(tree, leaves) == WGS84
    # the gas name is static metadata, not a leaf
    assert jax.tree_util.tree_leaves(CH4) == [CH4.molar_mass]
    assert len(jax.tree_util.tree_leaves(STANDARD)) == 8


@needs_jax
def test_a_traced_field_is_differentiable() -> None:
    import jax

    # d(R₁)/da = (2 + (1 − f)) / 3 for R₁ = (2a + a (1 − f)) / 3
    def mean_radius(body: Ellipsoid) -> float:
        return body.mean_radius

    grad = jax.jit(jax.grad(mean_radius))(WGS84)
    assert math.isclose(
        float(grad.semi_major), (3.0 - WGS84.flattening) / 3.0, rel_tol=1e-12
    )


@needs_jax
def test_tree_map_skips_validation() -> None:
    import jax

    # tree_map rebuilds the record with arbitrary leaves; checks apply to numbers only
    doubled = jax.tree_util.tree_map(lambda v: 2.0 * v, sphere(1.0))
    assert doubled.semi_major == 2.0
