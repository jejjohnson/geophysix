"""Gas columns: units, the dry-air column and mixing ratios."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from geophysix.columns import (
    COLUMN_UNITS,
    column_from_mixing_ratio,
    convert,
    dry_air_column,
    mixing_ratio_from_column,
)
from geophysix.constants import CH4, CO2, STANDARD
from geophysix.exceptions import UnitError


AIR = {"gas": CH4, "pressure": 95_000.0, "temperature": 280.0}
VALUES = np.array([1.0, 250.0, 3.0e3])


@pytest.mark.parametrize(
    ("units_in", "units_out"), list(itertools.permutations(COLUMN_UNITS, 2))
)
def test_round_trip_is_the_identity(units_in: str, units_out: str) -> None:
    # G2-inverse
    there = convert(VALUES, units_in, units_out, **AIR)
    back = convert(there, units_out, units_in, **AIR)
    np.testing.assert_allclose(back, VALUES, rtol=1e-14)


def test_molar_volume_matches_geoops_at_its_conditions() -> None:
    # geoops fixes V_m = 0.024465 m³ mol⁻¹ (298.15 K, 1 atm), rounded to 5 digits
    mol = float(convert(1e6, "ppm_m", "mol_m2", pressure=101_325.0, temperature=298.15))
    assert math.isclose(1.0 / mol, 0.024465, rel_tol=2e-5)


def test_mass_and_molecules() -> None:
    assert math.isclose(
        float(convert(2.0, "mol_m2", "kg_m2", gas=CO2)),
        2.0 * CO2.molar_mass,
        rel_tol=1e-15,
    )
    assert math.isclose(
        float(convert(1.0, "mol_m2", "molecules_m2")), STANDARD.avogadro, rel_tol=1e-15
    )


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        (("ppm", "mol_m2"), {}, "unknown"),
        (("mol_m2", "kg_m2"), {}, "gas"),
        (("ppm_m", "mol_m2"), {"pressure": 1e5}, "pressure and temperature"),
    ],
)
def test_bad_requests_raise(args, kwargs, match) -> None:
    with pytest.raises(UnitError, match=match):
        convert(1.0, *args, **kwargs)


def test_dry_air_column() -> None:
    # p_s / g / M_dry; xrops: ≈ 2.12e29 molecules m⁻² at 10⁵ Pa
    n_dry = float(dry_air_column(1e5))
    assert math.isclose(
        n_dry,
        1e5 / STANDARD.standard_gravity / STANDARD.dry_air_molar_mass,
        rel_tol=1e-15,
    )
    assert math.isclose(n_dry * STANDARD.avogadro, 2.12e29, rel_tol=1e-2)
    # water vapour removes its mass from the column
    wet = float(dry_air_column(1e5, 30.0))
    assert math.isclose(n_dry - wet, 30.0 / STANDARD.dry_air_molar_mass, rel_tol=1e-12)


def test_mixing_ratio_pair_is_inverse() -> None:
    n_dry = dry_air_column(np.array([9e4, 1e5]))
    x = np.array([1.9e-6, 2.1e-6])
    back = mixing_ratio_from_column(column_from_mixing_ratio(x, n_dry), n_dry)
    np.testing.assert_allclose(back, x, rtol=1e-15)


def test_enhancement_to_mixing_ratio() -> None:
    # xrops' column_mass_to_delta_vmr is convert to mol m⁻², then the ratio
    n_dry = dry_air_column(1e5)
    mol = convert(1e-3, "kg_m2", "mol_m2", gas=CH4)
    dx = float(mixing_ratio_from_column(mol, n_dry))
    assert math.isclose(dx, 1e-3 / CH4.molar_mass / float(n_dry), rel_tol=1e-15)
