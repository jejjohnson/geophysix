"""Column amounts of a gas, and the conversions between their units."""

from __future__ import annotations

from typing import Any

from pipekit_array.functions import elementwise, floating_inputs, like

from geophysix._src.constants.records import STANDARD, Constants, Gas
from geophysix.exceptions import UnitError


#: The column units ``convert`` knows.
COLUMN_UNITS = ("kg_m2", "mol_m2", "molecules_m2", "ppm_m")


def _to_mol_m2(value, units, gas, pressure, temperature, constants, xp):
    """``value`` in ``units`` as mol m⁻²."""
    if units == "mol_m2":
        return value
    if units == "molecules_m2":
        # Ω_mol = N / N_A
        return value / like(constants.avogadro, value, xp)
    if units == "kg_m2":
        # Ω_mol = Ω_kg / M
        return value / like(gas.molar_mass, value, xp)
    # Ω_mol = 10⁻⁶ X / V_m, V_m = R T / p
    molar_volume = like(constants.gas_constant, value, xp) * temperature / pressure
    return 1e-6 * value / molar_volume


def _from_mol_m2(value, units, gas, pressure, temperature, constants, xp):
    """``value`` in mol m⁻² as ``units``."""
    if units == "mol_m2":
        return value
    if units == "molecules_m2":
        # N = N_A Ω_mol
        return value * like(constants.avogadro, value, xp)
    if units == "kg_m2":
        # Ω_kg = M Ω_mol
        return value * like(gas.molar_mass, value, xp)
    # X = 10⁶ Ω_mol V_m, V_m = R T / p
    molar_volume = like(constants.gas_constant, value, xp) * temperature / pressure
    return 1e6 * value * molar_volume


@elementwise()
def convert(
    value: Any,
    units_in: str,
    units_out: str,
    *,
    gas: Gas | None = None,
    pressure: Any = None,
    temperature: Any = None,
    constants: Constants = STANDARD,
) -> Any:
    """A gas column converted between ppm·m, mol m⁻², kg m⁻² and molecules m⁻².

    Ω_mol = 10⁻⁶ X / V_m,  Ω_kg = M Ω_mol,  N = N_A Ω_mol,  V_m = R T / p

    with X a column enhancement in ppm·m, V_m the molar volume of air at
    pressure p and temperature T, and M the gas's molar mass. The unit names
    are static strings, so the conversion traces under ``jit``.

    Args:
        value: The column, in ``units_in``.
        units_in: One of ``"ppm_m"``, ``"mol_m2"``, ``"kg_m2"``,
            ``"molecules_m2"``.
        units_out: One of the same.
        gas: The gas, for its molar mass M; needed for ``"kg_m2"``.
        pressure: Pressure p of the air, Pa; needed for ``"ppm_m"``.
        temperature: Temperature T of the air, K; needed for ``"ppm_m"``.
        constants: The constants record, for R and N_A.

    Returns:
        The column in ``units_out``, in ``value``'s namespace and dtype.

    Raises:
        UnitError: If a unit is unknown, or the gas, pressure or temperature
            that a unit needs is missing.

    Examples:
        >>> import numpy as np
        >>> from geophysix.columns import convert
        >>> from geophysix.constants import CH4
        >>> kg = convert(
        ...     np.array([1000.0]),
        ...     "ppm_m",
        ...     "kg_m2",
        ...     gas=CH4,
        ...     pressure=101_325.0,
        ...     temperature=298.15,
        ... )
        >>> bool(np.allclose(kg, 1000.0e-6 / 0.024465 * 0.0160425, rtol=1e-4))
        True
        >>> convert(1.0, "ppm_m", "mol_m2")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        geophysix.exceptions.UnitError: "ppm_m" needs pressure and temperature
    """
    for units in (units_in, units_out):
        if units not in COLUMN_UNITS:
            msg = f"unknown column unit {units!r}; expected one of {COLUMN_UNITS}"
            raise UnitError(msg)
        if units == "kg_m2" and gas is None:
            msg = '"kg_m2" needs the gas, for its molar mass'
            raise UnitError(msg)
        if units == "ppm_m" and (pressure is None or temperature is None):
            msg = '"ppm_m" needs pressure and temperature'
            raise UnitError(msg)
    molar_mass = () if gas is None else (gas.molar_mass,)
    xp, (value,) = floating_inputs(
        value, params=(*molar_mass, constants.avogadro, pressure, temperature)
    )
    p = None if pressure is None else like(pressure, value, xp)
    t = None if temperature is None else like(temperature, value, xp)
    mol = _to_mol_m2(value, units_in, gas, p, t, constants, xp)
    return _from_mol_m2(mol, units_out, gas, p, t, constants, xp)
