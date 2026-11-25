"""The dry-air column, and gas columns to and from dry-air mole fractions."""

from __future__ import annotations

from typing import Any

from pipekit_array.arrays import floating_inputs, like
from pipekit_array.kernels import elementwise

from geophysix._src.constants.records import STANDARD, Constants


@elementwise()
def dry_air_column(
    surface_pressure: Any,
    water_vapour_column: Any = 0.0,
    *,
    constants: Constants = STANDARD,
) -> Any:
    """The column of dry air above a point, from hydrostatic balance.

    n_dry = (p_s / g₀ − W) / M_dry

    with p_s the surface pressure, W the water-vapour mass column and M_dry
    the molar mass of dry air. Multiply by N_A for molecules m⁻².

    Args:
        surface_pressure: Surface pressure p_s, Pa.
        water_vapour_column: Water-vapour mass column W (total column water
            vapour), kg m⁻²; 0 treats the air as dry.
        constants: The constants record, for g₀ and M_dry.

    Returns:
        n_dry, mol m⁻², broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> import math
        >>> from geophysix.columns import dry_air_column
        >>> from geophysix.constants import STANDARD
        >>> molecules = float(dry_air_column(1e5)) * STANDARD.avogadro
        >>> math.isclose(molecules, 2.12e29, rel_tol=1e-2)
        True
    """
    xp, (pressure, water) = floating_inputs(
        surface_pressure,
        water_vapour_column,
        params=(constants.standard_gravity, constants.dry_air_molar_mass),
    )
    g = like(constants.standard_gravity, pressure, xp)
    m_dry = like(constants.dry_air_molar_mass, pressure, xp)
    # n_dry = (p_s / g₀ − W) / M_dry
    return (pressure / g - water) / m_dry


@elementwise()
def column_from_mixing_ratio(mixing_ratio: Any, dry_air_column: Any) -> Any:
    """The column of a gas from its dry-air mole fraction.

    Ω = x n_dry

    Args:
        mixing_ratio: Column-average dry-air mole fraction x (1, not ppm).
        dry_air_column: Dry-air column n_dry, in the units wanted for Ω
            (mol m⁻² from `dry_air_column`).

    Returns:
        Ω, in the units of ``dry_air_column``.

    Examples:
        >>> from geophysix.columns import column_from_mixing_ratio
        >>> float(column_from_mixing_ratio(2e-6, 3.5e5))
        0.7
    """
    _, (mixing_ratio, dry_air_column) = floating_inputs(mixing_ratio, dry_air_column)
    # Ω = x n_dry
    return mixing_ratio * dry_air_column


@elementwise()
def mixing_ratio_from_column(column: Any, dry_air_column: Any) -> Any:
    """The dry-air mole fraction of a gas from its column.

    x = Ω / n_dry

    ``column_mass_to_delta_vmr`` of xrops is `convert` to mol m⁻² followed by
    this function.

    Args:
        column: Column Ω of the gas, in the units of ``dry_air_column``.
        dry_air_column: Dry-air column n_dry.

    Returns:
        x, dimensionless.

    Examples:
        >>> from geophysix.columns import mixing_ratio_from_column
        >>> float(mixing_ratio_from_column(0.7, 3.5e5))
        2e-06
    """
    _, (column, dry_air_column) = floating_inputs(column, dry_air_column)
    # x = Ω / n_dry
    return column / dry_air_column
