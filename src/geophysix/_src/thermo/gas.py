"""The ideal gas: number density, density, and virtual temperature."""

from __future__ import annotations

from typing import Any

from geophysix._src._arrays import floating_inputs, like
from geophysix._src.constants.records import STANDARD, Constants


def number_density(
    pressure: Any, temperature: Any, *, constants: Constants = STANDARD
) -> Any:
    """The number of molecules per unit volume of an ideal gas.

    n = p / (k_B T)

    Args:
        pressure: Pressure p, Pa.
        temperature: Temperature T, K.
        constants: The constants record, for k_B; ``STANDARD`` by default.

    Returns:
        n, m⁻³, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> import math
        >>> from geophysix.thermo import number_density
        >>> loschmidt = 2.686780111e25
        >>> math.isclose(
        ...     float(number_density(101_325.0, 273.15)), loschmidt, rel_tol=1e-9
        ... )
        True
    """
    xp, (pressure, temperature) = floating_inputs(
        pressure, temperature, params=(constants.boltzmann,)
    )
    k_b = like(constants.boltzmann, pressure, xp)
    # n = p / (k_B T)
    return pressure / (k_b * temperature)
