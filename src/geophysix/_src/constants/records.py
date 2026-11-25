"""Physical constants and trace gases as frozen records, SI units throughout."""

from __future__ import annotations

import dataclasses

from pipekit_array.records import register_record

from geophysix._src._records import is_concrete
from geophysix.exceptions import GeophysixError


@dataclasses.dataclass(frozen=True, slots=True)
class Constants:
    """One convention's physical constants, SI throughout.

    The defaults are CODATA 2018 for the exact constants (k_B, N_A, h, c),
    standard gravity g₀, the standard atmosphere's p₀ and T₀, and the dry-air values
    used across the stack. Derived values (R, R_d, κ) are properties, never
    stored, so they cannot drift from their parts. A caller that wants
    another convention builds its own record and passes it in.

    Attributes:
        boltzmann: Boltzmann constant k_B, J K⁻¹ (exact).
        avogadro: Avogadro constant N_A, mol⁻¹ (exact).
        planck: Planck constant h, J s (exact).
        speed_of_light: Speed of light in vacuum c, m s⁻¹ (exact).
        standard_gravity: Standard gravity g₀, m s⁻².
        standard_pressure: One standard atmosphere p₀, Pa.
        dry_air_molar_mass: Molar mass of dry air M_dry, kg mol⁻¹.
        dry_air_heat_capacity: Specific heat of dry air at constant pressure
            c_p, J kg⁻¹ K⁻¹.
        standard_temperature: The International Standard Atmosphere's
            sea-level temperature T₀, K.

    Examples:
        >>> import math
        >>> from geophysix.constants import STANDARD
        >>> math.isclose(STANDARD.gas_constant, 8.314462618, rel_tol=1e-10)
        True
        >>> STANDARD.standard_gravity
        9.80665
        >>> from geophysix.constants import Constants
        >>> Constants(standard_gravity=9.81).standard_gravity
        9.81
    """

    boltzmann: float = 1.380649e-23
    avogadro: float = 6.02214076e23
    planck: float = 6.62607015e-34
    speed_of_light: float = 299_792_458.0
    standard_gravity: float = 9.80665
    standard_pressure: float = 101_325.0
    dry_air_molar_mass: float = 0.0289644
    dry_air_heat_capacity: float = 1004.5
    standard_temperature: float = 288.15

    @property
    def gas_constant(self) -> float:
        """The molar gas constant, R = k_B N_A, J mol⁻¹ K⁻¹."""
        return self.boltzmann * self.avogadro

    @property
    def dry_air_gas_constant(self) -> float:
        """The specific gas constant of dry air, R_d = R / M_dry, J kg⁻¹ K⁻¹."""
        return self.gas_constant / self.dry_air_molar_mass

    @property
    def kappa(self) -> float:
        """The Poisson constant of dry air, κ = R_d / c_p (dimensionless)."""
        return self.dry_air_gas_constant / self.dry_air_heat_capacity


@dataclasses.dataclass(frozen=True, slots=True)
class Gas:
    """A trace gas: its name and its molar mass M, kg mol⁻¹.

    Attributes:
        name: The gas's formula, e.g. ``"CH4"``; static under JAX.
        molar_mass: Molar mass M, kg mol⁻¹; > 0.

    Raises:
        GeophysixError: If the molar mass is not positive.

    Examples:
        >>> from geophysix.constants import CH4, Gas
        >>> CH4
        Gas(name='CH4', molar_mass=0.0160425)
        >>> Gas("X", -1.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        geophysix.exceptions.GeophysixError: molar_mass must be > 0, got -1.0
    """

    name: str
    molar_mass: float

    def __post_init__(self) -> None:
        if is_concrete(self.molar_mass) and not self.molar_mass > 0.0:
            msg = f"molar_mass must be > 0, got {self.molar_mass}"
            raise GeophysixError(msg)


register_record(Constants)
register_record(Gas, meta_fields=("name",))

#: CODATA 2018, standard gravity and the standard atmosphere.
STANDARD = Constants()

# Molar masses from the IUPAC standard atomic weights, kg mol⁻¹.
#: Methane.
CH4 = Gas("CH4", 0.0160425)
#: Carbon dioxide.
CO2 = Gas("CO2", 0.0440095)
#: Water vapour.
H2O = Gas("H2O", 0.01801528)
#: Nitrous oxide.
N2O = Gas("N2O", 0.0440128)
#: Carbon monoxide.
CO = Gas("CO", 0.0280101)
