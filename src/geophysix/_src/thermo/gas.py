"""The ideal gas and dry-air thermodynamics: density, virtual and potential
temperature and its inverse, the Exner function, hypsometric thickness, and
the hydrostatic conversion between pressure and vertical velocity."""

from __future__ import annotations

from typing import Any

from geophysix._src._arrays import floating_inputs, like
from geophysix._src.constants.records import H2O, STANDARD, Constants, Gas


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


def dry_air_density(
    pressure: Any, temperature: Any, *, constants: Constants = STANDARD
) -> Any:
    """The density of dry air from the ideal gas law.

    ρ = p / (R_d T)

    For moist air pass the virtual temperature (`virtual_temperature`).

    Args:
        pressure: Pressure p, Pa.
        temperature: Temperature T (or virtual temperature T_v), K.
        constants: The constants record, for R_d; ``STANDARD`` by default.

    Returns:
        ρ, kg m⁻³, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> import math
        >>> from geophysix.thermo import dry_air_density
        >>> math.isclose(
        ...     float(dry_air_density(101_325.0, 288.15)), 1.225, rel_tol=1e-3
        ... )
        True
    """
    xp, (pressure, temperature) = floating_inputs(
        pressure, temperature, params=(constants.boltzmann,)
    )
    r_d = like(constants.dry_air_gas_constant, pressure, xp)
    # ρ = p / (R_d T)
    return pressure / (r_d * temperature)


def virtual_temperature(
    temperature: Any,
    specific_humidity: Any,
    *,
    water_vapour: Gas = H2O,
    constants: Constants = STANDARD,
) -> Any:
    """The virtual temperature of moist air.

    T_v = T (1 + (1/ε − 1) q),  ε = M_H2O / M_dry

    the temperature dry air would need to have the moist air's density at
    the same pressure; 1/ε − 1 ≈ 0.608.

    Args:
        temperature: Temperature T, K.
        specific_humidity: Specific humidity q, kg kg⁻¹.
        water_vapour: The water-vapour gas, for M_H2O; ``H2O`` by default.
        constants: The constants record, for M_dry; ``STANDARD`` by default.

    Returns:
        T_v, K, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> from geophysix.thermo import virtual_temperature
        >>> float(virtual_temperature(300.0, 0.0))
        300.0
    """
    xp, (temperature, q) = floating_inputs(
        temperature, specific_humidity, params=(water_vapour.molar_mass,)
    )
    # 1/ε − 1 = M_dry / M_H2O − 1
    excess = like(
        constants.dry_air_molar_mass / water_vapour.molar_mass - 1.0, temperature, xp
    )
    # T_v = T (1 + (1/ε − 1) q)
    return temperature * (1.0 + excess * q)


def exner(
    pressure: Any,
    *,
    reference_pressure: float = 100_000.0,
    constants: Constants = STANDARD,
) -> Any:
    """The Exner function, the dimensionless pressure of dry air.

    Π = (p / p₀)^κ,  κ = R_d / c_p

    Args:
        pressure: Pressure p, Pa.
        reference_pressure: Reference pressure p₀, Pa; 1000 hPa by default.
        constants: The constants record, for κ; ``STANDARD`` by default.

    Returns:
        Π, dimensionless, in ``pressure``'s shape, namespace and dtype.

    Examples:
        >>> from geophysix.thermo import exner
        >>> float(exner(100_000.0))
        1.0
    """
    xp, (pressure,) = floating_inputs(pressure, params=(reference_pressure,))
    p0 = like(reference_pressure, pressure, xp)
    kappa = like(constants.kappa, pressure, xp)
    # Π = (p / p₀)^κ
    return (pressure / p0) ** kappa


def potential_temperature(
    temperature: Any,
    pressure: Any,
    *,
    reference_pressure: float = 100_000.0,
    constants: Constants = STANDARD,
) -> Any:
    """The potential temperature of dry air.

    θ = T (p₀ / p)^κ = T / Π

    the temperature the air would reach brought adiabatically to p₀.

    Args:
        temperature: Temperature T, K.
        pressure: Pressure p, Pa.
        reference_pressure: Reference pressure p₀, Pa; 1000 hPa by default.
        constants: The constants record, for κ; ``STANDARD`` by default.

    Returns:
        θ, K, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> from geophysix.thermo import potential_temperature
        >>> float(potential_temperature(290.0, 100_000.0))
        290.0
    """
    xp, (temperature, pressure) = floating_inputs(
        temperature, pressure, params=(reference_pressure,)
    )
    p0 = like(reference_pressure, temperature, xp)
    kappa = like(constants.kappa, temperature, xp)
    # θ = T (p₀ / p)^κ
    return temperature * (p0 / pressure) ** kappa


def temperature_from_potential_temperature(
    potential_temperature: Any,
    pressure: Any,
    *,
    reference_pressure: float = 100_000.0,
    constants: Constants = STANDARD,
) -> Any:
    """Temperature from the potential temperature of dry air.

    T = θ (p / p₀)^κ = θ Π

    the inverse of `potential_temperature`, as a WRF reader needs it to turn
    the perturbation potential temperature plus its base state into T.

    Args:
        potential_temperature: Potential temperature θ, K.
        pressure: Pressure p, Pa.
        reference_pressure: Reference pressure p₀, Pa; 1000 hPa by default.
        constants: The constants record, for κ; ``STANDARD`` by default.

    Returns:
        T, K, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> import math
        >>> from geophysix.thermo import (
        ...     potential_temperature,
        ...     temperature_from_potential_temperature,
        ... )
        >>> float(temperature_from_potential_temperature(300.0, 100_000.0))
        300.0
        >>> t = temperature_from_potential_temperature(300.0, 50_000.0)
        >>> math.isclose(float(potential_temperature(t, 50_000.0)), 300.0)
        True
    """
    xp, (theta, pressure) = floating_inputs(
        potential_temperature, pressure, params=(reference_pressure,)
    )
    p0 = like(reference_pressure, theta, xp)
    kappa = like(constants.kappa, theta, xp)
    # T = θ (p / p₀)^κ
    return theta * (pressure / p0) ** kappa


def hypsometric_thickness(
    pressure: Any, virtual_temperature: Any, *, constants: Constants = STANDARD
) -> Any:
    """The thickness of each layer between consecutive pressure levels.

    Δz_k = (R_d T̄_v / g₀) ln(p_k / p_{k+1}),  T̄_v = (T_v,k + T_v,k+1) / 2

    along the last axis. A layer is thicker where its air is warmer. The
    thickness is positive when pressure decreases along the axis (the
    surface first); heights are its cumulative sum from the surface.

    Args:
        pressure: Pressure p at the N levels, Pa, along the last axis.
        virtual_temperature: Virtual temperature T_v at the same levels, K.
        constants: The constants record, for R_d and g₀; ``STANDARD`` by default.

    Returns:
        Δz, m, with N − 1 entries along the last axis.

    Examples:
        >>> import numpy as np
        >>> from geophysix.constants import STANDARD
        >>> from geophysix.thermo import hypsometric_thickness
        >>> dz = hypsometric_thickness(
        ...     np.array([1e5, 5e4]), np.array([250.0, 250.0])
        ... )
        >>> scale = (
        ...     STANDARD.dry_air_gas_constant * 250.0 / STANDARD.standard_gravity
        ... )
        >>> bool(np.allclose(dz, scale * np.log(2.0)))
        True
    """
    xp, (pressure, t_v) = floating_inputs(
        pressure, virtual_temperature, params=(constants.standard_gravity,)
    )
    r_d = like(constants.dry_air_gas_constant, pressure, xp)
    g = like(constants.standard_gravity, pressure, xp)
    # T̄_v = (T_v,k + T_v,k+1) / 2
    mean_t_v = 0.5 * (t_v[..., :-1] + t_v[..., 1:])
    # Δz_k = (R_d T̄_v / g₀) ln(p_k / p_{k+1})
    return r_d * mean_t_v / g * xp.log(pressure[..., :-1] / pressure[..., 1:])


def vertical_velocity(
    pressure_velocity: Any,
    pressure: Any,
    temperature: Any,
    *,
    constants: Constants = STANDARD,
) -> Any:
    """Vertical velocity from pressure velocity, in hydrostatic balance.

    w = −ω / (ρ g₀),  ρ = p / (R_d T)

    from ω ≈ (∂p/∂z) w = −ρ g₀ w, neglecting the local tendency and the
    horizontal advection of p. Rising air has ω < 0 and w > 0. For moist air
    pass the virtual temperature (`virtual_temperature`).

    Args:
        pressure_velocity: Pressure velocity ω = Dp/Dt, Pa s⁻¹.
        pressure: Pressure p, Pa.
        temperature: Temperature T (or virtual temperature T_v), K.
        constants: The constants record, for R_d and g₀; ``STANDARD`` by default.

    Returns:
        w, m s⁻¹, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> from geophysix.thermo import vertical_velocity
        >>> round(float(vertical_velocity(-1.0, 50_000.0, 250.0)), 6)
        0.146359
    """
    xp, (omega, pressure, temperature) = floating_inputs(
        pressure_velocity,
        pressure,
        temperature,
        params=(constants.standard_gravity,),
    )
    g = like(constants.standard_gravity, omega, xp)
    # ρ = p / (R_d T)
    density = dry_air_density(pressure, temperature, constants=constants)
    # w = −ω / (ρ g₀)
    return -omega / (density * g)


def pressure_velocity(
    vertical_velocity: Any,
    pressure: Any,
    temperature: Any,
    *,
    constants: Constants = STANDARD,
) -> Any:
    """Pressure velocity from vertical velocity, in hydrostatic balance.

    ω = −ρ g₀ w,  ρ = p / (R_d T)

    the inverse of `vertical_velocity`.

    Args:
        vertical_velocity: Vertical velocity w, m s⁻¹, positive upward.
        pressure: Pressure p, Pa.
        temperature: Temperature T (or virtual temperature T_v), K.
        constants: The constants record, for R_d and g₀; ``STANDARD`` by default.

    Returns:
        ω, Pa s⁻¹, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> import math
        >>> from geophysix.thermo import pressure_velocity, vertical_velocity
        >>> w = vertical_velocity(-1.0, 50_000.0, 250.0)
        >>> math.isclose(float(pressure_velocity(w, 50_000.0, 250.0)), -1.0)
        True
    """
    xp, (w, pressure, temperature) = floating_inputs(
        vertical_velocity,
        pressure,
        temperature,
        params=(constants.standard_gravity,),
    )
    g = like(constants.standard_gravity, w, xp)
    # ρ = p / (R_d T)
    density = dry_air_density(pressure, temperature, constants=constants)
    # ω = −ρ g₀ w
    return -density * g * w
