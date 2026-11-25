"""The linear equation of state of sea water, buoyancy and reduced gravity."""

from __future__ import annotations

import dataclasses
from typing import Any

from pipekit_array.arrays import floating_inputs, like, register_record
from pipekit_array.kernels import elementwise

from geophysix._src._records import is_concrete
from geophysix._src.constants.records import STANDARD, Constants
from geophysix.exceptions import GeophysixError


@dataclasses.dataclass(frozen=True, slots=True)
class LinearEOS:
    """The parameters of the linear equation of state of sea water.

    ρ = ρ₀ (1 − α (T − T₀) + β_S (S − S₀))

    Attributes:
        reference_density: Reference density ρ₀, kg m⁻³; > 0.
        thermal_expansion: Thermal expansion coefficient α, K⁻¹; ≥ 0.
        haline_contraction: Haline contraction coefficient β_S, per unit of
            salinity; ≥ 0.
        reference_temperature: Reference temperature T₀, °C.
        reference_salinity: Reference salinity S₀.

    Raises:
        GeophysixError: If ρ₀ ≤ 0, or α or β_S is negative.

    Examples:
        >>> from geophysix.seawater import LINEAR_EOS
        >>> LINEAR_EOS.reference_density
        1025.0
    """

    reference_density: float = 1025.0
    thermal_expansion: float = 2e-4
    haline_contraction: float = 7e-4
    reference_temperature: float = 10.0
    reference_salinity: float = 35.0

    def __post_init__(self) -> None:
        rho0 = self.reference_density
        if is_concrete(rho0) and not rho0 > 0.0:
            msg = f"reference_density must be > 0, got {rho0}"
            raise GeophysixError(msg)
        for name in ("thermal_expansion", "haline_contraction"):
            value = getattr(self, name)
            if is_concrete(value) and value < 0.0:
                msg = f"{name} must be >= 0, got {value}"
                raise GeophysixError(msg)


register_record(LinearEOS)

#: gridax's defaults: ρ₀ = 1025 kg m⁻³, α = 2e-4 K⁻¹, β_S = 7e-4, T₀ = 10 °C, S₀ = 35.
LINEAR_EOS = LinearEOS()


@elementwise()
def linear_density(
    temperature: Any, salinity: Any, *, eos: LinearEOS = LINEAR_EOS
) -> Any:
    """Sea-water density from the linear equation of state.

    ρ = ρ₀ (1 − α (T − T₀) + β_S (S − S₀))

    Args:
        temperature: Temperature T, °C.
        salinity: Salinity S, in the units of ``eos.reference_salinity``.
        eos: The equation of state; ``LINEAR_EOS`` by default.

    Returns:
        ρ, kg m⁻³, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> from geophysix.seawater import linear_density
        >>> float(linear_density(10.0, 35.0))
        1025.0
    """
    xp, (t, s) = floating_inputs(temperature, salinity, params=_params(eos))
    rho0, alpha, beta, t0, s0 = (like(v, t, xp) for v in _params(eos))
    # ρ = ρ₀ (1 − α (T − T₀) + β_S (S − S₀))
    return rho0 * (1.0 - alpha * (t - t0) + beta * (s - s0))


@elementwise()
def linear_density_anomaly(
    temperature: Any, salinity: Any, *, eos: LinearEOS = LINEAR_EOS
) -> Any:
    """The density anomaly ρ − ρ₀ from the linear equation of state.

    ρ′ = ρ₀ (−α (T − T₀) + β_S (S − S₀))

    Written directly rather than as ``linear_density − ρ₀``, which would lose
    the anomaly's digits to cancellation.

    Args:
        temperature: Temperature T, °C.
        salinity: Salinity S.
        eos: The equation of state; ``LINEAR_EOS`` by default.

    Returns:
        ρ′, kg m⁻³, broadcast over the inputs, in their namespace and dtype.

    Examples:
        >>> import numpy as np
        >>> from geophysix.seawater import linear_density_anomaly
        >>> bool(np.isclose(linear_density_anomaly(11.0, 35.0), -1025.0 * 2e-4))
        True
    """
    xp, (t, s) = floating_inputs(temperature, salinity, params=_params(eos))
    rho0, alpha, beta, t0, s0 = (like(v, t, xp) for v in _params(eos))
    # ρ′ = ρ₀ (−α (T − T₀) + β_S (S − S₀))
    return rho0 * (-alpha * (t - t0) + beta * (s - s0))


def linear_drho_dT(*, eos: LinearEOS = LINEAR_EOS) -> Any:
    """The thermal derivative of the linear equation of state.

    ∂ρ/∂T = −ρ₀ α

    Args:
        eos: The equation of state; ``LINEAR_EOS`` by default.

    Returns:
        ∂ρ/∂T, kg m⁻³ K⁻¹; negative, since warmer water is lighter.

    Examples:
        >>> from geophysix.seawater import linear_drho_dT
        >>> round(linear_drho_dT(), 6)
        -0.205
    """
    return -eos.reference_density * eos.thermal_expansion


def linear_drho_dS(*, eos: LinearEOS = LINEAR_EOS) -> Any:
    """The haline derivative of the linear equation of state.

    ∂ρ/∂S = ρ₀ β_S

    Args:
        eos: The equation of state; ``LINEAR_EOS`` by default.

    Returns:
        ∂ρ/∂S, kg m⁻³ per unit of salinity; positive, since saltier water is
        heavier.

    Examples:
        >>> from geophysix.seawater import linear_drho_dS
        >>> round(linear_drho_dS(), 6)
        0.7175
    """
    return eos.reference_density * eos.haline_contraction


@elementwise()
def buoyancy(
    density: Any, *, eos: LinearEOS = LINEAR_EOS, constants: Constants = STANDARD
) -> Any:
    """Buoyancy relative to the reference density.

    b = −g₀ (ρ − ρ₀) / ρ₀

    Args:
        density: Density ρ, kg m⁻³.
        eos: The equation of state, for ρ₀; ``LINEAR_EOS`` by default.
        constants: The constants record, for g₀; ``STANDARD`` by default.

    Returns:
        b, m s⁻²; positive where the water is lighter than ρ₀.

    Examples:
        >>> from geophysix.seawater import buoyancy
        >>> float(buoyancy(1025.0))
        -0.0
    """
    xp, (rho,) = floating_inputs(
        density, params=(eos.reference_density, constants.standard_gravity)
    )
    rho0 = like(eos.reference_density, rho, xp)
    g = like(constants.standard_gravity, rho, xp)
    # b = −g₀ (ρ − ρ₀) / ρ₀
    return -g * (rho - rho0) / rho0


@elementwise()
def reduced_gravity(
    upper_density: Any,
    lower_density: Any,
    *,
    eos: LinearEOS = LINEAR_EOS,
    constants: Constants = STANDARD,
) -> Any:
    """The reduced gravity across the interface of two layers.

    g′ = g₀ (ρ_lower − ρ_upper) / ρ₀

    Args:
        upper_density: Density of the upper layer ρ_upper, kg m⁻³.
        lower_density: Density of the lower layer ρ_lower, kg m⁻³.
        eos: The equation of state, for ρ₀; ``LINEAR_EOS`` by default.
        constants: The constants record, for g₀; ``STANDARD`` by default.

    Returns:
        g′, m s⁻²; positive for a stable stratification.

    Examples:
        >>> import numpy as np
        >>> from geophysix.seawater import reduced_gravity
        >>> bool(
        ...     np.isclose(
        ...         reduced_gravity(1025.0, 1027.05), 9.80665 * 2.05 / 1025.0
        ...     )
        ... )
        True
    """
    xp, (upper, lower) = floating_inputs(
        upper_density,
        lower_density,
        params=(eos.reference_density, constants.standard_gravity),
    )
    rho0 = like(eos.reference_density, upper, xp)
    g = like(constants.standard_gravity, upper, xp)
    # g′ = g₀ (ρ_lower − ρ_upper) / ρ₀
    return g * (lower - upper) / rho0


def _params(eos: LinearEOS) -> tuple[Any, ...]:
    """ρ₀, α, β_S, T₀ and S₀ of ``eos``, in that order."""
    return (
        eos.reference_density,
        eos.thermal_expansion,
        eos.haline_contraction,
        eos.reference_temperature,
        eos.reference_salinity,
    )
