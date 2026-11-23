"""One call of every public function that takes arrays.

`test_array_kinds.py` runs each call on every array kind, and fails when a
public function has no call here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np

from geophysix import columns, earth, rotation, seawater, thermo, wind
from geophysix.constants import CH4


__all__ = [
    "CALLS",
    "CHANGES_A_DIMENSION",
    "MODULES",
    "TAKES_NO_ARRAYS",
    "Call",
]

MODULES = {
    module.__name__.rsplit(".", 1)[-1]: module
    for module in (columns, earth, rotation, seawater, thermo, wind)
}

# Public functions of numbers and records only.
TAKES_NO_ARRAYS = {
    "earth.sphere",
    "seawater.linear_drho_dS",
    "seawater.linear_drho_dT",
}

# Not elementwise: DataArrays are refused (#59).
CHANGES_A_DIMENSION = {"earth.cell_area", "thermo.hypsometric_thickness"}


@dataclasses.dataclass(frozen=True, slots=True)
class Call:
    """A function, float64 arrays to call it with, and its tolerances."""

    name: str
    args: tuple[np.ndarray, ...]
    fn: Callable[..., Any] | None = None
    rtol: float = 1e-14
    atol: float = 0.0
    rtol32: float = 1e-5
    atol32: float = 0.0

    @property
    def function(self) -> Callable[..., Any]:
        """``fn``, or the public function called ``name``."""
        if self.fn is not None:
            return self.fn
        module, name = self.name.split(" ")[0].split(".")
        return getattr(MODULES[module], name)


def _convert_ppm_m(column: Any, pressure: Any, temperature: Any) -> Any:
    return columns.convert(
        column, "ppm_m", "mol_m2", pressure=pressure, temperature=temperature
    )


def _convert_mass(column: Any) -> Any:
    return columns.convert(column, "mol_m2", "kg_m2", gas=CH4)


def _beta_plane(y: Any) -> Any:
    return rotation.beta_plane(y, f0=1e-4, beta=2e-11, y0=0.0)


LON = np.asarray([1.0, 20.0, -50.0, 170.0])
LAT = np.asarray([3.0, 45.0, -70.0, 10.0])
LON2 = np.asarray([5.0, 30.0, -20.0, -160.0])
LAT2 = np.asarray([0.0, 40.0, -60.0, 20.0])
HEIGHT = np.asarray([0.0, 100.0, 2000.0, -50.0])
XYZ = tuple(np.asarray(v) for v in earth.geodetic_to_ecef(LON, LAT, HEIGHT))

P = np.asarray([1e5, 8e4, 5e4, 2e4])
T = np.asarray([290.0, 280.0, 250.0, 220.0])
OMEGA = np.asarray([1.0, -1.0, 0.5, 0.1])
RHO = np.asarray([1020.0, 1025.5, 1030.0])
SALT = np.asarray([35.0, 34.0, 33.0])
SEA_T = np.asarray([11.0, 12.0, 14.0])
COLUMN = np.asarray([1.0, 2.0, 3.0, 4.0])
# calm air, a westerly, and two oblique winds
U = np.asarray([0.0, 5.0, 3.0, -2.0])
V = np.asarray([0.0, 0.0, -4.0, 1.0])

CALLS = [
    Call("columns.convert (to mass)", (COLUMN,), _convert_mass),
    Call("columns.convert (from ppm m)", (COLUMN, P, T), _convert_ppm_m),
    Call("columns.dry_air_column", (P, np.asarray([0.0, 10.0, 20.0, 30.0]))),
    Call("columns.column_from_mixing_ratio", (COLUMN * 1e-6, COLUMN * 1e5)),
    Call("columns.mixing_ratio_from_column", (COLUMN, COLUMN * 1e5)),
    # differences of nearby sines, in float32
    Call(
        "earth.cell_area",
        (np.linspace(-80, 80, 9), np.linspace(0, 40, 6)),
        rtol=1e-12,
        rtol32=1e-4,
    ),
    Call("earth.great_circle_distance", (LON, LAT, LON2, LAT2), rtol=1e-12),
    Call("earth.initial_azimuth", (LON, LAT, LON2, LAT2), rtol=1e-12, rtol32=1e-4),
    # an Earth-centred coordinate resolves 0.5 m in float32
    Call("earth.geodetic_to_ecef", (LON, LAT, HEIGHT), rtol=1e-12, atol32=2.0),
    # h is a difference of two 6.4e6 m terms: 1e-9 m in float64
    Call("earth.ecef_to_geodetic", XYZ, atol=1e-8, atol32=2.0),
    Call("earth.meridian_degree_length", (LAT,)),
    Call("earth.meridional_radius", (LAT,)),
    Call("earth.parallel_degree_length", (LAT,)),
    Call("earth.parallel_radius", (LAT,)),
    Call("earth.prime_vertical_radius", (LAT,)),
    Call("rotation.beta", (LAT,)),
    Call("rotation.beta_plane", (HEIGHT,), _beta_plane),
    Call("rotation.coriolis", (LAT,)),
    # ρ − ρ₀ in float32, where ρ resolves 1e-4 kg m⁻³
    Call("seawater.buoyancy", (RHO,), rtol=1e-12, rtol32=1e-4),
    Call("seawater.linear_density", (SEA_T, SALT)),
    Call("seawater.linear_density_anomaly", (SEA_T, SALT), rtol=1e-12),
    Call("seawater.reduced_gravity", (RHO[:-1], RHO[1:]), rtol=1e-12, rtol32=1e-4),
    Call("thermo.dry_air_density", (P, T)),
    Call("thermo.exner", (P,)),
    Call("thermo.hypsometric_thickness", (P, T)),
    Call("thermo.number_density", (P, T)),
    Call("thermo.potential_temperature", (T, P)),
    Call("thermo.pressure_velocity", (OMEGA, P, T)),
    Call("thermo.temperature_from_potential_temperature", (T, P)),
    Call("thermo.vertical_velocity", (OMEGA, P, T)),
    Call("thermo.virtual_temperature", (T, COLUMN * 1e-3)),
    Call("wind.wind_components", (COLUMN, np.asarray([10.0, 95.0, 200.0, 300.0]))),
    Call("wind.wind_direction", (U, V)),
    Call("wind.wind_speed", (U, V)),
]
