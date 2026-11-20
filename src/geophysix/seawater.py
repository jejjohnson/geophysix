"""The equation of state of sea water: the linear one, buoyancy, reduced gravity.

``LINEAR_EOS`` holds the linear equation of state's parameters (gridax's
defaults); each function takes it as ``eos=``. TEOS-10 waits on #34.
"""

from __future__ import annotations

from geophysix._src.seawater.linear import (
    LINEAR_EOS,
    LinearEOS,
    buoyancy,
    linear_density,
    linear_density_anomaly,
    linear_drho_dS,
    linear_drho_dT,
    reduced_gravity,
)


__all__ = [
    "LINEAR_EOS",
    "LinearEOS",
    "buoyancy",
    "linear_density",
    "linear_density_anomaly",
    "linear_drho_dS",
    "linear_drho_dT",
    "reduced_gravity",
]
