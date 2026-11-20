"""The thermodynamics of air: the ideal gas.

Each function takes the constants record as ``constants=`` (``STANDARD`` by
default).
"""

from __future__ import annotations

from geophysix._src.thermo.gas import number_density


__all__ = [
    "number_density",
]
