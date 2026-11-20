"""Physical constants, Earth geometry and atmospheric state, on NumPy with JAX optional.

``geophysix`` is the stack's library of closed-form physical facts about the
Earth, its air and its sea water. It sits at tier 0 of GeoNumerics and imports
nothing from the rest of the GeoStack.

## Planned modules

| Module | What it gives you |
|---|---|
| `geophysix.constants` | CODATA constants, standard gravity, dry air and trace gases |
| `geophysix.earth` | the ellipsoid and the sphere |
| `geophysix.rotation` | the Earth's rotation, Coriolis |
| `geophysix.thermo` | atmospheric thermodynamics |
| `geophysix.seawater` | the sea-water equation of state |
| `geophysix.columns` | column units |

Every module exists; TEOS-10 waits on #34.
"""

from __future__ import annotations

from geophysix.exceptions import (
    EarthError,
    GeophysixError,
    RangeError,
    UnitError,
)


__version__ = "0.0.2"

__all__ = [
    "EarthError",
    "GeophysixError",
    "RangeError",
    "UnitError",
    "__version__",
]
