"""Physical constants, Earth geometry and atmospheric state, on any array library.

``geophysix`` is the stack's library of closed-form physical facts about the
Earth, its air and its sea water. It sits at tier 0 of GeoNumerics and imports
nothing from the rest of the GeoStack but the array backend of pipekit-array.

## Planned modules

| Module | What it gives you |
|---|---|
| `geophysix.constants` | CODATA constants, standard gravity, dry air and trace gases |
| `geophysix.earth` | the ellipsoid and the sphere |
| `geophysix.rotation` | the Earth's rotation, Coriolis |
| `geophysix.thermo` | atmospheric thermodynamics |
| `geophysix.wind` | the horizontal wind's speed, bearing and components |
| `geophysix.seawater` | the sea-water equation of state |
| `geophysix.columns` | column units |

Every module exists; TEOS-10 waits on #34.
"""

from __future__ import annotations

from geophysix.exceptions import (
    BackendError,
    EarthError,
    GeophysixError,
    RangeError,
    UnitError,
)


__version__ = "0.0.5"

__all__ = [
    "BackendError",
    "EarthError",
    "GeophysixError",
    "RangeError",
    "UnitError",
    "__version__",
]
