"""Physical constants and trace gases, as frozen records in SI units.

``STANDARD`` holds CODATA 2018's exact constants, standard gravity, the
standard atmosphere and dry air; ``CH4``, ``CO2``, ``H2O``, ``N2O`` and ``CO``
hold molar masses. A function that needs a constant takes the record as a
keyword argument with ``STANDARD`` as its default.
"""

from __future__ import annotations

from geophysix._src.constants.records import (
    CH4,
    CO,
    CO2,
    H2O,
    N2O,
    STANDARD,
    Constants,
    Gas,
)


__all__ = [
    "CH4",
    "CO",
    "CO2",
    "Constants",
    "Gas",
    "H2O",
    "N2O",
    "STANDARD",
]
