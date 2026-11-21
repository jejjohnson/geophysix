"""The horizontal wind: its speed, its bearing and its components.

A bearing is clockwise from north. ``convention="from"`` (the default, the
meteorological convention) names where the wind blows from, so a westerly is
270°; ``convention="to"`` names where it blows towards.
"""

from __future__ import annotations

from geophysix._src.wind.horizontal import (
    wind_components,
    wind_direction,
    wind_speed,
)


__all__ = [
    "wind_components",
    "wind_direction",
    "wind_speed",
]
