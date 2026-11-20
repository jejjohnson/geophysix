"""The Earth's rotation: the Coriolis parameter, its gradient, and the β-plane.

Each function takes the body as ``earth=`` (WGS84 by default), whose
``rotation_rate`` is Ω.
"""

from __future__ import annotations

from geophysix._src.rotation.coriolis import beta, beta_plane, coriolis


__all__ = [
    "beta",
    "beta_plane",
    "coriolis",
]
