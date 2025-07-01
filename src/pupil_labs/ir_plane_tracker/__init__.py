"""pupil_labs.ir_plane_tracker package.

IR Plane Tracker
"""
from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__: list[str] = ["__version__"]
