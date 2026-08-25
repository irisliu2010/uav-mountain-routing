"""Small helpers for reporting and terrain statistics."""

from __future__ import annotations

import numpy as np

from .scenarios import REGION_SIZE


def terrain_max_elevation(terrain, n: int = 300) -> float:
    """Actual maximum ground elevation of a terrain on the region grid."""
    xs = np.linspace(0, REGION_SIZE, n)
    ys = np.linspace(0, REGION_SIZE, n)
    X, Y = np.meshgrid(xs, ys)
    return float(np.max(terrain.height(X, Y)))


def straight_line_length(origin, destination) -> float:
    """Horizontal straight-line distance between origin and destination."""
    o = np.asarray(origin, dtype=float)
    d = np.asarray(destination, dtype=float)
    return float(np.linalg.norm((d - o)[:2]))
