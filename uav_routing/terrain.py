"""Terrain models.

A terrain is any object that maps horizontal coordinates ``(x, y)`` to a
ground elevation ``z_g(x, y)``.  For the trajectory optimiser to work well
the terrain must also expose an analytic (or smooth numerical) gradient.

Two concrete terrains are provided:

* :class:`GaussianTerrain` -- a sum of 2-D Gaussian bumps, used for the
  synthetic mountain scenarios.  Its gradient is analytic.
* :class:`GridTerrain` -- a bilinearly interpolated regular grid, suitable
  for a real digital elevation model (DEM).  Its gradient is computed from
  finite differences of the interpolated surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Terrain:
    """Abstract terrain interface."""

    def height(self, x, y):
        """Ground elevation at horizontal coordinates ``(x, y)``."""
        raise NotImplementedError

    def gradient(self, x, y):
        """Return ``(dz/dx, dz/dy)`` at ``(x, y)``."""
        raise NotImplementedError

    # Convenience alias so a terrain is callable.
    def __call__(self, x, y):
        return self.height(x, y)


@dataclass
class Peak:
    """A single 2-D Gaussian bump.

    ``z(x, y) = amplitude * exp(-((x-cx)^2 + (y-cy)^2) / (2 sigma^2))``
    """

    cx: float
    cy: float
    amplitude: float
    sigma: float


@dataclass
class GaussianTerrain(Terrain):
    """Terrain built as a sum of :class:`Peak` Gaussians.

    Parameters
    ----------
    peaks : list of Peak
        The Gaussian bumps that make up the surface.
    base : float
        A constant elevation offset (a flat "sea level").
    """

    peaks: list[Peak]
    base: float = 0.0

    def height(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.full(np.broadcast(x, y).shape, float(self.base))
        for p in self.peaks:
            r2 = (x - p.cx) ** 2 + (y - p.cy) ** 2
            z = z + p.amplitude * np.exp(-r2 / (2.0 * p.sigma ** 2))
        return z

    def gradient(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        gx = np.zeros(np.broadcast(x, y).shape, dtype=float)
        gy = np.zeros_like(gx)
        for p in self.peaks:
            dx = x - p.cx
            dy = y - p.cy
            g = p.amplitude * np.exp(-(dx ** 2 + dy ** 2) / (2.0 * p.sigma ** 2))
            gx = gx - g * dx / p.sigma ** 2
            gy = gy - g * dy / p.sigma ** 2
        return gx, gy

    @property
    def max_height(self) -> float:
        return float(self.base + sum(max(p.amplitude, 0.0) for p in self.peaks))


@dataclass
class GridTerrain(Terrain):
    """Bilinearly interpolated regular-grid DEM.

    Parameters
    ----------
    x_coords, y_coords : 1-D arrays
        Strictly increasing grid coordinates (metres).
    heights : 2-D array of shape (len(y_coords), len(x_coords))
        Elevation samples; ``heights[j, i]`` is the elevation at
        ``(x_coords[i], y_coords[j])``.
    """

    x_coords: np.ndarray
    y_coords: np.ndarray
    heights: np.ndarray
    _dx: float = field(init=False)
    _dy: float = field(init=False)

    def __post_init__(self):
        self.x_coords = np.asarray(self.x_coords, dtype=float)
        self.y_coords = np.asarray(self.y_coords, dtype=float)
        self.heights = np.asarray(self.heights, dtype=float)
        self._dx = float(self.x_coords[1] - self.x_coords[0])
        self._dy = float(self.y_coords[1] - self.y_coords[0])

    def _locate(self, coords, grid, spacing):
        idx = np.clip(
            np.floor((coords - grid[0]) / spacing).astype(int), 0, len(grid) - 2
        )
        frac = (coords - grid[idx]) / spacing
        return idx, frac

    def height(self, x, y):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        ix, fx = self._locate(x, self.x_coords, self._dx)
        iy, fy = self._locate(y, self.y_coords, self._dy)
        h = self.heights
        z00 = h[iy, ix]
        z10 = h[iy, ix + 1]
        z01 = h[iy + 1, ix]
        z11 = h[iy + 1, ix + 1]
        z = (
            z00 * (1 - fx) * (1 - fy)
            + z10 * fx * (1 - fy)
            + z01 * (1 - fx) * fy
            + z11 * fx * fy
        )
        return z if z.size > 1 else float(z[0])

    def gradient(self, x, y):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        eps_x = 0.5 * self._dx
        eps_y = 0.5 * self._dy
        gx = (np.asarray(self.height(x + eps_x, y)) - np.asarray(self.height(x - eps_x, y))) / (2 * eps_x)
        gy = (np.asarray(self.height(x, y + eps_y)) - np.asarray(self.height(x, y - eps_y))) / (2 * eps_y)
        if gx.size == 1:
            return float(gx), float(gy)
        return gx, gy

    @property
    def max_height(self) -> float:
        return float(np.max(self.heights))
