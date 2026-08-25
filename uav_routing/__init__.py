"""Energy-efficient UAV route planning over mountainous terrain.

A compact, reproducible SciPy implementation of the trajectory-optimisation
model described in the accompanying paper.
"""

from .params import DroneParams
from .terrain import Terrain, GaussianTerrain, GridTerrain, Peak
from .optimizer import RoutePlanner, RouteResult
from . import scenarios

__all__ = [
    "DroneParams",
    "Terrain",
    "GaussianTerrain",
    "GridTerrain",
    "Peak",
    "RoutePlanner",
    "RouteResult",
    "scenarios",
]

__version__ = "1.0.0"
