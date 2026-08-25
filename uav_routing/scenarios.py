"""Synthetic mountain scenarios (the numerical dataset).

This module defines a **new** set of five synthetic terrain scenarios used
for the numerical experiments.  Each scenario is a
:class:`~uav_routing.terrain.GaussianTerrain` on a 15 km x 15 km region,
together with a fixed origin/destination pair for the delivery task.

The five scenarios span a gradient of routing difficulty:

===========  ==========================================================
S1 Central   one dominant central massif that blocks the straight route
S2 TwinSaddle  two tall peaks with a low saddle corridor between them
S3 Ridge     a ridge running across the origin-destination diagonal
S4 Massif    a dense cluster of many peaks (procedurally generated)
S5 Hills     scattered low hills (procedurally generated)
===========  ==========================================================

The procedurally generated scenarios (S4, S5) use ``numpy.random.default_rng``
with fixed seeds, whose PCG64 bit-stream is stable across NumPy versions, so
the whole dataset is fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terrain import GaussianTerrain, Peak

# Common mission geometry for every synthetic scenario.
REGION_SIZE = 15000.0                      # metres (square region side)
ORIGIN = (0.0, 0.0, 100.0)                 # (x, y, z) metres
DESTINATION = (15000.0, 15000.0, 200.0)    # (x, y, z) metres


@dataclass
class Scenario:
    key: str
    name: str
    terrain: GaussianTerrain
    origin: tuple = ORIGIN
    destination: tuple = DESTINATION

    @property
    def max_height(self) -> float:
        return self.terrain.max_height


def _random_peaks(seed, n_peaks, amp_range, sigma_range, margin=2000.0):
    """Deterministically sample ``n_peaks`` Gaussian bumps."""
    rng = np.random.default_rng(seed)
    lo, hi = margin, REGION_SIZE - margin
    peaks = []
    for _ in range(n_peaks):
        cx = rng.uniform(lo, hi)
        cy = rng.uniform(lo, hi)
        amp = rng.uniform(*amp_range)
        sigma = rng.uniform(*sigma_range)
        peaks.append(Peak(cx, cy, amp, sigma))
    return peaks


# --------------------------------------------------------------------------- #
# S1  Central barrier: one big massif right on the diagonal.
# --------------------------------------------------------------------------- #
_S1 = GaussianTerrain(
    peaks=[
        Peak(7500, 7500, 2200, 1500),
        Peak(6000, 8600, 1300, 900),
        Peak(9000, 6400, 1400, 950),
        Peak(4000, 4200, 700, 1600),
        Peak(11200, 11000, 800, 1600),
    ]
)

# --------------------------------------------------------------------------- #
# S2  Twin saddle: two tall peaks flanking a low corridor near the diagonal.
# --------------------------------------------------------------------------- #
_S2 = GaussianTerrain(
    peaks=[
        Peak(6000, 8600, 2300, 1000),
        Peak(9200, 6200, 2300, 1000),
        Peak(3600, 6000, 1200, 1100),
        Peak(11400, 9200, 1200, 1100),
        Peak(7600, 7300, 500, 700),
    ]
)

# --------------------------------------------------------------------------- #
# S3  Diagonal ridge: overlapping peaks forming a wall across O->D.
# --------------------------------------------------------------------------- #
_S3 = GaussianTerrain(
    peaks=[
        Peak(4500, 10500, 1700, 1200),
        Peak(6500, 8500, 1950, 1100),
        Peak(8500, 6500, 1950, 1100),
        Peak(10500, 4500, 1700, 1200),
    ]
)

# --------------------------------------------------------------------------- #
# S4  Dense massif: many peaks, procedurally generated.
# --------------------------------------------------------------------------- #
_S4 = GaussianTerrain(
    peaks=_random_peaks(
        seed=20260824, n_peaks=18, amp_range=(700, 2400), sigma_range=(500, 1400)
    )
)

# --------------------------------------------------------------------------- #
# S5  Scattered hills: fewer, lower, gentler bumps.
# --------------------------------------------------------------------------- #
_S5 = GaussianTerrain(
    peaks=_random_peaks(
        seed=71013, n_peaks=10, amp_range=(400, 1200), sigma_range=(900, 1800)
    )
)


SCENARIOS = {
    "S1": Scenario("S1", "Central barrier", _S1),
    "S2": Scenario("S2", "Twin saddle", _S2),
    "S3": Scenario("S3", "Diagonal ridge", _S3),
    "S4": Scenario("S4", "Dense massif", _S4),
    "S5": Scenario("S5", "Scattered hills", _S5),
}


def get_scenario(key: str) -> Scenario:
    return SCENARIOS[key]


def all_scenarios() -> list[Scenario]:
    return [SCENARIOS[k] for k in ("S1", "S2", "S3", "S4", "S5")]
