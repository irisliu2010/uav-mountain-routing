#!/usr/bin/env python3
"""Minimal end-to-end example: solve one scenario and save two figures.

    python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_routing import DroneParams, RoutePlanner
from uav_routing.scenarios import get_scenario
from uav_routing import plotting


def main():
    # Pick a scenario and a vehicle.
    scenario = get_scenario("S1")                 # "Central barrier"
    params = DroneParams(payload=10.0, h_max=None)  # DJI FlyCart 30 class

    # Build the planner and solve (multi-start to avoid poor local minima).
    planner = RoutePlanner(
        scenario.terrain, params, scenario.origin, scenario.destination, n_nodes=40
    )
    result = planner.solve_multistart()

    print(f"converged      : {result.success}")
    print(f"total energy   : {result.energy/1e3:.1f} kJ")
    print(f"  climb part   : {result.energy_gravity/1e3:.1f} kJ")
    print(f"  drag part    : {result.energy_drag/1e3:.1f} kJ")
    print(f"path length    : {result.path_length:.0f} m")
    print(f"flight time    : {result.flight_time:.0f} s")

    # Plot.
    ax = plotting.plot_route_top(scenario, result)
    ax.figure.tight_layout()
    ax.figure.savefig("quickstart_top.png", dpi=150)

    ax2 = plotting.plot_profile(scenario, result)
    ax2.figure.tight_layout()
    ax2.figure.savefig("quickstart_profile.png", dpi=150)
    print("\nsaved quickstart_top.png and quickstart_profile.png")


if __name__ == "__main__":
    main()
