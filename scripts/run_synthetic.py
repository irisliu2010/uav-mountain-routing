#!/usr/bin/env python3
"""Run the five synthetic scenarios and produce all route figures and tables.

Outputs (under ``results/``):

* ``figures/routes_3d.png``       3-D terrain + route for every scenario
* ``figures/routes_top.png``      top-down contour + horizontal route
* ``figures/profiles.png``        vertical cross-sections
* ``figures/convergence.png``     normalised convergence curves
* ``tables/synthetic_results.csv``  and ``.md``  summary table

Run from the repository root::

    python scripts/run_synthetic.py
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running the script directly from a clean clone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_routing import DroneParams, RoutePlanner            # noqa: E402
from uav_routing.scenarios import all_scenarios              # noqa: E402
from uav_routing.metrics import terrain_max_elevation, straight_line_length  # noqa: E402
from uav_routing import plotting                             # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG = os.path.join(RESULTS, "figures")
TAB = os.path.join(RESULTS, "tables")
N_NODES = 40


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)

    params = DroneParams(h_max=None)     # no altitude cap for synthetic study
    scenarios = all_scenarios()
    results = {}
    rows = []

    for sc in scenarios:
        planner = RoutePlanner(sc.terrain, params, sc.origin, sc.destination, n_nodes=N_NODES)
        res = planner.solve_multistart()
        results[sc.key] = res
        straight = straight_line_length(sc.origin, sc.destination)
        rows.append(
            {
                "scenario": sc.key,
                "name": sc.name,
                "n_peaks": len(sc.terrain.peaks),
                "max_terrain_m": round(terrain_max_elevation(sc.terrain), 1),
                "energy_kJ": round(res.energy / 1e3, 1),
                "climb_energy_kJ": round(res.energy_gravity / 1e3, 1),
                "drag_energy_kJ": round(res.energy_drag / 1e3, 1),
                "path_length_m": round(res.path_length, 1),
                "detour_pct": round(100 * (res.path_length - straight) / straight, 2),
                "flight_time_s": round(res.flight_time, 1),
                "max_route_alt_m": round(float(res.nodes[:, 2].max()), 1),
                "iterations": res.n_iter,
                "converged": res.success,
            }
        )
        print(
            f"{sc.key} {sc.name:<16} E={res.energy/1e3:8.1f} kJ  "
            f"L={res.path_length:8.0f} m  iters={res.n_iter:3d}  ok={res.success}"
        )

    _write_tables(rows)
    _plot_all(scenarios, results)
    print(f"\nWrote figures to {FIG} and tables to {TAB}")


def _write_tables(rows):
    fields = list(rows[0].keys())
    csv_path = os.path.join(TAB, "synthetic_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(TAB, "synthetic_results.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in fields) + " |\n")


def _plot_all(scenarios, results):
    n = len(scenarios)

    # 3-D routes
    fig = plt.figure(figsize=(5 * n, 4.4))
    for i, sc in enumerate(scenarios):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plotting.plot_route_3d(sc, results[sc.key], ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "routes_3d.png"), dpi=150)
    plt.close(fig)

    # top-down
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.6))
    for ax, sc in zip(axes, scenarios):
        plotting.plot_route_top(sc, results[sc.key], ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "routes_top.png"), dpi=150)
    plt.close(fig)

    # profiles
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.6 * n))
    for ax, sc in zip(axes, scenarios):
        plotting.plot_profile(sc, results[sc.key], ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "profiles.png"), dpi=150)
    plt.close(fig)

    # convergence
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plotting.plot_convergence(results, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
