#!/usr/bin/env python3
"""Mount Huangshan case study on a real DEM.

Runs two delivery tasks -- AB (Tangkou Town -> Xihai Hotel) and CD (Kuzhuxi
Village -> camping base) -- each under two altitude caps (120 m and 300 m AGL),
and produces 3-D, top-down, and vertical-profile figures plus a results table.

Usage
-----
1. Put the DEM path in ``DEM_PATH`` below (a GeoTIFF such as an ASTER GDEM V3
   tile, or any raster rasterio can read).
2. Fill in the four delivery points in ``POINTS`` as (lon, lat) in degrees, or
   set ``POINTS_ARE_LOCAL = True`` and give them as (x, y) metres in the DEM's
   local frame.
3. Run from the repository root::

       python scripts/run_huangshan.py

Requires ``rasterio`` (``pip install rasterio``).
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_routing import DroneParams, RoutePlanner            # noqa: E402
from uav_routing.dem import load_dem, lonlat_to_local        # noqa: E402

# --------------------------------------------------------------------------- #
# CONFIGURATION -- edit these.
# --------------------------------------------------------------------------- #
# Place your DEM here (e.g. the ASTER GDEM V3 tile ASTGTMV003_N30E118_dem.tif).
DEM_PATH = "data/ASTGTMV003_N30E118_dem.tif"
DST_EPSG = 32650                             # UTM zone 50N (as in the paper)
BBOX_LONLAT = (118.10, 30.06, 118.22, 30.18) # crop to the Huangshan core area
DOWNSAMPLE = 1                               # increase to coarsen a very large tile
SMOOTH_SIGMA = 2.0                           # Gaussian smoothing (pixels) for stable gradients

POINTS_ARE_LOCAL = False                     # False: POINTS are (lon, lat) degrees
POINTS = {
    "A": (118.1447, 30.0876),   # Tangkou Town, southern foot (origin of task AB)
    "B": (118.1556, 30.1375),   # Xihai Hotel, summit region  (destination of task AB)
    "C": (118.1500, 30.1700),   # northern access village     (origin of task CD)
    "D": (118.1566, 30.1289),   # Bright Summit camp          (destination of task CD)
}

ALT_CAPS = [120.0, 300.0]                    # h_max settings (m AGL)
N_NODES = 60
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG = os.path.join(RESULTS, "figures")
TAB = os.path.join(RESULTS, "tables")


def _endpoint(terrain, xy, params):
    """Lift a horizontal point to a valid start/end altitude."""
    x, y = xy
    zg = float(np.asarray(terrain.height(x, y)))
    return (x, y, zg + params.h_min + 5.0)


def _feasibility(terrain, params, res, tol=1e-2):
    """Return (all_constraints_satisfied, objective_plateaued)."""
    zg = np.asarray(terrain.height(res.nodes[:, 0], res.nodes[:, 1]))
    clr = res.nodes[:, 2] - zg
    ok = bool(
        clr.min() >= params.h_min - 1.0
        and (params.h_max is None or clr.max() <= params.h_max + 1.0)
        and res.speeds.min() >= params.v_min - 1e-2
        and res.speeds.max() <= params.v_max + 1e-2
    )
    h = np.asarray(res.history, dtype=float)
    tail = h[-15:] if h.size >= 15 else h
    stable = bool(tail.size > 0 and (tail.max() - tail.min()) / max(abs(tail[-1]), 1.0) < tol)
    return ok, stable


def _to_local(terrain, meta):
    pts = {}
    for k, v in POINTS.items():
        if v[0] is None:
            raise SystemExit(f"Point {k} is not set -- fill in POINTS in this script.")
        pts[k] = tuple(v) if POINTS_ARE_LOCAL else lonlat_to_local(v[0], v[1], meta)
    return pts


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    if not os.path.exists(DEM_PATH):
        raise SystemExit(f"DEM not found at {DEM_PATH}. Set DEM_PATH in this script.")

    terrain, meta = load_dem(DEM_PATH, bbox_lonlat=BBOX_LONLAT, dst_epsg=DST_EPSG,
                             downsample=DOWNSAMPLE, smooth_sigma=SMOOTH_SIGMA)
    pts = _to_local(terrain, meta)
    print(f"DEM loaded: {terrain.heights.shape} grid, "
          f"elevation {terrain.heights.min():.0f}-{terrain.heights.max():.0f} m")

    tasks = {"AB": ("A", "B"), "CD": ("C", "D")}
    results = {}   # (task, cap) -> RouteResult
    rows = []
    for task, (o_key, d_key) in tasks.items():
        dist = float(np.hypot(pts[d_key][0] - pts[o_key][0], pts[d_key][1] - pts[o_key][1]))
        offs = tuple(f * dist for f in (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4))
        for cap in ALT_CAPS:
            params = DroneParams(h_max=cap)
            origin = _endpoint(terrain, pts[o_key], params)
            dest = _endpoint(terrain, pts[d_key], params)
            planner = RoutePlanner(terrain, params, origin, dest, n_nodes=N_NODES)
            res = planner.solve_multistart(offsets=offs, max_iter=300)
            feasible, stable = _feasibility(terrain, params, res)
            results[(task, cap)] = res
            rows.append({
                "task": task, "h_max_m": int(cap),
                "energy_kJ": round(res.energy / 1e3, 1),
                "climb_kJ": round(res.energy_gravity / 1e3, 1),
                "drag_kJ": round(res.energy_drag / 1e3, 1),
                "length_m": round(res.path_length, 2),
                "max_alt_m": round(float(res.nodes[:, 2].max()), 1),
                "flight_time_s": round(res.flight_time, 1),
                "feasible": feasible, "obj_stable": stable,
            })
            print(f"  task {task}  h_max={cap:>4.0f} m : E={res.energy/1e3:8.1f} kJ  "
                  f"L={res.path_length:9.1f} m  feasible={feasible} stable={stable}")

    _write_table(rows)
    _plot(terrain, pts, tasks, results)
    print(f"\nWrote figures to {FIG} and tables to {TAB}")


def _write_table(rows):
    fields = list(rows[0].keys())
    with open(os.path.join(TAB, "huangshan_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(os.path.join(TAB, "huangshan_results.md"), "w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in fields) + " |\n")


def _grid(terrain, n=220):
    xs = np.linspace(terrain.x_coords[0], terrain.x_coords[-1], n)
    ys = np.linspace(terrain.y_coords[0], terrain.y_coords[-1], n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[float(np.asarray(terrain.height(x, y))) for x in xs] for y in ys])
    return X, Y, Z


def _plot(terrain, pts, tasks, results):
    X, Y, Z = _grid(terrain)
    colors = {"AB": "crimson", "CD": "royalblue"}

    # 3-D surface with both routes (use the 300 m cap for the 3-D view).
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X / 1000, Y / 1000, Z, cmap="terrain", alpha=0.7, linewidth=0)
    for task in tasks:
        r = results[(task, 300.0)]
        ax.plot(r.nodes[:, 0] / 1000, r.nodes[:, 1] / 1000, r.nodes[:, 2],
                color=colors[task], lw=2.5, label=f"route {task}")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_zlabel("elevation (m)")
    ax.set_title("Mount Huangshan — optimized routes (h_max = 300 m)")
    ax.legend(); ax.view_init(elev=45, azim=-120)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "huangshan_3d.png"), dpi=150); plt.close(fig)

    # Top-down contour with both routes and both caps.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cf = ax.contourf(X / 1000, Y / 1000, Z, levels=20, cmap="terrain")
    ax.contour(X / 1000, Y / 1000, Z, levels=20, colors="k", linewidths=0.2, alpha=0.35)
    styles = {120.0: "-", 300.0: "--"}
    for task in tasks:
        for cap in [120.0, 300.0]:
            r = results[(task, cap)]
            ax.plot(r.nodes[:, 0] / 1000, r.nodes[:, 1] / 1000, styles[cap],
                    color=colors[task], lw=2.0, label=f"{task}, h_max={int(cap)} m")
    for k, (x, y) in pts.items():
        ax.scatter(x / 1000, y / 1000, color="black", s=35, zorder=5)
        ax.annotate(k, (x / 1000, y / 1000), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_aspect("equal")
    ax.set_title("Mount Huangshan — route top view")
    ax.legend(fontsize=8, loc="best")
    plt.colorbar(cf, ax=ax, label="elevation (m)", shrink=0.85)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "huangshan_top.png"), dpi=150); plt.close(fig)

    # Vertical profiles: 2x2 (task x cap).
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for i, task in enumerate(tasks):
        for j, cap in enumerate([120.0, 300.0]):
            ax = axes[i, j]
            r = results[(task, cap)]
            seg = np.linalg.norm(np.diff(r.nodes, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)]) / 1000
            ground = np.array([float(np.asarray(terrain.height(x, y)))
                               for x, y in r.nodes[:, :2]])
            ax.fill_between(s, ground.min() - 50, ground, color="0.75")
            ax.plot(s, r.nodes[:, 2], color=colors[task], lw=2.0)
            ax.set_title(f"route {task}, h_max = {int(cap)} m")
            ax.set_xlabel("distance (km)"); ax.set_ylabel("altitude (m)"); ax.margins(x=0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "huangshan_profiles.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
