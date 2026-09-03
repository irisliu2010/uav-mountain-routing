"""Plotting helpers for terrain, routes, profiles, and convergence."""

from __future__ import annotations

import numpy as np

# Use a non-interactive backend so scripts run headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

from .scenarios import REGION_SIZE


def _terrain_grid(terrain, n=160):
    xs = np.linspace(0, REGION_SIZE, n)
    ys = np.linspace(0, REGION_SIZE, n)
    X, Y = np.meshgrid(xs, ys)
    Z = terrain.height(X, Y)
    return X, Y, Z


def plot_route_3d(scenario, result, ax=None):
    """3-D surface with the optimized route overlaid."""
    if ax is None:
        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = _terrain_grid(scenario.terrain)
    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.75, linewidth=0, antialiased=True)
    nodes = result.nodes
    ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="crimson", lw=2.4, label="route")
    ax.scatter(*nodes[0], color="black", s=40, marker="o")
    ax.scatter(*nodes[-1], color="black", s=40, marker="^")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{scenario.key} — {scenario.name}")
    ax.view_init(elev=42, azim=-125)
    return ax


def plot_route_top(scenario, result, ax=None):
    """Top-down contour view with the horizontal route projection."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))
    X, Y, Z = _terrain_grid(scenario.terrain)
    cf = ax.contourf(X, Y, Z, levels=18, cmap="terrain")
    ax.contour(X, Y, Z, levels=18, colors="k", linewidths=0.25, alpha=0.4)
    nodes = result.nodes
    ax.plot(nodes[:, 0], nodes[:, 1], color="crimson", lw=2.2, label="route")
    ax.plot([nodes[0, 0], nodes[-1, 0]], [nodes[0, 1], nodes[-1, 1]],
            "--", color="white", lw=1.0, alpha=0.8, label="straight line")
    ax.scatter(nodes[0, 0], nodes[0, 1], color="black", s=40, marker="o", label="origin")
    ax.scatter(nodes[-1, 0], nodes[-1, 1], color="black", s=45, marker="^", label="dest.")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{scenario.key} — {scenario.name}")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    plt.colorbar(cf, ax=ax, label="elevation (m)", shrink=0.85)
    return ax


def plot_profile(scenario, result, ax=None):
    """Vertical cross-section: route altitude and ground beneath it."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))
    nodes = result.nodes
    seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)]) / 1000.0  # km along path
    ground = np.asarray(scenario.terrain.height(nodes[:, 0], nodes[:, 1]))
    ax.fill_between(s, 0, ground, color="0.75", label="ground beneath route")
    ax.plot(s, nodes[:, 2], color="crimson", lw=2.0, label="route altitude")
    ax.set_xlabel("distance along route (km)")
    ax.set_ylabel("altitude (m)")
    ax.set_title(f"{scenario.key} — {scenario.name}")
    ax.legend(loc="upper left", fontsize=8)
    ax.margins(x=0)
    return ax


def plot_routes_grid(scenarios, results_by_key, kind="top", ncols=3):
    """Lay out per-scenario route plots on a grid (default 2 rows x 3 cols).

    ``kind="top"`` uses :func:`plot_route_top`; ``kind="3d"`` uses
    :func:`plot_route_3d`. Any unused cells (e.g. the 6th cell for five
    scenarios) are left blank. Returns the Matplotlib figure.
    """
    n = len(scenarios)
    nrows = int(np.ceil(n / ncols))
    if kind == "top":
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for i, sc in enumerate(scenarios):
            plot_route_top(sc, results_by_key[sc.key], ax=axes[i])
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
    elif kind == "3d":
        fig = plt.figure(figsize=(5.5 * ncols, 5.75 * nrows))
        for i, sc in enumerate(scenarios):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            plot_route_3d(sc, results_by_key[sc.key], ax=ax)
            ax.set_title(f"{sc.key} — {sc.name}", pad=12)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02,
                            wspace=0.10, hspace=0.16)
    else:
        raise ValueError("kind must be 'top' or '3d'")
    return fig


def plot_convergence(results_by_key, ax=None):
    """Normalized convergence curves (best objective so far) per scenario.

    SLSQP evaluates trial points inside its line search that can transiently
    raise the objective; we plot the running minimum (best feasible value seen
    so far), normalized by each run's first value, which is the standard way to
    display convergence and removes line-search transients.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, res in results_by_key.items():
        h = np.asarray(res.history, dtype=float)
        if h.size == 0:
            continue
        best = np.minimum.accumulate(h)
        ax.plot(np.arange(1, best.size + 1), best / best[0], lw=1.9, label=key)
    ax.set_xlabel("SLSQP iteration")
    ax.set_ylabel("normalized objective  E / E$_0$")
    ax.set_title("Convergence (best objective so far)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax
