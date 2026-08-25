#!/usr/bin/env python3
"""Payload-mass sensitivity analysis.

For each scenario the total mass is swept over ``{40, 45, 50, 55} kg``
(payload ``{10, 15, 20, 25} kg``) and the route is re-optimised.  The study
is run under both energy models:

* ``non-recuperative`` (default, descent power floored at zero): climbing is
  genuinely costly, so heavier drones detour more to avoid climbs and total
  energy grows super-linearly.
* ``recuperative`` (paper eq. 5, descent credited): the gravity term
  telescopes to ``m g (z_D - z_O)/eta`` and the *optimisable* part of the
  objective is mass-independent, so the optimal route is invariant to mass
  and energy is exactly affine in mass.  This contrast is itself a result.

Outputs (under ``results/``):

* ``figures/sensitivity.png``
* ``tables/sensitivity.csv`` and ``.md``

Run from the repository root::

    python scripts/run_sensitivity.py
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_routing import DroneParams, RoutePlanner            # noqa: E402
from uav_routing.scenarios import all_scenarios              # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIG = os.path.join(RESULTS, "figures")
TAB = os.path.join(RESULTS, "tables")

PAYLOADS = [10.0, 15.0, 20.0, 25.0]      # kg  -> total mass 40..55 kg
N_NODES = 40


def _x0_from(res):
    """Rebuild an SLSQP decision vector from a previous solution."""
    import numpy as np

    interior = res.nodes[1:-1].reshape(-1)
    return np.concatenate([interior, [res.flight_time]])


def _sweep(recovery: bool):
    """Return {scenario_key: {mass: (energy_kJ, length_m)}}.

    The lightest payload is solved with a full lateral multi-start to locate a
    good route basin; each heavier payload is then solved by *continuation*,
    warm-starting from the previous mass's solution.  This keeps the whole
    route family inside one consistent corridor, so the mass trend is not
    contaminated by the optimiser jumping between distinct local minima.
    """
    out = {}
    for sc in all_scenarios():
        per_mass = {}
        prev = None
        for pay in PAYLOADS:
            params = DroneParams(payload=pay, h_max=None)
            planner = RoutePlanner(
                sc.terrain, params, sc.origin, sc.destination,
                n_nodes=N_NODES, descent_recovery=recovery,
            )
            if prev is None:
                res = planner.solve_multistart()
            else:
                res = planner.solve(x0=_x0_from(prev))
            prev = res
            per_mass[params.mass] = (res.energy / 1e3, res.path_length)
        out[sc.key] = per_mass
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)

    print("Sweeping payload mass (non-recuperative model)...")
    nonrec = _sweep(recovery=False)
    print("Sweeping payload mass (recuperative model)...")
    rec = _sweep(recovery=True)

    _write_table(nonrec, rec)
    _plot(nonrec, rec)
    print(f"\nWrote figures to {FIG} and tables to {TAB}")


def _write_table(nonrec, rec):
    masses = sorted(next(iter(nonrec.values())).keys())
    rows = []
    for key in nonrec:
        for m in masses:
            e_nr, l_nr = nonrec[key][m]
            e_r, l_r = rec[key][m]
            rows.append(
                {
                    "scenario": key,
                    "mass_kg": m,
                    "payload_kg": m - 30,
                    "energy_nonrec_kJ": round(e_nr, 1),
                    "length_nonrec_m": round(l_nr, 1),
                    "energy_rec_kJ": round(e_r, 1),
                    "length_rec_m": round(l_r, 1),
                }
            )
    fields = list(rows[0].keys())
    with open(os.path.join(TAB, "sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(TAB, "sensitivity.md"), "w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in fields) + " |\n")


def _plot(nonrec, rec):
    masses = sorted(next(iter(nonrec.values())).keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) energy vs mass, non-recuperative
    for key, d in nonrec.items():
        axes[0].plot(masses, [d[m][0] for m in masses], "o-", label=key)
    axes[0].set_title("(a) Energy vs mass — non-recuperative")
    axes[0].set_xlabel("total mass (kg)")
    axes[0].set_ylabel("total energy (kJ)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # (b) path length vs mass, non-recuperative
    for key, d in nonrec.items():
        axes[1].plot(masses, [d[m][1] / 1000 for m in masses], "s-", label=key)
    axes[1].set_title("(b) Path length vs mass — non-recuperative")
    axes[1].set_xlabel("total mass (kg)")
    axes[1].set_ylabel("path length (km)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # (c) energy vs mass, recuperative (affine, mass-independent route)
    for key, d in rec.items():
        axes[2].plot(masses, [d[m][0] for m in masses], "^--", label=key)
    axes[2].set_title("(c) Energy vs mass — recuperative (affine)")
    axes[2].set_xlabel("total mass (kg)")
    axes[2].set_ylabel("total energy (kJ)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sensitivity.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
