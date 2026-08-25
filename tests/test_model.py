"""Tests for the UAV routing model.

Run with ``pytest`` or directly with ``python tests/test_model.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_routing import DroneParams, RoutePlanner, GaussianTerrain, Peak
from uav_routing.scenarios import get_scenario, all_scenarios


def _planner(dr=True, n=12):
    sc = get_scenario("S1")
    p = DroneParams(h_max=None)
    return RoutePlanner(sc.terrain, p, sc.origin, sc.destination, n_nodes=n, descent_recovery=dr), p


def test_analytic_gradient_matches_finite_difference():
    for dr in (True, False):
        pl, _ = _planner(dr=dr)
        x0 = pl.initial_guess()
        rng = np.random.default_rng(0)
        x = x0 + rng.normal(scale=40, size=x0.shape)
        x[-1] = abs(x[-1]) + 120.0
        g = pl._gradient(x)
        gn = np.zeros_like(x)
        eps = 1e-2
        for i in range(len(x)):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            gn[i] = (pl._objective(xp) - pl._objective(xm)) / (2 * eps)
        rel = np.linalg.norm(g - gn) / np.linalg.norm(gn)
        # descent_recovery=False has a kink at dz=0; keep the tolerance loose.
        assert rel < (1e-6 if dr else 1e-3), f"dr={dr} rel err {rel}"


def test_recuperative_gravity_telescopes():
    """Under descent_recovery=True, gravity energy = m g (zD - zO) / eta."""
    pl, p = _planner(dr=True, n=20)
    rng = np.random.default_rng(1)
    x = pl.initial_guess() + rng.normal(scale=200, size=pl.initial_guess().shape)
    x[-1] = abs(x[-1]) + 100
    nodes, T = pl._unpack(x)
    d = np.diff(nodes, axis=0)
    e_grav = p.mass * p.gravity * np.sum(d[:, 2]) / p.prop_efficiency
    expected = p.mass * p.gravity * (pl.destination[2] - pl.origin[2]) / p.prop_efficiency
    assert abs(e_grav - expected) < 1e-6


def test_solution_is_feasible():
    sc = get_scenario("S2")
    p = DroneParams(h_max=None)
    pl = RoutePlanner(sc.terrain, p, sc.origin, sc.destination, n_nodes=40)
    res = pl.solve_multistart()
    assert res.success
    # clearance at every node
    zg = np.asarray(sc.terrain.height(res.nodes[:, 0], res.nodes[:, 1]))
    assert np.all(res.nodes[:, 2] - zg >= p.h_min - 1.0)
    # speed bounds (allow tiny numerical slack)
    assert res.speeds.min() >= p.v_min - 1e-3
    assert res.speeds.max() <= p.v_max + 1e-3
    # endpoints respected
    assert np.allclose(res.nodes[0], sc.origin)
    assert np.allclose(res.nodes[-1], sc.destination)


def test_recuperative_energy_is_affine_in_mass():
    sc = get_scenario("S4")
    energies = []
    prev = None
    for pay in (10.0, 15.0, 20.0, 25.0):
        p = DroneParams(payload=pay, h_max=None)
        pl = RoutePlanner(sc.terrain, p, sc.origin, sc.destination, n_nodes=40, descent_recovery=True)
        res = pl.solve_multistart() if prev is None else pl.solve(
            x0=np.concatenate([prev.nodes[1:-1].reshape(-1), [prev.flight_time]])
        )
        prev = res
        energies.append(res.energy)
    diffs = np.diff(energies)
    # equal steps (affine) — the per-5kg increment g*dz/eta*5 is constant.
    assert np.allclose(diffs, diffs[0], rtol=1e-3)


def test_scenarios_are_deterministic():
    a = get_scenario("S4").terrain.peaks
    from uav_routing import scenarios as s2
    import importlib
    importlib.reload(s2)
    b = s2.get_scenario("S4").terrain.peaks
    for pa, pb in zip(a, b):
        assert abs(pa.cx - pb.cx) < 1e-9 and abs(pa.amplitude - pb.amplitude) < 1e-9


def test_gaussian_terrain_gradient():
    terr = GaussianTerrain(peaks=[Peak(5000, 5000, 1500, 1200), Peak(8000, 6000, 900, 800)])
    rng = np.random.default_rng(3)
    for _ in range(5):
        x, y = rng.uniform(2000, 12000, size=2)
        gx, gy = terr.gradient(x, y)
        eps = 1.0
        gxn = (terr.height(x + eps, y) - terr.height(x - eps, y)) / (2 * eps)
        gyn = (terr.height(x, y + eps) - terr.height(x, y - eps)) / (2 * eps)
        assert abs(gx - gxn) < 1e-3 and abs(gy - gyn) < 1e-3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
