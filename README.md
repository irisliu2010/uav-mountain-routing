# Energy-Efficient UAV Route Planning over Mountainous Terrain

A compact, reproducible **SciPy** implementation of the trajectory-optimization
model described in the paper *"Energy-Efficient Route Planning for Delivery UAVs
over Mountainous Terrain: A Trajectory Optimization Approach"* (Iris Liu).

The code formulates single-origin / single-destination delivery routing over
rough terrain as a continuous-time energy-minimization problem, discretizes it
into a nonlinear program (NLP) with a piecewise-linear trajectory, and solves it
with SciPy's `SLSQP` optimizer. It reproduces every figure and table in the
paper's numerical section from scratch in under a minute on a laptop.

---

## The model in one screen

The drone is a point mass `m` flying a 3-D trajectory `r(t) = (x(t), y(t), z(t))`
from origin to destination. Under a quasi-steady assumption the instantaneous
**mechanical power** is the sum of a gravity term and an aerodynamic-drag term:

```
P_mech(t) = m g · v_z(t)            (gravity, v_z = vertical speed)
          + 0.5 ρ C_D A · |v(t)|³   (quadratic drag)
```

Battery (electrical) power is `P_elec = P_mech / η` with propeller efficiency
`η`. Total energy is the time integral of `P_elec`. The optimizer minimizes this
energy subject to:

* **terrain clearance** — stay at least `h_min` above the ground surface `z_g(x,y)`;
* **altitude cap** — stay below `h_max` above ground level (optional);
* **speed bounds** — `v_min ≤ |v| ≤ v_max`;
* **boundary conditions** — fixed start and end points.

### Two gravity models (an important detail)

The gravity term can be handled two ways, and the choice matters:

| mode | gravity power | behavior |
|---|---|---|
| **non-recuperative** (default) | `m g · max(v_z, 0)` | climbing costs energy that is **not** refunded on descent — physically correct for a multirotor. Routes go **around** peaks / cross at **saddles**; heavier payloads cost more. |
| **recuperative** (paper eq. 5, literal) | `m g · v_z` (signed) | descent is credited as free energy. The gravity integral then **telescopes** to the constant `m g (z_D − z_O)/η`, so only the drag term is optimizable and the optimal **route is independent of payload mass**. |

The non-recuperative model is the default here because it is physically faithful
and produces the terrain-avoiding routes the paper describes. The recuperative
model is kept because its telescoping property is a clean analytical result and a
useful sanity check (see `scripts/run_sensitivity.py`). Switch with
`RoutePlanner(..., descent_recovery=True/False)`.

---

## Install

```bash
git clone https://github.com/irisliu2010/uav-mountain-routing.git
cd uav-mountain-routing
pip install -r requirements.txt      # or: pip install -e .
```

Requires Python ≥ 3.9, NumPy, SciPy, and Matplotlib. No other dependencies.

## Quick start

```bash
python examples/quickstart.py
```

or in Python:

```python
from uav_routing import DroneParams, RoutePlanner
from uav_routing.scenarios import get_scenario

scenario = get_scenario("S1")                     # a synthetic mountain
params   = DroneParams(payload=10.0, h_max=None)  # DJI FlyCart 30 class
planner  = RoutePlanner(scenario.terrain, params,
                        scenario.origin, scenario.destination, n_nodes=40)

result = planner.solve_multistart()
print(result.energy/1e3, "kJ", "| length", result.path_length, "m")
```

## Reproduce the paper's results

```bash
python scripts/run_synthetic.py      # routes, profiles, convergence, tables
python scripts/run_sensitivity.py    # payload-mass sensitivity (both models)
python tests/test_model.py           # or: pytest
```

Figures land in `results/figures/`, tables in `results/tables/`.

---

## The dataset

Five synthetic 15 km × 15 km mountain scenarios, each a sum of 2-D Gaussian
bumps, spanning a gradient of routing difficulty:

| key | name | what it tests |
|---|---|---|
| S1 | Central barrier | one dominant massif blocking the straight route |
| S2 | Twin saddle | two tall peaks with a low corridor between them |
| S3 | Diagonal ridge | a continuous ridge across the origin–destination diagonal |
| S4 | Dense massif | a cluster of many peaks (procedurally generated) |
| S5 | Scattered hills | scattered low hills (procedurally generated) |

The procedural scenarios use `numpy.random.default_rng` with fixed seeds
(stable PCG64 bit-stream), so the whole dataset is reproducible. Parameters live
in [`uav_routing/scenarios.py`](uav_routing/scenarios.py).

### Headline results (non-recuperative model, 40 nodes, payload 10 kg)

| scenario | energy (kJ) | climb (kJ) | drag (kJ) | length (m) | max route alt (m) |
|---|---|---|---|---|---|
| S1 Central barrier | 1058.0 | 56.1 | 1002.0 | 22903 | 200 |
| S2 Twin saddle | 1070.1 | 56.1 | 1014.0 | 23178 | 200 |
| S3 Diagonal ridge | 1829.8 | 885.9 | 943.9 | 21454 | 1680 |
| S4 Dense massif | 1149.9 | 81.0 | 1068.9 | 24431 | 245 |
| S5 Scattered hills | 1801.0 | 794.9 | 1006.1 | 22997 | 1518 |

Where a detour exists (S1, S2, S4) the route stays low and skirts the massifs, so
the climb energy is small and drag dominates. Where a ridge blocks the whole
corridor (S3, S5) the route crosses at the lowest saddle and climb energy is
large. Payload mass raises total energy monotonically — steeply in the
climb-dominated cases, gently in the detour cases — while the route *structure*
stays stable (see `results/tables/sensitivity.md`).

---

## Using a real DEM

`GaussianTerrain` is one implementation of the `Terrain` interface; a bilinearly
interpolated grid DEM is also provided:

```python
from uav_routing import GridTerrain, DroneParams, RoutePlanner
import numpy as np

terrain = GridTerrain(x_coords, y_coords, heights)   # heights[j, i] at (x_i, y_j)
planner = RoutePlanner(terrain, DroneParams(h_max=300),
                       origin=(x0, y0, z0), destination=(x1, y1, z1), n_nodes=60)
route = planner.solve_multistart()
```

Any object exposing `height(x, y)` and `gradient(x, y)` works as a terrain.

For a georeferenced raster (e.g. an ASTER GDEM V3 GeoTIFF), `uav_routing/dem.py`
loads and reprojects it to a metric UTM frame automatically:

```bash
pip install -r requirements-dem.txt        # rasterio + pyproj
# edit DEM_PATH and the four delivery points in scripts/run_huangshan.py, then:
python scripts/run_huangshan.py
```

This runs the Mount Huangshan case study (tasks AB and CD, altitude caps 120 m
and 300 m) and writes `huangshan_3d.png`, `huangshan_top.png`,
`huangshan_profiles.png`, and `huangshan_results.{csv,md}`.

## Repository layout

```
uav_routing/
  params.py       DroneParams: vehicle / aero / mission parameters
  terrain.py      Terrain interface, GaussianTerrain, GridTerrain (DEM)
  scenarios.py    the five synthetic scenarios (the dataset)
  optimizer.py    RoutePlanner: discretization, analytic gradients, SLSQP, multi-start
  dem.py          load + reproject a real GeoTIFF DEM into a GridTerrain
  plotting.py     route / profile / convergence plots
  metrics.py      terrain and path statistics
scripts/          run_synthetic.py, run_sensitivity.py, run_huangshan.py
examples/         quickstart.py
tests/            test_model.py  (gradients, feasibility, telescoping, determinism)
results/          generated figures and tables
```

## Notes and limitations

* **SLSQP is a local method.** `solve_multistart` runs several laterally-offset
  initial guesses and keeps the lowest-energy feasible one; this is enough to
  find the around-the-peak route on these scenarios but does not guarantee a
  global optimum.
* The aerodynamic model is deliberately simple (gravity + `v³` drag, constant
  `η`). A hover-heavy mission would need an induced-power term. Substituting a
  richer power model changes only `optimizer._objective`/`_gradient`.
* Wind is not modeled.

## Citation

If you use this code, please cite the accompanying paper. A `CITATION` entry
will be added once the arXiv identifier is assigned.

## License

MIT — see [LICENSE](LICENSE).
