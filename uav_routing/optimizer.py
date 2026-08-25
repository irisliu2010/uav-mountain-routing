"""Direct-transcription trajectory optimiser.

The continuous-time energy-minimisation problem (P) of the paper is
discretised with ``N`` time nodes and a piecewise-linear trajectory, and
the resulting finite-dimensional nonlinear program is solved with SciPy's
SLSQP method.

Decision variables
------------------
The optimiser varies the ``N - 2`` interior node positions (each in 3-D)
and the total flight time ``T``; the first and last nodes are fixed to the
origin and destination.  The decision vector is::

    x = [ r_1, r_2, ..., r_{N-2}, T ]        (length 3*(N-2) + 1)

Energy model
------------
The discrete battery energy is (paper eq. 11)::

    E_hat = (1/eta) * sum_i [ m g * dz_i  +  K * |d_i|^3 / dt^2 ]

with ``d_i = r_{i+1} - r_i``, ``dz_i = z_{i+1} - z_i``, ``dt = T/(N-1)``
and ``K = 0.5 rho C_D A``.  With ``descent_recovery=True`` the gravity
term uses the signed ``dz_i`` (paper eq. 5); the sum then telescopes to the
constant ``m g (z_D - z_O)/eta`` and only drag is optimisable.  With
``descent_recovery=False`` the gravity term uses ``max(dz_i, 0)`` (descent
power floored at zero), which is physically non-recuperative and makes the
optimal route depend on payload mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .params import DroneParams
from .terrain import Terrain


@dataclass
class RouteResult:
    """Output of a single optimisation run."""

    nodes: np.ndarray          # (N, 3) optimised trajectory node positions
    flight_time: float         # optimal total flight time T (s)
    energy: float              # total battery energy (J)
    energy_gravity: float      # gravity component of the energy (J)
    energy_drag: float         # drag component of the energy (J)
    path_length: float         # 3-D path length (m)
    horizontal_length: float   # horizontal (map-projected) path length (m)
    speeds: np.ndarray         # (N-1,) per-segment speeds (m/s)
    success: bool
    n_iter: int
    message: str
    history: list[float]       # objective value per iteration (for convergence plots)


class RoutePlanner:
    """Set up and solve the discretised energy-minimising routing problem."""

    def __init__(
        self,
        terrain: Terrain,
        params: DroneParams,
        origin,
        destination,
        n_nodes: int = 40,
        descent_recovery: bool = False,
    ):
        self.terrain = terrain
        self.params = params
        self.origin = np.asarray(origin, dtype=float)
        self.destination = np.asarray(destination, dtype=float)
        self.N = int(n_nodes)
        self.descent_recovery = bool(descent_recovery)
        if self.N < 3:
            raise ValueError("n_nodes must be at least 3")

    # ------------------------------------------------------------------ #
    # Packing / unpacking the decision vector
    # ------------------------------------------------------------------ #
    def _unpack(self, x):
        """Return (full nodes array (N,3), T) from a decision vector."""
        interior = x[:-1].reshape(self.N - 2, 3)
        T = x[-1]
        nodes = np.empty((self.N, 3))
        nodes[0] = self.origin
        nodes[-1] = self.destination
        nodes[1:-1] = interior
        return nodes, T

    # ------------------------------------------------------------------ #
    # Objective and analytic gradient
    # ------------------------------------------------------------------ #
    def _objective(self, x):
        nodes, T = self._unpack(x)
        dt = T / (self.N - 1)
        d = np.diff(nodes, axis=0)              # (N-1, 3) segment vectors
        seg_len = np.linalg.norm(d, axis=1)     # (N-1,)
        K = self.params.drag_factor
        mg = self.params.mass * self.params.gravity
        if self.descent_recovery:
            grav = mg * np.sum(d[:, 2])
        else:
            grav = mg * np.sum(np.maximum(d[:, 2], 0.0))
        drag = K * np.sum(seg_len ** 3) / dt ** 2
        return (grav + drag) / self.params.prop_efficiency

    def _gradient(self, x):
        nodes, T = self._unpack(x)
        dt = T / (self.N - 1)
        d = np.diff(nodes, axis=0)
        seg_len = np.linalg.norm(d, axis=1)
        K = self.params.drag_factor
        mg = self.params.mass * self.params.gravity
        eta = self.params.prop_efficiency

        # d(|d_i|^3)/d(d_i) = 3 |d_i| d_i
        drag_dd = (3.0 * K / dt ** 2) * seg_len[:, None] * d   # (N-1, 3)

        # gravity contribution to d/d(d_i)
        grav_dd = np.zeros_like(d)
        if self.descent_recovery:
            grav_dd[:, 2] = mg
        else:
            grav_dd[:, 2] = mg * (d[:, 2] > 0.0)

        seg_grad = (drag_dd + grav_dd) / eta   # gradient wrt each segment vector d_i

        # node j (interior) affects d_{j-1} (=+) and d_j (=-)
        grad_nodes = np.zeros((self.N, 3))
        grad_nodes[1:] += seg_grad
        grad_nodes[:-1] -= seg_grad
        grad_interior = grad_nodes[1:-1].reshape(-1)

        # dE/dT : only drag depends on T through 1/dt^2 = (N-1)^2 / T^2
        drag_sum = np.sum(seg_len ** 3)
        dE_dT = (K * (self.N - 1) ** 2 / eta) * drag_sum * (-2.0 / T ** 3)

        return np.concatenate([grad_interior, [dE_dT]])

    # ------------------------------------------------------------------ #
    # Constraints (SLSQP convention: g(x) >= 0)
    # ------------------------------------------------------------------ #
    def _clearance_lower(self, x):
        nodes, _ = self._unpack(x)
        interior = nodes[1:-1]
        zg = np.asarray(self.terrain.height(interior[:, 0], interior[:, 1]))
        return interior[:, 2] - (zg + self.params.h_min)

    def _clearance_lower_jac(self, x):
        nodes, _ = self._unpack(x)
        interior = nodes[1:-1]
        gx, gy = self.terrain.gradient(interior[:, 0], interior[:, 1])
        gx = np.atleast_1d(gx)
        gy = np.atleast_1d(gy)
        n = self.N - 2
        J = np.zeros((n, 3 * n + 1))
        for k in range(n):
            J[k, 3 * k + 0] = -gx[k]
            J[k, 3 * k + 1] = -gy[k]
            J[k, 3 * k + 2] = 1.0
        return J

    def _clearance_upper(self, x):
        nodes, _ = self._unpack(x)
        interior = nodes[1:-1]
        zg = np.asarray(self.terrain.height(interior[:, 0], interior[:, 1]))
        return (zg + self.params.h_max) - interior[:, 2]

    def _clearance_upper_jac(self, x):
        nodes, _ = self._unpack(x)
        interior = nodes[1:-1]
        gx, gy = self.terrain.gradient(interior[:, 0], interior[:, 1])
        gx = np.atleast_1d(gx)
        gy = np.atleast_1d(gy)
        n = self.N - 2
        J = np.zeros((n, 3 * n + 1))
        for k in range(n):
            J[k, 3 * k + 0] = gx[k]
            J[k, 3 * k + 1] = gy[k]
            J[k, 3 * k + 2] = -1.0
        return J

    def _speeds(self, x):
        nodes, T = self._unpack(x)
        dt = T / (self.N - 1)
        d = np.diff(nodes, axis=0)
        return np.linalg.norm(d, axis=1) / dt

    def _speed_lower(self, x):
        return self._speeds(x) - self.params.v_min

    def _speed_upper(self, x):
        return self.params.v_max - self._speeds(x)

    def _speed_jac(self, x, sign):
        """Jacobian of ``sign * speed_i`` wrt the decision vector."""
        nodes, T = self._unpack(x)
        dt = T / (self.N - 1)
        d = np.diff(nodes, axis=0)
        seg_len = np.linalg.norm(d, axis=1)
        seg_len_safe = np.where(seg_len > 0, seg_len, 1.0)
        unit = d / seg_len_safe[:, None]
        n = self.N - 2
        m = self.N - 1                      # number of segments
        J = np.zeros((m, 3 * n + 1))
        factor = (self.N - 1) / T           # 1/dt
        for i in range(m):                  # segment i connects node i and i+1
            # d(speed_i)/d(node i)   = -unit_i / dt
            # d(speed_i)/d(node i+1) = +unit_i / dt
            if 1 <= i <= n:                 # node i is interior (index i-1 in interior block)
                col = 3 * (i - 1)
                J[i, col:col + 3] = -unit[i] * factor
            if 1 <= i + 1 <= n:             # node i+1 is interior
                col = 3 * (i + 1 - 1)
                J[i, col:col + 3] = unit[i] * factor
            J[i, -1] = -seg_len[i] * (self.N - 1) / T ** 2
        return sign * J

    def _build_constraints(self):
        cons = [
            {"type": "ineq", "fun": self._clearance_lower, "jac": self._clearance_lower_jac},
            {"type": "ineq", "fun": self._speed_lower, "jac": lambda x: self._speed_jac(x, +1.0)},
            {"type": "ineq", "fun": self._speed_upper, "jac": lambda x: self._speed_jac(x, -1.0)},
        ]
        if self.params.h_max is not None:
            cons.append(
                {"type": "ineq", "fun": self._clearance_upper, "jac": self._clearance_upper_jac}
            )
        return cons

    # ------------------------------------------------------------------ #
    # Initial guess
    # ------------------------------------------------------------------ #
    def initial_guess(self):
        """Straight line in the horizontal plane, lifted above the terrain."""
        taus = np.linspace(0.0, 1.0, self.N)
        line = self.origin[None, :] + taus[:, None] * (self.destination - self.origin)[None, :]
        zg = np.asarray(self.terrain.height(line[:, 0], line[:, 1]))
        # lift z to satisfy clearance with a little slack
        clearance_z = zg + self.params.h_min + 5.0
        line[:, 2] = np.maximum(line[:, 2], clearance_z)
        if self.params.h_max is not None:
            line[:, 2] = np.minimum(line[:, 2], zg + self.params.h_max - 1.0)
        line[0] = self.origin
        line[-1] = self.destination
        horiz = np.linalg.norm((self.destination - self.origin)[:2])
        T0 = horiz / (0.5 * (self.params.v_min + self.params.v_max))
        x0 = np.concatenate([line[1:-1].reshape(-1), [T0]])
        return x0

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def solve(self, x0=None, max_iter: int = 300, ftol: float = 1e-6, verbose: bool = False):
        if x0 is None:
            x0 = self.initial_guess()
        history: list[float] = []

        def callback(xk):
            history.append(float(self._objective(xk)))

        cons = self._build_constraints()
        # T must stay positive.
        bounds = [(None, None)] * (3 * (self.N - 2)) + [(1e-3, None)]

        res = minimize(
            self._objective,
            x0,
            jac=self._gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            callback=callback,
            options={"maxiter": max_iter, "ftol": ftol, "disp": verbose},
        )

        nodes, T = self._unpack(res.x)
        return self._make_result(nodes, T, res, history)

    def solve_multistart(
        self,
        offsets=(-6000.0, -4000.0, -2000.0, 0.0, 2000.0, 4000.0, 6000.0),
        max_iter: int = 300,
        ftol: float = 1e-6,
    ) -> RouteResult:
        """Solve from several laterally-offset initial guesses, keep the best.

        SLSQP is a local method, so the straight-line initial guess can settle
        on a route that flies over a peak when a detour around it is cheaper.
        Each offset bows the initial horizontal path sideways (perpendicular to
        the origin-destination line, zero at the endpoints) by a fixed amount,
        so the different starts explore going around a peak on either side.
        The lowest-energy feasible solution is returned; its ``history`` is the
        convergence trace of the winning start.
        """
        origin = self.origin
        dest = self.destination
        taus = np.linspace(0.0, 1.0, self.N)
        base = origin[None, :] + taus[:, None] * (dest - origin)[None, :]
        diag = dest[:2] - origin[:2]
        perp = np.array([-diag[1], diag[0], 0.0])
        norm = np.linalg.norm(perp)
        if norm > 0:
            perp = perp / norm
        bump = np.sin(np.pi * taus)
        horiz = np.linalg.norm(diag)
        T0 = horiz / (0.5 * (self.params.v_min + self.params.v_max))

        best: RouteResult | None = None
        for off in offsets:
            guess = base.copy()
            guess[:, 0] += off * bump * perp[0]
            guess[:, 1] += off * bump * perp[1]
            zg = np.asarray(self.terrain.height(guess[:, 0], guess[:, 1]))
            guess[:, 2] = np.maximum(guess[:, 2], zg + self.params.h_min + 5.0)
            if self.params.h_max is not None:
                guess[:, 2] = np.minimum(guess[:, 2], zg + self.params.h_max - 1.0)
            guess[0] = origin
            guess[-1] = dest
            x0 = np.concatenate([guess[1:-1].reshape(-1), [T0]])
            res = self.solve(x0=x0, max_iter=max_iter, ftol=ftol)
            if res.success and (best is None or res.energy < best.energy):
                best = res
        if best is None:  # fall back to the plain single start
            best = self.solve(max_iter=max_iter, ftol=ftol)
        return best

    def _make_result(self, nodes, T, res, history) -> RouteResult:
        d = np.diff(nodes, axis=0)
        seg_len = np.linalg.norm(d, axis=1)
        dt = T / (self.N - 1)
        speeds = seg_len / dt
        mg = self.params.mass * self.params.gravity
        eta = self.params.prop_efficiency
        K = self.params.drag_factor
        if self.descent_recovery:
            e_grav = mg * np.sum(d[:, 2]) / eta
        else:
            e_grav = mg * np.sum(np.maximum(d[:, 2], 0.0)) / eta
        e_drag = K * np.sum(seg_len ** 3) / dt ** 2 / eta
        horiz = np.sum(np.linalg.norm(d[:, :2], axis=1))
        if not history:
            history = [float(res.fun)]
        return RouteResult(
            nodes=nodes,
            flight_time=float(T),
            energy=float(e_grav + e_drag),
            energy_gravity=float(e_grav),
            energy_drag=float(e_drag),
            path_length=float(np.sum(seg_len)),
            horizontal_length=float(horiz),
            speeds=speeds,
            success=bool(res.success),
            n_iter=int(res.get("nit", len(history))),
            message=str(res.message),
            history=history,
        )
