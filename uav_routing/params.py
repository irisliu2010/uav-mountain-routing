"""Physical and vehicle parameters for the UAV energy model.

All quantities are SI units (metres, seconds, kilograms, Joules).

The default values correspond to a DJI FlyCart 30 class heavy-lift
multirotor and are the ones used in the paper's numerical experiments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DroneParams:
    """Vehicle, aerodynamic, and mission parameters.

    Attributes
    ----------
    empty_mass : float
        Empty (no-payload) vehicle mass ``m0`` in kg.
    payload : float
        Payload mass in kg. Total mass is ``empty_mass + payload``.
    frontal_area : float
        Effective frontal area ``A`` in m^2 used in the drag law.
    drag_coeff : float
        Aerodynamic drag coefficient ``C_D`` (dimensionless).
    prop_efficiency : float
        Propeller/motor efficiency ``eta`` in (0, 1]. Electrical power is
        mechanical power divided by ``eta``.
    air_density : float
        Air density ``rho`` in kg/m^3.
    gravity : float
        Gravitational acceleration ``g`` in m/s^2.
    v_min, v_max : float
        Lower and upper airspeed bounds in m/s.
    h_min : float
        Minimum terrain clearance (above ground level) in m.
    h_max : float or None
        Maximum altitude above ground level in m. ``None`` disables the
        altitude cap (used in the synthetic experiments so route shape is
        driven purely by energy).
    """

    empty_mass: float = 30.0
    payload: float = 10.0
    frontal_area: float = 0.5
    drag_coeff: float = 1.0
    prop_efficiency: float = 0.7
    air_density: float = 1.225
    gravity: float = 9.81
    v_min: float = 10.0
    v_max: float = 20.0
    h_min: float = 30.0
    h_max: float | None = None

    @property
    def mass(self) -> float:
        """Total mass ``m = empty_mass + payload`` in kg."""
        return self.empty_mass + self.payload

    @property
    def drag_factor(self) -> float:
        """The constant ``K = 0.5 * rho * C_D * A`` in the drag power law."""
        return 0.5 * self.air_density * self.drag_coeff * self.frontal_area

    def with_payload(self, payload: float) -> "DroneParams":
        """Return a copy of these parameters with a different payload."""
        from dataclasses import replace

        return replace(self, payload=payload)
