"""Physics engine for the parallel-plate capacitor classroom simulator.

A positive top plate and a negative bottom plate create a uniform electric
field E pointing straight down between them (from y=d at the top plate to
y=0 at the bottom plate). A positive charge q released inside this field
falls under a constant force F = qE, so its motion is the electric analogue
of free fall with acceleration a = qE / m.

The simulation steps this motion forward in small fixed timesteps (see
step_state) rather than solving one closed-form trajectory, so that q, E or
m can change mid-fall and the charge's acceleration updates immediately
from wherever it currently is.

Potential energy is measured from the bottom plate (U = 0 at y = 0), giving
U(y) = qEy, matching U = qEd at the top plate (y = d).
"""

from __future__ import annotations

import math


def field_component(y: float, E: float, d: float) -> float:
    """Return the y-component of the field: -E between the plates, else 0."""
    if 0.0 <= y <= d:
        return -E
    return 0.0


def potential(y: float, E: float, d: float) -> float:
    """Return V(y), taking the bottom (negative) plate as the V=0 reference."""
    y_clamped = min(max(y, 0.0), d)
    return E * y_clamped


def potential_energy(y: float, q: float, E: float, d: float) -> float:
    """Return the potential energy of charge q at height y."""
    y_clamped = min(max(y, 0.0), d)
    return q * E * y_clamped


def acceleration(q: float, E: float, m: float) -> float:
    """Return the downward acceleration magnitude a = qE / m."""
    return q * E / m


def time_to_land(y: float, v: float, a: float) -> float:
    """Return the time for a charge at height y with downward speed v and
    downward acceleration a to reach the bottom plate (y=0).

    Solves 0 = y - v*tau - 1/2*a*tau^2 for the positive root tau.
    """
    if y <= 0.0:
        return 0.0
    if a <= 0.0:
        return y / v if v > 0.0 else float("inf")
    discriminant = max(v * v + 2.0 * a * y, 0.0)
    return (-v + math.sqrt(discriminant)) / a


def step_state(y: float, v: float, a: float, dt: float) -> tuple[float, float, float, bool]:
    """Advance one timestep of duration dt under constant acceleration a.

    Returns (y_new, v_new, dt_used, landed). If the charge would cross the
    bottom plate partway through this step, dt_used is shortened to the
    exact landing instant so the reported landing speed stays accurate
    regardless of the timer's step size.
    """
    y_end = y - v * dt - 0.5 * a * dt * dt
    if y_end > 0.0:
        return y_end, v + a * dt, dt, False

    tau = time_to_land(y, v, a)
    tau = min(max(tau, 0.0), dt)
    return 0.0, v + a * tau, tau, True
