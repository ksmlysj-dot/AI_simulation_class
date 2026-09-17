"""Physics engine for the classroom electric-field simulator.

Relative units are used intentionally so the first lesson can focus on:
- Coulomb's law
- superposition
- electric-field direction
- electric potential

SI units can be introduced later by multiplying by Coulomb's constant.
"""

from __future__ import annotations

import numpy as np


def field_and_potential(
    x: np.ndarray,
    y: np.ndarray,
    charges: list[tuple[float, float, float]],
    softening: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Ex, Ey and V from point charges.

    Each charge is represented by:
        (q, x0, y0)

    A small softening term prevents numerical divergence exactly at a charge.
    """
    ex = np.zeros_like(x, dtype=float)
    ey = np.zeros_like(y, dtype=float)
    v = np.zeros_like(x, dtype=float)

    for q, x0, y0 in charges:
        dx = x - x0
        dy = y - y0
        r2 = dx**2 + dy**2 + softening**2
        r = np.sqrt(r2)

        ex += q * dx / (r2 * r)
        ey += q * dy / (r2 * r)
        v += q / r

    return ex, ey, v


def probe_measurement(
    x: float,
    y: float,
    charges: list[tuple[float, float, float]],
) -> tuple[float, float, float, float, float]:
    """Return Ex, Ey, |E| and V at a single probe position."""
    xx = np.array([[x]], dtype=float)
    yy = np.array([[y]], dtype=float)
    ex, ey, v = field_and_potential(xx, yy, charges)

    ex0 = float(ex[0, 0])
    ey0 = float(ey[0, 0])
    v0 = float(v[0, 0])
    emag = float(np.hypot(ex0, ey0))
    theta = float(np.degrees(np.arctan2(ey0, ex0)))
    return ex0, ey0, emag, v0, theta
