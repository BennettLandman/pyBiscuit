"""Synthetic 'two hemispheres joined by a bridge' surfaces (brief 12 fixtures).

A unit icosphere is stretched along x into a dumbbell: two ellipsoid lobes
at x = -/+ lobe_x joined by a narrow neck at x = 0.  The surface is closed,
genus 0 and outward oriented, and a plane x = c near 0 crosses only the neck,
so removing the cut band leaves two boundary loops.  Deterministic (no randomness).
"""

from __future__ import annotations

import numpy as np

from pybiscuit.mesh.model import Mesh
from pybiscuit.sphere.icosphere import icosphere


def bridged_ellipsoids(
    level: int = 5,
    lobe_x: float = 35.0,
    lobe_half: float = 28.0,
    radius_yz: float = 40.0,
    neck: float = 0.22,
    wobble: float = 0.0,
) -> Mesh:
    u = icosphere(level).vertices.copy()
    u /= np.linalg.norm(u, axis=1)[:, None]
    ux, uy, uz = u.T
    s = np.abs(ux)
    prof = neck + (1 - neck) * np.sin(np.clip(s * 1.6, 0, 1) * np.pi / 2) ** 2
    rho = radius_yz * np.sqrt(np.clip(1 - ux * ux, 0, 1)) * prof
    x = np.sign(ux) * lobe_x * s**0.8 + lobe_half * ux * 0.7
    yz = np.hypot(uy, uz)
    yz = np.where(yz > 1e-12, yz, 1.0)
    v = np.column_stack([x, rho * uy / yz * 0.8, rho * uz / yz])
    if wobble:
        v[:, 1] += wobble * np.sin(v[:, 0] * 0.4) * np.cos(v[:, 2] * 0.3)
        v[:, 2] += wobble * np.cos(v[:, 0] * 0.3) * np.sin(v[:, 1] * 0.4)
    return Mesh(v, icosphere(level).faces, frame="RAS", units="mm")
