"""Seeded fold fixtures for the LGI port (brief 07).

``fold_case`` returns a closed folded sphere r = R (1 + a Y_lm + noise) on an icosphere and its
"hull": the same vertices moved radially onto a sphere of radius ``hull_scale * radius`` (NOT an
enclosing hull: it is chosen so that hull area < surface area, i.e. LGI > 1, which the hull-area cap
in ``precomputeArea`` would otherwise flatten to 1).  Topology is identical, as the
surface/hull topology is identical (what klaplace's surface correspondence delivers).  The
Gaussian radial noise (seeded) removes the exact mirror ties of the bare icosphere, which would
otherwise make nearest-source and heap-order decisions platform dependent.
"""

from __future__ import annotations

import numpy as np

from pybiscuit.mesh import Mesh

from .meshes import icosphere, real_sph_harm

SEED = 20261020


def fold_case(
    a: float = 0.15, l: int = 8, m: int = 3, level: int = 4, radius: float = 50.0,  # noqa: E741
    noise: float = 0.004, seed: int = SEED, hull_scale: float = 0.8,
) -> tuple[Mesh, Mesh]:  # fmt: skip
    base = icosphere(level)
    u = base.vertices
    theta = np.arccos(np.clip(u[:, 2], -1.0, 1.0))
    phi = np.arctan2(u[:, 1], u[:, 0])
    y = real_sph_harm(l, m, theta, phi)
    eps = np.random.default_rng(seed).standard_normal(u.shape[0]) * noise
    r = radius * (1.0 + a * y + eps)
    surf = Mesh(u * r[:, None], base.faces, frame="RAS", units="mm")
    r_hull = radius * hull_scale
    hull = Mesh(u * r_hull, base.faces, frame="RAS", units="mm")
    return surf, hull
