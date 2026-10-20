"""Seeded unit-sphere fixture points for the spherical-harmonics oracle (float32)."""

from __future__ import annotations

import numpy as np

SH_SEED = 20260511

_SPECIAL = [
    [0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
    [1, 0, 1e-4], [1e-6, 0, 1], [1e-3, 1e-3, 1], [0, 1e-7, -1], [1, 1, 0],
    [-1, -1e-9, 0], [0.6, 0, 0.8], [1.5e-8, 0, 1], [0, 1.6e-8, -1], [1.2e-8, 1.2e-8, 1],
]  # fmt: skip


def sh_fixture_points(n_random: int = 2000, ico_level: int = 3) -> np.ndarray:
    """Random (PCG64 seed 20260511) + special axis/pole points + icosphere vertices, float32."""
    from pybiscuit.sphere.icosphere import icosphere

    rng = np.random.default_rng(SH_SEED)
    r = rng.normal(size=(n_random, 3))
    r /= np.linalg.norm(r, axis=1)[:, None]
    sp = np.array(_SPECIAL, float)
    ico = np.asarray(icosphere(ico_level).vertices, float)
    ico /= np.linalg.norm(ico, axis=1)[:, None]
    return np.vstack([r, sp, ico]).astype(np.float32)
