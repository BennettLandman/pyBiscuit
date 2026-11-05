"""Fixture meshes of the gw triangular-interpolation goldens (row B2.7).

Shared by ``scripts/oracles/make_golden_interpolation.py`` and ``tests/test_interpolation.py``;
``command.json`` of each golden records the spec it was built from.
"""

from __future__ import annotations

import numpy as np

from fixtures.meshes import cut_sphere, folded_sphere, perturb
from pybiscuit.mesh import Mesh

CASES = {
    # closed, smooth folds: every face has three neighbours
    "fold_l3": {"mesh": "folded_sphere", "a": 0.15, "l": 6, "m": 3, "level": 3, "radius": 50.0,
                "source": 0, "seed": 20261105, "n_fixed": 32},
    # open (one boundary loop) and noisy: the missing-neighbour fallbacks
    "cut_noisy_l3": {"mesh": "cut_sphere", "level": 3, "z_cut": 0.5, "sigma": 0.02, "noise_seed": 7,
                     "radius": 30.0, "source": 5, "seed": 20261106, "n_fixed": 32},
}  # fmt: skip


def build_mesh(spec: dict) -> Mesh:
    """The case mesh with float32-representable vertices (what MeshLib reads from the VTK)."""
    if spec["mesh"] == "folded_sphere":
        m = folded_sphere(spec["a"], spec["l"], spec["m"], spec["level"])
    else:
        m = perturb(cut_sphere(spec["level"], spec["z_cut"]), spec["sigma"], spec["noise_seed"])
    v = (m.vertices * spec["radius"]).astype(np.float32).astype(np.float64)
    return Mesh(v, m.faces)
