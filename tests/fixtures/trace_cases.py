"""Seeded TRACE / geodesic / slicer test cases shared by the golden generator and the tests."""

from __future__ import annotations

import numpy as np

from fixtures.meshes import folded_sphere
from pybiscuit.mesh import Mesh

RADIUS = 50.0
# name -> (a, l, m, icosphere level); radius 50 mm, seeds -0.01 / 0.01 (the defaults are tuned
# for cortex, not for these synthetic surfaces; see docs/ports/oracles.md)
TRACE_CASES = {
    "fold_l5m3": (0.3, 5, 3, 5),
    "fold_l8m4": (0.2, 8, 4, 5),
}
TRACE_ARGS = ["--sulcus", "--gyrus", "--bary", "--interm", "--sulcalSeed", "-0.01",
              "--gyralSeed", "0.01"]  # fmt: skip


def trace_mesh(case: str) -> Mesh:
    a, l, m, level = TRACE_CASES[case]  # noqa: E741
    mesh = folded_sphere(a, l, m, level)
    # float32 coordinates, as every MeshLib file holds
    return Mesh((mesh.vertices * RADIUS).astype(np.float32).astype(np.float64), mesh.faces)


def write_ascii_vtk(path, vertices, faces) -> None:
    """ASCII legacy VTK read by MeshLib's ``VtkIO``; coordinates are float32 values (%.9g)."""
    v = np.asarray(vertices, dtype=np.float32)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n")
        fh.write(f"POINTS {len(v)} float\n")
        fh.write("\n".join(" ".join(f"{x:.9g}" for x in r) for r in v) + "\n")
        fh.write(f"POLYGONS {len(faces)} {4 * len(faces)}\n")
        fh.write("\n".join(f"3 {r[0]} {r[1]} {r[2]}" for r in faces) + "\n")
