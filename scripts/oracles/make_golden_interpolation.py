#!/usr/bin/env python
"""Build the committed gw triangular-interpolation goldens (row B2.7).

Usage: make_golden_interpolation.py HARNESS_BINARY [case ...]

``HARNESS_BINARY`` is ``harness_interpolation`` from ``build_curve_extraction_nofma.sh``
(MeshLib dcef807, -ffp-contract=off).  For each case: build the fixture mesh (vertices rounded to
float32 so the ASCII float VTK round-trips into MeshLib's ``float``), a distance field (the port's
fast marching from one vertex; the harness only reads it, so its origin does not matter) and random
queries over every face, run the harness, and write ``tests/data/golden/interpolation/<case>/``:
``dist.txt``, ``queries.txt``, ``out.txt`` and ``command.json``.  The mesh itself is rebuilt from
the fixture parameters at test time.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

from fixtures.interpolation_cases import CASES, build_mesh  # noqa: E402
from pybiscuit.io import write_vtk  # noqa: E402
from pybiscuit.mesh.geodesic import GeodesicSolver, _build_connectivity  # noqa: E402


def queries(mesh, seed: int, n_fixed: int) -> list[str]:
    """One random query per face (kind cycling L, Q, C; random vertex order and point), and the
    vertices, an edge midpoint and the centroid in every kind for ``n_fixed`` random faces plus every
    face with a missing neighbour."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(mesh.n_faces):
        perm = int(rng.integers(6))
        x, y = rng.random(2)
        if x + y > 1:
            x, y = 1 - x, 1 - y
        rows.append(f"{'LQC'[f % 3]} {f} {perm} {float(x)!r} {float(y)!r}")
    _, fnbr = _build_connectivity(mesh.faces.astype(np.int32), mesh.n_vertices)
    fixed = set(rng.choice(mesh.n_faces, n_fixed, replace=False).tolist())
    fixed |= set(np.flatnonzero((fnbr < 0).any(axis=1)).tolist())
    for f in sorted(fixed):
        for kind in "LQC":
            for x, y in ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.5, 0.5), (1 / 3, 1 / 3)):
                rows.append(f"{kind} {f} 0 {x!r} {y!r}")
    return rows


def main() -> None:
    harness = Path(sys.argv[1]).resolve()
    names = sys.argv[2:] or list(CASES)
    ml = REPO.parent / "reference" / "repos" / "MeshLib"
    commit = subprocess.run(
        ["git", "-C", str(ml), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    cxx = subprocess.run(
        ["/usr/bin/clang++", "--version"], capture_output=True, text=True
    ).stdout.splitlines()[0]
    for name in names:
        spec = CASES[name]
        mesh = build_mesh(spec)
        solver = GeodesicSolver(mesh.vertices, mesh.faces)
        solver.perform_front_propagation(spec["source"])
        out = REPO / "tests" / "data" / "golden" / "interpolation" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "dist.txt").write_text("".join(f"{float(d)!r}\n" for d in solver.dist))
        (out / "queries.txt").write_text(
            "\n".join(queries(mesh, spec["seed"], spec["n_fixed"])) + "\n"
        )
        with tempfile.TemporaryDirectory() as td:
            vtk = Path(td) / "mesh.vtk"
            write_vtk(mesh, vtk, dtype="float")
            subprocess.run(
                [
                    str(harness),
                    str(vtk),
                    str(out / "dist.txt"),
                    str(out / "queries.txt"),
                    str(out / "out.txt"),
                ],
                check=True,
            )
        (out / "command.json").write_text(
            json.dumps(
                {
                    "tool": "harness_interpolation (scripts/oracles/harness_interpolation.cpp)",
                    "argv": [
                        "harness_interpolation",
                        "mesh.vtk",
                        "dist.txt",
                        "queries.txt",
                        "out.txt",
                    ],
                    "fixture": spec,
                    "meshlib": commit,
                    "compiler": cxx + " -O2 -std=c++11 -DNDEBUG -ffp-contract=off",
                    "oracle": "upstream-macos-arm64 (-ffp-contract=off)",
                },
                indent=2,
            )
            + "\n"
        )
        print(name, mesh.n_vertices, mesh.n_faces, (out / "out.txt").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
