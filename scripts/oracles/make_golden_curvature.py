#!/usr/bin/env python
"""Build the committed MeshLib curvature / spherical-mapping goldens (row B2.4b).

Usage: make_golden_curvature.py HARNESS_BINARY [case ...]

For each case: write the fixture mesh as ASCII float VTK (vertices rounded to float32 first, so the
text round-trips exactly into MeshLib's ``float`` vertices), run ``harness_curvature`` and pack every
raw dump into ``tests/data/golden/meshlib/<case>/golden.npz``.  ``command.json`` records argv,
commit, compiler and the oracle kind.  The harness is built from ``harness_curvature.cpp`` (see its
header) against the reference MeshLib sources; this script never needs the binary afterwards --
the tests read ``golden.npz`` only.
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

from fixtures.meshes import cut_sphere, ellipsoid, folded_sphere, icosphere, perturb  # noqa: E402
from pybiscuit.io import write_vtk  # noqa: E402
from pybiscuit.mesh import Mesh  # noqa: E402

SPHERE_ITERS = "1,2,5,20,100,10000"
CASES = {
    # name: (builder, tensor_iters, sphere_iters)
    "curv_ico3": (lambda: icosphere(3), "0,3", ""),
    "curv_ellipsoid": (lambda: ellipsoid(), "0,3", ""),
    "curv_folded": (lambda: folded_sphere(0.1, 4, 2, 3), "0,3", ""),
    "curv_cut": (lambda: cut_sphere(3), "0,3", ""),
    "curv_perturbed": (lambda: perturb(icosphere(3), 0.01, 7), "0,3", ""),
    "spheremap_ellipsoid": (lambda: ellipsoid(), "", SPHERE_ITERS),
    "spheremap_folded": (lambda: folded_sphere(0.3, 4, 2, 3), "", SPHERE_ITERS),
    "spheremap_perturbed": (lambda: perturb(icosphere(3), 0.05, 11), "", SPHERE_ITERS),
}


def f32_mesh(m: Mesh) -> Mesh:
    return Mesh(m.vertices.astype(np.float32).astype(np.float64), m.faces)


def main() -> None:
    harness = Path(sys.argv[1]).resolve()
    names = sys.argv[2:] or list(CASES)
    ml = REPO.parent / "reference" / "repos" / "MeshLib"
    commit = subprocess.run(
        ["git", "-C", str(ml), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    cxx = subprocess.run(
        ["/Library/Developer/CommandLineTools/usr/bin/clang++", "--version"], capture_output=True, text=True
    ).stdout.splitlines()[0]
    for name in names:
        build, titers, siters = CASES[name]
        mesh = f32_mesh(build())
        out = REPO / "tests" / "data" / "golden" / "meshlib" / name
        out.mkdir(parents=True, exist_ok=True)
        write_vtk(mesh, out / "input.vtk", dtype="float")
        with tempfile.TemporaryDirectory() as td:
            argv = [harness.name, "input.vtk", "<outdir>", titers, siters]
            subprocess.run(
                [str(harness), str(out / "input.vtk"), td, titers, siters], check=True
            )
            arrs = {}
            for p in sorted(Path(td).iterdir()):
                dt = {"f32": "<f4", "f64": "<f8", "i32": "<i4"}[p.suffix[1:]]
                arrs[p.stem] = np.fromfile(p, dtype=dt)
            np.savez_compressed(out / "golden.npz", **arrs)
        (out / "command.json").write_text(
            json.dumps(
                {
                    "argv": argv,
                    "commit": commit,
                    "compiler": cxx + " -O2 -std=c++11 -ffp-contract=off",
                    "oracle": "upstream-macos-arm64",
                    "tool": "harness_curvature (scripts/oracles/harness_curvature.cpp)",
                    "fixture": name,
                },
                indent=2,
            )
            + "\n"
        )
        print(name, mesh.n_vertices, (out / "golden.npz").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
