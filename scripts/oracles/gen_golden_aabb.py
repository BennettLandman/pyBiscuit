"""Generate the committed AABB_Sphere goldens (tests/data/golden/meshlib/aabb_*).

Runs the upstream-MeshLib harness (scripts/oracles/harness_aabb.cpp) on small
meshes and stores mesh + queries + oracle output + command.json per case.

usage: python scripts/oracles/gen_golden_aabb.py /path/to/harness_aabb
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from pybiscuit.io import write_vtk  # noqa: E402
from pybiscuit.mesh import Mesh  # noqa: E402
from pybiscuit.sphere import icosphere  # noqa: E402

OUT = REPO / "tests" / "data" / "golden" / "meshlib"
MESHLIB = REPO.parent / "reference" / "repos" / "MeshLib"


def f32(mesh: Mesh) -> Mesh:
    return Mesh(mesh.vertices.astype(np.float32).astype(np.float64), mesh.faces)


def deform(mesh: Mesh) -> Mesh:
    """Star-shaped deformation strictly inside the unit sphere (radial + tangential)."""
    v = mesh.vertices
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0])
    ph2 = ph + 0.03 * np.sin(3 * th) * np.cos(2 * ph)
    th2 = th + 0.01 * np.sin(2 * ph) * np.sin(th)
    r = 0.9 - 0.05 * np.sin(3 * th) * np.cos(2 * ph)
    w = np.stack([np.sin(th2) * np.cos(ph2), np.sin(th2) * np.sin(ph2), np.cos(th2)], 1)
    return Mesh(w * r[:, None], mesh.faces)


def queries(mesh: Mesh, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    tri = mesh.vertices[mesh.faces]
    special = np.vstack(
        [
            mesh.vertices,  # exact vertices: many exact ties
            (tri[:, 0] + tri[:, 1]) / 2,  # edge midpoints
            [[0, 0, 1], [0, 0, -1], [-1, 0, 0], [-1, 1e-9, 0], [-1, -1e-9, 0]],
        ]
    )
    return np.vstack([p, special]).astype(np.float32)


def main(harness: str) -> None:
    harness = str(Path(harness).resolve())
    commit = subprocess.run(
        ["git", "-C", str(MESHLIB), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    compiler = subprocess.run(["clang++", "--version"], capture_output=True, text=True).stdout
    compiler = compiler.splitlines()[0]
    base = f32(icosphere(3))
    defo = f32(deform(base))
    cases = {
        "aabb_ico3_closest": ("closest", base, None, False),
        "aabb_ico3_bool_eps": ("bool:0.02:0.01", base, None, False),
        "aabb_ico3_xcand_eps": ("xcand:0.02:0.01", base, None, False),
        "aabb_ico3_search": ("search:0.02:0.01", base, None, False),
        "aabb_deformed_update": ("closest", base, defo, False),
        "aabb_deformed_update_refresh": ("closest", base, defo, True),
    }
    for name, (mode, mesh, deformed, refresh) in cases.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        write_vtk(mesh, d / "mesh.vtk", dtype="float")
        q = queries(deformed if deformed is not None else mesh, 2000, 7)
        np.savetxt(d / "queries.txt", q, fmt="%.9g")
        argv = ["harness_aabb", mode, "mesh.vtk", "queries.txt", "expected.txt"]
        if deformed is not None:
            write_vtk(deformed, d / "deformed.vtk", dtype="float")
            argv.append("deformed.vtk")
            if refresh:
                argv.append("refresh")
        subprocess.run([harness, *argv[1:]], cwd=d, check=True)
        (d / "command.json").write_text(
            json.dumps(
                {
                    "argv": argv,
                    "commit": commit,
                    "compiler": compiler,
                    "oracle": "upstream-macos-arm64",
                    "build": "scripts/oracles/build_harness_aabb.sh (-O2 -std=c++11 -ffp-contract=off)",
                    "queries": "2000 seeded unit-sphere points + mesh vertices + edge midpoints + poles",
                },
                indent=2,
            )
            + "\n"
        )
        print("wrote", d)


if __name__ == "__main__":
    main(sys.argv[1])
