#!/usr/bin/env python
"""Write MeshLib geometry goldens: tests/data/golden/meshlib/geometry_<case>/ (brief 02, B2.4a).

    python scripts/oracles/make_golden_geometry.py [--harness PATH] [case ...]

Uses ``oracles/meshlib/bin/harness_geometry`` if present, else compiles the harness
from ``reference/repos/MeshLib`` into a temp dir.  Either way the build used for
committed goldens must be ``-ffp-contract=off`` (matches the x86-64 image binary; arm64
clang otherwise fuses a*b+c and moves float32 results by ~1e-7 relative, up to ~1e-4
for stiff sigma).  Each case dir holds input.vtk (ASCII, float32 text), data.f32 /
data.f64 inputs, the harness outputs, and command.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ML = REPO.parent / "reference" / "repos" / "MeshLib"
GOLD = REPO / "tests" / "data" / "golden" / "meshlib"
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))

CFLAGS = ["-O2", "-std=c++11", "-ffp-contract=off"]
SOURCES = ["Mesh.cpp", "MeshIO.cpp", "MNIObjIO.cpp", "VtkIO.cpp", "Util/SurfaceUtil.cpp", "Util/Geom.cpp"]


def write_ascii_vtk(path, vertices, faces) -> None:
    """ASCII legacy VTK that MeshLib's ``VtkIO`` reads; coordinates are float32 values."""
    v = np.asarray(vertices, dtype=np.float32)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n")
        fh.write(f"POINTS {len(v)} float\n")
        fh.write("\n".join(" ".join(f"{x:.9g}" for x in r) for r in v) + "\n")
        fh.write(f"POLYGONS {len(faces)} {4 * len(faces)}\n")
        fh.write("\n".join(f"3 {r[0]} {r[1]} {r[2]}" for r in faces) + "\n")


def build_harness(outdir: Path) -> Path:
    exe = outdir / "harness_geometry"
    src = [str(ML / "src" / s) for s in SOURCES]
    cmd = ["clang++", *CFLAGS, f"-I{ML / 'include'}", f"-I{ML / 'include' / 'Util'}",
           str(REPO / "scripts" / "oracles" / "harness_geometry.cpp"), *src, "-o", str(exe)]
    subprocess.run(cmd, check=True)
    return exe


def cases():
    from fixtures.meshes import cut_sphere, icosphere, perturb

    return {
        "geometry_ico2": (icosphere(2), 0.3),
        "geometry_ico3": (icosphere(3), 0.3),
        "geometry_perturb": (perturb(icosphere(3), 0.02, seed=3), 0.2),
        "geometry_cut": (cut_sphere(3), 0.3),
    }


def make_case(name, mesh, sigma, harness: Path, commit: str, compiler: str) -> None:
    d = GOLD / name
    d.mkdir(parents=True, exist_ok=True)
    write_ascii_vtk(d / "input.vtk", mesh.vertices, mesh.faces)
    rng = np.random.default_rng(12345)
    rng.standard_normal(mesh.n_vertices).astype("<f4").tofile(d / "data.f32")
    rng.standard_normal(mesh.n_vertices).astype("<f8").tofile(d / "data.f64")
    ops = ["-s", "1", "-s", "4", "-g", f"{sigma:g}", "1", "-g", f"{sigma / 2:g}", "3",
           "-df", "3", "data.f32", "-dd", "3", "data.f64"]
    argv = ["harness_geometry", "input.vtk", ".", *ops]
    subprocess.run([str(harness), "input.vtk", ".", *ops], check=True, cwd=d)
    (d / "command.json").write_text(json.dumps({
        "argv": argv, "commit": commit, "compiler": compiler, "cflags": CFLAGS,
        "oracle": "upstream-macos-arm64", "fixture": name,
    }, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness")
    ap.add_argument("case", nargs="*")
    a = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(ML), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    compiler = subprocess.run(["clang++", "--version"], capture_output=True, text=True,
                              check=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(a.harness) if a.harness else build_harness(Path(tmp))
        for name, (mesh, sigma) in cases().items():
            if not a.case or name in a.case:
                make_case(name, mesh, sigma, harness, commit, compiler)
                print("wrote", name, sum(p.stat().st_size for p in (GOLD / name).iterdir()), "bytes")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
