"""Icosahedral mesh files for parcellation training: port of SphericalLabeling ``gen_ico.py``.

Source: ilwoolyu/SphericalLabeling @ 37fa7a0, ``gen_ico.py`` 1-49 (``write_vtk`` 6-18, ``main``
20-46) and ``meshcnn/mesh.py`` 8-14 (``export_spheres``).  See
``docs/ports/parcellation-training.md``.

Upstream writes ``icosphere_{0..7}.pkl`` (MeshConv operators) with ``export_spheres`` and only
writes ``icosphere_{0..7}.vtk`` in its download fallback.  The VTK files are what the
data-augmentation step needs (``SphericalRemesh -r mesh_files/icosphere_5.vtk``).  This port
writes the VTK files only: the network regenerates its operators
(:func:`pybiscuit.sphere.operators.build_operators`), so no pickle is ever written or read.  The
vertices are the UGSCNN icosphere (:func:`pybiscuit.sphere.icosphere.icosphere`, scheme
``"ugscnn"``), i.e. ``pkl['V']``/``pkl['F']`` of the operator files.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..sphere.icosphere import icosphere

__all__ = ["export_spheres", "main", "write_vtk"]


def write_vtk(fn: str | Path, v: np.ndarray, f: np.ndarray) -> None:
    """``gen_ico.write_vtk``: ASCII polydata, vertices printed with Python's ``str(float)``."""
    lines = [
        "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n",
        f"POINTS {v.shape[0]} float\n",
    ]
    lines += [f"{row[0]} {row[1]} {row[2]}\n" for row in np.asarray(v, dtype=np.float64).tolist()]
    lines.append(f"POLYGONS {f.shape[0]} {f.shape[0] * 4}\n")
    lines += [f"3 {row[0]} {row[1]} {row[2]}\n" for row in np.asarray(f).tolist()]
    Path(fn).write_text("".join(lines))


def export_spheres(levels: Iterable[int] = range(8), dest: str | Path = "mesh_files") -> list[Path]:
    """Write ``<dest>/icosphere_<level>.vtk`` for each level; returns the paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for i in levels:
        m = icosphere(int(i), scheme="ugscnn")
        p = dest / f"icosphere_{int(i)}.vtk"
        write_vtk(p, m.vertices, m.faces)
        out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit parc-gen-ico",
        description="Write the icosahedral meshes icosphere_<level>.vtk (port of gen_ico.py).",
    )
    p.add_argument("--dest", default="mesh_files", help="output folder (default: mesh_files)")
    p.add_argument(
        "--levels", type=int, nargs="+", default=list(range(8)), help="levels (default: 0..7)"
    )
    a = p.parse_args(argv)
    for path in export_spheres(a.levels, a.dest):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
