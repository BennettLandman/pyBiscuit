"""BISCUIT's LGI phase (the ``lgi`` script v1.3.3m driven by ``gyrification.m``; spec §9.3).

Chain per hemisphere: ``OuterHull`` (pial) -> hull correspondence -> TRACE curves (the ported
``CurveExtraction``, called as in spec §8, in place of the image's binary) -> ``Gyrification`` with
``-m 316 --speed 0.2 --refHullArea 77100`` -> ``<h>.gray.LGI.txt``.

The hull correspondence is the one step that is **not** the image's: ``klaplace
-surfaceCorrespondence`` stays out of scope (licence, Q7).  :func:`hull_correspondence` is a
stand-in that moves every pial vertex to the nearest vertex of the hull mesh (error at most half
the hull's vertex spacing, ~0.4 mm for the 256-voxel hull).  It keeps the pial topology, which is
all ``Gyrification --outer`` needs, but it is not klaplace's smooth Laplace-streamline map, so LGI
values are comparable to the image only after the image goldens (Q1) are in.  Comparisons against
the oracle chain share this correspondence, so they isolate the TRACE and Gyrification ports.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..io import read_vtk, write_vtk
from ..mesh import Mesh
from ..mesh.geometry import vertex_areas
from .hull import outer_hull
from .lgi import gyrification, write_lgi_maps

__all__ = [
    "LgiRun", "gyrification_argv", "hull_correspondence", "lgi_pipeline", "mesh_area",
]  # fmt: skip

DEFAULT_KERNEL = 316.0  # biscuit.conf lgi_kernel_size
DEFAULT_SPEED = 0.2  # lgi_anisotropy
DEFAULT_REF_AREA = 77100.0  # lgi_ref_area


def mesh_area(mesh: Mesh) -> float:
    """Total surface area (mm^2), float32 per-vertex areas as ``Gyrification`` uses them."""
    return float(vertex_areas(mesh.vertices.astype(np.float32), mesh.faces).sum())


def hull_correspondence(surface: Mesh, hull: Mesh) -> Mesh:
    """Stand-in for ``klaplace -surfaceCorrespondence`` + ``-conv``: the pial mesh with every vertex
    moved to its nearest hull vertex (same faces, so ``Gyrification --outer`` accepts it)."""
    _, idx = cKDTree(hull.vertices).query(surface.vertices)
    return Mesh(
        np.ascontiguousarray(hull.vertices[idx], dtype=np.float64), surface.faces,
        frame=surface.frame, units=surface.units,
    )  # fmt: skip


def gyrification_argv(
    pial, outer, scurve, gcurve, out, *, kernel: float = DEFAULT_KERNEL,
    speed: float = DEFAULT_SPEED, ref_area: float = DEFAULT_REF_AREA, n_threads: int = 1,
) -> list[str]:  # fmt: skip
    """The ``Gyrification`` call of the ``lgi`` script; floats go through ``%f``."""
    return [
        "-i", str(pial), "-o", str(out), "--outer", str(outer), "-s", str(scurve),
        "-g", str(gcurve), "-m", f"{kernel:f}", "--speed", f"{speed:f}",
        "--refHullArea", f"{ref_area:f}", "--nThreads", str(n_threads),
    ]  # fmt: skip


@dataclass
class LgiRun:
    hemi: str
    lgi_file: Path  # <h>.gray.LGI.txt
    hull_area: float  # OuterHull mesh area (mm^2)
    corr_area: float  # area of the correspondence mesh handed to --outer
    argv: list[str]
    lgi: np.ndarray


def lgi_pipeline(
    pial, scurve, gcurve, out_dir, *, hemi: str = "lh", kernel: float = DEFAULT_KERNEL,
    speed: float = DEFAULT_SPEED, ref_area: float = DEFAULT_REF_AREA, n_threads: int = 1,
    hull: Mesh | None = None, outer: Mesh | None = None, keep_tmp: bool = True,
    heap: str = "libcxx",
) -> LgiRun:  # fmt: skip
    """One hemisphere of the ``lgi`` script.  ``pial``: VTK path; ``scurve`` / ``gcurve``: the
    ``.scurve.bary`` of the pial and ``.gcurve.bary`` of the white surface (TRACE products).

    ``hull`` / ``outer`` may be supplied (e.g. by a test) to bypass ``OuterHull`` and the
    correspondence stand-in.  Scratch (hull, correspondence, raw map) is kept in
    ``<out_dir>/tmp-lgi-<hemi>.gray`` like the script, unless ``keep_tmp`` is false.
    """
    out_dir = Path(out_dir)
    tmp = out_dir / f"tmp-lgi-{hemi}.gray"
    tmp.mkdir(parents=True, exist_ok=True)
    surf = read_vtk(pial)
    if hull is None and outer is None:
        hull = outer_hull(surf)
    if outer is None:
        outer = hull_correspondence(surf, hull)
    hull_area = mesh_area(hull) if hull is not None else float("nan")
    if hull is not None:
        write_vtk(hull, tmp / f"{hemi}.gray.hull.vtk", dtype="float")
    write_vtk(outer, tmp / f"{hemi}.gray.hull.corr.vtk", dtype="float")
    argv = gyrification_argv(
        pial, tmp / f"{hemi}.gray.hull.corr.vtk", scurve, gcurve, tmp / f"{hemi}.gray",
        kernel=kernel, speed=speed, ref_area=ref_area, n_threads=n_threads,
    )  # fmt: skip
    res = gyrification(
        surf, outer, sulcus=scurve, gyrus=gcurve, max_area=kernel, speed=speed,
        ref_hull_area=ref_area, n_threads=n_threads, heap=heap,
    )  # fmt: skip
    maps = write_lgi_maps(res, tmp / f"{hemi}.gray")
    lgi_file = out_dir / f"{hemi}.gray.LGI.txt"
    shutil.copyfile(maps[0], lgi_file)  # the script moves <name>.lgi.map.*.txt; one map expected
    if not keep_tmp:
        shutil.rmtree(tmp)
    return LgiRun(hemi, lgi_file, hull_area, mesh_area(outer), argv, res.lgi[:, 0].copy())
