"""BISCUIT's LGI phase (the ``lgi`` script v1.3.3m driven by ``gyrification.m``; spec §9.3).

Chain per hemisphere: ``OuterHull`` (pial) -> hull correspondence -> TRACE curves (the ported
``CurveExtraction``, called as in spec §8, in place of the image's binary) -> ``Gyrification`` with
``-m 316 --speed 0.2 --refHullArea 77100`` -> ``<h>.gray.LGI.txt``.

The hull correspondence is the one step that is **not** the image's: ``klaplace
-surfaceCorrespondence`` stays out of scope (licence, Q7).  :func:`hull_correspondence` is a
stand-in that moves every pial vertex to the closest point of the hull surface (faces around the
nearest hull vertex).  Snapping to hull *vertices* was tried first: 72 % of the template's outer
vertices then coincide exactly, which makes the fast marching tie-dominated.  It keeps the pial
topology, which is all ``Gyrification --outer`` needs, but it is not klaplace's smooth
Laplace-streamline map, so LGI
values are comparable to the image only after the image goldens (Q1) are in.  Comparisons against
the oracle chain share this correspondence, so they isolate the TRACE and Gyrification ports.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numba
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


@numba.njit(cache=True)
def _closest_on_triangle(p, a, b, c):
    """Closest point of triangle abc to p (Ericson, Real-Time Collision Detection, 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w
    den = 1.0 / (va + vb + vc)
    return a + ab * (vb * den) + ac * (vc * den)


@numba.njit(cache=True)
def _project(points, hv, hf, nearest, indptr, tris):
    out = np.empty_like(points)
    for i in range(points.shape[0]):
        best = np.inf
        n = nearest[i]
        for k in range(indptr[n], indptr[n + 1]):
            f = hf[tris[k]]
            q = _closest_on_triangle(points[i], hv[f[0]], hv[f[1]], hv[f[2]])
            d = ((q - points[i]) ** 2).sum()
            if d < best:
                best = d
                out[i] = q
    return out


def hull_correspondence(surface: Mesh, hull: Mesh) -> Mesh:
    """Stand-in for ``klaplace -surfaceCorrespondence`` + ``-conv``: the pial mesh with every vertex
    moved to the closest point of the hull surface (same faces, so ``Gyrification --outer`` accepts
    it).  The search is over the faces around the nearest hull vertex, which is exact for a hull
    whose spacing is small against its distance to the surface."""
    hv = np.ascontiguousarray(hull.vertices, dtype=np.float64)
    hf = np.ascontiguousarray(hull.faces, dtype=np.int64)
    _, nearest = cKDTree(hv).query(surface.vertices)
    flat = hf.ravel()
    order = np.argsort(flat, kind="stable")
    indptr = np.zeros(hv.shape[0] + 1, dtype=np.int64)
    np.cumsum(np.bincount(flat, minlength=hv.shape[0]), out=indptr[1:])
    tris = (order // 3).astype(np.int64)
    pts = np.ascontiguousarray(surface.vertices, dtype=np.float64)
    out = _project(pts, hv, hf, nearest.astype(np.int64), indptr, tris)
    return Mesh(out, surface.faces, frame=surface.frame, units=surface.units)


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
    heap: str = "libcxx", corr_method: str = "klaplace",
) -> LgiRun:  # fmt: skip
    """One hemisphere of the ``lgi`` script.  ``pial``: VTK path; ``scurve`` / ``gcurve``: the
    ``.scurve.bary`` of the pial and ``.gcurve.bary`` of the white surface (TRACE products).

    ``hull`` / ``outer`` may be supplied (e.g. by a test) to bypass ``OuterHull`` and the
    correspondence calculation.  Scratch (hull, correspondence, raw map) is kept in
    ``<out_dir>/tmp-lgi-<hemi>.gray`` like the script, unless ``keep_tmp`` is false.
    """
    out_dir = Path(out_dir)
    tmp = out_dir / f"tmp-lgi-{hemi}.gray"
    tmp.mkdir(parents=True, exist_ok=True)
    surf = read_vtk(pial)
    if hull is None and outer is None:
        hull = outer_hull(surf)
    if outer is None:
        if corr_method == "klaplace":
            from .klaplace import surface_correspondence
            outer = surface_correspondence(surf, hull)
        else:
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
