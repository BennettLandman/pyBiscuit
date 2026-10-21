# ruff: noqa: E501
"""Scaled-down inputs for BISCUIT's four-call HSD chain (brief 21).

Same file layout and call structure as ``surf_reg_macruise.m``, one icosahedral level lower
(``sphere_5120/20480/81920/327680`` hold 642 / 2,562 / 10,242 / 40,962 vertices, so ``--icomesh``
levels are 3, 4, 5, 6 instead of 4-7): both optimiser branches still run (levels 3-5 gradient descent,
level 6 damped Newton).  The template mean/variance files have the *template* length (like the
163,842-line shipped files); subject property files are longer than the subject mesh of calls 1-2, so
HSD must read the first ``nVertex`` lines (spec H-2).  Spheres are the template-space icospheres of radius 100 (spec H-3); the subject enters only through its resampled
feature files ``J/<h>.res.*.txt`` (the pattern seen through a smooth warp, plus noise).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pybiscuit.io import write_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.sphere.icosphere import icosphere

from .hsd_cases import feature, warp

SPHERE_LEVELS = {"sphere_5120": 3, "sphere_20480": 4, "sphere_81920": 5, "sphere_327680": 6}
#: feature name -> template stem, seed of the common pattern
FEATURES = {"iH": ("iH", 21), "sulc": ("sulc", 22), "H": ("curv", 23)}
TOP = 6


def write_chain_case(root, hemi="lh", *, amp=0.12, warp_k=1, radius=100.0):
    """Write ``sphere/``, ``reg/`` and ``J/`` under ``root``; returns ``(sphere_dir, template_dir, surf_dir)``."""
    root = Path(root)
    sphere_dir, tdir, jdir = root / "sphere", root / "reg", root / "J"
    for d in (sphere_dir, tdir, jdir):
        d.mkdir(parents=True, exist_ok=True)
    top = icosphere(TOP)
    u = top.vertices / np.linalg.norm(top.vertices, axis=1, keepdims=True)
    for name, level in SPHERE_LEVELS.items():
        m = icosphere(level)
        write_vtk(Mesh(m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True) * radius, m.faces), sphere_dir / f"{name}.vtk")
    rng = np.random.default_rng(7)
    for feat, (stem, seed) in FEATURES.items():
        f = feature(u, seed)
        g = feature(warp(u, warp_k, amp), seed)  # subject feature: the template pattern seen through a warp
        np.savetxt(tdir / f"{hemi}.{stem}.mean.txt", f, fmt="%.6f")
        var = 0.4 + 0.2 * np.abs(f) + 0.1 * np.cos(3 * u[:, 0])
        np.savetxt(tdir / f"{hemi}.{stem}.var.txt", var, fmt="%.6f")
        subj = g + 0.05 * np.std(g) * rng.standard_normal(len(g))
        np.savetxt(jdir / f"{hemi}.res.{feat}.txt", subj, fmt="%.6f")
    return sphere_dir, tdir, jdir



def _rotation(axis, deg):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    t = np.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)


def write_parc_case(root, *, npts=12000, deg=12.0, seed=3, radius=100.0):
    """Parcellation alignment inputs (spec §6.3, ``parc.m:61``): a native-resolution sphere that is not an
    icosphere, rotated by ``deg`` degrees, carrying the template pattern; the template sphere / curvature mean
    are the level-6 stand-ins.  Returns ``(template_sphere, native_sphere, curv_mean, h_txt)``."""
    from scipy.spatial import ConvexHull

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((npts, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    hull = ConvexHull(u)
    faces = hull.simplices.astype(np.int32)
    n = np.cross(u[faces[:, 1]] - u[faces[:, 0]], u[faces[:, 2]] - u[faces[:, 0]])
    flip = (n * u[faces[:, 0]]).sum(1) < 0
    faces[flip] = faces[flip][:, ::-1]
    x = u @ _rotation([0.3, -0.5, 0.8], deg).T
    write_vtk(Mesh(x * radius, faces), root / "native.sphere.vtk")
    np.savetxt(root / "native.H.txt", feature(u, 23), fmt="%.6f")
    top = icosphere(TOP)
    ut = top.vertices / np.linalg.norm(top.vertices, axis=1, keepdims=True)
    write_vtk(Mesh(ut * radius, top.faces), root / "template.sphere.vtk")
    np.savetxt(root / "curv.mean.txt", feature(ut, 23), fmt="%.6f")
    return root / "template.sphere.vtk", root / "native.sphere.vtk", root / "curv.mean.txt", root / "native.H.txt"
