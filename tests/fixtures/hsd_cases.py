# ruff: noqa: E501
"""Seeded HSD fixtures: subjects = warped icospheres carrying one common smooth feature.

Subject k has native sphere ``x_i = unit(u_i + warp_k(u_i))`` (u_i = BISCUIT icosphere vertex i)
and property ``f(u_i)`` (a fixed low-degree real harmonic mix), so registration should undo the warps.
Subject 0 is unwarped.  ``write_case`` writes ``s<k>.sphere.vtk`` (radius 100, like shipped
spheres) and ``s<k>.curv.txt`` (``%.9g``) and returns the file names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pybiscuit.io import write_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.sphere.icosphere import icosphere

from .meshes import real_sph_harm


def feature(u: np.ndarray, seed: int = 11, degree: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.arccos(np.clip(u[:, 2], -1, 1))
    phi = np.arctan2(u[:, 1], u[:, 0])
    f = np.zeros(len(u))
    for l in range(1, degree + 1):  # noqa: E741
        for m in range(-l, l + 1):
            f += rng.standard_normal() / l * real_sph_harm(l, m, theta, phi)
    return f


def warp(u: np.ndarray, k: int, amp: float) -> np.ndarray:
    if k == 0:
        return u.copy()
    ph = 1.3 * k
    d = np.stack(
        [np.sin(2 * u[:, 2] + ph), np.cos(2 * u[:, 0] - ph), np.sin(u[:, 0] + 2 * u[:, 1] + ph)], 1
    )
    d -= (d * u).sum(1, keepdims=True) * u  # tangential
    x = u + amp * d
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def write_case(workdir, n_subj=3, level=4, amp=0.12, seed=11, radius=100.0):
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    m = icosphere(level)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    f = feature(u, seed)
    names = []
    for k in range(n_subj):
        x = warp(u, k, amp) * radius
        write_vtk(Mesh(x, m.faces), work / f"s{k}.sphere.vtk")
        np.savetxt(work / f"s{k}.curv.txt", f, fmt="%.9g")
        names.append((f"s{k}.sphere.vtk", f"s{k}.curv.txt"))
    # variance prior of the template (subject 0) for --tmpVar
    np.savetxt(work / "tmpvar.txt", 0.4 + 0.2 * np.abs(f) + 0.1 * np.cos(3 * u[:, 0]), fmt="%.9g")
    return names


# --------------------------------------------------------------------------- #
# case tables shared by scripts/oracles/make_golden_hsd.py and tests/test_hsd_*.py
# --------------------------------------------------------------------------- #
#: unit-level states evaluated with the oracle harness (see harness_hsd.cpp "eval")
EVAL_CASES = {
    "gd_d0": dict(deg=6, ico=4, deginc=0, beg=0, end=0, trans=0),
    "gd_d3": dict(deg=6, ico=4, deginc=3, beg=0, end=3, trans=1),
    "gd_final": dict(deg=6, ico=4, deginc=6, beg=0, end=6, trans=2),
    "newton_band3": dict(deg=6, ico=6, deginc=3, beg=3, end=3, trans=1),
    "newton_final": dict(deg=6, ico=6, deginc=6, beg=0, end=6, trans=2),
}

#: full CLI runs (flags after the -s/-p/-o/--outputcoeff lists)
RUN_CASES = {
    "gd_ico4_guess": dict(nsub=3, flags=["-d", "6", "--icosahedron", "4"]),
    "newton_ico6": dict(nsub=3, flags=["-d", "6", "--icosahedron", "6", "--maxIter", "4"]),
    "resample_ico5": dict(nsub=3, flags=["-d", "8", "--icosahedron", "5", "--resample"]),
    "fixed_ico5": dict(nsub=3, flags=["-d", "6", "--icosahedron", "5", "--noguess", "--fixedSubjects", "0"]),
    "pairwise_ico5": dict(
        nsub=2, flags=["-d", "6", "--icosahedron", "5", "--fixedSubjects", "0", "--tmpVar", "tmpvar.txt"]
    ),
}


def coeff_states(deg: int, deginc: int, nsub: int = 3, seed: int = 5):
    """Two seeded SH coefficient states (A, B), shape (nsub, 3, (deg+1)**2), zero above ``deginc``."""
    rng = np.random.default_rng(seed)
    K = (deg + 1) ** 2

    def state(scale):
        c = np.zeros((nsub, 3, K))
        for l in range(deginc + 1):  # noqa: E741
            for m in range(-l, l + 1):
                c[:, :, l * l + l + m] = rng.standard_normal((nsub, 3)) * scale / (1 + l) ** 2
        return c

    A = state(0.02)
    B = A + state(0.004)
    return A, B
