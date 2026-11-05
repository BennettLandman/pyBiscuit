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


# --------------------------------------------------------------------------- #
# landmarks / location features (upstream-disabled options; oracle hsd-lm-nofma)
# --------------------------------------------------------------------------- #
def landmark_ids(level: int = 4, per_curve: int = 10) -> np.ndarray:
    """Vertex indices of three curves on the reference icosphere (a latitude arc, a meridian arc,
    a small circle), snapped to the nearest vertex, duplicates dropped in order."""
    m = icosphere(level)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    t = np.linspace(0, 1, per_curve)
    lat = np.r_[np.full(per_curve, 30.0), -20 + 60 * t, np.full(per_curve, -45.0)]
    lon = np.r_[90 * t, np.full(per_curve, 120.0), np.full(per_curve, -100.0)]
    # the small circle: 15 deg around (lat -45, lon -100)
    a = 2 * np.pi * t[:-1]
    c0 = np.radians([-45.0, -100.0])
    cen = np.array([np.cos(c0[0]) * np.cos(c0[1]), np.cos(c0[0]) * np.sin(c0[1]), np.sin(c0[0])])
    e1 = np.cross(cen, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(cen, e1)
    r = np.radians(15)
    circ = np.cos(r) * cen + np.sin(r) * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)
    la, lo = np.radians(lat[: 2 * per_curve]), np.radians(lon[: 2 * per_curve])
    pts = np.r_[np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1), circ]
    ids = np.argmax(pts @ u.T, axis=1)
    _, first = np.unique(ids, return_index=True)
    return ids[np.sort(first)]


def write_landmark_case(workdir, n_subj=3, level=4, seed=11, location=True):
    """``write_case`` plus ``s<k>.lm.txt`` (same vertex ids for every subject: the warps keep the
    correspondence) and ``s<k>.surf.vtk`` (radius ``80 + 8 f(u)``, scaled by ``1 + 0.05 k``)."""
    names = write_case(workdir, n_subj, level=level, seed=seed)
    work = Path(workdir)
    ids = landmark_ids(level)
    m = icosphere(level)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    f = feature(u, seed)
    for k in range(n_subj):
        np.savetxt(work / f"s{k}.lm.txt", ids, fmt="%d")
        if location:
            write_vtk(Mesh(u * (80 + 8 * f)[:, None] * (1 + 0.05 * k), m.faces), work / f"s{k}.surf.vtk")
    return names


#: unit-level states with landmarks (harness "eval"); ``flags`` follow -s/-p
LM_EVAL_CASES = {
    "lm_gd_d3": dict(deg=6, ico=4, deginc=3, beg=0, end=3, trans=1, flags=["--weightMap", "0.5"]),
    "lm_loc_gd_final": dict(deg=6, ico=4, deginc=6, beg=0, end=6, trans=2, flags=["--weightMap", "0.5", "--locationWeight", "0.7"]),
    "lm_newton_band3": dict(deg=6, ico=6, deginc=3, beg=3, end=3, trans=1, flags=["--weightMap", "0.5"]),
    "lm_newton_final": dict(deg=6, ico=6, deginc=6, beg=0, end=6, trans=2, flags=["--weightMap", "0.5", "--locationWeight", "0.7"]),
}

#: full CLI runs with the upstream-disabled options (flags after -s/-p/-o/--outputcoeff; LM/SURF expand)
LM_RUN_CASES = {
    "lm_gd_ico4": dict(nsub=3, flags=["-d", "6", "--icosahedron", "4", "LM", "--weightMap", "0.5"]),
    "lm_loc_ico4": dict(nsub=3, flags=["-d", "6", "--icosahedron", "4", "--noguess", "SURF", "--locationWeight", "0.7"]),
    "lm_loc_resample_ico5": dict(nsub=3, flags=["-d", "6", "--icosahedron", "5", "--resample", "LM", "SURF", "--locationWeight", "1", "--weightMap", "0.3"]),
    "lm_newton_ico6": dict(nsub=3, flags=["-d", "6", "--icosahedron", "6", "--maxIter", "4", "--noguess", "LM", "--weightMap", "0.5"]),
    "lm_pairwise_ico5": dict(nsub=2, flags=["-d", "6", "--icosahedron", "5", "--fixedSubjects", "0", "LM", "--weightMap", "0.2"]),
    "lm_dirs_ico4": dict(nsub=3, flags=["-d", "4", "--icosahedron", "4", "--noguess", "--maxIter", "3", "--landmarkDir", "lmdir", "--surfaceDir", "surfdir", "--locationWeight", "0.5"]),
}


def expand_lm_flags(flags, nsub, work=None):
    """Replace the ``LM``/``SURF`` placeholders by ``-l``/``--surface`` file lists; for the ``*Dir``
    forms, create the directories (with one decoy file each) under ``work``."""
    out = []
    for f in flags:
        if f == "LM":
            out += ["-l"] + [f"s{k}.lm.txt" for k in range(nsub)]
        elif f == "SURF":
            out += ["--surface"] + [f"s{k}.surf.vtk" for k in range(nsub)]
        else:
            out.append(f)
    if work is not None and "--landmarkDir" in out:
        w = Path(work)
        (w / "lmdir").mkdir(exist_ok=True)
        (w / "surfdir").mkdir(exist_ok=True)
        for k in range(nsub):
            (w / "lmdir" / f"s{k}.lm.txt").write_text((w / f"s{k}.lm.txt").read_text())
            (w / "surfdir" / f"s{k}.surf.vtk").write_bytes((w / f"s{k}.surf.vtk").read_bytes())
        (w / "lmdir" / "other.lm.txt").write_text("0\n")  # trimmed: name matches no sphere
        (w / "lmdir" / "notes.dat").write_text("x\n")  # wrong suffix
    return out
