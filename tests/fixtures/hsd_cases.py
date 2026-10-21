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
    return names
