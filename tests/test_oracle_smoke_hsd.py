"""Smoke test for the upstream HSD oracle (skip when oracles absent)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from pybiscuit.io import read_vtk, write_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.sphere.icosphere import icosphere

ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
BIN = ORACLES / "hsd" / "bin" / "HSD"

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="HSD oracle not built")


def _prop(u):
    x, y, z = u.T
    return np.sin(3 * x) + np.cos(2 * y) + 0.5 * z**2 + 0.7 * np.sin(2 * x + z)


def test_help():
    r = subprocess.run([str(BIN), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "Hierarchical Spherical Deformation" in r.stdout


def test_two_subject_registration_recovers_rotation(tmp_path):
    m = icosphere(4)
    v = m.vertices
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    th = np.deg2rad(8.0)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    write_vtk(Mesh(v.copy(), m.faces), tmp_path / "s1.sphere.vtk")
    write_vtk(Mesh(v @ rot.T, m.faces), tmp_path / "s2.sphere.vtk")
    for n in ("s1", "s2"):
        np.savetxt(tmp_path / f"{n}.curv.txt", _prop(u), fmt="%.9g")
    argv = [
        str(BIN), "-s", "s1.sphere.vtk", "s2.sphere.vtk", "-p", "s1.curv.txt", "s2.curv.txt",
        "-o", "s1.reg.vtk", "s2.reg.vtk", "-d", "6", "--icosahedron", "3", "--maxIter", "2",
    ]  # fmt: skip
    subprocess.run(argv, cwd=tmp_path, check=True, capture_output=True)
    a = read_vtk(tmp_path / "s1.reg.vtk").vertices
    b = read_vtk(tmp_path / "s2.reg.vtk").vertices
    before = np.abs(v - v @ rot.T).max()
    after = np.abs(a - b).max()
    assert before > 0.1
    assert after < 0.02 * before + 1e-2
