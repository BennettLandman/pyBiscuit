"""Smoke test for the upstream Gyrification (LGI) oracle (skip when oracles absent)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from pybiscuit.io import write_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.sphere.icosphere import icosphere

ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
BIN = ORACLES / "gyrification" / "bin" / "Gyrification"
CURVE = ORACLES / "curve_extraction" / "bin" / "CurveExtraction"

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="Gyrification oracle not built")


def test_help():
    r = subprocess.run([str(BIN), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "Local Gyrification Index" in r.stdout


@pytest.mark.skipif(not CURVE.exists(), reason="CurveExtraction oracle not built")
def test_lgi_on_bumpy_sphere(tmp_path):
    m = icosphere(5)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    x, y, z = u.T
    r = 50 * (1 + 0.10 * np.sin(5 * x + 1) * np.cos(4 * y) + 0.07 * np.sin(6 * z + x * 3))
    write_vtk(Mesh(u * r[:, None], m.faces), tmp_path / "bumpy.vtk")
    write_vtk(Mesh(u * 55, m.faces), tmp_path / "hull.vtk")
    subprocess.run(
        [str(CURVE), "-i", "bumpy.vtk", "-o", "bumpy", "--sulcus", "--gyrus",
         "--sulcalSeed", "-0.01", "--gyralSeed", "0.01"],
        cwd=tmp_path, check=True, capture_output=True,
    )  # fmt: skip
    subprocess.run(
        [str(BIN), "-i", "bumpy.vtk", "--outer", "hull.vtk", "-g", "bumpy.gcurve",
         "-s", "bumpy.scurve", "-o", "lgi"],
        cwd=tmp_path, check=True, capture_output=True,
    )  # fmt: skip
    outs = list(tmp_path.glob("lgi.lgi.map.*.txt"))
    assert len(outs) == 1
    lgi = np.loadtxt(outs[0])
    assert lgi.shape == (m.n_vertices,)
    assert np.isfinite(lgi).all() and lgi.min() >= 1.0 - 1e-6 and lgi.max() > 1.0
