"""Smoke test for the upstream CurveExtraction (TRACE) oracle (skip when oracles absent)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from pybiscuit.io import write_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.sphere.icosphere import icosphere

ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
BIN = ORACLES / "curve_extraction" / "bin" / "CurveExtraction"

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="CurveExtraction oracle not built")


def bumpy(tmp_path):
    m = icosphere(5)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    x, y, z = u.T
    r = 50 * (1 + 0.10 * np.sin(5 * x + 1) * np.cos(4 * y) + 0.07 * np.sin(6 * z + x * 3))
    write_vtk(Mesh(u * r[:, None], m.faces), tmp_path / "bumpy.vtk")


def test_help():
    r = subprocess.run([str(BIN), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "TRACE" in r.stdout


def test_extracts_curves(tmp_path):
    bumpy(tmp_path)
    argv = [
        str(BIN), "-i", "bumpy.vtk", "-o", "bumpy", "--sulcus", "--gyrus", "--cart", "--bary",
        "--interm", "--sulcalSeed", "-0.01", "--gyralSeed", "0.01",
    ]  # fmt: skip
    subprocess.run(argv, cwd=tmp_path, check=True, capture_output=True)
    for ext in ("scurve", "gcurve", "scurve.bary", "gcurve.bary", "spoint", "gpoint"):
        assert (tmp_path / f"bumpy.{ext}").exists(), ext
    assert len((tmp_path / "bumpy.gcurve").read_text().split("\n")) > 10
    assert len((tmp_path / "bumpy.scurve").read_text().strip().split("\n")) >= 1
