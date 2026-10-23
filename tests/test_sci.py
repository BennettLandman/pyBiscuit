"""Tests for Shape Complexity Index (B9.6)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from fixtures.meshes import icosphere
from pybiscuit.quant.sci import compute_shape_index, shape_complexity_index
from pybiscuit.quant.sci_cli import main as sci_main

DATA_DIR = Path(__file__).resolve().parent / "data" / "open-examples" / "0050952"


def test_shape_index_formula():
    # Spherical cup: cmin == cmax > 0 -> 1.0
    cmin = np.array([1.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    cmax = np.array([1.0, -1.0, 0.0, 2.0, 1.0], dtype=np.float32)
    si = compute_shape_index(cmin, cmax)
    assert si[0] == pytest.approx(1.0)
    assert si[1] == pytest.approx(-1.0)
    assert si[2] == pytest.approx(1.0)
    # Saddle: cmin = -1, cmax = 1 -> (2/pi) * atan(0/2) = 0
    cmin_s = np.array([-1.0], dtype=np.float32)
    cmax_s = np.array([1.0], dtype=np.float32)
    si_saddle = compute_shape_index(cmin_s, cmax_s)
    assert si_saddle[0] == pytest.approx(0.0)


def test_sci_synthetic():
    mesh = icosphere(3, 10.0)
    sci = shape_complexity_index(mesh, kernel_size=3.0, step=5)
    assert sci.shape == (mesh.n_vertices,)
    assert np.all(np.isfinite(sci))
    assert np.all((sci >= 0.0) & (sci <= 1.0))


_MISSING_DATA = not (DATA_DIR / "surf" / "lh.mid.vtk.gz").exists()


@pytest.mark.skipif(_MISSING_DATA, reason="open-examples data missing")
def test_sci_matches_open_examples_0050952_lh():
    mesh_path = DATA_DIR / "surf" / "lh.mid.vtk.gz"
    golden_path = DATA_DIR / "quant" / "lh.mid.SCI.txt.gz"

    with gzip.open(golden_path, "rt") as f:
        golden_sci = np.array([float(line.strip()) for line in f if line.strip()], dtype=np.float32)

    pred_sci = shape_complexity_index(mesh_path, kernel_size=3.0, step=5)
    assert pred_sci.shape == golden_sci.shape

    diffs = np.abs(pred_sci - golden_sci)
    corr = float(np.corrcoef(pred_sci, golden_sci)[0, 1])

    # Tolerance check: Pearson r >= 0.995, mean abs error <= 0.02
    assert corr >= 0.995, f"Pearson r = {corr:.6f} < 0.995"
    assert np.mean(diffs) <= 0.02, f"Mean abs error = {np.mean(diffs):.6f} > 0.02"
    assert np.percentile(diffs, 95) <= 0.05, f"p95 error = {np.percentile(diffs, 95):.6f} > 0.05"


def test_sci_cli(tmp_path: Path):
    mesh = icosphere(2, 10.0)
    from pybiscuit.io.vtk import write_vtk

    mesh_file = tmp_path / "sphere.vtk"
    out_file = tmp_path / "sci.txt"
    write_vtk(mesh, mesh_file)

    rc = sci_main(["-i", str(mesh_file), "-o", str(out_file), "-k", "3.0", "-s", "5"])
    assert rc == 0
    assert out_file.exists()

    lines = out_file.read_text().splitlines()
    assert len(lines) == mesh.n_vertices
