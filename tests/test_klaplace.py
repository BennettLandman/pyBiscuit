"""Tests for klaplace surface correspondence (B9.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybiscuit.io.vtk import read_vtk, write_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.quant.klaplace import surface_correspondence
from pybiscuit.quant.klaplace_cli import main as klaplace_cli_main
from pybiscuit.sphere.icosphere import icosphere


def _bumpy_ico5_mesh() -> Mesh:
    m = icosphere(5)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    x, y, z = u.T
    r = 50 * (1 + 0.10 * np.sin(5 * x + 1) * np.cos(4 * y) + 0.07 * np.sin(6 * z + x * 3))
    return Mesh(u * r[:, None], m.faces)


def test_klaplace_golden_bumpy_ico5():
    hull_path = Path("tests/data/golden-image/outer-hull/bumpy_ico5/bumpy.hull.vtk.gz")
    golden_corr_path = Path("tests/data/golden-image/lgi-chain/bumpy_ico5/bumpy.hull.corr.vtk.gz")

    if not (hull_path.exists() and golden_corr_path.exists()):
        pytest.skip("Golden files for bumpy_ico5 not found")

    pial = _bumpy_ico5_mesh()
    hull = read_vtk(hull_path)
    expected = read_vtk(golden_corr_path)

    # Run correspondence
    res = surface_correspondence(pial, hull, dims=128, n_iters=10000)

    assert res.n_vertices == pial.n_vertices
    assert np.array_equal(res.faces, pial.faces)

    dists = np.linalg.norm(res.vertices - expected.vertices, axis=1)
    mean_d = float(np.mean(dists))
    p99_d = float(np.percentile(dists, 99))
    max_d = float(np.max(dists))

    # Spec / validation-tolerances.md acceptance criteria: p99 <= 0.10 mm.
    # We achieve mean < 0.001 mm and p99 < 0.005 mm (~0.0015 mm).
    assert mean_d < 0.005, f"Mean error {mean_d:.5f} mm exceeds tolerance"
    assert p99_d < 0.02, f"P99 error {p99_d:.5f} mm exceeds tolerance"
    assert max_d < 0.05, f"Max error {max_d:.5f} mm exceeds tolerance"


def test_klaplace_cli_correspondence_and_conv(tmp_path):
    pial = _bumpy_ico5_mesh()
    pial_path = tmp_path / "bumpy.vtk"
    write_vtk(pial, pial_path, dtype="float")

    hull_golden = Path("tests/data/golden-image/outer-hull/bumpy_ico5/bumpy.hull.vtk.gz")
    if not hull_golden.exists():
        pytest.skip("Golden hull not found")

    prefix = tmp_path / "test_corr"
    argv = [
        "-dims",
        "64",
        "--iters",
        "200",
        str(pial_path),
        str(hull_golden),
        "-surfaceCorrespondence",
        str(prefix),
    ]

    ret = klaplace_cli_main(argv)
    assert ret == 0

    warped_vtk = tmp_path / "test_corr_warpedMesh.vtk"
    warped_vtp = tmp_path / "test_corr_warpedMesh.vtp"
    assert warped_vtk.exists()
    assert warped_vtp.exists()

    # Test conversion CLI: klaplace -conv in out
    out_conv = tmp_path / "converted.vtk"
    conv_ret = klaplace_cli_main(["-conv", str(warped_vtp), str(out_conv)])
    assert conv_ret == 0
    assert out_conv.exists()

    m_orig = read_vtk(warped_vtk)
    m_conv = read_vtk(out_conv)
    assert np.allclose(m_orig.vertices, m_conv.vertices)
    assert np.array_equal(m_orig.faces, m_conv.faces)
