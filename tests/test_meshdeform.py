"""Tests for MeshDeform port (`B3.2`, `B5.6`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pybiscuit.cli import main as cli_main
from pybiscuit.io.vtk import read_vtk, write_vtk
from pybiscuit.mesh.deform import (
    add_gaussian_noise,
    apply_matrix,
    center_vertices,
    invert_faces,
    make_sphere,
    mesh_deform,
    negated_center,
    scale_vertices,
    translate_centered_vertices,
    translate_vertices,
)
from pybiscuit.mesh.model import Mesh
from pybiscuit.sphere.icosphere import icosphere

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "data" / "golden-image" / "meshdeform"


def _bumpy_vtk(tmp_path: Path, level: int = 4) -> Path:
    """Reproduce ``make_image_golden.bumpy_fixture`` input ``bumpy.vtk``."""
    m = icosphere(level)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    x, y, z = u.T
    r = 50 * (1 + 0.10 * np.sin(5 * x + 1) * np.cos(4 * y) + 0.07 * np.sin(6 * z + x * 3))
    out = tmp_path / "bumpy.vtk"
    write_vtk(Mesh(u * r[:, None], m.faces), out, dtype="float")
    return out


@pytest.mark.parametrize("case", ["translate", "matrix", "sphere1", "sphere2"])
def test_meshdeform_golden_cases_byte_identical(tmp_path: Path, case: str) -> None:
    """All golden-image MeshDeform cases match byte-for-byte on deformed.vtk."""
    case_dir = GOLDEN_DIR / case
    meta = json.loads((case_dir / "command.json").read_text())
    bumpy = _bumpy_vtk(tmp_path, level=meta["inputs"]["level"])
    out_vtk = tmp_path / "deformed.vtk"

    argv = ["mesh-deform"]
    raw_argv = meta["argv"][1:]
    i = 0
    while i < len(raw_argv):
        tok = raw_argv[i]
        if tok == "-i":
            argv.extend(["-i", str(bumpy)])
            i += 2
        elif tok == "-o":
            argv.extend(["-o", str(out_vtk)])
            i += 2
        else:
            argv.append(tok)
            i += 1

    rc = cli_main(argv)
    assert rc == 0
    assert out_vtk.is_file()

    gold_vtk = case_dir / "deformed.vtk"
    gen_mesh = read_vtk(out_vtk)
    gold_mesh = read_vtk(gold_vtk)
    np.testing.assert_array_equal(gen_mesh.faces, gold_mesh.faces)
    np.testing.assert_allclose(gen_mesh.vertices, gold_mesh.vertices, rtol=0, atol=1e-9)
    assert out_vtk.read_text() == gold_vtk.read_text()


def test_translate_subtracts_offset_and_center_prints_negated() -> None:
    """Verify MeshDeform -t subtracts the vector and -c returns the negated centroid."""
    ico = icosphere(2)
    shifted = translate_vertices(ico.vertices, [-10.0, 20.0, -5.0])
    # Subtracting [-10, 20, -5] shifts centroid to [+10, -20, +5]
    np.testing.assert_allclose(shifted.mean(axis=0), [10.0, -20.0, 5.0], atol=1e-5)
    # negated_center returns -centroid = [-10, +20, -5]
    nc = negated_center(shifted)
    np.testing.assert_allclose(nc, [-10.0, 20.0, -5.0], atol=1e-5)
    # Passing negated_center to translate_vertices subtracts [-10, 20, -5] -> shifts further!
    # Passing +centroid [10, -20, 5] to translate_vertices re-centers the mesh at origin:
    recentered = translate_vertices(shifted, [10.0, -20.0, 5.0])
    np.testing.assert_allclose(recentered.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-5)


def test_translate_centered_and_scale_and_invert_faces() -> None:
    """Verify translate_centered_vertices, scale_vertices, and invert_faces."""
    ico = icosphere(2)
    v_off = ico.vertices + np.array([100.0, -50.0, 25.0])
    tc = translate_centered_vertices(v_off, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(tc.mean(axis=0), [-1.0, -2.0, -3.0], atol=5e-5)

    scaled = scale_vertices(ico.vertices, 2.5)
    np.testing.assert_allclose(scaled, ico.vertices.astype(np.float32) * 2.5, rtol=0, atol=1e-6)
    # scale == 0.0 is a no-op in Mesh::scaling
    np.testing.assert_array_equal(
        scale_vertices(ico.vertices, 0.0), ico.vertices.astype(np.float32)
    )

    inv_f = invert_faces(ico.faces)
    np.testing.assert_array_equal(inv_f[:, 0], ico.faces[:, 2])
    np.testing.assert_array_equal(inv_f[:, 1], ico.faces[:, 1])
    np.testing.assert_array_equal(inv_f[:, 2], ico.faces[:, 0])


def test_apply_matrix_accepts_flat12_and_4x4() -> None:
    """apply_matrix accepts 12-float sequence, (3, 4), or (4, 4) array."""
    ico = icosphere(2)
    m12 = [1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.5]
    m44 = np.eye(4, dtype=np.float64)
    m44[:3, :4] = np.array(m12, dtype=np.float64).reshape(3, 4)
    v1 = apply_matrix(ico.vertices, m12)
    v2 = apply_matrix(ico.vertices, m44)
    np.testing.assert_array_equal(v1, v2)
    with pytest.raises(ValueError, match="matrix must have 12 elements"):
        apply_matrix(ico.vertices, [1.0, 2.0, 3.0])


def test_add_gaussian_noise_and_mesh_deform_pipeline(tmp_path: Path, capsys) -> None:
    """Verify add_gaussian_noise along vertex normals and combined CLI flags."""
    ico = icosphere(2)
    rng = np.random.default_rng(123)
    noisy = add_gaussian_noise(ico.vertices, ico.faces, 0.5, rng=rng)
    radii = np.linalg.norm(noisy, axis=1)
    assert np.std(radii) > 0.1
    np.testing.assert_allclose(np.mean(radii), 1.0, atol=0.15)

    in_vtk = tmp_path / "in.vtk"
    out_vtk = tmp_path / "out.vtk"
    write_vtk(
        Mesh((ico.vertices + np.array([3.0, -4.0, 5.0])).astype(np.float32), ico.faces),
        in_vtk,
        dtype="float",
    )

    rc = cli_main(
        [
            "mesh-deform",
            "-i",
            str(in_vtk),
            "-o",
            str(out_vtk),
            "-c",
            "--scale",
            "2.0",
            "--invertFace",
            "--smooth",
            "1",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out.strip().split()
    assert len(captured) == 3
    np.testing.assert_allclose([float(x) for x in captured], [-3.0, 4.0, -5.0], atol=1e-4)
    assert out_vtk.is_file()


def test_meshdeform_cli_error_codes(tmp_path: Path) -> None:
    """Verify CLI error exit codes match CLI11 conventions."""
    assert cli_main(["mesh-deform", "--help"]) == 0
    assert cli_main(["mesh-deform"]) == 106
    assert cli_main(["mesh-deform", "-i", str(tmp_path / "missing.vtk")]) == 105
    assert cli_main(["mesh-deform", "--unknown-flag"]) == 109
    assert cli_main(["mesh-deform", "--sphere", "3"]) == 114
    assert cli_main(["mesh-deform", "--scale", "-1"]) == 114
    with pytest.raises(ValueError, match="sphere_type must be 1 or 2"):
        make_sphere(icosphere(1).vertices, icosphere(1).faces, 3)
    assert center_vertices(np.empty((0, 3), dtype=np.float32)).shape == (0, 3)
    res = mesh_deform(
        icosphere(1), translate_centered=[1.0, 2.0, 3.0], noise=0.01, rng=np.random.default_rng(0)
    )
    assert res.mesh.n_vertices == icosphere(1).n_vertices
