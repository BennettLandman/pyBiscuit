"""Tests for MeshProperty port (`B3.1`) and BISCUIT curvature feature helpers (`B9.1`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pybiscuit.cli import main as cli_main
from pybiscuit.io.txt import read_vertex_txt
from pybiscuit.io.vtk import read_vtk, write_vtk
from pybiscuit.mesh.model import Mesh
from pybiscuit.mesh.property import (
    crop_percentile,
    curvedness_from_principal,
    format_meshlib_vtk,
    gaussian_curvature_from_principal,
    mean_curvature_from_principal,
    mesh_property,
    parc_normalize_feature,
    parc_white_mean_curvature,
    postprocess_property,
    quant_mid_features,
    reg_inflated_mean_curvature,
    reg_white_mean_curvature,
    shape_index_from_principal,
    soft_normalize,
    write_property_txt,
)
from pybiscuit.sphere.icosphere import icosphere

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "data" / "golden-image" / "meshproperty"
OPEN_EX_0050952 = (
    REPO_ROOT.parent / "open-examples" / "0050952"
    if (REPO_ROOT.parent / "open-examples" / "0050952").is_dir()
    else REPO_ROOT / "open-examples" / "0050952"
)


def _bumpy_vtk(tmp_path: Path, level: int = 4) -> Path:
    """Reproduce ``make_image_golden.bumpy_fixture`` input ``bumpy.vtk``."""
    m = icosphere(level)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    x, y, z = u.T
    r = 50 * (1 + 0.10 * np.sin(5 * x + 1) * np.cos(4 * y) + 0.07 * np.sin(6 * z + x * 3))
    out = tmp_path / "bumpy.vtk"
    write_vtk(Mesh(u * r[:, None], m.faces), out, dtype="float")
    return out


def test_analytic_sphere_curvatures_and_area() -> None:
    """On a sphere of radius R, |H| ~ 1/R, K ~ 1/R^2, C ~ 1/R, |S| ~ 1, sum(A) ~ 4*pi*R^2."""
    ico = icosphere(4)
    radius = 50.0
    sphere = Mesh((ico.vertices * radius).astype(np.float32), ico.faces)

    res = mesh_property(
        sphere,
        curvedness=True,
        shape_index=True,
        mean_curvature=True,
        gaussian_curvature=True,
        vertex_area=True,
    )
    assert res.curvedness is not None
    assert res.shape_index is not None
    assert res.mean_curvature is not None
    assert res.gaussian_curvature is not None
    assert res.vertex_area is not None

    np.testing.assert_allclose(np.median(np.abs(res.mean_curvature)), 1.0 / radius, rtol=5e-2)
    np.testing.assert_allclose(np.median(res.curvedness), 1.0 / radius, rtol=5e-2)
    np.testing.assert_allclose(
        np.median(res.gaussian_curvature), 1.0 / (radius * radius), rtol=8e-2
    )
    np.testing.assert_allclose(np.median(np.abs(res.shape_index)), 1.0, atol=5e-2)

    # Quadratic fitting (-q) is accurate to ~0.6% on a level-4 sphere
    res_q = mesh_property(
        sphere,
        curvedness=True,
        shape_index=True,
        mean_curvature=True,
        gaussian_curvature=True,
        quad=True,
    )
    assert res_q.mean_curvature is not None
    assert res_q.curvedness is not None
    assert res_q.gaussian_curvature is not None
    assert res_q.shape_index is not None
    np.testing.assert_allclose(np.median(np.abs(res_q.mean_curvature)), 1.0 / radius, rtol=1e-2)
    np.testing.assert_allclose(np.median(res_q.curvedness), 1.0 / radius, rtol=1e-2)
    np.testing.assert_allclose(
        np.median(res_q.gaussian_curvature), 1.0 / (radius * radius), rtol=2e-2
    )
    np.testing.assert_allclose(np.median(np.abs(res_q.shape_index)), 1.0, atol=1e-2)

    # Total vertex area equals exact sum of triangle areas on the discretized icosphere
    v32 = sphere.vertices.astype(np.float32)
    e1 = v32[sphere.faces[:, 1]] - v32[sphere.faces[:, 0]]
    e2 = v32[sphere.faces[:, 2]] - v32[sphere.faces[:, 0]]
    tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
    np.testing.assert_allclose(res.vertex_area.sum(), tri_area, rtol=1e-5)


def test_invert_negates_cmin_cmax_in_place() -> None:
    """``--invert`` negates both cmin and cmax without swapping them."""
    ico = icosphere(3)
    v = ico.vertices.copy()
    v[:, 0] *= 30.0
    v[:, 1] *= 40.0
    v[:, 2] *= 50.0
    ellipsoid = Mesh(v.astype(np.float32), ico.faces)

    normal = mesh_property(
        ellipsoid,
        curvedness=True,
        shape_index=True,
        mean_curvature=True,
        gaussian_curvature=True,
    )
    inverted = mesh_property(
        ellipsoid,
        curvedness=True,
        shape_index=True,
        mean_curvature=True,
        gaussian_curvature=True,
        invert=True,
    )
    assert normal.cmin is not None and inverted.cmin is not None
    assert normal.cmax is not None and inverted.cmax is not None
    np.testing.assert_allclose(inverted.cmin, -normal.cmin, rtol=0, atol=0)
    np.testing.assert_allclose(inverted.cmax, -normal.cmax, rtol=0, atol=0)
    assert normal.mean_curvature is not None and inverted.mean_curvature is not None
    np.testing.assert_allclose(inverted.mean_curvature, -normal.mean_curvature, rtol=0, atol=0)
    assert normal.curvedness is not None and inverted.curvedness is not None
    np.testing.assert_allclose(inverted.curvedness, normal.curvedness, rtol=0, atol=0)
    assert normal.gaussian_curvature is not None and inverted.gaussian_curvature is not None
    np.testing.assert_allclose(
        inverted.gaussian_curvature, normal.gaussian_curvature, rtol=0, atol=0
    )
    # Because cmin and cmax are negated without swapping, (cmax+cmin)/(cmax-cmin) is unchanged!
    assert normal.shape_index is not None and inverted.shape_index is not None
    np.testing.assert_allclose(inverted.shape_index, normal.shape_index, rtol=0, atol=0)


def test_crop_and_soft_normalize_semantics() -> None:
    """Verify crop_percentile index calculation and two-pass soft_normalize clamping."""
    vals = np.arange(100, dtype=np.float64)
    cropped = crop_percentile(vals, 5.0)
    assert cropped.min() == 5.0
    assert cropped.max() == 95.0
    # p > 50 mirrors to 100 - p
    np.testing.assert_array_equal(crop_percentile(vals, 95.0), cropped)

    # Create distribution with extreme outliers beyond +/- 5 sd
    rng = np.random.default_rng(42)
    x = rng.normal(loc=2.5, scale=3.0, size=5000)
    x[0] = 100.0
    x[1] = -100.0
    normed = soft_normalize(x, threshold=5.0, ddof=0)
    # Soft clamp asymptotes at +/- (threshold + 1) = +/- 6
    assert np.all(normed < 6.0)
    assert np.all(normed > -6.0)
    assert normed[0] > 5.0
    assert normed[1] < -5.0

    # Parcellation normalization uses threshold=3.0, ddof=1 -> asymptotes at +/- 4.0
    pnorm = parc_normalize_feature(x, threshold=3.0)
    assert np.all(pnorm < 4.0)
    assert np.all(pnorm > -4.0)
    assert pnorm[0] > 3.0
    assert pnorm[1] < -3.0


def test_postprocess_pipeline_order() -> None:
    """Verify postprocess_property executes crop -> normbefore -> datasmoothing -> normafter."""
    ico = icosphere(3)
    raw = np.linspace(-20.0, 20.0, ico.n_vertices, dtype=np.float64)
    raw[0] = 500.0
    raw[-1] = -500.0
    out = postprocess_property(
        raw,
        ico.faces,
        prctile=1.0,
        norm_before=True,
        data_smoothing=2,
        norm_after=True,
    )
    assert out.shape == (ico.n_vertices,)
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) < 6.0)


@pytest.mark.parametrize("case", ["all_maps", "quad", "smoothed"])
def test_meshproperty_golden_cases_byte_identical(tmp_path: Path, case: str) -> None:
    """All golden-image MeshProperty cases match byte-for-byte on both .txt and .property.vtk."""
    case_dir = GOLDEN_DIR / case
    meta = json.loads((case_dir / "command.json").read_text())
    bumpy = _bumpy_vtk(tmp_path, level=meta["inputs"]["level"])
    out_prefix = tmp_path / "bumpy"

    argv = ["mesh-property"]
    raw_argv = meta["argv"][1:]
    i = 0
    while i < len(raw_argv):
        tok = raw_argv[i]
        if tok == "-i":
            argv.extend(["-i", str(bumpy)])
            i += 2
        elif tok == "-o":
            argv.extend(["-o", str(out_prefix)])
            i += 2
        else:
            argv.append(tok)
            i += 1

    rc = cli_main(argv)
    assert rc == 0

    for fname in meta["outputs"]:
        gen_path = tmp_path / fname
        gold_path = case_dir / fname
        assert gen_path.is_file(), f"missing output {fname}"
        if fname.endswith(".txt"):
            gen_vals = read_vertex_txt(gen_path)
            gold_vals = read_vertex_txt(gold_path)
            np.testing.assert_allclose(gen_vals, gold_vals, rtol=0, atol=5e-7)
        assert gen_path.read_text() == gold_path.read_text(), f"byte mismatch in {case}/{fname}"


def test_biscuit_feature_helpers_on_bumpy(tmp_path: Path) -> None:
    """Exercise quant_mid_features, reg_white/inflated, and parc_white_mean_curvature."""
    bumpy = read_vtk(_bumpy_vtk(tmp_path, level=3))
    h, s, area = quant_mid_features(bumpy, curv_smoothness=2)
    assert h.shape == (bumpy.n_vertices,)
    assert s.shape == (bumpy.n_vertices,)
    assert area.shape == (bumpy.n_vertices,)

    reg_h = reg_white_mean_curvature(bumpy)
    parc_h_raw = parc_white_mean_curvature(bumpy, normalize=False)
    np.testing.assert_allclose(reg_h, 4.0 * parc_h_raw, rtol=0, atol=1e-15)

    parc_h_norm = parc_white_mean_curvature(bumpy, normalize=True)
    assert np.all(np.abs(parc_h_norm) < 4.0)

    ih = reg_inflated_mean_curvature(bumpy)
    assert ih.shape == (bumpy.n_vertices,)
    assert np.all(np.isfinite(ih))
    assert np.all(np.abs(ih) < 6.0)


def test_meshproperty_cli_error_codes(tmp_path: Path) -> None:
    """Verify CLI error exit codes match CLI11 conventions."""
    assert cli_main(["mesh-property", "--help"]) == 0
    assert cli_main(["mesh-property"]) == 106
    assert cli_main(["mesh-property", "-i", str(tmp_path / "missing.vtk"), "-o", "out"]) == 105
    assert cli_main(["mesh-property", "--unknown-flag"]) == 109
    assert cli_main(["mesh-property", "--prctile", "150"]) == 114


def test_principal_formula_helpers_and_writers(tmp_path: Path) -> None:
    """Unit test direct formula helpers and writers."""
    cmin = np.array([-0.1, 0.0, 0.2], dtype=np.float64)
    cmax = np.array([0.3, 0.0, 0.2], dtype=np.float64)
    c = curvedness_from_principal(cmin, cmax)
    s = shape_index_from_principal(cmin, cmax)
    h = mean_curvature_from_principal(cmin, cmax)
    k = gaussian_curvature_from_principal(cmin, cmax)
    np.testing.assert_allclose(h, [0.1, 0.0, 0.2])
    np.testing.assert_allclose(k, [-0.03, 0.0, 0.04])
    np.testing.assert_allclose(c, np.sqrt((cmin**2 + cmax**2) * 0.5))
    assert np.isnan(s[1])
    np.testing.assert_allclose(s[2], 1.0)

    txt_path = tmp_path / "prop.txt"
    write_property_txt(h, txt_path)
    assert txt_path.read_text().splitlines() == ["0.100000", "0.000000", "0.200000"]

    ico = icosphere(1)
    vtk_text = format_meshlib_vtk(
        ico.vertices, ico.faces, [("MeanCurvature", np.zeros(ico.n_vertices))]
    )
    assert "FIELD ScalarData 1" in vtk_text
    assert f"MeanCurvature 1 {ico.n_vertices} float" in vtk_text


@pytest.mark.reference
@pytest.mark.skipif(not OPEN_EX_0050952.is_dir(), reason="open-examples/0050952 not present")
@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_open_examples_0050952_quant_mid_features(hemi: str) -> None:
    """Compare quant_mid_features against biscuit_FC_v2.2.sif outputs on 0050952."""
    mid = read_vtk(OPEN_EX_0050952 / "surf" / f"{hemi}.mid.vtk")
    h, s, area = quant_mid_features(mid, curv_smoothness=2)

    for ext, arr in [("H", h), ("S", s), ("area", area)]:
        gold_path = OPEN_EX_0050952 / "quant" / f"{hemi}.mid.{ext}.txt"
        gold = read_vertex_txt(gold_path)
        np.testing.assert_allclose(arr, gold, rtol=0, atol=5e-7)
        gold_lines = gold_path.read_text().splitlines()
        gen_lines = [f"{x:f}" for x in arr.tolist()]
        assert gen_lines == gold_lines
