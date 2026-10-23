# ruff: noqa: E501
"""TRACE (CurveExtraction port): committed no-FMA oracle goldens, quirks, CLI (rows B8.1-B8.4)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from conftest import REPO
from fixtures.trace_cases import RADIUS, TRACE_ARGS, TRACE_CASES, trace_mesh
from pybiscuit.cli import main as cli_main
from pybiscuit.trace.extract import extract_curves

GOLD = REPO / "tests" / "data" / "golden" / "trace"
# tie-driven point differences (planes aligned with icosphere symmetry axes; std::sort order of
# equal start-edge ids is libc++-specific): allowed number of differing point flags per case
MAX_POINT_TIES = {"fold_l5m3": 0, "fold_l8m4": 1}


def read_curves(path):
    return [list(map(int, ln.split())) for ln in path.read_text().splitlines() if ln.strip()]


def bary_polylines(path, verts):
    lines = path.read_text().split("\n")
    n, i, out = int(lines[0]), 1, []
    for _ in range(n):
        m = int(lines[i])
        rows = [ln.split() for ln in lines[i + 1 : i + 1 + m]]
        i += 1 + m
        a = np.array([[int(r[0]), int(r[1]), float(r[2])] for r in rows])
        out.append(a[:, 2:3] * verts[a[:, 0].astype(int)] + (1 - a[:, 2:3]) * verts[a[:, 1].astype(int)])
    return np.vstack(out) if out else np.zeros((0, 3))


def hausdorff(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0 if len(a) == len(b) else np.inf
    return max(cKDTree(b).query(a)[0].max(), cKDTree(a).query(b)[0].max())


@pytest.fixture(scope="module", params=list(TRACE_CASES))
def run(request):
    case = request.param
    m = trace_mesh(case)
    res = extract_curves(
        m.vertices, m.faces, sulcus=True, gyrus=True, sulcal_seed=-0.01, gyral_seed=0.01,
        uninit_current="exit",  # the goldens are clang oracle runs; the image continues (brief 22)
    )  # fmt: skip
    return case, m, res


def test_points_match_oracle(run):
    case, m, res = run
    for name, ext in (("sulcal", "spoint"), ("gyral", "gpoint")):
        ref = np.loadtxt(GOLD / case / f"out.{ext}", dtype=int).astype(bool)
        assert res.points[name].shape == ref.shape
        assert (res.points[name] != ref).sum() <= MAX_POINT_TIES[case]


def test_curves_match_oracle(run):
    case, m, res = run
    for name, ext in (("sulcal", "scurve"), ("gyral", "gcurve")):
        ref = read_curves(GOLD / case / f"out.{ext}")
        got = res.curves[name].curve_vertex_lists(True)
        assert len(ref) > 5
        # brief 04: point-set Jaccard >= 0.99, polyline Hausdorff <= 1e-3 * radius
        a, b = {x for c in got for x in c}, {x for c in ref for x in c}
        assert len(a & b) / len(a | b) >= 0.99
        assert got == ref  # in practice byte-identical
        assert res.curves[name].integrity


def test_bary_path_hausdorff(run, tmp_path):
    case, m, res = run
    from pybiscuit.trace.extract import write_geodesic_path

    v = m.vertices.astype(np.float32)
    for name, ext in (("sulcal", "scurve"), ("gyral", "gcurve")):
        out = tmp_path / f"{name}.bary"
        write_geodesic_path(out, res.curves[name], True)
        assert out.read_text() == (GOLD / case / f"out.{ext}.bary").read_text()
        d = hausdorff(bary_polylines(out, v), bary_polylines(GOLD / case / f"out.{ext}.bary", v))
        assert d <= 1e-3 * RADIUS


def test_cli_writes_all_outputs(tmp_path):
    m = trace_mesh("fold_l5m3")
    from fixtures.trace_cases import write_ascii_vtk

    write_ascii_vtk(tmp_path / "in.vtk", m.vertices, m.faces)
    args = ["trace", "-i", str(tmp_path / "in.vtk"), "-o", str(tmp_path / "o"), *TRACE_ARGS, "--cart", "--uninit-current", "exit"]
    assert cli_main(args) == 0
    for ext in ("scurve", "gcurve", "scurve.vtk", "gcurve.vtk", "scurve.cart", "gcurve.cart",
                "scurve.bary", "gcurve.bary", "spoint", "gpoint"):  # fmt: skip
        assert (tmp_path / f"o.{ext}").exists(), ext
    for ext in ("scurve", "gcurve", "scurve.bary", "gcurve.bary", "spoint", "gpoint"):
        want = (GOLD / "fold_l5m3" / f"out.{ext}").read_text()
        assert (tmp_path / f"o.{ext}").read_text() == want
    assert cli_main(["trace", "-i", str(tmp_path / "in.vtk"), "-o", str(tmp_path / "x")]) == -1


def test_simplification_with_both_extractions_is_the_upstream_crash():
    m = trace_mesh("fold_l5m3")
    with pytest.raises(RuntimeError, match="NULL"):
        extract_curves(m.vertices, m.faces, sulcus=True, gyrus=True, simplification=True)


def test_command_json_records_provenance():
    for case in TRACE_CASES:
        meta = json.loads((GOLD / case / "command.json").read_text())
        assert meta["curve_extraction"] == "ded932d" and "ffp-contract=off" in meta["oracle"]


ORACLE = Path.home() / "dev" / "biscuit" / "oracles" / "curve_extraction-nofma" / "bin" / "CurveExtraction"


@pytest.mark.reference
@pytest.mark.skipif(
    not ORACLE.exists() or not os.environ.get("PYBISCUIT_SLOW"),
    reason="slow live oracle run (minutes): needs the no-FMA oracle and PYBISCUIT_SLOW=1",
)
def test_template_lh_white_live_vs_oracle(tmp_path):
    """Default parameters on reg/mb101/lh.white (163,842 vertices), sulcal + gyral: files identical."""
    from conftest import REG_DIR

    surf = REG_DIR / "mb101" / "lh.white.vtk"
    base = ["-i", str(surf), "--sulcus", "--gyrus", "--bary", "--cart", "--interm"]
    subprocess.run([str(ORACLE), *base, "-o", str(tmp_path / "o")], check=True, capture_output=True)
    assert cli_main(["trace", *base, "-o", str(tmp_path / "p")]) == 0
    for ext in ("spoint", "gpoint", "scurve", "gcurve", "scurve.bary", "gcurve.bary",
                "scurve.cart", "gcurve.cart", "scurve.vtk", "gcurve.vtk"):  # fmt: skip
        assert (tmp_path / f"p.{ext}").read_bytes() == (tmp_path / f"o.{ext}").read_bytes(), ext
