"""TRACE -> LGI chain (plan row B20.1): BISCUIT's exact calls, ported TRACE vs the oracle chain.

Golden: ``tests/data/golden/lgi-chain/`` from ``scripts/oracles/make_golden_lgi_chain.py`` (no-FMA
oracle CurveExtraction + Gyrification with the argv of spec §8 / §9.3).
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from _compare import assert_close_float32
from fixtures.lgi import fold_case
from pybiscuit.cli import main as cli_main
from pybiscuit.io import read_vtk, write_vtk
from pybiscuit.quant.hull import outer_hull
from pybiscuit.quant.lgi_pipeline import (
    gyrification_argv,
    hull_correspondence,
    lgi_pipeline,
    mesh_area,
)
from pybiscuit.trace.pipeline import curve_argv, curve_calls, curve_extract_pipeline, num2str

G = Path(__file__).parent / "data" / "golden" / "lgi-chain"
CURVES = ("lh.gray.scurve", "lh.gray.scurve.bary", "lh.white.gcurve", "lh.white.gcurve.bary")


def test_num2str_matches_matlab():
    assert [num2str(x) for x in (4, 4.0, 2.5, 0.05, 12.25, 3.14159265)] == [
        "4", "4", "2.5", "0.05", "12.25", "3.1416",
    ]  # fmt: skip


def test_biscuit_trace_argv_is_the_spec_call():
    """T-1: spec §8 — ``--epRad r --nhDist r --lsTh d``, nothing else non-default (named test)."""
    got = curve_argv("sulcus", "S/lh.gray.vtk", "T/lh.gray", trace_r=4, trace_delta=2.5,
                     n_threads=8)
    assert " ".join(got) == (
        "--sulcus -i S/lh.gray.vtk -o T/lh.gray --nThreads 8 --epRad 4 --nhDist 4 --lsTh 2.5 "
        "--bary --noVtk"
    )
    got = curve_argv("gyrus", "S/rh.white.vtk", "T/rh.white", trace_r=3, trace_delta=1.5)
    assert got[:2] == ["--gyrus", "-i"]
    assert got[-8:] == ["--epRad", "3", "--nhDist", "3", "--lsTh", "1.5", "--bary", "--noVtk"]


def test_trace_call_order_and_outputs():
    calls = curve_calls("S", "T")
    assert [(a[0], Path(a[a.index("-i") + 1]).name) for a, _ in calls] == [
        ("--sulcus", "lh.gray.vtk"), ("--gyrus", "lh.white.vtk"),
        ("--sulcus", "rh.gray.vtk"), ("--gyrus", "rh.white.vtk"),
    ]  # fmt: skip
    assert [c[0].name for _, c in calls] == [
        "lh.gray.scurve", "lh.white.gcurve", "rh.gray.scurve", "rh.white.gcurve"
    ]


def test_biscuit_gyrification_argv_has_percent_f_numbers():
    """L-1: the ``lgi`` script passes kernel / anisotropy / reference area through ``%f``."""
    a = gyrification_argv("p.vtk", "h.vtk", "s.bary", "g.bary", "o", n_threads=4)
    assert a[a.index("-m") :] == [
        "-m", "316.000000", "--speed", "0.200000", "--refHullArea", "77100.000000",
        "--nThreads", "4",
    ]  # fmt: skip


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    w = tmp_path_factory.mktemp("chain")
    for f in ("lh.gray.vtk", "lh.white.vtk"):
        shutil.copy(G / f, w / f)
    curve_extract_pipeline(w, w / "curve", hemis=("lh",))
    run = lgi_pipeline(
        w / "lh.gray.vtk", w / "curve/lh.gray.scurve.bary", w / "curve/lh.white.gcurve.bary", w,
        outer=read_vtk(G / "lh.gray.hull.corr.vtk"),
    )  # fmt: skip
    return w, run


@pytest.mark.parametrize("name", CURVES)
def test_ported_trace_curves_equal_oracle(chain, name):
    w, _ = chain
    assert (w / "curve" / name).read_bytes() == (G / name).read_bytes()


def test_lgi_on_ported_curves_matches_oracle_chain(chain):
    _, run = chain
    gold_file = next(G.glob("lh.gray.lgi.map.*.txt"))
    assert_close_float32(run.lgi, np.loadtxt(gold_file), label="LGI (port chain vs oracle chain)")
    assert run.lgi_file.name == "lh.gray.LGI.txt" and float(run.lgi.max()) > 1.0


def test_curve_extraction_is_skipped_when_outputs_exist(chain, monkeypatch):
    w, _ = chain
    import pybiscuit.trace.pipeline as tp

    monkeypatch.setattr(tp, "trace_main", lambda argv: pytest.fail("TRACE re-ran"))
    curve_extract_pipeline(w, w / "curve", hemis=("lh",))


def test_hull_correspondence_standin_properties():
    surf, _ = fold_case(a=0.25, level=4)
    hull = outer_hull(surf, grid=64)
    corr = hull_correspondence(surf, hull)
    assert corr.n_vertices == surf.n_vertices and np.array_equal(corr.faces, surf.faces)
    assert {tuple(v) for v in corr.vertices} <= {tuple(v) for v in hull.vertices}
    assert mesh_area(hull) > 0


def test_end_to_end_cli_with_outer_hull(tmp_path):
    surf, _ = fold_case(a=0.25, level=4)
    for f in ("lh.gray.vtk", "lh.white.vtk"):
        write_vtk(surf, tmp_path / f, binary=False, dtype="float")
    out = tmp_path / "o"
    argv = ["lgi-pipeline", "--surf-dir", str(tmp_path), "--out", str(out), "--hemi", "lh"]
    assert cli_main(argv) == 0
    lgi = np.loadtxt(out / "lh.gray.LGI.txt")
    assert lgi.shape == (surf.n_vertices,) and np.isfinite(lgi).all()
    assert (out / "curve" / "lh.white.gcurve.bary").exists()
    assert (out / "lgi_pipeline.json").exists()


def test_curves_unchanged_by_six_decimal_vtk_rounding(tmp_path):
    """T-7: BISCUIT writes VTK with %f (6 decimals); rounding the input must not matter."""
    surf = read_vtk(G / "lh.gray.vtk")
    for f in ("lh.gray.vtk", "lh.white.vtk"):
        text = (G / f).read_text().splitlines()
        i = next(k for k, ln in enumerate(text) if ln.startswith("POINTS"))
        n = int(text[i].split()[1])
        for k in range(i + 1, i + 1 + n):
            text[k] = " ".join(f"{float(x):.6f}" for x in text[k].split())
        (tmp_path / f).write_text("\n".join(text) + "\n")
    assert surf.n_vertices == n
    curve_extract_pipeline(tmp_path, tmp_path / "curve", hemis=("lh",))
    for name in CURVES:
        assert (tmp_path / "curve" / name).read_bytes() == (G / name).read_bytes(), name


def test_flat_patch_region_does_not_stop_the_chain(tmp_path):
    """T-4: the hemisphere meshes include a (flat) medial-wall patch; curves are not masked."""
    surf, hull = fold_case(a=0.25, level=4)
    v = surf.vertices.copy()
    cap = v[:, 2] > 0.8 * np.linalg.norm(v, axis=1).max()  # polar cap flattened to a plane
    v[cap, 2] = v[cap, 2].min()
    patched = type(surf)(v, surf.faces, frame=surf.frame, units=surf.units)
    for f in ("lh.gray.vtk", "lh.white.vtk"):
        write_vtk(patched, tmp_path / f, binary=False, dtype="float")
    curve_extract_pipeline(tmp_path, tmp_path / "curve", hemis=("lh",))
    run = lgi_pipeline(
        tmp_path / "lh.gray.vtk", tmp_path / "curve/lh.gray.scurve.bary",
        tmp_path / "curve/lh.white.gcurve.bary", tmp_path, outer=hull,
    )  # fmt: skip
    assert cap.sum() > 20 and np.isfinite(run.lgi).all()
