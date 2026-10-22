# ruff: noqa: E501
"""Replays on the open examples (brief 22): ABIDE subjects with real ``biscuit_FC_v2.2`` outputs.

All tests are ``reference``-marked and skip when ``~/dev/biscuit/open-examples`` is absent; the data is
never in git.  The full per-stage comparison lives in ``scripts/validation/open_examples.py``.
"""

from pathlib import Path

import pytest

OPEN = Path.home() / "dev" / "biscuit" / "open-examples"
SUBJECT = OPEN / "0050952"

pytestmark = pytest.mark.reference
needs_data = pytest.mark.skipif(not SUBJECT.exists(), reason="open-examples not available")


@needs_data
def test_gyral_curves_match_the_image_bytes(tmp_path):
    """GyralCurve::refineCurves reads an uninitialised int upstream; the image runs >= 2 passes, so the
    default (``uninit_current="continue"``) must reproduce its ``.gcurve`` and ``.gcurve.bary`` exactly."""
    from pybiscuit.trace.extract import main

    prefix = tmp_path / "lh.white"
    argv = ["--gyrus", "-i", str(SUBJECT / "surf" / "lh.white.vtk"), "-o", str(prefix),
            "--nThreads", "4", "--epRad", "4", "--nhDist", "4", "--lsTh", "2.5", "--bary", "--noVtk"]  # fmt: skip
    assert main(argv) == 0
    for ext in ("gcurve", "gcurve.bary"):
        got = (tmp_path / f"lh.white.{ext}").read_bytes()
        assert got == (SUBJECT / "curve" / f"lh.white.{ext}").read_bytes(), ext


def test_strel_ball_excludes_the_rim():
    """``strel('ball', r, h)``: neighbourhood ``x^2 + y^2 < r^2`` plus the centre (inferred from the image,
    see ``quant/hull.py::_ball``).  ``ball(1, 1)`` is the centre pixel, so the hull dilation is the identity."""
    from pybiscuit.quant.hull import _ball

    dx, dy, ht = _ball(1, 1)
    assert dx.tolist() == [0] and dy.tolist() == [0] and ht.tolist() == [1.0]
    dx, dy, ht = _ball(15, 15)
    assert len(dx) == 697 and ht.max() == 15.0  # 709 lattice points of the closed disc, minus 12 on the rim


@needs_data
def test_sulcal_depth_has_no_hull_bias_against_the_image():
    """Regression guard for the rim fix (not an acceptance bound): with the rim included the port's depth
    was +0.71 mm too large on average; the remaining differences are voxel-scale hull details."""
    import numpy as np

    from pybiscuit.quant.sulcal_depth import sulcal_depth

    ours = sulcal_depth(SUBJECT / "surf" / "lh.gray.vtk")
    ref = np.loadtxt(SUBJECT / "quant" / "lh.gray.SD.txt")
    assert abs(float((ours - ref).mean())) < 0.1
    assert np.corrcoef(ours, ref)[0, 1] > 0.9995


@needs_data
def test_fs_route_chain_reaches_the_images_basin(tmp_path):
    """Regression guard for the call-1 guess fix (Q17), not an acceptance bound: upstream's raw ``--icomesh``
    guess lands ~113 degrees (median) from the image's registered sphere; the default is within a degree."""
    import numpy as np

    from pybiscuit.hsd.chain import ChainPaths, run_chain
    from pybiscuit.io import read_vtk
    from pybiscuit.sphere.remesh import run as remesh_run
    from pybiscuit.surfprep.freesurfer import prepare_fs_subject

    ref = Path.home() / "dev" / "biscuit" / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit"
    if not ref.exists():
        pytest.skip("reference/ not available")
    prepare_fs_subject(SUBJECT, tmp_path, hemis=("lh",))
    uni = ref / "sphere" / "sphere_327680.vtk"
    for f in ("curv", "inflated.H", "sulc"):
        remesh_run(tmp_path / "lh.sphere.vtk", ref=uni, property=[tmp_path / f"lh.{f}.txt"],
                   output_property=tmp_path / "lh.res", quiet=True)  # fmt: skip
    run_chain(ChainPaths("lh", tmp_path, ref / "reg" / "mb101", ref / "sphere"), route="fs", n_threads=4)
    ours = tmp_path / "ours.vtk"
    remesh_run(tmp_path / "lh.sphere.vtk", coeff=tmp_path / "lh.coeff.txt", deform=ours, deg=10, quiet=True)
    a = read_vtk(ours).vertices
    b = read_vtk(SUBJECT / "reg" / "lh.sphere.reg.vtk").vertices
    a, b = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (a, b))
    ang = np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))
    assert np.median(ang) < 1.0 and np.percentile(ang, 99) < 3.0
