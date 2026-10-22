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
