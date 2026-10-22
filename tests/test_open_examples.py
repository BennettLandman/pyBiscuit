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
