"""Smoke tests for the upstream SphericalRemesh oracle (skip when oracles absent)."""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
BIN = ORACLES / "spherical-remesh" / "bin" / "SphericalRemesh"
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "data" / "golden" / "spherical-remesh"

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="SphericalRemesh oracle not built")


def _make_golden():
    path = ROOT / "scripts" / "oracles" / "make_golden.py"
    spec = importlib.util.spec_from_file_location("make_golden", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_help():
    r = subprocess.run([str(BIN), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "Spherical Remeshing Tool" in r.stdout


@pytest.mark.parametrize("case", ["down5to4", "up3to4"])
def test_oracle_reproduces_committed_golden(case, tmp_path):
    mg = _make_golden()
    mg.sr_fixture(tmp_path, *mg.SR_CASES[case])
    argv = mg.sr_argv(ORACLES)
    subprocess.run(argv, cwd=tmp_path, check=True, capture_output=True)
    gold = GOLDEN / case
    meta = json.loads((gold / "command.json").read_text())
    assert meta["oracle"] == "upstream-macos-arm64"
    got = np.loadtxt(tmp_path / "outprop.prop.txt")
    ref = np.loadtxt(gold / "outprop.prop.txt")
    np.testing.assert_allclose(got, ref, atol=2e-6)  # 6-decimal text output


def test_up5_to_6_runs(tmp_path):
    mg = _make_golden()
    mg.sr_fixture(tmp_path, 5, 6)
    subprocess.run(mg.sr_argv(ORACLES), cwd=tmp_path, check=True, capture_output=True)
    out = np.loadtxt(tmp_path / "outprop.prop.txt")
    assert out.shape == (40962,)
    assert np.isfinite(out).all()
