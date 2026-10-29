"""Tests for executable pipeline stages (plan B10.2)."""

import json

import pytest

from pybiscuit.config import load_config
from pybiscuit.pipeline import (
    InputFindings,
    Product,
    Run,
)
from pybiscuit.pipeline.model import sha256_file
from pybiscuit.pipeline.stages import get_executable_stages


def _make_findings(tmp_path):
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    products = {}
    for name in ("t1", "seg_refine", "inner_surface"):
        p = d / f"{name}.dat"
        p.write_text("data")
        products[name] = Product(name, str(p), sha256_file(p), p.stat().st_size, "other", "RAS")
    return InputFindings(
        "synthetic", str(d), "macruise", products, required=tuple(products), frames={"t1": "RAS"}
    )


def test_executable_stages_are_unblocked():
    assert all(s.blocked is None and s.body is not None for s in get_executable_stages())


def test_unusable_inputs_fail_loudly(tmp_path):
    """Stage bodies never write placeholder results: junk inputs stop the run at surfprep."""
    findings = _make_findings(tmp_path)
    cfg = load_config(overrides={"surf_recon": "macruise"})
    out = tmp_path / "out"

    with pytest.raises(ValueError):  # the VTK reader rejects the junk inner surface
        Run(out, findings, cfg, get_executable_stages()).execute()

    rj = json.loads((out / "run.json").read_text())
    assert rj["status"] == "failed"
    assert rj["stages"]["surfprep"]["status"] == "failed"
    assert rj["stages"]["surfprep"]["error"]
    assert not any(rj["stages"].get(s, {}).get("status") == "complete"
                   for s in ("features", "register", "parc", "trace", "quant", "stats", "report"))
    assert not (out / "stages/surfprep/attempt-01/lh.white.vtk").exists()
