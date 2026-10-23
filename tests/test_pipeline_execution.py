"""Tests for executable pipeline stages (plan B10.2)."""

import json

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


def test_executable_stages_run_to_completion(tmp_path):
    findings = _make_findings(tmp_path)
    cfg = load_config(overrides={"surf_recon": "macruise"})
    out = tmp_path / "out"

    stages = get_executable_stages()
    # Confirm none of the executable stages are marked blocked
    assert all(s.blocked is None for s in stages)

    state = Run(out, findings, cfg, stages).execute()
    assert state["status"] == "complete"

    rj = json.loads((out / "run.json").read_text())
    assert rj["status"] == "complete"
    assert "report" in rj["stages"]
    assert rj["stages"]["report"]["status"] == "complete"
    assert (out / "stages/report/attempt-01/biscuit.pdf").is_file()
