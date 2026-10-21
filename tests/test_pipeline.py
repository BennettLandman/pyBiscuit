import json
from dataclasses import replace

import pytest

from pybiscuit import assets
from pybiscuit.config import load_config
from pybiscuit.pipeline import (
    STAGES,
    InputError,
    InputFindings,
    Product,
    ResumeError,
    Run,
    Stage,
    StageBlocked,
    format_plan,
)
from pybiscuit.pipeline.model import sha256_file


def make_findings(tmp_path, **contents):
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    products = {}
    for name, text in (contents or {"t1": "t1-data", "inner_surface": "surf"}).items():
        p = d / f"{name}.dat"
        p.write_text(text)
        products[name] = Product(name, str(p), sha256_file(p), p.stat().st_size, "other", "RAS")
    return InputFindings(
        "synthetic", str(d), "macruise", products, required=tuple(products), frames={"t1": "RAS"}
    )


CALLS: list[str] = []


def _body(text):
    def body(ctx):
        CALLS.append(ctx.stage.name)
        for o in ctx.stage.outputs:
            params = sorted(ctx.params.items())
            (ctx.attempt_dir / o).write_text(f"{ctx.stage.name}:{o}:{text}:{params}")

    return body


def graph(text="v1", **kw):
    return [
        Stage("a", ("T",), inputs=("t1",), outputs=("a.txt",), body=_body(text)),
        Stage("b", ("T",), inputs=("a:a.txt", "inner_surface"), params=("trace_r",),
              outputs=("b.txt",), assets=("reg_template",), body=_body(text), **kw),
        Stage("c", ("T",), inputs=("b:b.txt",), outputs=("c.txt", "d.txt"), body=_body(text)),
    ]


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()


def sigs(findings, config, stages=None):
    return {p.stage.name: p.signature
            for p in Run("/nonexistent-out", findings, config, stages or graph()).plan()}


def test_signature_sensitivity(tmp_path, monkeypatch):
    f, cfg = make_findings(tmp_path), load_config()
    base = sigs(f, cfg)
    # input change: only consumers of that input (and downstream) change
    f2 = make_findings(tmp_path, t1="CHANGED", inner_surface="surf")
    s = sigs(f2, cfg)
    assert all(s[k] != base[k] for k in "abc")
    f3 = make_findings(tmp_path, t1="t1-data", inner_surface="OTHER")
    s = sigs(f3, cfg)
    assert s["a"] == base["a"] and s["b"] != base["b"] and s["c"] != base["c"]
    # param change touches b and downstream, not a
    s = sigs(f, cfg.replace(trace_r=5))
    assert s["a"] == base["a"] and s["b"] != base["b"] and s["c"] != base["c"]
    # undeclared param (threads) changes nothing
    assert sigs(f, cfg.replace(omp_num_threads=8)) == base
    # asset selection change (different template)
    s = sigs(f, cfg.replace(reg_temp="oasis45"))
    assert s["a"] == base["a"] and s["b"] != base["b"]
    # asset content change (registry hash)
    reg = assets.load_registry()
    key = next(k for k in reg if k.startswith("reg/mb101/"))
    patched = dict(reg)
    patched[key] = replace(reg[key], sha256="0" * 64)
    monkeypatch.setattr(assets, "load_registry", lambda: patched)
    s = sigs(f, cfg)
    assert s["a"] == base["a"] and s["b"] != base["b"]
    monkeypatch.undo()
    # stage code version
    s = sigs(f, cfg, graph(version="2"))
    assert s["a"] == base["a"] and s["b"] != base["b"]
    # deterministic
    assert sigs(f, cfg) == base


def test_full_run_writes_manifest(tmp_path):
    f, cfg = make_findings(tmp_path), load_config(overrides={"surf_recon": "macruise"})
    out = tmp_path / "out"
    state = Run(out, f, cfg, graph()).execute()
    assert CALLS == ["a", "b", "c"]
    rj = json.loads((out / "run.json").read_text())
    assert rj["schema_version"] == 1 and rj["status"] == "complete"
    assert rj["spatial_convention"] == "RAS" and rj["frames"] == {"t1": "RAS"}
    assert set(rj["inputs"]["products"]) == {"t1", "inner_surface"}
    assert rj["config"]["values"]["trace_r"] == 4 and "mc_surf" in rj["config"]["values"]
    assert "b:reg_template" in rj["assets"] and rj["run_signature"]
    assert [h["stage"] for h in rj["history"]] == ["a", "b", "c"]
    assert {o["path"] for o in rj["final_outputs"]} == {
        "stages/c/attempt-01/c.txt", "stages/c/attempt-01/d.txt"}
    for o in rj["final_outputs"]:
        assert sha256_file(out / o["path"]) == o["sha256"]
    prov = json.loads((out / "stages/b/attempt-01/provenance.json").read_text())
    assert prov["dependency_signature"] == rj["stages"]["b"]["dependency_signature"]
    assert state["status"] == "complete"
    assert not list(out.glob("run.json.tmp"))


def test_resume_skips_valid_stages(tmp_path):
    f, cfg = make_findings(tmp_path), load_config()
    out = tmp_path / "out"
    Run(out, f, cfg, graph()).execute(until="b")
    assert CALLS == ["a", "b"]
    assert json.loads((out / "run.json").read_text())["status"] == "partial"
    CALLS.clear()
    Run(out, f, cfg, graph(), resume=True).execute()
    assert CALLS == ["c"]
    CALLS.clear()
    Run(out, f, cfg, graph(), resume=True).execute()  # everything valid: nothing runs
    assert CALLS == []
    assert not (out / "stages/a/attempt-02").exists()
    plans = Run(out, f, cfg, graph(), resume=True).plan()
    assert [p.state for p in plans] == ["valid"] * 3


def test_existing_run_requires_resume(tmp_path):
    f, cfg = make_findings(tmp_path), load_config()
    out = tmp_path / "out"
    Run(out, f, cfg, graph()).execute()
    with pytest.raises(ResumeError, match="Use --resume"):
        Run(out, f, cfg, graph()).execute()
    with pytest.raises(ResumeError, match="no .*run.json"):
        Run(tmp_path / "empty", f, cfg, graph(), resume=True).execute()


def test_resume_refuses_changed_param(tmp_path):
    f, cfg = make_findings(tmp_path), load_config()
    out = tmp_path / "out"
    Run(out, f, cfg, graph()).execute()
    with pytest.raises(ResumeError) as e:
        Run(out, f, cfg.replace(trace_r=9), graph(), resume=True).execute()
    msg = str(e.value)
    assert "stage b" in msg and "param trace_r: 4 -> 9" in msg and "stage c" in msg
    assert "stage a" not in msg
    assert CALLS == ["a", "b", "c"]  # nothing re-ran


def test_resume_refuses_changed_input_and_asset_and_code(tmp_path):
    f, cfg = make_findings(tmp_path), load_config()
    out = tmp_path / "out"
    Run(out, f, cfg, graph()).execute()
    f2 = make_findings(tmp_path, t1="t1-data", inner_surface="CHANGED")
    with pytest.raises(ResumeError, match=r"inner_surface: .* -> "):
        Run(out, f2, cfg, graph(), resume=True).execute()
    with pytest.raises(ResumeError, match="asset reg_template"):
        Run(out, f, cfg.replace(reg_temp="oasis45"), graph(), resume=True).execute()
    with pytest.raises(ResumeError, match="stage_version: 1 -> 2"):
        Run(out, f, cfg, graph(version="2"), resume=True).execute()


def test_resume_detects_corrupted_and_missing_outputs(tmp_path):
    f, cfg = make_findings(tmp_path), load_config()
    out = tmp_path / "out"
    Run(out, f, cfg, graph()).execute()
    (out / "stages/b/attempt-01/b.txt").write_text("tampered")
    (out / "stages/c/attempt-01/d.txt").unlink()
    with pytest.raises(ResumeError) as e:
        Run(out, f, cfg, graph(), resume=True).execute()
    msg = str(e.value)
    assert "stage b" in msg and "stages/b/attempt-01/b.txt" in msg and "changed" in msg
    assert "output missing: stages/c/attempt-01/d.txt" in msg


def test_attempt_numbering_and_failure_recovery(tmp_path):
    f, cfg = make_findings(tmp_path), load_config()
    out = tmp_path / "out"
    state = {"fail": True}

    def flaky(ctx):
        if state["fail"]:
            raise RuntimeError("boom")
        (ctx.attempt_dir / "b.txt").write_text("ok")

    stages = graph()
    stages[1] = Stage("b", ("T",), inputs=("a:a.txt", "inner_surface"), params=("trace_r",),
                      outputs=("b.txt",), assets=("reg_template",), body=flaky)
    with pytest.raises(RuntimeError, match="boom"):
        Run(out, f, cfg, stages).execute()
    rj = json.loads((out / "run.json").read_text())
    assert rj["status"] == "failed" and "boom" in rj["error"]
    assert rj["stages"]["b"]["status"] == "failed"
    prov = json.loads((out / "stages/b/attempt-01/provenance.json").read_text())
    assert prov["status"] == "failed"
    state["fail"] = False
    Run(out, f, cfg, stages, resume=True).execute()
    assert (out / "stages/b/attempt-02/b.txt").is_file()
    assert (out / "stages/a/attempt-01").is_dir() and not (out / "stages/a/attempt-02").exists()
    rj = json.loads((out / "run.json").read_text())
    assert rj["stages"]["b"]["attempt"] == 2 and rj["status"] == "complete"
    assert [(h["stage"], h["attempt"], h["status"]) for h in rj["history"]] == [
        ("a", 1, "complete"), ("b", 1, "failed"), ("b", 2, "complete"), ("c", 1, "complete")]


def test_missing_declared_output_is_an_error(tmp_path):
    f, cfg = make_findings(tmp_path), load_config()
    stages = [Stage("a", ("T",), inputs=("t1",), outputs=("a.txt",), body=lambda ctx: None)]
    with pytest.raises(Exception, match="did not write declared outputs"):
        Run(tmp_path / "out", f, cfg, stages).execute()


def test_incomplete_findings_refused(tmp_path):
    f = make_findings(tmp_path)
    f.required = ("t1", "seg_refine")
    with pytest.raises(InputError, match="seg_refine"):
        Run(tmp_path / "out", f, load_config(), graph()).execute()


def test_blocked_stage_cites_plan_row(tmp_path):
    f, cfg = make_findings(tmp_path), load_config()
    out = tmp_path / "out"
    with pytest.raises(StageBlocked, match=r"surfprep.*B5") as e:
        Run(out, f, cfg, STAGES).execute()
    assert "brief 11" in str(e.value)
    rj = json.loads((out / "run.json").read_text())
    assert rj["status"] == "blocked"
    assert rj["stages"]["ingest"]["status"] == "complete"
    assert rj["stages"]["surfprep"]["status"] == "blocked"
    assert rj["stages"]["surfprep"]["plan_rows"][0] == "B5.1"
    inputs = json.loads((out / "stages/ingest/attempt-01/inputs.json").read_text())
    assert inputs["adapter"] == "synthetic"
    # resume re-checks ingest (valid) and reports the same block
    with pytest.raises(StageBlocked, match="surfprep"):
        Run(out, f, cfg, STAGES, resume=True).execute()
    assert not (out / "stages/ingest/attempt-02").exists()


def test_every_blocked_stage_names_a_plan_row():
    for st in STAGES:
        if st.blocked:
            assert st.blocked.split(":")[0].split("/")[0].startswith("B"), st.name
            assert st.plan_rows


def test_plan_formatting_marks_blocked(tmp_path):
    f, cfg = make_findings(tmp_path), load_config()
    text = format_plan(Run(tmp_path / "o", f, cfg, STAGES).plan())
    assert "ingest" in text and "will run" in text
    assert text.count("BLOCKED") == len(STAGES) - 1
    assert "reg_template=" in text and "reg_temp=mb101" in text


def test_freesurfer_route_has_no_surfprep(tmp_path):
    f = make_findings(tmp_path)
    f.surf_recon = "freesurfer"
    names = [p.stage.name for p in Run(tmp_path / "o", f, load_config(), STAGES).plan()]
    assert "surfprep" not in names and names[0] == "ingest" and "register" in names
