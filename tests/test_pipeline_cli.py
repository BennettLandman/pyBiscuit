import json

import pytest

from pybiscuit.cli import main
from test_pipeline_adapters import LEGACY, PYM_RUN, fs_subject, legacy_dir, pym_run  # noqa: F401

pytestmark = pytest.mark.filterwarnings("ignore:.*volume information:UserWarning",
                                        "ignore:Unknown extension code:UserWarning")


def run_cli(*args):
    return main(["run", *map(str, args)])


def test_dry_run_synthetic_pymacruise(pym_run, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "out"
    cmd = ["--dry-run", "--blocked", "--reg", "oasis45", "--trace_r", "5"]
    assert run_cli(pym_run, out, *cmd) == 0
    text = capsys.readouterr().out
    assert "reg_temp = oasis45" in text and "trace_r = 5" in text
    assert "adapter: pymacruise" in text and "surf_recon = macruise" in text
    assert "== stage plan ==" in text and "1. ingest" in text and "will run" in text
    assert text.count("BLOCKED") == 8 and "[B10.3]" in text
    assert not out.exists()  # dry-run writes nothing


def test_dry_run_reports_unusable_input(pym_run, tmp_path, capsys):  # noqa: F811
    d = json.loads((pym_run / "run.json").read_text())
    d["status"] = "running"
    (pym_run / "run.json").write_text(json.dumps(d))
    assert run_cli(pym_run, tmp_path / "o", "--dry-run") == 2
    assert "ERROR: run status is 'running'" in capsys.readouterr().out
    assert run_cli(pym_run, tmp_path / "o", "--dry-run", "--allow-incomplete") == 0


def test_dry_run_freesurfer_and_conf(fs_subject, tmp_path, capsys):  # noqa: F811
    conf = tmp_path / "b.conf"
    conf.write_text("reg_temp = hcp1058\nsci_bins = 9\n")
    assert run_cli(fs_subject, tmp_path / "o", "--dry-run", "--config", conf) == 0
    text = capsys.readouterr().out
    assert "adapter: freesurfer" in text and "surf_recon = freesurfer" in text
    assert "reg_temp = hcp1058" in text and "sci_bins = 9" in text
    assert "fs_surf_model_path = surf" in text and "mc_surf" not in text
    assert "surfprep" not in text


def test_conflicting_surf_recon(pym_run, tmp_path, capsys):  # noqa: F811
    assert run_cli(pym_run, tmp_path / "o", "--dry-run", "--surf_recon", "freesurfer") == 2
    assert "conflicts" in capsys.readouterr().err


def test_real_run_blocks_with_plan_row_then_resume(legacy_dir, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "out"
    assert run_cli(legacy_dir, out, "--blocked") == 3
    err = capsys.readouterr().err
    assert "stage 'surfprep' is blocked: B5" in err
    rj = json.loads((out / "run.json").read_text())
    assert rj["status"] == "blocked" and rj["stages"]["ingest"]["status"] == "complete"
    assert run_cli(legacy_dir, out, "--blocked") == 2  # existing run needs --resume
    assert "Use --resume" in capsys.readouterr().err
    assert run_cli(legacy_dir, out, "--blocked", "--resume") == 3
    assert not (out / "stages/ingest/attempt-02").exists()
    # a changed ingest-relevant setting is refused
    assert run_cli(legacy_dir, out, "--blocked", "--resume", "--surf_recon", "freesurfer") == 2
    assert run_cli(legacy_dir, out, "--blocked", "--resume", "--until", "ingest") == 0
    assert "partial" in capsys.readouterr().out


def test_real_run_executes_to_completion(pym_run, tmp_path):  # noqa: F811
    out = tmp_path / "exec_out"
    assert run_cli(pym_run, out) == 0
    rj = json.loads((out / "run.json").read_text())
    assert rj["status"] == "complete"
    assert "report" in rj["stages"]
    assert rj["stages"]["report"]["status"] == "complete"


def test_unknown_input(tmp_path, capsys):
    assert run_cli(tmp_path, tmp_path / "o") == 2


@pytest.mark.reference
@pytest.mark.parametrize("root", [PYM_RUN, LEGACY])
def test_dry_run_reference_inputs(root, tmp_path, capsys):
    assert run_cli(root, tmp_path / "o", "--dry-run") == 0
    text = capsys.readouterr().out
    assert "== stage plan ==" in text and text.count("BLOCKED") == 8
    assert "ERROR" not in text
    for stage in ("ingest", "surfprep", "features", "register", "parc", "trace", "quant",
                  "stats", "report"):
        assert f" {stage} " in text
    assert not (tmp_path / "o").exists()


@pytest.mark.reference
def test_ingest_stage_on_reference_run(tmp_path):
    out = tmp_path / "o"
    assert run_cli(LEGACY, out, "--until", "ingest") == 0
    rj = json.loads((out / "run.json").read_text())
    assert rj["inputs"]["adapter"] == "macruise-v3" and rj["frames"]["inner_surface"] == "RAS"
    assert rj["run_signature"] and rj["stages"]["ingest"]["outputs"][0]["bytes"] > 0
