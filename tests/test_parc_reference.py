"""Acceptance of brief 06 against the image's checkpoints and the original code (oracle).

Needs ``reference/`` (the 12 legacy ``.pth``) and the oracle golden made by
``scripts/oracles/run_parc_oracle.py`` (``~/dev/biscuit/oracles/parc-unet/golden``); skipped when
either is absent.  Weights are converted into ``tmp_path`` for the test; nothing is written to git.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from conftest import OPT, PARC_MESH_FILES  # noqa: E402
from pybiscuit.assets import load_registry  # noqa: E402
from pybiscuit.parc.checkpoint import convert_checkpoint, converted_registry  # noqa: E402
from pybiscuit.parc.infer import ParcModel, resample_to_native  # noqa: E402
from pybiscuit.sphere.operators import safe_load_pickle  # noqa: E402

GOLDEN = Path.home() / "dev/biscuit/oracles/parc-unet/golden"
PARC = OPT / "parc"
CKPTS = sorted(PARC.glob("*/?h*.pth")) if PARC.exists() else []
pytestmark = [
    pytest.mark.reference,
    pytest.mark.skipif(not GOLDEN.exists(), reason="parc oracle golden not generated"),
]
IDS = [f"{c.parent.name}/{c.stem}" for c in CKPTS]


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("converted")
    out = {}
    for c in CKPTS:
        out[c] = convert_checkpoint(c, root / c.parent.name / f"{c.stem}.safetensors")
        out[c]["file"] = root / c.parent.name / f"{c.stem}.safetensors"
    return out


def test_twelve_checkpoints(converted):
    assert len(CKPTS) == 12


def test_conversion_matches_committed_registry(converted):
    """Hashes of the converted files (deterministic) and of their sources are pinned."""
    reg = converted_registry()
    src_reg = load_registry()
    assert len(reg) == 12
    for e in converted.values():
        rec = reg[e["path"]]
        assert rec["sha256"] == e["sha256"] and rec["size"] == e["size"]
        assert rec["source_sha256"] == e["source_sha256"] == src_reg[e["source"]].sha256
        assert e["ns_ew_max_abs_diff"] < 1e-6
        assert e["globals_audited"] == {
            "collections.OrderedDict": 1,
            "torch._utils._rebuild_tensor_v2": 1,
            "torch.FloatStorage": 1,
        }


def _cases(ckpt):
    base = GOLDEN / ckpt.parent.name / ckpt.stem
    return sorted(p for p in base.iterdir() if (p / "raw.npy").exists())


@pytest.mark.parametrize("ckpt", CKPTS, ids=IDS)
def test_network_vs_original_code(ckpt, converted):
    """Acceptance: per-class max abs diff <= 1e-4, argmax agreement >= 99.9 %."""
    model = ParcModel.from_safetensors(converted[ckpt]["file"])
    cases = _cases(ckpt)
    assert {c.name for c in cases} == {"random", f"template-{ckpt.stem[:2]}"}
    meta = json.loads((cases[0] / "meta.json").read_text())
    assert meta["ckpt_sha256"] == converted[ckpt]["source_sha256"]
    for c in cases:
        feats, ref = np.load(c / "features.npy"), np.load(c / "raw.npy")
        out = model.predict(feats)
        assert out.shape == ref.shape
        assert np.abs(out - ref).max() <= 1e-4
        assert (out.argmax(0) == ref.argmax(0)).mean() >= 0.999


@pytest.mark.parametrize("ckpt", CKPTS[:2] + CKPTS[2:3], ids=IDS[:3])
def test_operator_source_makes_no_difference(ckpt, converted):
    """Regenerated vs shipped-pickle operators (safe unpickler): outputs agree to 1e-6."""
    f = converted[ckpt]["file"]
    regen = ParcModel.from_safetensors(f)
    shipped = ParcModel.from_safetensors(
        f, operators=lambda lv: safe_load_pickle(PARC_MESH_FILES / f"icosphere_{lv}.pkl")
    )
    for c in _cases(ckpt):
        feats = np.load(c / "features.npy")
        assert np.abs(regen.predict(feats) - shipped.predict(feats)).max() <= 1e-6


@pytest.mark.parametrize("ckpt", [CKPTS[0], CKPTS[2], CKPTS[-1]], ids=[IDS[0], IDS[2], IDS[-1]])
def test_resampling_vs_original_pipeline(ckpt, converted):
    """test.py's text round trip + SphericalRemesh (nofma oracle binary) vs our resampling."""
    model = ParcModel.from_safetensors(converted[ckpt]["file"])
    for c in _cases(ckpt):
        nat = np.load(c / "native.npy")
        out = resample_to_native(
            model.predict(np.load(c / "features.npy")),
            PARC_MESH_FILES / "icosphere_5.vtk",
            PARC_MESH_FILES / "icosphere_6.vtk",
        )
        assert out.shape == nat.shape
        assert np.abs(out - nat).max() <= 1e-4
        assert (out.argmax(0) == nat.argmax(0)).mean() >= 0.999


def test_inference_time_report(converted, capsys):
    """Not an assertion of speed: prints the CPU time per hemisphere for the report."""
    model = ParcModel.from_safetensors(converted[CKPTS[0]]["file"])
    f = np.load(_cases(CKPTS[0])[0] / "features.npy")
    model.predict(f)  # warm up
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        model.predict(f)
        t.append(time.perf_counter() - t0)
    with capsys.disabled():
        print(f"\nparc inference ({torch.get_num_threads()} threads): {min(t):.2f}-{max(t):.2f} s")
    assert max(t) < 60
