"""Spherical U-Net, checkpoint conversion and resampling (needs the ``[parc]`` extra).

Committed golden: ``tests/data/golden/parc-unet/tiny.npz`` -- a small random-weight net
(``max_level=3, fdim=8``) evaluated by the image's ORIGINAL ``model.py``/``meshcnn`` code under
current torch (``scripts/oracles/make_golden_parc.py``).  These weights are test data, not a
BISCUIT model.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from pybiscuit.parc.checkpoint import (  # noqa: E402
    UnsafeCheckpointError,
    canonicalize_state_dict,
    convert_checkpoint,
    load_safetensors,
)
from pybiscuit.parc.infer import ParcModel, text_roundtrip  # noqa: E402
from pybiscuit.parc.unet import SphericalUNet, infer_architecture  # noqa: E402

GOLDEN = Path(__file__).parent / "data/golden/parc-unet/tiny.npz"


@pytest.fixture(scope="module")
def tiny():
    z = np.load(GOLDEN, allow_pickle=False)
    keys = ("in_ch", "out_ch", "max_level", "min_level", "fdim")
    arch = dict(zip(keys, map(int, z["arch"]), strict=True))
    sd = {k[3:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("sd/")}
    return arch, sd, z["x"], z["y"]


def test_canonicalize_merges_aliases_and_drops_derived(tiny):
    arch, sd, _, _ = tiny
    canon, rep = canonicalize_state_dict(sd)
    assert rep["aliases_merged"] > 0 and rep["ns_ew_max_abs_diff"] < 1e-6
    assert not any(k.endswith((".NS", ".EW")) or "seq" in k for k in canon)
    assert infer_architecture(canon) == {**arch, "min_level": 0}
    # same ResBlock layers under their canonical names
    assert {"down.0.conv.conv1.weight", "down.0.conv.bn_.running_var", "up.0.conv.bn3.bias"} <= set(
        canon
    )


def test_canonicalize_rejects_diverging_alias_and_bad_operator(tiny):
    _, sd, _, _ = tiny
    bad = dict(sd)
    bad["module.down.0.conv.seq1.0.weight"] = bad["module.down.0.conv.seq1.0.weight"] + 1
    with pytest.raises(UnsafeCheckpointError, match="alias"):
        canonicalize_state_dict(bad)
    bad = dict(sd)
    k = next(k for k in sd if k.endswith("in_conv.NS"))
    bad[k] = bad[k] + 1e-3
    with pytest.raises(UnsafeCheckpointError, match="regenerated operator"):
        canonicalize_state_dict(bad)


def test_network_matches_original_code(tiny):
    arch, sd, x, y = tiny
    canon, _ = canonicalize_state_dict(sd)
    net = SphericalUNet(**arch).eval()
    net.load_state_dict(canon, strict=False)
    with torch.no_grad():
        out = net(torch.from_numpy(x)[None])[0].numpy()
    assert out.shape == y.shape == (arch["out_ch"], 10 * 4 ** arch["max_level"] + 2)
    assert np.abs(out - y).max() < 1e-5  # bit-identical on the machine that made the golden


def test_legacy_pth_to_safetensors_roundtrip(tiny, tmp_path):
    """Write a legacy (non-zip) torch pickle like the shipped ones, convert, run."""
    arch, sd, x, y = tiny
    src = tmp_path / "toy" / "lh.pth"
    src.parent.mkdir()
    torch.save(
        {"state_dict": {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}},
        src,
        _use_new_zipfile_serialization=False,
    )
    assert src.read_bytes()[:4] == b"\x80\x02\x8a\x0a"
    dst1, dst2 = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    e1 = convert_checkpoint(src, dst1)
    e2 = convert_checkpoint(src, dst2)
    assert e1["sha256"] == e2["sha256"], "conversion must be deterministic"
    assert e1["arch"] == {**arch, "min_level": 0}
    _, meta = load_safetensors(dst1, expected_sha256=e1["sha256"])
    assert meta["source_sha256"] == e1["source_sha256"]
    model = ParcModel.from_safetensors(dst1, expected_sha256=e1["sha256"])
    out = model.predict(x)
    assert np.abs(out - y).max() < 1e-5
    # hash pinning
    with pytest.raises(ValueError, match="sha256"):
        load_safetensors(dst1, expected_sha256="0" * 64)
    with pytest.raises(UnsafeCheckpointError, match="sha256"):
        convert_checkpoint(src, tmp_path / "c.safetensors", expected_sha256="0" * 64)


def test_predict_truncates_finer_features_and_validates(tiny, tmp_path):
    arch, sd, x, y = tiny
    canon, _ = canonicalize_state_dict(sd)
    net = SphericalUNet(**arch).eval()
    net.load_state_dict(canon, strict=False)
    model = ParcModel(net)
    # a finer-level feature file is truncated to the first nv vertices (loader_individual.py)
    fine = np.concatenate([x, np.ones((4, x.shape[1]), dtype=x.dtype)], axis=0)  # extra channels
    wide = np.concatenate([x, np.full((3, 50), 7.0)], axis=1)
    np.testing.assert_array_equal(model.predict(wide), model.predict(x))
    np.testing.assert_array_equal(model.predict(fine), model.predict(x))
    with pytest.raises(ValueError, match="need at least"):
        model.predict(x[:, :-1])
    with pytest.raises(ValueError, match="need at least"):
        model.predict(x[:2])


def test_strict_loading_flags_missing_weights(tiny):
    from pybiscuit.parc.infer import _load_strict

    arch, sd, _, _ = tiny
    canon, _ = canonicalize_state_dict(sd)
    canon.pop("up.0.conv.bn3.bias")
    with pytest.raises(ValueError, match="mismatch"):
        _load_strict(SphericalUNet(**arch), canon)


def test_text_roundtrip_is_six_decimals():
    v = np.array([0.1234567, -2.3750005, 1e-7, 123.4567891], dtype=np.float32)
    t = text_roundtrip(v)
    assert t.dtype == np.float32
    np.testing.assert_allclose(t, np.round(v.astype(np.float64), 6), atol=3e-6, rtol=0)
    assert t[2] == 0.0
