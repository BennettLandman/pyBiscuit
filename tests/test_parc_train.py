"""Parcellation-network training (port of SphericalLabeling train.py / loader.py / gen_ico.py).

Synthetic fixture (generator below, seed 0): 10 subjects on the level-2 UGSCNN icosphere
(162 vertices), augmentations ``aug0``/``aug1``, three feature channels (noisy, randomly rotated
coordinates) and labels from z bands (labels 0/5/9, plus label 7 outside ``--classes``).  The
network is tiny (``max_level=2, feat=4``) and runs on CPU in seconds.

Tests marked with the upstream reference compare against ilwoolyu/SphericalLabeling @ 37fa7a0 in
``reference/repos`` and skip when it is absent.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from conftest import REPOS  # noqa: E402
from pybiscuit.parc.gen_ico import export_spheres  # noqa: E402
from pybiscuit.parc.infer import ParcModel  # noqa: E402
from pybiscuit.parc.train import (  # noqa: E402
    BEST_NAME,
    SegDataset,
    TrainConfig,
    build_parser,
    class_scores,
    load_resume,
    split_subjects,
    train,
)
from pybiscuit.parc.unet import OperatorCache, SphericalUNet  # noqa: E402
from pybiscuit.sphere.operators import build_operators  # noqa: E402

UPSTREAM = REPOS / "SphericalLabeling"
LEVEL = 2
NV = 162
CLASSES = [0, 5, 9]
IN_CH = ["curv", "iH", "sulc"]
N_SUBJ = 10


def make_dataset(root: Path, seed: int = 0, deg: int = 1, hemi: str = "lh") -> Path:
    """Write the synthetic training set (features float64, labels int16) under ``root``."""
    rng = np.random.default_rng(seed)
    v = build_operators(LEVEL)["V"]
    (root / "features").mkdir(parents=True)
    (root / "labels").mkdir()
    z = v[:, 2]
    label = np.where(z > 0.35, 5, np.where(z < -0.35, 9, 0)).astype(np.int16)
    label[(np.abs(z) < 0.05) & (v[:, 0] > 0)] = 7  # not in CLASSES -> class 0
    for s in range(N_SUBJ):
        for k in range(deg + 1):
            ang = rng.normal(scale=0.05, size=3)
            c, si = np.cos(ang), np.sin(ang)
            rx = np.array([[1, 0, 0], [0, c[0], -si[0]], [0, si[0], c[0]]])
            ry = np.array([[c[1], 0, si[1]], [0, 1, 0], [-si[1], 0, c[1]]])
            feats = (v @ (rx @ ry).T).T + rng.normal(scale=0.05, size=(3, NV))
            stem = f"subj{s:02d}.{hemi}.aug{k}"
            for name, f in zip(IN_CH, feats, strict=True):
                f.astype(np.float64).tofile(root / "features" / f"{stem}.{name}.dat")
            label.tofile(root / "labels" / f"{stem}.label.dat")
    return root


def config(data: Path, log: Path, **kw) -> TrainConfig:
    base = dict(
        classes=CLASSES, in_ch=IN_CH, batch_size=2, epochs=2, lr=0.01, seed=1,
        data_folder=str(data), log_interval=100, max_level=LEVEL, min_level=0, feat=4,
        log_dir=str(log), deg=1, hemi="lh", drop=[0], no_cuda=True,
    )  # fmt: skip
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def ops():
    return OperatorCache()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("parc-train") / "data")


@pytest.fixture(scope="module")
def run(data, ops, tmp_path_factory):
    log = tmp_path_factory.mktemp("parc-train-log")
    return config(data, log, epochs=6), train(config(data, log, epochs=6), operators=ops)


# --------------------------------------------------------------------------- loader


def test_split_partitions_all_subjects():
    subj = [f"s{i:02d}.lh" for i in range(23)]
    for kcv in (3, 5):
        tests = []
        for fold in range(1, kcv + 1):
            tr, va, te = split_subjects(subj, fold, kcv, seed=1)
            assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
            assert len(tr) + len(va) + len(te) == len(subj)
            tests += te
        assert sorted(tests) == sorted(subj)  # every subject is tested exactly once


def test_dataset_items(data):
    ds = SegDataset(data, "train", 1, LEVEL, CLASSES, IN_CH, seed=1, deg=1, kcv=5, hemi="lh")
    assert len(ds) == 6 * 2  # 6 training subjects x (aug0, aug1)
    x, y = ds[0]
    assert x.dtype == np.float32 and x.shape == (3, NV)
    assert y.dtype == np.int64 and set(np.unique(y)) <= {0, 1, 2}
    raw = np.fromfile(ds.flist[0][-1], dtype=np.int16)
    assert np.array_equal(y == 0, np.isin(raw, [0, 7]))  # unknown label 7 -> class 0
    val = SegDataset(data, "val", 1, LEVEL, CLASSES, IN_CH, seed=1, deg=1, kcv=5, hemi="lh")
    assert all(".aug0." in f.name for files in val.flist for f in files)


def _load_upstream_loader():
    path = UPSTREAM / "loader.py"
    if not path.exists():
        pytest.skip("upstream SphericalLabeling not in reference/repos")
    spec = importlib.util.spec_from_file_location("_upstream_sl_loader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fold", [1, 4, 5])
@pytest.mark.parametrize("partition", ["train", "val", "test"])
def test_loader_matches_upstream(data, partition, fold, monkeypatch):
    if "." in str(data):  # upstream parses the whole path on '.'
        pytest.skip("temporary path contains '.'")
    up = _load_upstream_loader()
    monkeypatch.setattr(np, "int", int, raising=False)  # removed in NumPy 1.24
    kw = dict(fold=fold, sp_level=LEVEL, classes=CLASSES, seed=1, deg=1, kcv=5, hemi="lh")
    ref = up.S2D3DSegLoader(str(data), partition, in_ch=[*IN_CH, "label"], **kw)
    ours = SegDataset(data, partition, in_ch=IN_CH, **kw)
    assert [[Path(f) for f in fl] for fl in ref.flist] == ours.flist
    for i in range(len(ours)):
        rx, ry = ref[i]
        ox, oy = ours[i]
        assert np.array_equal(rx, ox) and rx.dtype == ox.dtype
        assert np.array_equal(ry, oy)


# --------------------------------------------------------------------------- metrics


def _upstream_metric_functions():
    path = UPSTREAM / "train.py"
    if not path.exists():
        pytest.skip("upstream SphericalLabeling not in reference/repos")
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("iou_score", "accuracy", "dice")]  # fmt: skip
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
    return ns


def test_metrics_match_upstream():
    ns = _upstream_metric_functions()
    g = torch.Generator().manual_seed(3)
    pred = torch.randint(0, 6, (4, 500), generator=g)
    true = torch.randint(0, 6, (4, 500), generator=g)
    true[true == 4] = 3  # class 4 absent from the truth
    args = type("A", (), {"classes": list(range(6)), "drop": [0, 2]})()
    ours = class_scores(pred, true, 6, args.drop)
    i, u = ns["iou_score"](args, pred, true)
    t, c = ns["accuracy"](args, pred, true)
    d, dd = ns["dice"](args, pred, true)
    assert np.array_equal(ours["intersect"], i) and np.array_equal(ours["union"], u)
    assert np.array_equal(ours["tpos"], t) and np.array_equal(ours["true_count"], c)
    assert np.array_equal(ours["tpos"], d) and np.array_equal(ours["dice_denom"], dd)


def test_parser_flags_match_upstream():
    path = UPSTREAM / "train.py"
    if not path.exists():
        pytest.skip("upstream SphericalLabeling not in reference/repos")
    upstream = set(re.findall(r"add_argument\('(--[\w-]+)'", path.read_text()))
    ours = {s for a in build_parser()._actions for s in a.option_strings} - {"-h", "--help"}
    assert ours == upstream


# --------------------------------------------------------------------------- training


def test_training_runs_and_loss_decreases(run):
    cfg, res = run
    losses = [h["train_loss"] for h in res["history"]]
    assert [h["epoch"] for h in res["history"]] == list(range(1, 7))
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert np.mean(losses[-2:]) < np.mean(losses[:2])
    assert res["best"] is not None and res["best_dice"] > 0
    log = (Path(cfg.log_dir) / "train.log").read_text()
    assert "paramerters in total" in log and "[Epoch 1 val stats]: MIoU:" in log
    assert (Path(cfg.log_dir) / f"{BEST_NAME}.pth").exists()


def test_training_is_deterministic(data, ops, tmp_path):
    a = train(config(data, tmp_path / "a"), operators=ops)
    b = train(config(data, tmp_path / "b"), operators=ops)
    assert [h["train_loss"] for h in a["history"]] == [h["train_loss"] for h in b["history"]]
    assert [h["val"] for h in a["history"]] == [h["val"] for h in b["history"]]
    sa, sb = a["model"].state_dict(), b["model"].state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_checkpoint_roundtrips_through_inference(run, data, ops):
    cfg, res = run
    pm = ParcModel.from_safetensors(res["best"], operators=ops)
    assert pm.meta["arch"] == {"in_ch": 3, "out_ch": 3, "fdim": 4, "max_level": LEVEL,
                               "min_level": 0}  # fmt: skip
    assert pm.meta["classes"] == CLASSES
    assert pm.meta["dice"] == pytest.approx(res["best_dice"])
    # the same weights through the upstream-style .pth
    ck = load_resume(Path(cfg.log_dir) / f"{BEST_NAME}.pth")
    net = SphericalUNet(in_ch=3, out_ch=3, max_level=LEVEL, fdim=4, operators=ops)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    ds = SegDataset(data, "test", 1, LEVEL, CLASSES, IN_CH, seed=1, deg=1, kcv=5, hemi="lh")
    x, y = ds[0]
    with torch.no_grad():
        ref = net(torch.from_numpy(x)[None])[0].numpy()
    assert np.array_equal(pm.predict(x), ref)
    # and the trained net actually labels the held-out subject better than chance
    assert (pm.predict(x).argmax(0) == y).mean() > 0.5
    if res["history"][-1]["is_best"]:
        with torch.no_grad():
            res["model"].eval()
            assert np.array_equal(res["model"](torch.from_numpy(x)[None])[0].numpy(), ref)


def test_resume_continues_epochs(data, ops, tmp_path):
    first = train(config(data, tmp_path, epochs=2), operators=ops)
    assert first["best"] is not None
    ck = load_resume(tmp_path / f"{BEST_NAME}.pth")
    more = train(
        config(data, tmp_path / "r", epochs=ck["epoch"] + 1,
               resume=str(tmp_path / f"{BEST_NAME}.pth")),
        operators=ops,
    )  # fmt: skip
    assert [h["epoch"] for h in more["history"]] == [ck["epoch"] + 1]


def test_decay_schedule(data, ops, tmp_path):
    res = train(config(data, tmp_path, epochs=6, decay=True, optim="sgd"), operators=ops)
    lrs = [h["lr"] for h in res["history"]]
    assert lrs == pytest.approx([0.01] * 4 + [0.009] * 2)  # StepLR(5, 0.9).step(epoch)


def test_cli(data, tmp_path, capsys):
    from pybiscuit.cli import main

    rc = main([
        "parc-train", "--data_folder", str(data), "--log_dir", str(tmp_path),
        "--max_level", str(LEVEL), "--feat", "4", "--batch-size", "4", "--epochs", "1",
        "--in_ch", *IN_CH, "--classes", *map(str, CLASSES), "--deg", "1", "--no-cuda",
    ])  # fmt: skip
    assert rc == 0
    assert "best val Dice" in capsys.readouterr().out
    assert "Train Epoch: 1" in (tmp_path / "train.log").read_text()


# --------------------------------------------------------------------------- gen_ico


def test_gen_ico_writes_operator_vertices(tmp_path):
    from pybiscuit.io import read_vtk

    paths = export_spheres(range(3), tmp_path)
    assert [p.name for p in paths] == [f"icosphere_{i}.vtk" for i in range(3)]
    for lvl, p in enumerate(paths):
        ref = build_operators(lvl)
        text = p.read_text().splitlines()
        assert text[4] == f"POINTS {len(ref['V'])} float"
        pts = np.array([list(map(float, r.split())) for r in text[5 : 5 + len(ref["V"])]])
        assert np.array_equal(pts, ref["V"])  # str(float) round-trips float64 exactly
        m = read_vtk(p)
        assert np.array_equal(m.faces, ref["F"])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
