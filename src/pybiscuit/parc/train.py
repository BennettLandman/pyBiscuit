"""Training of the spherical U-Net: port of SphericalLabeling ``train.py`` + ``loader.py``.

Source: ilwoolyu/SphericalLabeling @ 37fa7a0 (MIT): ``train.py`` 1-300, ``loader.py`` 1-131 and the
defaults of ``train.sh``.  The network, operators and checkpoint format are the existing
:mod:`pybiscuit.parc.unet` / :mod:`pybiscuit.parc.checkpoint`; nothing is duplicated here.  See
``docs/ports/parcellation-training.md`` for the line map and the deviations.

Data layout (``loader.py``; produced by ``SphericalRemesh --deg0 0 --deg D --ftype double`` for
features and ``--ftype int16 --nneighbor`` for labels, see :mod:`pybiscuit.sphere.remesh`)::

    <data_folder>/features/<subj>.<hemi>.aug<k>.<feature>.dat   float64, >= nv values
    <data_folder>/labels/<subj>.<hemi>.aug<k>.label.dat         int16,   >= nv values

Outputs in ``log_dir``: ``train.log`` (the upstream log lines), the upstream best checkpoint
``checkpoint_latest.pth_SUNet_best.pth`` (``epoch``, ``state_dict``, ``dice``, ``optimizer``;
torch zip format, read back only with ``weights_only=True``) and next to it
``checkpoint_latest.pth_SUNet_best.safetensors``, a ``pybiscuit-parc-unet/1`` file that
:meth:`pybiscuit.parc.infer.ParcModel.from_safetensors` loads directly.
"""

from __future__ import annotations

import argparse
import collections
import logging
import random
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "pybiscuit.parc.train needs PyTorch: install it with `pip install pybiscuit[parc]`"
    ) from exc

from .checkpoint import LEGACY_MAGIC, canonicalize_state_dict, save_safetensors
from .unet import OperatorCache, SphericalUNet

__all__ = [
    "BEST_NAME",
    "NV_SPHERE",
    "SegDataset",
    "TrainConfig",
    "build_parser",
    "class_scores",
    "evaluate",
    "load_resume",
    "main",
    "split_subjects",
    "train",
    "train_epoch",
]

#: sphere mesh size at different levels (``loader.nv_sphere``)
NV_SPHERE = (12, 42, 162, 642, 2562, 10242, 40962, 163842)

#: ``save_checkpoint(..., output_folder=<log_dir>/checkpoint_latest.pth, filename="_SUNet")``
#: concatenates the two strings, so the only file upstream ever writes is this one.
BEST_NAME = "checkpoint_latest.pth_SUNet_best"

LOG = logging.getLogger("pybiscuit.parc.train")


# --------------------------------------------------------------------------- data (loader.py)


def split_subjects(
    subjects: Sequence[str], fold: int, kcv: int, seed: int
) -> tuple[list[str], list[str], list[str]]:
    """``loader.py`` 49-76: seeded shuffle, rotate by fold, carve test/val off the end.

    ``random.seed(seed); random.shuffle(subj)`` is reproduced with a private
    :class:`random.Random` (same Mersenne Twister sequence, no global state touched).
    """
    subj = sorted(subjects)
    random.Random(seed).shuffle(subj)
    total = len(subj)
    fold_batch = int(total / kcv)
    fold_batch_im = total - fold_batch * (kcv - 1)
    dq = collections.deque(subj)
    dq.rotate(fold_batch * (fold - 1))
    subj = list(dq)
    if fold >= kcv - 1:
        test = subj[total - fold_batch - fold_batch_im : total]
    else:
        test = subj[total - 2 * fold_batch : total]
    test_set = set(test)
    train = [s for s in subj if s not in test_set]
    if fold == kcv - 1:
        val = test[0:fold_batch_im]
        test = test[fold_batch_im : fold_batch_im + fold_batch]
    elif fold == kcv:
        val = test[0:fold_batch]
        test = test[fold_batch : fold_batch + fold_batch_im]
    else:
        val = test[0:fold_batch]
        test = test[fold_batch : 2 * fold_batch]
    return train, val, test


class SegDataset(Dataset):
    """``loader.S2D3DSegLoader``.

    ``in_ch`` lists the input features **without** ``'label'`` (upstream ``train.py`` appends it
    before constructing the loader; it is appended here).  Items are ``(float32 (C, nv),
    int64 (nv,))``; labels are mapped to class indices through ``classes`` and every label not in
    ``classes`` becomes class 0 (``defaultdict(lambda: 0)``).
    """

    def __init__(
        self,
        data_dir: str | Path,
        partition: str,
        fold: int,
        sp_level: int,
        classes: Sequence[int],
        in_ch: Sequence[str],
        seed: int,
        deg: int,
        kcv: int,
        hemi: str,
    ):
        if partition not in ("train", "test", "val"):
            raise ValueError(f"partition {partition!r} not in train/test/val")
        channels = [*in_ch, "label"]
        self.in_ch = len(channels) - 1
        self.nv = NV_SPHERE[sp_level]
        self.partition = partition
        data_dir = Path(data_dir)

        # file name <subj>.<hemi>.aug<k>.<feature>.dat: subject key "<subj>.<hemi>", group
        # "aug" + digits of the third field, feature = second-to-last field
        groups: dict[str, dict[str, list[Path]]] = {}
        for sub in ("features", "labels"):
            for p in sorted((data_dir / sub).glob(f"*.{hemi}.*.dat")):
                parts = p.name.split(".")
                key = ".".join(parts[0:2])
                aug = "aug" + re.sub("[^0-9]", "", parts[2])
                groups.setdefault(key, {}).setdefault(aug, []).append(p)
        data: dict[str, dict[str, list[Path]]] = {}
        for key, augs in groups.items():
            data[key] = {}
            for aug, files in augs.items():
                names = [f.name.split(".")[-2] for f in files]
                try:
                    data[key][aug] = [files[names.index(a)] for a in channels]
                except ValueError as exc:
                    raise ValueError(f"{key} {aug}: missing channel ({exc})") from None

        train, val, test = split_subjects(list(data), fold, kcv, seed)
        try:
            if partition == "train":
                self.flist = [data[s][f"aug{a}"] for s in train for a in range(deg + 1)]
            else:
                self.flist = [data[s]["aug0"] for s in (val if partition == "val" else test)]
        except KeyError as exc:
            raise ValueError(f"augmentation {exc} missing for a {partition} subject") from None
        self.subjects = {"train": train, "val": val, "test": test}[partition]

        self.classes = [int(c) for c in classes]

    def __len__(self) -> int:
        return len(self.flist)

    def _read(self, path: Path, dtype) -> np.ndarray:
        a = np.fromfile(path, count=self.nv, dtype=dtype)
        if a.size < self.nv:
            raise ValueError(f"{path}: {a.size} values, need {self.nv}")
        return a

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        files = self.flist[idx]
        data = np.stack([self._read(f, np.float64) for f in files[:-1]]).astype(np.float32)
        raw = self._read(files[-1], np.int16)
        labels = np.zeros(self.nv, dtype=np.int64)
        for i, c in enumerate(self.classes):  # later duplicates win, as in the defaultdict
            labels[raw == c] = i
        return data, labels


# --------------------------------------------------------------------------- metrics (train.py)


def class_scores(
    pred: torch.Tensor, true: torch.Tensor, nclass: int, drop: Sequence[int]
) -> dict[str, np.ndarray]:
    """Per-class integer counts of ``iou_score``, ``accuracy`` and ``dice`` (``train.py`` 27-71).

    Classes in ``drop`` (class indices, not labels) are left out, as upstream.  Returns
    ``intersect, union, tpos, true_count, dice_denom`` (``dice_denom`` = (|true| + |pred|) / 2).
    """
    keep = [i for i in range(nclass) if i not in drop]
    p = pred.reshape(-1).cpu().numpy()
    t = true.reshape(-1).cpu().numpy()
    pc = np.bincount(p, minlength=nclass)[keep]
    tc = np.bincount(t, minlength=nclass)[keep]
    inter = np.bincount(t[p == t], minlength=nclass)[keep]
    return {
        "intersect": inter,
        "union": pc + tc - inter,
        "tpos": inter,
        "true_count": tc,
        "dice_denom": (tc + pc) / 2,
    }


# --------------------------------------------------------------------------- config / CLI


@dataclass
class TrainConfig:
    """``train.py`` flags (``main`` 149-187); defaults are ``train.py``'s, not ``train.sh``'s."""

    classes: list[int]
    in_ch: list[str]
    batch_size: int = 64
    test_batch_size: int = 64  # parsed but unused upstream (val/test use batch_size)
    epochs: int = 100
    lr: float = 1e-2
    no_cuda: bool = False
    seed: int = 1
    mesh_folder: str | None = None  # upstream "./mesh_files"; None = regenerate operators
    data_folder: str = "processed_data"
    log_interval: int = 10
    max_level: int = 7
    min_level: int = 0
    feat: int = 4
    log_dir: str = "log"
    decay: bool = False
    optim: str = "adam"
    resume: str | None = None
    kcv: int = 5
    fold: int = 1
    blackout_id: str = ""
    ts: float = 10.0
    deg: int = 0
    hemi: str = "lh"
    drop: list[int] = field(default_factory=list)
    train_stats_freq: int = 0


#: ``train.sh`` (the authors' training run; ``data_folder``/``log_dir`` are site paths there)
TRAIN_SH_DEFAULTS = {
    "batch_size": 4,
    "test_batch_size": 10,
    "epochs": 30,
    "max_level": 5,
    "min_level": 0,
    "feat": 32,
    "log_interval": 70,
    "fold": 1,
    "lr": 0.01,
    "classes": [0, 1, 2, 3, 4, 6, 7, 8, 10],
    "in_ch": ["curv", "iH", "sulc"],
    "deg": 15,
    "seed": 1,
    "hemi": "lh",
    "drop": [0],
}


def build_parser() -> argparse.ArgumentParser:
    """Same flag names, types, defaults and help as ``train.py``."""
    p = argparse.ArgumentParser(prog="pybiscuit parc-train", description="Segmentation")
    a = p.add_argument
    # fmt: off
    a("--batch-size", type=int, default=64, metavar="N",
      help="input batch size for training (default: 64)")
    a("--test-batch-size", type=int, default=64, metavar="N",
      help="input batch size for testing (default: 64; unused, as upstream)")
    a("--epochs", type=int, default=100, metavar="N", help="number of epochs to train")
    a("--lr", type=float, default=1e-2, metavar="LR", help="learning rate (default: 0.01)")
    a("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    a("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    a("--mesh_folder", type=str, default=None,
      help="folder with icosphere_<level>.pkl operator files (read with the restricted "
           "unpickler); default: regenerate the operators")
    a("--data_folder", type=str, default="processed_data",
      help="path to data folder (default: processed_data)")
    a("--log-interval", type=int, default=10, metavar="N",
      help="how many batches to wait before logging training status")
    a("--max_level", type=int, default=7, help="max mesh level")
    a("--min_level", type=int, default=0, help="min mesh level")
    a("--feat", type=int, default=4, help="filter dimensions")
    a("--log_dir", type=str, default="log", help="log directory for run")
    a("--decay", action="store_true", help="switch to decay learning rate")
    a("--optim", type=str, default="adam", choices=["adam", "sgd"])
    a("--resume", type=str, default=None, help="path to checkpoint if resume is needed")
    a("--kcv", type=int, default=5, required=False, help="k-fold cross-validation")
    a("--fold", type=int, default=1, required=False, help="choice fold for cross-validation")
    a("--blackout_id", type=str, default="", help="path to file storing blackout_id")
    a("--in_ch", type=str, nargs="+", required=True, help="input channels (list of features)")
    a("--classes", type=int, nargs="+", help="list of classes", required=True)
    a("--ts", type=float, default=10, help="temperature scaling", required=False)
    a("--deg", type=int, default=0, required=False,
      help="degree of spherical harmonics for data augmentation")
    a("--hemi", type=str, default="lh", choices=["lh", "rh"])
    a("--drop", type=int, nargs="+", required=False,
      help="list of labels not considered for log (but still used for gradient update)")
    a("--train_stats_freq", default=0, type=int,
      help="frequency for printing training set stats. 0 for never.")
    # fmt: on
    return p


def config_from_args(ns: argparse.Namespace) -> TrainConfig:
    d = vars(ns).copy()
    d["drop"] = d["drop"] or []
    return TrainConfig(**d)


# --------------------------------------------------------------------------- train / test


def train_epoch(cfg, model, loader, optimizer, epoch, device, keep_id=None) -> float:
    """``train.py`` 73-100: one pass, cross entropy of ``output / ts``; returns the mean loss."""
    model.train()
    tot_loss = 0.0
    count = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data) / cfg.ts
        if keep_id is not None:
            output = output[:, :, keep_id]
            target = target[:, keep_id]
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        count += 1
        if batch_idx % cfg.log_interval == 0:
            seen, pct = batch_idx * len(data), 100.0 * batch_idx / len(loader)
            LOG.info(
                f"Train Epoch: {epoch} [{seen}/{len(loader.dataset)} ({pct:.0f}%)] "
                f"Loss: {loss.item():.6f}"
            )
    tot_loss /= count
    LOG.info(f"[Epoch {epoch} {loader.dataset.partition} stats]: Avg loss: {tot_loss:.4f}")
    return tot_loss


def evaluate(cfg, model, loader, epoch, device, keep_id=None) -> tuple[float, float, float, float]:
    """``train.py`` 103-144 (``test``): ``(mIoU, mean accuracy, mean Dice, mean batch loss)``.

    Counts are summed over the whole partition before dividing; a class absent from both
    prediction and truth gives 0/0 = NaN and makes the mean NaN, as upstream.
    """
    model.eval()
    nclass = len(cfg.classes)
    tot: dict[str, np.ndarray] = {}
    test_loss = 0.0
    count = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data) / cfg.ts
            if keep_id is not None:
                output = output[:, :, keep_id]
                target = target[:, keep_id]
            test_loss += F.cross_entropy(output, target).item()
            pred = output.max(dim=1, keepdim=False)[1]
            for k, v in class_scores(pred, target, nclass, cfg.drop).items():
                tot[k] = tot.get(k, 0) + v
            count += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ious = tot["intersect"] / tot["union"]
        accs = tot["tpos"] / tot["true_count"]
        dices = tot["tpos"] / tot["dice_denom"]
    test_loss /= count
    res = (float(np.mean(ious)), float(np.mean(accs)), float(np.mean(dices)), test_loss)
    LOG.info(
        "[Epoch {} {} stats]: MIoU: {:.4f}; Mean Accuracy: {:.4f}; Mean Dice: {:.4f}; "
        "Avg loss: {:.4f}".format(epoch, loader.dataset.partition, *res)
    )
    return res


# --------------------------------------------------------------------------- checkpoints


def _model_arch(model: SphericalUNet) -> dict:
    return {
        "in_ch": model.in_ch,
        "out_ch": model.out_ch,
        "fdim": model.fdim,
        "max_level": model.max_level,
        "min_level": model.min_level,
    }


def save_best(cfg: TrainConfig, model, optimizer, epoch: int, dice: float) -> Path:
    """``save_checkpoint`` for ``is_best``: the upstream ``.pth`` plus an inference safetensors."""
    out = Path(cfg.log_dir)
    sd = model.state_dict()
    torch.save(
        {"epoch": epoch, "state_dict": sd, "dice": dice, "optimizer": optimizer.state_dict()},
        out / f"{BEST_NAME}.pth",
    )
    tensors = {k: v.detach().cpu() for k, v in sd.items() if not k.endswith("num_batches_tracked")}
    meta = {
        "source": "pybiscuit parc-train",
        "arch": _model_arch(model),
        "epoch": epoch,
        "dice": dice,
        "classes": list(cfg.classes),
        "in_ch": list(cfg.in_ch),
        "hemi": cfg.hemi,
        "fold": cfg.fold,
        "kcv": cfg.kcv,
        "seed": cfg.seed,
    }
    dst = out / f"{BEST_NAME}.safetensors"
    save_safetensors(tensors, dst, meta)
    return dst


def load_resume(path: str | Path) -> dict:
    """Read a resume checkpoint without executing pickled code.

    Our own (zip) files and legacy (non-zip) files alike go through
    ``torch.load(weights_only=True)``; legacy files are first audited by
    :func:`pybiscuit.parc.checkpoint.audit_legacy_pickle`.  ``epoch`` and ``dice`` are required,
    as upstream (``resume_dict['epoch']``).
    """
    from .checkpoint import audit_legacy_pickle

    path = Path(path)
    with open(path, "rb") as fh:
        if fh.read(4) == LEGACY_MAGIC:
            audit_legacy_pickle(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("epoch", "dice", "state_dict"):
        if k not in obj:
            raise KeyError(f"{path}: resume checkpoint has no {k!r}")
    return obj


def _load_tolerant(model, state_dict) -> None:
    """``load_my_state_dict``: copy every known name, silently skip unknown ones."""
    canon, _ = canonicalize_state_dict(state_dict)
    own = model.state_dict()
    with torch.no_grad():
        for name, t in canon.items():
            if name in own:
                own[name].copy_(t)


# --------------------------------------------------------------------------- driver


def _operator_source(mesh_folder: str | None) -> Callable[[int], dict] | None:
    if mesh_folder is None:
        return None
    from ..sphere.operators import safe_load_pickle

    return lambda level: safe_load_pickle(Path(mesh_folder) / f"icosphere_{level}.pkl")


def train(cfg: TrainConfig, *, operators: OperatorCache | None = None) -> dict:
    """``train.py`` ``main`` 189-300.  Returns ``{"model", "history", "best_dice", "best"}``.

    The RNG is seeded once with ``cfg.seed`` (inside :func:`torch.random.fork_rng`, so the caller's
    global state is untouched) and consumed in upstream order: model initialisation, then one
    sampler seed per training epoch.  ``history`` has one dict per epoch.
    """
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "train.log")]
    old_level, old_prop = LOG.level, LOG.propagate
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False
    for h in handlers:
        LOG.addHandler(h)
    try:
        with torch.random.fork_rng(devices=[]):
            return _train(cfg, device, operators)
    finally:
        for h in handlers:
            LOG.removeHandler(h)
            h.close()
        LOG.setLevel(old_level)
        LOG.propagate = old_prop


def _train(cfg: TrainConfig, device, operators) -> dict:
    LOG.info("%s", repr(cfg))
    torch.manual_seed(cfg.seed)

    def ds(part):
        return SegDataset(
            cfg.data_folder, part, fold=cfg.fold, sp_level=cfg.max_level, in_ch=cfg.in_ch,
            classes=cfg.classes, seed=cfg.seed, deg=cfg.deg, kcv=cfg.kcv, hemi=cfg.hemi,
        )  # fmt: skip

    trainset, valset, testset = ds("train"), ds("val"), ds("test")
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    if len(train_loader) == 0:
        raise ValueError(
            f"{len(trainset)} training samples < batch size {cfg.batch_size} (drop_last)"
        )

    model = SphericalUNet(
        in_ch=len(cfg.in_ch), out_ch=len(cfg.classes), max_level=cfg.max_level,
        min_level=cfg.min_level, fdim=cfg.feat,
        operators=operators or _operator_source(cfg.mesh_folder),
    ).to(device)  # fmt: skip

    keep_id = None
    if cfg.blackout_id:
        blackout = np.load(cfg.blackout_id)
        keep_id = torch.from_numpy(
            np.flatnonzero(np.isin(np.arange(model.nv), blackout, invert=True))
        ).to(device)

    start_ep, best_dice = 0, 0.0
    if cfg.resume:
        resume = load_resume(cfg.resume)
        start_ep, best_dice = int(resume["epoch"]), float(resume["dice"])
        _load_tolerant(model, resume["state_dict"])

    # (sic) "paramerters": upstream log text kept verbatim
    LOG.info(f"{sum(x.numel() for x in model.parameters())} paramerters in total")

    if cfg.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    best = None
    for epoch in range(start_ep + 1, cfg.epochs + 1):
        if cfg.decay:  # StepLR(step_size=5, gamma=0.9).step(epoch): closed form
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr * 0.9 ** (epoch // 5)
        tr_loss = train_epoch(cfg, model, train_loader, optimizer, epoch, device, keep_id)
        val = evaluate(cfg, model, val_loader, epoch, device, keep_id)
        tst = evaluate(cfg, model, test_loader, epoch, device, keep_id)
        if cfg.train_stats_freq > 0 and epoch % cfg.train_stats_freq == 0:
            evaluate(cfg, model, train_loader, epoch, device, keep_id)
        dice = val[2]
        is_best = dice > best_dice  # NaN never wins, as upstream
        if is_best:
            best_dice = dice
            best = save_best(cfg, model, optimizer, epoch, dice)
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "val": val, "test": tst, "is_best": is_best,
             "lr": optimizer.param_groups[0]["lr"]}
        )  # fmt: skip
    return {"model": model, "history": history, "best_dice": best_dice, "best": best}


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(build_parser().parse_args(argv))
    res = train(cfg)
    print(f"best val Dice {res['best_dice']:.4f}" + (f": {res['best']}" if res["best"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
