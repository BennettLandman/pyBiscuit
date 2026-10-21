#!/usr/bin/env python
"""Print the brief-06 acceptance table: our network vs the original code, all 12 checkpoints.

    PYBISCUIT_ASSETS=~/dev/biscuit/reference/converted python scripts/oracles/parc_acceptance_metrics.py

Needs the converted safetensors (``scripts/convert_parc_checkpoints.py``) and the oracle golden
(``scripts/oracles/run_parc_oracle.py``).  Also times CPU inference (1 thread and default threads).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from pybiscuit.assets import cache_dir
from pybiscuit.parc.infer import ParcModel, resample_to_native
from pybiscuit.sphere.operators import safe_load_pickle

ROOT = Path.home() / "dev/biscuit"
GOLDEN = ROOT / "oracles/parc-unet/golden"
MESH = ROOT / "reference/ctf_plain/SurfPostproc/common/parc/mesh_files"


def timed(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def main() -> None:
    print("| checkpoint | case | max abs diff (scores) | argmax agree | max abs diff (native) "
          "| native argmax agree | pickle-vs-regen ops | oracle test.py s |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|")
    t_all = []
    for f in sorted((cache_dir() / "parc").glob("*/*.safetensors")):
        key = f"{f.parent.name}/{f.stem}"
        model = ParcModel.from_safetensors(f)
        shipped = ParcModel.from_safetensors(
            f, operators=lambda lv: safe_load_pickle(MESH / f"icosphere_{lv}.pkl")
        )
        for c in sorted((GOLDEN / key).iterdir()):
            feats, ref, nat = (np.load(c / n) for n in ("features.npy", "raw.npy", "native.npy"))
            meta = json.loads((c / "meta.json").read_text())
            out = model.predict(feats)
            res = resample_to_native(out, MESH / "icosphere_5.vtk", MESH / "icosphere_6.vtk")
            d_ops = np.abs(out - shipped.predict(feats)).max()
            print(f"| {key} | {c.name} | {np.abs(out - ref).max():.2e} "
                  f"| {(out.argmax(0) == ref.argmax(0)).mean():.5f} | {np.abs(res - nat).max():.2e} "
                  f"| {(res.argmax(0) == nat.argmax(0)).mean():.5f} | {d_ops:.1e} "
                  f"| {meta['test_py_seconds']} |")  # fmt: skip
    model = ParcModel.from_safetensors(cache_dir() / "parc/braincolor/lh.safetensors")
    feats = np.load(GOLDEN / "braincolor/lh/random/features.npy")
    t_def = timed(lambda: model.predict(feats))
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    t_one = timed(lambda: model.predict(feats))
    print(f"\ninference per hemisphere (braincolor/lh, 50 classes, level 5): "
          f"{t_def:.2f} s with {nt} threads, {t_one:.2f} s with 1 thread; "
          f"torch {torch.__version__}")  # fmt: skip
    del t_all


if __name__ == "__main__":
    main()
