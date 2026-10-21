#!/usr/bin/env python
"""Drive the parcellation oracle for every checkpoint and input case (plan row B7.2-oracle).

    python scripts/oracles/run_parc_oracle.py [--jobs 3] [--only braincolor/lh] [--cases random,template]

Needs ``scripts/oracles/build_parc_unet_oracle.sh`` (throwaway venv + staged original code) and, for
the resampling step, the non-FMA SphericalRemesh oracle.  Results go to
``~/dev/biscuit/oracles/parc-unet/golden/<protocol>/<ckpt>/<case>/``:

    features.npy   (3, 10242) float32 input
    raw.npy        (out_ch, 10242) float32 -- exact network output (np.savetxt shim)
    native.npy     (out_ch, 40962) float64 -- SphericalRemesh output text (what BISCUIT consumes)
    meta.json      sha256 of checkpoint / features, seconds for the whole ``test.py`` run

Input cases (all level 5, 10,242 vertices): ``random`` (seeded N(0,1)), ``template-lh`` and
``template-rh`` (first 10,242 values of the mb101 template mean H, iH and sulc: real cortical
fields; ico vertex order is hierarchical so the prefix is the level-5 icosphere).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path.home() / "dev/biscuit"
ORA = ROOT / "oracles"
OPT = ROOT / "reference/sif_FC_v2.2/root/opt/biscuit"
RUNNER = Path(__file__).with_name("parc_unet_oracle.py")
NV = 10242
SEED = 20261020


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_features(case: str) -> np.ndarray:
    if case == "random":
        return np.random.default_rng(SEED).standard_normal((3, NV)).astype(np.float32)
    hemi = case.split("-")[1]
    cols = [
        np.loadtxt(OPT / "reg/mb101" / f"{hemi}.{n}.mean.txt")[:NV]
        for n in ("curv", "iH", "sulc")
    ]  # fmt: skip
    return np.stack(cols).astype(np.float32)


def run_one(job):
    ckpt, case, out_ch = job
    rel = ckpt.relative_to(OPT / "parc")
    out = ORA / "parc-unet/golden" / rel.parent / ckpt.stem / case
    out.mkdir(parents=True, exist_ok=True)
    feats = make_features(case)
    np.save(out / "features.npy", feats)
    work = ORA / "parc-unet/work" / f"{rel.parent.name}-{ckpt.stem}-{case}"
    stage = ORA / "parc-unet/stage"
    t0 = time.perf_counter()
    subprocess.run(
        [
            str(ORA / "parc-unet/venv/bin/python"), "-I", str(RUNNER),
            "--stage", str(stage), "--ckpt", str(ckpt), "--out-ch", str(out_ch),
            "--features", str(out / "features.npy"), "--workdir", str(work),
            "--isphere", str(stage / "mesh_files/icosphere_5.vtk"),
            "--osphere", str(stage / "mesh_files/icosphere_6.vtk"),
            "--remesh-bin", str(ORA / "spherical-remesh-nofma/bin"),
        ],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    dt = time.perf_counter() - t0
    raw = np.stack([np.load(work / "raw" / f"out.mat.{i + 1}.txt.npy") for i in range(out_ch)])
    nat = np.stack([np.loadtxt(work / f"out.mat.{i + 1}.txt") for i in range(out_ch)])
    np.save(out / "raw.npy", raw)
    np.save(out / "native.npy", nat)
    meta = {"ckpt": str(rel), "ckpt_sha256": sha256(ckpt), "case": case, "out_ch": out_ch,
            "features_sha256": sha256(out / "features.npy"), "test_py_seconds": round(dt, 1)}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{rel} {case}: {dt:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", help="substring filter on '<protocol>/<ckpt>'")
    ap.add_argument("--cases", default="random,template")
    a = ap.parse_args()
    out_ch = {"braincolor": 50, "dkt31": 32}
    jobs = []
    for ck in sorted((OPT / "parc").glob("*/?h*.pth")):
        key = f"{ck.parent.name}/{ck.stem}"
        if a.only and a.only not in key:
            continue
        hemi = ck.stem[:2]
        for case in a.cases.split(","):
            case = f"template-{hemi}" if case == "template" else case
            jobs.append((ck, case, out_ch[ck.parent.name]))
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run_one, jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
