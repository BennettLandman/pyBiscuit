#!/usr/bin/env python
"""Golden traces of upstream HSD ``src/bobyqa.h`` (black box) for ``pybiscuit.hsd.bobyqa``.

    /usr/bin/clang++ -std=c++11 -O2 -ffp-contract=off -w -include cstddef \
        -I$REF/MeshLib/include/Util -I$REF/HSD/src scripts/oracles/harness_bobyqa.cpp \
        -o ~/dev/biscuit/oracles/bobyqa/bin/bobyqa_harness
    .venv/bin/python scripts/oracles/make_golden_bobyqa.py

Writes ``tests/data/golden/bobyqa/{golden.npz,command.json}``: for every (case, maxfun) the
evaluated points and values (``<case>_<maxfun>_evals``, shape (nf, 3)) and the result
(``<case>_<maxfun>_x``).  The header is compiled and run only (docs/clean-room.md, Q19).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BIN = Path.home() / "dev" / "biscuit" / "oracles" / "bobyqa" / "bin" / "bobyqa_harness"
OUT = REPO / "tests" / "data" / "golden" / "bobyqa"
CASES = ("quad", "rosen", "bound", "cosf", "sphere")
MAXFUN = (3, 5, 8, 20, 200)


def run(case: str, maxfun: int):
    r = subprocess.run([str(BIN), case, str(maxfun)], capture_output=True, text=True, check=True)
    evals, x = [], None
    for line in r.stdout.splitlines():
        t = line.split()
        if t[0] == "eval":
            evals.append([float(v) for v in t[1:4]])
        elif t[0] == "result":
            x = [float(t[1]), float(t[2])]
    return np.array(evals).reshape(-1, 3), np.array(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for c in CASES:
        for m in MAXFUN:
            ev, x = run(c, m)
            data[f"{c}_{m}_evals"] = ev
            data[f"{c}_{m}_x"] = x
    np.savez_compressed(OUT / "golden.npz", **data)
    cc = subprocess.run(["/usr/bin/clang++", "--version"], capture_output=True, text=True).stdout
    meta = {
        "tool": "bobyqa.h (HSD f169595, black box)", "harness": "scripts/oracles/harness_bobyqa.cpp",
        "cases": CASES, "maxfun": MAXFUN, "x0": "0,0 (rosen: -1.2,1)", "bounds": "[-PI, PI]^2",
        "rhobeg": 0.1, "rhoend": 1e-6, "npt": 5, "compiler": cc.splitlines()[0],
        "flags": "-O2 -ffp-contract=off", "libm": "macOS (cosf/sinf/acosf in cases cosf, sphere)",
    }  # fmt: skip
    (OUT / "command.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
