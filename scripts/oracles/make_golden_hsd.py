# ruff: noqa: E501
#!/usr/bin/env python
"""Golden outputs for HSD from the non-FMA oracle (``oracles/hsd-nofma``, ``build_hsd_nofma.sh``).

    .venv/bin/python scripts/oracles/make_golden_hsd.py            # all cases
    .venv/bin/python scripts/oracles/make_golden_hsd.py gd_d3 ...  # some

Writes ``tests/data/golden/hsd/<case>/{golden.npz,command.json}``:
* eval cases (``EVAL_CASES``): harness ``eval`` dump = energies at states A and B, damped Newton /
  gradient directions, raw gradient, Hessian (fine levels), features/mean/variance (small cases);
* run cases (``RUN_CASES``): full ``HSD`` CLI run = energy log, final sphere vertices, coefficients.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
ORA = Path.home() / "dev" / "biscuit" / "oracles" / "hsd-nofma" / "bin"
GOLDEN = REPO / "tests" / "data" / "golden" / "hsd"

from fixtures.hsd_cases import EVAL_CASES, RUN_CASES, coeff_states, write_case  # noqa: E402
from pybiscuit.io import read_vtk  # noqa: E402


def compiler() -> str:
    out = subprocess.run(["/usr/bin/clang++", "--version"], capture_output=True, text=True)
    return out.stdout.splitlines()[0]


def eval_case(name: str, c: dict, work: Path) -> dict:
    nsub = 3
    write_case(work, nsub)
    A, B = coeff_states(c["deg"], c["deginc"], nsub)
    A.tofile(work / "A.bin")
    B.tofile(work / "B.bin")
    out = work / "out"
    out.mkdir()
    import os

    env = dict(
        os.environ, HSD_H_MODE="eval", HSD_H_OUT=str(out), HSD_H_COEFF_A=str(work / "A.bin"),
        HSD_H_COEFF_B=str(work / "B.bin"), HSD_H_DEGINC=str(c["deginc"]), HSD_H_BEG=str(c["beg"]),
        HSD_H_END=str(c["end"]), HSD_H_TRANS=str(c["trans"]), HSD_H_NSUCC="2",
    )  # fmt: skip
    argv = (
        ["-s"] + [f"s{k}.sphere.vtk" for k in range(nsub)]
        + ["-p"] + [f"s{k}.curv.txt" for k in range(nsub)]
        + ["-d", str(c["deg"]), "--icosahedron", str(c["ico"]), "--noguess"]
    )  # fmt: skip
    subprocess.run([str(ORA / "hsd_harness"), *argv], cwd=work, env=env, check=True, stdout=subprocess.DEVNULL)
    L = lambda n: np.load(out / f"{n}.npy")  # noqa: E731
    d = {"energy_A": L("A.energy"), "energy_B": L("B.energy")}
    for s in range(nsub):
        d[f"gradient_raw_{s}"] = L(f"A.s{s}.gradient_raw")
        d[f"gradient_dir_{s}"] = L(f"A.s{s}.gradient_dir")
        if c["ico"] >= 6 and (c["beg"] != 0 or s == 0):
            d[f"hessian_{s}"] = L(f"A.s{s}.hessian")
        d[f"vertex_A_{s}"] = L(f"A.s{s}.vertex").astype(np.float32)
    if c["ico"] <= 4:
        for n in ("feature", "mean", "variance"):
            d[f"{n}_A"] = L(f"A.{n}")
    return {"argv": argv, "data": d, "env": {k: v for k, v in env.items() if k.startswith("HSD_H_") and "COEFF" not in k}}


def run_case(name: str, c: dict, work: Path) -> dict:
    nsub = c["nsub"]
    write_case(work, nsub)
    tag = "o"
    argv = (
        ["-s"] + [f"s{k}.sphere.vtk" for k in range(nsub)]
        + ["-p"] + [f"s{k}.curv.txt" for k in range(nsub)]
        + ["-o"] + [f"{tag}{k}.reg.vtk" for k in range(nsub)]
        + ["--outputcoeff"] + [f"{tag}{k}.coeff" for k in range(nsub)]
        + c["flags"]
    )  # fmt: skip
    r = subprocess.run([str(ORA / "HSD"), *argv], cwd=work, capture_output=True, text=True, check=True)
    energies = [float(m.group(1)) for m in re.finditer(r"^\[\d+\] ([0-9.e+-]+)", r.stdout, re.M)]
    d = {"energies": np.array(energies)}
    fixed = [int(x) for x in c["flags"][c["flags"].index("--fixedSubjects") + 1 :][:1]] if "--fixedSubjects" in c["flags"] else []
    for k in range(nsub):
        if k in fixed:
            continue
        d[f"vertex_{k}"] = read_vtk(work / f"{tag}{k}.reg.vtk").vertices.astype(np.float32)
        toks = (work / f"{tag}{k}.coeff").read_text().split()
        d[f"coeff_{k}"] = np.array(toks[3:], dtype=np.float64).reshape(-1, 3).T
        d[f"pole_{k}"] = np.array(toks[1:3], dtype=np.float64)
    return {"argv": argv, "data": d}


def init_case(name: str, c: dict, work: Path) -> dict:
    """Harness ``init`` dump: icosahedron samples/faces and the state after construction (B6.3)."""
    import os

    nsub = 3
    write_case(work, nsub)
    d = {}
    for lvl in (3, 4, 5):
        out = work / f"o{lvl}"
        out.mkdir()
        env = dict(os.environ, HSD_H_MODE="init", HSD_H_OUT=str(out))
        argv = (
            ["-s"] + [f"s{k}.sphere.vtk" for k in range(nsub)]
            + ["-p"] + [f"s{k}.curv.txt" for k in range(nsub)]
            + ["-d", "6", "--icosahedron", str(lvl)]
        )  # fmt: skip
        subprocess.run([str(ORA / "hsd_harness"), *argv], cwd=work, env=env, check=True, stdout=subprocess.DEVNULL)
        d[f"samples_{lvl}"] = np.load(out / "samples.npy")
        d[f"faces_{lvl}"] = np.load(out / "ico_faces.npy")
        if lvl == 4:
            for n in ("feature", "mean", "variance"):
                d[f"{n}"] = np.load(out / f"init.{n}.npy")
            for s in range(nsub):
                d[f"vertex_{s}"] = np.load(out / f"init.s{s}.vertex.npy")
                d[f"property_{s}"] = np.load(out / f"init.s{s}.property.npy")
                d[f"tree_cache_{s}"] = np.load(out / f"init.s{s}.tree_cache.npy")
                d[f"area0_{s}"] = np.load(out / f"init.s{s}.area0.npy")
    return {"argv": ["-d", "6", "--icosahedron", "3|4|5"], "data": d}


def main(names):
    cases = {"init": ("init", {})}
    cases.update({n: ("eval", c) for n, c in EVAL_CASES.items()})
    cases.update({n: ("run", c) for n, c in RUN_CASES.items()})
    for n in names or cases:
        kind, c = cases[n]
        with tempfile.TemporaryDirectory() as td:
            res = {"eval": eval_case, "run": run_case, "init": init_case}[kind](n, c, Path(td))
        out = GOLDEN / n
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "golden.npz", **res["data"])
        meta = {
            "tool": "hsd", "case": n, "kind": kind, "argv": res["argv"], "params": c,
            "commit": "f169595", "version": "v1.3.9", "meshlib": "dcef807", "compiler": compiler(),
            "oracle": "upstream-macos-arm64, -ffp-contract=off, Accelerate LAPACK/BLAS",
            "inputs": "tests/fixtures/hsd_cases.py:write_case", "env": res.get("env", {}),
        }  # fmt: skip
        (out / "command.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(n, "->", out, (out / "golden.npz").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1:])
