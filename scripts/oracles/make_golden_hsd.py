# ruff: noqa: E501
#!/usr/bin/env python
"""Golden outputs for HSD from the non-FMA oracle (``oracles/hsd-nofma``, ``build_hsd_nofma.sh``).

    .venv/bin/python scripts/oracles/make_golden_hsd.py            # all cases
    .venv/bin/python scripts/oracles/make_golden_hsd.py gd_d3 ...  # some

Writes ``tests/data/golden/hsd/<case>/{golden.npz,command.json}``:
* eval cases (``EVAL_CASES``): harness ``eval`` dump = energies at states A and B, damped Newton /
  gradient directions, raw gradient, Hessian (fine levels), features/mean/variance (small cases);
* run cases (``RUN_CASES``): full ``HSD`` CLI run = energy log, final sphere vertices, coefficients.
* landmark / location cases (``LM_EVAL_CASES``, ``LM_RUN_CASES``, ``lm_pole``): the same against
  ``oracles/hsd-lm-nofma`` (``HSD_PATCH_LANDMARKS=1 build_hsd_nofma.sh``: the upstream-disabled
  options uncommented, nothing else changed).
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
ORA_LM = Path.home() / "dev" / "biscuit" / "oracles" / "hsd-lm-nofma" / "bin"
GOLDEN = REPO / "tests" / "data" / "golden" / "hsd"

from fixtures.hsd_cases import (  # noqa: E402
    EVAL_CASES,
    LM_EVAL_CASES,
    LM_RUN_CASES,
    RUN_CASES,
    coeff_states,
    expand_lm_flags,
    write_case,
    write_landmark_case,
)
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


def _lm_argv(nsub, flags):
    argv = ["-s"] + [f"s{k}.sphere.vtk" for k in range(nsub)] + ["-p"] + [f"s{k}.curv.txt" for k in range(nsub)]
    argv += ["-l"] + [f"s{k}.lm.txt" for k in range(nsub)]
    if "--locationWeight" in flags:
        argv += ["--surface"] + [f"s{k}.surf.vtk" for k in range(nsub)]
    return argv + list(flags)


def lm_eval_case(name: str, c: dict, work: Path) -> dict:
    """Harness ``eval`` with landmarks (and location features when ``--locationWeight`` is set)."""
    import os

    nsub = 3
    write_landmark_case(work, nsub)
    A, B = coeff_states(c["deg"], c["deginc"], nsub)
    A.tofile(work / "A.bin")
    B.tofile(work / "B.bin")
    out = work / "out"
    out.mkdir()
    env = dict(
        os.environ, HSD_H_MODE="eval", HSD_H_OUT=str(out), HSD_H_COEFF_A=str(work / "A.bin"),
        HSD_H_COEFF_B=str(work / "B.bin"), HSD_H_DEGINC=str(c["deginc"]), HSD_H_BEG=str(c["beg"]),
        HSD_H_END=str(c["end"]), HSD_H_TRANS=str(c["trans"]), HSD_H_NSUCC="2",
    )  # fmt: skip
    argv = _lm_argv(nsub, c["flags"]) + ["-d", str(c["deg"]), "--icosahedron", str(c["ico"]), "--noguess"]
    subprocess.run([str(ORA_LM / "hsd_harness"), *argv], cwd=work, env=env, check=True, stdout=subprocess.DEVNULL)
    L = lambda n: np.load(out / f"{n}.npy")  # noqa: E731
    d = {"energy_A": L("A.energy"), "energy_B": L("B.energy"), "cost_lm_A": L("A.cost_lm"),
         "cost_lm_B": L("B.cost_lm"), "landmark_A": L("A.landmark")}  # fmt: skip
    for s in range(nsub):
        d[f"gradient_raw_{s}"] = L(f"A.s{s}.gradient_raw")
        d[f"gradient_dir_{s}"] = L(f"A.s{s}.gradient_dir")
        if c["ico"] >= 6 and (c["beg"] != 0 or s == 0):
            d[f"hessian_{s}"] = L(f"A.s{s}.hessian")
        d[f"vertex_A_{s}"] = L(f"A.s{s}.vertex").astype(np.float32)
        d[f"pole_{s}"] = L(f"A.s{s}.pole")
        d[f"tan_{s}"] = L(f"A.s{s}.tan")
    if c["ico"] <= 4:
        for n in ("feature", "mean", "variance"):
            d[f"{n}_A"] = L(f"A.{n}")
    return {"argv": argv, "data": d, "env": {k: v for k, v in env.items() if k.startswith("HSD_H_") and "COEFF" not in k}}


def lm_run_case(name: str, c: dict, work: Path) -> dict:
    nsub = c["nsub"]
    write_landmark_case(work, nsub)
    argv = (
        ["-s"] + [f"s{k}.sphere.vtk" for k in range(nsub)]
        + ["-p"] + [f"s{k}.curv.txt" for k in range(nsub)]
        + ["-o"] + [f"o{k}.reg.vtk" for k in range(nsub)]
        + ["--outputcoeff"] + [f"o{k}.coeff" for k in range(nsub)]
        + expand_lm_flags(c["flags"], nsub, work)
    )  # fmt: skip
    r = subprocess.run([str(ORA_LM / "HSD"), *argv], cwd=work, capture_output=True, text=True, check=True)
    energies = [float(m.group(1)) for m in re.finditer(r"^\[\d+\] ([0-9.e+-]+)", r.stdout, re.M)]
    d = {"energies": np.array(energies)}
    fixed = [0] if "--fixedSubjects" in c["flags"] else []
    for k in range(nsub):
        if k in fixed:
            continue
        d[f"vertex_{k}"] = read_vtk(work / f"o{k}.reg.vtk").vertices.astype(np.float32)
        toks = (work / f"o{k}.coeff").read_text().split()
        d[f"coeff_{k}"] = np.array(toks[3:], dtype=np.float64).reshape(-1, 3).T
        d[f"pole_{k}"] = np.array(toks[1:3], dtype=np.float64)
    if c["flags"][c["flags"].index("--icosahedron") + 1] in ("6", "7"):
        d.update(lm_sensitivity(work, argv, nsub, d))
    return {"argv": argv, "data": d}


K_ENS = 6  # as make_golden_hsd_chain.py: even members one value +-1 ulp, odd members every value 0 / +-1 ulp


def lm_sensitivity(work: Path, argv: list, nsub: int, base: dict) -> dict:
    """The oracle's own spread (Q15) on a damped-Newton run: K_ENS runs with subject 1's property file
    perturbed at the one-ulp level.  Stored as ``ens_E``, ``ens_iters``, ``ens_coeff`` (max |diff| to the
    base run), ``ens_ang_med`` / ``ens_ang_p99`` (max over subjects of the angle to the base sphere, deg)."""
    rng = np.random.default_rng(1)
    src = work / "s1.curv.txt"
    vals = np.array(src.read_text().split(), dtype=np.float64).astype(np.float32)
    e_, it_, c_, am_, ap_ = [], [], [], [], []
    for k in range(K_ENS):
        a = vals.copy()
        if k % 2 == 0:
            j = int(rng.integers(0, len(a)))
            a[j] = np.nextafter(a[j], np.float32(np.inf if rng.random() < 0.5 else -np.inf))
        else:
            hit = rng.random(len(a)) < 0.5
            up = np.where(rng.random(len(a)) < 0.5, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
            a = np.where(hit, np.nextafter(a, up), a)
        np.savetxt(work / "pert.curv.txt", a, fmt="%.9g")
        pa = [("pert.curv.txt" if x == "s1.curv.txt" else x) for x in argv]
        r = subprocess.run([str(ORA_LM / "HSD"), *pa], cwd=work, capture_output=True, text=True, check=True)
        en = [float(m.group(1)) for m in re.finditer(r"^\[\d+\] ([0-9.e+-]+)", r.stdout, re.M)]
        e_.append(en[-1]), it_.append(len(en))
        dc, med, p99 = 0.0, 0.0, 0.0
        for q in range(nsub):
            if f"vertex_{q}" not in base:
                continue
            toks = (work / f"o{q}.coeff").read_text().split()
            dc = max(dc, float(np.abs(np.array(toks[3:], dtype=np.float64).reshape(-1, 3).T - base[f"coeff_{q}"]).max()))
            v = read_vtk(work / f"o{q}.reg.vtk").vertices
            b = base[f"vertex_{q}"].astype(np.float64)
            ang = np.degrees(np.arccos(np.clip((v * b).sum(1) / np.linalg.norm(v, axis=1) / np.linalg.norm(b, axis=1), -1, 1)))
            med, p99 = max(med, float(np.median(ang))), max(p99, float(np.percentile(ang, 99)))
        c_.append(dc), am_.append(med), ap_.append(p99)
    return {"ens_E": np.array(e_), "ens_iters": np.array(it_), "ens_coeff": np.array(c_),
            "ens_ang_med": np.array(am_), "ens_ang_p99": np.array(ap_)}  # fmt: skip


def lm_pole_case(name: str, c: dict, work: Path) -> dict:
    """Harness ``init`` for several ``--maxIter`` (BOBYQA ``maxfun``): pole, tangents, landmark features."""
    import os

    nsub = 3
    write_landmark_case(work, nsub, location=False)
    d = {}
    for mi in (0, 3, 6, 20, 100):
        out = work / f"o{mi}"
        out.mkdir()
        env = dict(os.environ, HSD_H_MODE="init", HSD_H_OUT=str(out))
        argv = _lm_argv(nsub, []) + ["-d", "6", "--icosahedron", "3", "--noguess", "--maxIter", str(mi)]
        subprocess.run([str(ORA_LM / "hsd_harness"), *argv], cwd=work, env=env, check=True, stdout=subprocess.DEVNULL)
        for s in range(nsub):
            d[f"pole_{mi}_{s}"] = np.load(out / f"init.s{s}.pole.npy")
            d[f"tan_{mi}_{s}"] = np.load(out / f"init.s{s}.tan.npy")
        d[f"landmark_{mi}"] = np.load(out / "init.landmark.npy")
    return {"argv": _lm_argv(nsub, []) + ["-d", "6", "--icosahedron", "3", "--noguess", "--maxIter", "0|3|6|20|100"], "data": d}


def main(names):
    cases = {"init": ("init", {})}
    cases.update({n: ("eval", c) for n, c in EVAL_CASES.items()})
    cases.update({n: ("run", c) for n, c in RUN_CASES.items()})
    cases["lm_pole"] = ("lm_pole", {})
    cases.update({n: ("lm_eval", c) for n, c in LM_EVAL_CASES.items()})
    cases.update({n: ("lm_run", c) for n, c in LM_RUN_CASES.items()})
    for n in names or cases:
        kind, c = cases[n]
        with tempfile.TemporaryDirectory() as td:
            fn = {"eval": eval_case, "run": run_case, "init": init_case, "lm_eval": lm_eval_case,
                  "lm_run": lm_run_case, "lm_pole": lm_pole_case}[kind]  # fmt: skip
            res = fn(n, c, Path(td))
        out = GOLDEN / n
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "golden.npz", **res["data"])
        meta = {
            "tool": "hsd", "case": n, "kind": kind, "argv": res["argv"], "params": c,
            "commit": "f169595", "version": "v1.3.9", "meshlib": "dcef807", "compiler": compiler(),
            "oracle": "upstream-macos-arm64, -ffp-contract=off, Accelerate LAPACK/BLAS"
            + (" (hsd-lm-nofma: patches/hsd-landmarks)" if kind.startswith("lm_") else ""),
            "inputs": "tests/fixtures/hsd_cases.py:" + ("write_landmark_case" if kind.startswith("lm_") else "write_case"), "env": res.get("env", {}),
        }  # fmt: skip
        (out / "command.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(n, "->", out, (out / "golden.npz").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1:])
