#!/usr/bin/env python
"""BISCUIT's four-call HSD chain at template scale: port vs oracle (plan row B6.5, brief 21; not in CI).

    python scripts/validation/hsd_chain_vs_oracle.py WORKDIR [threads=4] [n_ens=3] [metrics.json]

Real inputs: the shipped ``reg/mb101`` template (``lh.{iH,sulc,curv}.{mean,var}.txt``, 163,842 lines) and
``sphere_{5120,20480,81920,327680}.vtk`` from the extracted image (``~/dev/biscuit/reference``).  No
registered subject exists locally, so the subject's resampled features ``J/lh.res.{iH,sulc,H}.txt`` are the
template means seen through a smooth warp of the sphere plus 5 % noise (seeded); registration should undo the warp.

For each of the four calls the oracle (``hsd-icomesh-nofma``, see ``build_hsd_nofma.sh``) and the port run from the
*oracle's* previous coefficient file (per-call comparison), then the port runs the whole chain (end to end).  The
oracle's own sensitivity is measured with ``n_ens`` runs per call and ``n_ens`` whole chains where one input
value is changed by one float32 ulp.  Results are cached in WORKDIR (rerun to resume) and written as JSON.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
from fixtures.hsd_cases import warp  # noqa: E402
from pybiscuit.hsd import kernels as K  # noqa: E402
from pybiscuit.hsd.chain import ChainPaths, chain_argv, run_chain  # noqa: E402
from pybiscuit.io import read_vtk  # noqa: E402
from pybiscuit.sphere.aabb import SphereAABB  # noqa: E402

REF = Path.home() / "dev" / "biscuit" / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit"
ORA = Path.home() / "dev" / "biscuit" / "oracles" / "hsd-icomesh-nofma" / "bin" / "HSD"
TEMPLATE = "mb101"
FEATURES = {"iH": "iH", "sulc": "sulc", "H": "curv"}  # res.<name> <- template stem


def angles(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cs = np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)
    return np.degrees(np.arccos(cs))


def energies(text):
    return np.array([float(m.group(1)) for m in re.finditer(r"^\[\d+\] ([0-9.e+-]+)", text, re.M)])


def coeff_of(path):
    t = Path(path).read_text().split()
    return np.array(t[1:3], np.float64), np.array(t[3:], np.float64).reshape(-1, 3).T


def make_subject(jdir, tdir, sphere):
    jdir.mkdir(parents=True, exist_ok=True)
    m = read_vtk(sphere)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    tree = SphereAABB(u.astype(np.float32), m.faces)
    fid, bary = tree.closest_face(warp(u, 1, 0.10).astype(np.float32))
    rng = np.random.default_rng(5)
    for feat, stem in FEATURES.items():
        t = np.loadtxt(tdir / f"lh.{stem}.mean.txt").astype(np.float32)
        out = np.empty(len(u), np.float32)
        K.interpolate(t, m.faces, fid, bary, out)
        out += (0.05 * t.std() * rng.standard_normal(len(u))).astype(np.float32)
        np.savetxt(jdir / f"lh.res.{feat}.txt", out, fmt="%.6f")


def ulp_perturb(src, dst, rng, n_first):
    lines = Path(src).read_text().split("\n")
    j = int(rng.integers(0, n_first))
    v = np.float32(float(lines[j]))
    lines[j] = format(np.nextafter(v, np.float32(np.inf if rng.random() < 0.5 else -np.inf)), ".9g")
    Path(dst).write_text("\n".join(lines))


def oracle(argv, threads):
    t = time.time()
    r = subprocess.run([str(ORA), *argv, "--nThreads", str(threads)], capture_output=True, text=True, check=True)
    return energies(r.stdout), time.time() - t


def main():
    work = Path(sys.argv[1])
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_ens = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    metrics_path = Path(sys.argv[4]) if len(sys.argv) > 4 else work / "metrics.json"
    work.mkdir(parents=True, exist_ok=True)
    sd, td, jd = REF / "sphere", REF / "reg" / TEMPLATE, work / "J"
    if not (jd / "lh.res.H.txt").exists():
        make_subject(jd, td, sd / "sphere_327680.vtk")
    paths = ChainPaths("lh", jd, td, sd)
    argvs = chain_argv(paths)
    M = {"template": TEMPLATE, "threads": threads, "n_ens": n_ens}
    if metrics_path.exists():
        M.update(json.loads(metrics_path.read_text()))
    save = lambda: metrics_path.write_text(json.dumps(M, indent=1))  # noqa: E731

    # ---- oracle chain (cached in WORKDIR/oracle: C after each call, registered sphere)
    odir = work / "oracle"
    odir.mkdir(exist_ok=True)
    for i in range(1, 5):
        cf = odir / f"C{i}.txt"
        if cf.exists():
            continue
        if i > 1:
            shutil.copy(odir / f"C{i - 1}.txt", paths.coeff)
        e, dt = oracle(argvs[i - 1], threads)
        shutil.copy(paths.coeff, cf)
        M.setdefault("oracle", {})[f"call{i}"] = {"iters": len(e), "E_final": float(e[-1]), "seconds": dt}
        if i == 4:
            shutil.copy(paths.registered, odir / "registered.vtk")
        save()
    ov = read_vtk(odir / "registered.vtk").vertices

    # ---- port per call (from the oracle's C_{i-1}) and end to end
    M["per_call"] = {}
    for i in range(1, 5):
        if i > 1:
            shutil.copy(odir / f"C{i - 1}.txt", paths.coeff)
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        t = time.time()
        with redirect_stdout(buf):
            run_chain(paths, stages=[i], n_threads=threads)
        dt = time.time() - t
        e = energies(buf.getvalue())
        pole, c = coeff_of(paths.coeff)
        po, co = coeff_of(odir / f"C{i}.txt")
        rec = {"iters_port": len(e), "iters_oracle": M["oracle"][f"call{i}"]["iters"],
               "E_rel": abs(e[-1] - M["oracle"][f"call{i}"]["E_final"]) / M["oracle"][f"call{i}"]["E_final"],
               "coeff_maxdiff": float(np.abs(c - co).max()), "pole_maxdiff": float(np.abs(pole - po).max()),
               "seconds_port": dt}  # fmt: skip
        if i == 4:
            a = angles(read_vtk(paths.registered).vertices, ov)
            rec |= {"ang_median": float(np.median(a)), "ang_p99": float(np.percentile(a, 99)), "ang_max": float(a.max())}
        M["per_call"][f"call{i}"] = rec
        print(f"per call {i}", rec, flush=True)
    save()
    buf = io.StringIO()
    t = time.time()
    with redirect_stdout(buf):
        run_chain(paths, n_threads=threads)
    a = angles(read_vtk(paths.registered).vertices, ov)
    M["end_to_end"] = {"ang_median": float(np.median(a)), "ang_p99": float(np.percentile(a, 99)), "ang_max": float(a.max()),
                       "E_rel": abs(energies(buf.getvalue())[-1] - M["oracle"]["call4"]["E_final"]) / M["oracle"]["call4"]["E_final"],
                       "coeff_maxdiff": float(np.abs(coeff_of(paths.coeff)[1] - coeff_of(odir / "C4.txt")[1]).max()),
                       "seconds_port": time.time() - t}  # fmt: skip
    print("end to end", M["end_to_end"], flush=True)
    save()

    # ---- oracle sensitivity: one float32 ulp in one input value
    rng = np.random.default_rng(1)
    ens = M.setdefault("oracle_sensitivity", {"per_call": {}, "end_to_end": []})
    for i in range(1, 5):
        if f"call{i}" in ens["per_call"]:
            continue
        base = argvs[i - 1]
        pf = base.index("-p") + 2
        nsub = len(read_vtk(base[base.index("-s") + 2]).vertices)
        rows = []
        for _ in range(n_ens):
            a_ = list(base)
            pert = work / "pert.txt"
            ulp_perturb(base[pf], pert, rng, nsub)
            a_[pf] = str(pert)
            if i > 1:
                shutil.copy(odir / f"C{i - 1}.txt", paths.coeff)
            e, _dt = oracle(a_, threads)
            _, c = coeff_of(paths.coeff)
            row = {"iters": len(e), "E_rel": abs(e[-1] - M["oracle"][f"call{i}"]["E_final"]) / M["oracle"][f"call{i}"]["E_final"],
                   "coeff_maxdiff": float(np.abs(c - coeff_of(odir / f"C{i}.txt")[1]).max())}  # fmt: skip
            if i == 4:
                ang = angles(read_vtk(paths.registered).vertices, ov)
                row |= {"ang_median": float(np.median(ang)), "ang_p99": float(np.percentile(ang, 99))}
            rows.append(row)
            print("oracle ulp, call", i, row, flush=True)
        ens["per_call"][f"call{i}"] = rows
        save()
    for _ in range(max(0, n_ens - len(ens["end_to_end"]))):
        pert = work / "pert_e2e.txt"
        ulp_perturb(argvs[0][argvs[0].index("-p") + 2], pert, rng, len(read_vtk(argvs[0][argvs[0].index("-s") + 2]).vertices))
        for i, base in enumerate(argvs, 1):
            a_ = list(base)
            if i == 1:
                a_[a_.index("-p") + 2] = str(pert)
            e, _dt = oracle(a_, threads)
        ang = angles(read_vtk(paths.registered).vertices, ov)
        row = {"ang_median": float(np.median(ang)), "ang_p99": float(np.percentile(ang, 99)), "ang_max": float(ang.max()),
               "E_rel": abs(e[-1] - M["oracle"]["call4"]["E_final"]) / M["oracle"]["call4"]["E_final"]}  # fmt: skip
        ens["end_to_end"].append(row)
        print("oracle ulp, end to end", row, flush=True)
        save()
    print(json.dumps(M, indent=1))


if __name__ == "__main__":
    main()
