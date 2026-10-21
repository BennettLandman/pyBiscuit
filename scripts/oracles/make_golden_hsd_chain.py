# ruff: noqa: E501
#!/usr/bin/env python
"""Goldens for BISCUIT's four-call HSD chain from the non-FMA oracle (``oracles/hsd-nofma``), brief 21.

    .venv/bin/python scripts/oracles/make_golden_hsd_chain.py [case ...]

Cases (``tests/fixtures/hsd_chain.py`` inputs, spec §5.3 argv from ``pybiscuit.hsd.chain``):
``chain_lh``, ``chain_rh_macruise`` (rh, call 2 spheres swapped as in the script), ``chain_rh_lhorder``,
``guess_icomesh`` (harness ``guess`` on call 1: initial guess with ``--icomesh`` radius-100 spheres, H-3),
``parc_align`` (``-d 0 --noguess --icosahedron 5``, ``-o`` = input, H-8).
Writes ``tests/data/golden/hsd-chain/<case>/{golden.npz,command.json}``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
ORA = Path.home() / "dev" / "biscuit" / "oracles" / "hsd-icomesh-nofma" / "bin"  # build_hsd_nofma.sh with HSD_PATCH_ICOMESH=1
GOLDEN = REPO / "tests" / "data" / "golden" / "hsd-chain"

from fixtures.hsd_chain import write_chain_case, write_parc_case  # noqa: E402
from pybiscuit.hsd.chain import ChainPaths, chain_argv, parc_alignment_argv  # noqa: E402
from pybiscuit.io import read_vtk  # noqa: E402

CHAINS = {
    "chain_lh": ("lh", 1, "macruise"),
    "chain_rh_macruise": ("rh", 2, "macruise"),
    "chain_rh_lhorder": ("rh", 2, "lh"),
}


def parse_coeff(path):
    toks = Path(path).read_text().split()
    return np.array(toks[1:3], np.float64), np.array(toks[3:], np.float64).reshape(-1, 3).T


def energies(stdout):
    return np.array([float(m.group(1)) for m in re.finditer(r"^\[\d+\] ([0-9.e+-]+)", stdout, re.M)])


def run_chain_case(name, work):
    hemi, k, order = CHAINS[name]
    sd, td, jd = write_chain_case(work, hemi, warp_k=k)
    paths = ChainPaths(hemi, jd, td, sd)
    data, argvs = {}, []
    for i, a in enumerate(chain_argv(paths, rh_stage2=order), 1):
        a = [*a, "--nThreads", "4"]
        r = subprocess.run([str(ORA / "HSD"), *a], capture_output=True, text=True, check=True)
        argvs.append([x.replace(str(work), "<root>") for x in a])
        data[f"energies_{i}"] = energies(r.stdout)
        pole, c = parse_coeff(paths.coeff)
        data[f"pole_{i}"], data[f"coeff_{i}"] = pole, c
    data["vertex"] = read_vtk(paths.registered).vertices.astype(np.float32)
    data.update(sensitivity(work, paths, hemi, order, data))
    return {"argv": argvs, "data": data, "inputs": f"tests/fixtures/hsd_chain.py:write_chain_case({hemi!r}, warp_k={k})"}


K_ENS = 6  # members alternate: one value +-1 ulp / every value 0 or +-1 ulp (see perturb)


def angles(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cs = np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)
    return np.degrees(np.arccos(cs))


def ulp_perturbed(src, dst, rng, n_first):
    """Copy a property file changing one of its first ``n_first`` values by one float32 ulp."""
    lines = Path(src).read_text().split("\n")
    j = int(rng.integers(0, n_first))
    v = np.float32(float(lines[j]))
    lines[j] = format(np.nextafter(v, np.float32(np.inf if rng.random() < 0.5 else -np.inf)), ".9g")
    Path(dst).write_text("\n".join(lines))


def ulp_noise(src, dst, rng, p=0.5):
    """Copy a property file moving every value by one float32 ulp (up or down) with probability ``p``: the size of change
    that another libm / summation order spreads over all vertices."""
    a = np.array(Path(src).read_text().split(), dtype=np.float64).astype(np.float32)
    hit = rng.random(len(a)) < p
    up = np.where(rng.random(len(a)) < 0.5, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
    np.savetxt(dst, np.where(hit, np.nextafter(a, up), a), fmt="%.9g")


def perturb(src, dst, rng, n_first, k):
    """Ensemble member ``k``: even = one value changed by one ulp, odd = noise on all values."""
    (ulp_perturbed if k % 2 == 0 else lambda s_, d_, r_, n_: ulp_noise(s_, d_, r_))(src, dst, rng, n_first)


def sensitivity(work, paths, hemi, order, data):
    """The oracle's own sensitivity: K_ENS runs per call (from the oracle's input state) and K_ENS whole chains with
    the call-1 property file perturbed at the one-ulp level.  Stored as ``ens_*`` / ``e2e_*`` arrays."""
    from pybiscuit.sphere.spharm import ShDeformation, write_sh_coefficients

    rng = np.random.default_rng(1)
    argvs = chain_argv(paths, rh_stage2=order)
    base_v = data["vertex"]
    out = {}

    def set_coeff(i):
        write_sh_coefficients(str(paths.coeff), ShDeformation(10, data[f"pole_{i}"].astype(np.float32), data[f"coeff_{i}"]))

    for i in (1, 2, 3, 4):
        base = argvs[i - 1]
        pf = base.index("-p") + 2
        nsub = len(read_vtk(base[base.index("-s") + 2]).vertices)
        e_, it_, c_, am_, ap_ = [], [], [], [], []
        for k in range(K_ENS):
            a = [*base, "--nThreads", "4"]
            pert = work / "pert.txt"
            perturb(base[pf], pert, rng, nsub, k)
            a[pf] = str(pert)
            if i > 1:
                set_coeff(i - 1)
            r = subprocess.run([str(ORA / "HSD"), *a], capture_output=True, text=True, check=True)
            en = energies(r.stdout)
            e_.append(en[-1]), it_.append(len(en)), c_.append(parse_coeff(paths.coeff)[1])
            if i == 4:
                ang = angles(read_vtk(paths.registered).vertices, base_v)
                am_.append(np.median(ang)), ap_.append(np.percentile(ang, 99))
        out[f"ens_E_{i}"], out[f"ens_iters_{i}"], out[f"ens_coeff_{i}"] = np.array(e_), np.array(it_), np.array(c_)
        if i == 4:
            out["ens_ang_med_4"], out["ens_ang_p99_4"] = np.array(am_), np.array(ap_)
    # whole chain with the 1-ulp change in call 1's property file
    em, ep, ec, ee = [], [], [], []
    for k in range(K_ENS):
        pert = work / "pert_e2e.txt"
        perturb(argvs[0][argvs[0].index("-p") + 2], pert, rng, len(read_vtk(argvs[0][argvs[0].index("-s") + 2]).vertices), k)
        for i, base in enumerate(argvs, 1):
            a = [*base, "--nThreads", "4"]
            if i == 1:
                a[a.index("-p") + 2] = str(pert)
            r = subprocess.run([str(ORA / "HSD"), *a], capture_output=True, text=True, check=True)
        ang = angles(read_vtk(paths.registered).vertices, base_v)
        em.append(np.median(ang)), ep.append(np.percentile(ang, 99)), ec.append(parse_coeff(paths.coeff)[1]), ee.append(energies(r.stdout)[-1])
    out["e2e_ang_med"], out["e2e_ang_p99"], out["e2e_coeff4"], out["e2e_E"] = map(np.array, (em, ep, ec, ee))
    return out


def guess_case(work):
    sd, td, jd = write_chain_case(work, "lh", warp_k=1)
    a = chain_argv(ChainPaths("lh", jd, td, sd))[0]
    out = work / "out"
    out.mkdir()
    env = dict(os.environ, HSD_H_MODE="guess", HSD_H_OUT=str(out))
    subprocess.run([str(ORA / "hsd_harness"), *a], env=env, check=True, capture_output=True)
    data = {"coeff_all": np.load(out / "guess.coeff_all.npy"), "mean": np.load(out / "guess.mean.npy"),
            "feature": np.load(out / "guess.feature.npy")}  # fmt: skip
    return {"argv": [x.replace(str(work), "<root>") for x in a], "data": data, "inputs": "write_chain_case('lh')"}


def parc_case(work):
    t, n, cm, h = write_parc_case(work)
    a = parc_alignment_argv(t, n, cm, h)
    r = subprocess.run([str(ORA / "HSD"), *a, "--nThreads", "4"], capture_output=True, text=True, check=True)
    # -o == input: the native sphere was overwritten
    data = {"energies": energies(r.stdout), "vertex": read_vtk(n).vertices.astype(np.float32)}
    return {"argv": [x.replace(str(work), "<root>") for x in a], "data": data, "inputs": "write_parc_case()"}


def main(names):
    cases = {n: ("chain", None) for n in CHAINS} | {"guess_icomesh": ("guess", None), "parc_align": ("parc", None)}
    for n in names or cases:
        with tempfile.TemporaryDirectory() as td:
            w = Path(td)
            kind = cases[n][0]
            res = run_chain_case(n, w) if kind == "chain" else guess_case(w) if kind == "guess" else parc_case(w)
        out = GOLDEN / n
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "golden.npz", **res["data"])
        meta = {"tool": "hsd", "case": n, "argv": res["argv"], "commit": "f169595", "version": "v1.3.9", "meshlib": "dcef807",
                "oracle": "upstream-macos-arm64, -ffp-contract=off, Accelerate LAPACK/BLAS (hsd-nofma)",
                "inputs": res["inputs"]}  # fmt: skip
        (out / "command.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(n, "->", (out / "golden.npz").stat().st_size // 1024, "KB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
