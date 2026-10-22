#!/usr/bin/env python
"""FreeSurfer-route replay against the real ``biscuit_FC_v2.2`` outputs in ``open-examples`` (brief 22).

    python scripts/validation/open_examples.py WORKDIR [--subjects 0050952 ...] [--hemis lh rh]
        [--stages input reg remesh trace sd lgi] [--threads 8] [--ens 3] [--report DIR]
    python scripts/validation/open_examples.py WORKDIR --summary-only [--report DIR]

Subjects are ABIDE (NYU) cases processed with FreeSurfer 7.4.1 and then ``biscuit_FC_v2.2 --surf_recon
freesurfer`` (template ``mb101``).  The data stays outside git; only the metrics JSON and the summary table
are written into the report directory (default ``docs/reports/open-examples-2026-10-22``).

Stages (per subject and hemisphere; each caches its JSON in WORKDIR and is skipped on a rerun):

* ``input``  -- ``pybiscuit.surfprep.freesurfer`` vs the image's ``surf/*.vtk``, ``roi/mask``, ``quant/thickness``
  and ``mid.area`` (byte / print-precision identity).
* ``reg``    -- the FreeSurfer-route HSD chain (``surf_reg_fs2.m``) on the image's inputs vs ``reg/<h>.coeff.txt``
  and the deformed native sphere ``reg/<h>.sphere.reg.vtk``.  HSD is chaotic (Q15), so the chain is rerun from
  1-ulp-perturbed inputs to measure this port's own spread, and the verdict is given under the policy bound and
  under the (proposed, not adopted) ``max(policy, 1.5 x spread)`` rule.  The chain with upstream's raw
  ``--icomesh`` initial guess (the macOS-oracle behaviour) is run once for comparison.
* ``remesh`` -- stable stages given *the image's* coefficient / maps: ``SphericalRemesh`` of gray/white/mid onto
  the template tessellation (``reg/*.reg.vtk``) and of the quant maps (``stat/*.reg.*``).
* ``trace``  -- the four ``CurveExtraction`` calls vs ``curve/`` (byte identity, vertex Jaccard).
* ``sd``     -- sulcal depth vs ``quant/<h>.gray.SD.txt``.
* ``lgi``    -- ``OuterHull`` + hull-correspondence stand-in + ``Gyrification`` on the image's curves vs
  ``quant/<h>.gray.LGI.txt``, with the Q14/Q15 distribution criterion and the port's own 1-ulp spread.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pybiscuit.hsd.chain import ChainPaths, run_chain  # noqa: E402
from pybiscuit.io import read_vtk  # noqa: E402
from pybiscuit.mesh.geometry import vertex_areas  # noqa: E402
from pybiscuit.sphere.remesh import resample  # noqa: E402
from pybiscuit.sphere.remesh import run as remesh_run  # noqa: E402
from pybiscuit.surfprep.freesurfer import prepare_fs_subject  # noqa: E402

OPEN = Path.home() / "dev" / "biscuit" / "open-examples"
REF = Path.home() / "dev" / "biscuit" / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit"
TEMPLATE = "mb101"
SUBJECTS = ("0050952", "0050961", "0050971")
STAGES = ("input", "reg", "remesh", "trace", "sd", "lgi")
UNI = REF / "sphere" / "sphere_327680.vtk"

# E-class bounds of docs/validation-tolerances.md (never loosened here)
POLICY = {"sphere_median_deg": 0.01, "sphere_p99_deg": 0.1, "coeff_abs": 1e-4, "energy_rel": 1e-3,
          "scalar_rel": 1e-4, "scalar_r": 0.9999, "jaccard": 0.99, "hausdorff_mm": 0.5}  # fmt: skip
# Q14 (proposed, not adopted): distributional LGI criterion
Q14 = {"r": 0.9999, "p99_rel": 1e-3, "roi_mean_rel": 1e-3}


# ------------------------------------------------------------------ metrics helpers
def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def angles(a, b):
    cs = np.clip((unit(np.asarray(a, np.float64)) * unit(np.asarray(b, np.float64))).sum(1), -1, 1)
    return np.degrees(np.arccos(cs))


def ang_stats(a, b):
    d = angles(a, b)
    return {"median_deg": float(np.median(d)), "p99_deg": float(np.percentile(d, 99)), "max_deg": float(d.max())}


def robust_range(x):
    return float(np.percentile(x, 99) - np.percentile(x, 1))


def vec(ours, ref):
    """Per-vertex scalar agreement metrics (ours vs the image's file)."""
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    if ours.shape != ref.shape:
        return {"shape_ours": list(ours.shape), "shape_ref": list(ref.shape), "ok": False}
    d = np.abs(ours - ref)
    rr = robust_range(ref) or 1.0
    return {
        "n": int(ref.size), "pearson_r": float(np.corrcoef(ours, ref)[0, 1]) if ref.std() else 1.0,
        "max_abs": float(d.max()), "max_abs_over_range": float(d.max() / rr),
        "p99_abs_over_range": float(np.percentile(d, 99) / rr), "mean_diff": float((ours - ref).mean()),
        "frac_exact_6dp": float((d < 5e-7 + 1e-12).mean()),
    }  # fmt: skip


def e_class(m):
    return bool(m.get("pearson_r", 0) >= POLICY["scalar_r"] and m.get("max_abs_over_range", 1) <= POLICY["scalar_rel"])


def read_txt(p):
    return np.loadtxt(p, dtype=np.float64)


def read_coeff(p):
    t = Path(p).read_text().split()
    return np.array(t[1:3], np.float64), np.array(t[3:], np.float64).reshape(-1, 3)


def same_bytes(a, b):
    return Path(a).read_bytes() == Path(b).read_bytes()


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def timed(fn):
    t = time.time()
    out = fn()
    return out, time.time() - t


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


# ------------------------------------------------------------------ stage: input
def stage_input(ctx):
    src, job, h = ctx["src"], ctx["job"], ctx["hemi"]
    prepare_fs_subject(src, job, hemis=(h,))
    m = {}
    for n in ("white", "gray", "mid", "inflated", "sphere"):
        m[f"{n}_vtk_identical"] = same_bytes(job / f"{h}.{n}.vtk", src / "surf" / f"{h}.{n}.vtk")
    m["mask_identical"] = same_bytes(job / f"{h}.mask.txt", src / "roi" / f"{h}.mask.txt")
    m["thickness_identical"] = same_bytes(job / f"{h}.thickness.txt", src / "quant" / f"{h}.thickness.txt")
    mid = read_vtk(job / f"{h}.mid.vtk")
    area = vertex_areas(mid.vertices.astype(np.float32), mid.faces)
    m["mid_area"] = vec(area, read_txt(src / "quant" / f"{h}.mid.area.txt"))
    m["pass"] = all(v for k, v in m.items() if k.endswith("identical")) and m["mid_area"]["max_abs"] < 1e-6
    return m


# ------------------------------------------------------------------ stage: reg
def ulp_one(src, dst, rng, n_first):
    """Copy a property file changing one of its first ``n_first`` values by one float32 ulp."""
    lines = Path(src).read_text().split("\n")
    j = int(rng.integers(0, n_first))
    v = np.float32(float(lines[j]))
    lines[j] = format(np.nextafter(v, np.float32(np.inf if rng.random() < 0.5 else -np.inf)), ".9g")
    Path(dst).write_text("\n".join(lines))
    return j


def ulp_noise(src, dst, rng, p=0.5):
    a = np.array(Path(src).read_text().split(), dtype=np.float64).astype(np.float32)
    hit = rng.random(len(a)) < p
    up = np.where(rng.random(len(a)) < 0.5, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
    np.savetxt(dst, np.where(hit, np.nextafter(a, up), a), fmt="%.9g")


def res_features(ctx):
    job, h = ctx["job"], ctx["hemi"]
    for f in ("curv", "inflated.H", "sulc"):
        if not (job / f"{h}.res.{f.split('.')[-1]}.txt").exists():
            remesh_run(job / f"{h}.sphere.vtk", ref=UNI, property=[job / f"{h}.{f}.txt"],
                       output_property=job / f"{h}.res", quiet=True)  # fmt: skip


def zscore(x):
    return (x - x.mean()) / x.std(ddof=1)


def energy(tmpl, x):
    return float(np.linalg.norm(zscore(tmpl) - zscore(x)) ** 2 / len(tmpl))


def chain_dir(ctx, name):
    d = ctx["work"] / f"chain-{ctx['hemi']}-{name}"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("curv", "H", "sulc"):
        shutil.copy(ctx["job"] / f"{ctx['hemi']}.res.{f}.txt", d / f"{ctx['hemi']}.res.{f}.txt")
    return d


def deform_native(ctx, coeff, out):
    remesh_run(ctx["job"] / f"{ctx['hemi']}.sphere.vtk", coeff=coeff, deform=out, deg=10, quiet=True)
    return read_vtk(out).vertices


def reg_energy(ctx, coeff, tag):
    """The PDF's E1: template ``curv.mean`` against ``res.curv`` seen through the registered uniform sphere."""
    d, h = ctx["work"], ctx["hemi"]
    s = d / f"{tag}.{h}.res.sphere.reg.vtk"
    remesh_run(UNI, coeff=coeff, deform=s, deg=10, quiet=True)
    remesh_run(s, ref=UNI, property=[ctx["job"] / f"{h}.res.curv.txt"], output_property=d / f"{tag}.{h}.reg", quiet=True)
    tm = read_txt(REF / "reg" / TEMPLATE / f"{h}.curv.mean.txt")
    return energy(tm, read_txt(d / f"{tag}.{h}.reg.curv.txt"))


def run_one_chain(ctx, name, threads, raw=False, perturb=None, rng=None):
    h = ctx["hemi"]
    d = chain_dir(ctx, name)
    if perturb == "H":
        ulp_one(ctx["job"] / f"{h}.res.H.txt", d / f"{h}.res.H.txt", rng, 10242)
    elif perturb == "sulc":
        ulp_one(ctx["job"] / f"{h}.res.sulc.txt", d / f"{h}.res.sulc.txt", rng, 40962)
    elif perturb == "noise":
        for f in ("curv", "H", "sulc"):
            ulp_noise(ctx["job"] / f"{h}.res.{f}.txt", d / f"{h}.res.{f}.txt", rng)
    paths = ChainPaths(h, d, REF / "reg" / TEMPLATE, REF / "sphere")
    t = time.time()
    try:
        quiet(run_chain, paths, route="fs", n_threads=threads, guess_raw_icomesh=raw)
    except RuntimeError as exc:  # upstream's "no closest triangle found" abort (Q16)
        return {"failed": str(exc), "seconds": time.time() - t}, None
    return {"seconds": time.time() - t}, paths.coeff


def stage_reg(ctx):
    h, src, threads = ctx["hemi"], ctx["src"], ctx["threads"]
    res_features(ctx)
    img_coeff = src / "reg" / f"{h}.coeff.txt"
    img_sphere = read_vtk(src / "reg" / f"{h}.sphere.reg.vtk").vertices
    tm = read_txt(REF / "reg" / TEMPLATE / f"{h}.curv.mean.txt")
    res_curv = read_txt(ctx["job"] / f"{h}.res.curv.txt")
    m = {"E0_unregistered": energy(tm, res_curv), "E1_image": reg_energy(ctx, img_coeff, "image")}
    # the image's own coefficient pushed through our SphericalRemesh: stable stage
    d = deform_native(ctx, img_coeff, ctx["work"] / f"image-coeff.{h}.sphere.reg.vtk")
    m["remesh_of_image_coeff_vs_image_sphere_reg"] = {"max_abs": float(np.abs(d - img_sphere).max())}
    ic = read_coeff(img_coeff)[1]

    def compare(coeff, tag):
        c = read_coeff(coeff)[1]
        v = deform_native(ctx, coeff, ctx["work"] / f"{tag}.{h}.sphere.reg.vtk")
        return {
            "coeff_max_abs": float(np.abs(c - ic).max()),
            "coeff_rel_norm": float(np.linalg.norm(c - ic) / np.linalg.norm(ic)),
            "sphere": ang_stats(v, img_sphere), "E1": reg_energy(ctx, coeff, tag), "_v": v,
        }  # fmt: skip

    base, coeff = run_one_chain(ctx, "base", threads)
    if coeff is None:
        m["port"] = base
        return m
    port = compare(coeff, "base")
    base_v = port.pop("_v")
    port["seconds"] = base["seconds"]
    port["E1_rel_diff_vs_image"] = abs(port["E1"] - m["E1_image"]) / m["E1_image"]
    m["port"] = port
    # upstream-faithful initial guess (macOS oracle behaviour)
    raw, rcoeff = run_one_chain(ctx, "raw", threads, raw=True)
    if rcoeff is not None:
        r = compare(rcoeff, "raw")
        r.pop("_v")
        m["port_raw_icomesh_guess"] = r | raw
    else:
        m["port_raw_icomesh_guess"] = raw
    # the port's own 1-ulp spread
    rng = np.random.default_rng(22)
    kinds = ["H", "sulc", "noise", "H", "sulc", "noise"][: ctx["ens"]]
    spreads = []
    for i, kind in enumerate(kinds):
        info, c = run_one_chain(ctx, f"ens{i}", threads, perturb=kind, rng=rng)
        if c is None:
            spreads.append({"kind": kind} | info)
            continue
        v = deform_native(ctx, c, ctx["work"] / f"ens{i}.{h}.sphere.reg.vtk")
        spreads.append({"kind": kind, "sphere_vs_base": ang_stats(v, base_v),
                        "coeff_max_abs_vs_base": float(np.abs(read_coeff(c)[1] - read_coeff(coeff)[1]).max()),
                        "E1": reg_energy(ctx, c, f"ens{i}")})  # fmt: skip
    m["ulp_ensemble"] = spreads
    ok = [s for s in spreads if "sphere_vs_base" in s]
    sp = {
        "median_deg": max((s["sphere_vs_base"]["median_deg"] for s in ok), default=0.0),
        "p99_deg": max((s["sphere_vs_base"]["p99_deg"] for s in ok), default=0.0),
        "coeff_abs": max((s["coeff_max_abs_vs_base"] for s in ok), default=0.0),
        "E1_rel": max((abs(s["E1"] - port["E1"]) / port["E1"] for s in ok), default=0.0),
    }
    m["port_own_spread"] = sp
    s, b = port["sphere"], POLICY
    q15 = {
        "sphere_median_deg": max(b["sphere_median_deg"], 1.5 * sp["median_deg"]),
        "sphere_p99_deg": max(b["sphere_p99_deg"], 1.5 * sp["p99_deg"]),
        "coeff_abs": max(b["coeff_abs"], 1.5 * sp["coeff_abs"]),
        "energy_rel": max(b["energy_rel"], 1.5 * sp["E1_rel"]),
    }
    m["verdict"] = {
        "policy": {"sphere": s["median_deg"] <= b["sphere_median_deg"] and s["p99_deg"] <= b["sphere_p99_deg"],
                   "coeff": port["coeff_max_abs"] <= b["coeff_abs"],
                   "energy": port["E1_rel_diff_vs_image"] <= b["energy_rel"]},
        "q15_proposed_bounds": q15,
        "q15_proposed": {"sphere": s["median_deg"] <= q15["sphere_median_deg"] and s["p99_deg"] <= q15["sphere_p99_deg"],
                         "coeff": port["coeff_max_abs"] <= q15["coeff_abs"],
                         "energy": port["E1_rel_diff_vs_image"] <= q15["energy_rel"]},
    }  # fmt: skip
    return m


# ------------------------------------------------------------------ stage: remesh (image coefficient / maps)
STAT_MAPS = {  # stat file -> (quant source, kind)
    "gray.reg.LGI": "gray.LGI", "gray.reg.SD": "gray.SD", "mid.reg.H": "mid.H",
    "mid.reg.S": "mid.S", "mid.reg.SCI": "mid.SCI",
}  # fmt: skip


def stage_remesh(ctx):
    h, src, work = ctx["hemi"], ctx["src"], ctx["work"]
    img_coeff, sphere = src / "reg" / f"{h}.coeff.txt", ctx["job"] / f"{h}.sphere.vtk"
    m = {"surfaces": {}, "maps": {}}
    for s in ("gray", "white", "mid"):
        out = work / f"image-coeff.{h}.{s}.reg.vtk"
        remesh_run(sphere, input=ctx["job"] / f"{h}.{s}.vtk", coeff=img_coeff, ref=UNI, output=out, deg=10, quiet=True)
        a, b = read_vtk(out), read_vtk(src / "reg" / f"{h}.{s}.reg.vtk")
        m["surfaces"][s] = {"faces_identical": bool(np.array_equal(a.faces, b.faces)),
                            "max_abs": float(np.abs(a.vertices - b.vertices).max())}  # fmt: skip
    # property maps through the image's registered native sphere (T40, no --deg)
    names = list(STAT_MAPS) + ["reg.thickness"]
    srcs = [read_txt(src / "quant" / f"{h}.{STAT_MAPS[n]}.txt") for n in STAT_MAPS]
    srcs.append(read_txt(src / "quant" / f"{h}.thickness.txt"))
    outs = resample(src / "reg" / f"{h}.sphere.reg.vtk", UNI, srcs)
    for n, o in zip(names, outs, strict=True):
        m["maps"][n] = vec(o, read_txt(src / "stat" / f"{h}.{n}.txt"))
        m["maps"][n]["e_class"] = e_class(m["maps"][n])
    mid_reg = read_vtk(src / "reg" / f"{h}.mid.reg.vtk")
    area = vertex_areas(mid_reg.vertices.astype(np.float32), mid_reg.faces)
    m["maps"]["mid.reg.area"] = vec(area, read_txt(src / "stat" / f"{h}.mid.reg.area.txt"))
    m["maps"]["mid.reg.area"]["e_class"] = e_class(m["maps"]["mid.reg.area"])
    m["pass"] = all(s["faces_identical"] and s["max_abs"] < 1e-4 for s in m["surfaces"].values()) and all(
        v["e_class"] for v in m["maps"].values())
    return m  # fmt: skip


# ------------------------------------------------------------------ stage: trace
def stage_trace(ctx):
    from pybiscuit.quant.lgi import curve_vertex_ids, read_curves
    from pybiscuit.trace.pipeline import curve_extract_pipeline

    h, src, work = ctx["hemi"], ctx["src"], ctx["work"]
    out = work / "curve"
    _, secs = timed(lambda: quiet(curve_extract_pipeline, ctx["job"], out, hemis=(h,), n_threads=ctx["threads"], force=True))
    m = {"seconds": secs, "files": {}}
    for name in (f"{h}.gray.scurve", f"{h}.gray.scurve.bary", f"{h}.white.gcurve", f"{h}.white.gcurve.bary"):
        m["files"][name] = {"identical": same_bytes(out / name, src / "curve" / name)}
    for kind, name in (("sulcal", f"{h}.gray.scurve"), ("gyral", f"{h}.white.gcurve")):
        a, b = curve_vertex_ids(out / name), curve_vertex_ids(src / "curve" / name)
        m[f"{kind}_vertex_jaccard"] = float(len(set(a) & set(b)) / max(len(set(a) | set(b)), 1))
        m[f"{kind}_curves"] = {"ours": len(read_curves(out / name)), "image": len(read_curves(src / "curve" / name))}
    m["pass"] = all(f["identical"] for f in m["files"].values())
    return m


# ------------------------------------------------------------------ stage: sd
def stage_sd(ctx):
    from pybiscuit.quant.sulcal_depth import sulcal_depth

    h, src = ctx["hemi"], ctx["src"]
    sd, secs = timed(lambda: sulcal_depth(src / "surf" / f"{h}.gray.vtk"))
    ref = read_txt(src / "quant" / f"{h}.gray.SD.txt")
    m = vec(sd, ref) | {"seconds": secs}
    d = np.abs(sd - ref)
    m["frac_within_0p1mm"] = float((d < 0.1).mean())
    m["frac_exact"] = float((d < 1e-3).mean())
    m["e_class"] = e_class(m)
    return m


# ------------------------------------------------------------------ stage: lgi
def roi_means(x, labels):
    return {int(k): float(x[labels == k].mean()) for k in np.unique(labels) if k > 0}


def q14(x, ref, labels):
    rel = np.abs(x - ref) / np.abs(ref)
    rm, rr = roi_means(x, labels), roi_means(ref, labels)
    roi_rel = max(abs(rm[k] - rr[k]) / abs(rr[k]) for k in rr)
    r = float(np.corrcoef(x, ref)[0, 1])
    out = {"pearson_r": r, "rel_p50": float(np.percentile(rel, 50)), "rel_p99": float(np.percentile(rel, 99)),
           "rel_max": float(rel.max()), "roi_mean_rel_max": roi_rel, "n_rois": len(rr),
           "mean_ratio": float((x / ref).mean())}  # fmt: skip
    out["q14_pass"] = bool(r >= Q14["r"] and out["rel_p99"] <= Q14["p99_rel"] and roi_rel <= Q14["roi_mean_rel"])
    return out


def stage_lgi(ctx):
    from pybiscuit.io import write_vtk
    from pybiscuit.quant.lgi_pipeline import lgi_pipeline

    h, src, work = ctx["hemi"], ctx["src"], ctx["work"]
    labels = np.loadtxt(src / "roi" / f"{h}.parc.dkt31.txt", dtype=int)
    ref = read_txt(src / "quant" / f"{h}.gray.LGI.txt")
    pial = src / "surf" / f"{h}.gray.vtk"
    sc, gc = src / "curve" / f"{h}.gray.scurve.bary", src / "curve" / f"{h}.white.gcurve.bary"
    run, secs = timed(lambda: lgi_pipeline(pial, sc, gc, work / "lgi", hemi=h, n_threads=ctx["threads"]))
    m = {"seconds": secs, "hull_area": run.hull_area, "correspondence_area": run.corr_area,
         "image_range": [float(ref.min()), float(ref.max())], "ours_range": [float(run.lgi.min()), float(run.lgi.max())]}  # fmt: skip
    m["vs_image"] = q14(run.lgi, ref, labels) | vec(run.lgi, ref)
    # the port's own 1-ulp spread: perturb every pial coordinate by 0/1 float32 ulp and rerun
    rng = np.random.default_rng(22)
    surf = read_vtk(pial)
    v32 = surf.vertices.astype(np.float32)
    up = np.where(rng.random(v32.shape) < 0.5, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
    v = np.where(rng.random(v32.shape) < 0.5, np.nextafter(v32, up), v32)
    pert = type(surf)(v.astype(np.float64), surf.faces, frame=surf.frame, units=surf.units)
    pdir = work / "lgi-ulp"
    pdir.mkdir(exist_ok=True)
    write_vtk(pert, pdir / f"{h}.gray.vtk", dtype="float")
    run2, secs2 = timed(lambda: lgi_pipeline(pdir / f"{h}.gray.vtk", sc, gc, pdir, hemi=h, n_threads=ctx["threads"]))
    m["ulp_spread_vs_base"] = q14(run2.lgi, run.lgi, labels) | {"seconds": secs2}
    sp = m["ulp_spread_vs_base"]
    b = m["vs_image"]
    m["q15_proposed_bounds"] = {
        "p99_rel": max(Q14["p99_rel"], 1.5 * sp["rel_p99"]),
        "roi_mean_rel": max(Q14["roi_mean_rel"], 1.5 * sp["roi_mean_rel_max"]),
    }
    m["q15_proposed_pass"] = bool(b["pearson_r"] >= Q14["r"] and b["rel_p99"] <= m["q15_proposed_bounds"]["p99_rel"]
                                  and b["roi_mean_rel_max"] <= m["q15_proposed_bounds"]["roi_mean_rel"])  # fmt: skip
    return m


# ------------------------------------------------------------------ driver
STAGE_FN = {"input": stage_input, "reg": stage_reg, "remesh": stage_remesh, "trace": stage_trace,
            "sd": stage_sd, "lgi": stage_lgi}  # fmt: skip


def run_subject(subj, args, work_root, report):
    src = OPEN / subj
    mpath = report / f"metrics-{subj}.json"
    M = json.loads(mpath.read_text()) if mpath.exists() else {"subject": subj}
    M["template"], M["threads"] = TEMPLATE, args.threads
    for h in args.hemis:
        work = work_root / subj / h
        work.mkdir(parents=True, exist_ok=True)
        ctx = {"src": src, "job": work / "job", "work": work, "hemi": h, "threads": args.threads, "ens": args.ens}
        for st in args.stages:
            key = f"{h}/{st}"
            if key in M.get("stages", {}) and not args.force:
                continue
            print(f"[{subj} {h}] {st} ...", flush=True)
            t = time.time()
            if st != "input":
                prepare_fs_subject(src, ctx["job"], hemis=(h,))
            try:
                res = STAGE_FN[st](ctx)
            except Exception as exc:  # keep going; the error is part of the record
                import traceback

                traceback.print_exc()
                res = {"error": f"{type(exc).__name__}: {exc}"}
            res["wall_seconds"] = time.time() - t
            M.setdefault("stages", {})[key] = res
            jdump(M, mpath)
            print(f"[{subj} {h}] {st} done in {res['wall_seconds']:.0f} s", flush=True)
    return M


def fmt(x, p=3):
    return f"{x:.{p}g}" if isinstance(x, float) else str(x)


def summary(report):
    rows = ["| Stage | Subject | Hemi | Result | Rule | Verdict |", "|---|---|---|---|---|---|"]
    for mp in sorted(report.glob("metrics-*.json")):
        M = json.loads(mp.read_text())
        for key, r in M.get("stages", {}).items():
            h, st = key.split("/")
            subj = M["subject"]
            if "error" in r:
                rows.append(f"| {st} | {subj} | {h} | error: {r['error']} | - | FAIL |")
                continue
            if st == "input":
                rows.append(f"| input VTK/mask/thickness | {subj} | {h} | all identical: {r['pass']}; mid.area max abs {fmt(r['mid_area']['max_abs'])} | byte identity | {'PASS' if r['pass'] else 'FAIL'} |")
            elif st == "reg":
                p = r.get("port", {})
                if "sphere" not in p:
                    rows.append(f"| HSD chain | {subj} | {h} | failed: {p.get('failed')} | - | FAIL |")
                    continue
                v = r["verdict"]
                rows.append(f"| HSD sphere (native, deformed) | {subj} | {h} | median {fmt(p['sphere']['median_deg'])} deg, p99 {fmt(p['sphere']['p99_deg'])} deg; own spread median {fmt(r['port_own_spread']['median_deg'])}, p99 {fmt(r['port_own_spread']['p99_deg'])} | policy 0.01/0.1 deg; Q15 proposed {fmt(v['q15_proposed_bounds']['sphere_median_deg'])}/{fmt(v['q15_proposed_bounds']['sphere_p99_deg'])} | policy {'PASS' if v['policy']['sphere'] else 'FAIL'}; Q15 {'PASS' if v['q15_proposed']['sphere'] else 'FAIL'} |")
                rows.append(f"| HSD coefficients | {subj} | {h} | max abs {fmt(p['coeff_max_abs'])}; own spread {fmt(r['port_own_spread']['coeff_abs'])} | policy 1e-4; Q15 {fmt(v['q15_proposed_bounds']['coeff_abs'])} | policy {'PASS' if v['policy']['coeff'] else 'FAIL'}; Q15 {'PASS' if v['q15_proposed']['coeff'] else 'FAIL'} |")
                rows.append(f"| HSD energy E1 (PDF) | {subj} | {h} | image {fmt(r['E1_image'])}, ours {fmt(p['E1'])} (rel {fmt(p['E1_rel_diff_vs_image'])}); own spread {fmt(r['port_own_spread']['E1_rel'])} | policy 1e-3; Q15 {fmt(v['q15_proposed_bounds']['energy_rel'])} | policy {'PASS' if v['policy']['energy'] else 'FAIL'}; Q15 {'PASS' if v['q15_proposed']['energy'] else 'FAIL'} |")
                raw = r.get("port_raw_icomesh_guess", {})
                if "sphere" in raw:
                    rows.append(f"| HSD with upstream raw icomesh guess | {subj} | {h} | sphere median {fmt(raw['sphere']['median_deg'])} deg, E1 {fmt(raw['E1'])} | (documents the Q17 fix) | - |")
            elif st == "remesh":
                worst = max(s["max_abs"] for s in r["surfaces"].values())
                bad = [n for n, v in r["maps"].items() if not v["e_class"]]
                rows.append(f"| SphericalRemesh (image coeff) surfaces + stat maps | {subj} | {h} | surf max abs {fmt(worst)} mm; maps not E-class: {bad or 'none'} | E | {'PASS' if r['pass'] else 'FAIL'} |")
            elif st == "trace":
                ids = sum(f["identical"] for f in r["files"].values())
                rows.append(f"| TRACE | {subj} | {h} | {ids}/4 files byte-identical; Jaccard sulcal {fmt(r['sulcal_vertex_jaccard'])}, gyral {fmt(r['gyral_vertex_jaccard'])} | E (Jaccard >= 0.99) | {'PASS' if r['pass'] else 'FAIL'} |")
            elif st == "sd":
                rows.append(f"| Sulcal depth | {subj} | {h} | r {fmt(r['pearson_r'], 5)}, bias {fmt(r['mean_diff'])} mm, max abs {fmt(r['max_abs'])} mm, exact {fmt(r['frac_exact'])} | E: r >= 0.9999, max/range <= 1e-4 | {'PASS' if r['e_class'] else 'FAIL'} |")
            elif st == "lgi":
                b = r["vs_image"]
                rows.append(f"| LGI (klaplace stand-in) | {subj} | {h} | r {fmt(b['pearson_r'], 5)}, rel p99 {fmt(b['rel_p99'])}, ROI-mean rel {fmt(b['roi_mean_rel_max'])}; own spread p99 {fmt(r['ulp_spread_vs_base']['rel_p99'])} | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 {'PASS' if b['q14_pass'] else 'FAIL'}; Q15 {'PASS' if r['q15_proposed_pass'] else 'FAIL'} |")
    text = "\n".join(rows) + "\n"
    (report / "summary.md").write_text(text)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workdir")
    p.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    p.add_argument("--hemis", nargs="+", default=["lh", "rh"], choices=["lh", "rh"])
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--ens", type=int, default=3, help="number of 1-ulp HSD chain reruns per hemisphere (max 6)")
    p.add_argument("--report", default=str(REPO / "docs" / "reports" / "open-examples-2026-10-22"))
    p.add_argument("--force", action="store_true", help="rerun stages that are already cached")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        import numba

        numba.set_num_threads(max(1, min(args.threads, numba.config.NUMBA_NUM_THREADS)))
        for s in args.subjects:
            run_subject(s, args, Path(args.workdir), report)
    print(summary(report))


if __name__ == "__main__":
    main()
