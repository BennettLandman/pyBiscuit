#!/usr/bin/env python
"""End-to-end comparison of a ``pybiscuit run --legacy`` output with the image's outputs (B11).

    python scripts/validation/e2e_open_examples.py RUN_DIR IMAGE_SUBJECT_DIR [--json OUT.json]
    python scripts/validation/e2e_open_examples.py --summary OUT1.json OUT2.json ...

``RUN_DIR`` is the output of ``pybiscuit run <subject> RUN_DIR --legacy`` (its ``legacy/`` tree is
compared); ``IMAGE_SUBJECT_DIR`` holds the ``biscuit_FC_v2.2.sif`` products (``surf/ reg/ roi/
curve/ quant/ stat/``), e.g. one ``open-examples`` subject.  Unlike ``open_examples.py`` (which
feeds every stage the image's own inputs) each stage here consumes the port's upstream results,
so differences accumulate along the chain as they would for a user.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pybiscuit.io import read_vtk  # noqa: E402
from pybiscuit.quant.lgi import curve_vertex_ids  # noqa: E402

HEMIS = ("lh", "rh")
PROTOCOLS = ("braincolor", "dkt31")


def txt(p):
    return np.loadtxt(p, dtype=np.float64)


def vec(ours, ref):
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    if ours.shape != ref.shape:
        return {"shape_ours": list(ours.shape), "shape_ref": list(ref.shape)}
    d = np.abs(ours - ref)
    lo, hi = np.percentile(ref, [1, 99])
    rng = float(hi - lo) or 1.0
    return {
        "pearson_r": float(np.corrcoef(ours, ref)[0, 1]) if ref.std() and ours.std() else None,
        "max_abs": float(d.max()), "p99_abs": float(np.percentile(d, 99)),
        "mean_diff": float((ours - ref).mean()), "p99_abs_over_range": float(np.percentile(d, 99) / rng),
        "frac_exact_6dp": float((d < 5e-7 + 1e-12).mean()),
    }  # fmt: skip


def same_bytes(a, b):
    return a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()


def sphere_angles(a, b):
    ua = a / np.linalg.norm(a, axis=1, keepdims=True)
    ub = b / np.linalg.norm(b, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((ua * ub).sum(1), -1, 1)))
    return {"median_deg": float(np.median(ang)), "p99_deg": float(np.percentile(ang, 99)),
            "max_deg": float(ang.max())}  # fmt: skip


def surface(ours, ref, *, sphere=False):
    a, b = read_vtk(ours), read_vtk(ref)
    m = {"byte_identical": same_bytes(ours, ref), "faces_identical": bool(np.array_equal(a.faces, b.faces))}
    if a.vertices.shape == b.vertices.shape:
        m["vertex_max_abs"] = float(np.abs(a.vertices - b.vertices).max())
        if sphere:
            m |= sphere_angles(a.vertices, b.vertices)
    return m


def labels(ours, ref):
    a, b = np.loadtxt(ours, dtype=int), np.loadtxt(ref, dtype=int)
    dice = []
    for k in np.unique(b[b > 0]):
        x, y = a == k, b == k
        dice.append(2 * (x & y).sum() / (x.sum() + y.sum()))
    return {"agreement": float((a == b).mean()), "dice_median": float(np.median(dice)),
            "dice_min": float(np.min(dice)), "n_rois": len(dice)}  # fmt: skip


def stats_csv(ours, ref):
    def load(p):
        with open(p, newline="") as fh:
            rows = list(csv.DictReader(fh))
        return {r["ID"]: r for r in rows}, list(rows[0].keys())

    a, cols = load(ours)
    b, _ = load(ref)
    out = {"rows_ours": len(a), "rows_image": len(b), "ids_identical": sorted(a) == sorted(b), "columns": {}}
    ids = [i for i in b if i in a]
    for c in cols[2:]:
        x = np.array([float(a[i][c]) for i in ids])
        y = np.array([float(b[i][c]) for i in ids])
        ok = np.isfinite(x) & np.isfinite(y)
        rel = np.abs(x[ok] - y[ok]) / np.maximum(np.abs(y[ok]), 1e-12)
        out["columns"][c] = {
            "pearson_r": float(np.corrcoef(x[ok], y[ok])[0, 1]) if y[ok].std() and x[ok].std() else None,
            "rel_median": float(np.median(rel)), "rel_max": float(rel.max()),
            "nan_mismatch": int((np.isnan(x) != np.isnan(y)).sum()),
        }  # fmt: skip
    return out


def compare(run: Path, img: Path) -> dict:
    leg = run / "legacy"
    r: dict = {"surf": {}, "reg": {}, "roi": {}, "curve": {}, "quant": {}, "stat": {}}
    for h in HEMIS:
        for s in ("white", "gray", "mid", "inflated", "sphere"):
            r["surf"][f"{h}.{s}"] = surface(leg / "surf" / f"{h}.{s}.vtk", img / "surf" / f"{h}.{s}.vtk")
        r["reg"][f"{h}.sphere.reg"] = surface(leg / "reg" / f"{h}.sphere.reg.vtk",
                                              img / "reg" / f"{h}.sphere.reg.vtk", sphere=True)  # fmt: skip
        for s in ("gray", "white", "mid"):
            r["reg"][f"{h}.{s}.reg"] = surface(leg / "reg" / f"{h}.{s}.reg.vtk", img / "reg" / f"{h}.{s}.reg.vtk")
        r["roi"][f"{h}.mask"] = {"byte_identical": same_bytes(leg / "roi" / f"{h}.mask.txt",
                                                              img / "roi" / f"{h}.mask.txt")}  # fmt: skip
        for p in PROTOCOLS:
            r["roi"][f"{h}.parc.{p}"] = labels(leg / "roi" / f"{h}.parc.{p}.txt", img / "roi" / f"{h}.parc.{p}.txt")
        for name in (f"{h}.gray.scurve", f"{h}.gray.scurve.bary", f"{h}.white.gcurve", f"{h}.white.gcurve.bary"):
            m = {"byte_identical": same_bytes(leg / "curve" / name, img / "curve" / name)}
            if not name.endswith(".bary"):
                x, y = set(curve_vertex_ids(leg / "curve" / name)), set(curve_vertex_ids(img / "curve" / name))
                m["vertex_jaccard"] = len(x & y) / max(len(x | y), 1)
            r["curve"][name] = m
        for f in ("mid.H", "mid.S", "mid.area", "thickness", "gray.SD", "gray.LGI", "mid.SCI"):
            o, i = leg / "quant" / f"{h}.{f}.txt", img / "quant" / f"{h}.{f}.txt"
            r["quant"][f"{h}.{f}"] = {"byte_identical": same_bytes(o, i)} | vec(txt(o), txt(i))
        for f in ("mid.reg.H", "mid.reg.S", "mid.reg.SCI", "mid.reg.area", "gray.reg.SD", "gray.reg.LGI",
                  "reg.thickness"):  # fmt: skip
            o, i = leg / "stat" / f"{h}.{f}.txt", img / "stat" / f"{h}.{f}.txt"
            r["stat"][f"{h}.{f}"] = {"byte_identical": same_bytes(o, i)} | vec(txt(o), txt(i))
    for p in PROTOCOLS:
        r["stat"][f"stats.{p}.csv"] = stats_csv(leg / "stat" / f"stats.{p}.csv", img / "stat" / f"stats.{p}.csv")
    return r


def _f(x, p=4):
    return "—" if x is None else f"{x:.{p}g}"


def summary(paths: list[Path]) -> str:
    """Markdown table (one row per subject × hemisphere) from saved comparison JSON files."""
    head = ("| Subject | Hemi | surf/ mask curve/ identical | sphere.reg median / p99 ° | parc agreement bc / dkt "
            "| mid H,S,area,thickness identical | SD r / bias mm | LGI r / bias | SCI r |")  # fmt: skip
    rows = [head, "|" + "---|" * 9]
    for p in paths:
        d = json.loads(Path(p).read_text())
        for h in HEMIS:
            ident = all(v["byte_identical"] for k, v in d["surf"].items() if k.startswith(h)) and all(
                v["byte_identical"] for k, v in d["curve"].items() if k.startswith(h)
            ) and d["roi"][f"{h}.mask"]["byte_identical"]
            sr = d["reg"][f"{h}.sphere.reg"]
            q = d["quant"]
            mid_ok = all(q[f"{h}.{f}"]["byte_identical"] for f in ("mid.H", "mid.S", "mid.area", "thickness"))
            rows.append(
                f"| {Path(p).name.split('.')[0]} | {h} | {'yes' if ident else 'NO'} | "
                f"{_f(sr['median_deg'], 3)} / {_f(sr['p99_deg'], 3)} | "
                f"{_f(d['roi'][f'{h}.parc.braincolor']['agreement'])} / "
                f"{_f(d['roi'][f'{h}.parc.dkt31']['agreement'])} | {'yes' if mid_ok else 'NO'} | "
                f"{_f(q[f'{h}.gray.SD']['pearson_r'], 5)} / {_f(q[f'{h}.gray.SD']['mean_diff'], 3)} | "
                f"{_f(q[f'{h}.gray.LGI']['pearson_r'])} / {_f(q[f'{h}.gray.LGI']['mean_diff'], 3)} | "
                f"{_f(q[f'{h}.mid.SCI']['pearson_r'])} |"
            )
    return "\n".join(rows)


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--summary":
        print(summary([Path(p) for p in argv[1:]]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    res = compare(a.run, a.image)
    text = json.dumps(res, indent=1, default=float)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
