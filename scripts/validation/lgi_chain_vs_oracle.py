#!/usr/bin/env python
"""TRACE -> LGI chain: port vs oracle chain on one surface (plan row B20.2; not in CI).

    python scripts/validation/lgi_chain_vs_oracle.py SURFACE.vtk WORKDIR [threads=4] [metrics.json]

``SURFACE.vtk`` is used as both ``lh.gray.vtk`` (sulci) and ``lh.white.vtk`` (gyri) -- the template
only ships white surfaces.  Both chains run BISCUIT's calls (spec §8, §9.3) on identical inputs:

* port:   ``pybiscuit.trace.pipeline`` -> ``pybiscuit.quant.lgi_pipeline``
* oracle: no-FMA ``CurveExtraction`` x2 -> no-FMA ``Gyrification``

The hull (``OuterHull`` port) and the correspondence stand-in are computed once and shared, so the
comparison isolates TRACE and Gyrification.  Prints and writes metrics (curve identity, curve
vertex Jaccard, LGI per-vertex agreement vs docs/validation-tolerances.md).
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pybiscuit.io import read_vtk, write_vtk  # noqa: E402
from pybiscuit.quant.hull import outer_hull  # noqa: E402
from pybiscuit.quant.lgi import curve_vertex_ids, read_curves  # noqa: E402
from pybiscuit.quant.lgi_pipeline import (  # noqa: E402
    gyrification_argv,
    hull_correspondence,
    lgi_pipeline,
    mesh_area,
)
from pybiscuit.trace.pipeline import curve_argv, curve_extract_pipeline  # noqa: E402

O = Path.home() / "dev" / "biscuit" / "oracles"  # noqa: E741
CE = O / "curve_extraction-nofma" / "bin" / "CurveExtraction"
GYR = O / "gyrification-nofma" / "bin" / "Gyrification"
CURVES = ("lh.gray.scurve", "lh.gray.scurve.bary", "lh.white.gcurve", "lh.white.gcurve.bary")


def timed(fn):
    t = time.time()
    out = fn()
    return out, time.time() - t


def main():
    surface, work = Path(sys.argv[1]), Path(sys.argv[2])
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    metrics_path = Path(sys.argv[4]) if len(sys.argv) > 4 else work / "metrics.json"
    work.mkdir(parents=True, exist_ok=True)
    for f in ("lh.gray.vtk", "lh.white.vtk"):
        shutil.copy(surface, work / f)
    pial = work / "lh.gray.vtk"
    m = {"surface": surface.name, "threads": threads}
    surf = read_vtk(pial)
    m["vertices"] = surf.n_vertices

    hull, t = timed(lambda: outer_hull(surf))
    corr = hull_correspondence(surf, hull)
    write_vtk(corr, work / "lh.gray.hull.corr.vtk", dtype="float")
    m.update(hull_seconds=t, hull_area=mesh_area(hull), corr_area=mesh_area(corr),
             surface_area=mesh_area(surf))  # fmt: skip
    print(f"hull {t:.0f} s; hull area {m['hull_area']:.0f}, surface area {m['surface_area']:.0f}")

    # ---- oracle chain
    oc = work / "oracle" / "curve"
    oc.mkdir(parents=True, exist_ok=True)

    def oracle_curves():
        for kind, stem in (("sulcus", "gray"), ("gyrus", "white")):
            argv = curve_argv(kind, work / f"lh.{stem}.vtk", oc / f"lh.{stem}", n_threads=threads)
            subprocess.run([str(CE), *argv], check=True, capture_output=True, cwd=oc)

    _, m["oracle_trace_seconds"] = timed(oracle_curves)
    print(f"oracle TRACE {m['oracle_trace_seconds']:.0f} s")

    # ---- port chain
    pc = work / "port" / "curve"
    _, m["port_trace_seconds"] = timed(
        lambda: curve_extract_pipeline(work, pc, hemis=("lh",), n_threads=threads, force=True)
    )
    print(f"port TRACE {m['port_trace_seconds']:.0f} s")
    same = {n: (pc / n).read_bytes() == (oc / n).read_bytes() for n in CURVES}
    m["curve_files_identical"] = same
    a = curve_vertex_ids(*[read_curves(pc / f) for f in ("lh.gray.scurve", "lh.white.gcurve")])
    b = curve_vertex_ids(*[read_curves(oc / f) for f in ("lh.gray.scurve", "lh.white.gcurve")])
    m["curve_vertices_port"], m["curve_vertices_oracle"] = len(a), len(b)
    m["curve_vertex_jaccard"] = len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
    print("curve files identical:", same, "jaccard", m["curve_vertex_jaccard"])

    def run_oracle_gyr():
        argv = gyrification_argv(
            pial, work / "lh.gray.hull.corr.vtk", oc / "lh.gray.scurve.bary",
            oc / "lh.white.gcurve.bary", work / "oracle" / "lh.gray", n_threads=threads,
        )  # fmt: skip
        subprocess.run([str(GYR), *argv], check=True, capture_output=True, cwd=work / "oracle")
        return np.loadtxt(next((work / "oracle").glob("lh.gray.lgi.map.*.txt")))

    gold, m["oracle_gyrification_seconds"] = timed(run_oracle_gyr)
    run, m["port_gyrification_seconds"] = timed(
        lambda: lgi_pipeline(
            pial, pc / "lh.gray.scurve.bary", pc / "lh.white.gcurve.bary", work / "port",
            outer=corr, n_threads=threads,
        )  # fmt: skip
    )
    lgi = run.lgi
    scale = np.abs(gold).max()
    rel = np.abs(lgi - gold) / np.abs(gold)
    m.update(
        lgi_median=float(np.median(gold)), lgi_min=float(gold.min()), lgi_max=float(gold.max()),
        lgi_max_abs_diff_over_max=float(np.abs(lgi - gold).max() / scale),
        lgi_max_rel=float(rel.max()), lgi_p99_rel=float(np.percentile(rel, 99)),
        lgi_n_rel_gt_1e4=int((rel > 1e-4).sum()), lgi_pearson_r=float(np.corrcoef(lgi, gold)[0, 1]),
        lgi_pass_E=bool(rel.max() <= 1e-4 and np.corrcoef(lgi, gold)[0, 1] >= 0.9999),
    )  # fmt: skip
    print(json.dumps(m, indent=1))
    metrics_path.write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
