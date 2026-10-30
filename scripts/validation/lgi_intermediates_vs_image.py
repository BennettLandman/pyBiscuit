#!/usr/bin/env python
"""LGI chain vs the image's ``tmp-lgi-<h>.gray`` intermediates (B9.2, B9.3, B9.4, Q14).

    python scripts/validation/lgi_intermediates_vs_image.py IMAGE_TMP_LGI_DIR IMAGE_SUBJECT_DIR RUN_DIR \
        [--json OUT.json] [--threads N] [--no-rerun]
    python scripts/validation/lgi_intermediates_vs_image.py --summary OUT1.json ...

``IMAGE_TMP_LGI_DIR`` holds ``tmp-lgi-{lh,rh}.gray/`` as the image's ``lgi`` script left them
(``<h>.gray.hull.vtk``, ``<h>.gray.hull.corr.vtk``, the two ``.bary`` files); ``IMAGE_SUBJECT_DIR``
the image's products (``surf/``, ``curve/``, ``quant/``); ``RUN_DIR`` a ``pybiscuit run`` whose quant
stage kept its own ``tmp-lgi-*``.  Per hemisphere it compares each intermediate, then isolates the
stages by feeding the image's intermediates to the port (unless ``--no-rerun``):

* ``gyr_on_image_corr``: Gyrification on the image's correspondence mesh -> LGI vs the image's map;
* ``klaplace_on_image_hull``: klaplace on the image's hull -> correspondence vs the image's.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pybiscuit.io import read_vtk  # noqa: E402
from pybiscuit.quant.klaplace import surface_correspondence  # noqa: E402
from pybiscuit.quant.lgi import gyrification  # noqa: E402
from pybiscuit.quant.lgi_pipeline import mesh_area  # noqa: E402

HEMIS = ("lh", "rh")


def _pct(d):
    return {"p50": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max())}


def lgi_metrics(ours, ref):
    d = ours - ref
    return {"pearson_r": float(np.corrcoef(ours, ref)[0, 1]), "mean_diff": float(d.mean()),
            "mean_rel": float(d.mean() / ref.mean()), "p99_abs": float(np.percentile(np.abs(d), 99)),
            "max_abs": float(np.abs(d).max())}  # fmt: skip


def corr_metrics(a, b):
    return {"area_ours": mesh_area(a), "area_image": mesh_area(b),
            "vertex_dist": _pct(np.linalg.norm(a.vertices - b.vertices, axis=1))}  # fmt: skip


def compare(img_tmp: Path, img: Path, run: Path, *, threads: int, rerun: bool) -> dict:
    (quant,) = sorted((run / "stages" / "quant").glob("attempt-*"))[-1:]
    out: dict = {}
    for h in HEMIS:
        it, ot = img_tmp / f"tmp-lgi-{h}.gray", quant / f"tmp-lgi-{h}.gray"
        r: dict = {}
        r["bary_identical"] = {
            n: filecmp.cmp(it / f"{h}.{n}.bary", img / "curve" / f"{h}.{n}.bary", shallow=False)
            and filecmp.cmp(it / f"{h}.{n}.bary", quant / f"{h}.{n}.bary", shallow=False)
            for n in ("gray.scurve", "white.gcurve")
        }
        ih, oh = read_vtk(it / f"{h}.gray.hull.vtk"), read_vtk(ot / f"{h}.gray.hull.vtk")
        r["hull"] = {
            "n_vertices_ours": oh.n_vertices, "n_vertices_image": ih.n_vertices,
            "area_ours": mesh_area(oh), "area_image": mesh_area(ih),
            "image_to_ours": _pct(cKDTree(oh.vertices).query(ih.vertices)[0]),
            "ours_to_image": _pct(cKDTree(ih.vertices).query(oh.vertices)[0]),
        }  # fmt: skip
        ic = read_vtk(it / f"{h}.gray.hull.corr.vtk")
        r["corr"] = corr_metrics(read_vtk(ot / f"{h}.gray.hull.corr.vtk"), ic)
        ref = np.loadtxt(img / "quant" / f"{h}.gray.LGI.txt")
        r["lgi_chain"] = lgi_metrics(np.loadtxt(quant / f"{h}.gray.LGI.txt"), ref)
        if rerun:
            pial = read_vtk(img / "surf" / f"{h}.gray.vtk")
            res = gyrification(
                pial, ic, sulcus=it / f"{h}.gray.scurve.bary", gyrus=it / f"{h}.white.gcurve.bary",
                max_area=316.0, speed=0.2, ref_hull_area=77100.0, n_threads=threads,
            )  # fmt: skip
            r["gyr_on_image_corr"] = lgi_metrics(res.lgi[:, 0], ref)
            r["klaplace_on_image_hull"] = corr_metrics(surface_correspondence(pial, ih), ic)
        out[h] = r
        print(f"{h}: {json.dumps(r)}", flush=True)
    return out


def summary(paths) -> str:
    rows = [
        "| Subject | Hemi | hull area ours/image | hull p50 / p99 mm | corr p50 / p99 / max mm | LGI chain r / rel "
        "| Gyrification on image corr r / rel / max abs | klaplace on image hull p50 / p99 / max mm |",
        "|" + "---|" * 8,
    ]
    for p in paths:
        d = json.loads(Path(p).read_text())
        for h in HEMIS:
            r = d[h]
            hu, c, lc = r["hull"], r["corr"]["vertex_dist"], r["lgi_chain"]
            g, k = r.get("gyr_on_image_corr"), r.get("klaplace_on_image_hull")
            rows.append(
                f"| {Path(p).name.split('.')[0]} | {h} | {hu['area_ours'] / hu['area_image']:.4f} | "
                f"{hu['image_to_ours']['p50']:.3f} / {hu['image_to_ours']['p99']:.2f} | "
                f"{c['p50']:.3f} / {c['p99']:.2f} / {c['max']:.1f} | "
                f"{lc['pearson_r']:.4f} / {100 * lc['mean_rel']:+.2f} % | "
                + (f"{g['pearson_r']:.6f} / {100 * g['mean_rel']:+.3f} % / {g['max_abs']:.2g} | " if g else "— | ")
                + (f"{k['vertex_dist']['p50']:.3f} / {k['vertex_dist']['p99']:.2f} / {k['vertex_dist']['max']:.1f} |"
                   if k else "— |")
            )  # fmt: skip
    return "\n".join(rows)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--summary":
        print(summary(argv[1:]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("image_tmp", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-rerun", action="store_true")
    a = ap.parse_args(argv)
    res = compare(a.image_tmp, a.image, a.run, threads=a.threads, rerun=not a.no_rerun)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
