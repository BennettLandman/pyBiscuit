#!/usr/bin/env python
"""LGI port vs the numbers the image's ``lgi`` script prints (B9.2, B9.3, B9.4).

    python scripts/validation/lgi_logs_vs_image.py LOG_DIR IMAGE_TMP_LGI_DIR IMAGE_SUBJECT_DIR RUN_DIR \
        [--json OUT.json]

``LOG_DIR`` holds the image's ``<subject>_<h>.lgi.log`` (stdout of ``lgi`` with ``$TMP`` kept);
the other arguments are as in ``lgi_intermediates_vs_image.py``; the subject id is the name of
``IMAGE_SUBJECT_DIR``.  Per hemisphere it checks

* the logged ``klaplace`` and ``Gyrification`` flags against the port's defaults;
* ``klaplace``'s ``Grid Spacing`` and ``# of seed points`` against the image's and the port's hull;
* ``Gyrification``'s ``Adjusted ratio`` and ``outer hull regions corrected`` against the port's area
  step on the image's correspondence mesh and on the port's own (``RUN_DIR``).
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pybiscuit.io import read_vtk  # noqa: E402
from pybiscuit.mesh.geometry import vertex_areas, vertex_neighbor_lists  # noqa: E402
from pybiscuit.quant import lgi  # noqa: E402
from pybiscuit.quant.klaplace import surface_correspondence  # noqa: E402

HEMIS = ("lh", "rh")


def _num(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def parse_log(text: str) -> dict:
    kl = re.search(r"/klaplace (-thread \d+ -dims (\d+)) ", text)
    gy = re.search(r"/Gyrification .* -m (\S+) --speed (\S+) --refHullArea (\S+)", text)
    return {
        "dims": int(kl.group(2)), "max_area": float(gy.group(1)), "speed": float(gy.group(2)),
        "ref_hull_area": float(gy.group(3)),
        "grid_spacing": _num(r"Grid Spacing: (\S+)", text),
        "seed_points": _num(r"# of seed points: (\d+)", text, int),
        "regions_corrected": _num(r"Total (\d+) outer hull regions corrected", text, int),
        "adjusted_ratio": _num(r"Adjusted ratio: (\S+)", text),
    }  # fmt: skip


def area_step(surf, hull, ref_hull_area):
    """``gyrification``'s kernel-area step (``lgi.py``, ``a1``/``a2`` and ``ratio``)."""
    a1 = vertex_areas(surf.vertices, surf.faces, vertex_neighbor_lists(surf.faces, surf.n_vertices))
    a2 = vertex_areas(hull.vertices, hull.faces, vertex_neighbor_lists(hull.faces, hull.n_vertices))
    defects = a1[: a2.size] < a2
    a2 = np.where(defects, a1[: a2.size], a2)
    return lgi._seq_sum(a2) / ref_hull_area, int(defects.sum())


def grid_spacing(hull, dims):
    """``klaplace``'s spacing: longest side of the hull's bounding box / ``dims``."""
    return float(np.ptp(hull.vertices, axis=0).max() / dims)


def compare(logs: Path, img_tmp: Path, img: Path, run: Path) -> dict:
    subject = img.name
    (quant,) = sorted((run / "stages" / "quant").glob("attempt-*"))[-1:]
    g_def = inspect.signature(lgi.gyrification).parameters
    k_def = inspect.signature(surface_correspondence).parameters
    out: dict = {}
    for h in HEMIS:
        log = parse_log((logs / f"{subject}_{h}.lgi.log").read_text())
        surf = read_vtk(img / "surf" / f"{h}.gray.vtk")
        it, ot = img_tmp / f"tmp-lgi-{h}.gray", quant / f"tmp-lgi-{h}.gray"
        ratio_i, n_i = area_step(surf, read_vtk(it / f"{h}.gray.hull.corr.vtk"), log["ref_hull_area"])
        ratio_o, n_o = area_step(surf, read_vtk(ot / f"{h}.gray.hull.corr.vtk"), log["ref_hull_area"])
        out[h] = {
            "log": log,
            "flags_match_port_defaults": (
                log["dims"] == k_def["dims"].default
                and log["max_area"] == g_def["max_area"].default
                and log["speed"] == g_def["speed"].default
                and log["ref_hull_area"] == g_def["ref_hull_area"].default
            ),
            "seed_points_match": log["seed_points"] == surf.n_vertices,
            "grid_spacing": {"image_hull": grid_spacing(read_vtk(it / f"{h}.gray.hull.vtk"), log["dims"]),
                             "port_hull": grid_spacing(read_vtk(ot / f"{h}.gray.hull.vtk"), log["dims"])},
            "on_image_corr": {"adjusted_ratio": ratio_i, "regions_corrected": n_i},
            "port_chain": {"adjusted_ratio": ratio_o, "regions_corrected": n_o,
                           "ratio_rel": ratio_o / log["adjusted_ratio"] - 1},
        }  # fmt: skip
    return {"subject": subject, "hemispheres": out}


def summary_line(subject, h, r) -> str:
    log, oi, pc, gs = r["log"], r["on_image_corr"], r["port_chain"], r["grid_spacing"]
    return (
        f"| {subject} | {h} | {'yes' if r['flags_match_port_defaults'] else 'NO'} "
        f"| {log['adjusted_ratio']:.6f} / {log['regions_corrected']} "
        f"| {oi['adjusted_ratio']:.6f} / {oi['regions_corrected']} "
        f"| {pc['adjusted_ratio']:.6f} / {pc['regions_corrected']} ({pc['ratio_rel'] * 100:+.2f} %) "
        f"| {log['grid_spacing']:g} / {gs['image_hull']:.6g} / {gs['port_hull']:.6g} |"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", type=Path)
    ap.add_argument("image_tmp_lgi", type=Path)
    ap.add_argument("image_subject", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    res = compare(a.logs, a.image_tmp_lgi, a.image_subject, a.run)
    for h, r in res["hemispheres"].items():
        print(summary_line(res["subject"], h, r))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
