# ruff: noqa: E501
"""``CurveExtraction`` driver, writers and CLI (``Extraction.cpp``, ``wrapper/{main,PARSE_ARGS}``).

``pybiscuit trace`` takes the original flags (``-i -o --sulcus --gyrus --cart --bary --noVtk
--interm --nojunction --gyralSeed --sulcalSeed --epRad --nhDist --lsTh --prune --simplification
-s/--smoothIter --smoothTensorIter --nThreads --inputPoints``); ``--nThreads`` is accepted and
ignored (single-threaded, deterministic).  Output files: ``<o>.scurve`` / ``.gcurve`` (vertex ids
per curve, one curve per line, each id followed by a space), ``.scurve.vtk`` / ``.gcurve.vtk``,
``.cart`` (``%f %f %f`` per path point), ``.bary`` (``%d %d %f``), ``.spoint`` / ``.gpoint``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..io import read_vtk
from ..mesh.geodesic import GeodesicPath, GeodesicSolver
from ..mesh.geometry import smooth_vertices
from .curves import Curves
from .points import detect_points, point_fields, seed_mask

__all__ = ["TraceResult", "extract_curves", "main"]

TRACE_VERSION = "v1.1.2"


@dataclass
class TraceResult:
    curves: dict[str, Curves] = field(default_factory=dict)  # "sulcal" / "gyral"
    points: dict[str, np.ndarray] = field(default_factory=dict)  # detected point masks
    vertices: np.ndarray | None = None  # float32 (after --smoothIter)
    faces: np.ndarray | None = None


def extract_curves(
    vertices, faces, *, sulcus=True, gyrus=False, sulcal_seed=-0.05, gyral_seed=0.03,
    ep_rad=4.0, nh_dist=4.0, ls_threshold=2.5, prune=5.0, smooth_iter=0, smooth_tensor_iter=3,
    simplification=False, input_points=None, verbose=False,
) -> TraceResult:  # fmt: skip
    """Run TRACE on a surface.  ``input_points`` maps ``"sulcal"`` / ``"gyral"`` to boolean masks
    that replace the point detection (``--inputPoints``)."""
    if not sulcus and not gyrus:
        raise ValueError("Specify at least one extraction type! --sulcus / --gyrus")
    if simplification and sulcus and gyrus:
        raise RuntimeError(
            "--simplification with --sulcus and --gyrus dereferences a NULL GyralCurve in the "
            "original (Extraction.cpp L136-L139) and crashes; run the two extractions separately"
        )
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if smooth_iter > 0:
        v = smooth_vertices(v, f, int(smooth_iter))
    res = TraceResult(vertices=v, faces=f)
    solver = GeodesicSolver(v, f)
    for name, enabled, gyral, seed in (
        ("sulcal", sulcus, False, sulcal_seed), ("gyral", gyrus, True, gyral_seed)
    ):  # fmt: skip
        if not enabled:
            continue
        fields = point_fields(v, f, smooth_tensor_iter, gyral)
        if input_points is None:
            seeds = seed_mask(fields, seed, gyral)
            if verbose:
                print(f"# of Seed Points: {int(seeds.sum())}")
            mask = detect_points(v, f, fields, seeds, ls_threshold, gyral)
        else:
            mask = np.asarray(input_points[name], dtype=bool)
        res.points[name] = mask
        cv = Curves(v, f, mask, fields.direction, gyral=gyral, solver=solver, verbose=verbose)
        cv.set_threshold(nh_dist, nh_dist, ep_rad, prune)
        cv.run()
        if simplification:
            cv.simplify(np.float32(2.5 if gyral else 4.5))
        res.curves[name] = cv
    return res


# --------------------------------------------------------------------------- writers
def write_curve_file(path, cv: Curves, include_junction: bool = True) -> None:
    """``saveSulcalCurves`` / ``saveGyralCurves``."""
    with open(path, "w") as fh:
        for row in cv.curve_vertex_lists(include_junction):
            fh.write("".join(f"{x} " for x in row) + "\n")


def write_curve_vtk(path, cv: Curves, vertices, faces) -> None:
    """``Mesh::saveFile(vtk)`` followed by the ``LINES`` / ``Labels`` block of ``saveVTK``."""
    v = np.asarray(vertices, dtype=np.float32)
    lines = ["# vtk DataFile Version 3.0", "vtk output", "ASCII", "DATASET POLYDATA",
             f"POINTS {len(v)} float"]  # fmt: skip
    lines += [" ".join(f"{float(x):.9f}" for x in r) for r in v]
    lines.append(f"POLYGONS {len(faces)} {4 * len(faces)}")
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    lines.append("")
    curves = cv.all_curves()
    n_elem = sum(len(c) for c in curves)
    lines.append(f"LINES {len(curves)} {len(curves) + n_elem}")
    lines += [f"{len(c)} " + "".join(f"{x} " for x in c) for c in curves]
    lines.append("")
    label = np.zeros(len(v), dtype=np.int64)
    for k, val in cv.junction_labels().items():
        label[k] = val
    lines.append(f"POINT_DATA {len(v)}\nSCALARS Labels int\nLOOKUP_TABLE default")
    body = "".join(
        f"{int(x)} " + ("\n" if (i + 1) % 256 == 0 else "") for i, x in enumerate(label)
    )
    text = "\n".join(lines) + "\n" + body + "\n"
    Path(path).write_text(text)


def write_geodesic_path(path, cv: Curves, barycentric: bool) -> None:
    """``saveGeodesicPath``: the full geodesic trajectory of every curve."""
    solver = cv.solver
    gp = GeodesicPath(cv.v32, cv.faces)
    out = [f"{cv.n_curves()}"]
    for lst in cv._iter():
        path_rows = []
        items = [int(cv.vid[e]) for e in lst.item]
        for i in range(len(items)):
            target = min(i + 1, len(items) - 1)
            solver.perform_front_propagation(items[i], items[target])
            pts, bary = gp.compute(items[target], solver.dist)
            src = bary if barycentric else pts
            if int(bary[-1, 0]) != items[i]:
                solver.perform_front_propagation(items[target], items[i])
                pts, bary = gp.compute(items[i], solver.dist)
                src = bary if barycentric else pts
                rows = src[: len(src) - 1]
            else:
                rows = src[len(src) - 1 : 0 : -1]
            path_rows.extend(r.copy() for r in rows)
        out.append(f"{len(path_rows)}")
        for r in path_rows:
            if barycentric:
                if r[2] == 1:
                    r[1] = r[0]
                if r[2] == 0:
                    r[0] = r[1]
                out.append(f"{int(r[0])} {int(r[1])} {float(r[2]):.6f}")
            else:
                out.append(f"{float(r[0]):.6f} {float(r[1]):.6f} {float(r[2]):.6f}")
    Path(path).write_text("\n".join(out) + "\n")


def write_point_file(path, mask) -> None:
    Path(path).write_text("".join(f"{int(b)}\n" for b in mask))


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pybiscuit trace",
        description="TRACE: Topological Graph Representation for Automatic Sulcal Curve "
        f"Extraction {TRACE_VERSION} (port of CurveExtraction; same flags)",
    )
    p.add_argument("-i", "--inputSurface", required=True)
    p.add_argument("--inputPoints")
    p.add_argument("--sulcus", action="store_true")
    p.add_argument("--gyrus", action="store_true")
    p.add_argument("-o", "--outputCurve", required=True)
    p.add_argument("--cart", action="store_true")
    p.add_argument("--bary", action="store_true")
    p.add_argument("--noVtk", action="store_true")
    p.add_argument("--interm", action="store_true")
    p.add_argument("--nojunction", action="store_true")
    p.add_argument("--gyralSeed", type=float, default=0.03)
    p.add_argument("--sulcalSeed", type=float, default=-0.05)
    p.add_argument("--epRad", type=float, default=4.0)
    p.add_argument("--nhDist", type=float, default=4.0)
    p.add_argument("--lsTh", type=float, default=2.5)
    p.add_argument("--prune", type=float, default=5.0)
    p.add_argument("--simplification", action="store_true")
    p.add_argument("-s", "--smoothIter", type=int, default=0)
    p.add_argument("--smoothTensorIter", type=int, default=3)
    p.add_argument("--nThreads", type=int, default=0, help="accepted for compatibility; ignored")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if not a.sulcus and not a.gyrus:
        print("Specify at least one extraction type! --sulcus / --gyrus")
        return -1
    mesh = read_vtk(a.inputSurface)
    out = a.outputCurve
    ip = None
    if a.inputPoints:
        ip = {}
        if a.sulcus:
            ip["sulcal"] = np.loadtxt(f"{a.inputPoints}.spoint", dtype=int).astype(bool)
        if a.gyrus:
            ip["gyral"] = np.loadtxt(f"{a.inputPoints}.gpoint", dtype=int).astype(bool)
    res = extract_curves(
        mesh.vertices, mesh.faces, sulcus=a.sulcus, gyrus=a.gyrus, sulcal_seed=a.sulcalSeed,
        gyral_seed=a.gyralSeed, ep_rad=a.epRad, nh_dist=a.nhDist, ls_threshold=a.lsTh,
        prune=a.prune, smooth_iter=a.smoothIter, smooth_tensor_iter=a.smoothTensorIter,
        simplification=a.simplification, input_points=ip, verbose=True,
    )  # fmt: skip
    for name, ext in (("sulcal", "scurve"), ("gyral", "gcurve")):
        if name not in res.curves:
            continue
        cv = res.curves[name]
        info = cv.info()
        print(f"# of the {name} points: {info['points']}\n# of the connected points: "
              f"{info['connected']}\n# of the curves: {info['curves']}\n# of the junction "
              f"points: {info['junctions']}")  # fmt: skip
        if a.interm and ip is None:
            write_point_file(f"{out}.{'spoint' if name == 'sulcal' else 'gpoint'}", res.points[name])
        write_curve_file(f"{out}.{ext}", cv, not a.nojunction)
        if not a.noVtk:
            write_curve_vtk(f"{out}.{ext}.vtk", cv, res.vertices, res.faces)
        if a.cart:
            write_geodesic_path(f"{out}.{ext}.cart", cv, False)
        if a.bary:
            write_geodesic_path(f"{out}.{ext}.bary", cv, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
