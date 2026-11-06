"""Surface summary: counts, topology, geometry and per-vertex arrays (``pybiscuit surf-info``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .model import Mesh
from .topology import topology_summary, unique_edges


def surface_info(mesh: Mesh) -> dict:
    """Return a JSON-serialisable summary of ``mesh``.

    Keys: ``counts`` (vertices, faces, edges), ``topology`` (Euler characteristic, genus,
    components, boundary loops, manifold/closed/orientation flags), ``geometry``
    (bounding box, centroid, area, edge lengths, degenerate faces and, for closed
    surfaces, enclosed volume and normal direction) and ``point_data`` (per-array
    dtype, shape and range). ``frame``, ``units`` and GIfTI metadata are copied over.
    """
    v, f = mesh.vertices, mesh.faces
    top = topology_summary(mesh)
    out: dict = {
        "format": mesh.meta.get("format"),
        "frame": mesh.frame,
        "units": mesh.units,
        "counts": {"vertices": top.n_vertices, "faces": top.n_faces, "edges": top.n_edges},
        "topology": {
            "euler_characteristic": top.euler_characteristic,
            "genus": top.genus,
            "components": top.n_components,
            "boundary_loops": top.n_boundary_loops,
            "boundary_edges": top.n_boundary_edges,
            "non_manifold_edges": top.n_non_manifold_edges,
            "non_manifold_vertices": top.n_non_manifold_vertices,
            "unreferenced_vertices": top.n_unreferenced_vertices,
            "closed": top.is_closed,
            "manifold": top.is_manifold,
            "consistently_oriented": top.is_consistently_oriented,
        },
    }
    if "gifti" in mesh.meta:
        out["gifti"] = mesh.meta["gifti"]
    geo: dict = {}
    if len(v):
        lo, hi = v.min(axis=0), v.max(axis=0)
        geo.update(
            bbox_min=lo.tolist(), bbox_max=hi.tolist(), extent=(hi - lo).tolist(),
            centroid=v.mean(axis=0).tolist(),
        )  # fmt: skip
    if len(f):
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        e = unique_edges(mesh).edges
        el = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
        geo.update(
            area=float(area.sum()),
            degenerate_faces=int((area <= 0).sum()),
            edge_length={"min": float(el.min()), "mean": float(el.mean()), "max": float(el.max())},
        )
        if top.is_closed_manifold and top.is_consistently_oriented:
            vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
            geo["volume"] = abs(vol)
            geo["normals"] = "outward" if vol > 0 else "inward"
    out["geometry"] = geo
    pdata = {}
    for name, arr in mesh.point_data.items():
        arr = np.asarray(arr)
        d: dict = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
        if arr.size and arr.dtype.kind in "biuf":
            fin = np.isfinite(arr) if arr.dtype.kind == "f" else np.ones(arr.shape, bool)
            if fin.any():
                d.update(min=float(arr[fin].min()), max=float(arr[fin].max()),
                         mean=float(arr[fin].mean()))  # fmt: skip
            d["non_finite"] = int((~fin).sum())
        pdata[name] = d
    out["point_data"] = pdata
    return out


def gifti_data_info(path) -> dict:
    """Summary of a GIfTI file without geometry (``*.func.gii``, ``*.label.gii``, ...)."""
    import nibabel as nib
    from nibabel import gifti

    img = nib.load(str(path))
    arrays = []
    for i, da in enumerate(img.darrays):
        arr = np.asarray(da.data)
        d = {
            "name": (da.meta or {}).get("Name") or f"array{i}",
            "intent": gifti.gifti.intent_codes.niistring[da.intent] if da.intent else "",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
        if arr.size and arr.dtype.kind in "biuf":
            fin = np.isfinite(arr) if arr.dtype.kind == "f" else np.ones(arr.shape, bool)
            if fin.any():
                d.update(min=float(arr[fin].min()), max=float(arr[fin].max()))
        arrays.append(d)
    out: dict = {"format": "gifti", "kind": "data", "meta": dict(img.meta or {}), "arrays": arrays}
    if img.labeltable is not None and img.labeltable.labels:
        out["labels"] = {int(lb.key): lb.label for lb in img.labeltable.labels}
    return out


def describe(path, *, frame: str | None = None) -> dict:
    """:func:`surface_info` of a surface, or :func:`gifti_data_info` for data-only GIfTI."""
    from pybiscuit.io.surface import read_surface, surface_format

    if surface_format(path) == "gifti":
        import nibabel as nib

        intents = {da.intent for da in nib.load(str(path)).darrays}
        if 1008 not in intents:  # NIFTI_INTENT_POINTSET
            return {"path": str(path), **gifti_data_info(path)}
    return {"path": str(path), **surface_info(read_surface(path, frame=frame))}


def _fmt_num(x) -> str:
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def _vec(x) -> str:
    return "(" + ", ".join(f"{t:.3f}" for t in x) + ")"


def format_info(info: dict) -> str:
    """Human-readable text for a :func:`describe` result."""
    lines = [f"{info['path']}"]
    if info.get("kind") == "data":
        lines.append("  format: gifti (data arrays, no geometry)")
        for k, val in info["meta"].items():
            lines.append(f"  {k}: {val}")
        for a in info["arrays"]:
            rng = f"  [{_fmt_num(a['min'])}, {_fmt_num(a['max'])}]" if "min" in a else ""
            lines.append(f"  array {a['name']}: {a['intent']} {a['dtype']} {a['shape']}{rng}")
        if "labels" in info:
            lines.append(f"  label table: {len(info['labels'])} entries")
        return "\n".join(lines)
    c, t, g = info["counts"], info["topology"], info["geometry"]
    lines.append(
        f"  format: {info['format']}  frame: {info['frame'] or 'unknown'}"
        f"  units: {info['units'] or 'unknown'}"
    )
    for k in ("AnatomicalStructurePrimary", "AnatomicalStructureSecondary", "GeometricType",
              "dataspace"):  # fmt: skip
        if k in info.get("gifti", {}):
            lines.append(f"  {k}: {info['gifti'][k]}")
    lines.append(f"  vertices: {c['vertices']}  faces: {c['faces']}  edges: {c['edges']}")
    flags = [n for n in ("closed", "manifold", "consistently_oriented") if t[n]]
    lines.append(
        f"  euler: {t['euler_characteristic']}  genus: {t['genus']}  components: {t['components']}"
        f"  boundary loops: {t['boundary_loops']}  [{', '.join(flags) or 'open/irregular'}]"
    )
    problems = {k: t[k] for k in ("non_manifold_edges", "non_manifold_vertices",
                                   "unreferenced_vertices") if t[k]}  # fmt: skip
    if g.get("degenerate_faces"):
        problems["degenerate_faces"] = g["degenerate_faces"]
    if problems:
        lines.append("  issues: " + ", ".join(f"{k}={n}" for k, n in problems.items()))
    if "bbox_min" in g:
        lines.append(f"  bbox: {_vec(g['bbox_min'])} .. {_vec(g['bbox_max'])}"
                     f"  extent {_vec(g['extent'])}")  # fmt: skip
        lines.append(f"  centroid: {_vec(g['centroid'])}")
    if "area" in g:
        el = g["edge_length"]
        lines.append(f"  area: {g['area']:.6g}  edge length min/mean/max:"
                     f" {el['min']:.4g}/{el['mean']:.4g}/{el['max']:.4g}")  # fmt: skip
    if "volume" in g:
        lines.append(f"  volume: {g['volume']:.6g}  normals: {g['normals']}")
    for name, d in info["point_data"].items():
        rng = (
            f"  [{_fmt_num(d['min'])}, {_fmt_num(d['max'])}] mean {_fmt_num(d['mean'])}"
            if "min" in d
            else ""
        )
        lines.append(f"  point data {name}: {d['dtype']} {d['shape']}{rng}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit surf-info",
        description="Summarise surfaces (.vtk, .gii, .asc, FreeSurfer) and GIfTI data files.",
    )
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--json", action="store_true", help="print a JSON list instead of text")
    p.add_argument("--frame", help="override the frame tag (.asc defaults to RAS)")
    a = p.parse_args(argv)
    infos, status = [], 0
    for path in a.files:
        try:
            infos.append(describe(path, frame=a.frame))
        except (OSError, ValueError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            status = 1
    if a.json:
        print(json.dumps(infos, indent=2))
    else:
        print("\n\n".join(format_info(i) for i in infos))
    return status
