"""MeshProperty port (`B3.1`) and BISCUIT curvature-derived feature maps (`B9.1`).

Black-box and binary-level exact port of ``MeshProperty v1.0`` (``/opt/biscuit/bin/MeshProperty``
in ``biscuit_FC_v2.2.sif``, linked against ``libMesh.so.1.0``) plus the BISCUIT pipeline
feature-map recipes from ``docs/reference/orchestrator-spec.md`` (§5.1, §6.2, §9.1).

Exact ``MeshProperty`` pipeline (`MeshProperty::init` + per-map methods):

1. Load mesh vertices as ``float32`` and triangle faces as ``int32`` (``Mesh::openFile``).
2. If ``surf_smoothing > 0``, smooth the vertex positions in place with
   ``SurfaceUtil::smoothing(mesh, surf_smoothing)``
   (:func:`pybiscuit.mesh.geometry.smooth_vertices`). Subsequent curvature estimation,
   vertex-area calculation, data smoothing, and ``--writevtk`` surface export all operate on this
   smoothed mesh.
3. Compute principal curvatures ``(cmin, cmax)`` in ``float64``:
   - Default (``quad=False``):
     ``SurfaceUtil::curvature(mesh, cmin, cmax, NULL, NULL, tensor_smoothing)``
     (:func:`pybiscuit.mesh.curvature.curvature`).
   - Quadratic fit (``-q`` / ``quad=True``):
     ``SurfaceUtil::curvature_quad(mesh, cmin, cmax, NULL, NULL)``
     (:func:`pybiscuit.mesh.curvature.curvature_quad`; ``tensor_smoothing`` is ignored by the binary
     when ``-q`` is set).
4. If ``--invert`` (``invert=True``), negate ``cmin`` and ``cmax`` in place
   (``cmin = -cmin; cmax = -cmax``) **without** swapping ``cmin`` and ``cmax``.
5. Evaluate requested maps in exact binary order:
   - Curvedness (``-c``, ``.C.txt``, VTK field ``Curvedness``):
     :math:`C = \\sqrt{(c_{\\min}^2 + c_{\\max}^2) / 2}`.
   - Shape index (``-s``, ``.S.txt``, VTK field ``ShapeIndex``):
     :math:`S = (2 / \\pi) \\arctan((c_{\\max} + c_{\\min}) / (c_{\\max} - c_{\\min}))`.
   - Mean curvature (``-m``, ``.H.txt``, VTK field ``MeanCurvature``):
     :math:`H = (c_{\\max} + c_{\\min}) / 2`.
   - Gaussian curvature (``-k``, ``.K.txt``, VTK field ``GaussianCurvature``):
     :math:`K = c_{\\max} \\cdot c_{\\min}`.
   - Vertex area (``-v``, ``.VertArea.txt``, VTK field ``VertexArea``):
     one-third of the incident 1-ring triangle area sum on the (possibly surface-smoothed)
     ``float32`` mesh (:func:`pybiscuit.mesh.geometry.vertex_areas`).
6. For each of ``C``, ``S``, ``H``, ``K`` (never ``VertArea``), post-process in exact order:
   - If ``prctile > 0.0``: :func:`crop_percentile` (clamps at sorted index
     ``int(float(N) * p / 100)`` and ``int(float(N) * (100 - p) / 100)``, with ``p = 100 - p``
     if ``p > 50``).
   - If ``norm_before``: :func:`soft_normalize` with threshold ``5.0`` (two-pass z-score +
     exponential soft clamp outside ``[-5, 5]``).
   - If ``data_smoothing > 0``: ``SurfaceUtil::smoothing(mesh, data_smoothing, data)``
     (:func:`pybiscuit.mesh.geometry.smooth_data`).
   - If ``norm_after``: :func:`soft_normalize` with threshold ``5.0``.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from pybiscuit.mesh.curvature import curvature, curvature_quad
from pybiscuit.mesh.geometry import (
    NeighborLists,
    smooth_data,
    smooth_vertices,
    vertex_areas,
    vertex_neighbor_lists,
)
from pybiscuit.mesh.model import Mesh

__all__ = [
    "MeshPropertyResult",
    "crop_percentile",
    "curvedness_from_principal",
    "format_meshlib_vtk",
    "gaussian_curvature_from_principal",
    "main",
    "mean_curvature_from_principal",
    "mesh_property",
    "parc_normalize_feature",
    "parc_white_mean_curvature",
    "postprocess_property",
    "quant_mid_features",
    "reg_inflated_mean_curvature",
    "reg_white_mean_curvature",
    " run_mesh_property".strip(),
    "shape_index_from_principal",
    "soft_normalize",
    "write_property_txt",
    "write_property_vtk",
]

VERSION = "v1.0"
_DESC = f"Mesh Property {VERSION}\nAuthor: Ilwoo Lyu\n"


def _coerce_f32(mesh, faces=None) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, Mesh):
        v, f = mesh.vertices, mesh.faces
    elif faces is not None:
        v, f = mesh, faces
    else:
        raise TypeError("pass a Mesh, or (vertices, faces)")
    return np.ascontiguousarray(v, dtype=np.float32), np.ascontiguousarray(f, dtype=np.int32)


def curvedness_from_principal(cmin: np.ndarray, cmax: np.ndarray) -> np.ndarray:
    """``MeshProperty::computeCurvedness`` raw formula: ``sqrt((cmin^2 + cmax^2) * 0.5)``."""
    cmin64 = np.asarray(cmin, dtype=np.float64)
    cmax64 = np.asarray(cmax, dtype=np.float64)
    return np.sqrt((cmin64 * cmin64 + cmax64 * cmax64) * 0.5)


def shape_index_from_principal(cmin: np.ndarray, cmax: np.ndarray) -> np.ndarray:
    """``MeshProperty::computeShapeIndex``: ``2 * atan((cmax + cmin) / (cmax - cmin)) / pi``."""
    cmin64 = np.asarray(cmin, dtype=np.float64)
    cmax64 = np.asarray(cmax, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * np.arctan((cmax64 + cmin64) / (cmax64 - cmin64)) / math.pi


def mean_curvature_from_principal(cmin: np.ndarray, cmax: np.ndarray) -> np.ndarray:
    """``MeshProperty::computeMeanCurvature`` raw formula: ``(cmax + cmin) * 0.5``."""
    cmin64 = np.asarray(cmin, dtype=np.float64)
    cmax64 = np.asarray(cmax, dtype=np.float64)
    return (cmax64 + cmin64) * 0.5


def gaussian_curvature_from_principal(cmin: np.ndarray, cmax: np.ndarray) -> np.ndarray:
    """``MeshProperty::computeGaussianCurvature`` raw formula: ``cmax * cmin``."""
    cmin64 = np.asarray(cmin, dtype=np.float64)
    cmax64 = np.asarray(cmax, dtype=np.float64)
    return cmax64 * cmin64


def crop_percentile(values: np.ndarray, prctile: float) -> np.ndarray:
    """Port of ``MeshProperty::crop(double p)`` (`0x3baf0`).

    Clamps ``values`` between sorted index ``int(double(float(N)) * p / 100.0)`` and
    ``int(double(float(N)) * (100.0 - p) / 100.0)``, where ``p = 100.0 - p`` if ``p > 50.0``.
    """
    arr = np.array(values, dtype=np.float64, copy=True)
    p = float(prctile)
    if p <= 0.0 or arr.size == 0:
        return arr
    if p > 50.0:
        p = 100.0 - p
    n_f = float(np.float32(arr.size))
    lo_idx = min(max(int(n_f * p / 100.0), 0), arr.size - 1)
    hi_idx = min(max(int(n_f * (100.0 - p) / 100.0), 0), arr.size - 1)
    sorted_vals = np.sort(arr)
    lo_val = sorted_vals[lo_idx]
    hi_val = sorted_vals[hi_idx]
    return np.clip(arr, lo_val, hi_val)


@njit(cache=True, error_model="numpy")
def _stats_mean_f64(a: np.ndarray) -> float:
    n = a.shape[0]
    s = 0.0
    for i in range(n):
        s += a[i]
    return s / float(n)


@njit(cache=True, error_model="numpy")
def _stats_var_f64(a: np.ndarray, ddof: int) -> float:
    n = a.shape[0]
    m = _stats_mean_f64(a)
    denom = float(n - ddof)
    res = 0.0
    for i in range(n):
        d = a[i] - m
        res += (d * d) / denom
    return res


@njit(cache=True, error_model="numpy")
def _soft_normalize_f64(a: np.ndarray, th: float, ddof: int) -> np.ndarray:
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    m1 = _stats_mean_f64(a)
    v1 = _stats_var_f64(a, ddof)
    sd1 = math.sqrt(v1)
    inv_sd1 = 1.0 / sd1 if sd1 > 0.0 else 0.0
    neg_th = -th
    for i in range(n):
        x = (a[i] - m1) * inv_sd1
        if x < neg_th:
            x = neg_th - (1.0 - math.exp(x + th))
        elif x > th:
            x = th + (1.0 - math.exp(th - x))
        out[i] = x
    v2 = _stats_var_f64(out, ddof)
    sd2 = math.sqrt(v2)
    inv_sd2 = 1.0 / sd2 if sd2 > 0.0 else 0.0
    for i in range(n):
        x = out[i] * inv_sd2
        if x < neg_th:
            x = neg_th - (1.0 - math.exp(x + th))
        elif x > th:
            x = th + (1.0 - math.exp(th - x))
        out[i] = x
    return out


def soft_normalize(values: np.ndarray, threshold: float = 5.0, *, ddof: int = 0) -> np.ndarray:
    """Port of ``MeshProperty::normalize(double th)`` (`0x3b4b0`, called with ``th = 5.0``).

    Pass 1 standardises by ``(x - mean) * (1 / sd)`` and applies the exponential soft clamp
    outside ``[-threshold, threshold]``:

    .. math::

        x \\leftarrow \\begin{cases}
            -\\text{th} - (1 - \\exp(x + \\text{th})) & x < -\\text{th}, \\\\
            \\text{th} + (1 - \\exp(\\text{th} - x)) & x > \\text{th}.
        \\end{cases}

    Pass 2 recomputes ``sd = sqrt(var(x))`` and scales ``x * (1 / sd)`` (without subtracting the
    mean), then soft-clamps outside ``[-threshold, threshold]`` a second time.
    ``MeshProperty`` uses ``Statistics::var`` (``ddof=0``, ``threshold=5.0``); MATLAB's
    ``extract_features.m`` uses ``zscore``/``std`` (``ddof=1``, ``threshold=3.0``).
    """
    arr = np.ascontiguousarray(values, dtype=np.float64)
    return _soft_normalize_f64(arr, float(threshold), int(ddof))


def postprocess_property(
    values: np.ndarray,
    faces: np.ndarray | None = None,
    *,
    prctile: float = 0.0,
    norm_before: bool = False,
    data_smoothing: int = 0,
    norm_after: bool = False,
    neighbors: NeighborLists | None = None,
) -> np.ndarray:
    """Apply the 4-stage post-processing pipeline of ``MeshProperty`` to a raw curvature map."""
    out = np.array(values, dtype=np.float64, copy=True)
    if prctile > 0.0:
        out = crop_percentile(out, prctile)
    if norm_before:
        out = soft_normalize(out, 5.0, ddof=0)
    if data_smoothing > 0:
        if faces is None and neighbors is None:
            raise ValueError("faces or neighbors required when data_smoothing > 0")
        out = smooth_data(
            faces if faces is not None else np.empty((0, 3), dtype=np.int32),
            out,
            int(data_smoothing),
            neighbors=neighbors,
        )
    if norm_after:
        out = soft_normalize(out, 5.0, ddof=0)
    return out


@dataclass
class MeshPropertyResult:
    """Result of :func:`mesh_property`."""

    mesh: Mesh
    cmin: np.ndarray | None = None
    cmax: np.ndarray | None = None
    curvedness: np.ndarray | None = None
    shape_index: np.ndarray | None = None
    mean_curvature: np.ndarray | None = None
    gaussian_curvature: np.ndarray | None = None
    vertex_area: np.ndarray | None = None

    def vtk_fields(self) -> list[tuple[str, np.ndarray]]:
        """Ordered ``(field_name, array)`` pairs as written by ``MeshProperty --writevtk``."""
        fields: list[tuple[str, np.ndarray]] = []
        if self.curvedness is not None:
            fields.append(("Curvedness", self.curvedness))
        if self.shape_index is not None:
            fields.append(("ShapeIndex", self.shape_index))
        if self.mean_curvature is not None:
            fields.append(("MeanCurvature", self.mean_curvature))
        if self.gaussian_curvature is not None:
            fields.append(("GaussianCurvature", self.gaussian_curvature))
        if self.vertex_area is not None:
            fields.append(("VertexArea", self.vertex_area))
        return fields


def mesh_property(
    mesh: Mesh | np.ndarray,
    faces: np.ndarray | None = None,
    *,
    curvedness: bool = False,
    shape_index: bool = False,
    mean_curvature: bool = False,
    gaussian_curvature: bool = False,
    vertex_area: bool = False,
    quad: bool = False,
    invert: bool = False,
    tensor_smoothing: int = 0,
    surf_smoothing: int = 0,
    data_smoothing: int = 0,
    prctile: float = 0.0,
    norm_before: bool = False,
    norm_after: bool = False,
    neighbors: NeighborLists | None = None,
) -> MeshPropertyResult:
    """Compute surface property maps with exact ``MeshProperty`` semantics."""
    v, f = _coerce_f32(mesh, faces)
    tensor_smoothing = max(0, int(tensor_smoothing))
    surf_smoothing = max(0, int(surf_smoothing))
    data_smoothing = max(0, int(data_smoothing))
    nb = neighbors or vertex_neighbor_lists(f, v.shape[0])

    if surf_smoothing > 0:
        v = smooth_vertices(v, f, surf_smoothing, neighbors=nb)

    need_curv = curvedness or shape_index or mean_curvature or gaussian_curvature
    cmin = cmax = None
    if need_curv:
        if quad:
            cmin, cmax, _, _ = curvature_quad(v, f, neighbors=nb, dtype=np.float64)
        else:
            cmin, cmax, _, _ = curvature(
                v,
                f,
                n_iter_smoothing_tensor=tensor_smoothing,
                neighbors=nb,
                dtype=np.float64,
            )
        if invert:
            cmin = -cmin
            cmax = -cmax

    c_map = s_map = h_map = k_map = va_map = None
    if curvedness and cmin is not None and cmax is not None:
        c_map = postprocess_property(
            curvedness_from_principal(cmin, cmax),
            f,
            prctile=prctile,
            norm_before=norm_before,
            data_smoothing=data_smoothing,
            norm_after=norm_after,
            neighbors=nb,
        )
    if shape_index and cmin is not None and cmax is not None:
        s_map = postprocess_property(
            shape_index_from_principal(cmin, cmax),
            f,
            prctile=prctile,
            norm_before=norm_before,
            data_smoothing=data_smoothing,
            norm_after=norm_after,
            neighbors=nb,
        )
    if mean_curvature and cmin is not None and cmax is not None:
        h_map = postprocess_property(
            mean_curvature_from_principal(cmin, cmax),
            f,
            prctile=prctile,
            norm_before=norm_before,
            data_smoothing=data_smoothing,
            norm_after=norm_after,
            neighbors=nb,
        )
    if gaussian_curvature and cmin is not None and cmax is not None:
        k_map = postprocess_property(
            gaussian_curvature_from_principal(cmin, cmax),
            f,
            prctile=prctile,
            norm_before=norm_before,
            data_smoothing=data_smoothing,
            norm_after=norm_after,
            neighbors=nb,
        )
    if vertex_area:
        va_map = vertex_areas(v, f, neighbors=nb).astype(np.float64)

    out_mesh = Mesh(v.astype(np.float64), f)
    res = MeshPropertyResult(
        mesh=out_mesh,
        cmin=cmin,
        cmax=cmax,
        curvedness=c_map,
        shape_index=s_map,
        mean_curvature=h_map,
        gaussian_curvature=k_map,
        vertex_area=va_map,
    )
    for name, arr in res.vtk_fields():
        out_mesh.point_data[name] = arr
    return res


def write_property_txt(values: np.ndarray, path: str | Path) -> None:
    """Write a 1-D property array with ``%f\\n`` per vertex (``MeshProperty::saveFile``)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    text = "".join(f"{x:f}\n" for x in arr.tolist())
    Path(path).write_text(text)


def format_meshlib_vtk(
    vertices: np.ndarray,
    faces: np.ndarray,
    fields: list[tuple[str, np.ndarray]] | None = None,
) -> str:
    """Format a mesh and optional ``FIELD ScalarData`` arrays exactly like ``VtkIO::save``."""
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    n, m = v.shape[0], f.shape[0]
    lines = [
        "# vtk DataFile Version 3.0",
        "vtk output",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {n} float",
    ]
    lines.extend(f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f}" for row in v.tolist())
    lines.append(f"POLYGONS {m} {m * 4}")
    lines.extend(f"3 {row[0]} {row[1]} {row[2]}" for row in f.tolist())
    lines.append("")
    if fields is not None:
        lines.append(f"POINT_DATA {n}")
        lines.append(f"FIELD ScalarData {len(fields)}")
        for name, arr in fields:
            arr64 = np.asarray(arr, dtype=np.float64).ravel()
            lines.append(f"{name} 1 {n} float")
            lines.extend(f"{x:f}" for x in arr64.tolist())
            lines.append("")
    return "\n".join(lines) + "\n"


def write_property_vtk(
    mesh: Mesh | np.ndarray,
    fields: list[tuple[str, np.ndarray]],
    path: str | Path,
    faces: np.ndarray | None = None,
) -> None:
    """Write ``<output>.property.vtk`` with ``VtkIO::save`` + ``FIELD ScalarData`` layout."""
    v, f = _coerce_f32(mesh, faces)
    Path(path).write_text(format_meshlib_vtk(v, f, fields))


# --------------------------------------------------------------------------- #
# BISCUIT pipeline feature-map helpers (B9.1; orchestrator-spec.md §5.1, §6.2, §9.1)
# --------------------------------------------------------------------------- #
def quant_mid_features(
    mid_mesh: Mesh | np.ndarray,
    faces: np.ndarray | None = None,
    *,
    curv_smoothness: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantification mid-surface maps ``(H, S, area)`` (spec §9.1 T31 and §5.5 T25).

    * ``H`` and ``S`` come from
      ``MeshProperty -i <mid.vtk> -o <mid> -s -m --surfsmoothing N --tensorsmoothing N``
      with ``N = curv_smoothness`` (default ``2`` in ``biscuit.conf``).
    * ``area`` comes from ``MeshProperty -i <mid.vtk> -o <mid> -v`` on the unsmoothed mid surface.
    """
    v, f = _coerce_f32(mid_mesh, faces)
    nb = vertex_neighbor_lists(f, v.shape[0])
    hs = mesh_property(
        v,
        f,
        mean_curvature=True,
        shape_index=True,
        surf_smoothing=curv_smoothness,
        tensor_smoothing=curv_smoothness,
        neighbors=nb,
    )
    va = vertex_areas(v, f, neighbors=nb).astype(np.float64)
    assert hs.mean_curvature is not None and hs.shape_index is not None
    return hs.mean_curvature, hs.shape_index, va


def reg_white_mean_curvature(
    white_mesh: Mesh | np.ndarray,
    faces: np.ndarray | None = None,
    *,
    surf_smoothing: int = 5,
    tensor_smoothing: int = 5,
    round_intermediate: bool = False,
) -> np.ndarray:
    """Registration white-surface curvature feature ``-4 * H`` (spec §5.1 T17, ``<h>.H.txt``).

    Runs ``MeshProperty -i <white.vtk> -o <h> -m --surfsmoothing 5 --tensorsmoothing 5`` and
    multiplies by ``-4``. When ``round_intermediate=True``, rounds the intermediate ``H`` to 6
    decimal places (matching MATLAB re-reading ``<h>.H.txt`` before rewriting ``-4 * H``).
    """
    res = mesh_property(
        white_mesh,
        faces,
        mean_curvature=True,
        surf_smoothing=surf_smoothing,
        tensor_smoothing=tensor_smoothing,
    )
    assert res.mean_curvature is not None
    h = np.round(res.mean_curvature, 6) if round_intermediate else res.mean_curvature
    return -4.0 * h


def reg_inflated_mean_curvature(
    inflated_mesh: Mesh | np.ndarray,
    faces: np.ndarray | None = None,
    *,
    prctile: float = 0.1,
    data_smoothing: int = 25,
) -> np.ndarray:
    """Registration inflated-surface mean curvature ``iH`` (spec §5.1 T19, ``<h>.iH.txt``).

    Runs ``MeshProperty -i <inflated.vtk> -o <h>.inflated --normbefore --normafter --prctile 0.1
    -m --datasmoothing 25 --invert -q``.
    """
    res = mesh_property(
        inflated_mesh,
        faces,
        mean_curvature=True,
        quad=True,
        invert=True,
        prctile=prctile,
        norm_before=True,
        data_smoothing=data_smoothing,
        norm_after=True,
    )
    assert res.mean_curvature is not None
    return res.mean_curvature


def parc_normalize_feature(values: np.ndarray, *, threshold: float = 3.0) -> np.ndarray:
    """Parcellation feature normalisation from ``extract_features.m:8-28`` (spec §6.2).

    Two-pass standardisation using sample standard deviation (``ddof=1``) and soft clamping at
    ``±threshold`` (default ``3.0``).
    """
    return soft_normalize(values, threshold=threshold, ddof=1)


def parc_white_mean_curvature(
    white_mesh: Mesh | np.ndarray,
    faces: np.ndarray | None = None,
    *,
    surf_smoothing: int = 5,
    tensor_smoothing: int = 5,
    normalize: bool = True,
    round_intermediate: bool = False,
) -> np.ndarray:
    """Parcellation white-surface mean curvature feature ``-H`` (spec §6.2 T26).

    Computes ``H = -(T26 output)`` where T26 is ``MeshProperty -i <white.vtk> -o <tmp> -m
    --surfsmoothing 5 --tensorsmoothing 5``, then optionally applies :func:`parc_normalize_feature`.
    """
    res = mesh_property(
        white_mesh,
        faces,
        mean_curvature=True,
        surf_smoothing=surf_smoothing,
        tensor_smoothing=tensor_smoothing,
    )
    assert res.mean_curvature is not None
    h = np.round(res.mean_curvature, 6) if round_intermediate else res.mean_curvature
    neg_h = -h
    return parc_normalize_feature(neg_h) if normalize else neg_h


# --------------------------------------------------------------------------- #
# CLI (`pybiscuit mesh-property` / `MeshProperty`)
# --------------------------------------------------------------------------- #
class _ArgError(Exception):
    def __init__(self, msg: str, code: int) -> None:
        super().__init__(msg)
        self.code = code


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        code = 109 if "unrecognized" in message else 114
        raise _ArgError(message, code)


def _nonneg_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not convert: {s}") from None
    if v < 0:
        raise argparse.ArgumentTypeError(f"Number {s} not in range [0 - inf]")
    return v


def _prctile_float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not convert: {s}") from None
    if not (0.0 <= v <= 100.0):
        raise argparse.ArgumentTypeError(f"Number {s} not in range [0 - 100]")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="MeshProperty",
        description=_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("-i", "--input", help="provides an input filename")
    p.add_argument("-o", "--output", help="provides a prefix of output")
    p.add_argument("-c", dest="curvedness", action="store_true", help="computes a curvedness map")
    p.add_argument("-s", dest="shapeindex", action="store_true", help="computes a shape index map")
    p.add_argument(
        "-k", dest="gaussiancurv", action="store_true", help="computes a Gaussian curvature map"
    )
    p.add_argument("-m", dest="meancurv", action="store_true", help="computes a mean curvature map")
    p.add_argument("-v", dest="vertexarea", action="store_true", help="computes a vertex area map")
    p.add_argument("--writevtk", action="store_true", help="writes a vtk file with properties")
    p.add_argument(
        "--tensorsmoothing",
        type=_nonneg_int,
        default=0,
        help="applies a smoothing over the tensor field with the specific number of iterations",
    )
    p.add_argument(
        "--surfsmoothing",
        type=_nonneg_int,
        default=0,
        help="applies a smoothing over the input surface with the specific number of iterations",
    )
    p.add_argument(
        "--datasmoothing",
        type=_nonneg_int,
        default=0,
        help=(
            "applies a smoothing over the output (except for vertex area) "
            "with the specific number of iterations"
        ),
    )
    p.add_argument(
        "-q",
        "--quad",
        action="store_true",
        help="approximates curvatures using quadratic fitting",
    )
    p.add_argument(
        "--prctile",
        type=_prctile_float,
        default=0.0,
        help="crops the output between prctile and 100 - prctile",
    )
    p.add_argument(
        "--normbefore", action="store_true", help="normalizes the output before data smoothing"
    )
    p.add_argument(
        "--normafter", action="store_true", help="normalizes the output after data smoothing"
    )
    p.add_argument("--invert", action="store_true", help="inverts curvature signs")
    return p


def run_mesh_property(
    input_path: str | Path,
    output_prefix: str | Path,
    *,
    curvedness: bool = False,
    shape_index: bool = False,
    mean_curvature: bool = False,
    gaussian_curvature: bool = False,
    vertex_area: bool = False,
    write_vtk: bool = False,
    tensor_smoothing: int = 0,
    surf_smoothing: int = 0,
    data_smoothing: int = 0,
    quad: bool = False,
    prctile: float = 0.0,
    norm_before: bool = False,
    norm_after: bool = False,
    invert: bool = False,
) -> MeshPropertyResult:
    """Run ``MeshProperty`` on ``input_path`` and write requested outputs to ``output_prefix``."""
    from pybiscuit.io.vtk import read_vtk

    mesh = read_vtk(input_path)
    res = mesh_property(
        mesh,
        curvedness=curvedness,
        shape_index=shape_index,
        mean_curvature=mean_curvature,
        gaussian_curvature=gaussian_curvature,
        vertex_area=vertex_area,
        quad=quad,
        invert=invert,
        tensor_smoothing=tensor_smoothing,
        surf_smoothing=surf_smoothing,
        data_smoothing=data_smoothing,
        prctile=prctile,
        norm_before=norm_before,
        norm_after=norm_after,
    )
    prefix = str(output_prefix)
    if res.curvedness is not None:
        write_property_txt(res.curvedness, f"{prefix}.C.txt")
    if res.shape_index is not None:
        write_property_txt(res.shape_index, f"{prefix}.S.txt")
    if res.mean_curvature is not None:
        write_property_txt(res.mean_curvature, f"{prefix}.H.txt")
    if res.gaussian_curvature is not None:
        write_property_txt(res.gaussian_curvature, f"{prefix}.K.txt")
    if res.vertex_area is not None:
        write_property_txt(res.vertex_area, f"{prefix}.VertArea.txt")
    if write_vtk:
        write_property_vtk(res.mesh, res.vtk_fields(), f"{prefix}.property.vtk")
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``MeshProperty`` / ``pybiscuit mesh-property``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(x in ("-h", "--help") for x in argv):
        print(build_parser().format_help())
        return 0
    try:
        a = build_parser().parse_args(argv)
        if a.input is None:
            raise _ArgError("--input is required", 106)
        if a.output is None:
            raise _ArgError("--output is required", 106)
        if not Path(a.input).is_file():
            raise _ArgError(f"--input: File does not exist: {a.input}", 105)
    except _ArgError as e:
        print(f"{e}\nRun with --help for more information.", file=sys.stderr)
        return e.code
    run_mesh_property(
        a.input,
        a.output,
        curvedness=a.curvedness,
        shape_index=a.shapeindex,
        mean_curvature=a.meancurv,
        gaussian_curvature=a.gaussiancurv,
        vertex_area=a.vertexarea,
        write_vtk=a.writevtk,
        tensor_smoothing=a.tensorsmoothing,
        surf_smoothing=a.surfsmoothing,
        data_smoothing=a.datasmoothing,
        quad=a.quad,
        prctile=a.prctile,
        norm_before=a.normbefore,
        norm_after=a.normafter,
        invert=a.invert,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
