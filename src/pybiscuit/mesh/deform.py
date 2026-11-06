"""MeshDeform port (`B3.2`, `B5.6`).

Black-box and binary-level exact port of ``MeshDeform v1.0`` (``/opt/biscuit/bin/MeshDeform``
in ``biscuit_FC_v2.2.sif``, linked against ``libMesh.so.1.0``).

Exact operation order in ``MeshDeform::main`` (`0x70f0`):

1. Load input mesh as ``float32`` vertices and ``int32`` triangle faces (``Mesh::openFile``).
2. If ``--matrix`` (12 values, row-major 3x4) is provided: ``MeshDeform::applyMatrix`` (`0x3b390`).
   Accumulates each row term in ``float64`` with intermediate ``float32`` casts between columns
   0, 1, and 2, adds the translation column ``m[r, 3]`` in ``float64``, and stores ``float32``.
3. If ``-t`` / ``--translate`` (3 values) is provided: ``MeshDeform::translate`` (`0x3b1f0`).
   **Subtracts** ``(tx, ty, tz)`` from each vertex in ``float64`` and casts back to ``float32``:
   ``v[k] = float32(double(v[k]) - double(t[k]))``.
4. If ``--translateCentered`` (3 values) is provided: ``MeshDeform::translateCentered`` (`0x3b100`).
   First centers the mesh (``Mesh::centering``: sequential ``float32`` mean subtracted in
   ``float32``), then subtracts ``(tx, ty, tz)`` as in step 3.
5. If ``-c`` / ``--center`` is set: ``MeshDeform::getCenter`` (`0x3b570`).
   Computes the ``float32`` vertex centroid ``(cx, cy, cz)`` and prints ``-cx -cy -cz`` to stdout
   (the negated centroid, ready to pass to ``MeshDeform`` or ``mris_convert``).
6. If ``--scale`` is non-zero: ``MeshDeform::scale`` (`0x3b710`) -> ``Mesh::scaling(float(scale))``.
7. If ``--invertFace`` is set: ``MeshDeform::invertFace`` (`0x3b2e0`), reversing each triangle's
   vertex indices ``(f0, f1, f2) -> (f2, f1, f0)``.
8. If ``--sphere`` is 1 or 2: ``MeshDeform::makeSphere`` (`0x3b720`).
   First runs ``SurfaceUtil::smoothing(mesh, 200)`` (200 iterations of surface smoothing), then
   ``SurfaceUtil::sphere(mesh, sphere - 1, 10000)`` (``sphere=1`` -> area preservation ``type=0``;
   ``sphere=2`` -> angle/conformal preservation ``type=1``).
9. If ``--smooth`` > 0: ``MeshDeform::smooth`` (`0x3b8a0`) ->
   ``SurfaceUtil::smoothing(mesh, iter)``.
10. If ``--noise`` > 0: ``MeshDeform::addNoise`` (`0x3b760`), displacing each vertex along its
    recomputed unit normal by ``N(0, sigma^2)``.
11. If ``-o`` / ``--output`` is non-empty: ``MeshDeform::saveFile`` ->
    ``VtkIO::save(output, mesh, false)``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from pybiscuit.mesh.geometry import smooth_vertices, vertex_normals
from pybiscuit.mesh.model import Mesh
from pybiscuit.mesh.property import format_meshlib_vtk
from pybiscuit.mesh.sphere_map import sphere_map

__all__ = [
    "MeshDeformResult",
    "add_gaussian_noise",
    "apply_matrix",
    "center_vertices",
    "invert_faces",
    "main",
    "make_sphere",
    "mesh_deform",
    "negated_center",
    "run_mesh_deform",
    "scale_vertices",
    "translate_centered_vertices",
    "translate_vertices",
]

VERSION = "v1.0"
_DESC = f"Mesh Deformation {VERSION}\nAuthor: Ilwoo Lyu\n"


def _coerce_f32(mesh, faces=None) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, Mesh):
        v, f = mesh.vertices, mesh.faces
    elif faces is not None:
        v, f = mesh, faces
    else:
        raise TypeError("pass a Mesh, or (vertices, faces)")
    return np.ascontiguousarray(v, dtype=np.float32), np.ascontiguousarray(f, dtype=np.int32)


@njit(cache=True, error_model="numpy")
def _apply_matrix_f32(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    n = v.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        vx = float(v[i, 0])
        vy = float(v[i, 1])
        vz = float(v[i, 2])
        for r in range(3):
            acc = float(np.float32(0.0))
            acc = float(np.float32(acc + vx * m[r, 0]))
            acc = float(np.float32(acc + vy * m[r, 1]))
            acc = float(np.float32(acc + vz * m[r, 2]))
            out[i, r] = np.float32(acc + m[r, 3])
    return out


def apply_matrix(vertices: np.ndarray, matrix: Sequence[float] | np.ndarray) -> np.ndarray:
    """Port of ``MeshDeform::applyMatrix`` (`0x3b390`).

    Accepts a 12-element flat sequence, a ``(3, 4)`` array, or a ``(4, 4)`` homogeneous matrix
    (top ``3x4`` block used). Reproduces the binary's mixed ``float32``/``float64`` accumulation.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape == (4, 4):
        m = np.ascontiguousarray(arr[:3, :4], dtype=np.float64)
    elif arr.size == 12:
        m = np.ascontiguousarray(arr.reshape(3, 4), dtype=np.float64)
    else:
        raise ValueError(f"matrix must have 12 elements (3x4) or shape (4, 4), got {arr.shape}")
    return _apply_matrix_f32(v, m)


def translate_vertices(vertices: np.ndarray, offset: Sequence[float] | np.ndarray) -> np.ndarray:
    """Port of ``MeshDeform::translate`` (`0x3b1f0`): subtracts ``offset`` in ``float64``."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    t = np.asarray(offset, dtype=np.float64).ravel()
    if t.size < 3:
        raise ValueError(f"translate offset must have at least 3 elements, got {t.size}")
    return (v.astype(np.float64) - t[:3]).astype(np.float32)


@njit(cache=True, error_model="numpy")
def _centroid_f32(v: np.ndarray) -> tuple[np.float32, np.float32, np.float32]:
    cx = np.float32(0.0)
    cy = np.float32(0.0)
    cz = np.float32(0.0)
    n = v.shape[0]
    if n == 0:
        return cx, cy, cz
    for i in range(n):
        cx = np.float32(cx + v[i, 0])
        cy = np.float32(cy + v[i, 1])
        cz = np.float32(cz + v[i, 2])
    nf = np.float32(n)
    return np.float32(cx / nf), np.float32(cy / nf), np.float32(cz / nf)


@njit(cache=True, error_model="numpy")
def _center_vertices_f32(v: np.ndarray) -> np.ndarray:
    cx, cy, cz = _centroid_f32(v)
    n = v.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        out[i, 0] = np.float32(v[i, 0] - cx)
        out[i, 1] = np.float32(v[i, 1] - cy)
        out[i, 2] = np.float32(v[i, 2] - cz)
    return out


def center_vertices(vertices: np.ndarray) -> np.ndarray:
    """Port of ``Mesh::centering`` (`Mesh.cpp:556-571`)."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    return _center_vertices_f32(v)


def translate_centered_vertices(
    vertices: np.ndarray, offset: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Port of ``MeshDeform::translateCentered`` (`0x3b100`): ``centering()`` + ``translate()``."""
    return translate_vertices(center_vertices(vertices), offset)


def negated_center(vertices: np.ndarray) -> tuple[float, float, float]:
    """Port of ``MeshDeform::getCenter`` (`0x3b570`): returns ``(-cx, -cy, -cz)`` in ``float32``."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    cx, cy, cz = _centroid_f32(v)
    return (float(np.float32(-cx)), float(np.float32(-cy)), float(np.float32(-cz)))


def scale_vertices(vertices: np.ndarray, scale: float) -> np.ndarray:
    """Port of ``MeshDeform::scale`` (`0x3b710`) -> ``Mesh::scaling(float(scale))``."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    factor = np.float32(scale)
    if factor == np.float32(0.0):
        return v.copy()
    return (v * factor).astype(np.float32)


def invert_faces(faces: np.ndarray) -> np.ndarray:
    """Port of ``MeshDeform::invertFace`` (`0x3b2e0`): reverses each triangle's vertex order."""
    f = np.asarray(faces, dtype=np.int32)
    return np.ascontiguousarray(f[:, ::-1], dtype=np.int32)


def make_sphere(vertices: np.ndarray, faces: np.ndarray, sphere_type: int) -> np.ndarray:
    """Port of ``MeshDeform::makeSphere`` (`0x3b720`).

    ``sphere_type`` is ``1`` (area-preserving, ``SurfaceUtil::sphere(..., 0, 10000)``) or
    ``2`` (angle/conformal, ``SurfaceUtil::sphere(..., 1, 10000)``), preceded by 200 iterations of
    ``SurfaceUtil::smoothing``.
    """
    if sphere_type not in (1, 2):
        raise ValueError(f"sphere_type must be 1 or 2, got {sphere_type}")
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    v_sm = smooth_vertices(v, f, 200)
    res = sphere_map(v_sm, f, type=sphere_type - 1, max_iter=10000)
    return np.ascontiguousarray(res.mesh.vertices, dtype=np.float32)


def add_gaussian_noise(
    vertices: np.ndarray,
    faces: np.ndarray,
    sigma: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Port of ``MeshDeform::addNoise(float sigma)`` (`0x3b760`).

    Updates vertex normals and displaces each vertex along its unit normal by ``N(0, sigma^2)``.
    """
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if sigma <= 0.0:
        return v.copy()
    vn = vertex_normals(v, f)
    gen = rng if rng is not None else np.random.default_rng()
    disp = gen.normal(0.0, float(np.float32(sigma)), size=v.shape[0]).astype(np.float32)
    return (v + vn * disp[:, None]).astype(np.float32)


@dataclass
class MeshDeformResult:
    """Result of :func:`mesh_deform`."""

    mesh: Mesh
    center_printed: tuple[float, float, float] | None = None


def mesh_deform(
    mesh: Mesh | np.ndarray,
    faces: np.ndarray | None = None,
    *,
    matrix: Sequence[float] | np.ndarray | None = None,
    translate: Sequence[float] | np.ndarray | None = None,
    translate_centered: Sequence[float] | np.ndarray | None = None,
    center: bool = False,
    scale: float = 0.0,
    invert_face: bool = False,
    sphere: int = 0,
    smooth: int = 0,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> MeshDeformResult:
    """Apply ``MeshDeform`` operations in the exact order of ``MeshDeform::main``."""
    v, f = _coerce_f32(mesh, faces)
    if matrix is not None and len(matrix) > 0:
        v = apply_matrix(v, matrix)
    if translate is not None and len(translate) > 0:
        v = translate_vertices(v, translate)
    if translate_centered is not None and len(translate_centered) > 0:
        v = translate_centered_vertices(v, translate_centered)
    c_out: tuple[float, float, float] | None = None
    if center:
        c_out = negated_center(v)
    if scale != 0.0:
        v = scale_vertices(v, scale)
    if invert_face:
        f = invert_faces(f)
    if sphere in (1, 2):
        v = make_sphere(v, f, sphere)
    if smooth > 0:
        v = smooth_vertices(v, f, int(smooth))
    if noise > 0.0:
        v = add_gaussian_noise(v, f, noise, rng=rng)
    return MeshDeformResult(mesh=Mesh(v.astype(np.float64), f), center_printed=c_out)


# --------------------------------------------------------------------------- #
# CLI (`pybiscuit mesh-deform` / `MeshDeform`)
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


def _nonneg_float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not convert: {s}") from None
    if v < 0.0:
        raise argparse.ArgumentTypeError(f"Number {s} not in range [0 - inf]")
    return v


def _pos_float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not convert: {s}") from None
    if v <= 0.0:
        raise argparse.ArgumentTypeError(f"Number {s} not in range (0 - inf)")
    return v


def _sphere_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not convert: {s}") from None
    if v not in (1, 2):
        raise argparse.ArgumentTypeError(f"Number {s} not in range [1 - 2]")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="MeshDeform",
        description=_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--matrix", type=float, nargs="+", help="provides a matrix (3 by 4)")
    p.add_argument("-t", "--translate", type=float, nargs="+", help="translates points")
    p.add_argument(
        "--translateCentered",
        type=float,
        nargs="+",
        help="provides an origin point after centering",
    )
    p.add_argument(
        "--sphere",
        type=_sphere_int,
        default=0,
        help="makes a sphere (1: area preservation, 2: conformal map)",
    )
    p.add_argument("--scale", type=_pos_float, default=0.0, help="sets a scaling factor")
    p.add_argument(
        "--noise",
        type=_nonneg_float,
        default=0.0,
        help="sets a random displacement (uniform distribution)",
    )
    p.add_argument(
        "--smooth",
        type=_nonneg_int,
        default=0,
        help="sets # of iterations for surface smoothing",
    )
    p.add_argument("-i", "--input", help="provides an input filename")
    p.add_argument("-o", "--output", default="", help="provides an output filename")
    p.add_argument("-c", "--center", action="store_true", help="prints a center of the input mesh")
    p.add_argument("--invertFace", action="store_true", help="inverts triangles")
    return p


def run_mesh_deform(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    matrix: Sequence[float] | np.ndarray | None = None,
    translate: Sequence[float] | np.ndarray | None = None,
    translate_centered: Sequence[float] | np.ndarray | None = None,
    center: bool = False,
    scale: float = 0.0,
    invert_face: bool = False,
    sphere: int = 0,
    smooth: int = 0,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> MeshDeformResult:
    """Run ``MeshDeform`` on ``input_path`` and optionally write ``output_path``."""
    from pybiscuit.io.surface import read_surface

    mesh = read_surface(input_path)
    res = mesh_deform(
        mesh,
        matrix=matrix,
        translate=translate,
        translate_centered=translate_centered,
        center=center,
        scale=scale,
        invert_face=invert_face,
        sphere=sphere,
        smooth=smooth,
        noise=noise,
        rng=rng,
    )
    if res.center_printed is not None:
        cx, cy, cz = res.center_printed
        print(f"{cx:.6g} {cy:.6g} {cz:.6g}")
    if output_path is not None and str(output_path) != "":
        Path(output_path).write_text(format_meshlib_vtk(res.mesh.vertices, res.mesh.faces))
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``MeshDeform`` / ``pybiscuit mesh-deform``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(x in ("-h", "--help") for x in argv):
        print(build_parser().format_help())
        return 0
    try:
        a = build_parser().parse_args(argv)
        if a.input is None:
            raise _ArgError("--input is required", 106)
        if not Path(a.input).is_file():
            raise _ArgError(f"--input: File does not exist: {a.input}", 105)
    except _ArgError as e:
        print(f"{e}\nRun with --help for more information.", file=sys.stderr)
        return e.code
    run_mesh_deform(
        a.input,
        a.output if a.output else None,
        matrix=a.matrix,
        translate=a.translate,
        translate_centered=a.translateCentered,
        center=a.center,
        scale=a.scale,
        invert_face=a.invertFace,
        sphere=a.sphere,
        smooth=a.smooth,
        noise=a.noise,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
