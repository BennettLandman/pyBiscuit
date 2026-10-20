"""MeshLib spherical mapping: port of ``SurfaceUtil::sphere(mesh, type, max_iter)``.

Port of ``MeshLib/src/Util/SurfaceUtil.cpp`` L284-L418 (plus ``Mesh::centering``,
Mesh.cpp L556-L571).
This is a *port only*: it is NOT adopted as the replacement for FreeSurfer's inflation / spherical
mapping (owner question Q3).  See ``docs/ports/meshlib-geometry.md`` (section "Spherical mapping").

The algorithm is an iterated, weighted 1-ring averaging with re-projection onto the unit sphere
(Jacobi: all vertices are updated from the previous positions).  ``type == 0`` ("area
preservation") seeds the weights from triangle areas and keeps the vertex itself in the average;
any other ``type`` ("angle preservation") uses cotangent-like angle weights without the vertex.
Iteration stops when no face is inverted (``(V1 - V0) x (V2 - V0) . (V0 + V1 + V2) < 0``) or after
``max_iter`` passes; at least one pass is always done (do-while).

Precision: float32 vertices and weights, as in MeshLib.  The angle weights are computed in double
(``double x = P.unit() * Q.unit()``) and added to the float weight via a double sum, exactly as in
the C++.  The result is deterministic and (on arm64/x86-64 without FMA contraction) bit-identical
to the C++ float arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from .geometry import NeighborLists, vertex_neighbor_lists
from .model import Mesh

__all__ = ["SphereMapResult", "count_inverted_faces", "sphere_map"]


@dataclass
class SphereMapResult:
    """Output of :func:`sphere_map`."""

    mesh: Mesh  # unit-sphere vertices (float32 values stored as float64), same faces
    iterations: int  # number of passes performed (>= 1)
    n_inverted: int  # faces with (V1-V0)x(V2-V0).(V0+V1+V2) < 0 at the end
    converged: bool  # True if the loop stopped because no face was inverted


@njit(cache=True, error_model="numpy")
def _count_inverted(v, faces, stop_at_first):
    c = 0
    for i in range(faces.shape[0]):
        a, b, d = faces[i, 0], faces[i, 1], faces[i, 2]
        ux, uy, uz = v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2]
        wx, wy, wz = v[d, 0] - v[a, 0], v[d, 1] - v[a, 1], v[d, 2] - v[a, 2]
        cx = uy * wz - uz * wy
        cy = uz * wx - ux * wz
        cz = ux * wy - uy * wx
        sx = v[a, 0] + v[b, 0] + v[d, 0]
        sy = v[a, 1] + v[b, 1] + v[d, 1]
        sz = v[a, 2] + v[b, 2] + v[d, 2]
        if cx * sx + cy * sy + cz * sz < 0:
            c += 1
            if stop_at_first:
                return c
    return c


@njit(cache=True, error_model="numpy")
def _sphere(v0, faces, indptr, indices, typ, max_iter):
    n = v0.shape[0]
    nf = faces.shape[0]
    v = v0.copy()
    f32 = np.float32
    # Mesh::centering (sequential float32 sums)
    cx = f32(0.0)
    cy = f32(0.0)
    cz = f32(0.0)
    for i in range(n):
        cx += v[i, 0]
        cy += v[i, 1]
        cz += v[i, 2]
    nn32 = f32(n)
    cx /= nn32
    cy /= nn32
    cz /= nn32
    for i in range(n):
        v[i, 0] = v[i, 0] - cx
        v[i, 1] = v[i, 1] - cy
        v[i, 2] = v[i, 2] - cz

    w = np.zeros(indices.shape[0], dtype=np.float32)
    if typ == 0:
        for i in range(nf):
            for j in range(3):
                idv = faces[i, j]
                id1 = faces[i, (j + 1) % 3]
                id2 = faces[i, (j + 2) % 3]
                px, py, pz = v[id1, 0] - v[idv, 0], v[id1, 1] - v[idv, 1], v[id1, 2] - v[idv, 2]
                qx, qy, qz = v[id2, 0] - v[idv, 0], v[id2, 1] - v[idv, 1], v[id2, 2] - v[idv, 2]
                kx = py * qz - pz * qy
                ky = pz * qx - px * qz
                kz = px * qy - py * qx
                nrm = f32(math.sqrt(kx * kx + ky * ky + kz * kz))
                weight = nrm / f32(2) / f32(12)
                for k in range(indptr[id1], indptr[id1 + 1]):
                    if indices[k] == id2:
                        w[k] += weight
                for k in range(indptr[id2], indptr[id2 + 1]):
                    if indices[k] == id1:
                        w[k] += weight
    else:
        for i in range(n):
            nn = indptr[i + 1] - indptr[i]
            for j in range(nn):
                id1 = indices[indptr[i] + j]
                id2 = indices[indptr[i] + (j + 1) % nn]
                px, py, pz = v[id1, 0] - v[i, 0], v[id1, 1] - v[i, 1], v[id1, 2] - v[i, 2]
                qx, qy, qz = v[id2, 0] - v[i, 0], v[id2, 1] - v[i, 1], v[id2, 2] - v[i, 2]
                sp = f32(math.sqrt(px * px + py * py + pz * pz))
                if sp > 0:
                    px /= sp
                    py /= sp
                    pz /= sp
                sq = f32(math.sqrt(qx * qx + qy * qy + qz * qz))
                if sq > 0:
                    qx /= sq
                    qy /= sq
                    qz /= sq
                x = np.float64(px * qx + py * qy + pz * qz)
                a = x / math.sqrt(1 - x * x)
                weight = 0.1 if a < 0.1 else a  # std::max(a, 1e-1)
                for k in range(indptr[id1], indptr[id1 + 1]):
                    if indices[k] == id2:
                        w[k] = f32(np.float64(w[k]) + weight)
                for k in range(indptr[id2], indptr[id2 + 1]):
                    if indices[k] == id1:
                        w[k] = f32(np.float64(w[k]) + weight)
    for i in range(n):
        s = f32(0.0)
        for k in range(indptr[i], indptr[i + 1]):
            s += w[k]
        for k in range(indptr[i], indptr[i + 1]):
            w[k] /= s

    for i in range(n):  # project to the unit sphere
        s = f32(math.sqrt(v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1] + v[i, 2] * v[i, 2]))
        if s > 0:
            v[i, 0] /= s
            v[i, 1] /= s
            v[i, 2] /= s

    newv = np.empty_like(v)
    it = 0
    ninv = True
    while True:
        for i in range(n):
            if typ == 0:
                ax, ay, az = v[i, 0], v[i, 1], v[i, 2]
            else:
                ax, ay, az = f32(0.0), f32(0.0), f32(0.0)
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                ax += v[j, 0] * w[k]
                ay += v[j, 1] * w[k]
                az += v[j, 2] * w[k]
            s = f32(math.sqrt(ax * ax + ay * ay + az * az))
            if s > 0:
                ax /= s
                ay /= s
                az /= s
            newv[i, 0] = ax
            newv[i, 1] = ay
            newv[i, 2] = az
        v, newv = newv, v
        ninv = _count_inverted(v, faces, True) > 0
        it += 1
        if not (ninv and it < max_iter):
            break
    return v, it


def count_inverted_faces(vertices, faces) -> int:
    """Number of faces with ``(V1-V0) x (V2-V0) . (V0+V1+V2) < 0`` (float32), the stopping test of
    :func:`sphere_map`.  Meaningful for (near-)unit-sphere vertices."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    return int(_count_inverted(v, f, False))


def sphere_map(
    mesh,
    faces=None,
    type: int = 0,  # noqa: A002 - MeshLib parameter name
    max_iter: int = 10000,
    *,
    neighbors: NeighborLists | None = None,
) -> SphereMapResult:
    """``SurfaceUtil::sphere(mesh, type, max_iter)``: map a genus-0 surface to the unit sphere.

    ``type=0`` area-preserving weights (default), any other value angle weights.  The input is
    not modified (MeshLib mutates the mesh in place).  The vertices are centred (float32 mean) and
    projected to the unit sphere before iterating.  ``max_iter`` is the cap on the number of
    passes; at least one pass is always performed.
    """
    if faces is None:
        if not hasattr(mesh, "faces"):
            raise TypeError("pass a Mesh, or (vertices, faces)")
        base = mesh
        verts, faces = mesh.vertices, mesh.faces
    else:
        base = None
        verts = mesh
    v = np.ascontiguousarray(verts, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("vertices must be (N, 3) and faces (M, 3)")
    nb = neighbors if neighbors is not None else vertex_neighbor_lists(f, v.shape[0])
    with np.errstate(all="ignore"):
        out, it = _sphere(v, f, nb.indptr, nb.indices, int(type), int(max_iter))
    n_inv = int(_count_inverted(out, f, False))
    if base is not None:
        res = Mesh(
            out.astype(np.float64),
            f,
            point_data={k: a.copy() for k, a in base.point_data.items()},
            frame=base.frame,
            units=base.units,
            meta=dict(base.meta),
        )
    else:
        res = Mesh(out.astype(np.float64), f)
    return SphereMapResult(res, int(it), n_inv, n_inv == 0 and it <= int(max_iter))
