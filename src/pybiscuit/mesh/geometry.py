"""MeshLib geometry port: normals, areas, MeshLib vertex neighbour lists, smoothing.

Port of ``MeshLib/src/Mesh.cpp`` (``Mesh::updateNormal``, ``Face::updateFaceNormal``,
``Mesh::connectivity``) and ``MeshLib/src/Util/SurfaceUtil.cpp`` (``smoothing`` overloads),
plus ``GeodesicA::vertexArea`` (the only vertex-area routine in MeshLib).
See ``docs/ports/meshlib-geometry.md``.

MeshLib stores vertices, normals and vertex data as C ``float``.  Everything here
first casts vertices to float32 and keeps float32 arithmetic where MeshLib does.

Public API (stable, shared with ``pybiscuit.mesh.curvature``)::

    NeighborLists(indptr, indices)                  # CSR, MeshLib ring ordering
    vertex_neighbor_lists(faces, n_vertices) -> NeighborLists
    face_normals(vertices, faces)  -> float32 (M, 3)
    vertex_normals(vertices, faces) -> float32 (N, 3)
    face_areas(vertices, faces)    -> float32 (M,)
    vertex_areas(vertices, faces, neighbors=None) -> float64 (N,)
    smooth_vertices(vertices, faces, iterations, sigma=None, neighbors=None) -> float32 (N, 3)
    smooth_data(faces, data, iterations, neighbors=None) -> data.dtype (N,)

Every function accepts ``vertices`` as a ``Mesh`` (then ``faces`` is omitted) or an
``(N, 3)`` array.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

__all__ = [
    "NeighborLists",
    "face_areas",
    "face_normals",
    "smooth_data",
    "smooth_vertices",
    "vertex_areas",
    "vertex_neighbor_lists",
    "vertex_normals",
]


class NeighborLists(NamedTuple):
    """CSR vertex neighbour lists in MeshLib's ``Vertex::list`` order."""

    indptr: np.ndarray  # int64 (N + 1,)
    indices: np.ndarray  # int32 (nnz,)

    def row(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i] : self.indptr[i + 1]]


F32 = np.float32


# --------------------------------------------------------------------------- #
# input coercion
# --------------------------------------------------------------------------- #
def _coerce(vertices, faces):
    """``(vertices float32 (N,3) C-contiguous, faces int32 (M,3))`` from Mesh or arrays."""
    if faces is None:
        if not hasattr(vertices, "faces"):
            raise TypeError("pass a Mesh, or (vertices, faces)")
        vertices, faces = vertices.vertices, vertices.faces
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if f.size and (f.min() < 0 or f.max() >= v.shape[0]):
        raise ValueError("face index out of range")
    return v, f


def _nbrs(faces, n, neighbors):
    return neighbors if neighbors is not None else vertex_neighbor_lists(faces, n)


# --------------------------------------------------------------------------- #
# float32 vector helpers (MeshLib ``Vector``): every intermediate stays float32
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy", inline="always")
def _dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@njit(cache=True, error_model="numpy", inline="always")
def _norm(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


# --------------------------------------------------------------------------- #
# Mesh::connectivity
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _connectivity(faces, n):
    nf = faces.shape[0]
    cnt = np.zeros(n, dtype=np.int64)
    for i in range(nf):
        for j in range(3):
            cnt[faces[i, j]] += 1
    start = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        start[i + 1] = start[i] + cnt[i]
    e_idx = np.empty(start[n], dtype=np.int32)
    e_next = np.empty(start[n], dtype=np.int32)
    fill = np.zeros(n, dtype=np.int64)
    for i in range(nf):
        a, b, c = faces[i, 0], faces[i, 1], faces[i, 2]
        p = start[a] + fill[a]
        e_idx[p] = b
        e_next[p] = c
        fill[a] += 1
        p = start[b] + fill[b]
        e_idx[p] = c
        e_next[p] = a
        fill[b] += 1
        p = start[c] + fill[c]
        e_idx[p] = a
        e_next[p] = b
        fill[c] += 1
    # each boundary "not found" adds one entry; at most one per original entry
    out = np.empty(2 * start[n], dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int64)
    pos = 0
    for i in range(n):
        s0 = start[i]
        m = cnt[i]
        visited = np.zeros(m, dtype=np.bool_)
        nn = m
        nxt = 0
        j = 0
        while nxt < m:
            out[pos + j] = e_idx[s0 + nxt]
            j += 1
            visited[nxt] = True
            found = False
            target = e_next[s0 + nxt]
            for k in range(m):
                if e_idx[s0 + k] == target:
                    nxt = k
                    found = True
                    break
            if not found:
                nn += 1
                out[pos + j] = target
                j += 1
            if visited[nxt]:
                nxt = 0
                while nxt < m:
                    if not visited[nxt]:
                        break
                    nxt += 1
                if nxt == m:
                    break
        pos += j
        indptr[i + 1] = pos
    return indptr, out[:pos].copy()


def vertex_neighbor_lists(faces, n_vertices: int | None = None) -> NeighborLists:
    """MeshLib ``Mesh::connectivity``: ordered one-ring per vertex (boundary-aware).

    Interior vertices get a counter-clockwise-by-face-winding ring starting at the
    vertex's first face; boundary vertices get the open fan starting from the first
    face in file order (plus the closing vertex), as in ``Mesh.cpp`` L819-L911.
    Order matters: float32 smoothing sums in this order.
    """
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    n = int(n_vertices) if n_vertices is not None else (int(f.max()) + 1 if f.size else 0)
    indptr, indices = _connectivity(f, n)
    return NeighborLists(indptr, indices)


# --------------------------------------------------------------------------- #
# normals
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _face_normals(v, faces):
    nf = faces.shape[0]
    out = np.zeros((nf, 3), dtype=np.float32)
    for i in range(nf):
        a, b, c = faces[i, 0], faces[i, 1], faces[i, 2]
        # v1 = V1 - V0, v2 = V2 - V1 (Vector(p, q) = q - p)
        ux, uy, uz = v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2]
        wx, wy, wz = v[c, 0] - v[b, 0], v[c, 1] - v[b, 1], v[c, 2] - v[b, 2]
        nx = uy * wz - uz * wy
        ny = uz * wx - ux * wz
        nz = ux * wy - uy * wx
        s = _norm(nx, ny, nz)
        if s > 0:
            nx /= s
            ny /= s
            nz /= s
        out[i, 0], out[i, 1], out[i, 2] = nx, ny, nz
    return out


@njit(cache=True, error_model="numpy")
def _vertex_normals(v, faces):
    n = v.shape[0]
    acc = np.zeros((n, 3), dtype=np.float32)
    one = np.float32(1.0)
    for i in range(faces.shape[0]):
        i0, i1, i2 = faces[i, 0], faces[i, 1], faces[i, 2]
        # e1 = V0 - V1, e2 = V1 - V2, e3 = V2 - V0
        e1x, e1y, e1z = v[i0, 0] - v[i1, 0], v[i0, 1] - v[i1, 1], v[i0, 2] - v[i1, 2]
        e2x, e2y, e2z = v[i1, 0] - v[i2, 0], v[i1, 1] - v[i2, 1], v[i1, 2] - v[i2, 2]
        e3x, e3y, e3z = v[i2, 0] - v[i0, 0], v[i2, 1] - v[i0, 1], v[i2, 2] - v[i0, 2]
        fx = e1y * e2z - e1z * e2y  # FN = e1 x e2 (unnormalised)
        fy = e1z * e2x - e1x * e2z
        fz = e1x * e2y - e1y * e2x
        s = _norm(e1x, e1y, e1z)
        if s > 0:
            e1x /= s
            e1y /= s
            e1z /= s
        s = _norm(e2x, e2y, e2z)
        if s > 0:
            e2x /= s
            e2y /= s
            e2z /= s
        s = _norm(e3x, e3y, e3z)
        if s > 0:
            e3x /= s
            e3y /= s
            e3z /= s
        # e1 * -e3, e2 * -e1, e3 * -e2 (negation is exact)
        e13 = _dot(e1x, e1y, e1z, -e3x, -e3y, -e3z)
        e13 = min(max(e13, -one), one)
        e21 = _dot(e2x, e2y, e2z, -e1x, -e1y, -e1z)
        e21 = min(max(e21, -one), one)
        e32 = _dot(e3x, e3y, e3z, -e2x, -e2y, -e2z)
        e32 = min(max(e32, -one), one)
        w1 = math.acos(e13)
        w2 = math.acos(e21)
        w3 = math.acos(e32)
        acc[i0, 0] += fx * w1
        acc[i0, 1] += fy * w1
        acc[i0, 2] += fz * w1
        acc[i1, 0] += fx * w2
        acc[i1, 1] += fy * w2
        acc[i1, 2] += fz * w2
        acc[i2, 0] += fx * w3
        acc[i2, 1] += fy * w3
        acc[i2, 2] += fz * w3
    for i in range(n):
        s = _norm(acc[i, 0], acc[i, 1], acc[i, 2])
        if s > 0:
            acc[i, 0] /= s
            acc[i, 1] /= s
            acc[i, 2] /= s
    return acc


def face_normals(vertices, faces=None) -> np.ndarray:
    """Unit face normals (``Face::updateFaceNormal``), float32 (M, 3).

    ``unit(cross(V1 - V0, V2 - V1))``; a zero-length normal stays zero.
    """
    v, f = _coerce(vertices, faces)
    return _face_normals(v, f)


def vertex_normals(vertices, faces=None) -> np.ndarray:
    """Angle-weighted vertex normals (``Mesh::updateNormal``), float32 (N, 3).

    Each face adds its *unnormalised* normal (twice the area vector) times the
    interior angle at the vertex; the sum is normalised.  Vertices in no face get 0.
    """
    v, f = _coerce(vertices, faces)
    return _vertex_normals(v, f)


# --------------------------------------------------------------------------- #
# areas
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _face_areas(v, faces):
    nf = faces.shape[0]
    out = np.empty(nf, dtype=np.float32)
    two = np.float32(2.0)
    for i in range(nf):
        a, b, c = faces[i, 0], faces[i, 1], faces[i, 2]
        px, py, pz = v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2]
        qx, qy, qz = v[c, 0] - v[a, 0], v[c, 1] - v[a, 1], v[c, 2] - v[a, 2]
        cx = py * qz - pz * qy
        cy = pz * qx - px * qz
        cz = px * qy - py * qx
        out[i] = _norm(cx, cy, cz) / two
    return out


@njit(cache=True, error_model="numpy")
def _vertex_areas(v, indptr, indices):
    n = v.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nn = indptr[i + 1] - indptr[i]
        s = 0.0
        for j in range(nn):
            i1 = indices[indptr[i] + j]
            i2 = indices[indptr[i] + (j + 1) % nn]
            e1x, e1y, e1z = v[i1, 0] - v[i, 0], v[i1, 1] - v[i, 1], v[i1, 2] - v[i, 2]
            e2x, e2y, e2z = v[i2, 0] - v[i, 0], v[i2, 1] - v[i, 1], v[i2, 2] - v[i, 2]
            a = np.float64(_norm(e1x, e1y, e1z))
            b = np.float64(_norm(e2x, e2y, e2z))
            n1 = _norm(e1x, e1y, e1z)
            if n1 > 0:
                e1x /= n1
                e1y /= n1
                e1z /= n1
            n2 = _norm(e2x, e2y, e2z)
            if n2 > 0:
                e2x /= n2
                e2y /= n2
                e2z /= n2
            inner = np.float64(_dot(e1x, e1y, e1z, e2x, e2y, e2z))
            if inner > 1.0:
                inner = 1.0
            if inner < -1.0:
                inner = -1.0
            theta = math.acos(inner)
            s += a * b * math.sin(theta) / 2.0 / 3.0
        out[i] = s
    return out


def face_areas(vertices, faces=None) -> np.ndarray:
    """Triangle areas as MeshLib computes them in ``SurfaceUtil::sphere`` (float32, (M,)).

    ``|(V1 - V0) x (V2 - V0)| / 2`` with float32 arithmetic.  MeshLib has no
    stand-alone face-area function; this is its inline expression (without the /12).
    """
    v, f = _coerce(vertices, faces)
    return _face_areas(v, f)


def vertex_areas(vertices, faces=None, neighbors: NeighborLists | None = None) -> np.ndarray:
    """``GeodesicA::vertexArea``: sum over ring triangles of area / 3, float64 (N,).

    This is MeshLib's only vertex area (1/3 of the incident triangle areas, i.e.
    barycentric, not Voronoi).  Edge vectors / norms are float32, the rest double.
    For boundary vertices the ring wraps from the last neighbour to the first, so
    one spurious "triangle" closes the open fan, exactly as in MeshLib.
    """
    v, f = _coerce(vertices, faces)
    nb = _nbrs(f, v.shape[0], neighbors)
    return _vertex_areas(v, nb.indptr, nb.indices)


# --------------------------------------------------------------------------- #
# smoothing (SurfaceUtil.cpp L18-L35, L255-L331)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _smooth_mean(v, indptr, indices, iterations):
    n = v.shape[0]
    cur = v.copy()
    new = np.zeros_like(cur)
    for _ in range(iterations):
        for i in range(n):
            nn = indptr[i + 1] - indptr[i]
            x = np.float32(0.0)
            y = np.float32(0.0)
            z = np.float32(0.0)
            for j in range(indptr[i], indptr[i + 1]):
                k = indices[j]
                x += cur[k, 0]
                y += cur[k, 1]
                z += cur[k, 2]
            d = np.float32(nn)
            new[i, 0] = x / d
            new[i, 1] = y / d
            new[i, 2] = z / d
        cur, new = new, cur
    return cur


@njit(cache=True, error_model="numpy")
def _normal_pdf(x, sigma):
    # Statistics::normal_pdf(float x, float u=0, float sigma): float exp(), double sqrt/denominator
    xm = x - np.float32(0.0)
    num = np.float32(-1.0) * xm * xm
    e = math.exp(num / (np.float32(2.0) * sigma * sigma))  # float32 in, float32 out
    den = np.float64(sigma) * math.sqrt(2.0 * 3.141592653589793238462643)
    return np.float32(np.float64(e) / den)


@njit(cache=True, error_model="numpy")
def _smooth_sigma(v, indptr, indices, sigma, iterations):
    n = v.shape[0]
    cur = v.copy()
    new = np.zeros_like(cur)
    for _ in range(iterations):
        for i in range(n):
            x = np.float32(0.0)
            y = np.float32(0.0)
            z = np.float32(0.0)
            weight = np.float32(0.0)
            for j in range(indptr[i], indptr[i + 1]):
                k = indices[j]
                dist = _norm(cur[i, 0] - cur[k, 0], cur[i, 1] - cur[k, 1], cur[i, 2] - cur[k, 2])
                w = _normal_pdf(dist, sigma)
                weight += w
                x += cur[k, 0] * w
                y += cur[k, 1] * w
                z += cur[k, 2] * w
            new[i, 0] = x / weight
            new[i, 1] = y / weight
            new[i, 2] = z / weight
        cur, new = new, cur
    return cur


@njit(cache=True, error_model="numpy")
def _smooth_data32(d, indptr, indices, iterations):
    n = d.shape[0]
    cur = d.copy()
    new = np.zeros_like(cur)
    for _ in range(iterations):
        for i in range(n):
            s = np.float32(0.0)
            for j in range(indptr[i], indptr[i + 1]):
                s += cur[indices[j]]
            new[i] = s / np.float32(indptr[i + 1] - indptr[i])
        cur, new = new, cur
    return cur


@njit(cache=True, error_model="numpy")
def _smooth_data64(d, indptr, indices, iterations):
    n = d.shape[0]
    cur = d.copy()
    new = np.zeros_like(cur)
    for _ in range(iterations):
        for i in range(n):
            s = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                s += cur[indices[j]]
            new[i] = s / float(indptr[i + 1] - indptr[i])
        cur, new = new, cur
    return cur


def smooth_vertices(
    vertices,
    faces=None,
    iterations: int = 1,
    sigma: float | None = None,
    neighbors: NeighborLists | None = None,
) -> np.ndarray:
    """``SurfaceUtil::smoothing(mesh, iter)`` (sigma None) or ``(mesh, sigma, iter)``.

    Returns the new float32 vertices (the input is not modified).  ``sigma is None``:
    each vertex moves to the plain mean of its ring neighbours (Jacobi update, the
    vertex itself is not included).  Otherwise neighbours are weighted by the
    Gaussian pdf of the edge length with standard deviation ``sigma`` (cast to
    float32 as in MeshLib) and the weighted mean is taken.  Neighbour lists are
    those of the input topology (fixed).  Vertex normals after smoothing are
    ``vertex_normals(result, faces)`` (MeshLib calls ``updateNormal`` each pass).
    Isolated vertices (no neighbours) become NaN, as in MeshLib (0/0).
    """
    v, f = _coerce(vertices, faces)
    nb = _nbrs(f, v.shape[0], neighbors)
    it = int(iterations)
    with np.errstate(all="ignore"):
        if sigma is None:
            return _smooth_mean(v, nb.indptr, nb.indices, it)
        return _smooth_sigma(v, nb.indptr, nb.indices, F32(sigma), it)


def smooth_data(
    faces, data, iterations: int = 1, neighbors: NeighborLists | None = None
) -> np.ndarray:
    """``SurfaceUtil::smoothing(mesh, iter, float*/double* data)``; dtype preserved.

    Per iteration each value becomes the plain mean of its ring neighbours' values
    (Jacobi).  ``data`` float32 uses float32 sums (the ``float*`` overload); float64
    uses the ``double*`` overload; other dtypes are converted to float64.  Returns a new
    array.  ``faces`` may be a Mesh.
    """
    if hasattr(faces, "faces"):
        faces = faces.faces
    f = np.ascontiguousarray(faces, dtype=np.int32)
    d = np.asarray(data)
    d = np.ascontiguousarray(d, dtype=np.float32 if d.dtype == np.float32 else np.float64)
    nb = _nbrs(f, d.shape[0], neighbors)
    if d.ndim != 1:
        raise ValueError("data must be 1-D")
    fn = _smooth_data32 if d.dtype == np.float32 else _smooth_data64
    return fn(d, nb.indptr, nb.indices, int(iterations))
