"""``HSD::icosahedron``: the uniform sampling icosahedra of HSD (HSD.cpp 2890-3050).

Not the BISCUIT ``sphere_*.vtk`` ordering (``pybiscuit.sphere.icosphere``): HSD builds its own
subdivision with float32 vectors, a first-come edge map, and an orientation fix per child face.
The vertex order of the samples matters (``m_propertySamples``, feature order, smoothing).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, types
from numba.typed import Dict

from .geom import NJIT, F, _unit3, _cross3, dot3

__all__ = ["icosahedron", "icosahedron_counts"]


def icosahedron_counts(level: int) -> tuple[int, int]:
    return 10 * 4**level + 2, 20 * 4**level


@njit(cache=True)
def _base():
    t = F((1.0 + math.sqrt(5.0)) / 2.0)
    s = F(math.sqrt(np.float64(F(1) + t * t)))
    v = np.empty((12, 3), np.float32)
    one, zero = F(1), F(0)
    vals = (
        (t, one, zero), (-t, one, zero), (t, -one, zero), (-t, -one, zero),
        (one, zero, t), (one, zero, -t), (-one, zero, t), (-one, zero, -t),
        (zero, t, one), (zero, -t, one), (zero, t, -one), (zero, -t, -one),
    )  # fmt: skip
    for i in range(12):
        v[i, 0] = vals[i][0] / s
        v[i, 1] = vals[i][1] / s
        v[i, 2] = vals[i][2] / s
    return v


_BASE_FACES = np.array(
    [
        [0, 8, 4], [1, 10, 7], [2, 9, 11], [7, 3, 1], [0, 5, 10], [3, 9, 6], [3, 11, 9],
        [8, 6, 4], [2, 4, 9], [3, 7, 11], [4, 2, 0], [9, 4, 6], [2, 11, 5], [0, 10, 8],
        [5, 0, 2], [10, 5, 7], [1, 6, 8], [1, 8, 10], [6, 1, 3], [11, 7, 5],
    ],
    dtype=np.int32,
)  # fmt: skip


@njit(cache=True)
def _flip_if_inward(v, f, i1, i2):
    """swap(f[1], f[2]) when ``(v1 - v0) x (v2 - v0) . v0 < 0`` (float32)."""
    a, b, c = f[0], f[i1], f[i2]
    ux, uy, uz = v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2]
    wx, wy, wz = v[c, 0] - v[a, 0], v[c, 1] - v[a, 1], v[c, 2] - v[a, 2]
    cx, cy, cz = _cross3(ux, uy, uz, wx, wy, wz)
    return dot3(cx, cy, cz, v[a, 0], v[a, 1], v[a, 2]) < F(0)


@njit(cache=True, error_model="numpy")
def _subdivide(v0, f0, level):
    nv, nf = v0.shape[0], f0.shape[0]
    for _ in range(level):
        nv += 3 * nf // 2  # each face has 3 edges, shared by two faces
        nf *= 4
    v = np.zeros((nv, 3), np.float32)
    f = np.zeros((nf, 3), np.int32)
    v[: v0.shape[0]] = v0
    f[: f0.shape[0]] = f0
    nv, nf = v0.shape[0], f0.shape[0]
    tri = np.empty(3, np.int32)
    for _ in range(level):
        edges = Dict.empty(key_type=types.int64, value_type=types.int64)
        nfaces = nf
        for i in range(nfaces):
            a, b, c = f[i, 0], f[i, 1], f[i, 2]
            mids = np.empty(3, np.int64)
            ep = ((a, b), (c, a), (b, c))
            for e in range(3):
                p, q = ep[e]
                lo, hi = (p, q) if p <= q else (q, p)
                key = np.int64(lo) * 100000000 + np.int64(hi)
                if key in edges:
                    mids[e] = edges[key]
                else:
                    # a->v + b->v, then unit()
                    sx = v[p, 0] + v[q, 0]
                    sy = v[p, 1] + v[q, 1]
                    sz = v[p, 2] + v[q, 2]
                    ux, uy, uz = _unit3(sx, sy, sz)
                    v[nv, 0] = ux
                    v[nv, 1] = uy
                    v[nv, 2] = uz
                    edges[key] = nv
                    mids[e] = nv
                    nv += 1
            id1, id2, id3 = mids[0], mids[1], mids[2]
            # overwrite the original face with the middle triangle
            tri[0], tri[1], tri[2] = id1, id2, id3
            if _flip_if_inward(v, tri, 1, 2):
                tri[1], tri[2] = id3, id2
            f[i, 0], f[i, 1], f[i, 2] = tri[0], tri[1], tri[2]
            # (a, id1, id2), (c, id2, id3), (b, id3, id1)
            for k in range(3):
                if k == 0:
                    t0, t1, t2 = a, id1, id2
                elif k == 1:
                    t0, t1, t2 = c, id2, id3
                else:
                    t0, t1, t2 = b, id3, id1
                tri[0], tri[1], tri[2] = t0, t1, t2
                if _flip_if_inward(v, tri, 1, 2):
                    tri[1], tri[2] = t2, t1
                f[nf, 0], f[nf, 1], f[nf, 2] = tri[0], tri[1], tri[2]
                nf += 1
    return v[:nv], f[:nf]


def icosahedron(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (float32 (V,3)) and faces (int32 (F,3)) of ``HSD::icosahedron(level)``."""
    if level < 0:
        raise ValueError("level must be >= 0")
    v, f = _subdivide(_base(), _BASE_FACES, int(level))
    return np.ascontiguousarray(v), np.ascontiguousarray(f)
