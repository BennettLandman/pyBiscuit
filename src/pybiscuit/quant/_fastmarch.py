"""Fast-marching front propagation on a triangle mesh (MeshLib ``Geodesic`` / ``GeodesicA``).

Port of the Peyre ``GW_GeodesicMesh`` / Lyu ``GW_GeodesicMeshA`` fast marching used by
``Gyrification`` (``MeshLib/include/Geodesic/gw/gw_geodesic/GW_GeodesicMesh{,A}.inl``,
``src/Geodesic/{Geodesic,GeodesicA}.cpp``, commit ``dcef807``).  Two flavours:

* isotropic (``Geodesic``): Sethian update, speed 1; distance to the sulcal/gyral curves;
* anisotropic (``GeodesicA``): Qian "fast sweeping" tensor update, area-limited stopping rule;
  used for the kernel of every vertex.

All state is float64 (MeshLib ``GW_Float`` = double; vertex positions are the float32 file
coordinates widened to double).  Divisions ``v /= f`` in MeshLib ``GW_VectorStatic`` are
multiplications by ``1/f`` (and are skipped for ``f == 0``); that is reproduced because it changes
last bits.  The std heap is reproduced (``libc++`` by default, ``libstdc++`` selectable) because
exact distance ties are broken by heap order: the oracle was built with libc++, the Linux image
binary uses libstdc++.  Closed manifold meshes only (open fans raise).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

__all__ = ["GWMesh", "build_gw_mesh", "kernel_range", "propagate_iso", "set_tensor"]

GW_INF = 1e9
GW_EPS = 1e-9
FAR, ALIVE, DEAD = 0, 1, 2
HEAP_LIBCXX, HEAP_LIBSTDCXX = 0, 1


class GWMesh(NamedTuple):
    pos: np.ndarray  # (N,3) float64
    faces: np.ndarray  # (M,3) int32
    fnbr: np.ndarray  # (M,3) int32, neighbour across the edge opposite slot i
    vf_ptr: np.ndarray  # CSR vertex -> faces (ascending face id)
    vf_idx: np.ndarray
    ring_ptr: np.ndarray  # CSR vertex -> neighbours in GW_VertexIterator order
    ring_idx: np.ndarray
    normal: np.ndarray  # (N,3) float64 BuildRawNormal


# --------------------------------------------------------------------------- #
# small vector helpers (tuples of float64)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy", inline="always")
def _dot(a, b):
    r = 0.0
    r += a[0] * b[0]
    r += a[1] * b[1]
    r += a[2] * b[2]
    return r


@njit(cache=True, error_model="numpy", inline="always")
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


@njit(cache=True, error_model="numpy", inline="always")
def _mul(a, f):
    return (a[0] * f, a[1] * f, a[2] * f)


@njit(cache=True, error_model="numpy", inline="always")
def _div(a, f):
    """``GW_VectorStatic::operator/=``: multiply by 1/f, unchanged when f == 0."""
    if f != 0:
        g = 1.0 / f
        return (a[0] * g, a[1] * g, a[2] * g)
    return a


@njit(cache=True, error_model="numpy", inline="always")
def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@njit(cache=True, error_model="numpy", inline="always")
def _norm(a):
    return math.sqrt(_dot(a, a))


@njit(cache=True, error_model="numpy", inline="always")
def _p(pos, i):
    return (pos[i, 0], pos[i, 1], pos[i, 2])


# --------------------------------------------------------------------------- #
# mesh connectivity (GW_Mesh::BuildConnectivity, GW_VertexIterator, BuildRawNormal)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _edge_number(faces, f, v1, v2):
    """``GW_Face::GetEdgeNumber(Vert1, Vert2)``: slot opposite the edge (0 if absent)."""
    for i in range(3):
        if faces[f, i] == v1:
            if faces[f, (i + 1) % 3] == v2:
                return (i + 2) % 3
            if faces[f, (i + 2) % 3] == v2:
                return (i + 1) % 3
    return 0


@njit(cache=True, error_model="numpy", inline="always")
def _third(faces, f, v1, v2):
    """``GW_Face::GetVertex(Vert1, Vert2)``."""
    return faces[f, _edge_number(faces, f, v1, v2)]


@njit(cache=True, error_model="numpy", inline="always")
def _face_nbr_of(faces, fnbr, f, v):
    """``GW_Face::GetFaceNeighbor(Vert)``: neighbour opposite ``v`` (-1 if none)."""
    for i in range(3):
        if faces[f, i] == v:
            return fnbr[f, i]
    return -1


@njit(cache=True, error_model="numpy")
def _connectivity(pos, faces):
    n = pos.shape[0]
    nf = faces.shape[0]
    vf_ptr = np.zeros(n + 1, dtype=np.int64)
    for f in range(nf):
        for i in range(3):
            vf_ptr[faces[f, i] + 1] += 1
    for v in range(n):
        vf_ptr[v + 1] += vf_ptr[v]
    fill = vf_ptr[:-1].copy()
    vf_idx = np.empty(vf_ptr[n], dtype=np.int32)
    for f in range(nf):
        for i in range(3):
            v = faces[f, i]
            vf_idx[fill[v]] = f
            fill[v] += 1
    fnbr = -np.ones((nf, 3), dtype=np.int32)
    for f in range(nf):
        for i in range(3):
            v1 = faces[f, (i + 1) % 3]
            v2 = faces[f, (i + 2) % 3]
            nb = -1
            for a in range(vf_ptr[v1], vf_ptr[v1 + 1]):
                fa = vf_idx[a]
                if fa == f:
                    continue
                for b in range(vf_ptr[v2], vf_ptr[v2 + 1]):
                    if vf_idx[b] == fa:
                        nb = fa
                        break
                if nb >= 0:
                    break
            fnbr[f, i] = nb
            if nb >= 0:
                fnbr[nb, _edge_number(faces, nb, v1, v2)] = f
    # vertex rings (first face = lowest face id containing the vertex, direction = next vertex)
    ring_ptr = np.zeros(n + 1, dtype=np.int64)
    ring_idx = np.empty(vf_ptr[n], dtype=np.int32)  # closed manifold: valence == #faces
    normal = np.zeros((n, 3), dtype=np.float64)
    pos_ring = 0
    for v in range(n):
        ring_ptr[v] = pos_ring
        if vf_ptr[v + 1] == vf_ptr[v]:
            continue
        f0 = vf_idx[vf_ptr[v]]
        k = 0
        for i in range(3):
            if faces[f0, i] == v:
                k = i
        d = faces[f0, (k + 1) % 3]
        f = f0
        nx = 0.0
        ny = 0.0
        nz = 0.0
        it = 0
        cnt = 0
        while True:
            if pos_ring >= ring_idx.shape[0] or cnt > vf_ptr[v + 1] - vf_ptr[v]:
                return vf_ptr, vf_idx, fnbr, ring_ptr, ring_idx, normal, False
            ring_idx[pos_ring] = d
            pos_ring += 1
            cnt += 1
            # BuildRawNormal accumulates the first 21 faces of the ring (FaceIterator order)
            if it <= 20:
                p0 = _p(pos, faces[f, 0])
                p1 = _p(pos, faces[f, 1])
                p2 = _p(pos, faces[f, 2])
                fn = _cross(_sub(p0, p1), _sub(p0, p2))
                nn = _norm(fn)
                if nn < GW_EPS:
                    fn = (1.0, 0.0, 0.0)
                else:
                    fn = _div(fn, nn)
                nx += fn[0]
                ny += fn[1]
                nz += fn[2]
                it += 1
            nf_ = _face_nbr_of(faces, fnbr, f, d)
            if nf_ < 0:
                return vf_ptr, vf_idx, fnbr, ring_ptr, ring_idx, normal, False
            if nf_ == f0:
                break
            d = _third(faces, f, v, d)
            f = nf_
        nv = (nx, ny, nz)
        nn = _norm(nv)
        if nn < GW_EPS:
            nv = (1.0, 0.0, 0.0)
        else:
            nv = _div(nv, nn)
        normal[v, 0] = nv[0]
        normal[v, 1] = nv[1]
        normal[v, 2] = nv[2]
    ring_ptr[n] = pos_ring
    return vf_ptr, vf_idx, fnbr, ring_ptr, ring_idx, normal, True


def build_gw_mesh(vertices, faces) -> GWMesh:
    """Connectivity, vertex rings, raw normals as MeshLib builds them (float32 vertices)."""
    pos = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32), dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    vf_ptr, vf_idx, fnbr, ring_ptr, ring_idx, normal, ok = _connectivity(pos, f)
    if not ok:
        raise ValueError("fast marching requires a closed, manifold triangle mesh (no open fans)")
    return GWMesh(pos, f, fnbr, vf_ptr, vf_idx, ring_ptr, ring_idx[: ring_ptr[-1]].copy(), normal)


# --------------------------------------------------------------------------- #
# std heap (comparator: a is "less" when dist[a] > dist[b], i.e. a min-heap on distance)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy", inline="always")
def _comp(dist, a, b):
    return dist[a] > dist[b]


@njit(cache=True, error_model="numpy")
def _sift_down_cxx(h, dist, ln, start):
    """libc++ ``__sift_down`` (used by make_heap)."""
    child = start
    if ln < 2 or (ln - 2) // 2 < child:
        return
    child = 2 * child + 1
    ci = child
    if child + 1 < ln and _comp(dist, h[ci], h[ci + 1]):
        ci += 1
        child += 1
    if _comp(dist, h[ci], h[start]):
        return
    top = h[start]
    st = start
    while True:
        h[st] = h[ci]
        st = ci
        if (ln - 2) // 2 < child:
            break
        child = 2 * child + 1
        ci = child
        if child + 1 < ln and _comp(dist, h[ci], h[ci + 1]):
            ci += 1
            child += 1
        if _comp(dist, h[ci], top):
            break
    h[st] = top


@njit(cache=True, error_model="numpy")
def _sift_up_cxx(h, dist, ln):
    """libc++ ``__sift_up`` over the first ``ln`` elements."""
    if ln > 1:
        ln2 = (ln - 2) // 2
        last = ln - 1
        ptr = ln2
        if _comp(dist, h[ptr], h[last]):
            t = h[last]
            while True:
                h[last] = h[ptr]
                last = ptr
                if ln2 == 0:
                    break
                ln2 = (ln2 - 1) // 2
                ptr = ln2
                if not _comp(dist, h[ptr], t):
                    break
            h[last] = t


@njit(cache=True, error_model="numpy")
def _adjust_heap_gnu(h, dist, hole, ln, value):
    """libstdc++ ``__adjust_heap`` (with the trailing ``__push_heap``)."""
    top = hole
    second = hole
    while second < (ln - 1) // 2:
        second = 2 * (second + 1)
        if _comp(dist, h[second], h[second - 1]):
            second -= 1
        h[hole] = h[second]
        hole = second
    if (ln & 1) == 0 and second == (ln - 2) // 2:
        second = 2 * (second + 1)
        h[hole] = h[second - 1]
        hole = second - 1
    parent = (hole - 1) // 2
    while hole > top and _comp(dist, h[parent], value):
        h[hole] = h[parent]
        hole = parent
        parent = (hole - 1) // 2
    h[hole] = value


@njit(cache=True, error_model="numpy")
def make_heap(h, dist, ln, kind):
    if ln < 2:
        return
    if kind == HEAP_LIBCXX:
        for start in range((ln - 2) // 2, -1, -1):
            _sift_down_cxx(h, dist, ln, start)
    else:
        parent = (ln - 2) // 2
        while True:
            _adjust_heap_gnu(h, dist, parent, ln, h[parent])
            if parent == 0:
                return
            parent -= 1


@njit(cache=True, error_model="numpy")
def push_heap(h, dist, ln, kind):
    """The new element is at ``h[ln-1]``."""
    if kind == HEAP_LIBCXX:
        _sift_up_cxx(h, dist, ln)
    else:
        hole = ln - 1
        value = h[hole]
        parent = (hole - 1) // 2
        while hole > 0 and _comp(dist, h[parent], value):
            h[hole] = h[parent]
            hole = parent
            parent = (hole - 1) // 2
        h[hole] = value


@njit(cache=True, error_model="numpy")
def pop_heap(h, dist, ln, kind):
    """Moves the minimum to ``h[ln-1]`` (caller then shrinks the heap)."""
    if ln < 2:
        return
    if kind == HEAP_LIBCXX:  # __pop_heap with __floyd_sift_down
        top = h[0]
        hole = 0
        child = 0
        while True:
            ci = hole + child + 1
            child = 2 * child + 1
            if child + 1 < ln and _comp(dist, h[ci], h[ci + 1]):
                ci += 1
                child += 1
            h[hole] = h[ci]
            hole = ci
            if child > (ln - 2) // 2:
                break
        last = ln - 1
        if hole == last:
            h[hole] = top
        else:
            h[hole] = h[last]
            h[last] = top
            _sift_up_cxx(h, dist, hole + 1)
    else:
        value = h[ln - 1]
        h[ln - 1] = h[0]
        _adjust_heap_gnu(h, dist, 0, ln - 1, value)


# --------------------------------------------------------------------------- #
# Sethian update (isotropic) and Qian update (anisotropic)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _update_sethian(d1, d2, a, b, dot, F):
    t = GW_INF
    cosa = dot
    sina = math.sqrt(1 - dot * dot)
    u = d2 - d1
    f2 = a * a + b * b - 2 * a * b * cosa
    f1 = b * u * (a * cosa - b)
    f0 = b * b * (u * u - F * F * a * a * sina * sina)
    delta = f1 * f1 - f0 * f2
    if delta >= 0:
        if abs(f2) > GW_EPS:
            t = (-f1 - math.sqrt(delta)) / f2
            if t < u or b * (t - u) / t < a * cosa or a / cosa < b * (t - u) / t:
                t = (-f1 + math.sqrt(delta)) / f2
        else:
            if f1 != 0:
                t = -f0 / f1
            else:
                t = -GW_INF
    else:
        t = -GW_INF
    if u < t and a * cosa < b * (t - u) / t and b * (t - u) / t < a / cosa:
        return t + d1
    bb = b * F + d1
    aa = a * F + d2
    return bb if bb < aa else aa


@njit(cache=True, error_model="numpy")
def _tri_inside(xa, ya, xb, yb, vx, vy):
    dot00 = xa * xa + ya * ya
    dot01 = xa * xb + ya * yb
    dot02 = xa * vx + ya * vy
    dot11 = xb * xb + yb * yb
    dot12 = xb * vx + yb * vy
    inv = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u * v >= 0


@njit(cache=True, error_model="numpy")
def _update_fsm(C, P1, P2, d1, d2, dot, F1, F2, v1, a, b):
    """``ComputeUpdate_FastSweepingMethod``; C current position, P1/P2 the two other corners."""
    c = 0.0
    A = _sub(P1, C)
    B = _sub(P2, C)
    N = _cross(A, B)
    N = _div(N, _norm(N))
    p = _dot(N, v1)
    U1 = _sub(v1, _mul(N, p))
    U1 = _div(U1, _norm(U1))
    U2 = _cross(N, U1)
    U2 = _div(U2, _norm(U2))
    xa = _dot(U1, A)
    xb = _dot(U1, B)
    xc = 0.0
    ya = _dot(U2, A)
    yb = _dot(U2, B)
    yc = 0.0
    lb = math.sqrt(xa * xa + ya * ya + 0.0 * 0.0)
    la = math.sqrt(xb * xb + yb * yb + 0.0 * 0.0)
    det = 1 / (1 - dot * dot)
    n11 = (xc - xa) / lb
    n12 = (yc - ya) / lb
    n21 = (xc - xb) / la
    n22 = (yc - yb) / la
    p11 = (n11 - n21 * dot) * det
    p12 = (n21 - n11 * dot) * det
    p21 = (n12 - n22 * dot) * det
    p22 = (n22 - n12 * dot) * det
    g1 = p11 / lb + p12 / la
    g2 = -(p11 / lb * d1 + p12 / la * d2)
    g3 = p21 / lb + p22 / la
    g4 = -(p21 / lb * d1 + p22 / la * d2)
    w1 = a * g1 * g1 + b * g3 * g3 - 2 * c * g1 * g3
    w2 = a * g1 * g2 + b * g3 * g4 - c * (g1 * g4 + g2 * g3)
    w3 = a * g2 * g2 + b * g4 * g4 - 2 * c * g2 * g4 - 1 / a / b
    disc = w2 * w2 - w1 * w3
    dm = d1 if d1 > d2 else d2
    if disc >= 0:
        if abs(w1) < GW_EPS:
            t = -w3 / w2
            if t > dm:
                return t
        else:
            r1 = -w2 / w1
            r2 = math.sqrt(disc) / w1
            t = r1 - r2
            if t > dm:
                pp = g1 * t + g2
                qq = g3 * t + g4
                if _tri_inside(xa, ya, xb, yb, a * pp - c * qq, b * qq - c * pp):
                    return t
            t = r1 + r2
            if t > dm:
                pp = g1 * t + g2
                qq = g3 * t + g4
                if _tri_inside(xa, ya, xb, yb, a * pp - c * qq, b * qq - c * pp):
                    return t
    x = lb * F1 + d1
    y = la * F2 + d2
    return x if x < y else y


@njit(cache=True, error_model="numpy")
def _unfold(m_pos, faces, fnbr, f, v, v1, v2):
    """``UnfoldTriangle``: returns (vertex or -1, dist, dot1, dot2)."""
    pv = _p(m_pos, v)
    e1 = _sub(_p(m_pos, v1), pv)
    n1 = _norm(e1)
    e1 = _div(e1, n1)
    e2 = _sub(_p(m_pos, v2), pv)
    n2 = _norm(e2)
    e2 = _div(e2, n2)
    dot = _dot(e1, e2)
    eq1x = dot
    eq1y = math.sqrt(1 - dot * dot)
    eq2x = 1.0
    eq2y = 0.0
    x1x = n1
    x1y = 0.0
    x2x = eq1x * n2
    x2y = eq1y * n2
    xs1x, xs1y, xs2x, xs2y = x1x, x1y, x2x, x2y
    pV1 = v1
    pV2 = v2
    cur = _face_nbr_of(faces, fnbr, f, v)
    num = 0
    while num < 50 and cur >= 0:
        pV = _third(faces, cur, pV1, pV2)
        e1 = _sub(_p(m_pos, pV2), _p(m_pos, pV1))
        rn1 = _norm(e1)
        e1 = _div(e1, rn1)
        e2 = _sub(_p(m_pos, pV), _p(m_pos, pV1))
        rn2 = _norm(e2)
        e2 = _div(e2, rn2)
        # vv = (x2 - x1) * rn2 / rn1   (vector / f == multiply by 1/f, zero vector if f == 0)
        vvx = (x2x - x1x) * rn2
        vvy = (x2y - x1y) * rn2
        if rn1 != 0:
            g = 1.0 / rn1
            vvx = vvx * g
            vvy = vvy * g
        else:
            vvx = 0.0
            vvy = 0.0
        dot = _dot(e1, e2)
        ang = -math.acos(dot)
        rx = math.cos(ang) * vvx - math.sin(ang) * vvy
        ry = math.sin(ang) * vvx + math.cos(ang) * vvy
        xx = rx + x1x
        xy = ry + x1y
        lam11 = -(x1x * eq1x + x1y * eq1y) / ((xx - x1x) * eq1x + (xy - x1y) * eq1y)
        lam12 = -(x1x * eq2x + x1y * eq2y) / ((xx - x1x) * eq2x + (xy - x1y) * eq2y)
        lam21 = -(x2x * eq1x + x2y * eq1y) / ((xx - x2x) * eq1x + (xy - x2y) * eq1y)
        lam22 = -(x2x * eq2x + x2y * eq2y) / ((xx - x2x) * eq2x + (xy - x2y) * eq2y)
        i11 = (lam11 >= 0) and (lam11 <= 1)
        i12 = (lam12 >= 0) and (lam12 <= 1)
        i21 = (lam21 >= 0) and (lam21 <= 1)
        i22 = (lam22 >= 0) and (lam22 <= 1)
        if i11 and i12:
            cur = _face_nbr_of(faces, fnbr, cur, pV2)
            pV2 = pV
            x2x = xx
            x2y = xy
        elif i21 and i22:
            cur = _face_nbr_of(faces, fnbr, cur, pV1)
            pV1 = pV
            x1x = xx
            x1y = xy
        else:
            dist = math.sqrt(xx * xx + xy * xy)
            nxs1 = math.sqrt(xs1x * xs1x + xs1y * xs1y)
            nxs2 = math.sqrt(xs2x * xs2x + xs2y * xs2y)
            dot1 = (xx * xs1x + xy * xs1y) / (dist * nxs1)
            dot2 = (xx * xs2x + xy * xs2y) / (dist * nxs2)
            return pV, dist, dot1, dot2
        num += 1
    return -1, 0.0, 0.0, 0.0


@njit(cache=True, error_model="numpy", inline="always")
def _speed_weight(pos, normal, T, i, q):
    """``GeodesicA::WeightCallback(Vert i, position q)``: 1 / (e . T e), e projected."""
    E = _sub(q, _p(pos, i))
    N = (normal[i, 0], normal[i, 1], normal[i, 2])
    p = _dot(N, E)
    E = _sub(E, _mul(N, p))
    E = _div(E, _norm(E))
    speed = E[0] * (T[i, 0] * E[0] + T[i, 1] * E[1] + T[i, 2] * E[2])
    speed += E[1] * (T[i, 3] * E[0] + T[i, 4] * E[1] + T[i, 5] * E[2])
    speed += E[2] * (T[i, 6] * E[0] + T[i, 7] * E[1] + T[i, 8] * E[2])
    return 1 / speed


@njit(cache=True, error_model="numpy")
def _vertex_distance(
    aniso, pos, faces, fnbr, normal, T, V1, lam1, lam2, dist, state, front, f, cur, a1, a2, cfront
):
    """``ComputeVertexDistance`` for ``cur`` in face ``f``; a1 is the nearer neighbour."""
    if state[a1] == FAR and state[a2] == FAR:
        return GW_INF
    Pc = _p(pos, cur)
    P1 = _p(pos, a1)
    P2 = _p(pos, a2)
    if aniso:
        F1 = _speed_weight(pos, normal, T, cur, P1)
        F2 = _speed_weight(pos, normal, T, cur, P2)
    else:
        F1 = 1.0
        F2 = 1.0
    Edge1 = _sub(P1, Pc)
    b = _norm(Edge1)
    Edge1 = _div(Edge1, b)
    Edge2 = _sub(P2, Pc)
    a = _norm(Edge2)
    Edge2 = _div(Edge2, a)
    d1 = dist[a1]
    d2 = dist[a2]
    u1 = state[a1] != FAR and front[a1] == cfront
    u2 = state[a2] != FAR and front[a2] == cfront
    if (not u1) and u2:
        return d2 + a * F2
    if u1 and (not u2):
        return d1 + b * F1
    if u1 and u2:
        dot = _dot(Edge1, Edge2)
        # V1 is a 1-row dummy in isotropic runs: only read it for anisotropic updates
        vv = (V1[cur, 0], V1[cur, 1], V1[cur, 2]) if aniso else (0.0, 0.0, 0.0)
        if dot < 0:
            pv, c, dot1, dot2 = _unfold(pos, faces, fnbr, f, cur, a1, a2)
            if pv >= 0 and state[pv] != FAR:
                d3 = dist[pv]
                if aniso:
                    F3 = _speed_weight(pos, normal, T, cur, _p(pos, pv))
                    t = _update_fsm(
                        Pc, P1, _p(pos, pv), d1, d3, dot1, F1, F3, vv, lam1[cur], lam2[cur]
                    )
                    t2 = _update_fsm(
                        Pc, _p(pos, pv), P2, d3, d2, dot2, F3, F2, vv, lam1[cur], lam2[cur]
                    )
                else:
                    t = _update_sethian(d1, d3, c, b, dot1, 1.0)
                    t2 = _update_sethian(d3, d2, a, c, dot2, 1.0)
                return t if t < t2 else t2
        if aniso:
            return _update_fsm(Pc, P1, P2, d1, d2, dot, F1, F2, vv, lam1[cur], lam2[cur])
        return _update_sethian(d1, d2, a, b, dot, 1.0)
    return GW_INF


@njit(cache=True, error_model="numpy")
def _march(
    aniso,
    kind,
    pos,
    faces,
    fnbr,
    vf_ptr,
    vf_idx,
    ring_ptr,
    ring_idx,
    normal,
    T,
    V1,
    lam1,
    lam2,
    varea,
    starts,
    dmax,
    niter_max_req,
    area_max,
    dist,
    state,
    front,
    cum_area,
    heap,
    touched,
):
    """One ``perform_front_propagation``.  Workspace arrays are reset here (only touched entries).

    Returns (number of touched vertices, nbr_iter).  ``state/dist/front`` of untouched vertices
    keep their reset values (FAR, 1e9, -1).
    """
    n = pos.shape[0]
    niter_max = int(n * 1.2) if (niter_max_req == 0 or n * 1.2 < niter_max_req) else niter_max_req
    nt = 0
    hs = 0
    for s in starts:
        if state[s] == FAR:
            touched[nt] = s
            nt += 1
        front[s] = s
        dist[s] = 0.0
        state[s] = ALIVE
        heap[hs] = s
        hs += 1
    make_heap(heap, dist, hs, kind)
    make_heap(heap, dist, hs, kind)
    nbr_iter = 0
    area_prop = 0.0
    while hs > 0:
        cur = heap[0]
        pop_heap(heap, dist, hs, kind)
        hs -= 1
        state[cur] = DEAD
        cfront = front[cur]
        for r in range(ring_ptr[cur], ring_ptr[cur + 1]):
            nw = ring_idx[r]
            if state[nw] == DEAD:
                continue
            rnew = GW_INF
            for k in range(vf_ptr[nw], vf_ptr[nw + 1]):
                f = vf_idx[k]
                s0 = 0
                for i in range(3):
                    if faces[f, i] == nw:
                        s0 = i
                a1 = faces[f, (s0 + 1) % 3]
                a2 = faces[f, (s0 + 2) % 3]
                if dist[a1] > dist[a2]:
                    t_ = a1
                    a1 = a2
                    a2 = t_
                val = _vertex_distance(
                    aniso,
                    pos,
                    faces,
                    fnbr,
                    normal,
                    T,
                    V1,
                    lam1,
                    lam2,
                    dist,
                    state,
                    front,
                    f,
                    nw,
                    a1,
                    a2,
                    cfront,
                )
                if val < rnew:
                    rnew = val
            if state[nw] == FAR:
                doins = nbr_iter <= niter_max
                nbr_iter += 1
                if doins:
                    dist[nw] = rnew
                    touched[nt] = nw
                    nt += 1
                    heap[hs] = nw
                    hs += 1
                    push_heap(heap, dist, hs, kind)
                    state[nw] = ALIVE
                    front[nw] = cfront
            else:  # ALIVE
                if rnew <= dist[nw]:
                    dist[nw] = rnew
                    front[nw] = cfront
                    make_heap(heap, dist, hs, kind)
        if hs > 0:
            if dist[cur] > dmax:
                break
            if aniso:
                if nbr_iter >= niter_max:
                    break
                if area_max > 0:
                    area_prop += varea[cur]
                    cum_area[cur] = area_prop
                    if area_prop > area_max:
                        break
    return nt, nbr_iter


@njit(cache=True, error_model="numpy")
def _reset(touched, nt, dist, state, front):
    for k in range(nt):
        v = touched[k]
        dist[v] = GW_INF
        state[v] = FAR
        front[v] = -1


def _dummy():
    return np.zeros((1, 9)), np.zeros((1, 3)), np.zeros(1), np.zeros(1), np.zeros(1)


def propagate_iso(mesh: GWMesh, starts, *, dmax: float = 1e9, heap: int = HEAP_LIBCXX):
    """``Geodesic::perform_front_propagation(starts, n, NULL, 0)``: returns (dist, state, source).

    ``source`` is the start vertex whose front reached each vertex (-1 if unreached).
    """
    n = mesh.pos.shape[0]
    dist = np.full(n, GW_INF)
    state = np.zeros(n, dtype=np.int8)
    front = -np.ones(n, dtype=np.int32)
    T, V1, lam1, lam2, varea = _dummy()
    s = np.ascontiguousarray(starts, dtype=np.int32)
    touched = np.empty(n + s.size + 1, dtype=np.int32)
    _march(
        False, heap, mesh.pos, mesh.faces, mesh.fnbr, mesh.vf_ptr, mesh.vf_idx, mesh.ring_ptr,
        mesh.ring_idx, mesh.normal, T, V1, lam1, lam2, varea, s, dmax, 0, 0.0,
        dist, state, front, np.zeros(n), np.empty(n + s.size + 1, dtype=np.int32), touched,
    )  # fmt: skip
    return dist, state, front


def set_tensor(mesh: GWMesh, u, lam1, lam2):
    """``GeodesicA::setTensor(u, lam1, lam2)``: returns (T (N,9), V1 (N,3), lam1, lam2)."""
    n = mesh.pos.shape[0]
    T = np.zeros((n, 9))
    V1 = np.zeros((n, 3))
    u = np.ascontiguousarray(u, dtype=np.float64)
    lam1 = np.ascontiguousarray(lam1, dtype=np.float64)
    lam2 = np.ascontiguousarray(lam2, dtype=np.float64)
    _set_tensor(mesh.normal, u, T, V1, lam1, lam2)
    return (
        T,
        V1,
        np.ascontiguousarray(lam1, dtype=np.float64),
        np.ascontiguousarray(lam2, dtype=np.float64),
    )


@njit(cache=True, error_model="numpy")
def _set_tensor(normal, u, T, V1, lam1, lam2):
    for i in range(u.shape[0]):
        N = (normal[i, 0], normal[i, 1], normal[i, 2])
        v = (u[i, 0], u[i, 1], u[i, 2])
        p = _dot(N, v)
        v = _sub(v, _mul(N, p))
        v = _div(v, _norm(v))
        v2 = _cross(N, v)
        v2 = _div(v2, _norm(v2))
        V1[i, 0] = v[0]
        V1[i, 1] = v[1]
        V1[i, 2] = v[2]
        for j in range(3):
            for k in range(j, 3):
                val = lam1[i] * V1[i, j] * V1[i, k] + lam2[i] * v2[j] * v2[k]
                T[i, j * 3 + k] = val
                T[i, k * 3 + j] = val


@njit(cache=True, error_model="numpy", nogil=True)
def kernel_range(
    kind, pos, faces, fnbr, vf_ptr, vf_idx, ring_ptr, ring_idx, normal, T, V1, lam1, lam2, varea,
    area1, area2, lookup, use_lookup, max_area, deltas, v_begin, v_end, out_k1, out_k2,
):  # fmt: skip
    """Area-limited anisotropic propagation from each vertex in [v_begin, v_end).

    ``out_k1/out_k2[v, t]`` = sum of surface / outer areas over dead vertices whose cumulative
    propagated area is below ``deltas[t]`` (``Gyrification::kernelArea``, index-ordered sums).
    """
    n = pos.shape[0]
    nd = deltas.shape[0]
    dist = np.full(n, GW_INF)
    state = np.zeros(n, dtype=np.int8)
    front = -np.ones(n, dtype=np.int32)
    cum = np.zeros(n)
    heap = np.empty(n + 1, dtype=np.int32)
    touched = np.empty(n + 2, dtype=np.int32)
    seen = np.zeros(area2.shape[0], dtype=np.int8) if use_lookup else np.zeros(1, dtype=np.int8)
    starts = np.empty(1, dtype=np.int32)
    for v in range(v_begin, v_end):
        starts[0] = v
        nt, _ = _march(
            True, kind, pos, faces, fnbr, vf_ptr, vf_idx, ring_ptr, ring_idx, normal, T, V1, lam1,
            lam2, varea, starts, 1e9, 0, max_area, dist, state, front, cum, heap, touched,
        )  # fmt: skip
        # dead vertices in index order (kernelArea loops i = 0..n-1)
        dead = np.empty(nt, dtype=np.int32)
        nd_ = 0
        for k in range(nt):
            if state[touched[k]] == DEAD:
                dead[nd_] = touched[k]
                nd_ += 1
        dead = np.sort(dead[:nd_])
        for t in range(nd):
            size = deltas[t]
            s1 = 0.0
            s2 = 0.0
            if use_lookup:
                seen[:] = 0
            for k in range(nd_):
                i = dead[k]
                if cum[i] < size:
                    s1 += area1[i]
                    if use_lookup:
                        j = lookup[i]
                        if seen[j] == 0:
                            s2 += area2[j]
                            seen[j] = 1
                    else:
                        s2 += area2[i]
            out_k1[v, t] = s1
            out_k2[v, t] = s2
        _reset(touched, nt, dist, state, front)
    return 0
