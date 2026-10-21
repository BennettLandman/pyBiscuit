"""Geodesic distances and paths: MeshLib ``Geodesic`` / ``GeodesicPath`` port (plan row B2.6).

Two tools live here:

* :func:`dijkstra_distances` - plain Dijkstra on the edge graph (SciPy).  A generic upper bound
  for geodesic distances (overestimates by up to ~8 % on a triangulated sphere); *not* what
  CurveExtraction uses.
* :class:`GeodesicSolver` / :class:`GeodesicPath` - exact ports of what the CurveExtraction call
  graph reaches: ``Geodesic/Geodesic.cpp`` (the wrapper), Peyre's fast-marching on triangulated
  surfaces (``gw/gw_geodesic/GW_GeodesicMesh.{cpp,inl}``, ``GW_GeodesicVertex.inl``,
  ``gw/gw_core/GW_{Mesh,Face,Vertex,FaceIterator,VertexIterator}.cpp/.inl``) and
  ``Geodesic/GeodesicPath.cpp``.  ``GeodesicA`` / ``GW_GeodesicMeshA`` / the Voronoi and
  parameterisation classes are *not* reached from CurveExtraction and are not ported.
  See ``docs/ports/geodesic.md``.

Precision (mirrors MeshLib): vertex positions are stored as float32 in ``Mesh`` and widened to
double in the GW mesh (``GW_Float = double``); the whole fast-marching update runs in double.
``Geodesic::dist()`` is double.  ``GeodesicPath`` mixes: distances double, edge vectors and the
2x2 solve inputs float32 (``Vector``), the Gram determinant is evaluated in float32 and only then
widened.  Quirks reproduced on purpose: ``GW_VectorStatic::operator/=`` multiplies by the
reciprocal; ``GW_Face::GetEdgeNumber(v, w)`` returns 0 when the edge is absent; the heap is a
binary heap of vertex ids ordered by distance whose *tie behaviour* depends on the C++ standard
library (``heap=\"libstdcxx\"``, the Linux image, or ``\"libcxx\"``, the macOS oracle).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse import csgraph

__all__ = [
    "GeodesicPath",
    "GeodesicSolver",
    "dijkstra_distances",
    "sparse_pairwise_distances",
]

FAR, ALIVE, DEAD = 0, 1, 2
GW_INFINITE = 1e9
GW_EPSILON = 1e-9
_HEAP_MODES = {"libstdcxx": 0, "libcxx": 1}


# --------------------------------------------------------------------------- #
# generic Dijkstra on the edge graph
# --------------------------------------------------------------------------- #
def dijkstra_distances(vertices, faces=None, sources=0, limit: float = np.inf) -> np.ndarray:
    """Shortest-path distances along mesh edges (Euclidean edge lengths, float64).

    ``sources`` is an int or sequence; returns ``(len(sources), N)`` (or ``(N,)`` for an int).
    ``limit`` truncates the search (unreached vertices are ``inf``).
    """
    if faces is None:
        vertices, faces = vertices.vertices, vertices.faces
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    n = v.shape[0]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    g = sparse.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), (n, n))
    single = np.isscalar(sources)
    d = csgraph.dijkstra(g.tocsr(), directed=False, indices=sources, limit=limit)
    return d if not single else d.reshape(-1)


# --------------------------------------------------------------------------- #
# GW_Mesh connectivity (BuildConnectivity + vertex/face iterators)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _edge_number(vert, f, a, b):
    """``GW_Face::GetEdgeNumber(v1, v2)``: index of the vertex opposite the edge; 0 if absent."""
    for i in range(3):
        if vert[f, i] == a:
            if vert[f, (i + 1) % 3] == b:
                return (i + 2) % 3
            if vert[f, (i + 2) % 3] == b:
                return (i + 1) % 3
    return 0


@njit(cache=True)
def _vertex_between(vert, f, a, b):
    """``GW_Face::GetVertex(v1, v2)``: the third vertex of face f."""
    return vert[f, _edge_number(vert, f, a, b)]


@njit(cache=True)
def _next_vertex(vert, f, v):
    """``GW_Face::GetNextVertex``: the vertex after v in the face's stored order (-1 if absent)."""
    for i in range(3):
        if vert[f, i] == v:
            return vert[f, (i + 1) % 3]
    return -1


@njit(cache=True)
def _nbr_across(vert, fnbr, f, v):
    """``GW_Face::GetFaceNeighbor(Vertex)``: neighbour face opposite vertex v (-1 = NULL)."""
    for i in range(3):
        if vert[f, i] == v:
            return fnbr[f, i]
    return -1


@njit(cache=True)
def _vertex_faces_csr(vert, n):
    cnt = np.zeros(n + 1, dtype=np.int64)
    for f in range(vert.shape[0]):
        for i in range(3):
            cnt[vert[f, i] + 1] += 1
    for i in range(n):
        cnt[i + 1] += cnt[i]
    pos = cnt[:-1].copy()
    idx = np.empty(cnt[n], dtype=np.int32)
    for f in range(vert.shape[0]):  # ascending face order, repeats kept (GW pushes one per slot)
        for i in range(3):
            v = vert[f, i]
            idx[pos[v]] = f
            pos[v] += 1
    return cnt, idx


@njit(cache=True)
def _build_connectivity(vert, n):
    """GW_Mesh::BuildConnectivity plus ``GW_Face::SetVertex`` (vertex -> first incident face)."""
    m = vert.shape[0]
    vf_ptr, vf_idx = _vertex_faces_csr(vert, n)
    vface = np.full(n, -1, dtype=np.int32)
    for f in range(m):
        for i in range(3):
            if vface[vert[f, i]] < 0:
                vface[vert[f, i]] = f
    fnbr = np.full((m, 3), -1, dtype=np.int32)
    for f in range(m):
        for i in range(3):
            i1 = (i + 1) % 3
            i2 = (i + 2) % 3
            a = vert[f, i1]
            b = vert[f, i2]
            nb = -1
            found = False
            for p in range(vf_ptr[a], vf_ptr[a + 1]):
                f1 = vf_idx[p]
                for q in range(vf_ptr[b], vf_ptr[b + 1]):
                    f2 = vf_idx[q]
                    if f1 == f2 and f1 != f:
                        nb = f1
                        found = True
                        break
                if found:
                    break
            fnbr[f, i] = nb
            if nb >= 0:
                e = _edge_number(vert, nb, a, b)
                fnbr[nb, e] = f
    return vface, fnbr


@njit(cache=True)
def _rings(vert, fnbr, vface, n):
    """Emulate ``GW_VertexIterator`` / ``GW_FaceIterator`` (incl. the open-fan boundary walk)."""
    vptr = np.zeros(n + 1, dtype=np.int64)
    fptr = np.zeros(n + 1, dtype=np.int64)
    vbuf = np.empty(n * 8 + 16, dtype=np.int32)
    fbuf = np.empty(n * 8 + 16, dtype=np.int32)
    vn = 0
    fn = 0
    for v in range(n):
        f0 = vface[v]
        if f0 >= 0:
            # ---- vertex iterator: yields pDirection_
            pf = f0
            pd = _next_vertex(vert, f0, v)
            pp = -1
            inc = 0
            while True:
                if vn >= vbuf.shape[0]:
                    vbuf = np.concatenate((vbuf, np.empty(vbuf.shape[0], dtype=np.int32)))
                vbuf[vn] = pd
                vn += 1
                if inc > 100:
                    break
                if pf == -1:
                    while pp != -1:
                        pf = pp
                        pp = _nbr_across(vert, fnbr, pf, pd)
                        pd = _vertex_between(vert, pf, v, pd)
                    if pf == f0:
                        break
                    inc += 1
                elif pd != -1:
                    nxt = _nbr_across(vert, fnbr, pf, pd)
                    if nxt == f0:
                        break
                    nd = _vertex_between(vert, pf, v, pd)
                    pp = pf
                    pf = nxt
                    pd = nd
                    inc += 1
                else:
                    break
            # ---- face iterator: yields pFace_
            pf = f0
            pd = _next_vertex(vert, f0, v)
            inc = 0
            while True:
                if fn >= fbuf.shape[0]:
                    fbuf = np.concatenate((fbuf, np.empty(fbuf.shape[0], dtype=np.int32)))
                fbuf[fn] = pf
                fn += 1
                if inc > 100:
                    break
                if pf != -1 and pd != -1:
                    nxt = _nbr_across(vert, fnbr, pf, pd)
                    if nxt == f0:
                        break
                    if nxt == -1:
                        pprev = pf
                        pd = _vertex_between(vert, pf, pd, v)
                        k = 0
                        stop = False
                        while True:
                            pf = pprev
                            pprev = _nbr_across(vert, fnbr, pprev, pd)
                            pd = _vertex_between(vert, pf, v, pd)
                            k += 1
                            if k >= 20:
                                stop = True
                                break
                            if pprev == -1:
                                break
                        if stop or pf == f0:
                            break
                        inc += 1
                    else:
                        nd = _vertex_between(vert, pf, v, pd)
                        pf = nxt
                        pd = nd
                        inc += 1
                else:
                    break
        vptr[v + 1] = vn
        fptr[v + 1] = fn
    return vptr, vbuf[:vn].copy(), fptr, fbuf[:fn].copy()


# --------------------------------------------------------------------------- #
# binary heap (std::make_heap / push_heap / pop_heap on vertex ids, comp = dist[a] > dist[b])
# --------------------------------------------------------------------------- #
@njit(cache=True, inline="always")
def _comp(dist, a, b):
    return dist[a] > dist[b]


@njit(cache=True)
def _cxx_sift_down(heap, dist, n, start):
    child = start
    if n < 2 or (n - 2) // 2 < child:
        return
    child = 2 * child + 1
    ci = child
    if child + 1 < n and _comp(dist, heap[ci], heap[ci + 1]):
        ci += 1
        child += 1
    if _comp(dist, heap[ci], heap[start]):
        return
    top = heap[start]
    while True:
        heap[start] = heap[ci]
        start = ci
        if (n - 2) // 2 < child:
            break
        child = 2 * child + 1
        ci = child
        if child + 1 < n and _comp(dist, heap[ci], heap[ci + 1]):
            ci += 1
            child += 1
        if _comp(dist, heap[ci], top):
            break
    heap[start] = top


@njit(cache=True)
def _cxx_sift_up(heap, dist, end, length):
    if length > 1:
        length = (length - 2) // 2
        ptr = length
        last = end - 1
        if _comp(dist, heap[ptr], heap[last]):
            t = heap[last]
            while True:
                heap[last] = heap[ptr]
                last = ptr
                if length == 0:
                    break
                length = (length - 1) // 2
                ptr = length
                if not _comp(dist, heap[ptr], t):
                    break
            heap[last] = t


@njit(cache=True)
def _cxx_floyd_sift_down(heap, dist, n):
    hole = 0
    ci = 0
    child = 0
    while True:
        ci += child + 1
        child = 2 * child + 1
        if child + 1 < n and _comp(dist, heap[ci], heap[ci + 1]):
            ci += 1
            child += 1
        heap[hole] = heap[ci]
        hole = ci
        if child > (n - 2) // 2:
            return hole


@njit(cache=True)
def _gcc_push(heap, dist, hole, top, value):
    parent = (hole - 1) // 2
    while hole > top and _comp(dist, heap[parent], value):
        heap[hole] = heap[parent]
        hole = parent
        parent = (hole - 1) // 2
    heap[hole] = value


@njit(cache=True)
def _gcc_adjust(heap, dist, hole, length, value):
    top = hole
    second = hole
    while second < (length - 1) // 2:
        second = 2 * (second + 1)
        if _comp(dist, heap[second], heap[second - 1]):
            second -= 1
        heap[hole] = heap[second]
        hole = second
    if (length & 1) == 0 and second == (length - 2) // 2:
        second = 2 * (second + 1)
        heap[hole] = heap[second - 1]
        hole = second - 1
    _gcc_push(heap, dist, hole, top, value)


@njit(cache=True)
def _make_heap(heap, dist, n, mode):
    if n < 2:
        return
    if mode == 1:  # libc++
        for start in range((n - 2) // 2, -1, -1):
            _cxx_sift_down(heap, dist, n, start)
    else:  # libstdc++
        parent = (n - 2) // 2
        while True:
            _gcc_adjust(heap, dist, parent, n, heap[parent])
            if parent == 0:
                return
            parent -= 1


@njit(cache=True)
def _push_heap(heap, dist, n, mode):
    """``n`` = size after the new element was appended at heap[n-1]."""
    if mode == 1:
        _cxx_sift_up(heap, dist, n, n)
    else:
        _gcc_push(heap, dist, n - 1, 0, heap[n - 1])


@njit(cache=True)
def _pop_heap(heap, dist, n, mode):
    """Move the top to heap[n-1] and re-heapify heap[:n-1]; caller then drops the last slot."""
    if mode == 1:
        if n > 1:
            top = heap[0]
            hole = _cxx_floyd_sift_down(heap, dist, n)
            last = n - 1
            if hole == last:
                heap[hole] = top
            else:
                heap[hole] = heap[last]
                hole += 1
                heap[last] = top
                _cxx_sift_up(heap, dist, hole, hole)
    else:
        if n > 1:
            value = heap[n - 1]
            heap[n - 1] = heap[0]
            _gcc_adjust(heap, dist, 0, n - 1, value)


# --------------------------------------------------------------------------- #
# fast-marching update (GW_GeodesicMesh::ComputeVertexDistance & friends), all double
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _unfold_triangle(pos, vert, fnbr, face, v, v1, v2):
    """``GW_GeodesicMesh::UnfoldTriangle`` -> (vertex id or -1, dist, dot1, dot2)."""
    e1x = pos[v1, 0] - pos[v, 0]
    e1y = pos[v1, 1] - pos[v, 1]
    e1z = pos[v1, 2] - pos[v, 2]
    rn1 = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    if rn1 != 0.0:
        r = 1.0 / rn1
        e1x *= r
        e1y *= r
        e1z *= r
    e2x = pos[v2, 0] - pos[v, 0]
    e2y = pos[v2, 1] - pos[v, 1]
    e2z = pos[v2, 2] - pos[v, 2]
    rn2 = math.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
    if rn2 != 0.0:
        r = 1.0 / rn2
        e2x *= r
        e2y *= r
        e2z *= r
    dot = e1x * e2x + e1y * e2y + e1z * e2z
    eq1x = dot
    eq1y = math.sqrt(1.0 - dot * dot)
    eq2x = 1.0
    eq2y = 0.0
    x1x = rn1
    x1y = 0.0
    x2x = eq1x * rn2
    x2y = eq1y * rn2
    xs1x, xs1y = x1x, x1y
    xs2x, xs2y = x2x, x2y
    p1 = v1
    p2 = v2
    cur = _nbr_across(vert, fnbr, face, v)
    num = 0
    while num < 50 and cur != -1:
        pv = _vertex_between(vert, cur, p1, p2)
        e1x = pos[p2, 0] - pos[p1, 0]
        e1y = pos[p2, 1] - pos[p1, 1]
        e1z = pos[p2, 2] - pos[p1, 2]
        rn1 = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        if rn1 != 0.0:
            r = 1.0 / rn1
            e1x *= r
            e1y *= r
            e1z *= r
        e2x = pos[pv, 0] - pos[p1, 0]
        e2y = pos[pv, 1] - pos[p1, 1]
        e2z = pos[pv, 2] - pos[p1, 2]
        rn2 = math.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
        if rn2 != 0.0:
            r = 1.0 / rn2
            e2x *= r
            e2y *= r
            e2z *= r
        # vv = (x2 - x1) * rNorm2 / rNorm1   (operator/ multiplies by the reciprocal)
        wx = (x2x - x1x) * rn2
        wy = (x2y - x1y) * rn2
        if rn1 != 0.0:
            r = 1.0 / rn1
            wx *= r
            wy *= r
        dot = e1x * e2x + e1y * e2y + e1z * e2z
        a = -math.acos(dot)
        # GW_VectorStatic::Rotate(a, 0, 1)
        xx = math.cos(a) * wx - math.sin(a) * wy + x1x
        xy = math.sin(a) * wx + math.cos(a) * wy + x1y
        dx1 = xx - x1x
        dy1 = xy - x1y
        dx2 = xx - x2x
        dy2 = xy - x2y
        l11 = -(x1x * eq1x + x1y * eq1y) / (dx1 * eq1x + dy1 * eq1y)
        l12 = -(x1x * eq2x + x1y * eq2y) / (dx1 * eq2x + dy1 * eq2y)
        l21 = -(x2x * eq1x + x2y * eq1y) / (dx2 * eq1x + dy2 * eq1y)
        l22 = -(x2x * eq2x + x2y * eq2y) / (dx2 * eq2x + dy2 * eq2y)
        i11 = l11 >= 0 and l11 <= 1
        i12 = l12 >= 0 and l12 <= 1
        i21 = l21 >= 0 and l21 <= 1
        i22 = l22 >= 0 and l22 <= 1
        if i11 and i12:
            cur = _nbr_across(vert, fnbr, cur, p2)
            p2 = pv
            x2x, x2y = xx, xy
        elif i21 and i22:
            cur = _nbr_across(vert, fnbr, cur, p1)
            p1 = pv
            x1x, x1y = xx, xy
        else:
            dist = math.sqrt(xx * xx + xy * xy)
            n1 = math.sqrt(xs1x * xs1x + xs1y * xs1y)
            n2 = math.sqrt(xs2x * xs2x + xs2y * xs2y)
            dot1 = (xx * xs1x + xy * xs1y) / (dist * n1)
            dot2 = (xx * xs2x + xy * xs2y) / (dist * n2)
            return pv, dist, dot1, dot2
        num += 1
    return -1, 0.0, 0.0, 0.0


@njit(cache=True)
def _sethian(d1, d2, a, b, dot, F):
    t = GW_INFINITE
    rcos = dot
    rsin = math.sqrt(1.0 - dot * dot)
    u = d2 - d1
    f2 = a * a + b * b - 2.0 * a * b * rcos
    f1 = b * u * (a * rcos - b)
    f0 = b * b * (u * u - F * F * a * a * rsin * rsin)
    delta = f1 * f1 - f0 * f2
    if delta >= 0:
        af2 = f2 if f2 > 0 else -f2
        if af2 > GW_EPSILON:
            t = (-f1 - math.sqrt(delta)) / f2
            if t < u or b * (t - u) / t < a * rcos or a / rcos < b * (t - u) / t:
                t = (-f1 + math.sqrt(delta)) / f2
        else:
            if f1 != 0:
                t = -f0 / f1
            else:
                t = -GW_INFINITE
    else:
        t = -GW_INFINITE
    if u < t and a * rcos < b * (t - u) / t and b * (t - u) / t < a / rcos:
        return t + d1
    x = b * F + d1
    y = a * F + d2
    return x if x < y else y


@njit(cache=True)
def _vertex_distance(pos, vert, fnbr, dist, state, front, face, cur, v1, v2, cur_front):
    """``GW_GeodesicMesh::ComputeVertexDistance`` with unit weights (``F = 1``)."""
    F = 1.0
    if state[v1] != FAR or state[v2] != FAR:
        e1x = pos[v1, 0] - pos[cur, 0]
        e1y = pos[v1, 1] - pos[cur, 1]
        e1z = pos[v1, 2] - pos[cur, 2]
        b = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        if b != 0.0:
            r = 1.0 / b
            e1x *= r
            e1y *= r
            e1z *= r
        e2x = pos[v2, 0] - pos[cur, 0]
        e2y = pos[v2, 1] - pos[cur, 1]
        e2z = pos[v2, 2] - pos[cur, 2]
        a = math.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
        if a != 0.0:
            r = 1.0 / a
            e2x *= r
            e2y *= r
            e2z *= r
        d1 = dist[v1]
        d2 = dist[v2]
        u1 = state[v1] != FAR and front[v1] == cur_front
        u2 = state[v2] != FAR and front[v2] == cur_front
        if (not u1) and u2:
            return d2 + a * F
        if u1 and (not u2):
            return d1 + b * F
        if u1 and u2:
            dot = e1x * e2x + e1y * e2y + e1z * e2z
            if dot < 0:  # bUseUnfolding_ is true
                pv, c, dot1, dot2 = _unfold_triangle(pos, vert, fnbr, face, cur, v1, v2)
                if pv != -1 and state[pv] != FAR:
                    d3 = dist[pv]
                    t = _sethian(d1, d3, c, b, dot1, F)
                    t2 = _sethian(d3, d2, a, c, dot2, F)
                    if t2 < t:
                        t = t2
                    return t
            return _sethian(d1, d2, a, b, dot, F)
    return GW_INFINITE


@njit(cache=True)
def _reset(dist, state, front, touched, nt):
    for k in range(nt):
        v = touched[k]
        dist[v] = GW_INFINITE
        state[v] = FAR
        front[v] = -1


@njit(cache=True)
def _march(pos, vert, fnbr, vptr, vidx, fptr, fidx, starts, ends, dmax, niter_max, mode,
           dist, state, front, heap, touched):
    """One ``Geodesic::perform_front_propagation``; arrays are reset by the caller.

    Returns the number of touched vertices (the entries of ``touched``).
    """
    nt = 0
    hn = 0
    for s in starts:
        front[s] = s
        dist[s] = 0.0
        state[s] = ALIVE
        heap[hn] = s
        hn += 1
        touched[nt] = s
        nt += 1
    _make_heap(heap, dist, hn, mode)
    _make_heap(heap, dist, hn, mode)
    nbr_iter = 0
    while hn > 0:
        cur = heap[0]
        _pop_heap(heap, dist, hn, mode)
        hn -= 1
        state[cur] = DEAD
        cf = front[cur]
        for p in range(vptr[cur], vptr[cur + 1]):
            new = vidx[p]
            rnew = GW_INFINITE
            for q in range(fptr[new], fptr[new + 1]):
                f = fidx[q]
                v1 = _next_vertex(vert, f, new)
                v2 = _next_vertex(vert, f, v1)
                if dist[v1] > dist[v2]:
                    v1, v2 = v2, v1
                c = _vertex_distance(pos, vert, fnbr, dist, state, front, f, new, v1, v2, cf)
                if c < rnew:
                    rnew = c
            st = state[new]
            if st == FAR:
                ins = nbr_iter <= niter_max
                nbr_iter += 1
                if ins:
                    dist[new] = rnew
                    heap[hn] = new
                    hn += 1
                    _push_heap(heap, dist, hn, mode)
                    state[new] = ALIVE
                    front[new] = cf
                    touched[nt] = new
                    nt += 1
            elif st == ALIVE:
                if rnew <= dist[new]:
                    dist[new] = rnew
                    front[new] = cf
                    _make_heap(heap, dist, hn, mode)
        if hn == 0:
            break
        stop = dist[cur] > dmax
        if not stop:
            for e in ends:
                if e == cur:
                    stop = True
                    break
        if stop:
            break
    return nt


# --------------------------------------------------------------------------- #
# public solver
# --------------------------------------------------------------------------- #
class GeodesicSolver:
    """``Geodesic`` wrapper over the Peyre fast-marching mesh (unit weights, no heuristic).

    ``dist`` (float64), ``state`` (0 far / 1 alive / 2 dead) and ``source`` (nearest start vertex or
    -1) are persistent arrays holding the result of the last propagation; they are views, not
    copies (copy them if you keep them across calls).
    """

    def __init__(self, vertices, faces=None, *, heap: str = "libstdcxx"):
        if faces is None:
            vertices, faces = vertices.vertices, vertices.faces
        if heap not in _HEAP_MODES:
            raise ValueError(f"heap must be one of {sorted(_HEAP_MODES)}")
        self.mode = _HEAP_MODES[heap]
        # MeshLib stores float; the GW mesh widens each coordinate to double
        v32 = np.ascontiguousarray(vertices, dtype=np.float32)
        self.pos = v32.astype(np.float64)
        self.vert = np.ascontiguousarray(faces, dtype=np.int32)
        n = self.pos.shape[0]
        self.n_vertices = n
        self.n_faces = self.vert.shape[0]
        if self.vert.size and (self.vert.min() < 0 or self.vert.max() >= n):
            raise ValueError("face index out of range")
        self.vface, self.fnbr = _build_connectivity(self.vert, n)
        self.vptr, self.vidx, self.fptr, self.fidx = _rings(self.vert, self.fnbr, self.vface, n)
        self.dist = np.full(n, GW_INFINITE)
        self.state = np.zeros(n, dtype=np.int8)
        self.source = np.full(n, -1, dtype=np.int32)
        self._heap = np.empty(n + 1, dtype=np.int32)
        self._touched = np.empty(n + 1, dtype=np.int32)
        self._nt = 0

    # ----- C++ API names
    def perform_front_propagation(self, start_point: int, end_point: int = -1) -> None:
        """``perform_front_propagation(int start, int end = -1)`` (dmax = 1e9)."""
        self.propagate([start_point], [] if end_point == -1 else [end_point], 1e9)

    def perform_front_propagation_dmax(self, start_point: int, dmax: float) -> None:
        """``perform_front_propagation(int start, double dmax)``."""
        self.propagate([start_point], [], float(dmax))

    def propagate(self, starts, ends=(), dmax: float = 1e9, niter_max: int = 0) -> None:
        """General ``perform_front_propagation(starts, nstart, ends, nend, dmax, niter_max)``."""
        starts = np.ascontiguousarray(starts, dtype=np.int32).reshape(-1)
        ends = np.ascontiguousarray(ends, dtype=np.int32).reshape(-1)
        n = self.n_vertices
        lim = int(n * 1.2)
        niter = lim if (niter_max == 0 or n * 1.2 < niter_max) else int(niter_max)
        _reset(self.dist, self.state, self.source, self._touched, self._nt)
        self._nt = _march(
            self.pos, self.vert, self.fnbr, self.vptr, self.vidx, self.fptr, self.fidx,
            starts, ends, float(dmax), niter, self.mode,
            self.dist, self.state, self.source, self._heap, self._touched,
        )  # fmt: skip

    @property
    def touched(self) -> np.ndarray:
        """Vertices left in a non-far state by the last propagation (dist < 1e9 possible)."""
        return self._touched[: self._nt]


# --------------------------------------------------------------------------- #
# truncated pairwise distances of a vertex subset (TRACE::detectNearestPoints)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _pairwise(pos, vert, fnbr, vptr, vidx, fptr, fidx, mode, vids, point_of, thr32,
              dist, state, front, heap, touched):  # fmt: skip
    n = vids.shape[0]
    cap = 4 * n + 64
    rows = np.empty(cap, dtype=np.int32)
    cols = np.empty(cap, dtype=np.int32)
    vals = np.empty(cap, dtype=np.float32)
    k = 0
    nt = 0
    nverts = pos.shape[0]
    niter = int(nverts * 1.2)
    starts = np.empty(1, dtype=np.int32)
    ends = np.empty(0, dtype=np.int32)
    for i in range(n):
        _reset(dist, state, front, touched, nt)
        starts[0] = vids[i]
        nt = _march(pos, vert, fnbr, vptr, vidx, fptr, fidx, starts, ends, float(thr32), niter,
                    mode, dist, state, front, heap, touched)
        for t in range(nt):
            j = point_of[touched[t]]
            if j >= i:
                d = np.float32(dist[touched[t]])
                if d <= thr32:
                    if k >= cap:
                        cap *= 2
                        rows = np.concatenate((rows, np.empty(rows.shape[0], dtype=np.int32)))
                        cols = np.concatenate((cols, np.empty(cols.shape[0], dtype=np.int32)))
                        vals = np.concatenate((vals, np.empty(vals.shape[0], dtype=np.float32)))
                    rows[k] = i
                    cols[k] = j
                    vals[k] = d
                    k += 1
    _reset(dist, state, front, touched, nt)
    return rows[:k], cols[:k], vals[:k], nt


def sparse_pairwise_distances(
    solver: GeodesicSolver, vertex_ids, threshold: float
) -> sparse.csr_matrix:
    """Symmetric float32 CSR of fast-marching distances among ``vertex_ids`` up to ``threshold``.

    Reproduces ``SulcalCurve::detectNearestPoints``: for each point ``i`` a propagation with
    ``dmax = (double)threshold`` is run and ``d[i, j] = d[j, i] = (float)dist[v_j] if <= threshold``
    for every ``j >= i`` (entries beyond the threshold are absent, i.e. ``FLT_MAX`` upstream).
    The diagonal is 0 and is stored.
    """
    vids = np.ascontiguousarray(vertex_ids, dtype=np.int32)
    n = vids.shape[0]
    point_of = np.full(solver.n_vertices, -1, dtype=np.int32)
    point_of[vids] = np.arange(n, dtype=np.int32)
    s = solver
    _reset(s.dist, s.state, s.source, s._touched, s._nt)
    s._nt = 0
    rows, cols, vals, _ = _pairwise(
        s.pos, s.vert, s.fnbr, s.vptr, s.vidx, s.fptr, s.fidx, s.mode, vids, point_of,
        np.float32(threshold), s.dist, s.state, s.source, s._heap, s._touched,
    )  # fmt: skip
    off = rows != cols
    r = np.concatenate([rows, cols[off]])
    c = np.concatenate([cols, rows[off]])
    v = np.concatenate([vals, vals[off]])
    m = sparse.coo_matrix((v, (r, c)), shape=(n, n), dtype=np.float32).tocsr()
    m.sort_indices()
    return m


# --------------------------------------------------------------------------- #
# GeodesicPath
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _lower_bound(keys, key):
    lo = 0
    hi = keys.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _edge(tkeys, v, w, nv):
    """``GeodesicPath::get_edge``: lower_bound in the sorted edge table (clamped like valid memory)."""
    k = _lower_bound(tkeys, np.int64(v) * nv + w)
    return k if k < tkeys.shape[0] else tkeys.shape[0] - 1


@njit(cache=True)
def _vertex_face(faces, f, v, w):
    """``GeodesicPath::get_vertex_face``: third vertex of face f (the C++ falls through to [1])."""
    a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
    if (a == v and b == w) or (a == w and b == v):
        return c
    if (b == v and c == w) or (b == w and c == v):
        return a
    return b


@njit(cache=True)
def _path(verts32, faces, nb_ptr, nb_idx, tkeys, tface, dist, x, nv):
    cap = 256
    vl1 = np.empty(cap, dtype=np.int32)
    vl2 = np.empty(cap, dtype=np.int32)
    pl = np.empty(cap, dtype=np.float64)
    pts = np.empty((cap, 3), dtype=np.float32)
    bary = np.empty((cap, 3), dtype=np.float32)

    # vertex_stepping(x, w, f)
    w = nb_idx[nb_ptr[x]]
    mn = dist[w]
    for i in range(1, nb_ptr[x + 1] - nb_ptr[x]):
        xx = nb_idx[nb_ptr[x] + i]
        if dist[xx] < mn:
            mn = dist[xx]
            w = xx
    f1 = tface[_edge(tkeys, x, w, nv)]
    f2 = tface[_edge(tkeys, w, x, nv)]
    z1 = _vertex_face(faces, f1, x, w)
    z2 = _vertex_face(faces, f2, x, w)
    f = f1 if dist[z1] < dist[z2] else f2
    n = 0
    k = _edge(tkeys, x, w, nv)
    vl1[0] = tkeys[k] // nv
    vl2[0] = tkeys[k] % nv
    pl[0] = 1.0
    for c in range(3):
        pts[0, c] = verts32[x, c]
    bary[0, 0] = np.float32(x)
    bary[0, 1] = np.float32(x)
    bary[0, 2] = np.float32(1.0)
    dprev = dist[x]
    count = 1
    guard = 0
    while True:
        guard += 1
        if guard > 50 * faces.shape[0] + 1000:
            break  # pathological (non-manifold) input: the C++ would loop forever
        i = vl1[n]
        j = vl2[n]
        kk = _vertex_face(faces, f, i, j)
        a = dist[i]
        b = dist[j]
        c = dist[kk]
        # get_face_face(f, i, kk), get_face_face(f, j, kk)
        g1 = tface[_edge(tkeys, i, kk, nv)]
        g2 = tface[_edge(tkeys, kk, i, nv)]
        ff1 = g2 if f == g1 else g1
        g1 = tface[_edge(tkeys, j, kk, nv)]
        g2 = tface[_edge(tkeys, kk, j, nv)]
        ff2 = g2 if f == g1 else g1
        xp = pl[n]
        yp = 1.0 - xp
        ux = verts32[i, 0] - verts32[kk, 0]
        uy = verts32[i, 1] - verts32[kk, 1]
        uz = verts32[i, 2] - verts32[kk, 2]
        vx = verts32[j, 0] - verts32[kk, 0]
        vy = verts32[j, 1] - verts32[kk, 1]
        vz = verts32[j, 2] - verts32[kk, 2]
        uu = ux * ux + uy * uy + uz * uz  # float32
        vv = vx * vx + vy * vy + vz * vz
        uv = ux * vx + uy * vy + uz * vz
        det = np.float64((uu * vv) - (uv * uv))  # evaluated in float32, then widened
        a00 = np.float64(vv) / det
        a01 = np.float64(-uv) / det
        a10 = a01
        a11 = np.float64(uu) / det
        nx = a00 * (a - c) + a01 * (b - c)
        ny = a10 * (a - c) + a11 * (b - c)
        etas = -yp / ny
        etat = -xp / nx
        s = xp + etas * nx
        t = yp + etat * ny
        if etas < 0 and s >= 0 and s <= 1 and ff1 >= 0:
            pnext = s
            e = _edge(tkeys, i, kk, nv)
            f = ff1
            nv1 = tkeys[e] // nv
            nv2 = tkeys[e] % nv
        elif etat < 0 and t >= 0 and t <= 1 and ff2 >= 0:
            pnext = t
            e = _edge(tkeys, j, kk, nv)
            f = ff2
            nv1 = tkeys[e] // nv
            nv2 = tkeys[e] % nv
        else:
            z = i if a <= b else j
            w = nb_idx[nb_ptr[z]]
            mn = dist[w]
            for q in range(1, nb_ptr[z + 1] - nb_ptr[z]):
                xx = nb_idx[nb_ptr[z] + q]
                if dist[xx] < mn:
                    mn = dist[xx]
                    w = xx
            f1 = tface[_edge(tkeys, z, w, nv)]
            f2 = tface[_edge(tkeys, w, z, nv)]
            z1 = _vertex_face(faces, f1, z, w)
            z2 = _vertex_face(faces, f2, z, w)
            f = f1 if dist[z1] < dist[z2] else f2
            e = _edge(tkeys, z, w, nv)
            nv1 = tkeys[e] // nv
            nv2 = tkeys[e] % nv
            pnext = 1.0
        n += 1
        if count >= cap:
            cap *= 2
            vl1 = np.concatenate((vl1, np.empty(vl1.shape[0], dtype=np.int32)))
            vl2 = np.concatenate((vl2, np.empty(vl2.shape[0], dtype=np.int32)))
            pl = np.concatenate((pl, np.empty(pl.shape[0], dtype=np.float64)))
            pts = np.concatenate((pts, np.empty_like(pts)))
            bary = np.concatenate((bary, np.empty_like(bary)))
        vl1[n] = nv1
        vl2[n] = nv2
        pl[n] = pnext
        dnew = dist[nv1] * pnext + dist[nv2] * (1.0 - pnext)
        wf = np.float32(pnext)
        wg = np.float32(1.0 - pnext)
        for cc in range(3):
            pts[n, cc] = verts32[nv1, cc] * wf + verts32[nv2, cc] * wg
        bary[n, 0] = np.float32(nv1)
        bary[n, 1] = np.float32(nv2)
        bary[n, 2] = np.float32(pnext)
        count += 1
        if dnew == 0 or (dprev == dnew and n > 0):
            break
        dprev = dnew
    return pts[:count].copy(), bary[:count].copy()


class GeodesicPath:
    """``GeodesicPath``: steepest-descent path over the faces of a precomputed distance field.

    ``compute(x, dist)`` walks from vertex ``x`` down ``dist`` (a ``Geodesic::dist()`` array) and
    returns ``(points, barycentric)``, float32 ``(k, 3)`` arrays: ``points`` are xyz, each
    ``barycentric`` row is ``(vid1, vid2, w)`` meaning ``w * v[vid1] + (1 - w) * v[vid2]``
    (``(x, x, 1)`` for the start).  The path runs from ``x`` towards the distance minimum.
    """

    def __init__(self, vertices, faces=None):
        from .geometry import vertex_neighbor_lists

        if faces is None:
            vertices, faces = vertices.vertices, vertices.faces
        self.verts32 = np.ascontiguousarray(vertices, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        n = self.verts32.shape[0]
        self.n_vertices = n
        nb = vertex_neighbor_lists(self.faces, n)
        self._nb_ptr, self._nb_idx = nb.indptr, nb.indices
        # m_table: 3 directed edges per face sorted by (vid1, vid2); equal keys keep face order
        f = self.faces.astype(np.int64)
        e1 = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        e2 = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
        fid = np.tile(np.arange(f.shape[0], dtype=np.int64), 3)
        keys = e1 * n + e2
        order = np.argsort(keys, kind="stable")
        self._keys = keys[order]
        self._face = fid[order].astype(np.int32)

    def compute(self, x: int, dist) -> tuple[np.ndarray, np.ndarray]:
        dist = np.ascontiguousarray(dist, dtype=np.float64)
        return _path(
            self.verts32, self.faces, self._nb_ptr, self._nb_idx, self._keys, self._face, dist,
            int(x), np.int64(self.n_vertices),
        )  # fmt: skip
