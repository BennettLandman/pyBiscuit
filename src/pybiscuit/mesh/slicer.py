"""Plane / mesh contours: MeshLib ``AABB`` tree + ``Slicer`` port (needed by TRACE point detection).

Port of ``MeshLib/src/Util/Slicer.cpp`` (``slicing``, ``contour``, ``tracePList``, ``getSlice``,
``getEdgeID``, ``boxIntersection``, ``searchTree``), the tree half of ``Util/AABB.cpp``
(``initTree``, ``construction``, ``boundingBox``, ``updateTree``) and
``Util/Geom.cpp::Coordinate::{intersection, proj2plane}``.  See ``docs/ports/trace.md``.

Everything is float32 (MeshLib ``Vertex``/``Normal``/``MathVector``) with the C++ association order
(no fused multiply-add).  Behaviour reproduced on purpose:

* the tree is built exactly like ``AABB::construction`` (median of the 2n interval ends, faces
  straddling the threshold go to both sides, a leaf when one side would receive every face), so the
  candidate-face set of a plane query equals upstream's;
* segments are sorted by the id of their start edge and linked with ``lower_bound`` on that id.  A
  plane that leaves an *open* mesh produces a ``nextID`` with no match; upstream then links to the
  next larger id (or reads past the end - undefined); here the index is clamped to the last
  segment.  Closed surfaces never hit this;
* ``std::sort`` of segments with *equal* start-edge ids (non-manifold or degenerate input) is
  unspecified upstream; a stable sort is used here.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .geometry import face_normals

__all__ = ["Slicer", "build_tree", "slice_plane"]

F32 = np.float32


# --------------------------------------------------------------------------- #
# AABB tree (AABB::initTree / construction / updateTree)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _face_boxes(v, faces):
    m = faces.shape[0]
    out = np.empty((m, 6), dtype=np.float32)
    big = np.float32(3.4028234663852886e38)
    for i in range(m):
        mnx = big
        mxx = -big
        mny = big
        mxy = -big
        mnz = big
        mxz = -big
        for j in range(3):
            x = v[faces[i, j], 0]
            y = v[faces[i, j], 1]
            z = v[faces[i, j], 2]
            if mnx > x:
                mnx = x
            if mxx < x:
                mxx = x
            if mny > y:
                mny = y
            if mxy < y:
                mxy = y
            if mnz > z:
                mnz = z
            if mxz < z:
                mxz = z
        out[i, 0] = mnx
        out[i, 1] = mxx
        out[i, 2] = mny
        out[i, 3] = mxy
        out[i, 4] = mnz
        out[i, 5] = mxz
    return out


@njit(cache=True, error_model="numpy")
def _build_tree(boxes):
    m = boxes.shape[0]
    cap = 2 * m + 16
    left = np.full(cap, -1, dtype=np.int32)
    right = np.full(cap, -1, dtype=np.int32)
    seg_start = np.zeros(cap, dtype=np.int64)
    seg_len = np.zeros(cap, dtype=np.int64)
    pcap = 4 * m + 64
    pool = np.empty(pcap, dtype=np.int32)
    for i in range(m):
        pool[i] = i
    pool_n = m
    seg_len[0] = m
    nn = 1
    stack = np.empty(cap, dtype=np.int32)
    sp = 0
    stack[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        s = seg_start[node]
        n = seg_len[node]
        if n == 0:
            continue
        r = np.empty(6, dtype=np.float32)
        for k in range(6):
            r[k] = boxes[pool[s], k]
        for i in range(1, n):
            f = pool[s + i]
            for dim in range(3):
                if r[dim * 2 + 0] > boxes[f, dim * 2 + 0]:
                    r[dim * 2 + 0] = boxes[f, dim * 2 + 0]
                if r[dim * 2 + 1] < boxes[f, dim * 2 + 1]:
                    r[dim * 2 + 1] = boxes[f, dim * 2 + 1]
        dim = 0
        if r[3] - r[2] >= r[1] - r[0] and r[3] - r[2] >= r[5] - r[4]:
            dim = 1
        if r[5] - r[4] >= r[1] - r[0] and r[5] - r[4] >= r[3] - r[2]:
            dim = 2
        data = np.empty(2 * n, dtype=np.float32)
        for i in range(n):
            f = pool[s + i]
            data[2 * i] = boxes[f, dim * 2 + 0]
            data[2 * i + 1] = boxes[f, dim * 2 + 1]
        data.sort()
        thr = data[n]  # data[(int)(data.size() / 2)]
        nl = 0
        nr = 0
        for i in range(n):
            f = pool[s + i]
            mn = boxes[f, dim * 2 + 0]
            mx = boxes[f, dim * 2 + 1]
            if thr > mx:
                nl += 1
            elif thr <= mn:
                nr += 1
            else:
                nl += 1
                nr += 1
        if nl == n or nr == n:
            continue  # leaf: keeps its candidate segment
        # two children with the classified candidate lists (order preserved)
        while pool_n + nl + nr > pool.shape[0]:
            pool = np.concatenate((pool, np.empty(pool.shape[0], dtype=np.int32)))
        if nn + 2 > left.shape[0]:
            g = left.shape[0]
            left = np.concatenate((left, np.full(g, -1, dtype=np.int32)))
            right = np.concatenate((right, np.full(g, -1, dtype=np.int32)))
            seg_start = np.concatenate((seg_start, np.zeros(g, dtype=np.int64)))
            seg_len = np.concatenate((seg_len, np.zeros(g, dtype=np.int64)))
            stack = np.concatenate((stack, np.empty(g, dtype=np.int32)))
        lnode = nn
        rnode = nn + 1
        nn += 2
        left[node] = lnode
        right[node] = rnode
        seg_start[lnode] = pool_n
        seg_len[lnode] = nl
        k = pool_n
        for i in range(n):
            f = pool[s + i]
            if not (thr <= boxes[f, dim * 2 + 0]):  # left: thr > max, or straddling
                pool[k] = f
                k += 1
        pool_n = k
        seg_start[rnode] = pool_n
        seg_len[rnode] = nr
        k = pool_n
        for i in range(n):
            f = pool[s + i]
            if not (thr > boxes[f, dim * 2 + 1]):  # right: thr <= min, or straddling
                pool[k] = f
                k += 1
        pool_n = k
        stack[sp] = lnode
        stack[sp + 1] = rnode
        sp += 2
    # node boxes (children always have larger ids than their parent)
    tbox = np.empty((nn, 6), dtype=np.float32)  # x0 y0 z0 x1 y1 z1
    for node in range(nn - 1, -1, -1):
        if left[node] == -1:
            s = seg_start[node]
            f = pool[s]
            x0 = boxes[f, 0]
            y0 = boxes[f, 2]
            z0 = boxes[f, 4]
            x1 = boxes[f, 1]
            y1 = boxes[f, 3]
            z1 = boxes[f, 5]
            for i in range(1, seg_len[node]):
                f = pool[s + i]
                x0 = x0 if x0 < boxes[f, 0] else boxes[f, 0]
                y0 = y0 if y0 < boxes[f, 2] else boxes[f, 2]
                z0 = z0 if z0 < boxes[f, 4] else boxes[f, 4]
                x1 = x1 if x1 > boxes[f, 1] else boxes[f, 1]
                y1 = y1 if y1 > boxes[f, 3] else boxes[f, 3]
                z1 = z1 if z1 > boxes[f, 5] else boxes[f, 5]
            tbox[node, 0] = x0
            tbox[node, 1] = y0
            tbox[node, 2] = z0
            tbox[node, 3] = x1
            tbox[node, 4] = y1
            tbox[node, 5] = z1
        else:
            lb = left[node]
            rb = right[node]
            for k in range(3):
                tbox[node, k] = tbox[lb, k] if tbox[lb, k] < tbox[rb, k] else tbox[rb, k]
                tbox[node, 3 + k] = (
                    tbox[lb, 3 + k] if tbox[lb, 3 + k] > tbox[rb, 3 + k] else tbox[rb, 3 + k]
                )
    return left[:nn].copy(), right[:nn].copy(), tbox, seg_start[:nn].copy(), seg_len[:nn].copy(), \
        pool[:pool_n].copy()  # fmt: skip


def build_tree(vertices, faces):
    """Tree arrays ``(left, right, box, seg_start, seg_len, pool)`` of ``AABB::initTree``."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    return _build_tree(_face_boxes(v, f))


@njit(cache=True, error_model="numpy", inline="always")
def _box_hit(a, b, c, d, x0, x1, y0, y1, z0, z1):
    i1 = a * x0 + b * y0 + c * z0 + d
    i2 = a * x1 + b * y0 + c * z0 + d
    i3 = a * x0 + b * y1 + c * z0 + d
    i4 = a * x1 + b * y1 + c * z0 + d
    i5 = a * x0 + b * y0 + c * z1 + d
    i6 = a * x1 + b * y0 + c * z1 + d
    i7 = a * x0 + b * y1 + c * z1 + d
    i8 = a * x1 + b * y1 + c * z1 + d
    if (i1 > 0 and i2 > 0 and i3 > 0 and i4 > 0 and i5 > 0 and i6 > 0 and i7 > 0 and i8 > 0) or (
        i1 < 0 and i2 < 0 and i3 < 0 and i4 < 0 and i5 < 0 and i6 < 0 and i7 < 0 and i8 < 0
    ):
        return False
    return True


@njit(cache=True, error_model="numpy")
def _plane_candidates(left, right, tbox, seg_start, seg_len, pool, a, b, c, d, nfaces):
    """Unique sorted face ids from the leaves reached by ``Slicer::searchTree``."""
    mark = np.zeros(nfaces, dtype=np.uint8)
    stack = np.empty(256, dtype=np.int32)
    stack[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if left[node] == -1:
            for i in range(seg_len[node]):
                mark[pool[seg_start[node] + i]] = 1
            continue
        for ch in (left[node], right[node]):
            if _box_hit(a, b, c, d, tbox[ch, 0], tbox[ch, 3], tbox[ch, 1], tbox[ch, 4],
                        tbox[ch, 2], tbox[ch, 5]):  # fmt: skip
                if sp >= stack.shape[0]:
                    stack = np.concatenate((stack, np.empty(stack.shape[0], dtype=np.int32)))
                stack[sp] = ch
                sp += 1
    return np.flatnonzero(mark).astype(np.int32)


# --------------------------------------------------------------------------- #
# Coordinate::intersection / proj2plane (float32)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _intersection(v, f, a, b, c, d, p0, p1):
    """``Coordinate::intersection`` for face ``f``; returns the vertex count and fills p0, p1."""
    eps = np.float32(1e-8)
    t0 = v[f[0]]
    t1 = v[f[1]]
    t2 = v[f[2]]
    it0 = a * t0[0] + b * t0[1] + c * t0[2]
    it1 = a * t1[0] + b * t1[1] + c * t1[2]
    it2 = a * t2[0] + b * t2[1] + c * t2[2]
    ait0 = abs(it0 + d)
    ait1 = abs(it1 + d)
    ait2 = abs(it2 + d)
    if ait0 <= eps and ait1 <= eps and ait2 <= eps:
        return 3
    elif ait0 <= eps and ait1 > eps and ait2 > eps and (it1 + d) * (it2 + d) > 0:
        for k in range(3):
            p0[k] = t0[k]
            p1[k] = t0[k]
        return 1
    elif ait1 <= eps and ait2 > eps and ait0 > eps and (it2 + d) * (it0 + d) > 0:
        for k in range(3):
            p0[k] = t1[k]
            p1[k] = t1[k]
        return 1
    elif ait2 <= eps and ait0 > eps and ait1 > eps and (it0 + d) * (it1 + d) > 0:
        for k in range(3):
            p0[k] = t2[k]
            p1[k] = t2[k]
        return 1
    elif ait0 <= eps and ait1 <= eps and ait2 > eps:
        for k in range(3):
            p0[k] = t0[k]
            p1[k] = t1[k]
        return 2
    elif ait1 <= eps and ait2 <= eps and ait0 > eps:
        for k in range(3):
            p0[k] = t1[k]
            p1[k] = t2[k]
        return 2
    elif ait2 <= eps and ait0 <= eps and ait1 > eps:
        for k in range(3):
            p0[k] = t2[k]
            p1[k] = t0[k]
        return 2
    else:
        if (it0 + d) * (it1 + d) > 0 and (it1 + d) * (it2 + d) > 0 and (it2 + d) * (it0 + d) > 0:
            return -1
        if (it1 + d) * (it2 + d) >= 0:
            v0, v1, v2 = t0, t1, t2
            k0, k1, k2 = it0, it1, it2
        elif (it2 + d) * (it0 + d) >= 0:
            v0, v1, v2 = t1, t2, t0
            k0, k1, k2 = it1, it2, it0
        else:
            v0, v1, v2 = t2, t0, t1
            k0, k1, k2 = it2, it0, it1
        one = np.float32(1.0)
        if abs(k1 + d) < eps:
            k1 = -d
        t = (d + k1) / (k1 - k0)
        for k in range(3):
            p0[k] = v0[k] * t + v1[k] * (one - t)
        if abs(k2 + d) < eps:
            k2 = -d
        t = (d + k2) / (k2 - k0)
        for k in range(3):
            p1[k] = v0[k] * t + v2[k] * (one - t)
        return 2


@njit(cache=True, error_model="numpy")
def _lower_bound64(keys, key):
    lo = 0
    hi = keys.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
# Slicer::slicing + contour
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _slice_segments(v, faces, fnorm, cand, a, b, c, d, nv_total):
    """Per-face cut segments in the order of ``Slicer::slicing`` (before the id sort)."""
    k = cand.shape[0]
    spos = np.empty((k, 3), dtype=np.float32)
    sid0 = np.empty((k, 2), dtype=np.int64)
    sid1 = np.empty((k, 2), dtype=np.int64)
    sface = np.empty(k, dtype=np.int32)
    ns = 0
    p0 = np.empty(3, dtype=np.float32)
    p1 = np.empty(3, dtype=np.float32)
    P = np.empty(3, dtype=np.float32)
    pp = np.empty(3, dtype=np.float32)
    one = np.float32(1.0)
    for ci in range(k):
        f = cand[ci]
        fv = faces[f]
        nv = _intersection(v, fv, a, b, c, d, p0, p1)
        if nv != 2:
            continue
        t0 = v[fv[0]]
        t1 = v[fv[1]]
        t2 = v[fv[2]]
        # consistent direction: P = proj(P0 + N)
        for q in range(3):
            P[q] = p0[q] + fnorm[f, q]
        portion = (a * P[0] + b * P[1] + c * P[2] + d) / (a * a + b * b + c * c)
        pp[0] = P[0] - a * portion
        pp[1] = P[1] - b * portion
        pp[2] = P[2] - c * portion
        it0 = a * t0[0] + b * t0[1] + c * t0[2] + d
        it1 = a * t1[0] + b * t1[1] + c * t1[2] + d
        it2 = a * t2[0] + b * t2[1] + c * t2[2] + d
        if (t0[0] == p0[0] and t0[1] == p0[1] and t0[2] == p0[2]) or (
            t0[0] == p1[0] and t0[1] == p1[1] and t0[2] == p1[2]
        ):
            it0 = np.float32(0.0)
        if (t1[0] == p0[0] and t1[1] == p0[1] and t1[2] == p0[2]) or (
            t1[0] == p1[0] and t1[1] == p1[1] and t1[2] == p1[2]
        ):
            it1 = np.float32(0.0)
        if (t2[0] == p0[0] and t2[1] == p0[1] and t2[2] == p0[2]) or (
            t2[0] == p1[0] and t2[1] == p1[1] and t2[2] == p1[2]
        ):
            it2 = np.float32(0.0)
        i00 = 0
        i01 = 0
        i10 = 0
        i11 = 0
        if abs(it1) == 0 and abs(it2) == 0:
            i00 = i01 = fv[1]
            i10 = i11 = fv[2]
        elif abs(it2) == 0 and abs(it0) == 0:
            i00 = i01 = fv[2]
            i10 = i11 = fv[0]
        elif abs(it0) == 0 and abs(it1) == 0:
            i00 = i01 = fv[0]
            i10 = i11 = fv[1]
        elif it1 * it2 >= 0:
            i00 = fv[1] if abs(it1) == 0 else fv[0]
            i10 = fv[2] if abs(it2) == 0 else fv[0]
            i01 = fv[1]
            i11 = fv[2]
        elif it2 * it0 >= 0:
            i00 = fv[2] if abs(it2) == 0 else fv[1]
            i10 = fv[0] if abs(it0) == 0 else fv[1]
            i01 = fv[2]
            i11 = fv[0]
        elif it0 * it1 >= 0:
            i00 = fv[0] if abs(it0) == 0 else fv[2]
            i10 = fv[1] if abs(it1) == 0 else fv[2]
            i01 = fv[0]
            i11 = fv[1]
        if i00 > i01:
            i00, i01 = i01, i00
        if i10 > i11:
            i10, i11 = i11, i10
        # PP0 = P0 - P, PP1 = P1 - P, PCP = PP0 x PP1; swap if PCP . (a, b, c) > 0
        ax = p0[0] - pp[0]
        ay = p0[1] - pp[1]
        az = p0[2] - pp[2]
        bx = p1[0] - pp[0]
        by = p1[1] - pp[1]
        bz = p1[2] - pp[2]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        if cx * a + cy * b + cz * c > 0:
            for q in range(3):
                tmp = p0[q]
                p0[q] = p1[q]
                p1[q] = tmp
            i00, i10 = i10, i00
            i01, i11 = i11, i01
        for q in range(3):
            spos[ns, q] = p0[q]
        sid0[ns, 0] = i00
        sid0[ns, 1] = i01
        sid1[ns, 0] = i10
        sid1[ns, 1] = i11
        sface[ns] = f
        ns += 1
    return spos[:ns], sid0[:ns], sid1[:ns], sface[:ns]


@njit(cache=True, error_model="numpy")
def _contour(sid0, sid1, nxt, prv, nv_total):
    """``Slicer::contour`` on id-sorted segments (arrays already ordered); returns the groups."""
    n = sid0.shape[0]
    keys = sid0[:, 0] * nv_total + sid0[:, 1]
    nkeys = sid1[:, 0] * nv_total + sid1[:, 1]
    trace = np.zeros(n, dtype=np.uint8)
    cap = n + 4
    gbeg = np.empty(cap, dtype=np.int64)
    gsize = np.empty(cap, dtype=np.int64)
    gclosed = np.empty(cap, dtype=np.uint8)
    ng = 0
    closed = False
    for i in range(n):
        if trace[i] == 0:
            # tracePList(i, i), iteratively
            cnt = 0
            cur = i
            while True:
                nx = _lower_bound64(keys, nkeys[cur])
                if nx >= n:
                    nx = n - 1
                trace[cur] = 1
                nxt[cur] = nx
                cnt += 1
                if trace[nx] == 0:
                    prv[nx] = cur
                    cur = nx
                else:
                    prv[i] = cur
                    break
            b_begin = i
            b_size = cnt
            b_closed = b_begin == nxt[prv[b_begin]]
            if not b_closed:
                inside = False
                part = nxt[prv[b_begin]]
                it = b_begin
                for j in range(b_size):
                    if it == part:
                        inside = True
                        break
                    it = nxt[it]
                if inside:
                    tmp = prv[b_begin]
                    prv[b_begin] = prv[part]
                    prv[part] = tmp
                    cc = 0
                    it = b_begin
                    while it != part and cc <= n + 2:
                        it = nxt[it]
                        cc += 1
                    b_size = cc
                    # new loop group
                    lb = part
                    cc = 1
                    it = lb
                    while it != prv[lb] and cc <= n + 2:
                        it = nxt[it]
                        cc += 1
                    gbeg[ng] = lb
                    gsize[ng] = cc
                    gclosed[ng] = 1
                    ng += 1
            closed = closed or (not b_closed)
            gbeg[ng] = b_begin
            gsize[ng] = b_size
            gclosed[ng] = 1 if b_closed else 0
            ng += 1
    # connected components
    gs = ng
    while closed:
        gi = 0
        while gi < ng:
            if gclosed[gi] == 0:
                for ci in range(ng):
                    if gclosed[ci] != 0:
                        continue
                    tail = nxt[prv[gbeg[gi]]]
                    head = gbeg[ci]
                    if tail == head:
                        tmp = prv[gbeg[gi]]
                        prv[gbeg[gi]] = prv[gbeg[ci]]
                        prv[gbeg[ci]] = tmp
                        gsize[gi] += gsize[ci]
                        for q in range(ci, ng - 1):
                            gbeg[q] = gbeg[q + 1]
                            gsize[q] = gsize[q + 1]
                            gclosed[q] = gclosed[q + 1]
                        ng -= 1
                        break
            gi += 1
        closed = gs != ng
        gs = ng
    return gbeg[:ng].copy(), gsize[:ng].copy()


@njit(cache=True, error_model="numpy")
def slice_plane(v, faces, fnorm, left, right, tbox, seg_start, seg_len, pool, a, b, c, d):
    """``Slicer::slicing(a, b, c, d)`` followed by ``getSlice`` / ``getEdgeID`` for every group.

    Returns ``(gptr, pos, eid)``: group ``g`` owns rows ``gptr[g]:gptr[g+1]`` of ``pos`` (float32
    xyz) and ``eid`` (float32 ``(id0, id1, 0)`` start-edge ids), in contour order.
    """
    cand = _plane_candidates(left, right, tbox, seg_start, seg_len, pool, a, b, c, d,
                             faces.shape[0])  # fmt: skip
    nvt = np.int64(v.shape[0])
    spos, sid0, sid1, sface = _slice_segments(v, faces, fnorm, cand, a, b, c, d, nvt)
    n = spos.shape[0]
    keys = sid0[:, 0] * nvt + sid0[:, 1]
    order = np.argsort(keys, kind="mergesort")
    spos = spos[order]
    sid0 = sid0[order]
    sid1 = sid1[order]
    nxt = np.full(n, -1, dtype=np.int64)
    prv = np.full(n, -1, dtype=np.int64)
    gbeg, gsize = _contour(sid0, sid1, nxt, prv, nvt)
    ng = gbeg.shape[0]
    gptr = np.zeros(ng + 1, dtype=np.int64)
    for g in range(ng):
        gptr[g + 1] = gptr[g] + gsize[g]
    pos = np.empty((gptr[ng], 3), dtype=np.float32)
    eid = np.zeros((gptr[ng], 3), dtype=np.float32)
    for g in range(ng):
        it = gbeg[g]
        for i in range(gsize[g]):
            r = gptr[g] + i
            for q in range(3):
                pos[r, q] = spos[it, q]
            eid[r, 0] = np.float32(sid0[it, 0])
            eid[r, 1] = np.float32(sid0[it, 1])
            it = nxt[it]
    return gptr, pos, eid


class Slicer:
    """``Slicer`` bound to a mesh: ``slicing(a, b, c, d)`` returns the plane's contour groups."""

    def __init__(self, vertices, faces=None):
        if faces is None:
            vertices, faces = vertices.vertices, vertices.faces
        self.v = np.ascontiguousarray(vertices, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        self.fnorm = face_normals(self.v, self.faces)
        self.tree = build_tree(self.v, self.faces)

    def slicing(self, a, b, c, d):
        """Groups of ``(positions float32 (n, 3), edge_ids float32 (n, 3))`` for the plane."""
        left, right, tbox, ss, sl, pool = self.tree
        gptr, pos, eid = slice_plane(
            self.v, self.faces, self.fnorm, left, right, tbox, ss, sl, pool,
            F32(a), F32(b), F32(c), F32(d),
        )  # fmt: skip
        return [(pos[gptr[g] : gptr[g + 1]], eid[gptr[g] : gptr[g + 1]])
                for g in range(len(gptr) - 1)]  # fmt: skip
