"""TRACE sulcal / gyral point detection (``SulcalPoint.cpp`` / ``GyralPoint.cpp``, ded932d).

``SulcalPoint`` and ``GyralPoint`` are the same algorithm with opposite polarity; they differ only in

========================  ==============================  ==============================
                          sulcal                          gyral
========================  ==============================  ==============================
seed test                 ``curv < t  and  SI < -0.1``    ``curv > t  and  SI > 0.2``
``m_curvature``           ``cmin``                        ``cmax``
``m_tangent``             ``umin``                        ``umax``
``m_direction``           ``umax``                        ``umin``
accepted when             ``away < 0.5f``                 ``away == 0.0f``
========================  ==============================  ==============================

(``cmin/cmax/umin/umax`` as named by ``SurfaceUtil::curvature``; see
:mod:`pybiscuit.mesh.curvature` for MeshLib's naming quirk.)  For every seed vertex the surface is
cut with the plane through the vertex whose normal is the principal direction; on the contour that
passes through the vertex the line-simplification of ``maxDeviation`` selects fundic points, and the
vertex becomes a sulcal/gyral point when it is (``away == 0``) or lies within 0.5 mm of one.

All arithmetic is float32 with the C++ association order (``MathVector`` is float); each seed is
independent.  See ``docs/ports/trace.md``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ..mesh.curvature import curvature
from ..mesh.geometry import face_normals, vertex_normals
from ..mesh.slicer import build_tree, slice_plane

__all__ = ["PointFields", "detect_points", "point_fields", "seed_mask"]

F32 = np.float32
_FLT_MAX = np.float32(3.4028234663852886e38)
_TWO_OVER_PI = 2.0 / 3.141592653589793238462643  # (double)2.0f / PI


class PointFields:
    """Per-vertex fields of ``SulcalPoint::setCurvature`` (all float32)."""

    __slots__ = ("curvature", "direction", "shape_index", "tangent")

    def __init__(self, curvature, shape_index, direction, tangent):
        self.curvature = curvature
        self.shape_index = shape_index
        self.direction = direction
        self.tangent = tangent


@njit(cache=True, error_model="numpy")
def _shape_index(cmin, cmax):
    n = cmin.shape[0]
    si = np.empty(n, dtype=np.float32)
    for i in range(n):
        if cmax[i] == cmin[i]:
            si[i] = 1.0 if (cmax[i] + cmin[i] > 0) else -1.0
        else:
            si[i] = np.float32(_TWO_OVER_PI * np.float64(math.atan((cmax[i] + cmin[i]) / (cmax[i] - cmin[i]))))
    return si


@njit(cache=True, error_model="numpy")
def _orient_tangent(direction, tangent, normal, gyral):
    n = direction.shape[0]
    for i in range(n):
        if gyral:
            ax, ay, az = tangent[i, 0], tangent[i, 1], tangent[i, 2]
            bx, by, bz = direction[i, 0], direction[i, 1], direction[i, 2]
        else:
            ax, ay, az = direction[i, 0], direction[i, 1], direction[i, 2]
            bx, by, bz = tangent[i, 0], tangent[i, 1], tangent[i, 2]
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx
        s = math.sqrt(nx * nx + ny * ny + nz * nz)
        if s > 0:
            nx /= s
            ny /= s
            nz /= s
        if nx * normal[i, 0] + ny * normal[i, 1] + nz * normal[i, 2] < 0:
            tangent[i, 0] *= -1
            tangent[i, 1] *= -1
            tangent[i, 2] *= -1


def point_fields(vertices, faces, iter_tensor: int, gyral: bool) -> PointFields:
    """``SulcalPoint`` / ``GyralPoint`` constructor: curvature, shape index, direction, tangent."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    c = curvature(v, f, int(iter_tensor), dtype=np.float32)
    si = _shape_index(c.cmin, c.cmax)
    if gyral:
        curv, tangent, direction = c.cmax, c.umax.copy(), c.umin.copy()
    else:
        curv, tangent, direction = c.cmin, c.umin.copy(), c.umax.copy()
    _orient_tangent(direction, tangent, vertex_normals(v, f), bool(gyral))
    return PointFields(curv, si, direction, tangent)


def seed_mask(fields: PointFields, threshold: float, gyral: bool) -> np.ndarray:
    """``setSeed(threshold)``: float32 ``threshold`` against the field, shape index in double."""
    t = np.float32(threshold)
    si = fields.shape_index.astype(np.float64)
    if gyral:
        return (fields.curvature > t) & (si > 0.2)
    return (fields.curvature < t) & (si < -0.1)


# --------------------------------------------------------------------------- #
# contour element helpers (circular doubly linked lists as index arrays)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy", inline="always")
def _nrm(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@njit(cache=True, error_model="numpy")
def _longest_axis(pos, nxt, deleted, start):
    """``longestAxis``: returns ``(begin, end)`` element indices."""
    v0x, v0y, v0z = pos[start, 0], pos[start, 1], pos[start, 2]
    v1x = np.float32(0.0)
    v1y = np.float32(0.0)
    v1z = np.float32(0.0)
    max_len = np.float32(-1.0)
    begin = start
    end = -1
    it = start
    while True:
        if not deleted[it]:
            ln = _nrm(v0x - pos[it, 0], v0y - pos[it, 1], v0z - pos[it, 2])
            if max_len < ln:
                max_len = ln
                v1x, v1y, v1z = pos[it, 0], pos[it, 1], pos[it, 2]
                end = it
        it = nxt[it]
        if it == start:
            break
    it = start
    while True:
        if not deleted[it]:
            ln = _nrm(v1x - pos[it, 0], v1y - pos[it, 1], v1z - pos[it, 2])
            if max_len < ln:
                max_len = ln
                begin = it
        it = nxt[it]
        if it == start:
            break
    return begin, end


@njit(cache=True, error_model="numpy")
def _arbitrary_axis(pos, nxt, deleted, start, dx, dy, dz, has_center, cx, cy, cz):
    """``arbitraryAxis``; ``has_center`` False -> the center-less overload.  Returns (begin, end)."""
    begin = -1
    end = -1
    s = _nrm(dx, dy, dz)  # T.unit()
    tx, ty, tz = dx, dy, dz
    if s > 0:
        tx /= s
        ty /= s
        tz /= s
    if not has_center:
        px, py, pz = pos[start, 0], pos[start, 1], pos[start, 2]
        max_dist = np.float32(0.0)
        it = start
        while nxt[it] != start:
            if not deleted[it]:
                qx = pos[it, 0] - px
                qy = pos[it, 1] - py
                qz = pos[it, 2] - pz
                dist = abs(qx * tx + qy * ty + qz * tz)
                if dist >= max_dist:
                    max_dist = dist
                    end = it
            it = nxt[it]
        begin = start
        if end == -1:
            end = start
    else:
        min_dist = _FLT_MAX
        max_dist = -_FLT_MAX
        n = nxt.shape[0]
        cand = np.empty(n, dtype=np.int64)
        nc = 0
        it = start
        while True:
            if not deleted[it]:
                d1x = pos[it, 0] - cx
                d1y = pos[it, 1] - cy
                d1z = pos[it, 2] - cz
                nx_ = nxt[it]
                d2x = pos[nx_, 0] - cx
                d2y = pos[nx_, 1] - cy
                d2z = pos[nx_, 2] - cz
                # T.cross(D1) * T.cross(D2)
                a0 = ty * d1z - tz * d1y
                a1 = tz * d1x - tx * d1z
                a2 = tx * d1y - ty * d1x
                b0 = ty * d2z - tz * d2y
                b1 = tz * d2x - tx * d2z
                b2 = tx * d2y - ty * d2x
                sign = a0 * b0 + a1 * b1 + a2 * b2
                if sign <= 0:
                    cand[nc] = it
                    nc += 1
            it = nxt[it]
            if it == start:
                break
        for q in range(nc):
            e = cand[q]
            dist = _nrm(pos[e, 0] - cx, pos[e, 1] - cy, pos[e, 2] - cz)
            if min_dist > dist:
                min_dist = dist
                begin = e
            elif max_dist < dist:
                max_dist = dist
                end = e
        if begin == -1:
            begin = start
        if end == -1:
            end = start
    return begin, end


@njit(cache=True, error_model="numpy")
def _tan_axis(pos, nxt, deleted, start, tangent, normal):
    """``tanAxis(list, ..., tangent, normal)`` -> (begin, end)."""
    tx, ty, tz = tangent[0], tangent[1], tangent[2]
    s = _nrm(tx, ty, tz)
    if s > 0:
        tx /= s
        ty /= s
        tz /= s
    nx, ny, nz = normal[0], normal[1], normal[2]
    s = _nrm(nx, ny, nz)
    if s > 0:
        nx /= s
        ny /= s
        nz /= s
    bx = ty * nz - tz * ny
    by = tz * nx - tx * nz
    bz = tx * ny - ty * nx
    s = _nrm(bx, by, bz)
    if s > 0:
        bx /= s
        by /= s
        bz /= s
    begin, end = _arbitrary_axis(pos, nxt, deleted, start, bx, by, bz, False,
                                 np.float32(0.0), np.float32(0.0), np.float32(0.0))  # fmt: skip
    p1x, p1y, p1z = pos[end, 0], pos[end, 1], pos[end, 2]
    p2x, p2y, p2z = pos[begin, 0], pos[begin, 1], pos[begin, 2]
    b1x = p1x - p2x
    b1y = p1y - p2y
    b1z = p1z - p2z
    ln = _nrm(b1x, b1y, b1z) * np.float32(0.5)
    s = _nrm(b1x, b1y, b1z)
    if s > 0:
        b1x /= s
        b1y /= s
        b1z /= s
    qx = pos[start, 0] + b1x * ln
    qy = pos[start, 1] + b1y * ln
    qz = pos[start, 2] + b1z * ln
    if (p2x - qx) * (p1x - qx) + (p2y - qy) * (p1y - qy) + (p2z - qz) * (p1z - qz) > 0:
        qx = pos[start, 0] - b1x * ln
        qy = pos[start, 1] - b1y * ln
        qz = pos[start, 2] - b1z * ln
    return _arbitrary_axis(pos, nxt, deleted, start, tangent[0], tangent[1], tangent[2], True,
                           qx, qy, qz)  # fmt: skip


@njit(cache=True, error_model="numpy")
def _max_deviation(pos, nxt, deleted, selected, is_vertex, begin, end, w, h):
    """``maxDeviation`` (the recursion is independent per interval, so a stack is equivalent)."""
    cap = nxt.shape[0] * 2 + 8
    sb = np.empty(cap, dtype=np.int64)
    se = np.empty(cap, dtype=np.int64)
    sb[0] = begin
    se[0] = end
    sp = 1
    while sp > 0:
        sp -= 1
        b = sb[sp]
        e = se[sp]
        v0x, v0y, v0z = pos[b, 0], pos[b, 1], pos[b, 2]
        dx = pos[e, 0] - v0x
        dy = pos[e, 1] - v0y
        dz = pos[e, 2] - v0z
        max_len = _nrm(dx, dy, dz)
        ax = dx / max_len
        ay = dy / max_len
        az = dz / max_len
        if max_len < w:
            continue
        max_h = np.float32(0.0)
        max_dev = -1
        it = nxt[b]
        while it != e:
            if not deleted[it]:
                rx = v0x - pos[it, 0]
                ry = v0y - pos[it, 1]
                rz = v0z - pos[it, 2]
                cx = ry * az - rz * ay
                cy = rz * ax - rx * az
                cz = rx * ay - ry * ax
                height = _nrm(cx, cy, cz)
                if max_h < height:
                    max_h = height
                    max_dev = it
            it = nxt[it]
        if max_h < h:
            continue
        if max_dev == -1:
            continue
        selected[max_dev] = True
        if is_vertex[max_dev]:
            continue
        if sp + 2 > sb.shape[0]:
            sb = np.concatenate((sb, np.empty(sb.shape[0], dtype=np.int64)))
            se = np.concatenate((se, np.empty(se.shape[0], dtype=np.int64)))
        sb[sp] = b
        se[sp] = max_dev
        sb[sp + 1] = max_dev
        se[sp + 1] = e
        sp += 2


@njit(cache=True, error_model="numpy")
def _find_candidate(pos, nxt, deleted, selected, is_vertex, begin, end, h):
    """``findCandidate(list, 0, h, ...)``; returns the new ``list->data`` (= ``end`` after swaps)."""
    w = np.float32(0.0)
    for i in range(2):
        if i == 1:
            tmp = begin
            begin = end
            end = tmp
        it = nxt[begin]
        while it != end:
            if is_vertex[it]:
                _max_deviation(pos, nxt, deleted, selected, is_vertex, begin, end, w, h)
                break
            it = nxt[it]
    return end


@njit(cache=True, error_model="numpy")
def _local_maxima(v, faces, fnorm, left, right, tbox, seg_start, seg_len, pool, idx, direction,
                  tangent, h, gyral):  # fmt: skip
    """``localMaxima(idx)``; returns True when ``idx`` becomes a sulcal / gyral point."""
    away = _FLT_MAX
    accepted = False
    px, py, pz = v[idx, 0], v[idx, 1], v[idx, 2]
    nx = direction[idx, 0]
    ny = direction[idx, 1]
    nz = direction[idx, 2]
    s = _nrm(nx, ny, nz)  # dn.unit()
    if s > 0:
        nx /= s
        ny /= s
        nz /= s
    d = (-px) * nx + (-py) * ny + (-pz) * nz
    gptr, pos_all, eid_all = slice_plane(v, faces, fnorm, left, right, tbox, seg_start, seg_len,
                                         pool, nx, ny, nz, d)  # fmt: skip
    ng = gptr.shape[0] - 1
    for g in range(ng):
        n = gptr[g + 1] - gptr[g]
        base = gptr[g]
        # contour(): a circular list per group; data = last element with id == (idx, idx)
        data = 0
        has_s = False
        is_vertex = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if eid_all[base + i, 0] == eid_all[base + i, 1] and eid_all[base + i, 0] == idx:
                has_s = True
                data = i
                is_vertex[i] = True
        if not has_s:
            continue
        pos = pos_all[base : base + n]
        nxt = np.empty(n, dtype=np.int64)
        for i in range(n):
            nxt[i] = i + 1 if i + 1 < n else 0
        deleted = np.zeros(n, dtype=np.bool_)
        selected = np.zeros(n, dtype=np.bool_)
        query = -1
        it = data
        while True:
            if eid_all[base + it, 0] == idx and eid_all[base + it, 1] == idx:
                query = it
            it = nxt[it]
            if it == data:
                break
        for k in range(2):
            if k == 0:
                begin, end = _longest_axis(pos, nxt, deleted, data)
            else:
                begin, end = _tan_axis(pos, nxt, deleted, data, tangent[idx], direction[idx])
            data = _find_candidate(pos, nxt, deleted, selected, is_vertex, begin, end, h)
            if selected[query]:
                break
        if selected[query]:
            away = np.float32(0.0)
        else:
            it = data
            while True:
                if (not deleted[it]) and selected[it]:
                    dist = _nrm(pos[it, 0] - px, pos[it, 1] - py, pos[it, 2] - pz)
                    if dist < away:
                        away = dist
                it = nxt[it]
                if it == data:
                    break
        if gyral:
            if away == np.float32(0.0):
                accepted = True
        else:
            if away < np.float32(0.5):
                accepted = True
    return accepted


@njit(cache=True, error_model="numpy")
def _detect(v, faces, fnorm, left, right, tbox, seg_start, seg_len, pool, seeds, direction,
            tangent, h, gyral):  # fmt: skip
    out = np.zeros(v.shape[0], dtype=np.bool_)
    for i in range(v.shape[0]):
        if seeds[i]:
            out[i] = _local_maxima(v, faces, fnorm, left, right, tbox, seg_start, seg_len, pool,
                                   i, direction, tangent, h, gyral)  # fmt: skip
    return out


def detect_points(
    vertices, faces, fields: PointFields, seeds: np.ndarray, ls_threshold: float, gyral: bool
) -> np.ndarray:
    """Sulcal / gyral point mask: ``localMaxima`` for every seed vertex (``Extraction.cpp`` L99-L118)."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    fn = face_normals(v, f)
    left, right, tbox, ss, sl, pool = build_tree(v, f)
    return _detect(
        v, f, fn, left, right, tbox, ss, sl, pool, np.ascontiguousarray(seeds, dtype=np.bool_),
        fields.direction, fields.tangent, np.float32(ls_threshold), bool(gyral),
    )  # fmt: skip
