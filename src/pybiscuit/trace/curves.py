# ruff: noqa: E501
"""TRACE curve delineation (``SulcalCurve.cpp`` / ``GyralCurve.cpp``, ded932d).

The two C++ classes are the same program with different angle defaults; every other line (the
Dijkstra tracing, junction bookkeeping, pruning) is identical in the code paths ``Extraction.cpp``
reaches (``run`` -> ``grouping`` / ``refineCurves``).  ``Curves(gyral=...)`` selects:

======================================  =================  =================
default argument                        sulcal             gyral
======================================  =================  =================
``curve_dijkstra`` inner2               0                  cos(pi/3)
``extendCurves_dijkstra`` inner1/inner2 cos(pi/3), cos(pi/3)  cos(pi/5), cos(pi/4)
``joinCurves(e1, e2)`` inner            cos(pi/4)          cos(pi/5)
``refineCurves`` first ``current``      ``nCurves()``      uninitialised (see below)
======================================  =================  =================

``GyralCurve::refineCurves`` leaves ``current`` uninitialised before its first
``previous = current`` (undefined behaviour upstream).  The clang-built oracle behaves as if the
loop test were already satisfied (one pass: verified on a folded sphere, where a second pass flips
the direction of some curves); a compiler that yields an arbitrary stack value would run at least
two passes.  Default ``uninit_current="exit"`` follows the oracle; the image's compiler is unknown
(open question).  Unreached upstream code (``curve``, ``extendCurves``,
``grouping`` defaults, ``joinCurves(float)``) is not ported.

State is kept as arrays (one slot per detected point, in vertex order); curve lists are a doubly
linked list of :class:`CurveList` objects in creation order, exactly as the C++ (the list order is
the output order).  Pairwise geodesic distances are a sparse symmetric float32 matrix whose
missing entries stand for upstream's ``FLT_MAX``.  All geometry is float32 (``MathVector``).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ..mesh.geodesic import GeodesicSolver, sparse_pairwise_distances

__all__ = ["CurveList", "Curves"]

F32 = np.float32
FLT_MAX = np.float32(3.4028234663852886e38)


# --------------------------------------------------------------------------- #
# float32 vector helpers
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _unit(x, y, z):
    s = math.sqrt(x * x + y * y + z * z)
    if s > 0:
        return x / s, y / s, z / s
    return x, y, z


@njit(cache=True, error_model="numpy")
def _rotation(ax, ay, az, theta):
    """``Coordinate::rotation(const float*, float, float*)`` -> 9 floats (row major)."""
    mat = np.zeros(9, dtype=np.float32)
    mat[0] = 1
    mat[4] = 1
    mat[8] = 1
    s = math.sqrt(ax * ax + ay * ay + az * az)
    if s == 0:
        return mat
    ax, ay, az = _unit(ax, ay, az)
    A = np.array([0, -az, ay, az, 0, -ax, -ay, ax, 0], dtype=np.float32)
    st = math.sin(theta)
    for i in range(9):
        mat[i] += A[i] * st
    om = np.float32(1.0) - math.cos(theta)
    for i in range(9):
        for j in range(3):
            mat[i] += A[j + (i // 3) * 3] * A[j * 3 + i % 3] * om
    return mat


@njit(cache=True, error_model="numpy")
def _sphmean(a0, a1, a2, b0, b1, b2):
    """``Coordinate::sphmean(v1, v2, v, 0.5f)``."""
    if a0 == b0 and a1 == b1 and a2 == b2:
        return a0, a1, a2
    u0, u1, u2 = _unit(a0, a1, a2)
    w0, w1, w2 = _unit(b0, b1, b2)
    c = u0 * w0 + u1 * w1 + u2 * w2
    angle = math.acos(c) * np.float32(0.5) * (np.float32(1.0) - np.float32(0.5))
    x0 = u1 * w2 - u2 * w1
    x1 = u2 * w0 - u0 * w2
    x2 = u0 * w1 - u1 * w0
    x0, x1, x2 = _unit(x0, x1, x2)
    m = _rotation(x0, x1, x2, angle)
    v0 = m[0] * a0 + m[1] * a1 + m[2] * a2
    v1 = m[3] * a0 + m[4] * a1 + m[5] * a2
    v2 = m[6] * a0 + m[7] * a1 + m[8] * a2
    # V1.cross(v) * V2.cross(v) > 0
    p0 = u1 * v2 - u2 * v1
    p1 = u2 * v0 - u0 * v2
    p2 = u0 * v1 - u1 * v0
    q0 = w1 * v2 - w2 * v1
    q1 = w2 * v0 - w0 * v2
    q2 = w0 * v1 - w1 * v0
    if p0 * q0 + p1 * q1 + p2 * q2 > 0:
        m = _rotation(x0, x1, x2, -angle)
        v0 = m[0] * a0 + m[1] * a1 + m[2] * a2
        v1 = m[3] * a0 + m[4] * a1 + m[5] * a2
        v2 = m[6] * a0 + m[7] * a1 + m[8] * a2
    return _unit(v0, v1, v2)


@njit(cache=True, error_model="numpy")
def _dget(ip, ix, dv, a, b):
    lo = ip[a]
    hi = ip[a + 1]
    end = hi
    while lo < hi:
        mid = (lo + hi) // 2
        if ix[mid] < b:
            lo = mid + 1
        else:
            hi = mid
    if lo < end and ix[lo] == b:
        return dv[lo]
    return FLT_MAX


@njit(cache=True, error_model="numpy")
def _exp32(x):
    return math.exp(x)


# --------------------------------------------------------------------------- #
# numba kernels for the O(n^2) loops
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _detect_end_points(ip, ix, dv, pos, deleted, header, cand, th, inner):
    c = 0
    n = pos.shape[0]
    for i in range(n):
        cand[i] = False
        if deleted[i] or header[i] != -1:
            continue
        cand[i] = True
        for p in range(ip[i], ip[i + 1]):
            if not cand[i]:
                break
            j = ix[p]
            if dv[p] < th and (not deleted[j]) and header[j] == -1:
                vx, vy, vz = _unit(pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1],
                                   pos[j, 2] - pos[i, 2])  # fmt: skip
                for q in range(p + 1, ip[i + 1]):
                    if not cand[i]:
                        break
                    k = ix[q]
                    if dv[q] < th and (not deleted[k]) and header[k] == -1:
                        wx, wy, wz = _unit(pos[k, 0] - pos[i, 0], pos[k, 1] - pos[i, 1],
                                           pos[k, 2] - pos[i, 2])  # fmt: skip
                        if vx * wx + vy * wy + vz * wz < inner:
                            cand[i] = False
        if cand[i]:
            c += 1
    return c


@njit(cache=True, error_model="numpy")
def _dijkstra(ip, ix, dv, pos, direction, orientation, deleted, header, traced, dij_node,
              dij_dist, start, hid, th, inner1, inner2, skip_deleted, abs_start):  # fmt: skip
    """Shared Dijkstra of ``curve_dijkstra`` / ``extendCurves_dijkstra`` (the loop up to the
    path selection).  ``hid`` is the id of the list being grown."""
    n = pos.shape[0]
    for i in range(n):
        dij_node[i] = -1
        dij_dist[i] = FLT_MAX
    dij_node[start] = start
    dij_dist[start] = 0
    q = np.empty(n + 1, dtype=np.int64)
    qn = 1
    q[0] = start
    one = np.float32(1.0)
    while qn > 0:
        min_d = FLT_MAX
        active = -1
        for i in range(qn):
            if dij_dist[q[i]] < min_d:
                min_d = dij_dist[q[i]]
                active = q[i]
        k = 0
        for i in range(qn):  # erase(remove(Q, active))
            if q[i] != active:
                q[k] = q[i]
                k += 1
        qn = k
        if active == -1:  # all remaining entries have dist >= FLT_MAX: unreachable upstream
            break
        vx, vy, vz = pos[active, 0], pos[active, 1], pos[active, 2]
        b0, b1, b2 = direction[active, 0], direction[active, 1], direction[active, 2]
        if b0 * orientation[active, 0] + b1 * orientation[active, 1] + b2 * orientation[active, 2] < 0:
            b0 = -b0
            b1 = -b1
            b2 = -b2
        for p in range(ip[active], ip[active + 1]):
            i = ix[p]
            if active == i:
                continue
            if skip_deleted and deleted[i]:
                continue
            dist = dv[p]
            if not (dist < th):
                continue
            wdist = dij_dist[active]
            c0, c1, c2 = direction[i, 0], direction[i, 1], direction[i, 2]
            if b0 * c0 + b1 * c1 + b2 * c2 < 0:
                c0 = -c0
                c1 = -c1
                c2 = -c2
            s0, s1, s2 = _sphmean(b0, b1, b2, c0, c1, c2)
            ex, ey, ez = pos[i, 0] - vx, pos[i, 1] - vy, pos[i, 2] - vz
            ux, uy, uz = _unit(ex, ey, ez)
            cx = uy * s2 - uz * s1
            cy = uz * s0 - ux * s2
            cz = ux * s1 - uy * s0
            w = _exp32(one * math.sqrt(cx * cx + cy * cy + cz * cz))
            wdist += dist * w
            adj = header[i] != -1 and header[i] != hid
            dev = s0 * ux + s1 * uy + s2 * uz
            if abs_start and active == start:
                dev = abs(dev)
            if (not adj and dev > inner1) or (adj and dev > inner2):
                if dij_dist[i] > wdist:
                    dij_dist[i] = wdist
                    dij_node[i] = active
                    if not traced[i]:
                        k = 0
                        for t in range(qn):
                            if q[t] != i:
                                q[k] = q[t]
                                k += 1
                        qn = k
                        q[qn] = i
                        qn += 1
                        orientation[i, 0] = s0
                        orientation[i, 1] = s1
                        orientation[i, 2] = s2
                        if ex * s0 + ey * s1 + ez * s2 < 0:
                            orientation[i, 0] *= -1
                            orientation[i, 1] *= -1
                            orientation[i, 2] *= -1


@njit(cache=True, error_model="numpy")
def _curve_target(ip, ix, dv, pos, traced, dij_node, dij_dist, dij_end, cand, is_end, start, th):
    """Target choice of ``curve_dijkstra`` after the Dijkstra (traced points, else endpoints)."""
    n = pos.shape[0]
    n_valid = 0
    target = start
    for i in range(n):
        dij_end[i] = False
    for i in range(n):
        if dij_node[i] == -1 or start == i or not traced[i]:
            continue
        dij_end[i] = True
        n_valid += 1
    if n_valid > 0:
        min_d = FLT_MAX
        for i in range(n):
            if dij_end[i] and dij_dist[i] < min_d:
                target = i
                min_d = dij_dist[i]
        return target
    for i in range(n):
        if dij_node[i] == -1:
            continue
        dij_end[i] = True
        for p in range(ip[i], ip[i + 1]):
            if not dij_end[i]:
                break
            j = ix[p]
            if i == j:
                continue
            if dij_node[j] != -1 and dv[p] < th:
                dij_end[i] = dij_dist[i] > dij_dist[j]
        if dij_end[i] and (cand[i] or is_end[i]):
            n_valid += 1
    if n_valid == 0:
        for i in range(n):
            if dij_node[i] == -1 or (not cand[i]) or (not is_end[i]):
                continue
            dij_end[i] = True
            n_valid += 1
    if n_valid == 0:
        for i in range(n):
            if dij_node[i] == -1:
                continue
            dij_end[i] = True
            for p in range(ip[i], ip[i + 1]):
                if not dij_end[i]:
                    break
                j = ix[p]
                if i == j:
                    continue
                if dij_node[j] != -1 and dv[p] < th:
                    dij_end[i] = dij_dist[i] > dij_dist[j]
    max_d = np.float32(0.0)
    for i in range(n):
        if dij_end[i] and dij_dist[i] > max_d:
            target = i
            max_d = dij_dist[i]
    return target


@njit(cache=True, error_model="numpy")
def _extend_target(pos, traced, dij_node, dij_dist, dij_end, header, hid, elem):
    n = pos.shape[0]
    n_valid = 0
    for i in range(n):
        dij_end[i] = False
    for i in range(n):
        if header[i] == hid or i == elem or (not traced[i]) or dij_node[i] == -1:
            continue
        dij_end[i] = True
        n_valid += 1
    target = elem
    min_d = FLT_MAX
    for i in range(n):
        if dij_end[i] and dij_dist[i] < min_d:
            target = i
            min_d = dij_dist[i]
    return n_valid, target


class CurveList:
    __slots__ = ("id", "item", "length", "next", "prev", "visit")

    def __init__(self, lid):
        self.id = lid
        self.item: list[int] = []
        self.length = F32(0)
        self.next = None
        self.prev = None
        self.visit = False


class Curves:
    """One ``SulcalCurve`` / ``GyralCurve`` object over the detected point vertices."""

    def __init__(self, vertices, faces, point_mask, direction, *, gyral: bool, solver=None,
                 verbose: bool = False, uninit_current: str = "exit"):  # fmt: skip
        if uninit_current not in ("exit", "continue"):
            raise ValueError("uninit_current must be 'exit' or 'continue'")
        self.uninit_current = uninit_current
        self.gyral = bool(gyral)
        self.verbose = verbose
        self.v32 = np.ascontiguousarray(vertices, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        self.solver = solver if solver is not None else GeodesicSolver(self.v32, self.faces)
        self.vid = np.flatnonzero(np.asarray(point_mask)).astype(np.int32)
        n = self.vid.shape[0]
        self.n = n
        self.pos = np.ascontiguousarray(self.v32[self.vid])
        self.direction = np.ascontiguousarray(np.asarray(direction, dtype=np.float32)[self.vid])
        self.orientation = np.zeros((n, 3), dtype=np.float32)
        self.header = np.full(n, -1, dtype=np.int64)
        self.is_end = np.zeros(n, dtype=np.bool_)
        self.is_junc = np.zeros(n, dtype=np.bool_)
        self.n_junc = np.zeros(n, dtype=np.int64)
        self.traced = np.zeros(n, dtype=np.bool_)
        self.deleted = np.zeros(n, dtype=np.bool_)
        self.cand = np.zeros(n, dtype=np.bool_)
        self.dij_node = np.full(n, -1, dtype=np.int64)
        self.dij_dist = np.full(n, FLT_MAX, dtype=np.float32)
        self.dij_end = np.zeros(n, dtype=np.bool_)
        self.head: CurveList | None = None
        self.lists: dict[int, CurveList] = {}
        self._next_id = 0
        pi = math.pi
        c = lambda a: F32(math.cos(pi / a))  # noqa: E731
        if self.gyral:
            self.inner_curve2 = c(3.0)
            self.inner_ext1, self.inner_ext2 = c(5.0), c(4.0)
            self.inner_join = c(5.0)
        else:
            self.inner_curve2 = F32(0.0)
            self.inner_ext1, self.inner_ext2 = c(3.0), c(3.0)
            self.inner_join = c(4.0)
        self.inner_curve1 = c(4.0)
        self.th = None

    # ------------------------------------------------------------------ lists
    def _new_list(self) -> CurveList:
        lst = CurveList(self._next_id)
        self._next_id += 1
        self.lists[lst.id] = lst
        return lst

    def _tail(self):
        t = self.head
        while t is not None and t.next is not None:
            t = t.next
        return t

    def _iter(self):
        it = self.head
        while it is not None:
            yield it
            it = it.next

    def _unlink(self, lst: CurveList):
        if lst.next is not None:
            lst.next.prev = lst.prev
        if lst.prev is not None:
            lst.prev.next = lst.next
        else:
            self.head = lst.next
        lst.item = []
        self.lists.pop(lst.id, None)

    def n_curves(self) -> int:
        return sum(1 for _ in self._iter())

    def _d(self, a, b):
        return _dget(self.ip, self.ix, self.dv, a, b)

    def _hl(self, e):
        h = self.header[e]
        return self.lists[h] if h != -1 else None

    # --------------------------------------------------------------- run
    def set_threshold(self, t1, t2, t3, t4):
        self.th = tuple(F32(t) for t in (t1, t2, t3, t4))

    def run(self):
        t1, t2, t3, t4 = self.th
        dm = sparse_pairwise_distances(self.solver, self.vid, float(t1))
        self.ip, self.ix, self.dv = dm.indptr.astype(np.int64), dm.indices.astype(np.int64), dm.data
        self._grouping(t1, t2, t3)
        self._refine(t1, t4)
        self.integrity = self.test_integrity()
        self.point_mask = np.zeros(self.n, dtype=np.bool_)
        for lst in self._iter():
            for e in lst.item:
                self.point_mask[e] = True

    def _grouping(self, th1, th2, th3):
        prev_n, cur = 0, 0
        while True:
            prev_n = cur
            self._delineation(th1, th2, th3)
            self._delete_nearest(th2)
            cur = self.n_curves()
            if prev_n == cur:
                break

    def _delete_nearest(self, th):
        for lst in self._iter():
            for e in lst.item:
                for p in range(self.ip[e], self.ip[e + 1]):
                    j = self.ix[p]
                    if self.deleted[j] or self.header[j] != -1:
                        continue
                    if self.dv[p] < th:
                        self._delete_elem(j)
                        self.cand[j] = False

    def _delete_elem(self, e):
        self.header[e] = -1
        self.deleted[e] = True
        self.is_end[e] = False
        self.is_junc[e] = False

    def _delineation(self, th1, th2, th3):
        n = self.n
        for i in range(n):
            if not self.deleted[i] and self.header[i] == -1:
                self.traced[i] = False
        inner = F32(0.0)
        while True:
            c = _detect_end_points(self.ip, self.ix, self.dv, self.pos, self.deleted, self.header,
                                   self.cand, th3, inner)  # fmt: skip
            inner = F32(np.float64(inner) - 0.1)
            if not (c == 0 and inner >= F32(-1.0)):
                break
        ids = np.flatnonzero(self.cand)
        order = np.lexsort((self.pos[ids, 2], self.pos[ids, 1], self.pos[ids, 0]))
        prev = self._tail()
        for i in ids[order]:
            if self.traced[i] or not self.cand[i] or self.deleted[i] or self.header[i] != -1:
                continue
            lst = self._new_list()
            end_point = self._curve_dijkstra(i, lst, th1)
            if len(lst.item) == 1:
                self._delete_elem(i)
                self.lists.pop(lst.id, None)
                continue
            lst.next = None
            lst.prev = prev
            lst.length = F32(0)
            if self.head is None:
                self.head = lst
            else:
                prev.next = lst
            prev = lst
            self.is_end[i] = True
            if self.header[end_point] != lst.id:
                self._join(lst.item[-1], end_point, self.inner_join)
        return self.n_curves()

    # ------------------------------------------------------------ tracing
    def _curve_dijkstra(self, cur, lst, th):
        self.traced[cur] = True
        self.header[cur] = lst.id
        self.orientation[cur] = self.direction[cur]
        _dijkstra(self.ip, self.ix, self.dv, self.pos, self.direction, self.orientation,
                  self.deleted, self.header, self.traced, self.dij_node, self.dij_dist, cur,
                  lst.id, th, self.inner_curve1, self.inner_curve2, True, True)  # fmt: skip
        target = _curve_target(self.ip, self.ix, self.dv, self.pos, self.traced, self.dij_node,
                               self.dij_dist, self.dij_end, self.cand, self.is_end, cur, th)  # fmt: skip
        marked = False
        i = target
        while i != cur:
            if not self.traced[i]:
                if not marked:
                    marked = True
                    self.is_end[i] = True
                lst.item.append(i)
                self.header[i] = lst.id
                self.traced[i] = True
                self.cand[i] = False
            i = self.dij_node[i]
        lst.item.append(cur)
        lst.item.reverse()
        self.cand[cur] = False
        if len(lst.item) > 1:
            self._update_orientation(lst)
        return target

    def _update_orientation(self, lst, only=None):
        it = lst.item
        for i in range(len(it)):
            e = it[i]
            if only is not None and only != e:
                continue
            if self.is_junc[e] and self.header[e] != lst.id:
                continue
            if i == 0:
                a, b = e, it[1]
            else:
                a, b = it[i - 1], e
            d = self.pos[b] - self.pos[a]
            vx, vy, vz = _unit(d[0], d[1], d[2])
            o = self.direction[e].copy()
            if o[0] * vx + o[1] * vy + o[2] * vz < 0:
                o = -o
            ux, uy, uz = _unit(vx, vy, vz)  # V.unit() again, as in the C++ argument
            s = _sphmean(o[0], o[1], o[2], ux, uy, uz)
            self.orientation[e] = s

    def _reverse(self, lst):
        lst.item.reverse()
        self._update_orientation(lst)

    def _join(self, e1, e2, inner):
        if not self.is_end[e1]:
            return
        l1 = self._hl(e1)
        l2 = self._hl(e2)
        if e1 == l1.item[0]:
            self._reverse(l1)
        if self.is_end[e2]:
            if e2 == l2.item[-1]:
                self._reverse(l2)
            p1, p2 = self.pos[e1], self.pos[e2]
            ux, uy, uz = _unit(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
            o = self.orientation[e2]
            q = self.pos[l2.item[1]]
            wx, wy, wz = _unit(q[0] - p2[0], q[1] - p2[1], q[2] - p2[2])
            if o[0] * ux + o[1] * uy + o[2] * uz > inner and wx * ux + wy * uy + wz * uz > inner:
                self.is_end[e1] = False
                self.is_end[e2] = False
                for j in l2.item:
                    l1.item.append(j)
                    if (not self.is_junc[j]) or (self.is_junc[j] and self.header[j] == l2.id):
                        self.header[j] = l1.id
                self._update_orientation(l1, e2)
                self._unlink(l2)
        else:
            self.is_junc[e2] = True
            self.n_junc[e2] += 1
            l1.item.append(e2)
            self.is_end[e1] = False

    def _test_loop(self, frm, to):
        frm.visit = True
        if to is None:
            return False
        if frm is to:
            return True
        for it in self._iter():
            if it.visit:
                continue
            a0, a1 = it.item[0], it.item[-1]
            b0, b1 = frm.item[0], frm.item[-1]
            if a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1:
                if self._test_loop(it, to):
                    return True
        return False

    def _extend(self, elem, th):
        if not self.is_end[elem]:
            return
        lst = self._hl(elem)
        if lst.item[0] == elem:
            self._reverse(lst)
        _dijkstra(self.ip, self.ix, self.dv, self.pos, self.direction, self.orientation,
                  self.deleted, self.header, self.traced, self.dij_node, self.dij_dist, elem,
                  lst.id, th, self.inner_ext1, self.inner_ext2, False, False)  # fmt: skip
        n_valid, target = _extend_target(self.pos, self.traced, self.dij_node, self.dij_dist,
                                         self.dij_end, self.header, lst.id, elem)  # fmt: skip
        if n_valid <= 0:
            return
        for it in self._iter():
            it.visit = False
        if self._test_loop(lst, self._hl(target)):
            return
        trace = []
        i = target
        while i != elem:
            trace.append(i)
            i = self.dij_node[i]
        trace.reverse()
        for k, i in enumerate(trace):
            hl = self._hl(elem)
            if self.header[i] != -1:
                last = hl.item[-1]
                self.is_end[last] = True
                self._join(last, i, F32(0))
            else:
                self.is_end[elem] = False
                hl.item.append(i)
                self.is_end[i] = k == len(trace) - 1
                self.header[i] = hl.id
                self.deleted[i] = False
                self.traced[i] = True
        self._update_orientation(self._hl(elem))

    # ------------------------------------------------------- refine / prune
    def _refine(self, th1, th4):
        self.cand[:] = False
        # gyral: `previous = current` reads an uninitialised int upstream (see module docstring);
        # `uninit_current="exit"` models the clang oracle (undef folds to "equal": one pass),
        # "continue" a value that never equals the count (>= two passes).
        current = self.n_curves() if not self.gyral else None
        while True:
            previous = current
            if previous is None:
                previous = -1 if self.uninit_current == "continue" else None
            it = self.head
            while it is not None:
                e1 = it.item[-1]
                if self.is_end[e1]:
                    self._extend(e1, th1)
                self._separate()
                e2 = it.item[0]
                if self.is_end[e2]:
                    self._extend(e2, th1)
                self._separate()
                it = it.next
            current = self.n_curves()
            if previous is None or previous == current:
                break
        self._separate()
        while self._prune(th4):
            pass

    def _separate(self):
        prev = self._tail()
        it = self.head
        while it is not None:
            for i in range(1, len(it.item) - 1):
                e = it.item[i]
                if self.is_junc[e]:
                    self.n_junc[e] += 1
                    new = self._new_list()
                    new.next = None
                    new.prev = prev
                    new.length = F32(0)
                    prev.next = new
                    size = len(it.item)
                    for j in range(i, size):
                        new.item.append(it.item[j])
                        if self.header[it.item[j]] == it.id:
                            self.header[it.item[j]] = new.id
                    del it.item[i + 1 :]
                    prev = new
                    break
            it = it.next

    def _update_length(self):
        with np.errstate(over="ignore"):
            for lst in self._iter():
                ln = F32(0)
                for i in range(1, len(lst.item)):
                    ln = F32(ln + self._d(lst.item[i], lst.item[i - 1]))
                lst.length = ln

    def _prune(self, th):
        self._update_length()
        min_len = FLT_MAX
        cand = None
        for lst in self._iter():
            if self.is_end[lst.item[0]] or self.is_end[lst.item[-1]]:
                elem = None
                if self.is_junc[lst.item[0]]:
                    elem = lst.item[0]
                if self.is_junc[lst.item[-1]]:
                    elem = lst.item[-1]
                w = F32(0)
                if elem is not None:
                    d = self.direction[elem]
                    for l1 in self._iter():
                        for end in (l1.item[0], l1.item[-1]):
                            if elem == end:
                                dot = F32(abs(d[0] * self.direction[end][0] + d[1] * self.direction[end][1]
                                              + d[2] * self.direction[end][2]))
                                w = max(w, F32(_exp32(F32(dot / F32(2)))))
                else:
                    w = F32(1)
                with np.errstate(over="ignore"):
                    prod = F32(lst.length * w)
                if min_len > prod:
                    min_len = prod
                    cand = lst
        if min_len < th:
            self._delete_curves(cand)
            self._update_length()
        return bool(min_len < th)

    def _delete_curves(self, lst):
        for e in list(lst.item):
            if self.is_junc[e]:
                self._delete_junction(e, lst)
            else:
                self._delete_elem(e)
        self._unlink(lst)

    def _delete_junction(self, e, parent):
        self.n_junc[e] -= 1
        if self.n_junc[e] == 0:
            for it in self._iter():
                if it is parent:
                    continue
                if e == it.item[0] or e == it.item[-1]:
                    self.is_junc[e] = False
                    self.is_end[e] = True
                    self.header[e] = it.id
                    break
        elif self.n_junc[e] == 1:
            l1 = l2 = None
            for it in self._iter():
                if it is parent:
                    continue
                if e == it.item[0] or e == it.item[-1]:
                    if l1 is None:
                        l1 = it
                    else:
                        l2 = it
                        break
            self.is_junc[e] = False
            self.is_end[e] = False
            self.n_junc[e] = 0
            if l1 is not None and l2 is not None:
                if l1.item[-1] != e:
                    self._reverse(l1)
                if l2.item[0] != e:
                    self._reverse(l2)
                self.header[e] = l1.id
                for i in range(1, len(l2.item)):
                    l1.item.append(l2.item[i])
                    if self.header[l2.item[i]] == l2.id:
                        self.header[l2.item[i]] = l1.id
                self._unlink(l2)
        else:
            for it in self._iter():
                if it is parent:
                    continue
                if e == it.item[0] or e == it.item[-1]:
                    self.header[e] = it.id
                    break

    # ------------------------------------------------------------ misc
    def simplify(self, threshold=4.5, inner=None):
        """``SimplifyCurves`` (sulcal defaults 4.5; gyral 2.5), bugs of the original kept."""
        threshold = F32(threshold)
        if inner is None:
            inner = F32(math.cos(math.pi / 10.0))
        it = self.head
        while it is not None:
            nxt = it.next
            arc = F32(0)
            i = 0
            while i < len(it.item):
                if 0 < i < len(it.item) - 1 and not self.is_junc[it.item[i]]:
                    v1 = self.pos[it.item[i]] - self.pos[it.item[i - 1]]
                    v2 = self.pos[it.item[i + 1]] - self.pos[it.item[i]]
                    arc = F32(arc + self._d(i - 1, i))  # upstream indexes by position, not id
                    a = _unit(v1[0], v1[1], v1[2])
                    b = _unit(v2[0], v2[1], v2[2])
                    if a[0] * b[0] + a[1] * b[1] + a[2] * b[2] > inner and arc < threshold:
                        self._delete_elem(it.item[i])
                        del it.item[i]
                        continue
                    arc = F32(0)
                i += 1
            if len(it.item) < 1:
                self._delete_curves(it)
            it = nxt

    def test_integrity(self) -> bool:
        for it in self._iter():
            a, z = it.item[0], it.item[-1]
            for e in (a, z):
                if (self.is_end[e] and it.id != self.header[e]) or (
                    self.is_junc[e] and self.header[e] == -1
                ) or (not self.is_end[e] and not self.is_junc[e]):
                    return False
            for e in it.item[1:-1]:
                if self.is_end[e] or it.id != self.header[e]:
                    return False
            for e in it.item:
                if (self.is_end[e] and self.is_junc[e]) or (self.n_junc[e] == 0 and self.is_junc[e]) or (
                    self.n_junc[e] > 0 and not self.is_junc[e]
                ) or (self.header[e] == -1 and self.is_junc[e]):
                    return False
                if self.deleted[e]:
                    return False
            if len(it.item) == 1:
                return False
        return True

    def info(self) -> dict:
        n_pts = n_curves = n_junc = 0
        for it in self._iter():
            for e in it.item:
                n_pts += 1
                n_junc += int(self.is_junc[e])
            n_curves += 1
        return {"points": self.n, "connected": n_pts, "curves": n_curves, "junctions": n_junc}

    def curve_vertex_lists(self, include_junction: bool = True):
        """``saveSulcalCurves``: vertex ids per curve (junction points of other curves optional)."""
        out = []
        for it in self._iter():
            row = []
            for e in it.item:
                if self.is_junc[e] and self.header[e] != it.id and not include_junction:
                    continue
                row.append(int(self.vid[e]))
            if row:
                out.append(row)
        return out

    def all_curves(self):
        return [[int(self.vid[e]) for e in it.item] for it in self._iter()]

    def junction_labels(self):
        """Per-curve (vertex, label) data used by ``saveVTK`` (label 1, or nJunctions+1)."""
        lab = {}
        for it in self._iter():
            for e in it.item:
                lab[int(self.vid[e])] = 1 if not self.is_junc[e] else int(self.n_junc[e]) + 1
        return lab
