"""MeshLib curvature port: edge-based curvature tensor, principal curvatures, quadratic fit.

Port of ``MeshLib/src/Util/SurfaceUtil.cpp`` (``tensor``, ``smoothingTensor``,
``principalCurvature``, ``curvature`` double/float, ``curvature_quad`` double/float) with the
helpers they call in ``Util/Geom.cpp`` (``LinearAlgebra::eig3symmetric``/``echelonEigv``/``Ab``/
``gaussElim``, ``Coordinate::rotation``/``rotPoint``/``rotPointInv``).  See
``docs/ports/meshlib-geometry.md`` (section "Curvature").

Precision (mirrors MeshLib): vertices, face/vertex normals and edge tangents are float32 and the
float32 arithmetic is kept (so the *same* decisions are taken); edge lengths, the tensors and
everything downstream are float64.  The ``float`` overloads of ``curvature``/``curvature_quad``
compute in double and cast the outputs to float32 (as in the C++).

Naming quirk preserved from MeshLib: ``cmin`` is the principal value of *middle* magnitude among
the three eigenvalues and ``cmax`` the one of largest magnitude (then swapped so ``cmin <= cmax``);
``umin`` is the eigenvector belonging to ``cmax`` and ``umax`` the one belonging to ``cmin``
(``principalCurvature`` L719-L758).  They are kept as is.

Known upstream quirks, reproduced deliberately (see the port note):

* "flat" vertices (every |T_jk| < 1e-6) get ``cmin = cmax = 0``; MeshLib writes ``umin[i][0]`` /
  ``umax[i][0]`` three times, leaving components 1, 2 uninitialised.  Here they are 0.
* If the eigenvalue magnitudes tie, none of the selection branches fires and MeshLib leaves the
  output slot untouched (uninitialised); here the slot keeps 0.  Same for exactly tied diagonal
  tensors in ``eig3symmetric`` (strict ``<`` comparisons leave ``lambda`` unset).
* ``echelonEigv`` only eliminates the second row, so eigenvectors of non-degenerate tensors are
  whatever that routine returns; it is ported verbatim, not replaced by ``numpy.linalg.eigh``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from .geometry import NeighborLists, face_normals, vertex_neighbor_lists, vertex_normals

__all__ = [
    "Curvature",
    "curvature",
    "curvature_quad",
    "gaussian_curvature",
    "mean_curvature",
    "principal_curvature",
    "smoothing_tensor",
    "tensor",
]

F32 = np.float32
_EPS_FLAT = 1e-6
_EPS_ECH = float(np.float32(1e-8))  # `float eps = 1e-8;` in echelonEigv
_PI = 3.141592653589793238462643


class Curvature(NamedTuple):
    """Principal curvatures/directions per vertex (``umin``/``umax`` are (N, 3))."""

    cmin: np.ndarray
    cmax: np.ndarray
    umin: np.ndarray
    umax: np.ndarray


def _coerce(mesh, faces=None):
    if faces is None:
        if not hasattr(mesh, "faces"):
            raise TypeError("pass a Mesh, or (vertices, faces)")
        mesh, faces = mesh.vertices, mesh.faces
    v = np.ascontiguousarray(mesh, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("vertices must be (N, 3) and faces (M, 3)")
    if f.size and (f.min() < 0 or f.max() >= v.shape[0]):
        raise ValueError("face index out of range")
    return v, f


# --------------------------------------------------------------------------- #
# SurfaceUtil::tensor  (L522-L624)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _tensor(v, faces, indptr, indices, fnorm):
    n = v.shape[0]
    nf = faces.shape[0]
    # edge list: for i ascending, neighbours id (list order) not yet "checked" (id > i)
    check = np.zeros(n, dtype=np.bool_)
    cap = indices.shape[0]
    ev1 = np.empty(cap, dtype=np.int64)
    ev2 = np.empty(cap, dtype=np.int64)
    etan = np.empty((cap, 3), dtype=np.float32)
    elen = np.empty(cap, dtype=np.float64)
    m = 0
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            idv = indices[p]
            if not check[idv]:
                tx = v[idv, 0] - v[i, 0]
                ty = v[idv, 1] - v[i, 1]
                tz = v[idv, 2] - v[i, 2]
                s = np.float32(math.sqrt(tx * tx + ty * ty + tz * tz))  # Vector::norm (float)
                elen[m] = np.float64(s)
                if s > 0:
                    tx /= s
                    ty /= s
                    tz /= s
                a = i
                b = idv
                if a > b:  # never true (check[] only passes id > i); kept from the source
                    a, b = b, a
                ev1[m] = a
                ev2[m] = b
                etan[m, 0] = tx
                etan[m, 1] = ty
                etan[m, 2] = tz
                m += 1
        check[i] = True

    key = ev1[:m] * n + ev2[:m]
    order = np.argsort(key, kind="mergesort")  # std::sort; keys are unique on manifold meshes
    skey = key[order]
    fid1 = np.full(m, -1, dtype=np.int64)
    fid2 = np.full(m, -1, dtype=np.int64)
    for i in range(nf):
        for k in range(3):
            a = faces[i, k]
            b = faces[i, (k + 1) % 3]
            lo = min(a, b)
            hi = max(a, b)
            pos = np.searchsorted(skey, lo * n + hi)  # lower_bound
            if a < b:
                fid1[pos] = i
            else:
                fid2[pos] = i

    mean_len = 0.0
    for q in range(m):
        mean_len += elen[order[q]]
    mean_len /= m

    nn = np.empty(n, dtype=np.float64)
    for i in range(n):
        nn[i] = np.float64(indptr[i + 1] - indptr[i])
    Tv = np.zeros((n, 3, 3), dtype=np.float64)
    for q in range(m):
        e = order[q]
        f1 = fid1[q]
        f2 = fid2[q]
        if f1 == -1 or f2 == -1:
            continue
        a0, a1, a2 = fnorm[f1, 0], fnorm[f1, 1], fnorm[f1, 2]
        b0, b1, b2 = fnorm[f2, 0], fnorm[f2, 1], fnorm[f2, 2]
        beta = np.float64(a0 * b0 + a1 * b1 + a2 * b2)  # float dot -> double
        c0 = a1 * b2 - a2 * b1
        c1 = a2 * b0 - a0 * b2
        c2 = a0 * b1 - a1 * b0
        t0, t1, t2 = etan[e, 0], etan[e, 1], etan[e, 2]
        cpt = c0 * t0 + c1 * t1 + c2 * t2  # float
        if beta > 1:
            beta = 1.0
        elif beta < -1:
            beta = -1.0
        if cpt > 0:
            beta = math.acos(beta)
        else:
            beta = -math.acos(beta)
        blen = beta * elen[e] / mean_len
        tt = (t0, t1, t2)
        va = ev1[e]
        vb = ev2[e]
        for j in range(3):
            for k in range(j, 3):
                tjk = np.float64(tt[j] * tt[k]) * blen  # (float*float) then * double
                Tv[va, j, k] += tjk / nn[va]
                Tv[vb, j, k] += tjk / nn[vb]
    for i in range(n):
        Tv[i, 1, 0] = Tv[i, 0, 1]
        Tv[i, 2, 0] = Tv[i, 0, 2]
        Tv[i, 2, 1] = Tv[i, 1, 2]
    return Tv


# --------------------------------------------------------------------------- #
# SurfaceUtil::smoothingTensor  (L626-L694)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _smoothing_tensor(Tv, indptr, indices, iters):
    n = Tv.shape[0]
    cur = Tv.copy()
    new = np.zeros_like(cur)
    for _ in range(iters):
        for i in range(n):
            weight = 1.0
            for j in range(3):
                for k in range(j, 3):
                    new[i, j, k] = 0.0 + cur[i, j, k]
            for p in range(indptr[i], indptr[i + 1]):
                idv = indices[p]
                for j in range(3):
                    for k in range(j, 3):
                        new[i, j, k] += cur[idv, j, k] * 1.0
                weight += 1.0
            for j in range(3):
                for k in range(j, 3):
                    new[i, j, k] /= weight
            new[i, 1, 0] = new[i, 0, 1]
            new[i, 2, 0] = new[i, 0, 2]
            new[i, 2, 1] = new[i, 1, 2]
        cur, new = new, cur
    return cur


# --------------------------------------------------------------------------- #
# LinearAlgebra::eig3symmetric / echelonEigv  (Geom.cpp L1816-1924, L2049-2099)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _echelon_eigv(M, x):
    n = 3
    A = M.copy()
    # i in [0, n-2): only i = 0; eliminate rows j in [1, n-1) i.e. only row 1
    for i in range(n - 2):
        max_elem = abs(A[i, i])
        r = i
        for j in range(i + 1, n):
            ab = abs(A[j, i])
            if max_elem < ab:
                max_elem = ab
                r = j
        if r != i:
            for k in range(n):
                t = A[r, k]
                A[r, k] = A[i, k]
                A[i, k] = t
        for j in range(i + 1, n - 1):
            c = A[j, i] / A[i, i]
            A[j, i] = 0.0
            for k in range(i + 1, n):
                A[j, k] -= A[i, k] * c
    for k in range(n - 1, -1, -1):
        if abs(A[k, k]) > _EPS_ECH:
            x[k] = 1.0
            for i in range(k - 1, -1, -1):
                x[i] = 0.0
                for j in range(i + 1, n):
                    x[i] -= A[i, j] * x[j]
                if A[i, i] != 0:
                    x[i] /= A[i, i]
            break
        else:
            x[k] = 0.0


@njit(cache=True, error_model="numpy")
def _eig3(M, lam, ev):
    p1 = M[0, 1] * M[0, 1] + M[0, 2] * M[0, 2] + M[1, 2] * M[1, 2]
    if p1 == 0:
        a, b, c = M[0, 0], M[1, 1], M[2, 2]
        if a < b and a < c:
            lam[0] = a
            if b < c:
                lam[1] = b
                lam[2] = c
                perm = (0, 1, 2)
            else:
                lam[1] = c
                lam[2] = b
                perm = (0, 2, 1)
        elif b < c and b < a:
            lam[0] = b
            if a < c:
                lam[1] = a
                lam[2] = c
                perm = (1, 0, 2)
            else:
                lam[1] = c
                lam[2] = a
                perm = (1, 2, 0)
        elif c < a and c < b:
            lam[0] = c
            if a < b:
                lam[1] = a
                lam[2] = b
                perm = (2, 0, 1)
            else:
                lam[1] = b
                lam[2] = a
                perm = (2, 1, 0)
        else:  # ties: MeshLib leaves lambda/eigv unset
            return
        for r in range(3):
            for c2 in range(3):
                ev[r, c2] = 0.0
            ev[r, perm[r]] = 1.0
    else:
        q = (M[0, 0] + M[1, 1] + M[2, 2]) / 3
        p2 = (
            (M[0, 0] - q) * (M[0, 0] - q)
            + (M[1, 1] - q) * (M[1, 1] - q)
            + (M[2, 2] - q) * (M[2, 2] - q)
            + 2 * p1
        )
        p = math.sqrt(p2 / 6)
        A = np.empty((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                A[i, j] = (1 / p) * (M[i, j] - q * (1.0 if i == j else 0.0))
        det = (
            A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
        )
        r = det / 2
        if r <= -1:
            phi = _PI / 3
        elif r >= 1:
            phi = 0.0
        else:
            phi = math.acos(r) / 3
        lam[0] = q + 2 * p * math.cos(phi + (2 * _PI / 3))
        lam[2] = q + 2 * p * math.cos(phi)
        lam[1] = 3 * q - lam[0] - lam[2]
        B = np.empty((3, 3), dtype=np.float64)
        x = np.empty(3, dtype=np.float64)
        for nn in range(3):
            for i in range(3):
                for j in range(3):
                    B[i, j] = M[i, j] - lam[nn] * (1.0 if i == j else 0.0)
            _echelon_eigv(B, x)
            s = 0.0
            for i in range(3):
                s += x[i] * x[i]
            s = math.sqrt(s)
            for i in range(3):
                ev[nn, i] = x[i] / s


# --------------------------------------------------------------------------- #
# SurfaceUtil::principalCurvature  (L696-L761)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _principal(Tv):
    n = Tv.shape[0]
    cmin = np.zeros(n, dtype=np.float64)
    cmax = np.zeros(n, dtype=np.float64)
    umin = np.zeros((n, 3), dtype=np.float64)
    umax = np.zeros((n, 3), dtype=np.float64)
    lam = np.zeros(3, dtype=np.float64)
    ev = np.zeros((3, 3), dtype=np.float64)
    for i in range(n):
        T = Tv[i]
        flat = True
        for j in range(3):
            for k in range(3):
                if not abs(T[j, k]) < _EPS_FLAT:
                    flat = False
        if flat:
            continue  # cmin = cmax = 0; umin/umax = 0 (upstream leaves comps 1, 2 undefined)
        lam[:] = 0.0
        ev[:, :] = 0.0
        _eig3(T, lam, ev)
        a0, a1, a2 = abs(lam[0]), abs(lam[1]), abs(lam[2])
        cmn = 0.0
        cmx = 0.0
        un = np.zeros(3)  # umin
        ux = np.zeros(3)  # umax
        if (a0 > a1 and a0 < a2) or (a0 < a1 and a0 > a2):
            cmn = lam[0]
            ux = ev[0].copy()
        elif (a1 > a2 and a1 < a0) or (a1 < a2 and a1 > a0):
            cmn = lam[1]
            ux = ev[1].copy()
        elif (a2 > a0 and a2 < a1) or (a2 < a0 and a2 > a1):
            cmn = lam[2]
            ux = ev[2].copy()
        if a0 > a1 and a0 > a2:
            cmx = lam[0]
            un = ev[0].copy()
        elif a1 > a2 and a1 > a0:
            cmx = lam[1]
            un = ev[1].copy()
        elif a2 > a0 and a2 > a1:
            cmx = lam[2]
            un = ev[2].copy()
        if cmx < cmn:
            cmn, cmx = cmx, cmn
            un, ux = ux, un
        cmin[i] = cmn
        cmax[i] = cmx
        umin[i] = un
        umax[i] = ux
    return cmin, cmax, umin, umax


# --------------------------------------------------------------------------- #
# public tensor-path API
# --------------------------------------------------------------------------- #
def tensor(mesh, faces=None, *, neighbors: NeighborLists | None = None) -> np.ndarray:
    """``SurfaceUtil::tensor``: per-vertex edge-based curvature tensor, float64 (N, 3, 3).

    Every interior edge contributes ``t t^T * beta * len / mean_len`` (``t`` the float32 unit edge
    tangent, ``beta`` the signed dihedral angle between the two float32 face normals, positive
    when convex per the sign of ``(n1 x n2) . t``) split evenly over its two end vertices
    (divided by their neighbour-list length).  Boundary edges (one face) are skipped.
    """
    v, f = _coerce(mesh, faces)
    nb = neighbors if neighbors is not None else vertex_neighbor_lists(f, v.shape[0])
    fn = face_normals(v, f)
    with np.errstate(all="ignore"):
        return _tensor(v, f, nb.indptr, nb.indices, fn)


def smoothing_tensor(
    T: np.ndarray, mesh, faces=None, iterations: int = 1, *, neighbors: NeighborLists | None = None
) -> np.ndarray:
    """``SurfaceUtil::smoothingTensor``: ``iterations`` Jacobi passes of the unweighted
    1-ring mean *including the vertex itself* applied to the tensor field.  Returns a new array."""
    v, f = _coerce(mesh, faces)
    nb = neighbors if neighbors is not None else vertex_neighbor_lists(f, v.shape[0])
    T = np.ascontiguousarray(T, dtype=np.float64)
    if T.shape != (v.shape[0], 3, 3):
        raise ValueError("T must have shape (N, 3, 3)")
    return _smoothing_tensor(T, nb.indptr, nb.indices, int(iterations))


def principal_curvature(T: np.ndarray) -> Curvature:
    """``SurfaceUtil::principalCurvature``: float64 ``Curvature`` from a (N, 3, 3) tensor field."""
    T = np.ascontiguousarray(T, dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (3, 3):
        raise ValueError("T must have shape (N, 3, 3)")
    with np.errstate(all="ignore"):
        return Curvature(*_principal(T))


def _as_dtype(c: Curvature, dtype) -> Curvature:
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    if dt == np.float64:
        return c
    return Curvature(*(a.astype(np.float32) for a in c))


def curvature(
    mesh,
    faces=None,
    n_iter_smoothing_tensor: int = 0,
    *,
    dtype=np.float64,
    neighbors: NeighborLists | None = None,
) -> Curvature:
    """``SurfaceUtil::curvature(mesh, cmin, cmax, umin, umax, nIterSmoothingTensor)``.

    ``dtype=np.float64`` is the ``double*`` overload, ``np.float32`` the ``float*`` overload
    (computed in double, outputs cast to float32 as in ``SurfaceUtil.cpp`` L98-L130).
    """
    T = tensor(mesh, faces, neighbors=neighbors)
    if n_iter_smoothing_tensor > 0:
        T = smoothing_tensor(T, mesh, faces, n_iter_smoothing_tensor, neighbors=neighbors)
    return _as_dtype(principal_curvature(T), dtype)


def mean_curvature(c: Curvature) -> np.ndarray:
    """``(cmin + cmax) / 2`` (float64 arithmetic on the stored values)."""
    return (c.cmin.astype(np.float64) + c.cmax.astype(np.float64)) / 2


def gaussian_curvature(c: Curvature) -> np.ndarray:
    """``cmin * cmax`` (float64)."""
    return c.cmin.astype(np.float64) * c.cmax.astype(np.float64)


# --------------------------------------------------------------------------- #
# SurfaceUtil::curvature_quad  (L132-L248)
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _rotation(ax0, ax1, ax2, theta, mat):
    # Coordinate::rotation(const double *axis, double theta, double *mat)
    for i in range(9):
        mat[i] = 0.0
    mat[0] = 1.0
    mat[4] = 1.0
    mat[8] = 1.0
    s = math.sqrt(ax0 * ax0 + ax1 * ax1 + ax2 * ax2)
    if s == 0:
        return
    if s > 0:
        ax0 /= s
        ax1 /= s
        ax2 /= s
    A = np.array([0.0, -ax2, ax1, ax2, 0.0, -ax0, -ax1, ax0, 0.0])
    st = math.sin(theta)
    for i in range(9):
        mat[i] += A[i] * st
    for i in range(9):
        for j in range(3):
            mat[i] += A[j + (i // 3) * 3] * A[j * 3 + i % 3] * (1 - math.cos(theta))


@njit(cache=True, error_model="numpy")
def _gauss_elim(M, x):
    # LinearAlgebra::gaussElim: x holds the right-hand side and receives the solution
    n = M.shape[0]
    A = np.empty((n, n + 1), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = M[i, j]
        A[i, n] = x[i]
    for i in range(n - 1):
        max_elem = abs(A[i, i])
        r = i
        for j in range(i + 1, n):
            ab = abs(A[j, i])
            if max_elem < ab:
                max_elem = ab
                r = j
        if r != i:
            for k in range(n + 1):
                t = A[r, k]
                A[r, k] = A[i, k]
                A[i, k] = t
            tx = x[r]
            x[r] = x[i]
            x[i] = tx
        for j in range(i + 1, n):
            c = A[j, i] / A[i, i]
            A[j, i] = 0.0
            for k in range(i + 1, n + 1):
                A[j, k] -= A[i, k] * c
    for i in range(n - 1, -1, -1):
        x[i] = A[i, n]
        for j in range(i + 1, n):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]


@njit(cache=True, error_model="numpy")
def _quad(v, indptr, indices, vnorm):
    n = v.shape[0]
    cmin = np.zeros(n, dtype=np.float64)
    cmax = np.zeros(n, dtype=np.float64)
    umin = np.zeros((n, 3), dtype=np.float64)
    umax = np.zeros((n, 3), dtype=np.float64)
    mat = np.zeros(9, dtype=np.float64)
    for i in range(n):
        # 2-ring (including the vertex itself), sorted unique
        cnt = 0
        for p in range(indptr[i], indptr[i + 1]):
            cnt += 1 + (indptr[indices[p] + 1] - indptr[indices[p]])
        buf = np.empty(cnt, dtype=np.int64)
        c = 0
        for p in range(indptr[i], indptr[i + 1]):
            idv = indices[p]
            buf[c] = idv
            c += 1
            for q in range(indptr[idv], indptr[idv + 1]):
                buf[c] = indices[q]
                c += 1
        nb = np.unique(buf)  # sort + unique

        # rotation taking the vertex normal to the -x axis (Ax = (-1, 0, 0))
        b0, b1, b2 = vnorm[i, 0], vnorm[i, 1], vnorm[i, 2]
        zero = np.float32(0.0)
        neg1 = np.float32(-1.0)
        rx0 = b1 * zero - b2 * zero  # B.cross(Ax), float32
        rx1 = b2 * neg1 - b0 * zero
        rx2 = b0 * zero - b1 * neg1
        theta = math.acos(np.float64(b0) * -1.0)
        _rotation(np.float64(rx0), np.float64(rx1), np.float64(rx2), theta, mat)

        m = nb.shape[0]
        M = np.empty((m, 6), dtype=np.float64)
        b = np.empty(m, dtype=np.float64)
        for j in range(m):
            k = nb[j]
            p0 = np.float64(v[k, 0])
            p1 = np.float64(v[k, 1])
            p2 = np.float64(v[k, 2])
            y1 = mat[3] * p0 + mat[4] * p1 + mat[5] * p2  # rotPoint rows 1, 2, 0
            y2 = mat[6] * p0 + mat[7] * p1 + mat[8] * p2
            x0 = mat[0] * p0 + mat[1] * p1 + mat[2] * p2
            M[j, 0] = y1 * y1
            M[j, 1] = y2 * y2
            M[j, 2] = y1 * y2
            M[j, 3] = y1
            M[j, 4] = y2
            M[j, 5] = 1.0
            b[j] = x0
        # LinearAlgebra::Ab: normal equations + gaussElim
        ATA = np.empty((6, 6), dtype=np.float64)
        x = np.empty(6, dtype=np.float64)
        for a in range(6):
            for bb in range(a, 6):
                s = 0.0
                for k in range(m):
                    s += M[k, a] * M[k, bb]
                ATA[a, bb] = s
                ATA[bb, a] = s
            s = 0.0
            for k in range(m):
                s += M[k, a] * b[k]
            x[a] = s
        _gauss_elim(ATA, x)

        dxx = 2 * x[0]
        dxy = x[2]
        dyy = 2 * x[1]
        tmp = math.sqrt((dxx - dyy) * (dxx - dyy) + 4 * dxy * dxy)
        v2x = 2 * dxy
        v2y = dyy - dxx + tmp
        nrm = math.sqrt(v2x * v2x + v2y * v2y)
        v2x /= nrm
        v2y /= nrm
        v1x = -v2y
        v1y = v2x
        mu1 = (dxx + dyy + tmp) / 2
        mu2 = (dxx + dyy - tmp) / 2
        if abs(mu1) < abs(mu2):
            cmin[i] = mu1
            cmax[i] = mu2
            u1y, u1z, u2y, u2z = v1x, v1y, v2x, v2y
        else:
            cmin[i] = mu2
            cmax[i] = mu1
            u1y, u1z, u2y, u2z = v2x, v2y, v1x, v1y
        # rotPointInv with u = (0, y, z)
        for r in range(3):
            umin[i, r] = mat[r + 3] * u1y + mat[r + 6] * u1z
            umax[i, r] = mat[r + 3] * u2y + mat[r + 6] * u2z
    return cmin, cmax, umin, umax


def curvature_quad(
    mesh, faces=None, *, dtype=np.float64, neighbors: NeighborLists | None = None
) -> Curvature:
    """``SurfaceUtil::curvature_quad``: curvature from a least-squares quadratic fit to the
    2-ring of each vertex, in a frame whose x axis is the (float32, angle-weighted) vertex normal.

    ``dtype=np.float32`` is the ``float*`` overload (computed in double, then cast).
    Hessian eigenvalues are ``mu = (Dxx + Dyy +- tmp) / 2``; ``cmin`` is the one of smaller
    magnitude (NOT the algebraically smaller one, unlike ``curvature``).
    """
    v, f = _coerce(mesh, faces)
    nb = neighbors if neighbors is not None else vertex_neighbor_lists(f, v.shape[0])
    vn = vertex_normals(v, f)
    with np.errstate(all="ignore"):
        out = Curvature(*_quad(v, nb.indptr, nb.indices, vn))
    return _as_dtype(out, dtype)
