"""Gradient and Gauss-Newton Hessian of the HSD energy (``updateGradientProperties`` /
``updateGradientDisplacement`` / ``updateHessian`` / ``updateNewGradient``, HSD.cpp 2022-2615).

The C++ builds the full Jacobian ``m_gradient_raw`` (samples x 3*size) and reduces it with BLAS
(``ATB`` with a ones vector for the gradient, ``ATDA`` for ``J^T diag(D) J``).  Here the Jacobian
rows are produced by a Numba kernel in chunks of samples (bounded memory) and reduced with NumPy
(float64 BLAS); the per-row arithmetic is the C++ float32/float64 mix, the reduction order is not
(documented in ``docs/ports/hsd.md``: ~1e-13 relative).
"""

from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, njit, prange
from scipy.linalg import lapack

from .geom import D, F, _rot_point, _sph2cart, _unit3, bary32_snap, dot3, norm3, rotation_into
from .geom import _cross3 as cross3

PAR = {"cache": True, "error_model": "numpy", "parallel": True}
SEQ = {"cache": True, "error_model": "numpy"}

CHUNK = 16384


@njit(cache=True, error_model="numpy", inline="always")
def _z_dot(d1, d2, zx, zy, zz, u1, u2, m, A):
    """z_dot = rotation of z_hat by |delta| about z_hat x unit(u1 d1 + u2 d2) (float32)."""
    f1 = F(d1)
    f2 = F(d2)
    ox = u1[0] * f1 + u2[0] * f2
    oy = u1[1] * f1 + u2[1] * f2
    oz = u1[2] * f1 + u2[2] * f2
    ox, oy, oz = _unit3(ox, oy, oz)
    degree = F(math.sqrt(d1 * d1 + d2 * d2))
    cx, cy, cz = cross3(zx, zy, zz, ox, oy, oz)
    cx, cy, cz = _unit3(cx, cy, cz)
    rotation_into(m, A, cx, cy, cz, degree)
    return _rot_point(m, zx, zy, zz)


@njit(cache=True, error_model="numpy", inline="always")
def _dxd(zdx, zdy, zdz, u1, u2, ax, ay, az, gx, gy, gz):
    """The three ``(M * a) . g`` products: M = [z_dot]x, [u2]x, [-u1]x (rows as in HSD.cpp)."""
    # [z_dot]x = {{0,-z2,z1},{z2,0,-z0},{-z1,z0,0}}
    zd0, zd1, zd2 = D(zdx), D(zdy), D(zdz)
    dxdg = (-zd2 * ay + zd1 * az) * gx + (zd2 * ax + -zd0 * az) * gy + (-zd1 * ax + zd0 * ay) * gz
    # [z_hatXu1]x with z_hatXu1 = u2
    b0, b1, b2 = D(u2[0]), D(u2[1]), D(u2[2])
    dxdu1 = (-b2 * ay + b1 * az) * gx + (b2 * ax + -b0 * az) * gy + (-b1 * ax + b0 * ay) * gz
    # [z_hatXu2]x with z_hatXu2 = -u1
    c0, c1, c2 = D(-u1[0]), D(-u1[1]), D(-u1[2])
    dxdu2 = (-c2 * ay + c1 * az) * gx + (c2 * ax + -c0 * az) * gy + (-c1 * ax + c0 * ay) * gz
    return dxdg, dxdu1, dxdu2


# --------------------------------------------------------------------------- #
# properties
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _prop_range(
    lo, hi, base, v, faces, indptr, indices, Y, samples, tree_cache, prop, mean, variance,
    feat, coeff, ncoeff, axis0, tan1, tan2, n0, n, fine, unit_var, normalization, J2, J1, diag,
):  # fmt: skip
    size = n - n0
    m = np.empty(9, np.float32)
    A = np.empty(9, np.float32)
    nid = np.empty(2, np.int64)
    for i in range(lo, hi):
        r = i - base
        yx, yy, yz = samples[i, 0], samples[i, 1], samples[i, 2]
        m_bar = mean[i]
        fid = tree_cache[i]
        a, b, c = faces[fid, 0], faces[fid, 1], faces[fid, 2]
        # N = unit((v2 - v1) x (v3 - v2)); Yproj = y * ((v1.N) / (y.N))
        nx, ny, nz = cross3(
            v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2],
            v[c, 0] - v[b, 0], v[c, 1] - v[b, 1], v[c, 2] - v[b, 2],
        )  # fmt: skip
        nx, ny, nz = _unit3(nx, ny, nz)
        sc = dot3(v[a, 0], v[a, 1], v[a, 2], nx, ny, nz) / dot3(yx, yy, yz, nx, ny, nz)
        px, py, pz = yx * sc, yy * sc, yz * sc
        cy0, cy1, cy2 = bary32_snap(
            v[a, 0], v[a, 1], v[a, 2], v[b, 0], v[b, 1], v[b, 2], v[c, 0], v[c, 1], v[c, 2],
            px, py, pz, F(1e-5),
        )  # fmt: skip
        id1, id2, id3 = a, b, c
        # delta from the barycentric blend of the basis rows of the three original vertices
        d0 = 0.0
        d1 = 0.0
        d2 = 0.0
        for j in range(ncoeff):
            y = Y[id1, j] * cy0 + Y[id2, j] * cy1 + Y[id3, j] * cy2
            d0 += y * coeff[0, j]
            d1 += y * coeff[1, j]
            d2 += y * coeff[2, j]
        mm = D(feat[i])
        zdx, zdy, zdz = _z_dot(d1, d2, axis0[0], axis0[1], axis0[2], tan1, tan2, m, A)
        # grad_m
        g0 = 0.0
        g1 = 0.0
        g2 = 0.0
        nid[0] = id2
        nid[1] = id3
        use_ring = False
        nnb = 2
        r1 = D(dot3(v[a, 0], v[a, 1], v[a, 2], nx, ny, nz) / dot3(yx, yy, yz, nx, ny, nz))
        if cy0 == F(1) or cy1 == F(1) or cy2 == F(1):
            if cy0 == F(1):
                id1 = a
            elif cy1 == F(1):
                id1 = b
            else:
                id1 = c
            use_ring = True
            nnb = indptr[id1 + 1] - indptr[id1]
            r1 = 1.0
        area = F(0)
        for j in range(nnb):
            if use_ring:
                id2 = indices[indptr[id1] + j]
                id3 = indices[indptr[id1] + (j + 1) % nnb]
            else:
                id2 = nid[j]
                id3 = nid[(j + 1) % nnb]
            e1x, e1y, e1z = v[id2, 0] - v[id1, 0], v[id2, 1] - v[id1, 1], v[id2, 2] - v[id1, 2]
            e2x, e2y, e2z = v[id3, 0] - v[id2, 0], v[id3, 1] - v[id2, 1], v[id3, 2] - v[id2, 2]
            Nx, Ny, Nz = cross3(e1x, e1y, e1z, e2x, e2y, e2z)
            area += norm3(Nx, Ny, Nz)
            Nx, Ny, Nz = _unit3(Nx, Ny, Nz)
            r2 = r1 / D(dot3(yx, yy, yz, Nx, Ny, Nz))
            ry0, ry1, ry2 = r2 * D(yx), r2 * D(yy), r2 * D(yz)
            p00 = r1 - ry0 * D(Nx)
            p01 = -ry0 * D(Ny)
            p02 = -ry0 * D(Nz)
            p10 = -ry1 * D(Nx)
            p11 = r1 - ry1 * D(Ny)
            p12 = -ry1 * D(Nz)
            p20 = -ry2 * D(Nx)
            p21 = -ry2 * D(Ny)
            p22 = r1 - ry2 * D(Nz)
            m1 = prop[id1]
            m2 = prop[id2]
            m3 = prop[id3]
            # DA1DP = unit(e23 x N) * |e23| * m1, e23 = v3 - v2 (v1,v2,v3 here are id1,id2,id3)
            a1x, a1y, a1z = e2x, e2y, e2z
            c1x, c1y, c1z = cross3(a1x, a1y, a1z, Nx, Ny, Nz)
            c1x, c1y, c1z = _unit3(c1x, c1y, c1z)
            l1 = norm3(a1x, a1y, a1z)
            t1x, t1y, t1z = c1x * l1 * m1, c1y * l1 * m1, c1z * l1 * m1
            # DA2DP = unit(e31 x N) * |e31| * m2, e31 = v1 - v3
            a2x, a2y, a2z = v[id1, 0] - v[id3, 0], v[id1, 1] - v[id3, 1], v[id1, 2] - v[id3, 2]
            c2x, c2y, c2z = cross3(a2x, a2y, a2z, Nx, Ny, Nz)
            c2x, c2y, c2z = _unit3(c2x, c2y, c2z)
            l2 = norm3(a2x, a2y, a2z)
            t2x, t2y, t2z = c2x * l2 * m2, c2y * l2 * m2, c2z * l2 * m2
            # DA3DP = unit(e12 x N) * |e12| * m3, e12 = v2 - v1
            a3x, a3y, a3z = e1x, e1y, e1z
            c3x, c3y, c3z = cross3(a3x, a3y, a3z, Nx, Ny, Nz)
            c3x, c3y, c3z = _unit3(c3x, c3y, c3z)
            l3 = norm3(a3x, a3y, a3z)
            t3x, t3y, t3z = c3x * l3 * m3, c3y * l3 * m3, c3z * l3 * m3
            dmx = (t1x + t2x) + t3x
            dmy = (t1y + t2y) + t3y
            dmz = (t1z + t2z) + t3z
            g0 += p00 * D(dmx) + p01 * D(dmy) + p02 * D(dmz)
            g1 += p10 * D(dmx) + p11 * D(dmy) + p12 * D(dmz)
            g2 += p20 * D(dmx) + p21 * D(dmy) + p22 * D(dmz)
            if nnb == 2:
                break
        g0 /= D(area)
        g1 /= D(area)
        g2 /= D(area)
        dxdg, dxdu1, dxdu2 = _dxd(zdx, zdy, zdz, tan1, tan2, D(yx), D(yy), D(yz), g0, g1, g2)
        dEdx = 2.0 * (mm - m_bar)
        var = 1.0 if unit_var else variance[i]
        if fine:
            diag[r] = normalization / var
        # basis row of the sample: barycentric blend of the three original vertices
        a_, b_, c_ = a, b, c
        for j in range(n0, n):
            yy_ = Y[a_, j] * cy0 + Y[b_, j] * cy1 + Y[c_, j] * cy2
            J2[r, j - n0] = yy_ * dxdg * dEdx / var
            J2[r, size + j - n0] = yy_ * dxdu1 * dEdx / var
            J2[r, 2 * size + j - n0] = yy_ * dxdu2 * dEdx / var
            if fine:
                J1[r, j - n0] = yy_ * dxdg
                J1[r, size + j - n0] = yy_ * dxdu1
                J1[r, 2 * size + j - n0] = yy_ * dxdu2


@njit(**PAR)
def _prop_rows(
    lo, hi, v, faces, indptr, indices, Y, samples, tree_cache, prop, mean, variance, feat,
    coeff, ncoeff, axis0, tan1, tan2, n0, n, fine, unit_var, normalization, J2, J1, diag, nchunk,
):  # fmt: skip
    cnt = hi - lo
    for c in prange(nchunk):
        a = lo + c * cnt // nchunk
        b = lo + (c + 1) * cnt // nchunk
        _prop_range(
            a, b, lo, v, faces, indptr, indices, Y, samples, tree_cache, prop, mean, variance,
            feat, coeff, ncoeff, axis0, tan1, tan2, n0, n, fine, unit_var, normalization,
            J2, J1, diag,
        )  # fmt: skip


def accumulate_properties(
    subj, k, deg_beg, deg_end, *, samples, mean_k, var_k, feat_k, normalization, fine,
    unit_var, axis0, ring,
):  # fmt: skip
    """``updateGradientProperties`` for one subject and one property ``k``.

    Returns ``(grad (3*size,), H (3*size, 3*size) or None)``; the caller adds ``grad * normalization``
    to the subject's gradient and ``H`` to its Hessian.
    """
    n = (deg_end + 1) ** 2
    n0 = deg_beg**2
    size = n - n0
    ns = samples.shape[0]
    grad = np.zeros(3 * size)
    H = np.zeros((3 * size, 3 * size)) if fine else None
    nthr = get_num_threads()
    for lo in range(0, ns, CHUNK):
        hi = min(lo + CHUNK, ns)
        J2 = np.empty((hi - lo, 3 * size))
        J1 = np.empty((hi - lo, 3 * size)) if fine else np.empty((1, 1))
        diag = np.empty(hi - lo) if fine else np.empty(1)
        _prop_rows(
            lo, hi, subj.v, subj.faces, ring[0], ring[1], subj.Y, samples, subj.tree_cache,
            subj.prop[k], mean_k, var_k, feat_k, subj.coeff, subj.coeff.shape[1], axis0, subj.tan1,
            subj.tan2, n0, n, fine, unit_var, normalization, J2, J1, diag, max(1, min(hi - lo, nthr * 4)),
        )  # fmt: skip
        grad += J2.sum(axis=0)
        if fine:
            H += J1.T @ (diag[:, None] * J1)
    return grad, H


# --------------------------------------------------------------------------- #
# displacement regulariser
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy")
def _disp_range(
    lo, hi, base, v, p0, Y, coeff, ncoeff, axis0, tan1, tan2, n0, n, fine, normalization, J2, J1, diag
):  # fmt: skip
    size = n - n0
    m = np.empty(9, np.float32)
    A = np.empty(9, np.float32)
    nv = v.shape[0]
    for i in range(lo, hi):
        r = i - base
        d0 = 0.0
        d1 = 0.0
        d2 = 0.0
        for j in range(ncoeff):
            y = Y[i, j]
            d0 += y * coeff[0, j]
            d1 += y * coeff[1, j]
            d2 += y * coeff[2, j]
        zdx, zdy, zdz = _z_dot(d1, d2, axis0[0], axis0[1], axis0[2], tan1, tan2, m, A)
        xbx, xby, xbz = p0[i, 0], p0[i, 1], p0[i, 2]
        xhx, xhy, xhz = v[i, 0], v[i, 1], v[i, 2]
        xdx = D(dot3(xhx, xhy, xhz, xbx, xby, xbz))
        dxdg = 0.0
        dxdu1 = 0.0
        dxdu2 = 0.0
        dEdx = 0.0
        diag[r] = 0.0
        if abs(xdx) < 0.999999:
            dxdg, dxdu1, dxdu2 = _dxd(
                zdx, zdy, zdz, tan1, tan2, D(xhx), D(xhy), D(xhz), D(xbx), D(xby), D(xbz)
            )
            omx = 1.0 - xdx * xdx
            dEdx = 2.0 * math.acos(xdx)
            drdx = -1.0 / math.sqrt(omx)
            dxdg *= drdx
            dxdu1 *= drdx
            dxdu2 *= drdx
            if fine:
                diag[r] = normalization / nv
        for j in range(n0, n):
            J2[r, j - n0] = Y[i, j] * dxdg * dEdx
            J2[r, size + j - n0] = Y[i, j] * dxdu1 * dEdx
            J2[r, 2 * size + j - n0] = Y[i, j] * dxdu2 * dEdx
            if fine:
                J1[r, j - n0] = Y[i, j] * dxdg
                J1[r, size + j - n0] = Y[i, j] * dxdu1
                J1[r, 2 * size + j - n0] = Y[i, j] * dxdu2


@njit(**PAR)
def _disp_rows(
    lo, hi, v, p0, Y, coeff, ncoeff, axis0, tan1, tan2, n0, n, fine, normalization, J2, J1, diag, nchunk
):  # fmt: skip
    cnt = hi - lo
    for c in prange(nchunk):
        a = lo + c * cnt // nchunk
        b = lo + (c + 1) * cnt // nchunk
        _disp_range(
            a, b, lo, v, p0, Y, coeff, ncoeff, axis0, tan1, tan2, n0, n, fine, normalization,
            J2, J1, diag,
        )  # fmt: skip


def accumulate_displacement(subj, deg_beg, deg_end, *, normalization, fine, axis0):
    """``updateGradientDisplacement`` for one subject: ``(grad, H or None)`` (caller scales grad)."""
    n = (deg_end + 1) ** 2
    n0 = deg_beg**2
    size = n - n0
    nv = subj.v.shape[0]
    grad = np.zeros(3 * size)
    H = np.zeros((3 * size, 3 * size)) if fine else None
    nthr = get_num_threads()
    for lo in range(0, nv, CHUNK):
        hi = min(lo + CHUNK, nv)
        J2 = np.empty((hi - lo, 3 * size))
        J1 = np.empty((hi - lo, 3 * size)) if fine else np.empty((1, 1))
        diag = np.empty(hi - lo)
        _disp_rows(
            lo, hi, subj.v, subj.p0, subj.Y, subj.coeff, subj.coeff.shape[1], axis0, subj.tan1,
            subj.tan2, n0, n, fine, normalization, J2, J1, diag, max(1, min(hi - lo, nthr * 4)),
        )  # fmt: skip
        grad += J2.sum(axis=0)
        if fine:
            H += J1.T @ (diag[:, None] * J1)
    return grad, H


# --------------------------------------------------------------------------- #
# damped Newton direction (updateNewGradient)
# --------------------------------------------------------------------------- #
def solve_direction(H, G, lam, alpha):
    """``updateNewGradient``: damp the diagonal and solve ``H x = G`` with LAPACK ``dsysv`` ('L').

    ``alpha != 0``: ``H_ii += H_ii * lam + alpha``; else ``H_ii *= 1 + lam``.  ``H`` is modified
    (copy).  Only the lower triangle is read, as in the C++ call.
    """
    M = np.array(H, dtype=np.float64, copy=True)
    idx = np.arange(M.shape[0])
    if alpha != 0:
        M[idx, idx] += M[idx, idx] * lam + alpha
    else:
        M[idx, idx] *= 1 + lam
    _, _, x, _ = lapack.dsysv(M, np.asarray(G, dtype=np.float64), lower=1)
    return x
