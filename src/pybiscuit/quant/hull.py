"""Cerebral outer hull: port of ``OuterHull.m`` (LocalGyrificationIndex ``matlab/``, ``b9fa760``).

``OuterHull.m`` (+ ``VOXELISE/`` and ``ICP_finite/``), compiled with MATLAB R2012a (MCR 7.17) in the
BISCUIT image, builds a smooth hull enveloping a cortical surface:

1. voxelise the closed surface, longest side 256 voxels (``VOXELISE`` ray casting,
   three directions, majority vote), pad by 20 voxels and fill holes (26-connectivity);
2. Gaussian smoothing (``smooth3``, size 9, sigma 2), grayscale morphological closing with
   ``strel('ball', 15, 15)``, dilation with ``strel('ball', 1, 1)``, threshold at 25;
3. iso-surface at level 1, coordinates mapped back to millimetres (``*1.02`` as in the script);
4. keep one connected component;
5. rigid ICP (``ICP_finite``, optimiser ``fminlbfgs``) of the hull onto the input vertices.

Known deviations (``docs/ports/outer-hull.md``): MATLAB library routines have no source here, so
``strel('ball')`` is applied as the exact nonflat ellipsoid (MATLAB's default N=8 line
decomposition approximates it), 2-D per z-slice (MATLAB pads a 2-D neighbourhood with singleton
dimensions).  The iso-surface is classic marching cubes (Lorensen & Cline table), as MATLAB's
``isosurface``; vertex and face order may still differ from MATLAB's.  No oracle exists (no MATLAB;
Octave is unavailable), so this row is provisional; the image's LGI maps are the indirect evidence.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from ..io import read_vtk, write_vtk
from ..mesh import Mesh
from . import _mc_tables
from ._voxelise import voxelise

__all__ = ["outer_hull", "icp_finite", "fminlbfgs", "write_hull"]

GRID = 256  # longest side in voxels (OuterHull.m)
PAD = 20
SMOOTH_SD = 2.0
CLOSING_R, CLOSING_H = 15, 15
DILATE_R, DILATE_H = 1, 1
THRESHOLD = 25.0
ISO = 1.0
MARGIN = 1.02


# --------------------------------------------------------------------------- #
# morphology
# --------------------------------------------------------------------------- #
def _ball(r: int, h: float):
    """``strel('ball', r, h, 0)``: offsets and heights of the nonflat ellipsoid.

    The neighbourhood excludes the rim (``x^2 + y^2 < r^2``; the centre is always kept), so
    ``strel('ball', 1, 1)`` is the single centre pixel and its dilation only adds the height.
    That is what the three open examples show (brief 22): with the rim included, the sulcal-depth
    hull is one voxel too large (mean bias +0.71 mm against the image's ``gray.SD``), without it
    the bias is -0.04 mm.  Inferred from image output, not from MATLAB source.
    """
    ax = np.arange(-r, r + 1)
    dx, dy = np.meshgrid(ax, ax, indexing="ij")
    rr = dx**2 + dy**2
    keep = (rr < r * r) | (rr == 0)
    hts = h * np.sqrt(1.0 - rr[keep] / float(r * r))
    return dx[keep].astype(np.int64), dy[keep].astype(np.int64), hts


@njit(cache=True, error_model="numpy", parallel=True)
def _dilate(A, mask, dx, dy, ht, base):
    nx, ny, nz = A.shape
    out = np.full(A.shape, base)
    for k in prange(nz):
        for i in range(nx):
            for j in range(ny):
                if not mask[i, j, k]:
                    continue
                best = -np.inf
                for q in range(dx.shape[0]):
                    ii = i - dx[q]
                    jj = j - dy[q]
                    if ii < 0 or jj < 0 or ii >= nx or jj >= ny:
                        continue
                    v = A[ii, jj, k] + ht[q]
                    if v > best:
                        best = v
                out[i, j, k] = best
    return out


@njit(cache=True, error_model="numpy", parallel=True)
def _erode(A, mask, dx, dy, ht):
    nx, ny, nz = A.shape
    out = np.zeros(A.shape)
    for k in prange(nz):
        for i in range(nx):
            for j in range(ny):
                if not mask[i, j, k]:
                    continue
                best = np.inf
                for q in range(dx.shape[0]):
                    ii = i + dx[q]
                    jj = j + dy[q]
                    if ii < 0 or jj < 0 or ii >= nx or jj >= ny:
                        continue
                    v = A[ii, jj, k] - ht[q]
                    if v < best:
                        best = v
                out[i, j, k] = best
    return out


def _window(mask, r):
    return ndimage.maximum_filter(mask, footprint=np.ones((2 * r + 1, 2 * r + 1, 1), bool))


def _close_ball(A, r, h):
    """``imclose(A, strel('ball', r, h))`` slice-wise along the last axis (A >= 0)."""
    dx, dy, ht = _ball(r, h)
    hmax = float(ht.max())
    d = _dilate(A, _window(A > 0, r), dx, dy, ht, hmax)  # all-zero windows: 0 + h(0)
    return _erode(d, _window(d != hmax, r), dx, dy, ht)


def _dilate_ball(A, r, h):
    dx, dy, ht = _ball(r, h)
    hmax = float(ht.max())
    return _dilate(A, _window(A > 0, r), dx, dy, ht, hmax)


def _smooth3_gaussian(A, size, sigma):
    """``smooth3(A, 'gaussian', size, sigma)`` (separable form of the 3-D kernel)."""
    half = (size - 1) // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    h1 = np.exp(-(x * x) / 2 / sigma**2)
    h1 /= h1.sum()
    out = A
    for ax in range(3):
        out = ndimage.convolve1d(out, h1, axis=ax, mode="nearest")
    return out


# --------------------------------------------------------------------------- #
# iso-surface: marching cubes (MATLAB ``isosurface``; Lorensen & Cline table)
# --------------------------------------------------------------------------- #
_CORNER = np.array(_mc_tables.CORNERS, dtype=np.int64)
_EDGE = np.array(_mc_tables.EDGES, dtype=np.int64)
_TRI = np.full((256, 16), -1, dtype=np.int64)
for _i, _row in enumerate(_mc_tables.TRIANGLES):
    _TRI[_i, : len(_row)] = _row


@njit(cache=True, error_model="numpy")
def _mc(vol, iso, cubes, tri_table, edge, corner):
    ni, nj, nk = vol.shape
    L = ni * nj * nk
    tri = np.empty((cubes.shape[0] * 5, 3), dtype=np.int64)
    nt = 0
    lin = np.empty(8, dtype=np.int64)
    for c in range(cubes.shape[0]):
        i0, j0, k0 = cubes[c, 0], cubes[c, 1], cubes[c, 2]
        case = 0
        for q in range(8):
            ii = i0 + corner[q, 0]
            jj = j0 + corner[q, 1]
            kk = k0 + corner[q, 2]
            lin[q] = (ii * nj + jj) * nk + kk
            if vol[ii, jj, kk] < iso:
                case |= 1 << q
        m = 0
        while m < 16 and tri_table[case, m] >= 0:
            # table winding: normals point from the >= iso side to the < iso side
            for r in range(3):
                e = tri_table[case, m + r]
                la, lb = lin[edge[e, 0]], lin[edge[e, 1]]
                if la > lb:
                    la, lb = lb, la
                tri[nt, r] = la * L + lb
            nt += 1
            m += 3
    return tri[:nt]


def isosurface(vol, iso=ISO):
    """Marching-cubes iso-surface of ``vol`` (axis0, axis1, axis2 voxel coordinates, 0-based).

    Returns ``(vertices float64 (V,3), faces int32 (F,3))``, faces oriented from the ``>= iso``
    side to the ``< iso`` side, vertices shared between triangles (one per cut voxel edge).
    """
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    inside = (vol >= iso).astype(np.uint8)
    s = np.zeros(tuple(n - 1 for n in vol.shape), dtype=np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                s += inside[
                    dx : vol.shape[0] - 1 + dx,
                    dy : vol.shape[1] - 1 + dy,
                    dz : vol.shape[2] - 1 + dz,
                ]
    cubes = np.argwhere((s > 0) & (s < 8)).astype(np.int64)
    tri = _mc(vol, iso, cubes, _TRI, _EDGE, _CORNER)
    keys, inv = np.unique(tri.ravel(), return_inverse=True)
    L = vol.size
    a, b = keys // L, keys % L
    nj, nk = vol.shape[1], vol.shape[2]
    pa = np.stack([a // (nj * nk), (a // nk) % nj, a % nk], axis=1).astype(np.float64)
    pb = np.stack([b // (nj * nk), (b // nk) % nj, b % nk], axis=1).astype(np.float64)
    va, vb = vol.ravel()[a], vol.ravel()[b]
    t = ((iso - va) / (vb - va))[:, None]
    return pa + t * (pb - pa), inv.reshape(-1, 3).astype(np.int32)


# --------------------------------------------------------------------------- #
# ICP_finite / fminlbfgs
# --------------------------------------------------------------------------- #
def _rotation(r):
    r = np.asarray(r) * (math.pi / 180)
    cx, sx, cy, sy, cz, sz = (
        math.cos(r[0]),
        math.sin(r[0]),
        math.cos(r[1]),
        math.sin(r[1]),
        math.cos(r[2]),
        math.sin(r[2]),
    )
    rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
    ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
    rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return rx @ ry @ rz


def _matrix(par, scale):
    p = np.asarray(par) * scale
    tra = np.eye(4)
    tra[:3, 3] = p[:3]
    return tra @ _rotation(p[3:6])  # size = eye, shear = eye for the rigid case


def _move(M, P):
    return P @ M[:3, :3].T + M[:3, 3]


class _Opt:
    rho, sigma, tau1, tau2, tau3 = 0.01, 0.9, 3.0, 0.1, 0.5
    tol_fun, max_iter = 1e-6, 400
    diff_max, diff_min = 1e-1, 1e-8


class _State:
    pass


def fminlbfgs(fun, x0, tol_x: float):
    """Port of ``fminlbfgs.m`` (D. Kroon) as ``ICP_finite`` calls it: BFGS (6 variables), forward
    differences (``GradObj='off'``, ``GradConstr=true``), exact-goals line search.
    Returns ``(x, f)``."""
    opt = _Opt()
    st = _State()
    n = x0.size
    st.max_fun = 100 * n - 1
    st.func_count = 0
    st.x = np.array(x0, dtype=np.float64).ravel()
    st.init_step = 1.0

    def f_only(x):
        st.func_count += 1
        return float(fun(x))

    def f_grad(x):
        f = float(fun(x))
        gstep = min(max(st.init_step / 1e6, opt.diff_min), opt.diff_max)
        g = np.zeros(n)
        for i in range(n):
            xt = x.copy()
            xt[i] += gstep
            st.func_count += 1
            g[i] = (float(fun(xt)) - f) / gstep
        return f, g

    f0, g0 = f_grad(st.x)
    st.f, st.g = f0, g0
    st.dir = -g0.copy()
    st.g_old, st.x_old, st.f_old = g0.copy(), st.x.copy(), None
    st.fprime0 = float(g0 @ st.dir)
    st.init_step = min(1.0 / np.max(np.abs(g0)), 5.0) if np.max(np.abs(g0)) > 0 else 5.0
    H = np.eye(n)
    exitflag = None
    iteration = 0
    while True:
        iteration += 1
        tolfun_lns = np.spacing(max(1.0, abs(st.f)))
        fmin = st.f - 1e16 * (1 + abs(st.f))
        res = _linesearch(f_only, f_grad, st, opt, fmin, tolfun_lns)
        if res is not None:  # line search failed: restore previous point
            exitflag = res
            st.x, st.f, st.g = st.x_old, st.f_old if st.f_old is not None else st.f, st.g_old
            break
        st.x = st.x + st.alpha * st.dir
        st.f = st.f_alpha
        st.g = st.grad
        st.init_step = 1.0
        if np.max(np.abs(st.g)) < opt.tol_fun:
            exitflag = 1
        if np.max(np.abs(st.x_old - st.x)) < tol_x:
            exitflag = 2
        if iteration >= opt.max_iter:
            exitflag = 0
        if exitflag is not None:
            break
        dX = st.alpha * st.dir
        dG = st.g - st.g_old
        if dX @ dG >= math.sqrt(np.finfo(float).eps) * max(
            np.finfo(float).eps, np.linalg.norm(dX) * np.linalg.norm(dG)
        ):
            pk = 1.0 / (dG @ dX)
            Vk = np.eye(n) - pk * np.outer(dG, dX)
            H = Vk.T @ H @ Vk + pk * np.outer(dX, dX)
            st.dir = -H @ st.g
        st.fprime0 = float(st.g @ st.dir)
        st.f_old, st.x_old, st.g_old = st.f, st.x.copy(), st.g.copy()
    return st.x, st.f


def _pick_alpha(lo, hi, a1, a2, f1, fp1, f2, fp2):
    c = np.array(
        [
            (fp1 + fp2) * (a2 - a1) - 2 * (f2 - f1),
            3 * (f2 - f1) - (2 * fp1 + fp2) * (a2 - a1),
            (a2 - a1) * fp1,
            f1,
        ]
    )
    lb, ub = (lo - a1) / (a2 - a1), (hi - a1) / (a2 - a1)
    if lb > ub:
        lb, ub = ub, lb
    s = np.roots([3 * c[0], 2 * c[1], c[2]])
    s = s[s.imag == 0].real if s.size else s.real
    s = s[(s >= lb) & (s <= ub)]
    pts = np.concatenate([[lb], s, [ub]])
    vals = np.polyval(c, pts)
    k = int(np.argmin(vals))
    return a1 + pts[k] * (a2 - a1), vals[k]


def _gstep(st, opt):
    return min(max(st.init_step / 1e6, opt.diff_min), opt.diff_max)


def _linesearch(f_only, f_grad, st, opt, fmin, tolfun_lns):
    """``linesearch`` = bracketing + sectioning phase; returns an exit flag or None on success."""
    x, d = st.x, st.dir
    alpha_prev = 0.0
    alpha = st.init_step
    f_alpha = st.f
    fp_alpha = st.fprime0
    alpha_max = (fmin - st.f) / (opt.rho * st.fprime0)
    bracket = None
    while True:
        f_prev, fp_prev = f_alpha, fp_alpha
        gstep = _gstep(st, opt)
        f_alpha = f_only(x + alpha * d)
        f2 = f_only(x + (alpha + gstep) * d)
        fp_alpha = (f2 - f_alpha) / gstep
        if f_alpha <= fmin:
            return 4
        if f_alpha > st.f + alpha * opt.rho * st.fprime0 or f_alpha >= f_prev:
            bracket = (alpha_prev, f_prev, fp_prev, alpha, f_alpha, fp_alpha)
            break
        if abs(fp_alpha) <= -opt.sigma * st.fprime0:
            f_alpha, grad = f_grad(x + alpha * d)
            st.alpha, st.f_alpha, st.grad = alpha, f_alpha, grad
            return None
        if fp_alpha >= 0:
            bracket = (alpha, f_alpha, fp_alpha, alpha_prev, f_prev, fp_prev)
            break
        if 2 * alpha - alpha_prev < alpha_max:
            lo = 2 * alpha - alpha_prev
            hi = min(alpha_max, alpha + opt.tau1 * (alpha - alpha_prev))
            new, _ = _pick_alpha(lo, hi, alpha_prev, alpha, f_prev, fp_prev, f_alpha, fp_alpha)
            alpha_prev = alpha
            alpha = new
        else:
            alpha = alpha_max
        if st.func_count > st.max_fun:
            return -1
    # ---- sectioning phase
    a_, fa, fpa, b_, fb, fpb = bracket
    while True:
        lo = a_ + min(opt.tau2, opt.sigma) * (b_ - a_)
        hi = b_ - opt.tau3 * (b_ - a_)
        alpha, _ = _pick_alpha(lo, hi, a_, b_, fa, fpa, fb, fpb)
        if abs((alpha - a_) * fpa) <= tolfun_lns:
            return -2
        gstep = _gstep(st, opt)
        f_alpha = f_only(x + alpha * d)
        f2 = f_only(x + (alpha + gstep) * d)
        fp_alpha = (f2 - f_alpha) / gstep
        a_prev, fa_prev, fpa_prev = a_, fa, fpa
        if f_alpha > st.f + alpha * opt.rho * st.fprime0 or f_alpha >= fa:
            b_, fb, fpb = alpha, f_alpha, fp_alpha
        else:
            if abs(fp_alpha) <= -opt.sigma * st.fprime0:
                f_alpha, grad = f_grad(x + alpha * d)
                st.alpha, st.f_alpha, st.grad = alpha, f_alpha, grad
                return None
            a_, fa, fpa = alpha, f_alpha, fp_alpha
            if (b_ - a_) * fp_alpha >= 0:
                b_, fb, fpb = a_prev, fa_prev, fpa_prev
        if abs(b_ - a_) < np.finfo(float).eps:
            return -2
        if st.func_count > st.max_fun:
            return -1


def icp_finite(static, moving, tol_p: float = 0.001):
    """``ICP_finite(Points_Static, Points_Moving, Options)`` with the rigid model and ``fminlbfgs``.

    ``static``: reference points (cortical vertices); ``moving``: points to align (hull).  Returns
    ``(moved, M)``; each iteration re-matches nearest static points of the *moved* points and
    re-optimises the 6 rigid parameters from the previous solution (parameters are scaled by
    ``[1 1 1 .01 .01 .01]``, rotations in 0.01-degree units).
    """
    static = np.asarray(static, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    scale = np.array([1, 1, 1, 0.01, 0.01, 0.01])
    par = np.zeros(6)
    tol_x = float((static.max(axis=0) - static.min(axis=0)).max() / 1000)
    tree = cKDTree(static)
    moved = moving.copy()
    fval_old = math.inf
    perc = 0.0
    M = np.eye(4)
    while perc < 1 - tol_p:
        match = static[tree.query(moved)[1]]

        def err(p, match=match):
            d = _move(_matrix(p, scale), moving) - match
            return float(np.sum(np.sum(d * d, axis=1)))

        par, fval = fminlbfgs(err, par, tol_x)
        perc = fval / fval_old
        fval_old = fval
        M = _matrix(par, scale)
        moved = _move(M, moving)
    return moved, M


# --------------------------------------------------------------------------- #
# OuterHull
# --------------------------------------------------------------------------- #
def _largest_component(faces, n):
    f = faces
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    A = sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    ncomp, lab = csgraph.connected_components(A, directed=False)
    return lab == np.bincount(lab).argmax()


def outer_hull(surface, *, grid: int = GRID, icp: bool = True) -> Mesh:
    """``OuterHull(vtk_input, vtk_output)``: returns the hull as a :class:`Mesh` (float64 vertices).

    ``surface``: :class:`Mesh` or VTK path.  ``grid``: voxels along the longest side (256 in the
    script).  ``icp=False`` skips the final rigid registration (diagnostics only).
    """
    mesh = surface if isinstance(surface, Mesh) else read_vtk(surface)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    ext = v.max(axis=0) - v.min(axis=0)
    vox = np.ceil(ext / ext.max() * grid).astype(int)
    vol, gx, gy, gz = voxelise(v, f, int(vox[0]), int(vox[1]), int(vox[2]))
    vol = np.pad(vol, PAD)
    vol = ndimage.binary_fill_holes(vol, structure=np.ones((3, 3, 3), bool))
    a = vol.astype(np.float64) * 255.0
    a = _smooth3_gaussian(a, 2 * math.ceil(2 * SMOOTH_SD) + 1, SMOOTH_SD)
    a = _close_ball(a, CLOSING_R, CLOSING_H)
    a = _dilate_ball(a, DILATE_R, DILATE_H)
    a = (a > THRESHOLD).astype(np.float64) * 255.0
    v1, f1 = isosurface(a, ISO)
    # script: swap to [y x z] after MATLAB's (x=col, y=row) convention -> our axes already (0, 1, 2)
    v2 = v1 - v1.min(axis=0)
    lo = np.minimum(np.array([gx.min(), gy.min(), gz.min()]), v.min(axis=0))
    hi = np.maximum(np.array([gx.max(), gy.max(), gz.max()]), v.max(axis=0))
    v2 = v2 * ((hi - lo) / v2.max(axis=0))
    v2 = v2 + lo + np.array([0.5, 0.0, -0.5])
    v2 = v2 * MARGIN
    keep = _largest_component(f1, v2.shape[0])
    remap = np.cumsum(keep) - 1
    f3 = remap[f1[keep[f1].all(axis=1)]]
    v3 = v2[keep]
    v4 = icp_finite(v, v3)[0] if icp else v3
    return Mesh(v4, f3.astype(np.int32), frame=mesh.frame, units=mesh.units)


def write_hull(mesh: Mesh, path) -> None:
    write_vtk(mesh, path, dtype="float")
