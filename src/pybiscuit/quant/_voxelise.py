"""Mesh voxelisation by ray casting: port of Adam Aitkenhead's ``VOXELISE.m`` (as shipped in
LocalGyrificationIndex ``matlab/VOXELISE``, with ``COMPUTE_mesh_normals.m``).

Only the call used by ``OuterHull.m`` is supported: ``VOXELISE(nx, ny, nz, fv, 'xyz')`` with integer
grid sizes (a grid of ``n`` voxel centres is laid out between ``min + w/2`` and ``max - w/2`` with
``w = (max - min) / (n + 1/2)`` on every axis) and the three ray directions combined by majority
vote.  The ray/facet logic, the rounding of crossings to 1e-12, the even/odd crossing rule and the
neighbour-based correction of failed rays are reproduced.  MATLAB indices are 1-based; here 0-based.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["voxelise"]


@njit(cache=True, error_model="numpy")
def _matlab_round(x):
    # MATLAB round(): half away from zero
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


@njit(cache=True, error_model="numpy")
def _face_normal_z(M, f):
    ax, ay, az = M[f, 0, 0], M[f, 1, 0], M[f, 2, 0]
    bx, by, bz = M[f, 0, 1], M[f, 1, 1], M[f, 2, 1]
    cx, cy, cz = M[f, 0, 2], M[f, 1, 2], M[f, 2, 2]
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    nx = aby * acz - abz * acy
    ny = abz * acx - abx * acz
    nz = abx * acy - aby * acx
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    return nz / n


@njit(cache=True, error_model="numpy")
def _share_vertex(M, f, g):
    for a in range(3):
        for b in range(3):
            if M[f, 0, a] == M[g, 0, b] and M[f, 1, a] == M[g, 1, b] and M[f, 2, a] == M[g, 2, b]:
                return True
    return False


@njit(cache=True, error_model="numpy")
def _internal(gx, gy, gz, M, fmin, fmax):
    """``VOXELISEinternal``: rays along z through the (gx, gy) grid.  Returns the bool volume."""
    nx, ny, nz = gx.shape[0], gy.shape[0], gz.shape[0]
    nf = M.shape[0]
    out = np.zeros((nx, ny, nz), dtype=np.bool_)
    zmin = M[:, 2, :].min()
    zmax = M[:, 2, :].max()
    xmin = M[:, 0, :].min()
    xmax = M[:, 0, :].max()
    ymin = M[:, 1, :].min()
    ymax = M[:, 1, :].max()
    # first index of the grid coordinate nearest to the mesh extremes (find(...==min(...)))
    xminp = np.argmin(np.abs(gx - xmin))
    xmaxp = np.argmin(np.abs(gx - xmax))
    yminp = np.argmin(np.abs(gy - ymin))
    ymaxp = np.argmin(np.abs(gy - ymax))
    if xminp > xmaxp:
        xminp, xmaxp = xmaxp, xminp
    if yminp > ymaxp:
        yminp, ymaxp = ymaxp, yminp
    corr = np.zeros((ny * nx + 1, 2), dtype=np.int64)
    ncorr = 0
    ylist = np.empty(nf, dtype=np.int64)
    plist = np.empty(nf, dtype=np.int64)
    for ly in range(yminp, ymaxp + 1):
        cy = gy[ly]
        ny_ = 0
        for f in range(nf):
            if fmin[f, 1] <= cy and fmax[f, 1] >= cy:
                ylist[ny_] = f
                ny_ += 1
        for lx in range(xminp, xmaxp + 1):
            cx = gx[lx]
            npos = 0
            for k in range(ny_):
                f = ylist[k]
                if fmin[f, 0] <= cx and fmax[f, 0] >= cx:
                    plist[npos] = f
                    npos += 1
            if npos == 0:
                continue
            fac = np.empty(npos * 2 + 8, dtype=np.int64)  # facetCROSSLIST (may hold duplicates)
            nfac = 0
            # facets with a vertex exactly on the ray
            vlist = np.empty(npos, dtype=np.int64)
            nv = 0
            for k in range(npos):
                f = plist[k]
                hit = False
                for c in range(3):
                    if M[f, 0, c] == cx and M[f, 1, c] == cy:
                        hit = True
                if hit:
                    vlist[nv] = f
                    nv += 1
            abandoned = False
            if nv > 0:
                checked = np.zeros(nv, dtype=np.bool_)
                while True:
                    first = -1
                    for k in range(nv):
                        if not checked[k]:
                            first = k
                            break
                    if first < 0:
                        break
                    checked[first] = True
                    zmx = -2.0
                    zmn = 2.0
                    for k in range(nv):
                        if _share_vertex(M, vlist[first], vlist[k]):
                            checked[k] = True
                            nzv = _face_normal_z(M, vlist[k])
                            if nzv > zmx:
                                zmx = nzv
                            if nzv < zmn:
                                zmn = nzv
                    if zmx < 0 or zmn > 0:
                        fac[nfac] = vlist[first]
                        nfac += 1
                    else:
                        abandoned = True
                        corr[ncorr, 0] = lx
                        corr[ncorr, 1] = ly
                        ncorr += 1
                        break
            if abandoned:
                continue
            for k in range(npos):
                f = plist[k]
                x1, x2, x3 = M[f, 0, 0], M[f, 0, 1], M[f, 0, 2]
                y1, y2, y3 = M[f, 1, 0], M[f, 1, 1], M[f, 1, 2]
                y1p = y2 - ((y2 - y3) * (x2 - x1) / (x2 - x3))
                yr = y2 - ((y2 - y3) * (x2 - cx) / (x2 - x3))
                if (y1p > y1 and yr > cy) or (y1p < y1 and yr < cy):
                    y2p = y3 - ((y3 - y1) * (x3 - x2) / (x3 - x1))
                    yr = y3 - ((y3 - y1) * (x3 - cx) / (x3 - x1))
                    if (y2p > y2 and yr > cy) or (y2p < y2 and yr < cy):
                        y3p = y1 - ((y1 - y2) * (x1 - x3) / (x1 - x2))
                        yr = y1 - ((y1 - y2) * (x1 - cx) / (x1 - x2))
                        if (y3p > y3 and yr > cy) or (y3p < y3 and yr < cy):
                            if nfac >= fac.shape[0]:
                                nf2 = np.empty(fac.shape[0] * 2, dtype=np.int64)
                                nf2[:nfac] = fac[:nfac]
                                fac = nf2
                            fac[nfac] = f
                            nfac += 1
            zc = np.empty(nfac, dtype=np.float64)
            for k in range(nfac):
                f = fac[k]
                X1, X2, X3 = M[f, 0, 0], M[f, 0, 1], M[f, 0, 2]
                Y1, Y2, Y3 = M[f, 1, 0], M[f, 1, 1], M[f, 1, 2]
                Z1, Z2, Z3 = M[f, 2, 0], M[f, 2, 1], M[f, 2, 2]
                A = Y1 * (Z2 - Z3) + Y2 * (Z3 - Z1) + Y3 * (Z1 - Z2)
                B = Z1 * (X2 - X3) + Z2 * (X3 - X1) + Z3 * (X1 - X2)
                C = X1 * (Y2 - Y3) + X2 * (Y3 - Y1) + X3 * (Y1 - Y2)
                D = -X1 * (Y2 * Z3 - Y3 * Z2) - X2 * (Y3 * Z1 - Y1 * Z3) - X3 * (Y1 * Z2 - Y2 * Z1)
                if abs(C) < 1e-14:
                    C = 0.0
                zc[k] = (-D - A * cx - B * cy) / C
            keep = np.empty(nfac, dtype=np.float64)
            nk = 0
            for k in range(nfac):
                z = zc[k]
                if z >= zmin - 1e-12 and z <= zmax + 1e-12:
                    keep[nk] = _matlab_round(z * 1e12) / 1e12
                    nk += 1
            ks = np.sort(keep[:nk])
            uq = np.empty(nk, dtype=np.float64)
            nu = 0
            for k in range(nk):
                if nu == 0 or ks[k] != uq[nu - 1]:
                    uq[nu] = ks[k]
                    nu += 1
            if nu % 2 == 0:
                for p in range(nu // 2):
                    lo = uq[2 * p]
                    hi = uq[2 * p + 1]
                    for kz in range(nz):
                        if gz[kz] > lo and gz[kz] < hi:
                            out[lx, ly, kz] = True
            elif nu != 0:
                corr[ncorr, 0] = lx
                corr[ncorr, 1] = ly
                ncorr += 1
    if ncorr > 0:
        pad = False
        for c in range(ncorr):
            if corr[c, 0] == 0 or corr[c, 0] == nx - 1 or corr[c, 1] == 0 or corr[c, 1] == ny - 1:
                pad = True
        if pad:
            big = np.zeros((nx + 2, ny + 2, nz), dtype=np.bool_)
            big[1:-1, 1:-1, :] = out
            out = big
            off = 1
        else:
            off = 0
        for c in range(ncorr):
            i = corr[c, 0] + off
            j = corr[c, 1] + off
            for kz in range(nz):
                s = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di != 0 or dj != 0:
                            if out[i + di, j + dj, kz]:
                                s += 1
                if s >= 4:
                    out[i, j, kz] = True
        if out.shape[0] > nx or out.shape[1] > ny:
            out = out[1:-1, 1:-1, :].copy()
    return out


def _grid(n: int, lo: float, hi: float) -> np.ndarray:
    w = (hi - lo) / (n + 1 / 2)
    return lo + w / 2 + np.arange(n) * w


def voxelise(vertices, faces, nx: int, ny: int, nz: int):
    """``VOXELISE(nx, ny, nz, struct(vertices, faces), 'xyz')``.

    Returns ``(volume bool (nx, ny, nz), gx, gy, gz)``: the cropped voxel-centre coordinates.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    M = np.ascontiguousarray(np.transpose(v[f], (0, 2, 1)))  # (F, xyz, corner)
    lo, hi = v.min(axis=0), v.max(axis=0)
    gx = _grid(nx, lo[0], hi[0])
    gy = _grid(ny, lo[1], hi[1])
    gz = _grid(nz, lo[2], hi[2])
    # only the first axis that needs it is extended (elseif chain): 'x' always does here
    gxe = np.concatenate([[lo[0]], gx]) if gx.min() > lo[0] else gx
    gcheck = 1 if gx.min() > lo[0] else 0
    if gxe.max() < hi[0]:
        gxe = np.concatenate([gxe, [hi[0]]])
        gcheck += 2

    def run(a, b, c, perm):
        Mp = np.ascontiguousarray(M[:, perm, :])
        fmin = np.ascontiguousarray(Mp.min(axis=2))
        fmax = np.ascontiguousarray(Mp.max(axis=2))
        return _internal(a, b, c, Mp, fmin, fmax)

    rx = np.transpose(run(gy, gz, gxe, [1, 2, 0]), (2, 0, 1))
    ry = np.transpose(run(gz, gxe, gy, [2, 0, 1]), (1, 2, 0))
    rz = run(gxe, gy, gz, [0, 1, 2])
    vol = (rx.astype(np.int8) + ry + rz) >= 1.5
    if gcheck == 1:
        vol, gxe = vol[1:], gxe[1:]
    elif gcheck == 2:
        vol, gxe = vol[:-1], gxe[:-1]
    elif gcheck == 3:
        vol, gxe = vol[1:-1], gxe[1:-1]
    return np.ascontiguousarray(vol), gxe, gy, gz
