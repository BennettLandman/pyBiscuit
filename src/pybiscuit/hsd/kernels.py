# ruff: noqa: E501
"""Numba kernels for HSD: deformation, areas, flips, energies, feature sampling, statistics.

Each kernel names the C++ function it ports (``HSD.cpp`` @ f169595) and keeps its float32 /
float64 mix (see ``docs/ports/hsd.md`` for the precision table).
"""

from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, njit, prange

from pybiscuit.sphere.aabb import _bary32

from .geom import (
    PI,
    D,
    F,
    _cart2sph,
    _rot_point,
    _sph2cart,
    _unit3,
    bary32_snap,
    dot3,
    norm3,
    rotation_into,
)
from .geom import _cross3 as cross3

PAR = {"cache": True, "error_model": "numpy", "parallel": True}
SEQ = {"cache": True, "error_model": "numpy"}


def _nchunk(n: int) -> int:
    return max(1, min(n, get_num_threads() * 8))


# --------------------------------------------------------------------------- #
# HSD::updateCoordinate for every vertex
# --------------------------------------------------------------------------- #
@njit(**SEQ)
def _update_coord_range(lo, hi, p, Y, coeff, n, axis0, tan1, tan2, out):
    m = np.empty(9, np.float32)
    A = np.empty(9, np.float32)
    for v in range(lo, hi):
        d0 = 0.0
        d1 = 0.0
        d2 = 0.0
        for i in range(n):
            y = Y[v, i]
            d0 += y * coeff[0, i]
            d1 += y * coeff[1, i]
            d2 += y * coeff[2, i]
        # axis = unit(axis0 + tan1 * d1 + tan2 * d2): float32 Vector arithmetic, delta -> float
        f1 = F(d1)
        f2 = F(d2)
        tx = axis0[0] + tan1[0] * f1
        ty = axis0[1] + tan1[1] * f1
        tz = axis0[2] + tan1[2] * f1
        tx = tx + tan2[0] * f2
        ty = ty + tan2[1] * f2
        tz = tz + tan2[2] * f2
        qx, qy, qz = _unit3(tx, ty, tz)
        angle = F(math.sqrt(d1 * d1 + d2 * d2))
        ax, ay, az = cross3(axis0[0], axis0[1], axis0[2], qx, qy, qz)
        ax, ay, az = _unit3(ax, ay, az)
        rotation_into(m, A, ax, ay, az, angle)
        rx, ry, rz = _rot_point(m, p[v, 0], p[v, 1], p[v, 2])
        if d0 == 0.0:
            out[v, 0] = rx
            out[v, 1] = ry
            out[v, 2] = rz
            continue
        rotation_into(m, A, qx, qy, qz, F(d0))
        px, py, pz = _rot_point(m, rx, ry, rz)
        out[v, 0] = px
        out[v, 1] = py
        out[v, 2] = pz


@njit(**PAR)
def _update_coord_all(p, Y, coeff, n, axis0, tan1, tan2, out, nchunk):
    nv = p.shape[0]
    for c in prange(nchunk):
        lo = c * nv // nchunk
        hi = (c + 1) * nv // nchunk
        _update_coord_range(lo, hi, p, Y, coeff, n, axis0, tan1, tan2, out)


def update_coordinates(p, Y, coeff, n, pole_axis, tan1, tan2):
    """``updateCoordinate`` for all vertices: returns the *unnormalised* float32 (N,3) result.

    ``coeff`` is the subject's float64 (3, K) block, ``n = (degree + 1)**2`` terms are summed
    (sequential double accumulation), ``pole_axis`` = ``sph2cart(pole)`` (float32).
    """
    out = np.empty_like(p)
    _update_coord_all(p, Y, coeff, int(n), pole_axis, tan1, tan2, out, _nchunk(p.shape[0]))
    return out


# --------------------------------------------------------------------------- #
# areas and flips
# --------------------------------------------------------------------------- #
@njit(**SEQ)
def face_areas(v, faces, out):
    """``updateArea`` / ``initArea`` loop: ``|(V2-V1)x(V3-V1)|/2`` clipped below at 1e-6 (float32)."""
    for i in range(faces.shape[0]):
        a, b, c = faces[i, 0], faces[i, 1], faces[i, 2]
        ux, uy, uz = cross3(
            v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2],
            v[c, 0] - v[a, 0], v[c, 1] - v[a, 1], v[c, 2] - v[a, 2],
        )  # fmt: skip
        area = norm3(ux, uy, uz) / F(2)
        if D(area) < 1e-6:
            area = F(1e-6)
        out[i] = area


@njit(**SEQ)
def init_flips(v, faces, flip):
    """``initTriangleFlipping``: returns (neg_area, n_initial_flips, n_flips)."""
    neg_area = 0.0
    ninit = 0
    nflip = 0
    for i in range(faces.shape[0]):
        a, b, c = faces[i, 0], faces[i, 1], faces[i, 2]
        sx = (v[a, 0] + v[b, 0]) + v[c, 0]
        sy = (v[a, 1] + v[b, 1]) + v[c, 1]
        sz = (v[a, 2] + v[b, 2]) + v[c, 2]
        vx, vy, vz = sx / F(3), sy / F(3), sz / F(3)
        ux, uy, uz = cross3(
            v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2],
            v[c, 0] - v[a, 0], v[c, 1] - v[a, 1], v[c, 2] - v[a, 2],
        )  # fmt: skip
        flip[i] = False
        if dot3(vx, vy, vz, ux, uy, uz) < F(0):
            neg = D(norm3(ux, uy, uz))
            neg_area += neg
            ninit += 1
            if neg > 1e-4:
                flip[i] = True
        if flip[i]:
            nflip += 1
    return neg_area, ninit, nflip


@njit(**SEQ)
def test_triangle_flip(v, faces, flip, threshold):
    """``testTriangleFlip``: accumulated area of faces whose orientation changed > threshold."""
    area = 0.0
    thr = D(threshold)
    for i in range(faces.shape[0]):
        a, b, c = faces[i, 0], faces[i, 1], faces[i, 2]
        sx = (v[a, 0] + v[b, 0]) + v[c, 0]
        sy = (v[a, 1] + v[b, 1]) + v[c, 1]
        sz = (v[a, 2] + v[b, 2]) + v[c, 2]
        vx, vy, vz = sx / F(3), sy / F(3), sz / F(3)
        ux, uy, uz = cross3(
            v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2],
            v[c, 0] - v[a, 0], v[c, 1] - v[a, 1], v[c, 2] - v[a, 2],
        )  # fmt: skip
        d = dot3(vx, vy, vz, ux, uy, uz)
        if (d < F(0) and not flip[i]) or (d > F(0) and flip[i]):
            area += D(norm3(ux, uy, uz))
        if area > thr:
            return True
    return False


# --------------------------------------------------------------------------- #
# energies
# --------------------------------------------------------------------------- #
@njit(**SEQ)
def var_area_add(area0, area1, cost):
    """Inner loop of ``varArea`` (one subject): ``cost += |log(a1/a0)| / nFace`` (float32 log)."""
    nf = area0.shape[0]
    for i in range(nf):
        ratio = D(abs(F(math.log(D(area1[i] / area0[i])))))
        cost += ratio / nf
    return cost


@njit(**SEQ)
def var_displacement_add(p0, v, cost):
    """Inner loop of ``varDisplacement`` (one subject)."""
    nv = v.shape[0]
    for i in range(nv):
        inner = dot3(p0[i, 0], p0[i, 1], p0[i, 2], v[i, 0], v[i, 1], v[i, 2])
        if inner > F(1):
            inner = F(1)
        if inner < F(-1):
            inner = F(-1)
        degree = D(F(math.acos(D(inner))))
        cost += degree * degree / nv
    return cost


@njit(**SEQ)
def var_properties(feature, mean, variance, active, nprop_total, use_unit_var):
    """``varProperties`` for subj_id == -1: feature (S, K, M) float32, mean/variance (K, M) float64."""
    cost = 0.0
    nsubj, nk, ns = feature.shape
    den = nprop_total * ns
    for k in range(nk):
        sd = 0.0
        for i in range(ns):
            m = mean[k, i]
            for s in range(nsubj):
                if not active[s]:
                    continue
                p = D(feature[s, k, i])
                pm = p - m
                var = 1.0 if use_unit_var else variance[k, i]
                sd += pm * pm / den / var
        cost += sd
    return cost / nsubj


# --------------------------------------------------------------------------- #
# feature sampling (updateProperties)
# --------------------------------------------------------------------------- #
@njit(**SEQ)
def _patch_one(v, faces, patch_cache, tc, yx, yy, yz, maxdist, flist):
    """Cached-patch search of ``updateProperties`` for one sample. Returns (fid, c0, c1, c2)."""
    size = 1
    flist[0] = tc
    init = 0
    ln = 1
    for _ in range(2):  # m_patch_range
        for idx in range(init, ln):
            for j in range(3):
                pc = patch_cache[flist[idx], j]
                if pc != -1:
                    flist[size] = pc
                    size += 1
        init = ln
        ln = size
    fid = -1
    c0 = F(0)
    c1 = F(0)
    c2 = F(0)
    for j in range(size):
        if fid != -1:
            break
        f = flist[j]
        dup = False
        for q in range(j):
            if flist[q] == f:
                dup = True
        if dup:
            continue
        a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
        nx, ny, nz = cross3(
            v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2],
            v[c, 0] - v[b, 0], v[c, 1] - v[b, 1], v[c, 2] - v[b, 2],
        )  # fmt: skip
        nx, ny, nz = _unit3(nx, ny, nz)
        scale = dot3(v[a, 0], v[a, 1], v[a, 2], nx, ny, nz) / dot3(yx, yy, yz, nx, ny, nz)
        if abs(scale - F(1)) > maxdist:
            continue
        px, py, pz = yx * scale, yy * scale, yz * scale
        c0, c1, c2 = _bary32(
            v[a, 0], v[a, 1], v[a, 2], v[b, 0], v[b, 1], v[b, 2], v[c, 0], v[c, 1], v[c, 2],
            px, py, pz,
        )  # fmt: skip
        if c0 >= F(0) and c1 >= F(0) and c2 >= F(0):
            fid = f
        else:
            fid = -1
    return fid, c0, c1, c2


@njit(**PAR)
def patch_search(v, faces, patch_cache, tree_cache, samples, out_fid, out_bary, nchunk):
    ns = samples.shape[0]
    maxdist = F(1.0 - math.cos(PI / 180.0 * 36.0))
    for c in prange(nchunk):
        lo = c * ns // nchunk
        hi = (c + 1) * ns // nchunk
        flist = np.empty(16, np.int32)
        for i in range(lo, hi):
            out_fid[i] = -1
            tc = tree_cache[i]
            if tc == -1:
                continue
            fid, c0, c1, c2 = _patch_one(
                v, faces, patch_cache, tc, samples[i, 0], samples[i, 1], samples[i, 2], maxdist, flist
            )
            out_fid[i] = fid
            out_bary[i, 0] = c0
            out_bary[i, 1] = c1
            out_bary[i, 2] = c2


@njit(**SEQ)
def interpolate(ref, faces, fid, bary, out):
    """``propertyInterpolation(float*)``: ``ref[a]*c0 + ref[b]*c1 + ref[c]*c2`` (0 when fid == -1)."""
    for i in range(fid.shape[0]):
        f = fid[i]
        if f == -1:
            out[i] = F(0)
        else:
            out[i] = (
                ref[faces[f, 0]] * bary[i, 0]
                + ref[faces[f, 1]] * bary[i, 1]
                + ref[faces[f, 2]] * bary[i, 2]
            )


@njit(**SEQ)
def interpolate64(ref, faces, fid, bary, out):
    """``propertyInterpolation(double*)`` (double ref, float coefficients)."""
    for i in range(fid.shape[0]):
        f = fid[i]
        if f == -1:
            out[i] = 0.0
        else:
            out[i] = (
                ref[faces[f, 0]] * D(bary[i, 0])
                + ref[faces[f, 1]] * D(bary[i, 1])
                + ref[faces[f, 2]] * D(bary[i, 2])
            )


# --------------------------------------------------------------------------- #
# property normalisation (initProperties) and statistics (Statistics::*, float32)
# --------------------------------------------------------------------------- #
@njit(**SEQ)
def _mean32(a):
    s = F(0)
    for i in range(a.shape[0]):
        s += a[i]
    return s / F(a.shape[0])


@njit(**SEQ)
def _var32(a):
    n = a.shape[0]
    m = _mean32(a)
    res = F(0)
    for i in range(n):
        res += (a[i] - m) * (a[i] - m) / F(n)
    return res


@njit(**SEQ)
def _median32(a):
    t = np.sort(a.copy())
    n = t.shape[0]
    if n % 2 == 1:
        return t[n // 2]
    return (t[n // 2 - 1] + t[n // 2]) * F(0.5)


@njit(**SEQ)
def _soft_clamp(a):
    for j in range(a.shape[0]):
        if a[j] < F(-3):
            a[j] = F(-3) - (F(1) - F(math.exp(D(F(3) + a[j]))))
        if a[j] > F(3):
            a[j] = F(3) + (F(1) - F(math.exp(D(F(3) - a[j]))))


@njit(**SEQ)
def normalize_property(a):
    """The median z-score block of ``initProperties`` for one property (float32, in place)."""
    med = _median32(a)
    for j in range(a.shape[0]):
        a[j] -= med
    sd = F(math.sqrt(D(_var32(a))))
    for j in range(a.shape[0]):
        a[j] /= sd
    _soft_clamp(a)
    sd = F(math.sqrt(D(_var32(a))))
    for j in range(a.shape[0]):
        a[j] /= sd
    _soft_clamp(a)


@njit(**SEQ)
def feature_mean(feature, select, ndef, nsubj, mean):
    """First loop of ``updatePropertyStats``."""
    nk = feature.shape[1]
    ns = feature.shape[2]
    for k in range(nk):
        for i in range(ns):
            m = 0.0
            for s in range(nsubj):
                if select[s]:
                    m += D(feature[s, k, i])
            m = m / nsubj if nsubj == ndef else m / (nsubj - ndef)
            mean[k, i] = m


@njit(**SEQ)
def feature_variance(feature, mean, nsubj, k, out):
    """Sample variance across subjects for property ``k`` (``updatePropertyStats`` else-branch)."""
    ns = feature.shape[2]
    for i in range(ns):
        m = mean[k, i]
        sd = 0.0
        for s in range(nsubj):
            p = D(feature[s, k, i])
            pm = p - m
            sd += pm * pm / (nsubj - 1)
        out[i] = sd


# --------------------------------------------------------------------------- #
# pole / tangent plane, rotations, misc helpers
# --------------------------------------------------------------------------- #
@njit(**SEQ)
def pole_axis(phi, theta):
    """``Coordinate::sph2cart(pole[0], pole[1], axis0)`` (float32)."""
    x, y, z = _sph2cart(F(phi), F(theta))
    return np.array([x, y, z], dtype=np.float32)


@njit(**SEQ)
def tangent_plane(phi, theta):
    """``HSD::initTangentPlane`` (no landmarks): returns (tan1, tan2) float32.

    ``tan1 = unit(proj - fmean)`` with ``proj`` = the sample at (pole + 0.1) projected onto the
    tangent plane (``proj2plane(fmean, d = -1)``); ``tan2 = unit(fmean x tan1)``.
    """
    fx, fy, fz = _sph2cart(F(phi), F(theta))
    sx, sy, sz = _sph2cart(F(D(F(phi)) + 0.1), F(D(F(theta)) + 0.1))
    d = F(-1)
    portion = (fx * sx + fy * sy + fz * sz + d) / (fx * fx + fy * fy + fz * fz)
    px = sx - fx * portion
    py = sy - fy * portion
    pz = sz - fz * portion
    tx, ty, tz = _unit3(px - fx, py - fy, pz - fz)
    cx, cy, cz = cross3(fx, fy, fz, tx, ty, tz)
    ux, uy, uz = _unit3(cx, cy, cz)
    return (
        np.array([tx, ty, tz], dtype=np.float32),
        np.array([ux, uy, uz], dtype=np.float32),
    )


@njit(**SEQ)
def rotmat(ax, ay, az, theta):
    """``Coordinate::rotation`` -> float32 (9,)."""
    m = np.empty(9, np.float32)
    A = np.empty(9, np.float32)
    rotation_into(m, A, F(ax), F(ay), F(az), F(theta))
    return m


@njit(**SEQ)
def rotate_points(pts, m, normalize):
    """``rotPoint`` of every row (float32), optionally followed by ``unit()``."""
    out = np.empty_like(pts)
    for i in range(pts.shape[0]):
        x, y, z = _rot_point(m, pts[i, 0], pts[i, 1], pts[i, 2])
        if normalize:
            x, y, z = _unit3(x, y, z)
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z
    return out


@njit(**SEQ)
def seq_sum_div(a, n):
    """``for i: m += a[i] / n`` (sequential double)."""
    m = 0.0
    for i in range(a.shape[0]):
        m += a[i] / n
    return m


@njit(**SEQ)
def guess_cost(feature_res, mean, idmap, nprop, nq):
    """Cost of one rotation candidate in ``guessInitCoeff`` (double accumulation)."""
    cost = 0.0
    den = nprop * nq
    for i in range(nq):
        idx = idmap[i]
        for k in range(nprop):
            pm = D(feature_res[k, idx]) - mean[k, i]
            cost += pm * pm / den
    return cost


@njit(**SEQ)
def pairwise_bary(v, faces, fid, samples, out):
    """Barycentric coordinates (err = 1e-5) used by ``initPairwise`` for the template variance."""
    for i in range(fid.shape[0]):
        f = fid[i]
        a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
        nx, ny, nz = cross3(
            v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2],
            v[c, 0] - v[b, 0], v[c, 1] - v[b, 1], v[c, 2] - v[b, 2],
        )  # fmt: skip
        nx, ny, nz = _unit3(nx, ny, nz)
        yx, yy, yz = samples[i, 0], samples[i, 1], samples[i, 2]
        sc = dot3(v[a, 0], v[a, 1], v[a, 2], nx, ny, nz) / dot3(yx, yy, yz, nx, ny, nz)
        c0, c1, c2 = bary32_snap(
            v[a, 0], v[a, 1], v[a, 2], v[b, 0], v[b, 1], v[b, 2], v[c, 0], v[c, 1], v[c, 2],
            yx * sc, yy * sc, yz * sc, F(1e-5),
        )  # fmt: skip
        out[i, 0] = c0
        out[i, 1] = c1
        out[i, 2] = c2


# --------------------------------------------------------------------------- #
# landmarks (opt-in; dead code in the shipped HSD CLI)
# --------------------------------------------------------------------------- #
@njit(**SEQ)
def var_landmarks(lm_x, select, nsubj):
    """``varLandmarks(-1)``: mean of the selected subjects' landmarks (double, normalised), then
    the mean squared arc length of every subject's landmark to it, summed over landmarks,
    divided by ``nsubj - 1``."""
    nlm = lm_x.shape[1]
    cost = 0.0
    for i in range(nlm):
        m0 = 0.0
        m1 = 0.0
        m2 = 0.0
        for s in range(nsubj):
            if select[s]:
                m0 += D(lm_x[s, i, 0])
                m1 += D(lm_x[s, i, 1])
                m2 += D(lm_x[s, i, 2])
        nrm = math.sqrt(m0 * m0 + m1 * m1 + m2 * m2)
        m0 /= nrm
        m1 /= nrm
        m2 /= nrm
        sd = 0.0
        for s in range(nsubj):
            inner = 0.0
            inner += m0 * D(lm_x[s, i, 0])
            inner += m1 * D(lm_x[s, i, 1])
            inner += m2 * D(lm_x[s, i, 2])
            if inner > 1:
                inner = 1.0
            elif inner < -1:
                inner = -1.0
            arclen = math.acos(inner)
            sd += arclen * arclen / nlm
        cost += sd
    return cost / (nsubj - 1)


@njit(**SEQ)
def landmark_pole_mean(phi, theta, lm):
    """Start of ``initTangentPlane`` with landmarks: ``unit(sph2cart(pole) + sum p_i)`` (float32)
    back to (phi, theta) with ``Coordinate::cart2sph``."""
    x, y, z = _sph2cart(F(phi), F(theta))
    for i in range(lm.shape[0]):
        x = x + lm[i, 0]
        y = y + lm[i, 1]
        z = z + lm[i, 2]
    x, y, z = _unit3(x, y, z)
    return _cart2sph(x, y, z)


@njit(**SEQ)
def landmark_pole_cost(phi, theta, d0, d1, lm):
    """``HSD::optimalMeanCost``: float32 sum of squared arc lengths from the pole
    ``(phi + (float)d0, theta + (float)d1)`` to the landmarks."""
    px, py, pz = _sph2cart(F(phi) + F(d0), F(theta) + F(d1))
    cost = F(0)
    for i in range(lm.shape[0]):
        inner = dot3(px, py, pz, lm[i, 0], lm[i, 1], lm[i, 2])
        if inner > F(1):
            inner = F(1)
        elif inner < F(-1):
            inner = F(-1)
        arclen = F(math.acos(D(inner)))
        cost += arclen * arclen
    return D(cost)
