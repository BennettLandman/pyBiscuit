"""``HSD::guessInitCoeff``: brute-force rigid initial guess (HSD.cpp 2617-2845, "improved initial guess").

For every deformable subject whose coefficients are all zero, a coarse-to-fine grid over
(pole displacement radius c1, direction c2, rotation about the pole delta0) is searched; the
candidate with the smallest feature mismatch against the current mean map sets the degree-0
coefficients.  The loops use floating-point loop variables exactly like the C++ (accumulated
``c1 += step``), so the visited candidates match.
"""

from __future__ import annotations

import math

import numpy as np

from pybiscuit.mesh.geometry import face_normals
from pybiscuit.sphere.aabb import SphereAABB

from . import kernels as K
from .geom import PI, D, F
from .icosahedron import icosahedron

FLT_MAX = 3.4028234663852886e38


def _stale_normals_tree(verts, faces, normals_from):
    """AABB_Sphere whose pole detection uses the face normals of ``normals_from`` (the mesh
    as it was when ``Mesh::setMesh`` computed them; ``setVertex`` never refreshes them)."""
    tree = SphereAABB(verts, faces)
    tree._fnormals = face_normals(normals_from, faces)
    tree._refresh()
    return tree


def guess_init_coeff(h) -> None:
    """Run the initial guess on ``h`` (an :class:`~pybiscuit.hsd.register.HSD`)."""
    nq = min(10242, h.nsamples)
    nprop = h.nprop
    ps = None
    if nq > 0:
        gv, gf = icosahedron(min(5, h.icosahedron))
        ps = gv.copy()
        axis = (-ps[0, 1], ps[0, 0], F(0))
        theta = F(-F(math.acos(D(ps[0, 2]))))
        rot = K.rotmat(axis[0], axis[1], axis[2], theta)
        ps = K.rotate_points(ps, rot, True)
        tree = _stale_normals_tree(ps, gf, gv)
        rot72 = K.rotmat(F(0), F(0), F(1), F(-PI * 72.0 / 180.0))
        v1 = K.rotate_points(ps, rot72, False)
        fid, bary = tree.closest_face(v1)
        if (fid < 0).any():
            raise RuntimeError("guessInitCoeff: no closest face for a candidate query")
        f = gf[fid]
        b0, b1, b2 = bary[:, 0], bary[:, 1], bary[:, 2]
        cand = np.where(
            (b0 > b1) & (b0 > b2), f[:, 0], np.where((b1 > b2) & (b1 > b0), f[:, 1], f[:, 2])
        ).astype(np.int64)
        tree_ico = _stale_normals_tree(h.ico_v, h.ico_f, h.ico_v_raw)
        fid, bary = tree_ico.closest_face(ps)
        mean = np.zeros((nprop, nq))
        for k in range(nprop):
            K.interpolate64(h.mean[k], h.ico_f, fid, bary, mean[k])
        idmaps = [np.arange(nq, dtype=np.int64)]
        for _ in range(4):
            idmaps.append(cand[idmaps[-1]])

    for s, sub in enumerate(h.subj):
        if sub.fixed:
            continue
        if np.any(sub.coeff != 0):
            continue
        mincost = FLT_MAX
        feature0 = np.empty((nprop, nq), np.float32)
        feature = np.empty((nprop, nq), np.float32)
        feature_res = np.empty((nprop, nq), np.float32)
        fid, bary = sub.tree.closest_face(ps)
        for k in range(nprop):
            K.interpolate(sub.prop[k], sub.faces, fid, bary, feature0[k])
        P = sub.axis0
        range_c1 = [0.0, PI / 2, PI / 8]
        range_c2 = [0.0, 2 * PI, PI / 8]
        intv = 3
        opt_c1 = opt_c2 = 0.0
        for _res in range(h.guess_res):
            c1 = range_c1[0]
            while c1 <= range_c1[1]:
                c2 = range_c2[0]
                while c2 < range_c2[1]:
                    if c1 == 0 and c2 > 0:
                        c2 += range_c2[2]
                        continue
                    d1 = c1 * math.cos(c2)
                    d2 = c1 * math.sin(c2)
                    # exponential map (linear): axis = unit(P + tan1 d1 + tan2 d2)
                    f1, f2 = F(d1), F(d2)
                    tx = P[0] + sub.tan1[0] * f1
                    ty = P[1] + sub.tan1[1] * f1
                    tz = P[2] + sub.tan1[2] * f1
                    tx = tx + sub.tan2[0] * f2
                    ty = ty + sub.tan2[1] * f2
                    tz = tz + sub.tan2[2] * f2
                    qx, qy, qz = _unit(tx, ty, tz)
                    ax, ay, az = _cross(P[0], P[1], P[2], qx, qy, qz)
                    ax, ay, az = _unit(ax, ay, az)
                    rotA = K.rotmat(ax, ay, az, F(-F(c1)))
                    pts = K.rotate_points(ps, rotA, False)
                    fid, bary = tree.closest_face(pts)
                    for k in range(nprop):
                        K.interpolate(feature0[k], gf, fid, bary, feature[k])
                    for w in range(intv):
                        for c3 in range(5):
                            d0 = 2 * PI / 5 * c3 + 2 * PI / 5 * float(w) / float(intv)
                            if c3 == 0:
                                rotP = K.rotmat(P[0], P[1], P[2], F(-d0))
                                pts = K.rotate_points(ps, rotP, False)
                                fid, bary = tree.closest_face(pts)
                                for k in range(nprop):
                                    K.interpolate(feature[k], gf, fid, bary, feature_res[k])
                            cost = K.guess_cost(feature_res, mean, idmaps[c3], nprop, nq)
                            if cost < mincost:
                                y0 = sub.Y[0, 0]
                                sub.coeff[0, 0] = d0 / y0
                                sub.coeff[1, 0] = d1 / y0
                                sub.coeff[2, 0] = d2 / y0
                                mincost = cost
                                opt_c1 = c1
                                opt_c2 = c2
                    c2 += range_c2[2]
                c1 += range_c1[2]
            range_c1[0] = opt_c1 - range_c1[2]
            range_c1[1] = opt_c1 + range_c1[2]
            range_c1[2] /= 2
            range_c1[0] += range_c1[2]
            range_c1[1] -= range_c1[2]
            range_c1[0] = range_c1[2] if range_c1[0] < 0 else range_c1[0]
            if opt_c1 != 0:
                range_c2[0] = opt_c2 - range_c2[2]
                range_c2[1] = opt_c2 + range_c2[2]
            range_c2[2] /= 2
            if opt_c1 != 0:
                range_c2[0] += range_c2[2]
            intv *= 2

    for s in range(h.nsubj):
        h.update_deformation(s, True)
    h.updated[:] = False
    h.trace()
    h.coeff_prev[:] = h.coeff
    h.guess_active = False


def _unit(x, y, z):
    s = F(math.sqrt(D(x * x + y * y + z * z)))
    if s > F(0):
        return x / s, y / s, z / s
    return x, y, z


def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
