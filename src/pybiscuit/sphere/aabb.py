"""Point location on a (deformed) sphere mesh: port of MeshLib ``AABB_Sphere``.

Source: ``reference/repos/MeshLib`` @ dcef807b (``src/Util/AABB_Sphere.cpp``,
``include/Util/AABB_Sphere.h``, ``Coordinate::cart2sph/cart2bary/dpoint2tri`` in
``src/Util/Geom.cpp``).  See ``docs/ports/aabb-sphere.md``.

Public API (stable)
===================

``SphereAABB(mesh)`` / ``SphereAABB(vertices, faces)``
    Builds the tree.  Vertices are converted to **float32** (MeshLib stores
    ``float``); every computation in the query path is float32, exactly as in
    the C++ source, so results (face index, barycentric coefficients) match the
    upstream binary bit-for-bit up to libm ulp differences in ``atan2`` (the only
    platform-dependent operation; see "Determinism" in ``docs/ports/aabb-sphere.md``).

``SphereAABB.closest_face(points, eps1=None, eps2=None, *, poles=True)``
    ``-> (face int32 (n,), bary float32 (n, 3))``.  Two overloads:

    * ``eps1 is None`` (default) -- ``AABB_Sphere::closestFace(v, coeff)``, the
      overload the tools call.  Attempt 0 uses enlargement ``0``; if no face is
      found, attempts ``trial = 0..8`` retry with ``rad = (float)0.0174533 * 2**trial``
      (1 degree, doubled each time, applied to *both* angles).  The poles are
      only used at enlargement 0.
    * ``eps1`` given (``eps2`` defaults to ``eps1``) -- one shot of the private
      ``closestFaceBool(v, coeff, visited, eps1, eps2)``: candidate leaves are
      those whose (phi, theta) box, enlarged by ``eps1`` in azimuth and ``eps2``
      in elevation, contains the query; if ``eps1 == 0`` and both pole faces
      exist the pole faces are added as candidates (``theta >= maxTheta`` ->
      +z pole face, ``theta <= minTheta`` -> -z pole face).  ``poles=False``
      skips the pole logic and gives the private ``closestFace(v, coeff, xcand,
      eps1, eps2)`` overload (``xcand`` empty); both overloads select the same
      winner whenever the pole logic does not fire.

    Among candidate faces the winner is the *valid* face of minimum
    ``dpoint2tri`` distance; exact (float32) distance ties go to the lowest face
    index.  A face is *valid* if the query is not behind its plane
    (``r1 >= r0 - 1e-5``) and the query, scaled radially onto the face plane,
    has all barycentric coefficients ``>= -1e-5``.  The returned coefficients
    are those of that radial projection (unsnapped, may be slightly negative,
    sum ~1).  If nothing is found: face ``-1`` and bary ``NaN`` (the C++ leaves
    ``coeff`` stale; callers must test ``face == -1``).

``SphereAABB.candidates(points, eps1=0.0, eps2=0.0)``
    ``-> (offsets int64 (n+1,), faces int32)`` -- ``AABB_Sphere::searchTree``
    for every point (sorted, unique per point; no pole faces).

``SphereAABB.update(vertices, *, refresh_face_normals=False)``
    ``AABB_Sphere::update()`` after the mesh vertices moved: the tree topology is
    kept (as upstream), bounding boxes are recomputed and the pole faces are
    re-detected.  Upstream callers (SphericalRemesh, HSD) change vertices with
    ``Vertex::setVertex`` and do **not** call ``Mesh::updateNormal``, so the
    pole detection sees the *stale* face normals of the original mesh; that is
    the default here.  ``refresh_face_normals=True`` recomputes them first.

``barycentric(a, b, c, p, dtype=np.float32)``
    ``Coordinate::cart2bary`` (``err = 0``) vectorised; ``dtype=np.float64``
    gives the ``double`` overload.

``cart2sph(points)``
    ``Coordinate::cart2sph`` (float32): ``(phi, theta)`` with phi the azimuth
    and ``theta`` the *elevation* (not the polar angle).
"""

from __future__ import annotations

import math

import numpy as np
from numba import get_num_threads, njit, prange

__all__ = ["SphereAABB", "barycentric", "cart2sph", "closest_face"]

F = np.float32
_PI = 3.141592653589793238462643
_TWO_PI = 2.0 * _PI
_TWO_PI_F32 = np.float32(_TWO_PI)
_RAD0 = np.float32(0.0174533)  # 1 degree, as the C++ literal converted to float
_N_TRIALS = 9  # trial = 0..8


# --------------------------------------------------------------------------- #
# float32 scalar kernels (ports of Geom.cpp)
# --------------------------------------------------------------------------- #
@njit(cache=True, inline="always")
def _atan2f(y, x):
    """float32 ``atan2f`` of the platform libm (the ONLY non-IEEE-exact operation in this module).

    Everything else here is ``+ - * / sqrt`` on float32, which is correctly rounded and, with
    no fast-math / FMA contraction (nothing in this module sets ``fastmath``; LLVM only fuses
    with ``contract`` flags or ``fmuladd``, which Numba does not emit), identical on every
    platform.  ``atan2f`` is not: results differ by 1 ulp between Apple libm, glibc and the
    Windows UCRT (and Apple returns pi - 1 ulp for ``atan2f(+-0, -x)``).  The (phi, theta) values
    feed the box tests and the ``maxphi - minphi > pi`` wrap decision, so the discrete outcome can
    differ between platforms for queries within an ulp of a box bound.  Kept as the platform libm
    on purpose (oracle semantics); tests treat such cases as ties
    (docs/ports/aabb-sphere.md, "Determinism").
    """
    return F(math.atan2(y, x))


@njit(cache=True, inline="always")
def _cart2sph(x, y, z):
    d = x * x + y * y
    if d == F(0):
        phi = F(0)
    else:
        phi = _atan2f(y, x)
    if z == F(0):
        theta = F(0)
    else:
        theta = _atan2f(z, F(math.sqrt(d)))
    return phi, theta


@njit(cache=True)
def _cart2sph_all(p):
    n = p.shape[0]
    phi = np.empty(n, np.float32)
    theta = np.empty(n, np.float32)
    for i in range(n):
        phi[i], theta[i] = _cart2sph(p[i, 0], p[i, 1], p[i, 2])
    return phi, theta


def _make_bary(T):
    @njit(cache=True, inline="always")
    def bary(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz):
        one = T(1)
        # N = (B-A) x (C-A)
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        nn = T(math.sqrt(nx * nx + ny * ny + nz * nz))
        abc = (nx * nx + ny * ny + nz * nz) / nn
        if nn > T(0):  # N.unit()
            nx = nx / nn
            ny = ny / nn
            nz = nz / nn
        # (B-P) x (C-P) . N / ABC
        ux, uy, uz = bx - px, by - py, bz - pz
        vx, vy, vz = cx - px, cy - py, cz - pz
        c0 = ((uy * vz - uz * vy) * nx + (uz * vx - ux * vz) * ny + (ux * vy - uy * vx) * nz) / abc
        # (C-P) x (A-P) . N / ABC
        ux, uy, uz = cx - px, cy - py, cz - pz
        vx, vy, vz = ax - px, ay - py, az - pz
        c1 = ((uy * vz - uz * vy) * nx + (uz * vx - ux * vz) * ny + (ux * vy - uy * vx) * nz) / abc
        c2 = one - c0 - c1
        return c0, c1, c2

    return bary


_bary32 = _make_bary(np.float32)
_bary64 = _make_bary(np.float64)


@njit(cache=True)
def _dpoint2tri(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz):  # noqa: C901
    """``Coordinate::dpoint2tri(t0,t1,t2,p0)`` (Eberly region tree), float32."""
    zero, one, two = F(0), F(1), F(2)
    e0x, e0y, e0z = bx - ax, by - ay, bz - az
    e1x, e1y, e1z = cx - ax, cy - ay, cz - az
    dx, dy, dz = ax - px, ay - py, az - pz
    a = e0x * e0x + e0y * e0y + e0z * e0z
    b = e0x * e1x + e0y * e1y + e0z * e1z
    c = e1x * e1x + e1y * e1y + e1z * e1z
    d = e0x * dx + e0y * dy + e0z * dz
    e = e1x * dx + e1y * dy + e1z * dz
    f = dx * dx + dy * dy + dz * dz

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e
    sqr = zero

    if (s + t) <= det:
        if s < zero:
            if t < zero:
                # region 4
                if d < zero:
                    t = zero
                    if -d >= a:
                        s = one
                        sqr = a + two * d + f
                    else:
                        s = -d / a
                        sqr = d * s + f
                else:
                    s = zero
                    if e >= zero:
                        t = zero
                        sqr = f
                    else:
                        if -e >= c:
                            t = one
                            sqr = c + two * e + f
                        else:
                            t = -e / c
                            sqr = e * t + f
            else:
                # region 3
                s = zero
                if e >= zero:
                    t = zero
                    sqr = f
                else:
                    if -e >= c:
                        t = one
                        sqr = c + two * e + f
                    else:
                        t = -e / c
                        sqr = e * t + f
        else:
            if t < zero:
                # region 5
                t = zero
                if d >= zero:
                    s = zero
                    sqr = f
                else:
                    if -d >= a:
                        s = one
                        sqr = a + two * d + f
                    else:
                        s = -d / a
                        sqr = d * s + f
            else:
                # region 0
                inv = one / det
                s = s * inv
                t = t * inv
                sqr = s * (a * s + b * t + two * d) + t * (b * s + c * t + two * e) + f
    else:
        if s < zero:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - two * b + c
                if numer >= denom:
                    s = one
                    t = zero
                    sqr = a + two * d + f
                else:
                    s = numer / denom
                    t = one - s
                    sqr = s * (a * s + b * t + two * d) + t * (b * s + c * t + two * e) + f
            else:
                s = zero
                if tmp1 <= zero:
                    t = one
                    sqr = c + two * e + f
                else:
                    if e >= zero:
                        t = zero
                        sqr = f
                    else:
                        t = -e / c
                        sqr = e * t + f
        else:
            if t < zero:
                # region 6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - two * b + c
                    if numer >= denom:
                        t = one
                        s = zero
                        sqr = c + two * e + f
                    else:
                        t = numer / denom
                        s = one - t
                        sqr = s * (a * s + b * t + two * d) + t * (b * s + c * t + two * e) + f
                else:
                    t = zero
                    if tmp1 <= zero:
                        s = one
                        sqr = a + two * d + f
                    else:
                        if d >= zero:
                            s = zero
                            sqr = f
                        else:
                            s = -d / a
                            sqr = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= zero:
                    s = zero
                    t = one
                    sqr = c + two * e + f
                else:
                    denom = a - two * b + c
                    if numer >= denom:
                        s = one
                        t = zero
                        sqr = a + two * d + f
                    else:
                        s = numer / denom
                        t = one - s
                        sqr = s * (a * s + b * t + two * d) + t * (b * s + c * t + two * e) + f
    if sqr < zero:
        sqr = zero
    return F(math.sqrt(sqr))


# --------------------------------------------------------------------------- #
# tree construction (AABB_Sphere::boundingBox(face), construction, boundingBox(node))
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _face_boxes(verts, faces):
    """Per-face (phi0, phi1, theta0, theta1) boxes and the wrap-around max phi.

    Returns ``(boxes float32 (m,4), max_wrap_phi float32)``; ``max_wrap_phi`` is
    the largest ``maxphi`` among faces crossing the +-pi cut, or ``-inf``.
    """
    n = verts.shape[0]
    m = faces.shape[0]
    ph = np.empty(n, np.float32)
    th = np.empty(n, np.float32)
    for i in range(n):
        ph[i], th[i] = _cart2sph(verts[i, 0], verts[i, 1], verts[i, 2])
    boxes = np.empty((m, 4), np.float32)
    wrap = F(-np.inf)
    for i in range(m):
        minphi = F(_TWO_PI)
        maxphi = F(-_TWO_PI)
        mint = F(_TWO_PI)
        maxt = F(-_TWO_PI)
        for k in range(3):
            v = faces[i, k]
            p = ph[v]
            t = th[v]
            if minphi > p:
                minphi = p
            if maxphi < p:
                maxphi = p
            if mint > t:
                mint = t
            if maxt < t:
                maxt = t
        if np.float64(maxphi - minphi) > _PI:
            minphi = F(np.float64(minphi) + _TWO_PI)
            tmp = minphi
            minphi = maxphi
            maxphi = tmp
            if wrap < maxphi:
                wrap = maxphi
        boxes[i, 0] = minphi
        boxes[i, 1] = maxphi
        boxes[i, 2] = mint
        boxes[i, 3] = maxt
    return boxes, wrap


@njit(cache=True)
def _wrap32(x):
    """Wrap an int64 to int32 (the C++ ``int`` overflow of ``na*na + nb*nb``)."""
    x = x & 0xFFFFFFFF
    if x >= 0x80000000:
        x -= 0x100000000
    return x


@njit(cache=True)
def _build_tree(boxes):
    """Port of ``AABB_Sphere::construction`` (same leaf grouping as upstream)."""
    m = boxes.shape[0]
    perm = np.arange(m).astype(np.int32)
    max_nodes = 2 * m + 1
    left = np.full(max_nodes, -1, np.int32)
    right = np.full(max_nodes, -1, np.int32)
    lo = np.zeros(max_nodes, np.int32)
    cnt = np.zeros(max_nodes, np.int32)
    stk_node = np.empty(max_nodes, np.int32)
    stk_lo = np.empty(max_nodes, np.int32)
    stk_hi = np.empty(max_nodes, np.int32)
    s_l = np.empty(m, np.int32)
    s_r = np.empty(m, np.int32)
    s_lr = np.empty(m, np.int32)
    nn = 1
    sp = 0
    stk_node[0] = 0
    stk_lo[0] = 0
    stk_hi[0] = m
    sp = 1
    while sp > 0:
        sp -= 1
        node = stk_node[sp]
        a0 = stk_lo[sp]
        a1 = stk_hi[sp]
        n = a1 - a0
        lo[node] = a0
        cnt[node] = n
        if n <= 1:
            continue
        if n == 2:
            # construction(): a 2-element candidate list is always split into two leaves
            nl = 1
            nr = 1
            lid = nn
            rid = nn + 1
            nn += 2
            left[node] = lid
            right[node] = rid
            stk_node[sp] = lid
            stk_lo[sp] = a0
            stk_hi[sp] = a0 + 1
            sp += 1
            stk_node[sp] = rid
            stk_lo[sp] = a0 + 1
            stk_hi[sp] = a1
            sp += 1
            continue
        cdim = 0
        cthr = F(0)
        score = F(-1)
        for dim in range(2):
            data = np.empty(2 * n, np.float32)
            for i in range(n):
                fi = perm[a0 + i]
                data[2 * i] = boxes[fi, dim * 2 + 0]
                data[2 * i + 1] = boxes[fi, dim * 2 + 1]
            data.sort()
            thr = data[(2 * n) // 2]
            na = 0
            nb = 0
            nc = 0
            for i in range(n):
                fi = perm[a0 + i]
                mn = boxes[fi, dim * 2 + 0]
                mx = boxes[fi, dim * 2 + 1]
                if thr >= mx:
                    na += 1
                elif thr <= mn:
                    nb += 1
                else:
                    nc += 1
            if (na > 0 and nb > 0) or nc > 0:
                cand_score = F(na) * F(na) + F(nb) * F(nb)
                if score < cand_score:
                    cdim = dim
                    cthr = thr
                    score = F(_wrap32(np.int64(na) * na + np.int64(nb) * nb))
        nl = 0
        nr = 0
        nlr = 0
        for i in range(n):
            fi = perm[a0 + i]
            mn = boxes[fi, cdim * 2 + 0]
            mx = boxes[fi, cdim * 2 + 1]
            if cthr >= mx:
                s_l[nl] = fi
                nl += 1
            elif cthr <= mn:
                s_r[nr] = fi
                nr += 1
            else:
                s_lr[nlr] = fi
                nlr += 1
        for i in range(nlr):
            if nl < nr:
                s_l[nl] = s_lr[i]
                nl += 1
            else:
                s_r[nr] = s_lr[i]
                nr += 1
        if nl == 0 or nr == 0:
            continue  # leaf holding the whole candidate list
        for i in range(nl):
            perm[a0 + i] = s_l[i]
        for i in range(nr):
            perm[a0 + nl + i] = s_r[i]
        lid = nn
        rid = nn + 1
        nn += 2
        left[node] = lid
        right[node] = rid
        stk_node[sp] = lid
        stk_lo[sp] = a0
        stk_hi[sp] = a0 + nl
        sp += 1
        stk_node[sp] = rid
        stk_lo[sp] = a0 + nl
        stk_hi[sp] = a1
        sp += 1
    return perm, left[:nn].copy(), right[:nn].copy(), lo[:nn].copy(), cnt[:nn].copy()


@njit(cache=True)
def _node_boxes(boxes, perm, left, right, lo, cnt):
    """``AABB_Sphere::boundingBox(node)``: leaf = union of its faces, inner = union of children."""
    nn = left.shape[0]
    nb = np.empty((nn, 4), np.float32)  # phi0, phi1, theta0, theta1
    for node in range(nn - 1, -1, -1):
        if left[node] < 0:
            fi = perm[lo[node]]
            p0 = boxes[fi, 0]
            p1 = boxes[fi, 1]
            t0 = boxes[fi, 2]
            t1 = boxes[fi, 3]
            for k in range(1, cnt[node]):
                fi = perm[lo[node] + k]
                p0 = p0 if p0 < boxes[fi, 0] else boxes[fi, 0]
                t0 = t0 if t0 < boxes[fi, 2] else boxes[fi, 2]
                p1 = p1 if p1 > boxes[fi, 1] else boxes[fi, 1]
                t1 = t1 if t1 > boxes[fi, 3] else boxes[fi, 3]
            nb[node, 0] = p0
            nb[node, 1] = p1
            nb[node, 2] = t0
            nb[node, 3] = t1
        else:
            lc = left[node]
            rc = right[node]
            nb[node, 0] = nb[lc, 0] if nb[lc, 0] < nb[rc, 0] else nb[rc, 0]
            nb[node, 2] = nb[lc, 2] if nb[lc, 2] < nb[rc, 2] else nb[rc, 2]
            nb[node, 1] = nb[lc, 1] if nb[lc, 1] > nb[rc, 1] else nb[rc, 1]
            nb[node, 3] = nb[lc, 3] if nb[lc, 3] > nb[rc, 3] else nb[rc, 3]
    return nb


@njit(cache=True)
def _face_normals(verts, faces):
    """``Face::updateFaceNormal``: unit((b-a) x (c-b)) in float32."""
    m = faces.shape[0]
    out = np.empty((m, 3), np.float32)
    for i in range(m):
        a = verts[faces[i, 0]]
        b = verts[faces[i, 1]]
        c = verts[faces[i, 2]]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - b[0], c[1] - b[1], c[2] - b[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        s = F(math.sqrt(nx * nx + ny * ny + nz * nz))
        if s > F(0):
            nx = nx / s
            ny = ny / s
            nz = nz / s
        out[i, 0] = nx
        out[i, 1] = ny
        out[i, 2] = nz
    return out


@njit(cache=True)
def _find_poles(verts, faces, fnormals):
    """Pole part of ``AABB_Sphere::update``. Returns (pole0, pole1, minTheta, maxTheta)."""
    pole0 = -1
    pole1 = -1
    min_theta = F(0)
    max_theta = F(0)
    zero = F(0)
    i = 0
    m = faces.shape[0]
    while i < m and (pole0 == -1 or pole1 == -1):
        nx, ny, nz = fnormals[i, 0], fnormals[i, 1], fnormals[i, 2]
        a = verts[faces[i, 0]]
        b = verts[faces[i, 1]]
        c = verts[faces[i, 2]]
        # N * (0,0,1) and N * (0,0,-1)
        d0 = nx * zero + ny * zero + nz * F(1)
        d1 = nx * zero + ny * zero + nz * F(-1)
        if np.float64(d0) > 0.98:
            c0, c1, c2 = _bary32(
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], zero, zero, F(1)
            )
            if c0 >= zero and c1 >= zero and c2 >= zero:
                dist = _dpoint2tri(
                    a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], zero, zero, F(1)
                )
                if np.float64(dist) < 1e-1:
                    pole0 = i
                    _, t = _cart2sph(a[0], a[1], a[2])
                    max_theta = t
                    _, t = _cart2sph(b[0], b[1], b[2])
                    max_theta = t if max_theta > t else max_theta
                    _, t = _cart2sph(c[0], c[1], c[2])
                    max_theta = t if max_theta > t else max_theta
        if np.float64(d1) > 0.98:
            c0, c1, c2 = _bary32(
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], zero, zero, F(-1)
            )
            if c0 >= zero and c1 >= zero and c2 >= zero:
                dist = _dpoint2tri(
                    a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], zero, zero, F(-1)
                )
                if np.float64(dist) < 1e-1:
                    pole1 = i
                    _, t = _cart2sph(a[0], a[1], a[2])
                    min_theta = t
                    _, t = _cart2sph(b[0], b[1], b[2])
                    min_theta = t if min_theta < t else min_theta
                    _, t = _cart2sph(c[0], c[1], c[2])
                    min_theta = t if min_theta < t else min_theta
        i += 1
    return pole0, pole1, min_theta, max_theta


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
@njit(cache=True, inline="always")
def _eval_face(verts, faces, fi, vx, vy, vz):
    """Validity test + distance of one candidate (body of the closestFace loops)."""
    ia, ib, ic = faces[fi, 0], faces[fi, 1], faces[fi, 2]
    ax, ay, az = verts[ia, 0], verts[ia, 1], verts[ia, 2]
    bx, by, bz = verts[ib, 0], verts[ib, 1], verts[ib, 2]
    cx, cy, cz = verts[ic, 0], verts[ic, 1], verts[ic, 2]
    nan = F(np.nan)
    # N = unit((b-a) x (c-b))   [Vector(a,b).cross(Vector(b,c)).unit()]
    ux, uy, uz = bx - ax, by - ay, bz - az
    wx, wy, wz = cx - bx, cy - by, cz - bz
    nx = uy * wz - uz * wy
    ny = uz * wx - ux * wz
    nz = ux * wy - uy * wx
    s = F(math.sqrt(nx * nx + ny * ny + nz * nz))
    if s > F(0):
        nx = nx / s
        ny = ny / s
        nz = nz / s
    r0 = ax * nx + ay * ny + az * nz
    r1 = vx * nx + vy * ny + vz * nz
    eps = F(-1e-5)
    if r1 < r0 + eps:
        return False, nan, nan, nan, nan
    k = r0 / r1
    c0, c1, c2 = _bary32(ax, ay, az, bx, by, bz, cx, cy, cz, vx * k, vy * k, vz * k)
    if c0 >= eps and c1 >= eps and c2 >= eps:
        dist = _dpoint2tri(ax, ay, az, bx, by, bz, cx, cy, cz, vx, vy, vz)
        return True, dist, c0, c1, c2
    return False, nan, nan, nan, nan


@njit(cache=True)
def _search_one(
    verts,
    faces,
    perm,
    left,
    right,
    lo,
    cnt,
    nbox,
    max_phi,
    phi,
    theta,
    eps1,
    eps2,
    vx,
    vy,
    vz,
    stack,
    best_d,
    best_i,
    best_c,
):
    """Walk the tree for one (phi, theta) and evaluate every candidate face.

    ``best_*`` are updated in place: ``best_d`` (1,), ``best_i`` (1,), ``best_c`` (3,).
    """
    for rep in range(2):
        if rep == 0:
            ph = phi
        else:
            if not (np.float64(phi) + _TWO_PI <= np.float64(max_phi)):
                break
            ph = F(np.float64(phi) + _TWO_PI)
        sp = 0
        stack[0] = 0
        sp = 1
        while sp > 0:
            sp -= 1
            node = stack[sp]
            lc = left[node]
            if lc < 0:
                for k in range(cnt[node]):
                    fi = perm[lo[node] + k]
                    ok, dist, c0, c1, c2 = _eval_face(verts, faces, fi, vx, vy, vz)
                    if ok:
                        if dist == best_d[0] and best_i[0] != fi:
                            best_c[3] = F(1)  # exact tie between two different faces
                        if dist < best_d[0] or (dist == best_d[0] and best_i[0] > fi):
                            if dist < best_d[0]:
                                best_c[3] = F(0)
                            best_i[0] = fi
                            best_d[0] = dist
                            best_c[0] = c0
                            best_c[1] = c1
                            best_c[2] = c2
                continue
            rc = right[node]
            # push right first so that left is visited first (order is irrelevant for the result)
            if (
                ph >= nbox[rc, 0] - eps1
                and theta >= nbox[rc, 2] - eps2
                and ph <= nbox[rc, 1] + eps1
                and theta <= nbox[rc, 3] + eps2
            ):
                stack[sp] = rc
                sp += 1
            if (
                ph >= nbox[lc, 0] - eps1
                and theta >= nbox[lc, 2] - eps2
                and ph <= nbox[lc, 1] + eps1
                and theta <= nbox[lc, 3] + eps2
            ):
                stack[sp] = lc
                sp += 1


@njit(cache=True)
def _closest_one(
    verts,
    faces,
    perm,
    left,
    right,
    lo,
    cnt,
    nbox,
    max_phi,
    pole0,
    pole1,
    min_theta,
    max_theta,
    vx,
    vy,
    vz,
    eps1,
    eps2,
    escalate,
    use_poles,
    stack,
    best_d,
    best_i,
    best_c,
):
    phi, theta = _cart2sph(vx, vy, vz)
    rad = _RAD0
    ntrial = _N_TRIALS if escalate else 0
    for trial in range(-1, ntrial):
        if trial == -1:
            e1 = eps1
            e2 = eps2
        else:
            e1 = rad
            e2 = rad
        best_i[0] = -1
        best_d[0] = F(3.4028234663852886e38)
        best_c[3] = F(0)
        _search_one(
            verts,
            faces,
            perm,
            left,
            right,
            lo,
            cnt,
            nbox,
            max_phi,
            phi,
            theta,
            e1,
            e2,
            vx,
            vy,
            vz,
            stack,
            best_d,
            best_i,
            best_c,
        )
        if trial == -1 and use_poles and e1 == F(0) and pole0 != -1 and pole1 != -1:
            if theta >= max_theta:
                ok, dist, c0, c1, c2 = _eval_face(verts, faces, pole0, vx, vy, vz)
                if ok:
                    if dist == best_d[0] and best_i[0] != pole0:
                        best_c[3] = F(1)
                    if dist < best_d[0] or (dist == best_d[0] and best_i[0] > pole0):
                        if dist < best_d[0]:
                            best_c[3] = F(0)
                        best_i[0] = pole0
                        best_d[0] = dist
                        best_c[0] = c0
                        best_c[1] = c1
                        best_c[2] = c2
            if theta <= min_theta:
                ok, dist, c0, c1, c2 = _eval_face(verts, faces, pole1, vx, vy, vz)
                if ok:
                    if dist == best_d[0] and best_i[0] != pole1:
                        best_c[3] = F(1)
                    if dist < best_d[0] or (dist == best_d[0] and best_i[0] > pole1):
                        if dist < best_d[0]:
                            best_c[3] = F(0)
                        best_i[0] = pole1
                        best_d[0] = dist
                        best_c[0] = c0
                        best_c[1] = c1
                        best_c[2] = c2
        if best_i[0] != -1:
            break
        if trial >= 0:
            rad = rad * F(2)
    return best_i[0]


@njit(cache=True, parallel=True)
def _closest_all(
    verts,
    faces,
    perm,
    left,
    right,
    lo,
    cnt,
    nbox,
    max_phi,
    pole0,
    pole1,
    min_theta,
    max_theta,
    pts,
    eps1,
    eps2,
    escalate,
    use_poles,
    nchunk,
    depth,
):
    n = pts.shape[0]
    idx = np.empty(n, np.int32)
    bary = np.empty((n, 3), np.float32)
    ties = np.zeros(n, np.bool_)
    per = (n + nchunk - 1) // nchunk
    for ch in prange(nchunk):
        stack = np.empty(depth, np.int32)
        best_d = np.empty(1, np.float32)
        best_i = np.empty(1, np.int64)
        best_c = np.empty(4, np.float32)
        for q in range(ch * per, min(n, (ch + 1) * per)):
            best_c[0] = np.nan
            best_c[1] = np.nan
            best_c[2] = np.nan
            r = _closest_one(
                verts,
                faces,
                perm,
                left,
                right,
                lo,
                cnt,
                nbox,
                max_phi,
                pole0,
                pole1,
                min_theta,
                max_theta,
                pts[q, 0],
                pts[q, 1],
                pts[q, 2],
                eps1,
                eps2,
                escalate,
                use_poles,
                stack,
                best_d,
                best_i,
                best_c,
            )
            idx[q] = r
            ties[q] = best_c[3] > F(0)
            if r == -1:
                bary[q, 0] = np.nan
                bary[q, 1] = np.nan
                bary[q, 2] = np.nan
            else:
                bary[q, 0] = best_c[0]
                bary[q, 1] = best_c[1]
                bary[q, 2] = best_c[2]
    return idx, bary, ties


@njit(cache=True)
def _candidates_all(perm, left, right, lo, cnt, nbox, max_phi, pts, eps1, eps2, depth):
    n = pts.shape[0]
    offsets = np.zeros(n + 1, np.int64)
    out = []
    stack = np.empty(depth, np.int32)
    for q in range(n):
        phi, theta = _cart2sph(pts[q, 0], pts[q, 1], pts[q, 2])
        mine = []
        for rep in range(2):
            if rep == 0:
                ph = phi
            else:
                if not (np.float64(phi) + _TWO_PI <= np.float64(max_phi)):
                    break
                ph = F(np.float64(phi) + _TWO_PI)
            sp = 1
            stack[0] = 0
            while sp > 0:
                sp -= 1
                node = stack[sp]
                lc = left[node]
                if lc < 0:
                    for k in range(cnt[node]):
                        mine.append(perm[lo[node] + k])
                    continue
                rc = right[node]
                if (
                    ph >= nbox[rc, 0] - eps1
                    and theta >= nbox[rc, 2] - eps2
                    and ph <= nbox[rc, 1] + eps1
                    and theta <= nbox[rc, 3] + eps2
                ):
                    stack[sp] = rc
                    sp += 1
                if (
                    ph >= nbox[lc, 0] - eps1
                    and theta >= nbox[lc, 2] - eps2
                    and ph <= nbox[lc, 1] + eps1
                    and theta <= nbox[lc, 3] + eps2
                ):
                    stack[sp] = lc
                    sp += 1
        arr = np.unique(np.array(mine, dtype=np.int32)) if len(mine) > 0 else np.empty(0, np.int32)
        offsets[q + 1] = offsets[q] + arr.shape[0]
        for v in arr:
            out.append(v)
    return offsets, np.array(out, dtype=np.int32) if len(out) > 0 else np.empty(0, np.int32)


@njit(cache=True)
def _tree_depth(left, right):
    nn = left.shape[0]
    depth = np.zeros(nn, np.int32)
    best = 1
    for node in range(nn):
        if left[node] >= 0:
            depth[left[node]] = depth[node] + 1
            depth[right[node]] = depth[node] + 1
        if depth[node] + 1 > best:
            best = depth[node] + 1
    return best


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _as_points(points) -> np.ndarray:
    p = np.ascontiguousarray(points, dtype=np.float32)
    if p.ndim == 1 and p.shape[0] == 3:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    return p


def cart2sph(points) -> tuple[np.ndarray, np.ndarray]:
    """``Coordinate::cart2sph`` (float32): azimuth ``phi`` and *elevation* ``theta``."""
    # Same kernel (and libm ``atan2f``) as the tree/query path; NumPy's float32 arctan2 is a
    # different implementation and would disagree with the decisions taken inside the tree.
    return _cart2sph_all(_as_points(points))


def barycentric(a, b, c, p, dtype=np.float32) -> np.ndarray:
    """Barycentric coefficients of ``p`` w.r.t. triangle ``(a, b, c)`` (``cart2bary``, err=0).

    ``a, b, c, p`` broadcast to ``(n, 3)``.  The coefficients are those of the
    *plane* through the triangle: ``p`` is not projected first, so for an
    off-plane ``p`` they correspond to the orthogonal projection onto the plane
    (the triple-product formula divides by ``|N|``).  They sum to 1 by
    construction (``c2 = 1 - c0 - c1``).  ``dtype=np.float32`` mirrors
    ``cart2bary(const float*...)``; ``np.float64`` the ``double`` overload.
    Returns ``(n, 3)`` of ``dtype``.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    arrs = np.broadcast_arrays(*[np.atleast_2d(np.asarray(x, dtype=dtype)) for x in (a, b, c, p)])
    a, b, c, p = (np.ascontiguousarray(x) for x in arrs)
    out = np.empty((a.shape[0], 3), dtype)
    _bary_all(a, b, c, p, out, dtype == np.dtype(np.float32))
    return out


@njit(cache=True)
def _bary_all(a, b, c, p, out, single):
    for i in range(a.shape[0]):
        if single:
            r = _bary32(
                a[i, 0],
                a[i, 1],
                a[i, 2],
                b[i, 0],
                b[i, 1],
                b[i, 2],
                c[i, 0],
                c[i, 1],
                c[i, 2],
                p[i, 0],
                p[i, 1],
                p[i, 2],
            )
        else:
            r = _bary64(
                a[i, 0],
                a[i, 1],
                a[i, 2],
                b[i, 0],
                b[i, 1],
                b[i, 2],
                c[i, 0],
                c[i, 1],
                c[i, 2],
                p[i, 0],
                p[i, 1],
                p[i, 2],
            )
        out[i, 0] = r[0]
        out[i, 1] = r[1]
        out[i, 2] = r[2]


class SphereAABB:
    """Closest-face / barycentric lookup on a sphere-like triangle mesh (``AABB_Sphere``).

    Parameters
    ----------
    mesh_or_vertices
        A :class:`pybiscuit.mesh.Mesh` or an ``(N, 3)`` vertex array (cast to float32).
    faces
        ``(M, 3)`` integer array; required when ``mesh_or_vertices`` is an array.

    The mesh must be star-shaped around the origin (a unit sphere or a radially/
    tangentially perturbed one); queries are mapped to (azimuth, elevation) boxes.
    """

    def __init__(self, mesh_or_vertices, faces=None) -> None:
        if faces is None:
            if not hasattr(mesh_or_vertices, "faces"):
                raise ValueError("pass a Mesh, or vertices and faces")
            verts, fcs = mesh_or_vertices.vertices, mesh_or_vertices.faces
        else:
            verts, fcs = mesh_or_vertices, faces
        self._verts = np.ascontiguousarray(verts, dtype=np.float32)
        self._faces = np.ascontiguousarray(fcs, dtype=np.int32)
        if self._verts.ndim != 2 or self._verts.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if self._faces.ndim != 2 or self._faces.shape[1] != 3 or len(self._faces) == 0:
            raise ValueError("faces must have shape (M, 3), M > 0")
        if self._faces.min() < 0 or self._faces.max() >= len(self._verts):
            raise ValueError("face indices out of range")
        # initTree(): topology of the tree is fixed from the construction-time vertices
        boxes, _ = _face_boxes(self._verts, self._faces)
        self._perm, self._left, self._right, self._lo, self._cnt = _build_tree(boxes)
        self._depth = int(_tree_depth(self._left, self._right)) + 2
        # face normals of the construction-time mesh (Mesh::openFile -> updateNormal)
        self._fnormals = _face_normals(self._verts, self._faces)
        self._refresh()

    # -- properties ---------------------------------------------------------
    @property
    def n_faces(self) -> int:
        return int(self._faces.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self._left.shape[0])

    @property
    def vertices(self) -> np.ndarray:
        """Current float32 vertices (read-only view)."""
        v = self._verts.view()
        v.flags.writeable = False
        return v

    @property
    def poles(self) -> tuple[int, int]:
        """``(m_pole0, m_pole1)``: faces containing +z and -z (``-1`` when not found)."""
        return int(self._pole0), int(self._pole1)

    # -- update --------------------------------------------------------------
    def _refresh(self) -> None:
        boxes, wrap = _face_boxes(self._verts, self._faces)
        self._nbox = _node_boxes(boxes, self._perm, self._left, self._right, self._lo, self._cnt)
        # m_maxPhi: reset to 2*pi by updateTree(root); only a single-leaf tree can raise it.
        max_phi = _TWO_PI_F32
        if self.n_nodes == 1 and wrap > max_phi:
            max_phi = wrap
        self._max_phi = F(max_phi)
        p0, p1, mint, maxt = _find_poles(self._verts, self._faces, self._fnormals)
        self._pole0, self._pole1 = int(p0), int(p1)
        self._min_theta, self._max_theta = F(mint), F(maxt)

    def update(self, vertices, *, refresh_face_normals: bool = False) -> None:
        """``AABB_Sphere::update()`` for moved vertices (same topology).

        Keeps the tree structure, recomputes bounding boxes, ``maxPhi`` and the
        pole faces.  With ``refresh_face_normals=False`` (default, what
        SphericalRemesh/HSD effectively do) the pole detection uses the face
        normals of the vertices given at construction; ``True`` recomputes them.
        """
        v = np.ascontiguousarray(vertices, dtype=np.float32)
        if v.shape != self._verts.shape:
            raise ValueError("update() needs the same number of vertices")
        self._verts = v
        if refresh_face_normals:
            self._fnormals = _face_normals(self._verts, self._faces)
        self._refresh()

    # -- queries ---------------------------------------------------------------
    def closest_face(
        self, points, eps1=None, eps2=None, *, poles: bool = True, return_ties: bool = False
    ):
        """Closest face and barycentric coefficients for each query point.

        See the module docstring for the exact semantics of the two overloads.

        Returns
        -------
        face : int32 (n,)   (-1 when no face was found)
        bary : float32 (n, 3)   (NaN rows when face == -1)
        ties : bool (n,)        only with ``return_ties=True``: another face had exactly
            the same (float32) distance in the winning attempt, so the lowest-index
            rule decided.
        """
        pts = _as_points(points)
        if eps1 is None:
            if eps2 is not None:
                raise ValueError("eps2 given without eps1")
            e1 = e2 = F(0)
            escalate = True
        else:
            e1 = F(eps1)
            e2 = e1 if eps2 is None else F(eps2)
            escalate = False
        n = len(pts)
        nchunk = max(1, min(n, get_num_threads() * 8))
        if n == 0:
            res = (np.empty(0, np.int32), np.empty((0, 3), np.float32), np.empty(0, bool))
            return res if return_ties else res[:2]
        res = _closest_all(
            self._verts,
            self._faces,
            self._perm,
            self._left,
            self._right,
            self._lo,
            self._cnt,
            self._nbox,
            self._max_phi,
            self._pole0,
            self._pole1,
            self._min_theta,
            self._max_theta,
            pts,
            e1,
            e2,
            escalate,
            bool(poles),
            nchunk,
            self._depth,
        )
        return res if return_ties else res[:2]

    def candidates(self, points, eps1: float = 0.0, eps2: float = 0.0):
        """``AABB_Sphere::searchTree``: ``(offsets int64 (n+1,), faces int32)`` (sorted unique)."""
        pts = _as_points(points)
        return _candidates_all(
            self._perm,
            self._left,
            self._right,
            self._lo,
            self._cnt,
            self._nbox,
            self._max_phi,
            pts,
            F(eps1),
            F(eps2),
            self._depth,
        )


def closest_face(vertices, faces, points, eps1=None, eps2=None, *, poles: bool = True):
    """One-shot convenience: ``SphereAABB(vertices, faces).closest_face(points, ...)``."""
    return SphereAABB(vertices, faces).closest_face(points, eps1, eps2, poles=poles)
