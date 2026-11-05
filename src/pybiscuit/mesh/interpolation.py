"""Interpolation of a geodesic distance in a face: gw ``GW_TriangularInterpolation_*`` (row B2.7).

Peyre's gw toolbox (inside MeshLib) interpolates the per-vertex distance of a fast-marching run
inside one face, for the gradient-descent geodesic path (``GW_GeodesicPath::AddNewPoint``).  Three
classes exist and all three are compiled into MeshLib (``CMakeLists.txt``, ``SRC_GEOD_FILES``):

* :class:`LinearInterpolation` - ``GW_TriangularInterpolation_Linear``: barycentric value, constant
  gradient.
* :class:`QuadraticInterpolation` - ``GW_TriangularInterpolation_Quadratic``: a 2nd-order 2D
  polynomial through the face's three vertices and the three vertices across its edges, after
  unfolding them into the face plane.
* :class:`CubicInterpolation` - ``GW_TriangularInterpolation_Cubic``.  Unfinished upstream: its
  set-up, value and gradient are the quadratic ones (the gradient without the ``|det| > eps``
  guard), and it reports the *quadratic* type.  Reproduced as it behaves; its private helpers
  ``ComputeLocalGradient`` / ``SetLocalGradient`` / ``ComputeLocalBasis`` are never called and are
  not ported.

:class:`InterpolationMesh` holds what the classes read from the GW mesh (positions, face
vertices, face neighbours, distances) and mirrors ``GW_GeodesicFace``'s dispatch
(:meth:`InterpolationMesh.set_up`): the type is a mesh-wide setting, quadratic by default, and
``cubic`` is refused there exactly where upstream asserts.

``value`` / ``gradient`` take the face's vertices in any order ``(v0, v1, v2)`` - the caller's
local frame, as ``GW_GeodesicPath`` passes them - and a point
``x * v0 + y * v1 + (1 - x - y) * v2``.  The gradient is returned in that frame:
``grad = dx * e0 + dy * e1`` with ``e0 = (v0 - v2) / |v0 - v2|`` and
``e1 = (v1 - v2) / |v1 - v2|``.  See ``docs/ports/interpolation.md``.

Precision mirrors gw: vertex positions are float32 (MeshLib ``Mesh``) widened to double, all
arithmetic is double (``GW_Float``).  ``GW_VectorStatic`` details kept: ``v / f`` multiplies by
``1 / f`` (zero vector when ``f == 0``), ``Normalize`` returns ``(1, 0, 0)`` below ``1e-9``, dot
products and norms sum from 0 in component order.  Upstream quirks kept on purpose:

* a missing neighbour across edge 1 or 2 is replaced by the midpoint of ``V0`` and ``V1`` (the
  *position*) while its *value* is the mean of the edge's own end points;
* ``GW_Face::GetEdgeNumber(v1, v2)`` returns 0 when the edge is absent.

The 6x6 solve of ``GW_Maths::Fit2ndOrderPolynomial2D`` is LU decomposition by Crout's method with
implicit (row-scaled) partial pivoting, ties going to the last row, a zero pivot replaced by
``1e-9``, then forward and back substitution.  It is implemented here from that description; the
operation order is what makes the result bit-identical to MeshLib's.
"""

from __future__ import annotations

import math

import numpy as np

from .geodesic import _build_connectivity

__all__ = [
    "CUBIC",
    "LINEAR",
    "QUADRATIC",
    "CubicInterpolation",
    "InterpolationMesh",
    "LinearInterpolation",
    "QuadraticInterpolation",
    "fit_2nd_order_polynomial_2d",
]

LINEAR, QUADRATIC, CUBIC = "linear", "quadratic", "cubic"
GW_EPSILON = 1e-9


# --------------------------------------------------------------------------- #
# GW_VectorStatic arithmetic on 3-tuples of Python floats (IEEE double)
# --------------------------------------------------------------------------- #
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, f):
    return (a[0] * f, a[1] * f, a[2] * f)


def _dot(a, b):
    r = 0.0
    r += a[0] * b[0]
    r += a[1] * b[1]
    r += a[2] * b[2]
    return r


def _norm(a):
    return math.sqrt(_dot(a, a))


def _div(a, f):
    """``operator/``: multiply by the reciprocal; the zero vector when ``f == 0``."""
    if f != 0:
        return _scale(a, 1 / f)
    return (0.0, 0.0, 0.0)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _recip(f):
    """``1 / f`` with IEEE semantics (``inf`` / ``nan`` instead of ZeroDivisionError)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(1.0) / np.float64(f))


def _acos(x):
    """C ``acos``: NaN outside [-1, 1] (the unfolding of a missing neighbour can land there)."""
    return math.acos(x) if -1.0 <= x <= 1.0 else math.nan


def _normalize(a):
    n = _norm(a)
    if n < GW_EPSILON:
        return (1.0, 0.0, 0.0)
    return _div(a, n)


# --------------------------------------------------------------------------- #
# GW_Maths::Fit2ndOrderPolynomial2D
# --------------------------------------------------------------------------- #
def _lu_solve(a: list[list[float]], b: list[float]) -> None:
    """Solve ``a x = b`` in place (``b`` <- x; ``a`` is overwritten by its LU factors).

    Crout's method with implicit partial pivoting: each row is scaled by its largest absolute
    entry when choosing the pivot (ties to the later row), a zero pivot becomes ``GW_EPSILON``;
    forward substitution starts at the first non-zero right-hand side entry.
    """
    n = len(a)
    scale = [0.0] * n
    for i in range(n):
        big = 0.0
        for j in range(n):
            t = abs(a[i][j])
            if t > big:
                big = t
        scale[i] = 1.0 / big  # big == 0 is an upstream assert (singular matrix)
    perm = [0] * n
    for j in range(n):
        for i in range(j):
            s = a[i][j]
            for k in range(i):
                s -= a[i][k] * a[k][j]
            a[i][j] = s
        big = 0.0
        imax = j
        for i in range(j, n):
            s = a[i][j]
            for k in range(j):
                s -= a[i][k] * a[k][j]
            a[i][j] = s
            d = scale[i] * abs(s)
            if d >= big:
                big = d
                imax = i
        if j != imax:
            a[imax], a[j] = a[j], a[imax]
            scale[imax] = scale[j]
        perm[j] = imax
        if a[j][j] == 0.0:
            a[j][j] = GW_EPSILON
        if j != n - 1:
            d = 1.0 / a[j][j]
            for i in range(j + 1, n):
                a[i][j] *= d
    first = -1
    for i in range(n):
        ip = perm[i]
        s = b[ip]
        b[ip] = b[i]
        if first >= 0:
            for j in range(first, i):
                s -= a[i][j] * b[j]
        elif s:
            first = i
        b[i] = s
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= a[i][j] * b[j]
        b[i] = s / a[i][i]


def fit_2nd_order_polynomial_2d(points, values) -> np.ndarray:
    """Coefficients ``(1, x, y, xy, x^2, y^2)`` of the quadratic through 6 points (``GW_Maths``)."""
    m = []
    for i in range(6):
        x, y = float(points[i][0]), float(points[i][1])
        m.append([1.0, x, y, x * y, x * x, y * y])
    c = [float(v) for v in values]
    _lu_solve(m, c)
    return np.array(c)


# --------------------------------------------------------------------------- #
# the mesh state the interpolations read
# --------------------------------------------------------------------------- #
class InterpolationMesh:
    """Positions, faces, face neighbours and per-vertex distances of a GW geodesic mesh.

    ``dist`` is the distance field to interpolate (e.g. ``GeodesicSolver.dist``); it may be
    replaced at any time (``mesh.dist = ...``).  ``kind`` is the mesh-wide type used by
    :meth:`set_up` (``GW_GeodesicFace::SetTriangularInterpolationType``).
    """

    def __init__(self, vertices, faces=None, dist=None, *, kind: str = QUADRATIC):
        if faces is None:
            vertices, faces = vertices.vertices, vertices.faces
        # MeshLib stores float; the GW mesh widens each coordinate to double
        self.pos = np.ascontiguousarray(vertices, dtype=np.float32).astype(np.float64)
        self.vert = np.ascontiguousarray(faces, dtype=np.int32)
        n = self.pos.shape[0]
        if self.vert.size and (self.vert.min() < 0 or self.vert.max() >= n):
            raise ValueError("face index out of range")
        _, self.fnbr = _build_connectivity(self.vert, n)
        self.dist = np.zeros(n) if dist is None else np.asarray(dist, dtype=np.float64)
        self.kind = kind
        self._faces: dict[int, LinearInterpolation] = {}

    @property
    def kind(self) -> str:
        return self._kind

    @kind.setter
    def kind(self, kind: str) -> None:
        if kind not in _CLASSES:
            raise ValueError(f"kind must be one of {sorted(_CLASSES)}")
        self._kind = kind

    def position(self, v: int) -> tuple[float, float, float]:
        p = self.pos[v]
        return (float(p[0]), float(p[1]), float(p[2]))

    def distance(self, v: int) -> float:
        return float(self.dist[v])

    def third_vertex(self, f: int, a: int, b: int) -> int:
        """``GW_Face::GetVertex(v1, v2)``; vertex 0 of the face if ``a, b`` is not an edge of it."""
        row = self.vert[f]
        for i in range(3):
            if row[i] == a:
                if row[(i + 1) % 3] == b:
                    return int(row[(i + 2) % 3])
                if row[(i + 2) % 3] == b:
                    return int(row[(i + 1) % 3])
        return int(row[0])

    def set_up(self, f: int):
        """``GW_GeodesicFace::SetUpTriangularInterpolation``: (re)fit face ``f``, return its object.

        The object is kept per face and replaced when the mesh-wide type changed, as upstream.
        """
        if self._kind == CUBIC:
            # upstream: GW_ASSERT(GW_False) and a NULL interpolation (crash in release builds)
            raise ValueError(
                "GW_GeodesicFace cannot set up a cubic interpolation (upstream assert)"
            )
        obj = self._faces.get(f)
        if obj is None or obj.get_type() != self._kind:
            obj = _CLASSES[self._kind]()
            self._faces[f] = obj
        obj.set_up(self, f)
        return obj

    def gradient(
        self, f: int, v0: int, v1: int, v2: int, x: float, y: float
    ) -> tuple[float, float]:
        """``GW_GeodesicFace::ComputeGradient`` (after :meth:`set_up` on ``f``)."""
        obj = self._faces.get(f)
        if obj is None:
            raise ValueError(f"face {f} has no interpolation set up")
        return obj.gradient(self, v0, v1, v2, x, y)


# --------------------------------------------------------------------------- #
# the three interpolations
# --------------------------------------------------------------------------- #
class LinearInterpolation:
    """``GW_TriangularInterpolation_Linear``."""

    def get_type(self) -> str:
        return LINEAR

    def set_up(self, mesh: InterpolationMesh, f: int) -> None:
        """Nothing to fit."""

    def value(
        self, mesh: InterpolationMesh, v0: int, v1: int, v2: int, x: float, y: float
    ) -> float:
        return mesh.distance(v0) * x + mesh.distance(v1) * y + mesh.distance(v2) * (1 - x - y)

    def gradient(self, mesh: InterpolationMesh, v0: int, v1: int, v2: int, x: float, y: float):
        d0, d1, d2 = mesh.distance(v0), mesh.distance(v1), mesh.distance(v2)
        p2 = mesh.position(v2)
        e0 = _sub(mesh.position(v0), p2)
        e1 = _sub(mesh.position(v1), p2)
        l0 = _norm(e0)
        l1 = _norm(e1)
        e0 = _div(e0, l0)
        e1 = _div(e1, l1)
        dot = _dot(e0, e1)
        det = 1 - dot * dot
        gu = (d0 - d2) / l0
        gv = (d1 - d2) / l1
        r = _recip(det)  # det == 0 is an upstream assert
        return r * (gu - dot * gv), r * (-dot * gu + gv)


class QuadraticInterpolation(LinearInterpolation):
    """``GW_TriangularInterpolation_Quadratic``.

    After :meth:`set_up`: ``coeffs`` (``cst, X, Y, XY, X^2, Y^2`` in the face frame), the frame axes
    ``u``, ``v`` and its origin ``w`` (the face's third vertex).
    """

    _guard = True  # ComputeGradient returns (0, 0) when |det| <= GW_EPSILON

    def __init__(self):
        self.coeffs = np.zeros(6)
        self.u = self.v = self.w = (0.0, 0.0, 0.0)

    def get_type(self) -> str:
        return QUADRATIC

    def set_up(self, mesh: InterpolationMesh, f: int) -> None:
        iv0, iv1, iv2 = (int(i) for i in mesh.vert[f])
        nb = mesh.fnbr[f]
        iw0 = mesh.third_vertex(int(nb[0]), iv1, iv2) if nb[0] >= 0 else -1
        iw1 = mesh.third_vertex(int(nb[1]), iv0, iv2) if nb[1] >= 0 else -1
        iw2 = mesh.third_vertex(int(nb[2]), iv0, iv1) if nb[2] >= 0 else -1
        V0, V1, V2 = mesh.position(iv0), mesh.position(iv1), mesh.position(iv2)
        W0 = mesh.position(iw0) if iw0 >= 0 else _scale(_add(V1, V2), 0.5)
        W1 = mesh.position(iw1) if iw1 >= 0 else _scale(_add(V0, V1), 0.5)  # sic: V0+V1
        W2 = mesh.position(iw2) if iw2 >= 0 else _scale(_add(V0, V1), 0.5)
        # edges of the face and of the side triangles
        e0, e1, e2 = _sub(V0, V2), _sub(V1, V2), _sub(V1, V0)
        s0, s1, s2 = _sub(W0, V2), _sub(W1, V2), _sub(W2, V0)
        l0, l1, l2 = _norm(e0), _norm(e1), _norm(e2)
        m0, m1, m2 = _norm(s0), _norm(s1), _norm(s2)
        # orthonormal frame of the interpolation
        self.u = _div(e0, l0)
        self.v = _normalize(_cross(_cross(self.u, e1), self.u))
        self.w = V2
        # unfold the side vertices into the face plane
        a = _acos(_dot(e0, e1) / (l0 * l1))
        b = _acos(_dot(e1, s0) / (l1 * m0))
        c = _acos(_dot(e0, s1) / (l0 * m1))
        d = _acos(-_dot(e0, e2) / (l0 * l2))
        e = _acos(_dot(e2, s2) / (l2 * m2))
        points = (
            (l0, 0.0),
            (l1 * math.cos(a), l1 * math.sin(a)),
            (0.0, 0.0),
            (m0 * math.cos(a + b), m0 * math.sin(a + b)),
            (m1 * math.cos(c), -m1 * math.sin(c)),
            (l0 - m2 * math.cos(d + e), m2 * math.sin(d + e)),
        )
        val0, val1, val2 = mesh.distance(iv0), mesh.distance(iv1), mesh.distance(iv2)
        values = (
            val0,
            val1,
            val2,
            mesh.distance(iw0) if iw0 >= 0 else (val1 + val2) * 0.5,
            mesh.distance(iw1) if iw1 >= 0 else (val0 + val2) * 0.5,
            mesh.distance(iw2) if iw2 >= 0 else (val0 + val1) * 0.5,
        )
        self.coeffs = fit_2nd_order_polynomial_2d(points, values)

    def _frame(self, mesh, v0, v1, v2, x, y):
        """Passage matrix from the caller's frame to (u, v) and the point's (s, t)."""
        e = mesh.position(v2)
        e0 = _sub(mesh.position(v0), e)
        e1 = _sub(mesh.position(v1), e)
        trans = _sub(e, self.w)
        p00, p01 = _dot(e0, self.u), _dot(e1, self.u)
        p10, p11 = _dot(e0, self.v), _dot(e1, self.v)
        s = x * p00 + y * p01 + _dot(trans, self.u)
        t = x * p10 + y * p11 + _dot(trans, self.v)
        return e0, e1, p00, p01, p10, p11, s, t

    def value(self, mesh, v0, v1, v2, x, y) -> float:
        *_, s, t = self._frame(mesh, v0, v1, v2, x, y)
        c = [float(k) for k in self.coeffs]
        return c[0] + c[1] * s + c[2] * t + c[3] * s * t + c[4] * s * s + c[5] * t * t

    def gradient(self, mesh, v0, v1, v2, x, y):
        e0, e1, p00, p01, p10, p11, s, t = self._frame(mesh, v0, v1, v2, x, y)
        c = [float(k) for k in self.coeffs]
        gu = c[1] + c[3] * t + c[4] * 2 * s
        gv = c[2] + c[3] * s + c[5] * 2 * t
        det = p00 * p11 - p01 * p10
        if self._guard and not abs(det) > GW_EPSILON:
            return 0.0, 0.0
        r = _recip(det)
        return r * (p11 * gu - p01 * gv) * _norm(e0), r * (-p10 * gu + p00 * gv) * _norm(e1)


class CubicInterpolation(QuadraticInterpolation):
    """``GW_TriangularInterpolation_Cubic``: the quadratic fit, gradient without the det guard.

    ``get_type`` returns ``"quadratic"``, as upstream's ``GetType`` does.
    """

    _guard = False


_CLASSES = {
    LINEAR: LinearInterpolation,
    QUADRATIC: QuadraticInterpolation,
    CUBIC: CubicInterpolation,
}
