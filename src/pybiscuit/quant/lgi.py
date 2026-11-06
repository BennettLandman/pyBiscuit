"""Local gyrification index: port of ``Gyrification`` (LocalGyrificationIndex ``b9fa760``, v1.2.2).

Source: ``src/Gyrification.cpp`` / ``Gyrification.h`` / ``wrapper/{main.cpp,PARSE_ARGS.h}``; the
fast-marching kernels it calls live in :mod:`pybiscuit.quant._fastmarch`.  See
``docs/ports/lgi.md``.

Pipeline (``Gyrification::run``):

1. isotropic geodesic distance (and nearest-source id) from all sulcal and gyral curve vertices
   (``setupGDist``), or loaded from ``--inputMap``;
2. barycentric vertex areas of the surface and of the outer (hull) surface, hull area capped at
   the surface area (``precomputeArea``);
3. kernel-size adjustment by hull area (``--refHullArea``, default 77100) or cortex area;
4. per-vertex anisotropic metric from the distance field (``setupTensorEqualArea``), then for
   every vertex an area-limited anisotropic fast-marching front whose kernel gives
   ``LGI = (surface area) / (hull area)`` for each requested kernel size.

Mesh coordinates are float32 in MeshLib (``Vector`` arithmetic); the tensor code below keeps float32
where MeshLib does and doubles elsewhere.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

from ..io import read_surface
from ..mesh import Mesh
from ..mesh.curvature import _eig3
from ..mesh.geometry import vertex_areas, vertex_neighbor_lists
from . import _fastmarch as fm
from ._libm import DEFAULT_LIBM, LIBMS, acosf_select

__all__ = [
    "LgiResult",
    "curve_vertex_ids",
    "gyrification",
    "read_curves",
    "read_geodesic_map",
    "vertex_normals_f32",
    "write_geodesic_map",
    "write_lgi_maps",
]

F32 = np.float32
_FLT_MAX = float(np.finfo(np.float32).max)
_HEAP = {"libcxx": fm.HEAP_LIBCXX, "libstdcxx": fm.HEAP_LIBSTDCXX}


# --------------------------------------------------------------------------- #
# curve files
# --------------------------------------------------------------------------- #
def read_curves(path) -> list[np.ndarray]:
    """Curve vertex ids as ``Gyrification::load{S,G}curve[Bary]`` reads them.

    ``*.bary`` (TRACE with ``--bary``): ``n`` curves, each ``p`` then ``p`` lines ``id1 id2 coeff``;
    a point is kept only when ``id1 == id2`` or ``coeff == 1`` (i.e. it is a mesh vertex) and then
    contributes ``id1``.  Anything else is the plain format: one curve per line, vertex ids
    separated by spaces.
    """
    path = Path(path)
    text = path.read_text().splitlines()
    curves: list[np.ndarray] = []
    if path.name.endswith(".bary"):
        n = int(text[0].split()[0]) if text and text[0].split() else 0
        k = 1
        for _ in range(n):
            p = int(text[k].split()[0])
            k += 1
            ids = []
            for i in range(p):
                tok = text[k + i].split()
                id1, id2, coeff = int(tok[0]), int(tok[1]), np.float32(float(tok[2]))
                if id1 == id2 or coeff == 1:
                    ids.append(id1)
            k += p
            if ids:
                curves.append(np.asarray(ids, dtype=np.int64))
    else:
        for line in text:
            ids = [int(t) for t in line.split() if t.strip("\r")]
            if ids:
                curves.append(np.asarray(ids, dtype=np.int64))
    return curves


def curve_vertex_ids(*curve_sets) -> np.ndarray:
    """Sorted unique vertex ids of all curves (paths or lists of id arrays)."""
    ids = []
    for cs in curve_sets:
        if isinstance(cs, (str, os.PathLike)):
            cs = read_curves(cs)
        for c in cs:
            ids.append(np.asarray(c, dtype=np.int64).ravel())
    return np.unique(np.concatenate(ids)) if ids else np.empty(0, dtype=np.int64)


def write_geodesic_map(path, dist, source) -> None:
    """``Gyrification::saveGMap``: ``"%f %d"`` per vertex."""
    with open(path, "w") as fh:
        for d, s in zip(dist, source, strict=True):
            fh.write(f"{d:f} {s:d}\n")


def read_geodesic_map(path, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """``Gyrification::loadGMap``: distances are parsed as C ``float``."""
    a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if a.shape[0] != n_vertices:
        raise ValueError(f"{path}: {a.shape[0]} rows for {n_vertices} vertices")
    return a[:, 0].astype(np.float32).astype(np.float64), a[:, 1].astype(np.int64)


@njit(cache=True, error_model="numpy")
def _vertex_normals(v, faces, host):
    """``Mesh::updateNormal`` (as ``mesh.geometry.vertex_normals``) with a selectable ``acosf``.

    A copy of the brief 02 kernel whose only change is the ``acosf`` of :mod:`._libm` (``host``
    False: pinned fdlibm), so the angle weights need not depend on the host libm
    (docs/ports/lgi.md, "libm").
    """
    n = v.shape[0]
    acc = np.zeros((n, 3), dtype=np.float32)
    one = F32(1.0)
    for i in range(faces.shape[0]):
        i0, i1, i2 = faces[i, 0], faces[i, 1], faces[i, 2]
        e1x, e1y, e1z = v[i0, 0] - v[i1, 0], v[i0, 1] - v[i1, 1], v[i0, 2] - v[i1, 2]
        e2x, e2y, e2z = v[i1, 0] - v[i2, 0], v[i1, 1] - v[i2, 1], v[i1, 2] - v[i2, 2]
        e3x, e3y, e3z = v[i2, 0] - v[i0, 0], v[i2, 1] - v[i0, 1], v[i2, 2] - v[i0, 2]
        fx = e1y * e2z - e1z * e2y
        fy = e1z * e2x - e1x * e2z
        fz = e1x * e2y - e1y * e2x
        s = _fnorm(e1x, e1y, e1z)
        if s > 0:
            e1x, e1y, e1z = e1x / s, e1y / s, e1z / s
        s = _fnorm(e2x, e2y, e2z)
        if s > 0:
            e2x, e2y, e2z = e2x / s, e2y / s, e2z / s
        s = _fnorm(e3x, e3y, e3z)
        if s > 0:
            e3x, e3y, e3z = e3x / s, e3y / s, e3z / s
        e13 = e1x * -e3x + e1y * -e3y + e1z * -e3z
        e13 = min(max(e13, -one), one)
        e21 = e2x * -e1x + e2y * -e1y + e2z * -e1z
        e21 = min(max(e21, -one), one)
        e32 = e3x * -e2x + e3y * -e2y + e3z * -e2z
        e32 = min(max(e32, -one), one)
        w1 = acosf_select(e13, host)
        w2 = acosf_select(e21, host)
        w3 = acosf_select(e32, host)
        acc[i0, 0] += fx * w1
        acc[i0, 1] += fy * w1
        acc[i0, 2] += fz * w1
        acc[i1, 0] += fx * w2
        acc[i1, 1] += fy * w2
        acc[i1, 2] += fz * w2
        acc[i2, 0] += fx * w3
        acc[i2, 1] += fy * w3
        acc[i2, 2] += fz * w3
    for i in range(n):
        s = _fnorm(acc[i, 0], acc[i, 1], acc[i, 2])
        if s > 0:
            acc[i, 0] /= s
            acc[i, 1] /= s
            acc[i, 2] /= s
    return acc


def vertex_normals_f32(vertices, faces, libm: str = DEFAULT_LIBM) -> np.ndarray:
    """float32 vertex normals as MeshLib computes them; ``libm`` picks the ``acosf`` (``_libm``)."""
    if libm not in LIBMS:
        raise ValueError(f"libm must be one of {LIBMS}")
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    return _vertex_normals(v, np.ascontiguousarray(faces, dtype=np.int32), libm == "host")


# --------------------------------------------------------------------------- #
# setupTensorEqualArea
# --------------------------------------------------------------------------- #
@njit(cache=True, error_model="numpy", inline="always")
def _fnorm(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@njit(cache=True, error_model="numpy")
def _initial_tensor(v, indptr, indices, nrm, adj):
    """First loop of ``setupTensorEqualArea``: rank-2 tensor from the distance gradient."""
    n = v.shape[0]
    T = np.zeros((n, 9))
    zero = F32(0.0)
    for i in range(n):
        nx, ny, nz = nrm[i, 0], nrm[i, 1], nrm[i, 2]
        lam1 = _FLT_MAX
        lam2 = 0.0
        v2x, v2y, v2z = zero, zero, zero
        for j in range(indptr[i], indptr[i + 1]):
            k = indices[j]
            ex, ey, ez = v[k, 0] - v[i, 0], v[k, 1] - v[i, 1], v[k, 2] - v[i, 2]
            p = nx * ex + ny * ey + nz * ez
            dx, dy, dz = ex - nx * p, ey - ny * p, ez - nz * p
            dn = _fnorm(dx, dy, dz)
            t = abs(adj[i] - adj[k]) / np.float64(dn)
            if dn > 0:
                dx, dy, dz = dx / dn, dy / dn, dz / dn
            if lam2 < t:
                lam2 = t
                v2x, v2y, v2z = dx, dy, dz
        # V1 = N x V2, unit
        v1x = ny * v2z - nz * v2y
        v1y = nz * v2x - nx * v2z
        v1z = nx * v2y - ny * v2x
        s = _fnorm(v1x, v1y, v1z)
        if s > 0:
            v1x, v1y, v1z = v1x / s, v1y / s, v1z / s
        for j in range(indptr[i], indptr[i + 1]):
            k = indices[j]
            ex, ey, ez = v[k, 0] - v[i, 0], v[k, 1] - v[i, 1], v[k, 2] - v[i, 2]
            p = nx * ex + ny * ey + nz * ez
            dx, dy, dz = ex - nx * p, ey - ny * p, ez - nz * p
            dn = _fnorm(dx, dy, dz)
            t = abs(adj[i] - adj[k]) / np.float64(dn)
            if dn > 0:
                dx, dy, dz = dx / dn, dy / dn, dz / dn
            x = np.float64(v1x * dx + v1y * dy + v1z * dz) * t
            y = np.float64(v2x * dx + v2y * dy + v2z * dz) * t
            if y == 0:
                lam1 = t
                break
            if y == lam2:
                continue
            l1 = abs(x / math.sqrt(1 - y * y / lam2 / lam2))
            if lam1 > l1:
                lam1 = l1
        if lam1 == 0:
            lam1 = 1e-5
        s1 = lam1
        s2 = lam2
        a0, a1, a2 = np.float64(v1x), np.float64(v1y), np.float64(v1z)
        b0, b1, b2 = np.float64(v2x), np.float64(v2y), np.float64(v2z)
        T[i, 0] = s1 * a0 * a0 + s2 * b0 * b0
        T[i, 4] = s1 * a1 * a1 + s2 * b1 * b1
        T[i, 8] = s1 * a2 * a2 + s2 * b2 * b2
        T[i, 1] = T[i, 3] = s1 * a0 * a1 + s2 * b0 * b1
        T[i, 2] = T[i, 6] = s1 * a2 * a0 + s2 * b2 * b0
        T[i, 5] = T[i, 7] = s1 * a1 * a2 + s2 * b1 * b2
    return T


@njit(cache=True, error_model="numpy")
def _smooth_tensor(T, v, indptr, indices, nrm, iters):
    """``smoothingTensor``: inverse projected-distance weighted ring mean (float32 w)."""
    n = v.shape[0]
    one = F32(1.0)
    cur = T.copy()
    for _ in range(iters):
        new = np.zeros((n, 9))
        for i in range(n):
            nx, ny, nz = nrm[i, 0], nrm[i, 1], nrm[i, 2]
            wsum = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                k = indices[j]
                ex, ey, ez = v[k, 0] - v[i, 0], v[k, 1] - v[i, 1], v[k, 2] - v[i, 2]
                p = nx * ex + ny * ey + nz * ez
                dx, dy, dz = ex - nx * p, ey - ny * p, ez - nz * p
                w = one / _fnorm(dx, dy, dz)
                wsum += np.float64(w)
                for c in range(9):
                    new[i, c] += cur[k, c] * np.float64(w)
            for c in range(9):
                new[i, c] /= wsum
        cur = new
    return cur


@njit(cache=True, error_model="numpy")
def _smooth_scalar(s, v, indptr, indices, iters):
    """``smoothingScalar``: inverse edge-length weighted ring mean (float32 weight)."""
    n = v.shape[0]
    one = F32(1.0)
    cur = s.copy()
    for _ in range(iters):
        new = np.zeros(n)
        for i in range(n):
            wsum = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                k = indices[j]
                ex, ey, ez = v[k, 0] - v[i, 0], v[k, 1] - v[i, 1], v[k, 2] - v[i, 2]
                w = one / _fnorm(ex, ey, ez)
                wsum += np.float64(w)
                new[i] += cur[k] * np.float64(w)
            new[i] /= wsum
        cur = new
    return cur


@njit(cache=True, error_model="numpy")
def _max_dist(adj, source, indptr, indices):
    """Largest distance reachable by greedy uphill walking inside the same source basin."""
    n = adj.shape[0]
    out = np.zeros(n)
    for i in range(n):
        mx = adj[i]
        cur = i
        while True:
            best = -1
            for j in range(indptr[cur], indptr[cur + 1]):
                k = indices[j]
                if source[cur] != source[k]:
                    continue
                if mx < adj[k]:
                    mx = adj[k]
                    best = k
            out[i] = mx
            if best == -1:
                break
            cur = best
    return out


@njit(cache=True, error_model="numpy")
def _principal_axis(T):
    """Largest-eigenvalue eigenvector of every 3x3 tensor (``eig3symmetric``)."""
    n = T.shape[0]
    u = np.zeros((n, 3))
    M = np.empty((3, 3))
    lam = np.zeros(3)
    ev = np.eye(3)
    for i in range(n):
        for a in range(3):
            for b in range(3):
                M[a, b] = T[i, a * 3 + b]
        lam[:] = 0.0
        ev[:] = 0.0
        ev[0, 0] = ev[1, 1] = ev[2, 2] = 1.0
        _eig3(M, lam, ev)
        u[i, 0] = ev[2, 0]
        u[i, 1] = ev[2, 1]
        u[i, 2] = ev[2, 2]
    return u


def setup_tensor_equal_area(v32, nb, nrm32, adj, source, speed):
    """Returns ``(u, lam1, lam2)`` for ``GeodesicA::setTensor`` (``area = 1``)."""
    T = _initial_tensor(v32, nb.indptr, nb.indices, nrm32, adj)
    T = _smooth_tensor(T, v32, nb.indptr, nb.indices, nrm32, 10)
    mx = _max_dist(adj, source, nb.indptr, nb.indices)
    mx = np.where(mx == 0, 1.0, mx)
    a, b = float(speed), 1.0
    param = (b - a) * (adj - 0) / (mx - 0) + a
    param = np.minimum(param, 1.0)
    param = _smooth_scalar(param, v32, nb.indptr, nb.indices, 10)
    param = np.minimum(param, 1.0)
    u = _principal_axis(T)
    lam1 = param
    lam2 = 1.0 / lam1
    return u, lam1, lam2


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
@dataclass
class LgiResult:
    """Outputs of :func:`gyrification`.

    ``deltas[t]`` is the (adjusted) kernel area of map ``t``; ``lgi[:, t]`` its per-vertex LGI;
    ``k1``/``k2`` the kernel areas on the surface / hull (``--kArea1/2``).
    """

    deltas: np.ndarray
    lgi: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    adj_dist: np.ndarray
    source: np.ndarray
    adj_ratio: float
    max_area: float
    intv: float
    n_hull_defects: int
    area_surface: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    area_hull: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def file_labels(self) -> list[int]:
        """The ``%d`` in ``<out>.lgi.map.%d.txt`` for every map (truncated adjusted area)."""
        return [int(d) for d in self.deltas]


def _as_mesh(m) -> Mesh:
    return m if isinstance(m, Mesh) else read_surface(m)


def gyrification(
    surface,
    outer,
    *,
    sulcus=None,
    gyrus=None,
    curve_ids=None,
    max_area: float = 316.0,
    intv_area: float = 0.0,
    speed: float = 0.2,
    ref_cortex_area: float = 0.0,
    ref_hull_area: float = 77100.0,
    input_map=None,
    corr=None,
    n_threads: int | None = None,
    heap: str = "libstdcxx",
    libm: str = DEFAULT_LIBM,
) -> LgiResult:
    """Run the Gyrification pipeline (defaults = ``Gyrification`` CLI defaults).

    ``surface`` / ``outer``: :class:`Mesh` or VTK path.  ``outer`` must have the surface's
    topology, with every vertex moved to the hull (klaplace ``-surfaceCorrespondence`` output);
    that file is what ``--outer`` takes.  Curves: ``sulcus`` / ``gyrus`` as TRACE ``.scurve`` /
    ``.gcurve`` (or ``.bary``) paths or lists of id arrays; ``curve_ids`` gives the union directly.
    ``corr``: optional lookup surface vertex -> outer vertex (``--corr``).
    ``heap``: ``"libcxx"`` (oracle build) or ``"libstdcxx"`` (Linux image) std::heap tie order.
    ``libm``: ``"fdlibm"`` (pinned ``acosf`` of the MeshLib normal weights, platform independent,
    provisional default) or ``"host"`` (the machine's libm; results then vary across platforms).
    """
    if heap not in _HEAP:
        raise ValueError("heap must be 'libcxx' or 'libstdcxx'")
    kind = _HEAP[heap]
    surf, hull = _as_mesh(surface), _as_mesh(outer)
    n = surf.n_vertices
    if corr is None and hull.n_vertices != n:
        raise ValueError(
            f"outer surface has {hull.n_vertices} vertices, surface {n}: pass a vertex-wise "
            "correspondence (klaplace output) or --corr"
        )
    if corr is not None:
        lookup = np.ascontiguousarray(np.asarray(corr, dtype=np.int64).ravel())
        if lookup.size != n:
            raise ValueError("corr must give one outer vertex per surface vertex")
        if hull.n_vertices != n:  # MeshLib reads out of bounds here (varea indexed by surface id)
            raise ValueError(
                "--corr with an outer surface of different vertex count is undefined upstream"
            )
    else:
        lookup = np.zeros(1, dtype=np.int64)

    v32 = np.ascontiguousarray(surf.vertices, dtype=np.float32)
    nb = vertex_neighbor_lists(surf.faces, n)
    gw = fm.build_gw_mesh(surf.vertices, surf.faces)

    # ---- geodesic distance from the curves
    if input_map is not None:
        adj, source = read_geodesic_map(input_map, n)
    else:
        sets = [c for c in (sulcus, gyrus) if c is not None]
        ids = (
            np.asarray(curve_ids, dtype=np.int64)
            if curve_ids is not None
            else curve_vertex_ids(*sets)
        )
        if ids.size == 0:
            raise ValueError("no sulcal/gyral curve vertices")
        if ids.min() < 0 or ids.max() >= n:
            raise ValueError("curve vertex id out of range")
        dist, _state, front = fm.propagate_iso(gw, np.unique(ids), heap=kind)
        adj, source = dist, front.astype(np.int64)

    # ---- areas
    a1 = vertex_areas(surf.vertices, surf.faces, nb)
    hnb = vertex_neighbor_lists(hull.faces, hull.n_vertices)
    a2 = vertex_areas(hull.vertices, hull.faces, hnb).copy()
    defects = a1[: a2.size] < a2
    a2 = np.where(defects, a1[: a2.size], a2)
    n_defects = int(defects.sum())

    ratio = 1.0
    if ref_hull_area > 0:
        ratio = _seq_sum(a2) / ref_hull_area
    elif ref_cortex_area > 0:
        ratio = _seq_sum(a1) / ref_cortex_area
    max_adj = max_area * ratio
    intv = intv_area * ratio
    if intv == 0:
        intv = max_adj
    nk = int(math.ceil(max_adj / intv))
    deltas = np.array([min(intv * (t + 1), max_adj) for t in range(nk)])

    # ---- metric tensor and kernels
    nrm = vertex_normals_f32(surf.vertices, surf.faces, libm)
    u, lam1, lam2 = setup_tensor_equal_area(v32, nb, nrm, adj, source, speed)
    T, V1, lam1, lam2 = fm.set_tensor(gw, u, lam1, lam2)

    k1 = np.zeros((n, nk))
    k2 = np.zeros((n, nk))
    nt = max(1, n_threads if n_threads else int(os.environ.get("OMP_NUM_THREADS", "1") or 1))
    bounds = np.linspace(0, n, min(nt, n) * 4 + 1 if nt > 1 else 2).astype(int)

    def work(lo_hi):
        lo, hi = lo_hi
        fm.kernel_range(
            kind,
            *gw[:2],
            gw.fnbr,
            gw.vf_ptr,
            gw.vf_idx,
            gw.ring_ptr,
            gw.ring_idx,
            gw.normal,
            T,
            V1,
            lam1,
            lam2,
            a2,
            a1,
            a2,
            lookup,
            corr is not None,
            max_adj,
            deltas,
            lo,
            hi,
            k1,
            k2,
        )

    chunks = list(zip(bounds[:-1], bounds[1:], strict=True))
    if nt > 1:
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(work, chunks))
    else:
        for c in chunks:
            work(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        lgi = k1 / k2
    return LgiResult(deltas, lgi, k1, k2, adj, source, ratio, max_adj, intv, n_defects, a1, a2)


@njit(cache=True, error_model="numpy")
def _seq_sum(a):
    s = 0.0
    for x in a:
        s += x
    return s


def write_lgi_maps(result: LgiResult, prefix, *, kind: str = "lgi") -> list[Path]:
    """``saveGI`` / ``saveKMap{1,2}``: ``<prefix>.{lgi,k1,k2}.map.<int(delta)>.txt``, ``%f``."""
    arr = {"lgi": result.lgi, "k1": result.k1, "k2": result.k2}[kind]
    out = []
    for t, d in enumerate(result.deltas):
        p = Path(f"{prefix}.{kind}.map.{int(d)}.txt")
        with open(p, "w") as fh:
            for x in arr[:, t]:
                fh.write(f"{x:f}\n")
        out.append(p)
    return out
