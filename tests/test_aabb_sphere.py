"""B2.5: ``pybiscuit.sphere.aabb`` (port of MeshLib ``AABB_Sphere``).

Layers
------
* committed goldens (``tests/data/golden/meshlib/aabb_*``) from the upstream MeshLib
  harness; no binaries needed;
* a pure-numpy float64 brute-force cross-check on small meshes;
* oracle tests that run ``scripts/oracles/harness_aabb.cpp`` (skipped when
  ``~/dev/biscuit/oracles`` is absent): 1e5 seeded queries on icosphere(5, 6, 7),
  a deformed sphere (fresh tree and ``update()``), structured queries (vertices,
  edge midpoints, poles, the +-pi cut) and all eps overloads;
* the 163,842-query timing budget on icosphere(7).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from _compare import (
    assert_candidate_sets_close,
    assert_close_float32,
    assert_same_choice_modulo_ties,
)
from conftest import REPO
from pybiscuit.io import write_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.sphere import icosphere
from pybiscuit.sphere.aabb import SphereAABB, barycentric, cart2sph, closest_face

ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
GOLDEN = REPO / "tests" / "data" / "golden" / "meshlib"
MESHLIB = REPO.parent / "reference" / "repos" / "MeshLib"
BARY_TOL = 1e-6


# --------------------------------------------------------------------------- helpers
def f32mesh(mesh: Mesh) -> Mesh:
    """Round vertices to float32 so that the text written for the oracle is exact."""
    return Mesh(mesh.vertices.astype(np.float32).astype(np.float64), mesh.faces)


def deformed_sphere(mesh: Mesh) -> Mesh:
    """Star-shaped deformation of a unit sphere strictly inside it.

    Radius 0.85..0.95 (so unit-sphere queries lie outside the surface, which is what
    AABB_Sphere requires) plus a small azimuth/elevation shift that keeps the
    direction map bijective.
    """
    v = mesh.vertices
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0])
    ph2 = ph + 0.03 * np.sin(3 * th) * np.cos(2 * ph)
    th2 = th + 0.01 * np.sin(2 * ph) * np.sin(th)
    r = 0.9 - 0.05 * np.sin(3 * th) * np.cos(2 * ph)
    w = np.stack([np.sin(th2) * np.cos(ph2), np.sin(th2) * np.sin(ph2), np.cos(th2)], 1)
    return Mesh(w * r[:, None], mesh.faces)


def random_unit(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p.astype(np.float32)


def structured_queries(mesh: Mesh) -> np.ndarray:
    """Vertices, edge midpoints, centroids (raw and normalised), poles, +-pi cut."""
    v = mesh.vertices.astype(np.float32)
    tri = v[mesh.faces]
    raw = np.vstack(
        [
            v,
            (tri[:, 0] + tri[:, 1]) / 2,
            tri.mean(1),
            [[0, 0, 1], [0, 0, -1], [1e-9, 0, 1], [0, 1e-9, -1]],
            [[-1, 0, 0], [-1, 1e-9, 0], [-1, -1e-9, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]],
        ]
    ).astype(np.float32)
    unit = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
    return np.vstack([raw, unit])


def parse_mode(mode: str):
    """'closest' | 'bool:E1:E2' | 'xcand:E1:E2' | 'search:E1:E2' -> (kind, e1, e2)."""
    kind, *rest = mode.split(":")
    e1 = float(rest[0]) if rest else 0.0
    e2 = float(rest[1]) if len(rest) > 1 else e1
    return kind, e1, e2


def python_result(mode, mesh, queries, deformed=None, refresh=False):
    """Run the Python port in the harness' output convention.

    Returns ``(face, bary)`` for closest/bool/xcand (bary zeroed where face == -1, like
    the harness) and a list of candidate arrays for ``search``.
    """
    tree = SphereAABB(mesh)
    if deformed is not None:
        tree.update(deformed.vertices, refresh_face_normals=refresh)
    kind, e1, e2 = parse_mode(mode)
    if kind == "search":
        off, fs = tree.candidates(queries, e1, e2)
        return [fs[off[i] : off[i + 1]] for i in range(len(queries))]
    if kind == "closest":
        f, b = tree.closest_face(queries)
    elif kind == "bool":
        f, b = tree.closest_face(queries, e1, e2)
    elif kind == "xcand":
        f, b = tree.closest_face(queries, e1, e2, poles=False)
    else:  # pragma: no cover
        raise ValueError(mode)
    b = np.where(f[:, None] == -1, np.float32(0), b)
    return f, b


def read_expected(path: Path, kind: str):
    if kind == "search":
        return [
            np.array(line.split()[1:], dtype=np.int32) for line in path.read_text().splitlines()
        ]
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0].astype(np.int64), a[:, 1:].astype(np.float32)


def assert_matches(mode, got, expected, label=""):
    """EXACT comparison (face / candidate list equality).

    Only for (a) the opt-in ``bitexact`` golden check on the oracle platform and (b) the live
    oracle tests, where harness and port run on the SAME machine and therefore share one libm.
    The committed goldens (made on macOS arm64) go through :func:`compare_golden` instead.
    """
    if parse_mode(mode)[0] == "search":
        bad = [
            i
            for i, (g, e) in enumerate(zip(got, expected, strict=True))
            if not np.array_equal(g, e)
        ]
        assert not bad, f"{label}: {len(bad)} candidate lists differ, first {bad[:5]}"
        return 0.0
    f, b = got
    ef, eb = expected
    bad = np.flatnonzero(f != ef)
    assert bad.size == 0, f"{label}: {bad.size} face indices differ, first {bad[:5]}"
    err = float(np.abs(b - eb).max()) if len(b) else 0.0
    assert err <= BARY_TOL, f"{label}: max bary error {err:.3e}"
    return err


# ----------------------------------------------------------------- cross-platform policy
# float32 unit roundoff scale; the unit-sphere queries and meshes have |x| <~ 1.
F32_EPS = float(np.finfo(np.float32).eps)  # 1.19e-7 = 1 ulp(1)
# Distance ties: query points are surface points stored as float32 (edge midpoints, vertices,
# centroids), so they sit up to ~1 ulp(1) off the mesh and competing faces' point-triangle
# distances differ by float32 rounding noise.  The observed gaps are <= 1e-8; 4 ulp(1) is the
# bound we accept as "same choice".
DIST_EPS = 4 * F32_EPS
# Box tests: (phi, theta) come from libm atan2f, whose result differs by 1 ulp between
# platforms (ulp(2*pi) = 4.8e-7).  A box membership decision is platform dependent only when
# the query is within a few ulps of a bound: 4 ulp(2*pi) ~ 1.9e-6.
ANGLE_EPS = 4 * 2.0**-21  # ulp(2*pi) = 2^-21
RAD0 = 0.0174533


class BoxAmbiguity:
    """Which (query, face) pairs have a platform dependent box decision?

    Recomputes every box in float64 from the tree's float32 vertices (no libm ``atan2f``
    involved).  Faces whose wrap decision ``maxphi - minphi > pi`` is within ``ANGLE_EPS`` of
    the threshold (e.g. a pole face with a vertex on the +-pi cut: Apple's ``atan2f(0, -x)``
    returns pi - 1 ulp, glibc/UCRT the correctly rounded pi, so the face box is ``[0, pi]`` on
    one and the wrap box ``[pi, 2 pi]`` on the other) get both variants; node boxes are built
    with none (``N``) and all (``A``) of these faces wrapped.  A node is *ambiguous* for a query
    if the query is within ``ANGLE_EPS`` of one of its (eps-enlarged) bounds, or if the
    visit/skip decision differs between ``N`` and ``A``.  Also ambiguous: the global
    thresholds ``phi <= maxPhi - 2 pi`` and ``theta >= maxTheta`` / ``<= minTheta``.

    ``near(i, face)`` is true if any node on the root-to-leaf path of ``face`` is ambiguous for
    query ``i``: a leaf is visited, and then ALL its faces are candidates, only if every
    ancestor box passes.
    """

    def __init__(self, tree: SphereAABB, queries: np.ndarray):
        self.t = tree
        v, fc = tree._verts, tree._faces
        phi_v, th_v = self._angles(v)
        pf, tf = phi_v[fc], th_v[fc]
        lo_p, hi_p = pf.min(1), pf.max(1)
        self.face_wrap_sens = np.abs((hi_p - lo_p) - np.pi) <= ANGLE_EPS
        wrapped = (hi_p - lo_p) > np.pi
        plain = np.stack([lo_p, hi_p, tf.min(1), tf.max(1)], 1)
        wrap = np.stack([hi_p, lo_p + 2 * np.pi, tf.min(1), tf.max(1)], 1)
        sens = self.face_wrap_sens
        box_n = np.where((wrapped & ~sens)[:, None], wrap, plain)
        box_a = np.where((wrapped | sens)[:, None], wrap, plain)
        nn = tree.n_nodes
        self.parent = np.full(nn, -1, np.int64)
        for n in range(nn):
            if tree._left[n] >= 0:
                self.parent[tree._left[n]] = n
                self.parent[tree._right[n]] = n
        self.leaf_of = np.full(tree.n_faces, -1, np.int64)
        self.nbox = []
        for fb in (box_n, box_a):
            nb = np.zeros((nn, 4))
            for n in range(nn - 1, -1, -1):  # children have larger indices than parents
                if tree._left[n] < 0:
                    fs = tree._perm[tree._lo[n] : tree._lo[n] + tree._cnt[n]]
                    self.leaf_of[fs] = n
                    k = fb[fs]
                    nb[n] = [k[:, 0].min(), k[:, 1].max(), k[:, 2].min(), k[:, 3].max()]
                else:
                    lc, rc = nb[tree._left[n]], nb[tree._right[n]]
                    nb[n] = [
                        min(lc[0], rc[0]),
                        max(lc[1], rc[1]),
                        min(lc[2], rc[2]),
                        max(lc[3], rc[3]),
                    ]
            self.nbox.append(nb)
        self.phi, self.theta = self._angles(np.asarray(queries, np.float32))
        self.global_amb = (
            (np.abs(self.phi - (float(tree._max_phi) - 2 * np.pi)) <= ANGLE_EPS)
            | (np.abs(self.theta - float(tree._max_theta)) <= ANGLE_EPS)
            | (np.abs(self.theta - float(tree._min_theta)) <= ANGLE_EPS)
        )

    @staticmethod
    def _angles(p):
        """float64 (phi, theta) of float32 points, with the kernel's zero special cases."""
        x, y, z = (p[:, k].astype(np.float64) for k in range(3))
        d32 = (p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1]).astype(np.float32)
        phi = np.where(d32 == 0, 0.0, np.arctan2(y, x))
        theta = np.where(z == 0, 0.0, np.arctan2(z, np.sqrt(d32).astype(np.float64)))
        return phi, theta

    def node_ambiguous(self, i, n, e1, e2):
        th = self.theta[i]
        visits = []
        for nb in self.nbox:
            b = nb[n]
            ok = False
            for ph in (self.phi[i], self.phi[i] + 2 * np.pi):
                if abs(ph - (b[0] - e1)) <= ANGLE_EPS or abs(ph - (b[1] + e1)) <= ANGLE_EPS:
                    return True
                ok |= b[0] - e1 <= ph <= b[1] + e1
            if abs(th - (b[2] - e2)) <= ANGLE_EPS or abs(th - (b[3] + e2)) <= ANGLE_EPS:
                return True
            visits.append(ok and b[2] - e2 <= th <= b[3] + e2)
        return visits[0] != visits[1]

    def near_fn(self, epss):
        """``near(query, face)`` for the given list of (eps1, eps2) box enlargements."""

        def near(i, f):
            if self.global_amb[i]:
                return True
            n = self.leaf_of[f]
            while n >= 0:
                if any(self.node_ambiguous(i, n, e1, e2) for e1, e2 in epss):
                    return True
                n = self.parent[n]
            return False

        return near


def _enlargements(mode):
    kind, e1, e2 = parse_mode(mode)
    if kind == "closest":  # trial 0 (no enlargement) then rad * 2^k on both angles
        return [(0.0, 0.0)] + [(RAD0 * 2.0**k,) * 2 for k in range(9)]
    return [(e1, e2)]


def compare_golden(mode, tree, queries, got, expected, label):
    """Cross-platform golden comparison.  Returns a stats dict.

    * barycentrics where the face agrees: abs <= 1e-6 (brief 02 acceptance, not loosened);
    * faces: :func:`assert_same_choice_modulo_ties` with the margin recomputed in float64
      (|dist(chosen) - dist(golden)| <= DIST_EPS), or, when the two faces are not tied in
      distance, a mismatch explained by a platform dependent box decision (``BoxAmbiguity``);
    * candidate lists: sets, differences only for faces whose leaf path is box-ambiguous.
    """
    amb = BoxAmbiguity(tree, queries)
    near = amb.near_fn(_enlargements(mode))
    if parse_mode(mode)[0] == "search":
        n_diff = assert_candidate_sets_close(got, expected, near, label=label)
        return {"differing_lists": n_diff, "n": len(queries)}
    f, b = got
    ef, eb = expected
    verts = tree._verts.astype(np.float64)
    q64 = np.asarray(queries, np.float64)
    cats = {"dist_tie": 0, "box_ambiguous": 0}

    def dist(i, fi):
        return np.inf if fi < 0 else _point_triangle_dist(q64[i], *verts[tree._faces[fi]])

    def margin(idx, chosen, golden):
        out = np.empty(len(idx))
        for k, (i, c, g) in enumerate(zip(idx, chosen, golden, strict=True)):
            gap = abs(dist(i, c) - dist(i, g))
            if gap <= DIST_EPS:
                cats["dist_tie"] += 1
            elif any(f >= 0 and near(i, f) for f in (c, g)):
                cats["box_ambiguous"] += 1
                gap = 0.0
            out[k] = gap
        return out

    ties = assert_same_choice_modulo_ties(f, ef, margin, DIST_EPS, label=label)
    same = f == ef
    if same.any():
        assert_close_float32(b[same], eb[same], rel=0.0, abs_=BARY_TOL, label=f"{label} bary")
    return {"mismatches": ties, **cats, "n": len(queries)}


# --------------------------------------------------------------------------- goldens
GOLDEN_CASES = sorted(p.name for p in GOLDEN.glob("aabb_*") if p.is_dir())
GOLDEN_STATS: dict[str, dict] = {}


def test_goldens_present():
    assert len(GOLDEN_CASES) >= 6


def _load_golden(case):
    from pybiscuit.io import read_vtk

    d = GOLDEN / case
    cmd = json.loads((d / "command.json").read_text())
    assert cmd["oracle"] == "upstream-macos-arm64" and cmd["commit"]
    argv = cmd["argv"]
    mode, extra = argv[1], argv[5:]
    mesh = read_vtk(d / argv[2])
    deformed = read_vtk(d / extra[0]) if extra else None
    refresh = "refresh" in extra
    q = np.loadtxt(d / argv[3], dtype=np.float32, ndmin=2)
    exp = read_expected(d / argv[4], parse_mode(mode)[0])
    return mode, mesh, q, deformed, refresh, exp


def _run(mode, tree, q):
    """Query an existing tree in the harness' output convention."""
    kind, e1, e2 = parse_mode(mode)
    if kind == "search":
        off, fs = tree.candidates(q, e1, e2)
        return [fs[off[i] : off[i + 1]] for i in range(len(q))]
    if kind == "closest":
        f, b = tree.closest_face(q)
    else:
        f, b = tree.closest_face(q, e1, e2, poles=(kind == "bool"))
    return f, np.where(f[:, None] == -1, np.float32(0), b)


def _golden_tree(mesh, deformed, refresh, tree_cls=SphereAABB):
    tree = tree_cls(mesh)
    if deformed is not None:
        tree.update(deformed.vertices, refresh_face_normals=refresh)
    return tree


@pytest.mark.parametrize("case", GOLDEN_CASES)
def test_golden(case):
    """Goldens come from macOS arm64 (Apple libm ``atan2f``); compare modulo explained ties."""
    mode, mesh, q, deformed, refresh, exp = _load_golden(case)
    tree = _golden_tree(mesh, deformed, refresh)
    stats = compare_golden(mode, tree, q, _run(mode, tree, q), exp, case)
    GOLDEN_STATS[case] = stats
    print(f"{case}: {stats}")


@pytest.mark.bitexact
@pytest.mark.parametrize("case", GOLDEN_CASES)
def test_golden_bitexact(case):
    """Diagnostic only (``pytest -m bitexact``): exact equality on the oracle platform (arm64)."""
    mode, mesh, q, deformed, refresh, exp = _load_golden(case)
    assert_matches(mode, python_result(mode, mesh, q, deformed, refresh), exp, case)


# --------------------------------------------------------------- emulated other platform
@pytest.fixture(scope="session")
def other_libm_module():
    """The port recompiled with a correctly rounded ``atan2f`` (float64 ``atan2``, then round).

    glibc and the Windows UCRT return this in practically every case, while Apple's libm
    differs by 1 ulp for ~6 % of the arguments and returns pi - 1 ulp for ``atan2f(0, -x)``.
    The recompiled module therefore reproduces the x86-64 CI discrepancies on any machine.
    """
    import sys
    import types

    from pybiscuit.sphere import aabb

    src = Path(aabb.__file__).read_text()
    needle = "    return F(math.atan2(y, x))\n"
    assert src.count(needle) == 1, "aabb._atan2f changed: update the emulation"
    src = src.replace(needle, "    return F(math.atan2(np.float64(y), np.float64(x)))\n")
    src = src.replace("cache=True", "cache=False")
    mod = types.ModuleType("_aabb_correctly_rounded_atan2")
    mod.__file__ = aabb.__file__
    sys.modules[mod.__name__] = mod
    exec(compile(src, aabb.__file__, "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("case", GOLDEN_CASES)
def test_golden_with_other_libm(case, other_libm_module):
    """Same goldens with a different ``atan2f``: must pass modulo explained ties."""
    mode, mesh, q, deformed, refresh, exp = _load_golden(case)
    tree = _golden_tree(mesh, deformed, refresh, other_libm_module.SphereAABB)
    stats = compare_golden(mode, tree, q, _run(mode, tree, q), exp, case + "[other libm]")
    GOLDEN_STATS[case + "[other libm]"] = stats
    print(f"{case}[other libm]: {stats}")


# ----------------------------------------------------- the comparison must not be vacuous
def test_untied_face_mismatch_fails():
    mesh = f32mesh(icosphere(3))
    tree = SphereAABB(mesh)
    q = random_unit(200, 1)
    f, b = tree.closest_face(q)
    exp_f = f.copy()
    exp_f[7] = (f[7] + mesh.n_faces // 2) % mesh.n_faces  # a face on the far side
    with pytest.raises(AssertionError, match="untied"):
        compare_golden("closest", tree, q, (f, b), (exp_f, b), "synthetic")
    exp_f[7] = f[7]
    exp_f[9] = -1  # found vs not found, with no box ambiguity
    with pytest.raises(AssertionError, match="untied"):
        compare_golden("closest", tree, q, (f, b), (exp_f, b), "synthetic")
    # barycentric tolerance stays 1e-6 absolute
    compare_golden("closest", tree, q, (f, b), (f, b + np.float32(5e-7)), "synthetic")
    with pytest.raises(AssertionError, match="outside"):
        compare_golden("closest", tree, q, (f, b), (f, b + np.float32(2e-6)), "synthetic")


def test_untied_candidate_mismatch_fails():
    mesh = f32mesh(icosphere(3))
    tree = SphereAABB(mesh)
    q = random_unit(50, 2)
    mode = "search:0.02:0.01"
    got = _run(mode, tree, q)
    assert (
        compare_golden(mode, tree, q, got, [g.copy() for g in got], "synthetic")["differing_lists"]
        == 0
    )
    bad = [g.copy() for g in got]
    far = np.setdiff1d(np.arange(mesh.n_faces), got[3])[0]
    bad[3] = np.append(bad[3], far).astype(np.int32)  # extra candidate in an unrelated leaf
    with pytest.raises(AssertionError, match="not near"):
        compare_golden(mode, tree, q, got, bad, "synthetic")


def test_near_flags_box_edge_queries_only():
    """A query on a node's bound is flagged; a generic query is not."""
    tree = SphereAABB(f32mesh(icosphere(3)))
    leaf = int(np.flatnonzero(tree._left < 0)[0])
    f = int(tree._perm[tree._lo[leaf]])
    b = tree._nbox[leaf].astype(np.float64)
    on_bound = [np.cos(b[2]) * np.cos(b[0]), np.cos(b[2]) * np.sin(b[0]), np.sin(b[2])]
    generic = np.array([0.31, -0.52, 0.79])
    q = np.array([on_bound, generic / np.linalg.norm(generic)], np.float32)
    near = BoxAmbiguity(tree, q).near_fn([(0.0, 0.0)])
    assert near(0, f)
    assert not near(1, f)


def test_cart2sph_uses_tree_kernel_and_no_fastmath():
    import re

    from pybiscuit.sphere import aabb

    q = random_unit(500, 3)
    phi, theta = cart2sph(q)
    p32, t32 = aabb._cart2sph_all(q)
    assert np.array_equal(phi, p32) and np.array_equal(theta, t32)
    code = re.sub(r'""".*?"""|#.*', "", Path(aabb.__file__).read_text(), flags=re.S)
    assert "fastmath" not in code and "fma" not in code.lower()


# --------------------------------------------------------------------------- brute force
def _point_triangle_dist(p, a, b, c):
    """Ericson closest point on triangle, float64 (independent of the port)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return np.linalg.norm(ap)
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return np.linalg.norm(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return np.linalg.norm(p - (a + ab * d1 / (d1 - d3)))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return np.linalg.norm(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return np.linalg.norm(p - (a + ac * d2 / (d2 - d6)))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.linalg.norm(p - (b + (c - b) * w))
    denom = 1.0 / (va + vb + vc)
    return np.linalg.norm(p - (a + ab * vb * denom + ac * vc * denom))


def brute_force(verts, faces, p):
    """Float64 re-statement of the closest-face rule over *all* faces.

    valid: the query is not behind the face plane (``r1 >= r0 - 1e-5``) and its radial
    projection onto the plane has barycentric coordinates ``>= -1e-5``; winner: smallest
    point-triangle distance.  Returns ``(face, bary, gap)`` where ``gap`` is the distance
    difference to the runner-up (``inf`` if unique).
    """
    best = []
    for fi, (ia, ib, ic) in enumerate(faces):
        a, b, c = verts[ia], verts[ib], verts[ic]
        n = np.cross(b - a, c - b)
        n /= np.linalg.norm(n)
        r0, r1 = a @ n, p @ n
        if r1 < r0 - 1e-5:
            continue
        q = p * (r0 / r1)
        area2 = np.cross(b - a, c - a)
        nn = area2 / np.linalg.norm(area2)
        den = np.linalg.norm(area2)
        c0 = np.cross(b - q, c - q) @ nn / den
        c1 = np.cross(c - q, a - q) @ nn / den
        co = np.array([c0, c1, 1 - c0 - c1])
        if (co >= -1e-5).all():
            best.append((_point_triangle_dist(p, a, b, c), fi, co))
    if not best:
        return -1, np.full(3, np.nan), np.inf
    best.sort(key=lambda t: (t[0], t[1]))
    gap = best[1][0] - best[0][0] if len(best) > 1 else np.inf
    return best[0][1], best[0][2], gap


@pytest.mark.parametrize("name", ["ico2", "ico2_deformed", "uv"])
def test_brute_force(name):
    from fixtures.meshes import uv_sphere

    if name == "uv":
        mesh = f32mesh(uv_sphere(8, 16))
    else:
        mesh = f32mesh(icosphere(2))
        if name.endswith("deformed"):
            mesh = f32mesh(deformed_sphere(mesh))
    verts = mesh.vertices.astype(np.float32).astype(np.float64)
    q = np.vstack([random_unit(300, 11), structured_queries(mesh)[:80]])
    face, bary = SphereAABB(mesh).closest_face(q)
    checked = 0
    for i in range(len(q)):
        bf, bb, gap = brute_force(verts, mesh.faces, q[i].astype(np.float64))
        if gap < 1e-5:  # float32 near-tie: either face is acceptable
            continue
        assert face[i] == bf, (i, q[i], face[i], bf)
        assert np.abs(bary[i] - bb).max() < 5e-4  # float32 triple-product conditioning
        checked += 1
    assert checked > len(q) * 0.6


# --------------------------------------------------------------------------- unit tests
def test_barycentric_basic():
    a = np.array([[0.0, 0, 0]])
    b = np.array([[1.0, 0, 0]])
    c = np.array([[0.0, 1, 0]])
    p = np.array([[0.25, 0.25, 0.0]])
    for dt in (np.float32, np.float64):
        w = barycentric(a, b, c, p, dtype=dt)
        assert w.dtype == dt
        np.testing.assert_allclose(w, [[0.5, 0.25, 0.25]], atol=1e-6)
    # off-plane point: coefficients of the orthogonal projection
    w = barycentric(a, b, c, np.array([[0.25, 0.25, 3.0]]))
    np.testing.assert_allclose(w, [[0.5, 0.25, 0.25]], atol=1e-6)
    # broadcasting of a single triangle against many points
    pts = np.random.default_rng(0).uniform(0, 1, (5, 3))
    assert barycentric(a, b, c, pts).shape == (5, 3)
    with pytest.raises(ValueError):
        barycentric(a, b, c, p, dtype=np.int32)


def test_cart2sph_is_elevation():
    phi, theta = cart2sph([[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, -1]])
    np.testing.assert_allclose(theta, [np.pi / 2, 0, 0, 0, -np.pi / 2], atol=1e-6)
    np.testing.assert_allclose(phi, [0, 0, np.pi / 2, np.pi, 0], atol=1e-6)


def test_basic_ico_and_outputs():
    mesh = icosphere(4)
    tree = SphereAABB(mesh)
    assert tree.n_faces == mesh.n_faces and tree.poles[0] >= 0 and tree.poles[1] >= 0
    q = random_unit(2000, 5)
    face, bary = tree.closest_face(q)
    assert face.dtype == np.int32 and bary.dtype == np.float32 and bary.shape == (2000, 3)
    assert (face >= 0).all()
    np.testing.assert_allclose(bary.sum(1), 1.0, atol=1e-5)
    assert (bary >= -1e-5).all()
    # reconstruct: barycentric combination of the face's vertices lies on the radial projection
    v = mesh.vertices.astype(np.float32)
    rec = np.einsum("nk,nkd->nd", bary, v[mesh.faces[face]])
    cosang = (rec * q).sum(1) / np.linalg.norm(rec, axis=1)
    assert (cosang > 1 - 1e-5).all()
    # convenience wrapper and single-point input agree
    f2, b2 = closest_face(mesh.vertices, mesh.faces, q)
    assert np.array_equal(face, f2) and np.array_equal(bary, b2)
    f3, _ = tree.closest_face(q[0])
    assert f3.shape == (1,) and f3[0] == face[0]


def test_not_found_and_inside():
    mesh = icosphere(3)
    tree = SphereAABB(mesh)
    # a query well inside the sphere is behind every face plane: nothing is found
    f, b = tree.closest_face(np.array([[0.1, 0.2, 0.3]]))
    assert f[0] == -1 and np.isnan(b).all()
    f, b = tree.closest_face(np.empty((0, 3)))
    assert f.shape == (0,) and b.shape == (0, 3)


def test_update_validation_and_ties_flag():
    mesh = icosphere(3)
    tree = SphereAABB(mesh)
    with pytest.raises(ValueError):
        tree.update(mesh.vertices[:-1])
    with pytest.raises(ValueError):
        tree.closest_face(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        tree.closest_face(np.zeros((3, 3)), eps2=0.1)
    with pytest.raises(ValueError):
        SphereAABB(mesh.vertices)
    q = mesh.vertices.astype(np.float32)  # exact vertices sit on 5-6 faces: exact ties
    f, _, ties = tree.closest_face(q, return_ties=True)
    assert ties.all()
    assert (f >= 0).all()
    # lowest-index rule: a vertex query returns the lowest face index among incident faces
    for vi in (10, 100, 300):
        inc = np.flatnonzero((mesh.faces == vi).any(1))
        assert f[vi] == inc.min()


def test_update_equals_fresh_when_pole_faces_unchanged():
    base = f32mesh(icosphere(4))
    defo = f32mesh(deformed_sphere(base))
    q = random_unit(20000, 3)
    t = SphereAABB(base)
    t.update(defo.vertices)
    fresh = SphereAABB(defo)
    fu, bu = t.closest_face(q)
    ff, bf = fresh.closest_face(q)
    # same faces wherever the (fixed-topology) tree finds them; tree grouping may differ only
    # in which candidates are visited, so allow differences only if one side found nothing
    both = (fu >= 0) & (ff >= 0)
    assert both.mean() > 0.99
    assert np.array_equal(fu[both], ff[both])
    np.testing.assert_allclose(bu[both], bf[both], atol=1e-6)


# --------------------------------------------------------------------------- timing
def test_ico7_163842_queries_within_2s():
    mesh = icosphere(7)
    t0 = time.perf_counter()
    tree = SphereAABB(mesh)
    t_build = time.perf_counter() - t0
    q = random_unit(163842, 42)
    t0 = time.perf_counter()
    tree.closest_face(q[:16])  # JIT warm-up / cache load
    t_warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    f, _ = tree.closest_face(q)
    t_query = time.perf_counter() - t0
    print(f"ico7 build {t_build:.2f}s, first call {t_warm:.2f}s, 163842 queries {t_query:.3f}s")
    assert (f >= 0).all()
    assert t_query <= 2.0


# --------------------------------------------------------------------------- oracle
def _find_harness(tmp_path_factory):
    if not ORACLES.exists():
        pytest.skip("~/dev/biscuit/oracles not available")
    for rel in ("meshlib/bin/harness_aabb", "meshlib/harness_aabb", "harness_aabb"):
        p = ORACLES / rel
        if p.exists():
            return p
    cxx = shutil.which("clang++") or shutil.which("c++")
    if cxx is None or not (MESHLIB / "src").exists():
        pytest.skip("no harness_aabb binary and cannot build it")
    out = tmp_path_factory.mktemp("harness_aabb")
    res = subprocess.run(
        ["sh", str(REPO / "scripts" / "oracles" / "build_harness_aabb.sh"), str(out)],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "CXX": cxx},
    )
    if res.returncode != 0 or not (out / "harness_aabb").exists():
        pytest.skip(f"harness build failed: {res.stderr[-300:]}")
    return out / "harness_aabb"


@pytest.fixture(scope="session")
def harness(tmp_path_factory):
    return _find_harness(tmp_path_factory)


@pytest.fixture(scope="session")
def oracle_run(harness, tmp_path_factory):
    work = tmp_path_factory.mktemp("aabb_oracle")

    def run(mode, mesh, queries, deformed=None, refresh=False):
        tag = f"{abs(hash((mode, mesh.n_faces, len(queries), deformed is not None, refresh)))}"
        mp, qp, op = work / f"m{tag}.vtk", work / f"q{tag}.txt", work / f"o{tag}.txt"
        write_vtk(f32mesh(mesh), mp, dtype="float")
        np.savetxt(qp, queries, fmt="%.9g")
        argv = [str(harness), mode, str(mp), str(qp), str(op)]
        if deformed is not None:
            dp = work / f"d{tag}.vtk"
            write_vtk(f32mesh(deformed), dp, dtype="float")
            argv.append(str(dp))
            if refresh:
                argv.append("refresh")
        subprocess.run(argv, check=True, timeout=600)
        return read_expected(op, parse_mode(mode)[0])

    return run


TIE_REPORT: dict[str, tuple[int, int]] = {}


@pytest.mark.parametrize("level", [5, 6, 7])
def test_oracle_random_icosphere(oracle_run, level):
    mesh = f32mesh(icosphere(level))
    q = random_unit(100_000, 1000 + level)
    exp = oracle_run("closest", mesh, q)
    tree = SphereAABB(mesh)
    f, b, ties = tree.closest_face(q, return_ties=True)
    err = assert_matches("closest", (f, b), exp, f"ico{level}")
    TIE_REPORT[f"ico{level}"] = (int(ties.sum()), len(q))
    print(
        f"ico{level}: 0 face mismatches / {len(q)}, exact ties {ties.sum()}, max bary err {err:.2e}"
    )


@pytest.mark.parametrize("level", [5, 7])
def test_oracle_deformed_fresh_and_update(oracle_run, level):
    base = f32mesh(icosphere(level))
    defo = f32mesh(deformed_sphere(base))
    q = random_unit(100_000, 2000 + level)
    # fresh tree on the deformed mesh
    exp = oracle_run("closest", defo, q)
    f, b, ties = SphereAABB(defo).closest_face(q, return_ties=True)
    assert (f >= 0).all()
    e1 = assert_matches("closest", (f, b), exp, f"deformed{level}")
    # tree built on the sphere, then update() (as SphericalRemesh does)
    for refresh in (False, True):
        exp = oracle_run("closest", base, q, deformed=defo, refresh=refresh)
        tree = SphereAABB(base)
        tree.update(defo.vertices, refresh_face_normals=refresh)
        f2, b2, ties2 = tree.closest_face(q, return_ties=True)
        e2 = assert_matches("closest", (f2, b2), exp, f"deformed{level}-update-{refresh}")
        assert (f2 >= 0).all()
        print(f"deformed{level} update refresh={refresh}: bary err {e2:.2e}, ties {ties2.sum()}")
    print(f"deformed{level} fresh: bary err {e1:.2e}, ties {ties.sum()}")


@pytest.mark.parametrize("scheme", ["biscuit", "ugscnn"])
def test_oracle_structured_queries(oracle_run, scheme):
    mesh = f32mesh(icosphere(4, scheme=scheme))
    q = structured_queries(mesh)
    for mode in ("closest", "bool:0:0", "bool:0.005:0.005", "xcand:0:0", "xcand:0.005:0.002"):
        kind, e1, e2 = parse_mode(mode)
        exp = oracle_run(mode, mesh, q)
        got = python_result(mode, mesh, q)
        assert_matches(mode, got, exp, f"{scheme}/{mode}")


def test_oracle_uv_and_fixture_icosphere(oracle_run):
    from fixtures.meshes import icosphere as fixture_ico
    from fixtures.meshes import uv_sphere

    for name, mesh in (("fixture_ico4", fixture_ico(4)), ("uv", uv_sphere(20, 40))):
        mesh = f32mesh(mesh)
        q = np.vstack([random_unit(30_000, 9), structured_queries(mesh)])
        exp = oracle_run("closest", mesh, q)
        assert_matches("closest", python_result("closest", mesh, q), exp, name)


@pytest.mark.parametrize("eps", ["0:0", "0.01:0.01", "0.02:0.005"])
def test_oracle_eps_overloads(oracle_run, eps):
    mesh = f32mesh(icosphere(5))
    q = random_unit(50_000, 77)
    for kind in ("bool", "xcand", "search"):
        mode = f"{kind}:{eps}"
        exp = oracle_run(mode, mesh, q[:20_000] if kind == "search" else q)
        qq = q[:20_000] if kind == "search" else q
        assert_matches(mode, python_result(mode, mesh, qq), exp, mode)


def test_oracle_search_deformed(oracle_run):
    defo = f32mesh(deformed_sphere(icosphere(5)))
    q = random_unit(20_000, 5)
    mode = "search:0.01:0.02"
    assert_matches(mode, python_result(mode, defo, q), oracle_run(mode, defo, q), mode)
