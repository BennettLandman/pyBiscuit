"""Tests for pybiscuit.mesh.topology on inline synthetic meshes."""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import OPT
from pybiscuit.mesh import topology as T
from pybiscuit.mesh.model import Mesh


# ----------------------------------------------------------------- generators
def icosphere(level: int = 2):
    t = (1 + 5**0.5) / 2
    v = np.array(
        [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]],
        dtype=float,
    )  # fmt: skip
    f = np.array(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]],
    )  # fmt: skip
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(level):
        n = len(v)
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uk, inv = np.unique(e[:, 0] * n + e[:, 1], return_inverse=True)
        mids = v[uk // n] + v[uk % n]
        v = np.vstack([v, mids / np.linalg.norm(mids, axis=1, keepdims=True)])
        inv = inv.reshape(-1) + n
        m = len(f)
        ab, bc, ca = inv[:m], inv[m : 2 * m], inv[2 * m :]
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        f = np.concatenate(
            [np.stack(x, 1) for x in ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))]
        )
    return v, f


def uv_sphere(n_lat: int = 12, n_lon: int = 16):
    """Closed UV sphere with pole vertices; outward winding."""
    th = np.linspace(0, np.pi, n_lat + 1)[1:-1]
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    ring = np.array(
        [[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)] for t in th for p in ph]
    )
    v = np.vstack([[0, 0, 1], ring, [0, 0, -1]])
    south = len(v) - 1
    idx = lambda i, j: 1 + i * n_lon + (j % n_lon)  # noqa: E731
    f = []
    for j in range(n_lon):
        f.append([0, idx(0, j), idx(0, j + 1)])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            f += [[a, b, c], [a, c, d]]
    for j in range(n_lon):
        f.append([south, idx(n_lat - 2, j + 1), idx(n_lat - 2, j)])
    return v, np.array(f)


def torus(nu: int = 20, nv: int = 12, R: float = 2.0, r: float = 0.7):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    w = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    v = np.array([[(R + r * np.cos(b)) * np.cos(a), (R + r * np.cos(b)) * np.sin(a), r * np.sin(b)]
                  for a in u for b in w])  # fmt: skip
    f = []
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            f += [[a, b, c], [a, c, d]]
    return v, np.array(f)


def drop_faces(f, mask):
    return f[~mask]


def two_spheres():
    v, f = icosphere(1)
    return np.vstack([v, v + 5.0]), np.vstack([f, f + len(v)])


# ---------------------------------------------------------------------- tests
@pytest.mark.parametrize("maker", [lambda: icosphere(2), uv_sphere])
def test_sphere_chi2_genus0(maker):
    v, f = maker()
    s = T.topology_summary(v, f)
    assert s.euler_characteristic == 2 == T.euler_characteristic(v, f)
    assert s.genus == 0 == T.genus(v, f)
    assert s.n_components == 1 and s.n_boundary_loops == 0
    assert s.is_closed_manifold and s.is_consistently_oriented
    assert s.n_non_manifold_edges == 0 and s.n_unreferenced_vertices == 0


def test_mesh_input_equivalent():
    v, f = icosphere(1)
    assert T.euler_characteristic(Mesh(v, f)) == T.euler_characteristic(v, f) == 2


def test_torus_genus1():
    v, f = torus()
    s = T.topology_summary(Mesh(v, f))
    assert s.euler_characteristic == 0
    assert s.genus == 1
    assert s.is_closed_manifold and s.is_consistently_oriented


def test_edges_and_adjacency_icosphere():
    v, f = icosphere(2)
    et = T.unique_edges(v, f)
    assert len(et) == 3 * len(f) // 2
    assert (et.valence == 2).all()
    assert (et.edge_faces >= 0).all()
    # each edge's two faces really contain both vertices
    for e in (0, len(et) // 2, len(et) - 1):
        for fa in et.edge_faces[e]:
            assert set(et.edges[e]) <= set(f[fa])
    adj = T.face_adjacency(v, f)
    assert adj.shape == (len(f), 3) and (adj >= 0).all()
    # symmetric, and neighbour across edge k shares that edge
    for fi in (0, 7, len(f) - 1):
        for k in range(3):
            g = adj[fi, k]
            assert {f[fi, k], f[fi, (k + 1) % 3]} <= set(f[g])
            assert fi in adj[g]
    m = T.face_adjacency_matrix(v, f)
    assert (m != m.T).nnz == 0 and (np.asarray(m.sum(1)).ravel() == 3).all()


def test_vertex_adjacency_and_one_ring():
    v, f = icosphere(2)
    vf = T.vertex_faces(v, f)
    deg = vf.degree()
    assert set(np.unique(deg)) == {5, 6} and deg.sum() == 3 * len(f)
    nb = T.vertex_neighbors(v, f)
    assert (nb.degree() == deg).all()
    ring = T.one_ring(v, f)
    assert ring.valid.all() and not ring.is_boundary.any()
    for vi in (0, 3, 11, len(v) - 1):
        r = ring.ring(vi)
        assert set(r) == set(nb.row(vi)) and len(r) == deg[vi]
        # consecutive ring vertices form faces (vi, r[i], r[i+1]) with ring winding
        for i in range(len(r)):
            tri = (vi, r[i], r[(i + 1) % len(r)])
            assert any(tuple(np.roll(ff, -list(ff).index(vi))) == tri for ff in f[vf.row(vi)])


def test_one_ring_open_and_nonmanifold():
    v, f = uv_sphere()
    keep = v[f].mean(1)[:, 2] < 0.8
    fc = f[keep]
    ring = T.one_ring(v, fc)
    bverts = np.unique(np.concatenate(T.boundary_loops(v, fc)))
    assert ring.is_boundary[bverts].all()
    r = ring.ring(int(bverts[0]))
    nb = T.vertex_neighbors(v, fc).row(int(bverts[0]))
    assert len(r) == len(nb) and set(r) == set(nb) and len(set(r)) == len(r)
    # bowtie: two triangles sharing only a vertex -> pinch vertex is non-manifold
    bow = np.array([[0, 1, 2], [0, 3, 4]])
    assert list(T.non_manifold_vertices(5, bow)) == [0]
    # three triangles on one edge -> non-manifold edge
    fan = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    assert len(T.non_manifold_edges(5, fan)) == 1
    assert not T.topology_summary(5, fan).is_manifold


def test_boundary_loops_single_cap_removed():
    v, f = icosphere(3)
    fc = drop_faces(f, v[f].mean(1)[:, 2] > 0.8)
    loops = T.boundary_loops(v, fc)
    assert len(loops) == 1
    lp = loops[0]
    assert len(lp) == len(T.boundary_edges(v, fc)) and len(set(lp)) == len(lp)
    # consecutive loop vertices are edges
    et = T.unique_edges(v, fc)
    ek = {tuple(e) for e in et.edges}
    assert all((min(a, b), max(a, b)) in ek for a, b in zip(lp, np.roll(lp, -1), strict=True))
    s = T.topology_summary(v, fc)
    assert s.euler_characteristic == 1 and s.genus == 0 and not s.is_closed
    assert s.n_unreferenced_vertices > 0  # cap interior vertices


def test_boundary_loops_two_caps_removed():
    v, f = icosphere(3)
    z = v[f].mean(1)[:, 2]
    fc = drop_faces(f, np.abs(z) > 0.8)
    loops = T.boundary_loops(v, fc)
    assert len(loops) == 2
    s = T.topology_summary(v, fc)
    assert s.n_boundary_loops == 2 and s.euler_characteristic == 0 and s.genus == 0
    assert s.n_components == 1
    # opposite caps are disjoint vertex sets
    assert not set(loops[0]) & set(loops[1])


def test_torus_with_hole_genus():
    v, f = torus()
    fc = f[1:]  # drop one face -> one boundary loop of 3 edges
    s = T.topology_summary(v, fc)
    assert s.n_boundary_loops == 1 and s.genus == 1


def test_orientation_flags_single_flipped_face():
    for v, f in (icosphere(2), torus()):
        assert T.check_orientation(v, f).is_consistent
        g = f.copy()
        k = 17
        g[k] = g[k][::-1]
        rep = T.check_orientation(v, g)
        assert not rep.is_consistent and rep.is_orientable
        assert list(rep.flipped_faces) == [k]
        assert len(rep.inconsistent_edges) == 3
        s = T.topology_summary(v, g)
        assert not s.is_consistently_oriented
        # flipping reported faces restores consistency
        g[rep.flipped_faces] = g[rep.flipped_faces][:, ::-1]
        assert T.check_orientation(v, g).is_consistent


def test_orientation_whole_component_flipped_is_consistent():
    v, f = icosphere(1)
    assert T.check_orientation(v, f[:, ::-1]).is_consistent


def test_orientation_moebius_not_orientable():
    n = 8
    f = []
    for i in range(n - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        f += [[a, b, c], [b, d, c]]
    # close with a twist
    a, b = 2 * (n - 1), 2 * (n - 1) + 1
    f += [[a, b, 1], [b, 0, 1]]
    rep = T.check_orientation(2 * n, np.array(f))
    assert not rep.is_orientable


def test_components_two_spheres():
    v, f = two_spheres()
    n, vl, fl = T.connected_components(v, f)
    assert n == 2
    assert (vl[: len(v) // 2] == 0).all() and (vl[len(v) // 2 :] == 1).all()
    assert np.bincount(fl).tolist() == [len(f) // 2] * 2
    s = T.topology_summary(v, f)
    assert s.euler_characteristic == 4 and s.genus == 0 and s.n_components == 2
    assert list(s.component_euler) == [2, 2]


def test_unreferenced_vertices():
    v, f = icosphere(1)
    v2 = np.vstack([v, [[9.0, 9, 9]]])
    assert list(T.unreferenced_vertices(v2, f)) == [len(v)]
    n, vl, _ = T.connected_components(v2, f)
    assert n == 1 and vl[-1] == -1
    assert T.euler_characteristic(v2, f) == 2


def test_empty_faces():
    s = T.topology_summary(np.zeros((3, 3)), np.zeros((0, 3), dtype=int))
    assert s.n_components == 0 and s.euler_characteristic == 0


def test_performance_large_mesh():
    v, f = icosphere(7)  # 327,680 faces
    assert len(f) == 327680
    t0 = time.perf_counter()
    s = T.topology_summary(v, f)
    ring = T.one_ring(v, f)
    dt = time.perf_counter() - t0
    assert s.euler_characteristic == 2 and s.genus == 0 and ring.valid.all()
    assert dt < 10.0  # target ~1-2 s; generous bound to keep CI stable


# ------------------------------------------------------------- reference data
def _read_ascii_vtk(path):
    """Minimal ASCII legacy-VTK polydata reader (points + triangle polygons)."""
    toks = path.read_text().split()
    ip = toks.index("POINTS")
    n = int(toks[ip + 1])
    pts = np.array(toks[ip + 3 : ip + 3 + 3 * n], dtype=float).reshape(n, 3)
    ig = toks.index("POLYGONS")
    m = int(toks[ig + 1])
    cells = np.array(toks[ig + 3 : ig + 3 + 4 * m], dtype=np.int64).reshape(m, 4)
    assert (cells[:, 0] == 3).all()
    return pts, cells[:, 1:]


def _load(path):
    try:
        from pybiscuit.io.vtk import read_vtk

        m = read_vtk(path)
        return m.vertices, m.faces
    except Exception:  # noqa: BLE001 - fall back to the private parser
        return _read_ascii_vtk(path)


@pytest.mark.reference
@pytest.mark.parametrize("n", [5120, 20480, 81920, 327680])
def test_reference_spheres(n):
    v, f = _load(OPT / "sphere" / f"sphere_{n}.vtk")
    s = T.topology_summary(v, f)
    assert len(f) == n
    assert s.euler_characteristic == 2 and s.genus == 0
    assert s.is_closed_manifold and s.n_components == 1
    assert s.is_consistently_oriented


@pytest.mark.reference
@pytest.mark.parametrize("subject", ["hcp1058", "mb101", "oasis45"])
@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_reference_template_white(subject, hemi):
    v, f = _load(OPT / "reg" / subject / f"{hemi}.white.vtk")
    s = T.topology_summary(v, f)
    assert s.euler_characteristic == 2 and s.genus == 0
    assert s.n_components == 1 and s.n_boundary_loops == 0
    assert s.is_closed_manifold
