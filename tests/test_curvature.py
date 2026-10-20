"""B2.4b: MeshLib curvature port (tensor, smoothingTensor, principalCurvature, curvature, quad).

Oracle: ``scripts/oracles/harness_curvature.cpp`` linked against the MeshLib sources.  Committed
goldens (``tests/data/golden/meshlib/curv_*``) need no binary; the live test on
``reg/mb101/lh.white.vtk`` needs ``reference/`` and a harness binary (skipped when absent).
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import REG_DIR
from fixtures.meshes import ellipsoid, icosphere
from pybiscuit.io.vtk import read_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.mesh import curvature as C
from pybiscuit.mesh.geometry import face_normals, vertex_neighbor_lists, vertex_normals

GOLD = Path(__file__).parent / "data" / "golden" / "meshlib"
CASES = sorted(p.name for p in GOLD.glob("curv_*"))
ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
HARNESS_CANDIDATES = [
    Path(os.environ.get("BISCUIT_HARNESS_CURVATURE", "/nonexistent")),
    ORACLES / "meshlib-b24b" / "harness_curvature",
]
RTOL64 = 1e-10  # acceptance (double paths)
RTOL32 = 1e-5  # acceptance (float32 paths)


def _eq(a, b, rtol):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=0, equal_nan=True)


def _dir_diff(a, b, keep):
    """Largest component difference of unit vectors up to sign, on `keep` vertices."""
    a, b = a[keep], b[keep]
    d = np.minimum(np.abs(a - b).max(1), np.abs(a + b).max(1))
    d = d[~(np.isnan(a).any(1) | np.isnan(b).any(1))]
    return d.max() if d.size else 0.0


def _compare_all(mesh, g):
    """Assert the port against the dumps `g` (dict-like of oracle arrays)."""
    n = mesh.n_vertices
    nb = vertex_neighbor_lists(mesh.faces, n)
    assert np.array_equal(nb.indices, g["nbr"])
    assert np.array_equal(np.diff(nb.indptr), g["nbr_count"])
    # float32 normals are bit-exact with a non-fused build of MeshLib
    assert np.array_equal(face_normals(mesh), g["fnormal"].reshape(-1, 3))
    assert np.array_equal(vertex_normals(mesh), g["vnormal"].reshape(-1, 3))
    stats = {}
    for it in sorted({int(k.split("_it")[1]) for k in g if k.startswith("T_it")}):
        T = C.tensor(mesh, neighbors=nb)
        if it:
            T = C.smoothing_tensor(T, mesh, iterations=it, neighbors=nb)
        Tg = g[f"T_it{it}"].reshape(n, 3, 3)
        _eq(T, Tg, RTOL64)
        flat = (np.abs(Tg) < 1e-6).all(axis=(1, 2))
        c = C.principal_curvature(T)
        _eq(c.cmin, g[f"cmin_it{it}"], RTOL64)
        _eq(c.cmax, g[f"cmax_it{it}"], RTOL64)
        for nm in ("umin", "umax"):
            ref = g[f"{nm}_it{it}"].reshape(n, 3)
            assert _dir_diff(getattr(c, nm), ref, ~flat) <= 1e-10
        _eq(C.mean_curvature(c), g[f"mean_it{it}"], RTOL64)
        _eq(C.gaussian_curvature(c), g[f"gauss_it{it}"], RTOL64)
        # public double overload == staged path
        c2 = C.curvature(mesh, n_iter_smoothing_tensor=it, neighbors=nb)
        _eq(c2.cmin, g[f"pub_cmin_it{it}"], RTOL64)
        _eq(c2.cmax, g[f"pub_cmax_it{it}"], RTOL64)
        # float overload: computed in double, cast to float32 (flat vertices' directions undefined)
        cf = C.curvature(mesh, n_iter_smoothing_tensor=it, dtype=np.float32, neighbors=nb)
        assert cf.cmin.dtype == np.float32 and cf.umin.dtype == np.float32
        _eq(cf.cmin, g[f"f_cmin_it{it}"], RTOL32)
        _eq(cf.cmax, g[f"f_cmax_it{it}"], RTOL32)
        _eq(cf.cmin, g[f"fnull_cmin_it{it}"], RTOL32)
        for nm in ("umin", "umax"):
            ref = g[f"f_{nm}_it{it}"].reshape(n, 3)
            assert _dir_diff(getattr(cf, nm), ref, ~flat) <= 1e-5
        stats[it] = int(flat.sum())
    q = C.curvature_quad(mesh, neighbors=nb)
    _eq(q.cmin, g["q_cmin"], RTOL64)
    _eq(q.cmax, g["q_cmax"], RTOL64)
    assert _dir_diff(q.umin, g["q_umin"].reshape(n, 3), np.ones(n, bool)) <= 1e-10
    assert _dir_diff(q.umax, g["q_umax"].reshape(n, 3), np.ones(n, bool)) <= 1e-10
    qf = C.curvature_quad(mesh, dtype=np.float32, neighbors=nb)
    _eq(qf.cmin, g["qf_cmin"], RTOL32)
    _eq(qf.cmax, g["qf_cmax"], RTOL32)
    assert _dir_diff(qf.umin, g["qf_umin"].reshape(n, 3), np.ones(n, bool)) <= 1e-5
    return stats


@pytest.mark.parametrize("case", CASES)
def test_matches_committed_oracle_dumps(case):
    mesh = read_vtk(GOLD / case / "input.vtk")
    with np.load(GOLD / case / "golden.npz") as g:
        _compare_all(mesh, g)


def test_goldens_have_provenance():
    assert CASES, "no curv_* goldens committed"
    import json

    for case in CASES:
        meta = json.loads((GOLD / case / "command.json").read_text())
        assert meta["oracle"] == "upstream-macos-arm64" and meta["commit"]
        assert sum(p.stat().st_size for p in (GOLD / case).iterdir()) < 1_000_000


# --------------------------------------------------------------------------- #
# behaviour checks that need no oracle
# --------------------------------------------------------------------------- #
def test_tensor_is_symmetric_and_scale_invariant():
    m = ellipsoid()
    T = C.tensor(m)
    assert np.array_equal(T, T.transpose(0, 2, 1))
    m2 = Mesh(m.vertices * 2.0, m.faces)  # power of two: float32 scaling is exact
    # beta * len / mean_len: invariant under uniform scaling
    np.testing.assert_allclose(C.tensor(m2), T, rtol=1e-12, atol=0)


def test_convex_positive_and_orientation_flip_negates():
    m = icosphere(4)
    c = C.curvature(m)
    assert (c.cmin > 0).all() and (c.cmax > 0).all() and (c.cmin <= c.cmax).all()
    flipped = Mesh(m.vertices, m.faces[:, ::-1])
    cf = C.curvature(flipped)
    assert (cf.cmax < 0).all()


@pytest.mark.parametrize("radius", [1.0, 2.5])
def test_quad_fit_recovers_inverse_radius(radius):
    q = C.curvature_quad(icosphere(4, radius))
    np.testing.assert_allclose(q.cmin.mean(), 1 / radius, rtol=0.03)
    np.testing.assert_allclose(q.cmax.mean(), 1 / radius, rtol=0.03)
    assert q.cmin.std() / q.cmin.mean() < 0.1


def test_flat_vertices_get_zero_curvature():
    # planar triangulated square: no dihedral angle anywhere -> every vertex is "flat"
    n = 5
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    v = np.c_[xs.ravel(), ys.ravel(), np.zeros(n * n)]
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            f += [(a, b, d), (a, d, c)]
    c = C.curvature(Mesh(v, np.array(f)))
    assert not c.cmin.any() and not c.cmax.any() and not c.umin.any() and not c.umax.any()


def test_smoothing_tensor_zero_iterations_and_mean_preserving_shape():
    m = ellipsoid()
    T = C.tensor(m)
    assert np.array_equal(C.smoothing_tensor(T, m, iterations=0), T)
    Ts = C.smoothing_tensor(T, m, iterations=2)
    assert Ts.shape == T.shape and Ts.std() < T.std()


def test_coerce_accepts_arrays_and_mesh():
    m = icosphere(2)
    a = C.curvature(m)
    b = C.curvature(m.vertices, m.faces)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    with pytest.raises(TypeError):
        C.curvature(m.vertices)


# --------------------------------------------------------------------------- #
# live oracle on a real white surface (reference + harness binary required)
# --------------------------------------------------------------------------- #
def _harness():
    for p in HARNESS_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.mark.reference
def test_lh_white_matches_live_oracle(tmp_path):
    harness = _harness()
    if harness is None:
        pytest.skip("harness_curvature binary not built (scripts/oracles/harness_curvature.cpp)")
    vtk = REG_DIR / "mb101" / "lh.white.vtk"
    subprocess.run([str(harness), str(vtk), str(tmp_path), "0,3", ""], check=True, timeout=600)
    g = {}
    for p in tmp_path.iterdir():
        dt = {"f32": "<f4", "f64": "<f8", "i32": "<i4"}[p.suffix[1:]]
        g[p.stem] = np.fromfile(p, dtype=dt)
    stats = _compare_all(read_vtk(vtk), g)
    assert set(stats) == {0, 3}
