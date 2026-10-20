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

from _compare import REL32, REL64, assert_close_float32, assert_close_float64
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
RTOL64 = REL64  # acceptance (double paths)
RTOL32 = REL32  # acceptance (float32 paths)
# Direction vectors are only as well conditioned as the eigen-gap of the tensor allows
# (first-order perturbation: |d(u)| ~ |dT| / gap), and the discrete eigenvalue/principal-curvature
# *selection* flips when two magnitudes tie to within the platform noise.  Directions are therefore
# compared with a per-vertex bound DIR_C * delta / gap, and vertices with gap < TIE_FACTOR * delta
# are counted as ties and skipped (reported, never silent).
LAST_TIES: dict = {}  # tie counts of the last _compare_all (reported, see the b03 report)
DIR_C = 10.0
TIE_FACTOR = 1e3


def _eq(a, b, rtol, label="values"):
    """Tolerance comparison (cross-platform policy): see tests/_compare.py."""
    check = assert_close_float32 if rtol >= 1e-6 else assert_close_float64
    check(a, b, rel=rtol, label=label)


def _tensor_gap(Tg):
    """Relative eigen-gap of each golden tensor: min over (|lambda| ordering, eigenvalue) gaps.

    Margin only (recomputed from the golden with ``numpy.linalg.eigvalsh``); the port itself never
    calls it.  Small gap <=> the eigenvector choice is a tie that may flip across platforms.
    """
    lam = np.linalg.eigvalsh(Tg)  # ascending, (n, 3)
    mag = np.sort(np.abs(lam), axis=1)
    scale = np.maximum(mag[:, 2], 1e-300)
    g_mag = np.minimum(mag[:, 1] - mag[:, 0], mag[:, 2] - mag[:, 1])
    g_val = np.minimum(lam[:, 1] - lam[:, 0], lam[:, 2] - lam[:, 1])
    return np.minimum(g_mag, g_val) / scale


def _quad_gap(cmin, cmax):
    """Relative gap of the 2x2 Hessian eigenvalues (selection is by magnitude)."""
    scale = np.maximum(np.maximum(np.abs(cmin), np.abs(cmax)), 1e-300)
    return np.minimum(np.abs(cmax) - np.abs(cmin), np.abs(cmax - cmin)) / scale


def _dir_diff(a, b, keep):
    """Per-vertex largest component difference of unit vectors up to sign (NaN -> 0)."""
    d = np.minimum(np.abs(a - b).max(1), np.abs(a + b).max(1))
    d[np.isnan(a).any(1) | np.isnan(b).any(1)] = 0.0
    return np.where(keep, d, 0.0)


def _frame_noise(vnormal):
    """Direction slack of the quadric-fit frame (``curvature_quad``).

    The frame is the rotation taking the float32 vertex normal to -x with
    ``theta = acos(-n_x)``.  One float32 ulp of ``n_x`` (6e-8, libm acosf differences move the
    normals by that much) changes ``theta`` by ``ulp / sin(theta)``; where the normal is within
    ~1e-4 of the x axis that is capped by ``sqrt(2 ulp)`` ~ 4e-4.  This is intrinsic to the C++
    algorithm (a float32 normal fed to a double ``acos``), not to the port.
    """
    nx = np.asarray(vnormal, np.float64).reshape(-1, 3)[:, 0]
    sin_t = np.sqrt(np.maximum(1.0 - nx * nx, 0.0))
    return np.minimum(1.2e-7 / np.maximum(sin_t, 1e-12), 4e-4)


def _frame_degenerate(vnormal):
    """Vertices whose float32 normal is (within 1.5 ulp) along the quadric-fit reference axis.

    There ``curvature_quad`` takes ``theta = acos(+-1)``: for ``n = +x`` exactly the rotation axis
    ``n x (-x)`` is the zero vector and ``Coordinate::rotation`` returns the identity (the fit then
    sees the surface from the wrong side, curvature sign flipped -- the oracle's behaviour, present
    in the committed goldens), whereas one ulp of platform noise in ``n_y, n_z`` gives a proper
    rotation by pi.  ``acos`` of an argument rounded just past 1 is NaN.  Both are a float32
    rounding decision, so these vertices are counted as ties and not compared.
    """
    nx = np.asarray(vnormal, np.float64).reshape(-1, 3)[:, 0]
    return np.abs(nx) >= 1.0 - 3e-7 if nx.size else np.zeros(0, bool)


def _eq_masked(a, b, keep, rtol, label):
    _eq(np.asarray(a)[keep], np.asarray(b)[keep], rtol, label)


def _check_dirs(a, b, keep, gap, delta, label, slack=0.0):
    """Direction check modulo ties; returns the number of tie vertices skipped."""
    d = _dir_diff(np.asarray(a, np.float64), np.asarray(b, np.float64), keep)
    tie = keep & (gap < TIE_FACTOR * delta)
    ok = d <= np.maximum(DIR_C * delta / np.maximum(gap, 1e-300), DIR_C * delta) + 2 * slack
    bad = np.flatnonzero(keep & ~tie & ~ok)
    assert bad.size == 0, (
        f"{label}: {bad.size} untied direction mismatches, first {bad[:5].tolist()}, "
        f"diff {d[bad][:5].tolist()}, gap {gap[bad][:5].tolist()}"
    )
    return int(tie.sum())


def _check_normals(mesh, g):
    """Float32 normals vs the arm64 dump, to 1e-5 (libm acosf / sqrtf may differ by an ulp)."""
    assert_close_float32(face_normals(mesh), g["fnormal"].reshape(-1, 3), label="face_normals")
    assert_close_float32(vertex_normals(mesh), g["vnormal"].reshape(-1, 3), label="vertex_normals")


def _compare_all(mesh, g):
    """Assert the port against the dumps `g` (dict-like of oracle arrays)."""
    n = mesh.n_vertices
    nb = vertex_neighbor_lists(mesh.faces, n)
    assert np.array_equal(nb.indices, g["nbr"])  # integer data: exact
    assert np.array_equal(np.diff(nb.indptr), g["nbr_count"])
    _check_normals(mesh, g)
    stats = {}
    ties = LAST_TIES
    ties.clear()
    for it in sorted({int(k.split("_it")[1]) for k in g if k.startswith("T_it")}):
        T = C.tensor(mesh, neighbors=nb)
        if it:
            T = C.smoothing_tensor(T, mesh, iterations=it, neighbors=nb)
        Tg = g[f"T_it{it}"].reshape(n, 3, 3)
        _eq(T, Tg, RTOL64, f"T_it{it}")
        flat = (np.abs(Tg) < 1e-6).all(axis=(1, 2))
        gap = _tensor_gap(Tg)
        c = C.principal_curvature(T)
        _eq(c.cmin, g[f"cmin_it{it}"], RTOL64, f"cmin_it{it}")
        _eq(c.cmax, g[f"cmax_it{it}"], RTOL64, f"cmax_it{it}")
        for nm in ("umin", "umax"):
            ref = g[f"{nm}_it{it}"].reshape(n, 3)
            ties[nm, it] = _check_dirs(getattr(c, nm), ref, ~flat, gap, RTOL64, f"{nm}_it{it}")
        _eq(C.mean_curvature(c), g[f"mean_it{it}"], RTOL64, f"mean_it{it}")
        _eq(C.gaussian_curvature(c), g[f"gauss_it{it}"], RTOL64, f"gauss_it{it}")
        # public double overload == staged path
        c2 = C.curvature(mesh, n_iter_smoothing_tensor=it, neighbors=nb)
        _eq(c2.cmin, g[f"pub_cmin_it{it}"], RTOL64, f"pub_cmin_it{it}")
        _eq(c2.cmax, g[f"pub_cmax_it{it}"], RTOL64, f"pub_cmax_it{it}")
        # float overload: computed in double, cast to float32 (flat vertices' directions undefined)
        cf = C.curvature(mesh, n_iter_smoothing_tensor=it, dtype=np.float32, neighbors=nb)
        assert cf.cmin.dtype == np.float32 and cf.umin.dtype == np.float32
        _eq(cf.cmin, g[f"f_cmin_it{it}"], RTOL32, f"f_cmin_it{it}")
        _eq(cf.cmax, g[f"f_cmax_it{it}"], RTOL32, f"f_cmax_it{it}")
        _eq(cf.cmin, g[f"fnull_cmin_it{it}"], RTOL32, f"fnull_cmin_it{it}")
        for nm in ("umin", "umax"):
            ref = g[f"f_{nm}_it{it}"].reshape(n, 3)
            _check_dirs(getattr(cf, nm), ref, ~flat, gap, RTOL32, f"f_{nm}_it{it}")
        stats[it] = int(flat.sum())
    # Quadric fit: a float64 computation, but it consumes the *float32* vertex normals (frame
    # rotation), which differ by an ulp across libm implementations -> float32-class tolerance.
    q = C.curvature_quad(mesh, neighbors=nb)
    degenerate = _frame_degenerate(g["vnormal"])
    allv = ~degenerate
    LAST_TIES["quad_frame_degenerate"] = int(degenerate.sum())
    _eq_masked(q.cmin, g["q_cmin"], allv, RTOL32, "q_cmin")
    _eq_masked(q.cmax, g["q_cmax"], allv, RTOL32, "q_cmax")
    qgap = _quad_gap(g["q_cmin"], g["q_cmax"])
    slack = _frame_noise(g["vnormal"])
    _check_dirs(q.umin, g["q_umin"].reshape(n, 3), allv, qgap, RTOL32, "q_umin", slack)
    _check_dirs(q.umax, g["q_umax"].reshape(n, 3), allv, qgap, RTOL32, "q_umax", slack)
    qf = C.curvature_quad(mesh, dtype=np.float32, neighbors=nb)
    _eq_masked(qf.cmin, g["qf_cmin"], allv, RTOL32, "qf_cmin")
    _eq_masked(qf.cmax, g["qf_cmax"], allv, RTOL32, "qf_cmax")
    _check_dirs(qf.umin, g["qf_umin"].reshape(n, 3), allv, qgap, RTOL32, "qf_umin", slack)
    return stats


@pytest.mark.parametrize("case", CASES)
def test_matches_committed_oracle_dumps(case):
    mesh = read_vtk(GOLD / case / "input.vtk")
    with np.load(GOLD / case / "golden.npz") as g:
        _compare_all(mesh, g)


@pytest.mark.bitexact
@pytest.mark.parametrize("case", CASES)
def test_normals_bitexact_with_arm64_oracle(case):
    """Diagnostic only (``pytest -m bitexact``): float32 normals are bit-identical with a non-fused
    MeshLib build on the oracle platform (arm64).  Not expected to hold on x86-64 / other libm."""
    mesh = read_vtk(GOLD / case / "input.vtk")
    with np.load(GOLD / case / "golden.npz") as g:
        assert np.array_equal(face_normals(mesh), g["fnormal"].reshape(-1, 3))
        assert np.array_equal(vertex_normals(mesh), g["vnormal"].reshape(-1, 3))


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
