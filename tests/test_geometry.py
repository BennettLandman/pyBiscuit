"""Tests for pybiscuit.mesh.geometry (MeshLib normals / areas / smoothing port, row B2.4a).

Three layers: analytic unit tests, committed MeshLib goldens (no binaries needed),
and live comparison with the oracle harness (skipped when ~/dev/biscuit/oracles or a
compiler is absent).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import OPT, REPO
from fixtures.meshes import cut_sphere, ellipsoid, folded_sphere, icosphere, perturb, uv_sphere
from pybiscuit.io import read_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.mesh import geometry as G

ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
GOLDEN = REPO / "tests" / "data" / "golden" / "meshlib"
HARNESS_SRC = REPO / "scripts" / "oracles" / "harness_geometry.cpp"
TOL32 = 1e-5  # brief 02 B2.4: float32 paths
TOL64 = 1e-10  # double paths


# ----------------------------------------------------------------- helpers
def write_ascii_vtk(path, vertices, faces) -> None:
    v = np.asarray(vertices, dtype=np.float32)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n")
        fh.write(f"POINTS {len(v)} float\n")
        fh.write("\n".join(" ".join(f"{x:.9g}" for x in r) for r in v) + "\n")
        fh.write(f"POLYGONS {len(faces)} {4 * len(faces)}\n")
        fh.write("\n".join(f"3 {r[0]} {r[1]} {r[2]}" for r in faces) + "\n")


def vec_rel(a, b) -> float:
    """max over vertices of ||a - b|| / ||b|| (b must be non-degenerate)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    nb = np.maximum(np.linalg.norm(b, axis=1), 1e-30)
    return float(np.max(np.linalg.norm(a - b, axis=1) / nb))


def sca_rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))


def load(d: Path, name: str, dtype: str, shape=None):
    x = np.fromfile(d / name, dtype=dtype)
    return x.reshape(shape) if shape else x


def check_outputs(d: Path, argv: list[str], vertices, faces, tol32=TOL32, tol64=TOL64) -> dict:
    """Compare every harness output in ``d`` with the Python port; returns max errors."""
    v = np.asarray(vertices, np.float32)
    err: dict[str, float] = {}
    nb = G.vertex_neighbor_lists(faces, len(v))
    assert np.array_equal(nb.indptr, load(d, "nbr_indptr.i32", "<i4"))
    assert np.array_equal(nb.indices, load(d, "nbr_indices.i32", "<i4"))
    err["face_normals"] = vec_rel(
        G.face_normals(v, faces), load(d, "face_normals.f32", "<f4", (-1, 3))
    )
    err["vertex_normals"] = vec_rel(
        G.vertex_normals(v, faces), load(d, "vertex_normals.f32", "<f4", (-1, 3))
    )
    err["face_areas"] = sca_rel(G.face_areas(v, faces), load(d, "face_areas.f32", "<f4"))
    # vertex areas: float32 edge lengths / angles feed a double sum -> float32-class path
    err["vertex_areas"] = sca_rel(G.vertex_areas(v, faces, nb), load(d, "vertex_areas.f64", "<f8"))
    i, ops = 3, argv
    while i < len(ops):
        op = ops[i]
        if op == "-s":
            it = int(ops[i + 1])
            pre = f"smooth_i{it}_"
            r = G.smooth_vertices(v, faces, it, neighbors=nb)
            err[pre + "vertices"] = vec_rel(r, load(d, pre + "vertices.f32", "<f4", (-1, 3)))
            err[pre + "normals"] = vec_rel(
                G.vertex_normals(r, faces), load(d, pre + "vertex_normals.f32", "<f4", (-1, 3))
            )
            i += 2
        elif op == "-g":
            sg, it = float(ops[i + 1]), int(ops[i + 2])
            pre = f"smoothsig_s{np.float32(sg):g}_i{it}_"
            r = G.smooth_vertices(v, faces, it, sigma=sg, neighbors=nb)
            err[pre + "vertices"] = vec_rel(r, load(d, pre + "vertices.f32", "<f4", (-1, 3)))
            i += 3
        elif op in ("-df", "-dd"):
            i += 3  # data ops are checked by check_data_ops
        else:
            raise AssertionError(op)
    return err


def check_data_ops(d: Path, argv: list[str], faces, orig: dict[str, np.ndarray]) -> dict:
    out = {}
    i = 3
    while i < len(argv):
        op = argv[i]
        if op in ("-s",):
            i += 2
        elif op == "-g":
            i += 3
        else:
            it, src = int(argv[i + 1]), argv[i + 2]
            kind = "f32" if op == "-df" else "f64"
            res = load(d, f"data_{kind}_i{it}.{kind}", "<f4" if kind == "f32" else "<f8")
            mine = G.smooth_data(faces, orig[src], it)
            assert mine.dtype == res.dtype
            out[f"data_{kind}"] = float(np.max(np.abs(mine - res) / np.maximum(np.abs(res), 1e-3)))
            i += 3
    return out


def run_case(d: Path, tol32=TOL32):
    cmd = json.loads((d / "command.json").read_text())
    argv = cmd["argv"]
    m = read_vtk(d / "input.vtk")
    orig = {"data.f32": load(d, "data.f32", "<f4"), "data.f64": load(d, "data.f64", "<f8")}
    err = check_outputs(d, argv, m.vertices, m.faces)
    err.update(check_data_ops(d, argv, m.faces, orig))
    bad = {k: x for k, x in err.items() if x > tol32}
    assert not bad, bad
    return err


# ----------------------------------------------------------------- analytic unit tests
def test_single_triangle_normals_and_area():
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], np.float64)
    f = np.array([[0, 1, 2]])
    assert np.allclose(G.face_normals(v, f), [[0, 0, 1]])
    assert np.allclose(G.vertex_normals(v, f), [[0, 0, 1]] * 3)
    assert G.face_areas(v, f).tolist() == [3.0]
    # MeshLib quirk: the ring of a boundary vertex wraps last->first, so a lone triangle
    # is counted twice at each corner: 2 * area / 3 = 2
    assert np.allclose(G.vertex_areas(v, f), [2.0, 2.0, 2.0])
    assert G.face_normals(v, f).dtype == np.float32


def test_degenerate_triangle_has_zero_normal():
    v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float64)
    f = np.array([[0, 1, 2]])
    assert np.all(G.face_normals(v, f) == 0)
    assert np.all(G.vertex_normals(v, f) == 0)


def test_sphere_normals_point_outward_and_areas_sum():
    m = icosphere(4)
    vn = G.vertex_normals(m)
    assert vn.dtype == np.float32
    assert np.min(np.sum(vn * m.vertices, axis=1)) > 0.999
    assert np.allclose(np.linalg.norm(vn, axis=1), 1, atol=2e-7)
    fa = G.face_areas(m)
    assert abs(float(fa.sum(dtype=np.float64)) - 4 * np.pi) < 0.05
    va = G.vertex_areas(m)
    assert va.dtype == np.float64
    assert abs(va.sum() - fa.sum(dtype=np.float64)) < 1e-5  # 1/3 of each triangle per corner


def test_vertex_area_closed_fan_equals_third_of_incident_areas():
    m = uv_sphere(12, 24)
    f = m.faces.astype(np.int64)
    fa = G.face_areas(m).astype(np.float64)
    ref = np.zeros(m.n_vertices)
    for k in range(3):
        np.add.at(ref, f[:, k], fa / 3)
    assert np.allclose(G.vertex_areas(m), ref, rtol=1e-5)


def test_neighbor_lists_closed_and_open():
    m = icosphere(2)
    nb = G.vertex_neighbor_lists(m.faces, m.n_vertices)
    deg = np.diff(nb.indptr)
    assert deg.min() == 5 and deg.max() == 6 and (deg == 5).sum() == 12
    for i in (0, 5, 17):
        ring = nb.row(i)
        assert len(set(ring)) == len(ring)
        for a, b in zip(ring, np.roll(ring, -1), strict=True):  # consecutive entries share a face
            fs = {tuple(sorted(t)) for t in m.faces if i in t and a in t and b in t}
            assert fs
    # open fan: single triangle fan around vertex 0 gets the extra closing vertex
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    nb = G.vertex_neighbor_lists(f, 4)
    assert nb.row(0).tolist() == [1, 2, 3]
    assert nb.row(2).tolist() == [3, 0, 1] or sorted(nb.row(2)) == [0, 1, 3]


def test_cut_sphere_boundary_ring_has_extra_vertex():
    m = cut_sphere(3)
    nb = G.vertex_neighbor_lists(m.faces, m.n_vertices)
    from pybiscuit.mesh.topology import boundary_loops, vertex_neighbors

    bv = np.concatenate(boundary_loops(m))
    topo = vertex_neighbors(m)
    # MeshLib lists contain exactly the topological neighbours (as a set) on manifold meshes
    for i in list(bv[:10]) + [0, 1, 2]:
        assert set(nb.row(i).tolist()) == set(topo.row(i).tolist())
        assert len(nb.row(i)) == len(topo.row(i))


def test_smooth_constants_and_input_not_modified():
    m = perturb(icosphere(2), 0.01, seed=0)
    v0 = m.vertices.copy()
    r = G.smooth_vertices(m, iterations=3)
    assert np.array_equal(m.vertices, v0)
    assert r.dtype == np.float32 and r.shape == v0.shape
    c = np.full(m.n_vertices, 2.5, np.float32)
    assert np.allclose(G.smooth_data(m, c, 4), 2.5)
    assert np.array_equal(G.smooth_vertices(m, iterations=0), v0.astype(np.float32))


def test_smooth_data_dtypes_and_manual_jacobi():
    m = icosphere(2)
    nb = G.vertex_neighbor_lists(m.faces, m.n_vertices)
    rng = np.random.default_rng(0)
    d = rng.standard_normal(m.n_vertices)
    r = G.smooth_data(m.faces, d, 1)
    assert r.dtype == np.float64
    man = np.array([d[nb.row(i)].mean() for i in range(m.n_vertices)])
    assert np.allclose(r, man, rtol=1e-12)
    assert G.smooth_data(m.faces, d.astype(np.float32), 1).dtype == np.float32
    assert G.smooth_data(m.faces, np.arange(m.n_vertices), 1).dtype == np.float64  # ints -> double
    with pytest.raises(ValueError):
        G.smooth_data(m.faces, np.zeros((m.n_vertices, 2)), 1)


def test_sigma_limit_matches_mean_for_large_sigma():
    # sigma -> infinity: Gaussian weights become uniform, equal to the plain mean
    m = icosphere(2)
    a = G.smooth_vertices(m, iterations=1)
    b = G.smooth_vertices(m, iterations=1, sigma=1e3)
    assert np.allclose(a, b, atol=1e-6)


def test_stiff_sigma_gives_nan_like_meshlib():
    m = icosphere(2)
    r = G.smooth_vertices(m, iterations=1, sigma=1e-3)  # all weights underflow -> 0/0
    assert np.isnan(r).all()


def test_input_validation():
    with pytest.raises(TypeError):
        G.face_normals(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        G.face_normals(np.zeros((3, 3)), np.array([[0, 1, 5]]))
    with pytest.raises(ValueError):
        G.face_normals(np.zeros((3, 2)), np.array([[0, 1, 2]]))


def test_accepts_mesh_and_arrays():
    m = icosphere(1)
    assert np.array_equal(G.vertex_normals(m), G.vertex_normals(m.vertices, m.faces))
    assert isinstance(m, Mesh)


# ----------------------------------------------------------------- committed goldens
GOLDEN_CASES = sorted(p.name for p in GOLDEN.glob("geometry_*") if p.is_dir())


def test_golden_cases_present():
    assert len(GOLDEN_CASES) >= 3


@pytest.mark.parametrize("case", GOLDEN_CASES)
def test_against_committed_golden(case):
    cmd = json.loads((GOLDEN / case / "command.json").read_text())
    assert cmd["oracle"] == "upstream-macos-arm64"
    run_case(GOLDEN / case)


# ----------------------------------------------------------------- live oracle
@pytest.fixture(scope="session")
def harness(tmp_path_factory) -> Path:
    exe = ORACLES / "meshlib" / "bin" / "harness_geometry"
    if exe.exists():
        return exe
    lib = ORACLES / "meshlib" / "lib" / "libMesh.a"
    cxx = shutil.which("clang++") or shutil.which("c++")
    if not (ORACLES / "meshlib" / "READY").exists() or not lib.exists() or cxx is None:
        pytest.skip("MeshLib oracle (or a C++ compiler) not available")
    out = tmp_path_factory.mktemp("harness") / "harness_geometry"
    inc = ORACLES / "meshlib" / "include"
    cmd = [
        cxx,
        "-std=c++11",
        "-O2",
        f"-I{inc}",
        f"-I{inc / 'Util'}",
        str(HARNESS_SRC),
        str(lib),
        "-o",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        pytest.skip("could not build harness: " + r.stderr[-300:])
    return out


def run_harness(harness: Path, tmp: Path, mesh_v, mesh_f, sigmas, seed=0):
    n = len(mesh_v)
    rng = np.random.default_rng(seed)
    d32 = rng.standard_normal(n).astype("<f4")
    d64 = rng.standard_normal(n)
    d32.tofile(tmp / "data.f32")
    d64.tofile(tmp / "data.f64")
    write_ascii_vtk(tmp / "in.vtk", mesh_v, mesh_f)
    ops = [
        "-s",
        "1",
        "-s",
        "5",
        "-g",
        f"{sigmas[0]:g}",
        "1",
        "-g",
        f"{sigmas[1]:g}",
        "3",
        "-df",
        "3",
        "data.f32",
        "-dd",
        "3",
        "data.f64",
    ]
    subprocess.run([str(harness), "in.vtk", ".", *ops], check=True, cwd=tmp)
    return tmp, ["h", "in.vtk", ".", *ops], {"data.f32": d32, "data.f64": d64}


FIXTURES = {
    "icosphere3": (lambda: icosphere(3), (0.3, 0.15)),
    "uv_sphere": (lambda: uv_sphere(20, 40), (0.3, 0.15)),
    "ellipsoid": (lambda: ellipsoid(), (0.5, 0.3)),
    "folded_sphere": (lambda: folded_sphere(level=4), (0.2, 0.1)),
    "perturb": (lambda: perturb(icosphere(3), 0.02, seed=1), (0.3, 0.15)),
    "cut_sphere": (lambda: cut_sphere(3), (0.3, 0.15)),
}


def compare_live(harness, tmp_path, mesh, sigmas):
    # compare on the float32-rounded geometry that MeshLib actually reads
    v32 = mesh.vertices.astype(np.float32)
    d, argv, orig = run_harness(harness, tmp_path, v32, mesh.faces, sigmas)
    err = check_outputs(d, argv, v32, mesh.faces)
    err.update(check_data_ops(d, argv, mesh.faces, orig))
    bad = {k: x for k, x in err.items() if x > TOL32}
    assert not bad, bad
    return err


@pytest.mark.parametrize("name", list(FIXTURES))
def test_against_oracle_fixtures(name, harness, tmp_path):
    make, sig = FIXTURES[name]
    compare_live(harness, tmp_path, make(), sig)


@pytest.mark.reference
def test_against_oracle_lh_white(harness, tmp_path):
    m = read_vtk(OPT / "reg" / "mb101" / "lh.white.vtk")
    err = compare_live(harness, tmp_path, m, (1.5, 0.7))
    assert max(err.values()) <= TOL32
