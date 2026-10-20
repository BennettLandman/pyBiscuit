import time

import numpy as np
import pytest

from conftest import CTF_PLAIN, MIPAV, OPT
from pybiscuit.io.vtk import read_vtk, write_vtk
from pybiscuit.mesh import Mesh


def _octa():
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    v = v * 1.234567890123 + 0.1
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])  # fmt: skip
    return v, f


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("ptype", ["float", "double"])
def test_roundtrip(tmp_path, binary, ptype):
    v, f = _octa()
    if ptype == "float":
        v = v.astype(np.float32).astype(np.float64)
    pd = {
        "curv": np.linspace(-1, 1, 6).astype(np.float32),
        "label": np.arange(6, dtype=np.int32),
        "d": np.linspace(0, 1, 6) ** 2,
        "vec": np.arange(18, dtype=np.float64).reshape(6, 3) / 7,
    }
    m = Mesh(v, f, point_data=pd, frame="RAS", units="mm")
    p = tmp_path / "m.vtk"
    write_vtk(m, p, binary=binary, dtype=ptype)
    assert (b"BINARY" in p.read_bytes()[:80]) == binary
    r = read_vtk(p)
    np.testing.assert_array_equal(r.vertices, v)
    np.testing.assert_array_equal(r.faces, f)
    assert (r.frame, r.units) == ("RAS", "mm")
    assert r.meta["points_type"] == ptype
    assert set(r.point_data) == set(pd)
    for k, a in pd.items():
        np.testing.assert_array_equal(r.point_data[k], a)
        assert r.point_data[k].dtype == a.dtype


def test_no_frame_title_preserved(tmp_path):
    v, f = _octa()
    p = tmp_path / "m.vtk"
    write_vtk(Mesh(v, f), p)
    r = read_vtk(p)
    assert r.frame is None and r.units is None
    assert r.meta["title"] == "vtk output"


def test_crlf_and_lookup_table(tmp_path):
    v, f = _octa()
    p = tmp_path / "m.vtk"
    write_vtk(Mesh(v, f, point_data={"s": np.arange(6, dtype=np.float32)}), p)
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    r = read_vtk(p)
    np.testing.assert_array_equal(r.faces, f)
    np.testing.assert_array_equal(r.point_data["s"], np.arange(6))


def test_non_triangle_raises(tmp_path):
    p = tmp_path / "q.vtk"
    p.write_text(
        "# vtk DataFile Version 3.0\nx\nASCII\nDATASET POLYDATA\nPOINTS 4 float\n"
        "0 0 0\n1 0 0\n1 1 0\n0 1 0\nPOLYGONS 1 5\n4 0 1 2 3\n"
    )
    with pytest.raises(ValueError, match="triangles only"):
        read_vtk(p)


def test_bad_header_raises(tmp_path):
    p = tmp_path / "b.vtk"
    p.write_text("hello\n")
    with pytest.raises(ValueError):
        read_vtk(p)


def test_large_ascii_speed(tmp_path):
    rng = np.random.default_rng(1)
    n, m = 163842, 327680
    v = rng.normal(size=(n, 3)).astype(np.float32).astype(np.float64)
    f = rng.integers(0, n, size=(m, 3))
    p = tmp_path / "big.vtk"
    write_vtk(Mesh(v, f), p, dtype="float")
    t = time.perf_counter()
    r = read_vtk(p)
    assert time.perf_counter() - t < 5
    assert r.n_faces == m
    np.testing.assert_array_equal(r.faces, f)


def _check(path, nv, nf, frame=None):
    t = time.perf_counter()
    m = read_vtk(path)
    dt = time.perf_counter() - t
    assert (m.n_vertices, m.n_faces) == (nv, nf), path
    assert np.isfinite(m.vertices).all()
    assert m.faces.min() >= 0 and m.faces.max() < nv
    assert dt < 10, f"{path} took {dt:.1f}s"
    return m


@pytest.mark.reference
@pytest.mark.parametrize("hemi", ["lh", "rh"])
@pytest.mark.parametrize("surf", ["white", "inflated"])
def test_reference_reg_vtk(hemi, surf):
    files = sorted((OPT / "reg").glob(f"*/{hemi}.{surf}.vtk"))
    assert files
    for f in files:
        m = _check(f, 163842, 327680)
        assert m.meta["points_type"] == "float"


@pytest.mark.reference
@pytest.mark.parametrize("nf", [5120, 20480, 81920, 327680])
def test_reference_sphere_vtk(nf):
    _check(OPT / "sphere" / f"sphere_{nf}.vtk", nf // 2 + 2, nf)


@pytest.mark.reference
@pytest.mark.parametrize("k,nv", [(5, 10242), (6, 40962)])
def test_reference_icosphere_vtk(k, nv):
    p = CTF_PLAIN / "SurfPostproc/common/parc/mesh_files" / f"icosphere_{k}.vtk"
    _check(p, nv, 2 * (nv - 2))


@pytest.mark.reference
def test_reference_pymacruise_surface(tmp_path):
    p = (
        MIPAV / "pymacruise/build/invivo-base-v3-dura002-no-regrid-2026-10-14/stages"
        / "surface-thickness/attempt-01/inner_surface.vtk"
    )  # fmt: skip
    if not p.exists():
        pytest.skip("PyMaCRUISE surface not present")
    m = _check(p, 241042, 482500)
    assert (m.frame, m.units) == ("RAS", "mm")
    assert m.meta["points_type"] == "double"
    q = tmp_path / "rt.vtk"
    write_vtk(m, q, binary=True)
    m2 = read_vtk(q)
    np.testing.assert_array_equal(m2.vertices, m.vertices)
    np.testing.assert_array_equal(m2.faces, m.faces)
    assert (m2.frame, m2.units) == ("RAS", "mm")
