import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from conftest import PARC_MESH_FILES
from pybiscuit.sphere import operators as ops
from pybiscuit.sphere.operators import (
    FIELDS,
    SafeUnpicklingError,
    build_operators,
    compare_operators,
    load_operators_npz,
    safe_load_pickle,
    safe_loads,
    save_operators_npz,
)

TOL = 1e-6


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned > /dev/null",))


class _EvilEval:
    def __reduce__(self):
        return (eval, ("1+1",))


@pytest.mark.parametrize("payload", [_Evil(), _EvilEval(), {"k": _Evil()}])
@pytest.mark.parametrize("protocol", [2, 4])
def test_rejects_malicious_pickles(payload, protocol, tmp_path):
    blob = pickle.dumps(payload, protocol=protocol)
    with pytest.raises(SafeUnpicklingError):
        safe_loads(blob)
    f = tmp_path / "evil.pkl"
    f.write_bytes(blob)
    with pytest.raises(SafeUnpicklingError):
        safe_load_pickle(f)


def test_rejects_codecs_encode_misuse():
    # _codecs.encode is on the allowlist but only as str -> latin1 bytes
    class _Rot13:
        def __reduce__(self):
            import codecs

            return (codecs.encode, ("abc", "rot13"))

    with pytest.raises(SafeUnpicklingError):
        safe_loads(pickle.dumps(_Rot13(), protocol=2))
    with pytest.raises(SafeUnpicklingError):
        ops._safe_codecs_encode(b"abc", "latin1")
    assert ops._safe_codecs_encode("a\x00", "latin1") == b"a\x00"


def test_rejects_modern_numpy_scipy_globals():
    obj = {"a": np.arange(5.0), "m": sp.csc_matrix(np.eye(3))}
    # modern pickles use numpy._core / scipy.sparse._csc globals that are
    # NOT on the allowlist (only the shipped py2-era names are).
    with pytest.raises(SafeUnpicklingError):
        safe_loads(pickle.dumps(obj, protocol=2))


@pytest.mark.parametrize("level", range(4))
def test_build_operators_shapes(level):
    o = build_operators(level)
    nf, nv = 20 * 4**level, 10 * 4**level + 2
    assert set(o) == set(FIELDS)
    assert o["V"].shape == (nv, 3) and o["F"].shape == (nf, 3)
    assert np.allclose(np.linalg.norm(o["V"], axis=1), 1.0)
    assert o["G"].shape == (3 * nf, nv) and o["L"].shape == (nv, nv)
    assert o["F2V"].shape == (nv, nf)
    assert len(o["Seq"]) == 11
    # Laplacian rows sum to zero; mass sums to sphere (polyhedron) area
    assert abs(o["L"].sum(axis=1)).max() < 1e-12
    assert o["M"].sum() == pytest.approx(0.5 * np.linalg.norm(
        np.cross(o["V"][o["F"][:, 1]] - o["V"][o["F"][:, 0]],
                 o["V"][o["F"][:, 2]] - o["V"][o["F"][:, 0]]), axis=1).sum())
    assert np.allclose(np.asarray(o["F2V"].sum(axis=1)).ravel(), 1.0)


def test_npz_roundtrip(tmp_path):
    o = build_operators(2)
    p = tmp_path / "ops.npz"
    save_operators_npz(p, o)
    r = load_operators_npz(p)
    diffs = compare_operators(o, r)
    assert all(d == 0 for d, _ in diffs.values())
    assert sp.isspmatrix_csc(r["Intp"]) and sp.isspmatrix_coo(r["G"])


@pytest.mark.reference
@pytest.mark.parametrize("level", range(6))
def test_matches_shipped_pickle(level):
    shipped = safe_load_pickle(PARC_MESH_FILES / f"icosphere_{level}.pkl")
    assert set(shipped) == set(FIELDS)
    diffs = compare_operators(shipped, build_operators(level))
    for field, (d, note) in diffs.items():
        assert d <= TOL, (level, field, d)
        assert note == "", (level, field, note)


@pytest.mark.reference
@pytest.mark.parametrize("level", range(6))
def test_shipped_sparse_pattern_identical(level):
    shipped = safe_load_pickle(PARC_MESH_FILES / f"icosphere_{level}.pkl")
    mine = build_operators(level)
    for k in ("G", "L", "F2V", "Intp"):
        a, b = sp.coo_matrix(shipped[k]), sp.coo_matrix(mine[k])
        ka = np.unique(a.row.astype(np.int64) * a.shape[1] + a.col)
        kb = np.unique(b.row.astype(np.int64) * b.shape[1] + b.col)
        assert np.array_equal(ka, kb), (level, k)


@pytest.mark.reference
@pytest.mark.parametrize("level", [5, 6])
def test_matches_shipped_vtk(level):
    from pybiscuit.io.vtk import read_vtk

    m = read_vtk(PARC_MESH_FILES / f"icosphere_{level}.vtk")
    o = build_operators(level)
    assert np.array_equal(np.asarray(m.faces), o["F"])
    # the .vtk stores float with 6 decimals
    assert np.abs(np.asarray(m.vertices) - o["V"]).max() < 1e-6
