from pathlib import Path

import numpy as np
import pytest

from pybiscuit.io.freesurfer import (
    read_asc,
    read_fs_curv,
    read_fs_surf,
    write_asc,
    write_fs_curv,
    write_fs_surf,
)
from pybiscuit.mesh.model import Mesh


def _tet():
    v = np.array([[0, 0, 0], [1.5, 0, 0], [0, 2.25, 0], [0, 0, -3.125]], float)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
    return Mesh(v, f)


def test_asc_roundtrip(tmp_path):
    m = _tet()
    write_asc(tmp_path / "a.asc", m)
    r = read_asc(tmp_path / "a.asc")
    assert r.frame == "tkRAS"
    np.testing.assert_allclose(r.vertices, m.vertices)
    np.testing.assert_array_equal(r.faces, m.faces)
    assert read_asc(tmp_path / "a.asc", frame=None).frame is None


def test_asc_bad_count(tmp_path):
    p = tmp_path / "b.asc"
    p.write_text("#!ascii\n3 1\n0 0 0 0\n1 0 0 0\n")
    with pytest.raises(ValueError):
        read_asc(p)


def test_surf_roundtrip(tmp_path):
    m = _tet()
    write_fs_surf(tmp_path / "lh.x", m)
    r = read_fs_surf(tmp_path / "lh.x")
    np.testing.assert_allclose(r.vertices, m.vertices)
    np.testing.assert_array_equal(r.faces, m.faces)
    assert r.frame == "tkRAS"


def test_curv_roundtrip(tmp_path):
    vals = np.array([0.5, -1.25, 3.0, 2.0])
    write_fs_curv(tmp_path / "lh.curv", vals, n_faces=4)
    np.testing.assert_allclose(read_fs_curv(tmp_path / "lh.curv"), vals)


MIPAV = Path.home() / "dev" / "mipav"  # same as tests/conftest.py
SAMPLE = MIPAV / "sample-macruise" / "output"


@pytest.mark.reference
@pytest.mark.parametrize("sub", ["Surfaces_FreeView", "Surfaces_MNI"])
@pytest.mark.parametrize(
    "kind,nv,nf",
    [
        ("inner", 250_874, 501_744),
        ("central", 243_822, 487_640),
    ],
)
def test_read_legacy_asc(sub, kind, nv, nf):
    # NB: the brief expected inner = 243,822 (FreeView) / 250,874 (MNI); in the sample
    # inner is 250,874 in both dirs and 243,822 is the *central* surface.
    m = read_asc(SAMPLE / sub / f"target_image_GMimg_{kind}Surf.asc")
    assert (m.n_vertices, m.n_faces) == (nv, nf)
    assert m.faces.min() >= 0 and m.faces.max() == nv - 1
