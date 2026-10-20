"""B4.1: icosphere generators vs the shipped BISCUIT / UGSCNN meshes."""

import numpy as np
import pytest

from conftest import PARC_MESH_FILES, REG_DIR, SPHERE_DIR
from pybiscuit.io.vtk import read_vtk
from pybiscuit.sphere.icosphere import (
    SCHEMES,
    SHIPPED_SPHERES,
    icosphere,
    icosphere_counts,
    quantize_shipped,
)

R_BISCUIT = 100.0
LSB = 1e-4 * R_BISCUIT  # one step of the 4-decimal quantisation (0.01 mm)
# Coordinates where the original arithmetic rounded an exact tie the other way
# (always by exactly one LSB): measured 8 / 10 / 57 / 277 of 3N coordinates.
MAX_TIE_FLIPS = {4: 8, 5: 10, 6: 57, 7: 277}


def _edge_stats(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq, counts


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("level", range(0, 7))
def test_counts_closed_unit_deterministic(level, scheme):
    m = icosphere(level, scheme)
    nv, nf = icosphere_counts(level)
    assert (m.n_vertices, m.n_faces) == (nv, nf)
    assert nv == 10 * 4**level + 2 and nf == 20 * 4**level
    assert m.vertices.dtype == np.float64 and m.faces.dtype == np.int32
    np.testing.assert_allclose(np.linalg.norm(m.vertices, axis=1), 1.0, atol=1e-12)
    # closed 2-manifold: every edge shared by exactly two faces, V - E + F = 2
    uniq, counts = _edge_stats(m.faces)
    assert (counts == 2).all()
    assert nv - len(uniq) + nf == 2
    assert m.faces.min() == 0 and m.faces.max() == nv - 1
    assert len(np.unique(m.faces)) == nv  # no unused vertices
    # outward winding
    tri = m.vertices[m.faces]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    assert (vol > 0).all()
    # determinism
    m2 = icosphere(level, scheme)
    assert np.array_equal(m.vertices, m2.vertices) and np.array_equal(m.faces, m2.faces)


@pytest.mark.parametrize("scheme", ["biscuit", "biscuit_ico4"])
def test_biscuit_levels_nested(scheme):
    """Vertex k of level L is vertex k of level L+1 (FreeSurfer-style ico)."""
    a, b = icosphere(3, scheme), icosphere(4, scheme)
    np.testing.assert_array_equal(a.vertices, b.vertices[: a.n_vertices])


def test_level_7_and_bounds():
    m = icosphere(7)
    assert (m.n_vertices, m.n_faces) == (163842, 327680)
    assert icosphere_counts(7) == (163842, 327680)
    with pytest.raises(ValueError):
        icosphere(8)
    with pytest.raises(ValueError):
        icosphere(-1)
    with pytest.raises(ValueError):
        icosphere(2, "nope")


def test_schemes_share_vertices_up_to_rotation():
    """biscuit and biscuit_ico4 differ by a 72 degree z rotation and face order."""
    a, b = icosphere(2, "biscuit"), icosphere(2, "biscuit_ico4")
    assert a.vertices.shape == b.vertices.shape
    assert not np.allclose(a.vertices, b.vertices)  # labelling differs
    c, s = np.cos(np.deg2rad(72)), np.sin(np.deg2rad(72))
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    # same point set after rotating one by -72 deg
    ra = a.vertices @ rot.T
    ra_sorted = ra[np.lexsort(np.round(ra, 9).T)]
    b_sorted = b.vertices[np.lexsort(np.round(b.vertices, 9).T)]
    np.testing.assert_allclose(ra_sorted, b_sorted, atol=1e-8)


# ---------------------------------------------------------------- reference


@pytest.mark.reference
@pytest.mark.parametrize("level", sorted(SHIPPED_SPHERES))
def test_biscuit_spheres_match_shipped(level):
    scheme, name = SHIPPED_SPHERES[level]
    ref = read_vtk(SPHERE_DIR / name)
    m = icosphere(level, scheme)
    assert m.faces.shape == ref.faces.shape
    assert np.array_equal(m.faces, ref.faces), "face list must be identical"
    # raw unit-sphere vertices vs file: within half a quantisation step
    raw = np.abs(m.vertices * R_BISCUIT - ref.vertices).max()
    assert raw < 0.5 * LSB + 1e-4  # half a step + tie-flip slack
    # after the files' own quantisation: exact except for rare 1-LSB tie flips
    d = np.abs(quantize_shipped(m.vertices, R_BISCUIT) - ref.vertices)
    flips = d > 1e-3
    assert flips.sum() <= MAX_TIE_FLIPS[level]
    # files hold float32(x) printed to 9 decimals: ulp is 7.6e-6 for |x| in [64, 128)
    assert d[~flips].max() <= 1e-5
    assert d.max() <= LSB + 1e-5


@pytest.mark.reference
@pytest.mark.parametrize("level", [5, 6])
def test_ugscnn_icospheres_match_parc_meshes(level):
    ref = read_vtk(PARC_MESH_FILES / f"icosphere_{level}.vtk")
    m = icosphere(level, "ugscnn")
    assert np.array_equal(m.faces, ref.faces)
    assert np.abs(m.vertices - ref.vertices).max() <= 1e-6  # files print 6 decimals


@pytest.mark.reference
def test_ugscnn_is_not_the_biscuit_order():
    ref = read_vtk(SPHERE_DIR / "sphere_20480.vtk")
    m = icosphere(5, "ugscnn")
    assert not np.array_equal(m.faces, ref.faces)


@pytest.mark.reference
def test_template_sphere_is_sphere_327680():
    """reg/*/?h.{white,inflated}.vtk (ico7) share the exact face array of sphere_327680."""
    sphere = read_vtk(SPHERE_DIR / "sphere_327680.vtk")
    files = sorted(REG_DIR.glob("*/?h.white.vtk")) + sorted(REG_DIR.glob("*/?h.inflated.vtk"))
    assert len(files) == 12
    for p in files:
        m = read_vtk(p)
        assert m.vertices.shape == sphere.vertices.shape, p
        assert np.array_equal(m.faces, sphere.faces), p
    assert np.array_equal(icosphere(7).faces, sphere.faces)
