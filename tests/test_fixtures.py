import numpy as np
import pytest

from fixtures import (
    cut_sphere,
    ellipsoid,
    folded_sphere,
    geometry_hash,
    icosphere,
    perturb,
    real_sph_harm,
    uv_sphere,
)

# SHA-256 of geometry_hash() (vertices quantised to 1e-9, int32 faces) at fixed params.
# Regenerate deliberately (and review) only if a generator is intentionally changed.
HASHES = {
    "uv_sphere(8,16)": (
        "e8add2851dee170432d75169b39bdea355e0181173f726b551aabb930a3fca77",
        lambda: uv_sphere(8, 16),
    ),
    "ellipsoid(3,2,1,12,24)": (
        "7eaaa40290a9885f90b126e6b9941ee73e289c058d4eea761e1ec38fb388fea9",
        lambda: ellipsoid(3.0, 2.0, 1.0, 12, 24),
    ),
    "icosphere(0)": (
        "47d05533271ff80f1013ab58306c11204e493aa614c07620ddcc889d55811d17",
        lambda: icosphere(0),
    ),
    "icosphere(3)": (
        "918dff3f719be3f13e9ec5a8a5fedb47c247eeb37998c9216687032098e8330c",
        lambda: icosphere(3),
    ),
    "icosphere(5)": (
        "70b8fa3a73579410680cee8fafbaf2f7190a9ac9b5d8e8749088e9b266ded00b",
        lambda: icosphere(5),
    ),
    "folded_sphere(0.1,4,2,3)": (
        "f925ba9f161955586bd1c890ac54c627a380bb7f9d7bb6cff31eeb2f2425ac37",
        lambda: folded_sphere(0.1, 4, 2, 3),
    ),
    "cut_sphere(3,0.5)": (
        "23c3d501031d1af9c3f01e46b937ace71ffa9a2af961ae0f43912b4490d5e47f",
        lambda: cut_sphere(3, 0.5),
    ),
    "perturb(icosphere(2),0.01,seed=7)": (
        "0e873076fea7b0084e0d3deda539713061f9c6278a4cc82c92be803a2e02d451",
        lambda: perturb(icosphere(2), 0.01, 7),
    ),
}


def _volume(m):
    v = m.vertices[m.faces]
    return np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0


def _edge_counts(m):
    e = np.concatenate([m.faces[:, [0, 1]], m.faces[:, [1, 2]], m.faces[:, [2, 0]]])
    return e


@pytest.mark.parametrize("name", list(HASHES))
def test_geometry_hash_constants(name):
    expected, make = HASHES[name]
    assert geometry_hash(make()) == expected
    assert geometry_hash(make()) == expected  # repeat: deterministic


@pytest.mark.parametrize("name", list(HASHES))
def test_hash_inputs_far_from_rounding_boundary(name):
    """Cross-platform guard for the constants above.

    libm / SIMD ``sin``, ``cos``, ``arccos``, ``arctan2`` may differ by a few ulp (~1e-16)
    between platforms.  The hash can only change if that moves a coordinate across a rounding
    boundary of the 1e-9 grid; require every coordinate to sit at least 1e-5 grid steps (1e-14
    absolute, ~100 ulp) away from one, so the hash is provably insensitive to such noise.
    """
    _, make = HASHES[name]
    x = make().vertices * 1e9
    margin = np.abs(x - np.floor(x) - 0.5).min()
    assert margin > 1e-5, f"{name}: coordinate {margin:.2e} grid steps from a rounding boundary"


@pytest.mark.parametrize("level", range(8))
def test_icosphere_counts_and_closed(level):
    m = icosphere(level)
    assert m.n_vertices == 10 * 4**level + 2
    assert m.n_faces == 20 * 4**level
    np.testing.assert_allclose(np.linalg.norm(m.vertices, axis=1), 1.0, atol=1e-12)
    if level <= 5:
        assert _closed_manifold(m)
        assert _volume(m) > 0
    else:  # cheaper check for large levels: directed edges unique
        e = _edge_counts(m)
        assert len(np.unique(e[:, 0].astype(np.int64) * m.n_vertices + e[:, 1])) == len(e)


def _closed_manifold(m):
    e = _edge_counts(m).astype(np.int64)
    key = e[:, 0] * m.n_vertices + e[:, 1]
    rev = e[:, 1] * m.n_vertices + e[:, 0]
    return len(np.unique(key)) == len(key) and np.array_equal(np.sort(key), np.sort(rev))


def test_uv_sphere_properties():
    m = uv_sphere(10, 20, radius=2.0)
    assert m.n_vertices == 2 + 9 * 20
    assert m.n_faces == 2 * 20 + 8 * 20 * 2
    assert m.n_vertices - 3 * m.n_faces // 2 + m.n_faces == 2  # Euler
    assert _closed_manifold(m)
    np.testing.assert_allclose(np.linalg.norm(m.vertices, axis=1), 2.0, atol=1e-12)
    assert _volume(m) > 0


def test_ellipsoid_properties():
    m = ellipsoid(3, 2, 1, 16, 32)
    x, y, z = m.vertices.T
    np.testing.assert_allclose((x / 3) ** 2 + (y / 2) ** 2 + (z / 1) ** 2, 1.0, atol=1e-12)
    assert _closed_manifold(m) and _volume(m) > 0


def test_real_sph_harm_orthonormal():
    # quadrature on a fine lat-lon grid
    nt, npf = 200, 400
    th = (np.arange(nt) + 0.5) * np.pi / nt
    ph = (np.arange(npf) + 0.5) * 2 * np.pi / npf
    T, P = np.meshgrid(th, ph, indexing="ij")
    w = np.sin(T) * (np.pi / nt) * (2 * np.pi / npf)
    lm = [(0, 0), (1, 0), (2, 1), (2, -1), (3, 2), (4, 2), (4, -3)]
    ys = [real_sph_harm(deg, m, T, P) for deg, m in lm]
    for i, yi in enumerate(ys):
        for j, yj in enumerate(ys):
            val = (yi * yj * w).sum()
            assert val == pytest.approx(1.0 if i == j else 0.0, abs=1e-3)


def test_real_sph_harm_known_values():
    th, ph = 0.7, 1.1
    assert real_sph_harm(1, 0, th, ph) == pytest.approx(np.sqrt(3 / (4 * np.pi)) * np.cos(th))
    # real l=1,m=1 is +sqrt(3/4pi) sin(th) cos(ph) (Condon-Shortley phase cancelled)
    assert real_sph_harm(1, 1, th, ph) == pytest.approx(
        np.sqrt(3 / (4 * np.pi)) * np.sin(th) * np.cos(ph)
    )
    assert real_sph_harm(1, -1, th, ph) == pytest.approx(
        np.sqrt(3 / (4 * np.pi)) * np.sin(th) * np.sin(ph)
    )


def test_folded_sphere_radius_analytic():
    a = 0.15
    m = folded_sphere(a, 5, 3, 4)
    r = np.linalg.norm(m.vertices, axis=1)
    np.testing.assert_allclose(r, 1.0 + a * m.point_data["ylm"], atol=1e-12)
    np.testing.assert_allclose(r, m.point_data["radius"], atol=1e-12)
    assert np.abs(m.point_data["ylm"]).max() > 0.1
    assert _closed_manifold(m)


def test_cut_sphere_open_with_single_boundary():
    m = cut_sphere(3, 0.5)
    e = _edge_counts(m).astype(np.int64)
    key = e[:, 0] * m.n_vertices + e[:, 1]
    rev = e[:, 1] * m.n_vertices + e[:, 0]
    boundary = ~np.isin(key, rev)
    assert boundary.sum() > 0
    # euler characteristic of a disc is 1
    n_edges = len(np.unique(np.sort(e, axis=1), axis=0))
    assert m.n_vertices - n_edges + m.n_faces == 1
    assert m.faces.max() == m.n_vertices - 1
    assert m.vertices[:, 2].max() <= 1.0


def test_perturb_seeded():
    base = icosphere(2)
    a, b, c = perturb(base, 0.01, 1), perturb(base, 0.01, 1), perturb(base, 0.01, 2)
    np.testing.assert_array_equal(a.vertices, b.vertices)
    assert not np.array_equal(a.vertices, c.vertices)
    np.testing.assert_array_equal(base.faces, a.faces)


def test_hash_quantisation_ignores_last_bit_noise():
    m = icosphere(2)
    h = geometry_hash(m)
    m.vertices = m.vertices * (1 + 1e-15)
    assert geometry_hash(m) == h


def test_argument_validation():
    with pytest.raises(ValueError):
        icosphere(8)
    with pytest.raises(ValueError):
        uv_sphere(1, 2)
    with pytest.raises(ValueError):
        real_sph_harm(2, 3, 0.1, 0.1)
