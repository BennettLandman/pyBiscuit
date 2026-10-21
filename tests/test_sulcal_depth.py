"""Sulcal depth port (plan row B9.5): provisional, no MATLAB/Octave oracle.

Building blocks are checked against analytic answers; the whole chain against sphere / grooved
sphere geometry (baseline offset of the smoothed hull, depth growing with groove depth).
"""

import math

import numpy as np
import pytest

from fixtures.meshes import icosphere
from pybiscuit.io import write_vtk
from pybiscuit.mesh import Mesh
from pybiscuit.quant.sulcal_depth import (
    bwdistgeodesic,
    bwperim,
    interp3_linear,
    sulcal_depth,
)


def test_bwperim_18_cube():
    bw = np.zeros((7, 7, 7), bool)
    bw[1:6, 1:6, 1:6] = True
    p = bwperim(bw, 18)
    assert not p[3, 3, 3] and p[1, 3, 3] and p[1, 1, 3]
    assert p.sum() == 125 - 27
    # array border alone is not a perimeter
    assert bwperim(np.ones((4, 4, 4), bool)).sum() == 0


def test_bwdistgeodesic_chamfer_steps():
    m = np.ones((7, 7, 7), bool)
    s = np.zeros_like(m)
    s[0, 0, 0] = True
    d = bwdistgeodesic(m, s)
    assert d[3, 0, 0] == 3.0
    assert d[2, 2, 0] == pytest.approx(2 * math.sqrt(2))
    assert d[2, 2, 2] == pytest.approx(2 * math.sqrt(3))
    assert d[3, 1, 0] == pytest.approx(math.sqrt(2) + 2)  # quasi-Euclidean, not Euclidean
    assert d[3, 1, 1] == pytest.approx(math.sqrt(3) + 2)


def test_bwdistgeodesic_wall_and_unreachable():
    m = np.ones((5, 5, 1), bool)
    m[2, :4, 0] = False  # wall with a gap at y = 4
    s = np.zeros_like(m)
    s[0, 0, 0] = True
    d = bwdistgeodesic(m, s)
    assert np.isnan(d[2, 0, 0])
    assert d[4, 0, 0] == pytest.approx(4 * math.sqrt(2) + 4)  # detour through the gap
    m[2, :, 0] = False
    assert np.isnan(bwdistgeodesic(m, s)[4, 4, 0])


def test_seeds_outside_mask_policy():
    m = np.zeros((5, 1, 1), bool)
    m[1:, 0, 0] = True
    s = np.zeros_like(m)
    s[0, 0, 0] = True
    assert np.isnan(bwdistgeodesic(m, s)).all()
    assert bwdistgeodesic(m, s, seeds_outside_mask="keep")[4, 0, 0] == 4.0


def test_interp3_linear_exact_and_inf():
    i, j, k = np.meshgrid(*[np.arange(6.0)] * 3, indexing="ij")
    vol = 2 * i + 3 * j - k
    p = np.array([[1.25, 2.5, 3.75], [0, 0, 0], [5, 5, 5]])
    np.testing.assert_allclose(interp3_linear(vol, p), 2 * p[:, 0] + 3 * p[:, 1] - p[:, 2])
    assert np.isnan(interp3_linear(vol, [[-0.1, 0, 0]])).all()
    vol[2, 2, 2] = np.inf
    assert interp3_linear(vol, [[1.5, 1.5, 1.5]])[0] == np.inf  # positive weight * Inf
    assert np.isnan(interp3_linear(vol, [[1.0, 1.0, 1.0]])[0])  # 0 * Inf -> NaN (IEEE)


def test_sphere_depth_is_small_baseline():
    sd = sulcal_depth(icosphere(5, 40.0))
    assert np.isfinite(sd).all()
    assert sd.min() >= 0 and sd.max() < 4.0  # hull offset of the smoothed solid, a few voxels
    assert sd.std() < 0.6
    sd30 = sulcal_depth(icosphere(5, 30.0))
    assert abs(sd30.mean() - sd.mean()) < 0.5  # baseline independent of radius


def _groove(d, R=40.0, w=2.5):
    m = icosphere(5, 1.0)
    u = m.vertices
    ang = np.arcsin(np.clip(u[:, 0], -1, 1)) * R
    r = R - d * np.exp(-(ang**2) / (2 * w * w))
    return Mesh(u * r[:, None], m.faces), ang


def test_grooved_sphere_depth_tracks_groove():
    base = sulcal_depth(_groove(0.0)[0]).mean()
    bottoms = []
    for d in (4.0, 8.0, 12.0):
        m, ang = _groove(d)
        sd = sulcal_depth(m)
        bottom = sd[np.abs(ang) < 0.7].mean()
        bottoms.append(bottom)
        if d >= 8:
            assert abs(bottom - base - d) < 2.5  # voxel tolerance
        # monotone along the groove profile and flat away from it
        prof = [sd[(np.abs(ang) >= a) & (np.abs(ang) < a + 1.2)].mean() for a in (0, 2, 4, 6)]
        assert prof[0] > prof[1] > prof[2] > prof[3] - 0.2
        assert abs(sd[np.abs(ang) > 12].mean() - base) < 0.6
    assert bottoms[0] < bottoms[1] < bottoms[2]


def test_cli_writes_one_value_per_vertex(tmp_path):
    from pybiscuit.quant.sulcal_depth_cli import main

    m = icosphere(3, 20.0)
    write_vtk(m, tmp_path / "in.vtk")
    assert main([str(tmp_path / "in.vtk"), str(tmp_path / "out.txt")]) == 0
    vals = np.loadtxt(tmp_path / "out.txt")
    assert vals.shape == (m.vertices.shape[0],)
    assert (tmp_path / "out.txt").read_text().splitlines()[0].count(".") == 1
