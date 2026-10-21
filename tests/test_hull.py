"""OuterHull port (plan row B9.2): provisional, no oracle (MATLAB R2012a / Octave unavailable).

Checks the building blocks against analytic or brute-force answers and the pipeline against
geometry the hull must satisfy (encloses the surface, closed, smooth, small margin).
"""

import numpy as np
import pytest
from tests.fixtures.meshes import ellipsoid, folded_sphere, icosphere

from pybiscuit.mesh import euler_characteristic
from pybiscuit.mesh.geometry import vertex_areas
from pybiscuit.quant._voxelise import voxelise
from pybiscuit.quant.hull import (
    _close_ball,
    fminlbfgs,
    icp_finite,
    isosurface,
    outer_hull,
)


def _volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def test_voxelise_sphere_matches_analytic():
    m = icosphere(4, 10.0)
    vol, gx, gy, gz = voxelise(m.vertices, m.faces, 40, 40, 40)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    inside_true = X**2 + Y**2 + Z**2 < 10.0**2
    # facets cut the sphere: only boundary voxels may disagree
    assert (vol != inside_true).sum() < 0.02 * inside_true.sum()
    assert vol.shape == (40, 40, 40) and gx.size == 40


def test_voxelise_ellipsoid_uvsphere_poles_on_rays():
    # uv-sphere poles/ring vertices can sit exactly on grid rays (vertex branch)
    m = ellipsoid(8, 8, 8, n_lat=16, n_lon=32)
    vol, *_ = voxelise(m.vertices, m.faces, 33, 33, 33)
    expect = 4 / 3 * np.pi * 8**3
    vox = (16 / 33.5) ** 3
    assert abs(vol.sum() * vox - expect) / expect < 0.08


def test_isosurface_closed_outward():
    vol = np.zeros((20, 20, 20))
    x, y, z = np.meshgrid(*[np.arange(20)] * 3, indexing="ij")
    vol[(x - 9.5) ** 2 + (y - 9.5) ** 2 + (z - 9.5) ** 2 < 36] = 255.0
    v, f = isosurface(vol, 1.0)
    assert euler_characteristic(v, f) == 2
    assert _volume(v, f) > 0  # faces outward
    r = np.linalg.norm(v - 9.5, axis=1)
    assert abs(r.mean() - 6.0) < 0.6


def test_closing_fills_narrow_gap_per_slice():
    a = np.zeros((60, 60, 5))
    a[10:50, 10:50, :] = 255.0
    a[30:32, 10:40, :] = 0.0  # narrow slot, open to the left
    c = _close_ball(a, 5, 5)
    assert c[31, 20, 2] > 200  # closed
    assert c[5, 5, 2] < 1e-9  # background untouched


def test_fminlbfgs_quadratic():
    A = np.diag([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    b = np.arange(1.0, 7.0)
    x, f = fminlbfgs(lambda p: float(0.5 * p @ A @ p - b @ p), np.zeros(6), 1e-9)
    assert np.allclose(x, np.linalg.solve(A, b), atol=1e-3)


def test_icp_recovers_small_rigid_motion():
    rng = np.random.default_rng(3)
    base = icosphere(3, 30.0).vertices * np.array([1.0, 1.3, 0.8])
    base = base + rng.normal(0, 0.05, base.shape)
    th = np.deg2rad(2.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    moving = base @ R.T + np.array([0.7, -0.4, 0.3])
    moved, M = icp_finite(base, moving)
    assert (
        np.abs(moved - base).max() < 0.5
    )  # TolX = extent/1000 ~ 0.06; nearest-point matching limits it


@pytest.mark.parametrize("grid", [48])
def test_outer_hull_encloses_ellipsoid(grid):
    m = ellipsoid(60, 45, 35, n_lat=30, n_lon=60)
    h = outer_hull(m, grid=grid)
    assert euler_characteristic(h.vertices, h.faces) == 2
    r = np.sqrt(((h.vertices / np.array([60, 45, 35])) ** 2).sum(axis=1))
    assert r.min() > 1.0 and r.max() < 1.12
    assert _volume(h.vertices, h.faces) > _volume(m.vertices, m.faces)


def test_outer_hull_folded_sphere_smoother_than_surface():
    m = folded_sphere(a=0.15, l=8, m=3, level=4)
    m.vertices = m.vertices * 50.0
    h = outer_hull(m, grid=64)
    area_s = vertex_areas(m.vertices, m.faces).sum()
    area_h = vertex_areas(h.vertices, h.faces).sum()
    assert area_h < 1.5 * area_s and _volume(h.vertices, h.faces) > _volume(m.vertices, m.faces)
