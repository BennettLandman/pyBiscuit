"""Unit tests for surface preparation and deformation (plan B5.4-B5.7; spec §4.6-4.8)."""

import numpy as np

from pybiscuit.mesh.model import Mesh
from pybiscuit.mesh.topology import topology_summary
from pybiscuit.surfprep.surface import (
    compute_mid_surface,
    compute_thickness,
    deform_pial,
    inflate_surface,
    map_to_sphere,
    refine_white,
)


def _sphere(r=10.0, n_sub=2):
    from pybiscuit.hsd.icosahedron import icosahedron

    v, f = icosahedron(n_sub)
    return Mesh(v * r, f, frame="RAS")


def test_refine_white_smoothing():
    m = _sphere(r=10.0)
    # With 0 smoothing, vertices are identical
    w0 = refine_white(m, surf_smooth=0)
    assert np.allclose(w0.vertices, m.vertices)

    # With smoothing > 0, vertices are smoothed
    w2 = refine_white(m, surf_smooth=2)
    assert w2.n_vertices == m.n_vertices
    assert np.all(w2.faces == m.faces)
    # Laplacian smoothing slightly shrinks convex sphere radius
    assert np.linalg.norm(w2.vertices, axis=1).mean() < np.linalg.norm(m.vertices, axis=1).mean()


def test_compute_mid_surface_and_thickness():
    m_white = _sphere(r=10.0)
    # Create pial as 2.0mm outward offset
    v_pial = m_white.vertices * 1.2  # radius 12.0
    m_pial = Mesh(v_pial, m_white.faces.copy(), frame="RAS")

    mid = compute_mid_surface(m_white, m_pial)
    assert np.allclose(mid.vertices, (m_white.vertices + m_pial.vertices) / 2.0)
    assert np.all(mid.faces == m_white.faces)

    th = compute_thickness(m_white, m_pial)
    assert np.allclose(th, 2.0)


def test_deform_pial_default_and_levelset():
    m_white = _sphere(r=10.0)
    # 1. Default deformation along normal
    pial = deform_pial(m_white, default_thickness=2.5)
    assert pial.n_vertices == m_white.n_vertices
    assert np.all(pial.faces == m_white.faces)
    th = compute_thickness(m_white, pial)
    assert np.allclose(th, 2.5, atol=1e-3)

    # 2. Levelset deformation: synthetic levelset where zero crossing is at r = 12.5
    # Grid: [-20, 20]^3 with 1mm spacing
    grid_coords = np.linspace(-20, 20, 41)
    X, Y, Z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    # Level set phi = R - 12.5 (negative inside r=12.5, zero at r=12.5)
    phi = (R - 12.5).astype(np.float32)
    # Affine: voxel [0,0,0] -> [-20, -20, -20], dx = dy = dz = 1.0
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = -20.0

    pial_ls = deform_pial(m_white, outer_levelset=phi, outer_affine=affine)
    th_ls = compute_thickness(m_white, pial_ls)
    # Thickness should be close to 12.5 - 10.0 = 2.5 mm
    assert np.allclose(th_ls, 2.5, atol=0.25)


def test_inflate_surface_and_sphere_mapping():
    m_white = _sphere(r=10.0)
    smoothwm, inflated, sulc = inflate_surface(m_white, n_smooth_wm=3, n_inflate_iters=10)

    assert smoothwm.n_vertices == m_white.n_vertices
    assert inflated.n_vertices == m_white.n_vertices
    assert len(sulc) == m_white.n_vertices

    # Check mapping to sphere
    sph = map_to_sphere(inflated, radius=100.0, max_iter=20)
    assert sph.n_vertices == m_white.n_vertices
    # Norms should be 100.0
    radii = np.linalg.norm(sph.vertices, axis=1)
    assert np.allclose(radii, 100.0, atol=1e-4)
    # Topology should remain closed genus 0
    topo = topology_summary(sph)
    assert topo.is_closed and topo.genus == 0
