"""Surface preparation and deformation from MaCRUISE inner surfaces (plan B5.4-B5.7; spec §4.6-4.8).

Provides:
- ``refine_white``: white surface preparation with optional smoothing (spec §4.6, T13);
- ``deform_pial``: pial surface deformed from white along PyMaCRUISE outer level set or outer
  mesh (B5.4);
- ``compute_mid_surface``: exact mid surface ``(white + pial) / 2`` with white faces
  (spec §4.8, B5.5);
- ``compute_thickness``: cortical thickness per vertex (Euclidean white-pial distance);
- ``inflate_surface``: smoothwm, inflated surface and sulcal depth (``sulc``) (spec §4.7, B5.6);
- ``map_to_sphere``: distortion-minimizing spherical mapping to radius-100 sphere (spec §4.7, B5.7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pybiscuit.mesh.geometry import (
    smooth_data,
    smooth_vertices,
    vertex_normals,
)
from pybiscuit.mesh.model import Mesh
from pybiscuit.mesh.sphere_map import sphere_map

__all__ = [
    "SurfacePrepResult",
    "compute_mid_surface",
    "compute_thickness",
    "deform_pial",
    "inflate_surface",
    "map_to_sphere",
    "prepare_hemisphere_surfaces",
    "refine_white",
]


@dataclass
class SurfacePrepResult:
    """Full surface prep suite for a single hemisphere."""

    white: Mesh
    pial: Mesh
    mid: Mesh
    smoothwm: Mesh
    inflated: Mesh
    sphere: Mesh
    thickness: np.ndarray
    sulc: np.ndarray


def refine_white(inner_mesh: Mesh, surf_smooth: int = 0) -> Mesh:
    """Prepare white surface from inner mesh, optionally smoothing (spec §4.6, T13)."""
    v = inner_mesh.vertices.copy()
    f = inner_mesh.faces.copy()
    if surf_smooth > 0:
        v = smooth_vertices(v, f, iterations=surf_smooth)
    return Mesh(v, f, frame=inner_mesh.frame, units=inner_mesh.units)


def compute_mid_surface(white_mesh: Mesh, pial_mesh: Mesh) -> Mesh:
    """Mid surface = (white + pial) / 2 by vertex index with white faces (spec §4.8)."""
    if white_mesh.n_vertices != pial_mesh.n_vertices:
        raise ValueError(
            f"vertex count mismatch: white has {white_mesh.n_vertices}, "
            f"pial has {pial_mesh.n_vertices}"
        )
    v_mid = (white_mesh.vertices + pial_mesh.vertices) / 2.0
    return Mesh(v_mid, white_mesh.faces.copy(), frame=white_mesh.frame, units=white_mesh.units)


def compute_thickness(white_mesh: Mesh, pial_mesh: Mesh) -> np.ndarray:
    """Cortical thickness = Euclidean distance between white and pial vertices."""
    return np.linalg.norm(pial_mesh.vertices - white_mesh.vertices, axis=1)


def deform_pial(
    white_mesh: Mesh,
    outer_levelset=None,
    outer_affine: np.ndarray | None = None,
    outer_mesh: Mesh | None = None,
    max_thickness: float = 6.0,
    default_thickness: float = 2.5,
    mask: np.ndarray | None = None,
) -> Mesh:
    """Deform white surface outward along normals to outer level set or mesh (B5.4).

    If outer_levelset (a 3D array or NIfTI object) is provided, traces rays outward
    along the unit normals to the zero crossing.
    If outer_mesh is provided, samples closest point outward.
    Otherwise, applies a smooth default thickness along vertex normals.
    Vertices where mask == 0 (medial wall patch) are kept coincident or with zero thickness.
    """
    v_white = white_mesh.vertices.astype(np.float64)
    f = white_mesh.faces
    normals = vertex_normals(v_white, f)

    n_verts = len(v_white)
    pial_v = v_white.copy()
    thickness = np.full(n_verts, default_thickness, dtype=np.float64)

    if mask is not None:
        cortex = mask != 0
    else:
        cortex = np.ones(n_verts, dtype=bool)

    if outer_levelset is not None:
        if hasattr(outer_levelset, "get_fdata"):
            data = outer_levelset.get_fdata()
            inv_aff = np.linalg.inv(outer_levelset.affine)
        elif hasattr(outer_levelset, "dataobj"):
            data = np.asarray(outer_levelset.dataobj, dtype=np.float32)
            inv_aff = np.linalg.inv(outer_levelset.affine)
        else:
            data = np.asarray(outer_levelset, dtype=np.float32)
            inv_aff = np.linalg.inv(outer_affine) if outer_affine is not None else np.eye(4)

        # Trilinear ray march along normal: t from 0.5 to max_thickness in 0.25mm steps
        steps = np.arange(0.5, max_thickness + 0.1, 0.25)
        n_steps = len(steps)

        # For cortex vertices, evaluate level set along ray
        cortex_indices = np.where(cortex)[0]
        pts_ray = (
            v_white[cortex_indices, None, :]
            + normals[cortex_indices, None, :] * steps[None, :, None]
        )
        # Reshape to (N * n_steps, 3) for affine map to voxel coordinates
        pts_flat = pts_ray.reshape(-1, 3)
        ijk = pts_flat @ inv_aff[:3, :3].T + inv_aff[:3, 3]

        # Sample with map_coordinates
        vals = ndimage.map_coordinates(
            data, [ijk[:, 0], ijk[:, 1], ijk[:, 2]], order=1, mode="nearest"
        )
        vals = vals.reshape(len(cortex_indices), n_steps)

        # Find zero crossings (level set goes from negative inside cortex to positive outside)
        for i, idx in enumerate(cortex_indices):
            v_profile = vals[i]
            # Check for sign change
            sign_changes = np.where(v_profile >= 0)[0]
            if len(sign_changes) > 0:
                first_pos = sign_changes[0]
                if first_pos == 0:
                    t_val = steps[0]
                else:
                    v0, v1 = v_profile[first_pos - 1], v_profile[first_pos]
                    t0, t1 = steps[first_pos - 1], steps[first_pos]
                    denom = v1 - v0
                    frac = (-v0) / denom if abs(denom) > 1e-6 else 0.5
                    t_val = t0 + frac * (t1 - t0)
                thickness[idx] = np.clip(t_val, 0.5, max_thickness)

    elif outer_mesh is not None:
        from scipy.spatial import cKDTree

        tree = cKDTree(outer_mesh.vertices)
        cortex_indices = np.where(cortex)[0]
        # Query along normal at estimated distance
        probe = v_white[cortex_indices] + normals[cortex_indices] * default_thickness
        dists, _ = tree.query(probe)
        thickness[cortex_indices] = np.clip(dists, 0.5, max_thickness)

    # Smooth thickness map slightly on cortex to avoid high-frequency noise
    smooth_thick = smooth_data(f, thickness, iterations=2)
    thickness[cortex] = np.clip(smooth_thick[cortex], 0.5, max_thickness)
    thickness[~cortex] = 0.0

    pial_v = v_white + normals * thickness[:, None]
    return Mesh(pial_v, f.copy(), frame=white_mesh.frame, units=white_mesh.units)


def inflate_surface(
    white_mesh: Mesh,
    n_smooth_wm: int = 3,
    n_inflate_iters: int = 200,
    inflation_rate: float = 0.05,
) -> tuple[Mesh, Mesh, np.ndarray]:
    """Produce smoothwm, inflated surface and sulcal depth (``sulc``) (spec §4.7, B5.6).

    1. smoothwm: 3 iterations of Laplacian smoothing of white surface.
    2. inflated: smoothing with normal outward expansion towards a sphere.
    3. sulc: signed normal displacement from smoothwm to inflated (negative in sulcal fundi).
    """
    v_white = white_mesh.vertices.copy()
    f = white_mesh.faces.copy()

    # 1. smoothwm
    v_smoothwm = smooth_vertices(v_white, f, iterations=n_smooth_wm)
    smoothwm_mesh = Mesh(v_smoothwm, f.copy(), frame=white_mesh.frame, units=white_mesh.units)

    # 2. Iterative inflation: outward normal expansion + smoothing
    v_cur = v_smoothwm.copy()
    centroid = v_cur.mean(axis=0)
    target_radius = 50.0  # approximate scale for inflated cortex

    for _it in range(n_inflate_iters):
        # Laplacian smoothing to untangle and unfold
        v_cur = smooth_vertices(v_cur, f, iterations=1)
        # Normal outward step from center
        radial_vec = v_cur - centroid
        r_dist = np.linalg.norm(radial_vec, axis=1, keepdims=True)
        r_dist = np.maximum(r_dist, 1e-6)
        r_dir = radial_vec / r_dist
        # Gently expand towards target sphere
        v_cur += r_dir * (inflation_rate * (target_radius - r_dist))

    inflated_mesh = Mesh(v_cur, f.copy(), frame=white_mesh.frame, units=white_mesh.units)

    # 3. sulcal depth (sulc):
    # FreeSurfer's sulc is the signed displacement along outward normal from smoothwm to inflated
    # (vertices that moved inward relative to average expansion have sulc < 0, crowns > 0)
    smoothwm_normals = vertex_normals(v_smoothwm, f)
    disp = v_cur - v_smoothwm
    sulc = np.sum(disp * smoothwm_normals, axis=1)
    # Mean-center sulc so zero represents average depth
    sulc -= float(np.mean(sulc))

    return smoothwm_mesh, inflated_mesh, sulc


def map_to_sphere(inflated_mesh: Mesh, radius: float = 100.0, max_iter: int = 2000) -> Mesh:
    """Map inflated surface to a sphere with radius 100 mm (spec §4.7, B5.7).

    Uses MeshLib distortion-minimizing spherical mapping.
    """
    res = sphere_map(inflated_mesh, max_iter=max_iter)
    v_sphere = res.mesh.vertices * radius
    return Mesh(
        v_sphere,
        inflated_mesh.faces.copy(),
        frame=inflated_mesh.frame,
        units=inflated_mesh.units,
    )


def prepare_hemisphere_surfaces(
    inner_mesh: Mesh,
    surf_smooth: int = 0,
    outer_levelset=None,
    mask: np.ndarray | None = None,
) -> SurfacePrepResult:
    """Run full surface preparation pipeline on a single hemisphere (B5.3-B5.7)."""
    white = refine_white(inner_mesh, surf_smooth=surf_smooth)
    pial = deform_pial(white, outer_levelset=outer_levelset, mask=mask)
    mid = compute_mid_surface(white, pial)
    thickness = compute_thickness(white, pial)
    smoothwm, inflated, sulc = inflate_surface(white)
    sphere = map_to_sphere(inflated, radius=100.0)

    return SurfacePrepResult(
        white=white,
        pial=pial,
        mid=mid,
        smoothwm=smoothwm,
        inflated=inflated,
        sphere=sphere,
        thickness=thickness,
        sulc=sulc,
    )
