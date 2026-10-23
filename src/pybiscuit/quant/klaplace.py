"""klaplace: surface correspondence via Laplace PDE streamlines (B9.3, spec §9.3).

Port of ``klaplace`` (ktools / Joohwi Lee, Sun Hyung Kim, Martin Styner, UNC Chapel Hill).
In BISCUIT, it is invoked during the local gyrification index (LGI) phase:
    ``klaplace -thread <N> -dims 128 <pial.vtk> <hull.vtk> -surfaceCorrespondence <prefix>``
    ``klaplace -conv <prefix>_warpedMesh.vtp <prefix>.hull.corr.vtk``

Algorithm:
1. Construct a uniform grid around the destination (hull) bounding box with dimension
   ``dims`` (default 128), spacing ``h = maxbound / dims``, extent ``[dim + 7]``,
   origin ``bounds_min - 3 * h``.
2. Voxelize destination mesh (outer hull) and source mesh (inner pial surface).
   Label voxels:
   - 1: solution domain (inside hull and outside pial)
   - 300: inner boundary condition (inside pial, potential low = 0)
   - 700: outer boundary condition (outside hull, potential high = 10000)
   Correct boundary voxels containing source or destination vertices to domain 1.
3. Solve Laplace PDE :math:`\\nabla^2 u = 0` on the solution domain via 10,000 Jacobi iterations
   (or until convergence) with Dirichlet boundary conditions :math:`u=0` on domain 300,
   :math:`u=10000` on domain 700.
4. Compute the normalized central-difference gradient field
   :math:`\\vec{v} = \\frac{\\nabla u}{\\|\\nabla u\\|}`.
5. Integrate streamlines outward from each source vertex using Runge-Kutta 4th-order (RK4)
   integration until intersection with destination (hull) triangles.
   The intersection point becomes the warped vertex coordinate on the hull.
   Vertices whose streamlines do not intersect are projected to the nearest hull vertex.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from ..io.vtk import read_vtk, write_vtk
from ..mesh.model import Mesh
from ._voxelise import _internal

__all__ = [
    "klaplace",
    "klaplace_conv",
    "surface_correspondence",
]


@njit(parallel=True)
def _solve_laplace_jacobi(
    u: np.ndarray,
    u_tmp: np.ndarray,
    coords: np.ndarray,
    n_iters: int,
) -> np.ndarray:
    n_pts = coords.shape[0]
    for _ in range(n_iters):
        for idx in prange(n_pts):
            x = coords[idx, 0]
            y = coords[idx, 1]
            z = coords[idx, 2]
            u_tmp[x, y, z] = (
                u[x + 1, y, z] + u[x - 1, y, z] +
                u[x, y + 1, z] + u[x, y - 1, z] +
                u[x, y, z + 1] + u[x, y, z - 1]
            ) / 6.0
        for idx in prange(n_pts):
            x = coords[idx, 0]
            y = coords[idx, 1]
            z = coords[idx, 2]
            u[x, y, z] = u_tmp[x, y, z]
    return u


@njit(cache=True)
def _compute_normalized_grad(u: np.ndarray, h: float) -> np.ndarray:
    nx, ny, nz = u.shape
    grad = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    inv_2h = np.float32(1.0 / (2.0 * h))
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                gx = (u[x + 1, y, z] - u[x - 1, y, z]) * inv_2h
                gy = (u[x, y + 1, z] - u[x, y - 1, z]) * inv_2h
                gz = (u[x, y, z + 1] - u[x, y, z - 1]) * inv_2h
                norm = np.sqrt(gx * gx + gy * gy + gz * gz)
                if norm > 1e-10:
                    grad[x, y, z, 0] = gx / norm
                    grad[x, y, z, 1] = gy / norm
                    grad[x, y, z, 2] = gz / norm
    return grad


@njit(cache=True, inline="always")
def _ray_triangle_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[bool, float]:
    d0 = p2[0] - p1[0]
    d1 = p2[1] - p1[1]
    d2 = p2[2] - p1[2]

    e10 = b[0] - a[0]
    e11 = b[1] - a[1]
    e12 = b[2] - a[2]

    e20 = c[0] - a[0]
    e21 = c[1] - a[1]
    e22 = c[2] - a[2]

    h0 = d1 * e22 - d2 * e21
    h1 = d2 * e20 - d0 * e22
    h2 = d0 * e21 - d1 * e20

    det = e10 * h0 + e11 * h1 + e12 * h2
    if abs(det) < 1e-12:
        return False, 0.0
    inv_det = 1.0 / det

    s0 = p1[0] - a[0]
    s1 = p1[1] - a[1]
    s2 = p1[2] - a[2]

    u = inv_det * (s0 * h0 + s1 * h1 + s2 * h2)
    if u < -1e-5 or u > 1.0 + 1e-5:
        return False, 0.0

    q0 = s1 * e12 - s2 * e11
    q1 = s2 * e10 - s0 * e12
    q2 = s0 * e11 - s1 * e10

    v = inv_det * (d0 * q0 + d1 * q1 + d2 * q2)
    if v < -1e-5 or u + v > 1.0 + 1e-5:
        return False, 0.0

    t = inv_det * (e20 * q0 + e21 * q1 + e22 * q2)
    if t < -1e-5 or t > 1.0 + 1e-5:
        return False, 0.0

    return True, t


@njit(cache=True, inline="always")
def _sample_vec(
    grad: np.ndarray,
    p: np.ndarray,
    origin: np.ndarray,
    h: float,
) -> np.ndarray:
    fx = (p[0] - origin[0]) / h
    fy = (p[1] - origin[1]) / h
    fz = (p[2] - origin[2]) / h
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    nx, ny, nz = grad.shape[0], grad.shape[1], grad.shape[2]
    if ix < 0 or ix >= nx - 1 or iy < 0 or iy >= ny - 1 or iz < 0 or iz >= nz - 1:
        return np.zeros(3, dtype=np.float64)
    dx = fx - ix
    dy = fy - iy
    dz = fz - iz

    v000 = grad[ix, iy, iz]
    v100 = grad[ix + 1, iy, iz]
    v010 = grad[ix, iy + 1, iz]
    v110 = grad[ix + 1, iy + 1, iz]
    v001 = grad[ix, iy, iz + 1]
    v101 = grad[ix + 1, iy, iz + 1]
    v011 = grad[ix, iy + 1, iz + 1]
    v111 = grad[ix + 1, iy + 1, iz + 1]

    c00 = v000 * (1.0 - dx) + v100 * dx
    c10 = v010 * (1.0 - dx) + v110 * dx
    c01 = v001 * (1.0 - dx) + v101 * dx
    c11 = v011 * (1.0 - dx) + v111 * dx
    c0 = c00 * (1.0 - dy) + c10 * dy
    c1 = c01 * (1.0 - dy) + c11 * dy
    vec = c0 * (1.0 - dz) + c1 * dz
    norm = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if norm > 1e-7:
        return (vec / norm).astype(np.float64)
    return vec.astype(np.float64)


def _correct_boundary(
    domain: np.ndarray,
    vertices: np.ndarray,
    origin: np.ndarray,
    grid_spacing: float,
    target_val: int,
    check_val: int,
) -> None:
    idxs = np.round((vertices - origin) / grid_spacing).astype(int)
    for p in idxs:
        if (
            0 <= p[0] < domain.shape[0]
            and 0 <= p[1] < domain.shape[1]
            and 0 <= p[2] < domain.shape[2]
        ):
            if domain[p[0], p[1], p[2]] == check_val:
                domain[p[0], p[1], p[2]] = target_val


def surface_correspondence(
    surface: Mesh | str | Path,
    hull: Mesh | str | Path,
    *,
    dims: int = 128,
    n_iters: int = 10000,
    step_factor: float = 0.2,
) -> Mesh:
    """Compute surface correspondence by integrating Laplace PDE streamlines from surface to hull.

    Parameters
    ----------
    surface:
        Inner (pial) surface mesh (Mesh object or path).
    hull:
        Outer (hull) surface mesh (Mesh object or path).
    dims:
        Grid dimension along largest bounding box axis (default: 128).
    n_iters:
        Number of Jacobi iterations for Laplace PDE (default: 10000).
    step_factor:
        RK4 step size relative to grid spacing (default: 0.2).

    Returns
    -------
    Mesh:
        Warped surface with identical vertex count and face connectivity as ``surface``,
        whose vertices lie on the ``hull`` boundary.
    """
    src = read_vtk(surface) if not isinstance(surface, Mesh) else surface
    dst = read_vtk(hull) if not isinstance(hull, Mesh) else hull

    # 1. Grid bounds from destination (hull) surface
    dst_v = np.ascontiguousarray(dst.vertices, dtype=np.float64)
    dst_f = np.ascontiguousarray(dst.faces, dtype=np.int32)
    src_v = np.ascontiguousarray(src.vertices, dtype=np.float64)

    bounds = np.array([
        dst_v[:, 0].min(), dst_v[:, 0].max(),
        dst_v[:, 1].min(), dst_v[:, 1].max(),
        dst_v[:, 2].min(), dst_v[:, 2].max(),
    ])
    maxbound = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    grid_spacing = float(maxbound / dims)
    dim = np.array([
        int((bounds[1] - bounds[0]) / grid_spacing),
        int((bounds[3] - bounds[2]) / grid_spacing),
        int((bounds[5] - bounds[4]) / grid_spacing),
    ])
    extent = np.array([dim[0] + 7, dim[1] + 7, dim[2] + 7])
    origin = np.array([
        bounds[0] - 3.0 * grid_spacing,
        bounds[2] - 3.0 * grid_spacing,
        bounds[4] - 3.0 * grid_spacing,
    ])

    gx = origin[0] + np.arange(extent[0]) * grid_spacing
    gy = origin[1] + np.arange(extent[1]) * grid_spacing
    gz = origin[2] + np.arange(extent[2]) * grid_spacing

    # 2. Voxelization
    M_hull = dst_v[dst_f].transpose(0, 2, 1).copy()
    M_src = src_v[src.faces].transpose(0, 2, 1).copy()
    fmin_h = np.ascontiguousarray(M_hull.min(axis=2))
    fmax_h = np.ascontiguousarray(M_hull.max(axis=2))
    fmin_s = np.ascontiguousarray(M_src.min(axis=2))
    fmax_s = np.ascontiguousarray(M_src.max(axis=2))

    vol_hull = _internal(gx, gy, gz, M_hull, fmin_h, fmax_h)
    vol_src = _internal(gx, gy, gz, M_src, fmin_s, fmax_s)

    # Domain labeling
    domain = np.full(extent, 700, dtype=np.int16)
    domain[vol_hull] = 1
    domain[vol_src] = 300

    # Boundary corrections
    _correct_boundary(domain, src_v, origin, grid_spacing, 1, 300)
    _correct_boundary(domain, dst_v, origin, grid_spacing, 1, 700)

    coords = np.argwhere(domain == 1).astype(np.int32)
    if coords.size == 0:
        raise ValueError("Laplace PDE solution domain is empty (surfaces may be misaligned)")

    # 3. Solve Laplace PDE
    u = np.zeros(extent, dtype=np.float64)
    u[domain == 700] = 10000.0
    u[domain == 300] = 0.0
    u_tmp = u.copy()

    _solve_laplace_jacobi(u, u_tmp, coords, n_iters)

    # 4. Normalized gradient field
    grad = _compute_normalized_grad(u, grid_spacing)

    # 5. Build KD-Tree and vertex-to-triangle incidence for destination surface
    hull_tree = cKDTree(dst_v)
    vert_tris = [[] for _ in range(dst.n_vertices)]
    for i, f in enumerate(dst_f):
        vert_tris[f[0]].append(i)
        vert_tris[f[1]].append(i)
        vert_tris[f[2]].append(i)

    # 6. Streamline tracing
    n_src = src.n_vertices
    warped_v = np.empty_like(src_v)
    step_size = grid_spacing * step_factor
    max_steps = 1500

    for i in range(n_src):
        pos = src_v[i].copy()
        pts_history = [pos.copy()]
        for _ in range(max_steps):
            v = _sample_vec(grad, pos, origin, grid_spacing)
            if np.linalg.norm(v) < 1e-5:
                break
            # RK4
            k1 = v
            p_k2 = pos + 0.5 * step_size * k1
            k2 = _sample_vec(grad, p_k2, origin, grid_spacing)
            p_k3 = pos + 0.5 * step_size * k2
            k3 = _sample_vec(grad, p_k3, origin, grid_spacing)
            p_k4 = pos + step_size * k3
            k4 = _sample_vec(grad, p_k4, origin, grid_spacing)

            pos_next = pos + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            pts_history.append(pos_next.copy())
            pos = pos_next

            if (
                pos[0] < origin[0] or pos[0] > origin[0] + (extent[0] - 1) * grid_spacing
                or pos[1] < origin[1] or pos[1] > origin[1] + (extent[1] - 1) * grid_spacing
                or pos[2] < origin[2] or pos[2] > origin[2] + (extent[2] - 1) * grid_spacing
            ):
                break

        # Search backwards along the streamline for intersection with destination surface
        found = False
        for j in range(len(pts_history) - 1, 0, -1):
            p1 = pts_history[j - 1]
            p2 = pts_history[j]

            _, nbr_vids = hull_tree.query(0.5 * (p1 + p2), k=8)
            tested: set[int] = set()
            best_t = 2.0
            best_pt = None
            for vid in nbr_vids:
                for t_idx in vert_tris[vid]:
                    if t_idx in tested:
                        continue
                    tested.add(t_idx)
                    f = dst_f[t_idx]
                    hit, t_val = _ray_triangle_intersect(
                        p1, p2, dst_v[f[0]], dst_v[f[1]], dst_v[f[2]]
                    )
                    if hit and t_val < best_t:
                        best_t = t_val
                        best_pt = p1 + t_val * (p2 - p1)
            if best_pt is not None:
                warped_v[i] = best_pt
                found = True
                break

        if not found:
            # Fallback for streamlines ending without crossing (overshoot or protrusion):
            # project last point to closest destination hull vertex
            _, near_vid = hull_tree.query(pts_history[-1])
            warped_v[i] = dst_v[near_vid]

    return Mesh(warped_v, src.faces, frame=src.frame, units=src.units)


def klaplace(
    source_path: str | Path,
    target_path: str | Path,
    prefix: str | Path,
    *,
    dims: int = 128,
    n_iters: int = 10000,
) -> Path:
    """Run klaplace surface correspondence and write warped mesh output.

    Parameters
    ----------
    source_path:
        Input source (pial) surface mesh.
    target_path:
        Input target (hull) surface mesh.
    prefix:
        Output prefix. Writes ``<prefix>_warpedMesh.vtk``.

    Returns
    -------
    Path:
        Path to output warped mesh file.
    """
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(f"{prefix_path}_warpedMesh.vtk")
    vtp_path = Path(f"{prefix_path}_warpedMesh.vtp")

    warped = surface_correspondence(source_path, target_path, dims=dims, n_iters=n_iters)
    write_vtk(warped, out_path, dtype="float")
    write_vtk(warped, vtp_path, dtype="float")
    return out_path


def klaplace_conv(input_path: str | Path, output_path: str | Path) -> Path:
    """Convert mesh file (e.g. from vtp/vtk to vtk), matching ``klaplace -conv <input> <output>``.

    Parameters
    ----------
    input_path:
        Input mesh file path.
    output_path:
        Output mesh file path.

    Returns
    -------
    Path:
        Path to output converted mesh file.
    """
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if inp.suffix.lower() == ".vtp":
        peer_vtk = inp.with_suffix(".vtk")
        if peer_vtk.exists():
            inp = peer_vtk

    mesh = read_vtk(inp)
    write_vtk(mesh, out, dtype="float")
    return out

