"""Coordinate-frame conversion between FreeSurfer tkRAS and scanner RAS.

``scanner RAS = T1.affine @ inv(vox2ras_tkr) @ tkRAS`` where ``vox2ras_tkr``
is the "tkregister" voxel-to-surface matrix. It depends only on the volume
shape and voxel sizes (volume centre at the origin, LIA axis convention):

    [[-sx,   0,  0,  sx*nx/2],
     [  0,   0, sz, -sz*nz/2],
     [  0, -sy,  0,  sy*ny/2]]
"""

from __future__ import annotations

import numpy as np

from pybiscuit.mesh.model import Mesh


def _header_of(obj):
    return obj.header if hasattr(obj, "header") else obj


def vox2ras_tkr(t1) -> np.ndarray:
    """4x4 tkregister voxel->surface-RAS matrix from a NIfTI image/header."""
    hdr = _header_of(t1)
    nx, ny, nz = (float(v) for v in hdr.get_data_shape()[:3])
    sx, sy, sz = (float(v) for v in hdr.get_zooms()[:3])
    return np.array(
        [
            [-sx, 0.0, 0.0, sx * nx / 2.0],
            [0.0, 0.0, sz, -sz * nz / 2.0],
            [0.0, -sy, 0.0, sy * ny / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _scanner_affine(t1) -> np.ndarray:
    if hasattr(t1, "affine"):
        return np.asarray(t1.affine, dtype=np.float64)
    return np.asarray(t1.get_best_affine(), dtype=np.float64)


def tkras_to_scanner_matrix(t1) -> np.ndarray:
    """4x4 matrix mapping tkRAS to scanner RAS for this volume."""
    return _scanner_affine(t1) @ np.linalg.inv(vox2ras_tkr(t1))


def _apply(m: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ m[:3, :3].T + m[:3, 3]


def tkras_to_scanner_ras(vertices, t1_header_or_img) -> np.ndarray:
    """Convert (N, 3) tkRAS coordinates to scanner RAS (mm)."""
    return _apply(tkras_to_scanner_matrix(t1_header_or_img), vertices)


def scanner_ras_to_tkras(vertices, t1_header_or_img) -> np.ndarray:
    """Inverse of :func:`tkras_to_scanner_ras`."""
    return _apply(np.linalg.inv(tkras_to_scanner_matrix(t1_header_or_img)), vertices)


def mesh_tkras_to_scanner(mesh: Mesh, t1_header_or_img) -> Mesh:
    """Return a copy of ``mesh`` converted tkRAS -> scanner RAS, frame="RAS"."""
    if mesh.frame not in (None, "tkRAS"):
        raise ValueError(f"mesh frame is {mesh.frame!r}, expected 'tkRAS'")
    return Mesh(
        tkras_to_scanner_ras(mesh.vertices, t1_header_or_img),
        mesh.faces.copy(),
        {k: v.copy() for k, v in mesh.point_data.items()},
        frame="RAS",
        units=mesh.units or "mm",
        meta=dict(mesh.meta),
    )


def mesh_scanner_to_tkras(mesh: Mesh, t1_header_or_img) -> Mesh:
    """Return a copy of ``mesh`` converted scanner RAS -> tkRAS, frame="tkRAS"."""
    if mesh.frame not in (None, "RAS"):
        raise ValueError(f"mesh frame is {mesh.frame!r}, expected 'RAS'")
    return Mesh(
        scanner_ras_to_tkras(mesh.vertices, t1_header_or_img),
        mesh.faces.copy(),
        {k: v.copy() for k, v in mesh.point_data.items()},
        frame="tkRAS",
        units=mesh.units or "mm",
        meta=dict(mesh.meta),
    )
