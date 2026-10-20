"""FreeSurfer surface I/O: ASCII ``.asc``, binary surf and curv files.

ASCII layout: a comment line (``#!ascii version of ...``), a line
``nverts nfaces``, ``nverts`` lines ``x y z 0`` and ``nfaces`` lines
``a b c 0`` (0-based vertex indices). The trailing column is a flag, always 0.

Frame: FreeSurfer-native binary surfaces are in tkRAS, so ``read_fs_surf`` tags
``mesh.frame = "tkRAS"`` by default (override with ``frame=``). ``.asc`` files
have no fixed frame (MaCRUISE ``.asc`` surfaces are scanner RAS), so
``read_asc`` requires an explicit keyword ``frame="RAS"`` or ``"tkRAS"``.
Convert with ``pybiscuit.io.frames``.
"""

from __future__ import annotations

from pathlib import Path

import nibabel.freesurfer as nfs
import numpy as np

from pybiscuit.mesh.model import Mesh

_DEFAULT_FRAME = "tkRAS"
_ASC_FRAMES = ("RAS", "tkRAS")


def read_asc(path, *, frame: str) -> Mesh:
    """Read a FreeSurfer ASCII surface.

    ``frame`` is a required keyword, ``"RAS"`` (scanner RAS; what MaCRUISE
    writes) or ``"tkRAS"``; there is deliberately no default.
    """
    if frame not in _ASC_FRAMES:
        raise ValueError(f"frame must be one of {_ASC_FRAMES}, got {frame!r}")
    path = Path(path)
    with open(path) as fh:
        comment = fh.readline().rstrip("\n")
        nv, nf = (int(t) for t in fh.readline().split()[:2])
        data = np.loadtxt(fh, dtype=np.float64, ndmin=2)
    if data.shape[0] != nv + nf:
        raise ValueError(f"{path}: expected {nv + nf} data lines, found {data.shape[0]}")
    verts = data[:nv, :3]
    faces = data[nv:, :3].astype(np.int32)
    return Mesh(verts, faces, frame=frame, units="mm", meta={"asc_comment": comment})


def write_asc(path, mesh: Mesh, comment: str | None = None) -> None:
    comment = comment or mesh.meta.get("asc_comment") or "#!ascii version of surface (pybiscuit)"
    with open(path, "w") as fh:
        fh.write(comment.rstrip("\n") + "\n")
        fh.write(f"{mesh.n_vertices} {mesh.n_faces}\n")
        np.savetxt(
            fh,
            np.column_stack([mesh.vertices, np.zeros(mesh.n_vertices)]),
            fmt="%.10g %.10g %.10g %d",
        )
        np.savetxt(
            fh,
            np.column_stack([mesh.faces, np.zeros(mesh.n_faces, dtype=np.int64)]),
            fmt="%d %d %d %d",
        )


def read_fs_surf(path, frame: str | None = _DEFAULT_FRAME) -> Mesh:
    """Read a FreeSurfer binary surface (``lh.white`` etc.) via NiBabel."""
    verts, faces = nfs.read_geometry(str(path))
    return Mesh(verts, faces, frame=frame, units="mm")


def write_fs_surf(path, mesh: Mesh) -> None:
    nfs.write_geometry(str(path), mesh.vertices, mesh.faces)


def read_fs_curv(path) -> np.ndarray:
    """Read a FreeSurfer binary per-vertex curv file as float64."""
    return np.asarray(nfs.read_morph_data(str(path)), dtype=np.float64)


def write_fs_curv(path, values, n_faces: int = 0) -> None:
    """Write per-vertex values as a curv file (stored as float32)."""
    nfs.write_morph_data(str(path), np.asarray(values, dtype=np.float32), fnum=n_faces)
