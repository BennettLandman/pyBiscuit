"""Format-dispatching surface I/O: VTK legacy, GIfTI, FreeSurfer ``.asc`` and binary surf.

``read_surface`` picks the reader from the file name (``.vtk``/``.vtk.gz``, ``.gii``,
``.asc``) and otherwise from the FreeSurfer binary magic number, so tool command
lines accept any of these where the original binary took a VTK file.

Frames: ``.asc`` has no fixed frame; ``read_surface`` reads it as scanner RAS (what
MaCRUISE writes) unless ``frame=`` says otherwise. FreeSurfer binary surfaces are
tagged ``tkRAS``; GIfTI takes its frame from the POINTSET coordinate system.
"""

from __future__ import annotations

from pathlib import Path

from pybiscuit.mesh.model import Mesh

FORMATS = ("vtk", "gifti", "asc", "freesurfer")
_FS_MAGIC = (b"\xff\xff\xfe", b"\xff\xff\xff")  # triangle, quad


def surface_format(path) -> str:
    """Return the surface format of ``path``: one of :data:`FORMATS`.

    Raises ``ValueError`` when neither the name nor the content identifies it.
    """
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".vtk", ".vtk.gz")):
        return "vtk"
    if name.endswith(".gii"):
        return "gifti"
    if name.endswith(".asc"):
        return "asc"
    if path.is_file():
        with open(path, "rb") as fh:
            if fh.read(3) in _FS_MAGIC:
                return "freesurfer"
    raise ValueError(
        f"{path}: unrecognised surface format (expected .vtk, .gii, .asc or a FreeSurfer surface)"
    )


def read_surface(path, *, frame: str | None = None) -> Mesh:
    """Read a triangle surface in any supported format (see :func:`surface_format`).

    ``frame`` overrides the frame tag; for ``.asc`` it defaults to ``"RAS"``.
    ``mesh.meta["format"]`` records the format read.
    """
    fmt = surface_format(path)
    if fmt == "vtk":
        from .vtk import read_vtk

        mesh = read_vtk(path)
    elif fmt == "gifti":
        from .gifti import read_gifti

        mesh = read_gifti(path)
    elif fmt == "asc":
        from .freesurfer import read_asc

        mesh = read_asc(path, frame=frame or "RAS")
    else:
        from .freesurfer import read_fs_surf

        mesh = read_fs_surf(path)
    if frame is not None:
        mesh.frame = frame
    mesh.meta["format"] = fmt
    return mesh


def write_surface(path, mesh: Mesh, *, format: str | None = None) -> None:
    """Write ``mesh`` in the format implied by ``path`` (or ``format``, one of :data:`FORMATS`).

    FreeSurfer binary has no extension, so it is written only when asked for by ``format``.
    """
    path = Path(path)
    if format is None:
        name = path.name.lower()
        if name.endswith(".vtk"):
            format = "vtk"
        elif name.endswith(".gii"):
            format = "gifti"
        elif name.endswith(".asc"):
            format = "asc"
        else:
            raise ValueError(f"{path}: cannot infer surface format; pass format=")
    if format == "vtk":
        from .vtk import write_vtk

        write_vtk(mesh, path)
    elif format == "gifti":
        from .gifti import write_gifti

        write_gifti(path, mesh)
    elif format == "asc":
        from .freesurfer import write_asc

        write_asc(path, mesh)
    elif format == "freesurfer":
        from .freesurfer import write_fs_surf

        write_fs_surf(path, mesh)
    else:
        raise ValueError(f"unknown surface format {format!r}; expected one of {FORMATS}")
