"""pybiscuit.io: mesh and per-vertex readers/writers."""

from .frames import (
    mesh_scanner_to_tkras,
    mesh_tkras_to_scanner,
    scanner_ras_to_tkras,
    tkras_to_scanner_matrix,
    tkras_to_scanner_ras,
    vox2ras_tkr,
)
from .freesurfer import (
    read_asc,
    read_fs_curv,
    read_fs_surf,
    write_asc,
    write_fs_curv,
    write_fs_surf,
)
from .gifti import (
    read_gifti,
    read_label_gii,
    write_func_gii,
    write_gifti,
    write_label_gii,
    write_surf_gii,
)
from .labels import LabelTable, read_label_table, read_protocol_table
from .stats import read_stats_tsv, write_stats_tsv
from .surface import read_surface, surface_format, write_surface
from .txt import read_vertex_txt, write_vertex_txt
from .vtk import read_vtk, write_vtk

__all__ = [
    "LabelTable",
    "read_label_gii",
    "read_label_table",
    "read_protocol_table",
    "read_stats_tsv",
    "read_surface",
    "surface_format",
    "write_surface",
    "write_func_gii",
    "write_label_gii",
    "write_stats_tsv",
    "write_surf_gii",
    "mesh_scanner_to_tkras",
    "mesh_tkras_to_scanner",
    "read_asc",
    "read_fs_curv",
    "read_fs_surf",
    "read_gifti",
    "read_vertex_txt",
    "read_vtk",
    "scanner_ras_to_tkras",
    "tkras_to_scanner_matrix",
    "tkras_to_scanner_ras",
    "vox2ras_tkr",
    "write_asc",
    "write_fs_curv",
    "write_fs_surf",
    "write_gifti",
    "write_vertex_txt",
    "write_vtk",
]
