"""pybiscuit.mesh"""

from . import curvature, geometry, sphere_map  # noqa: F401
from .model import Mesh
from .topology import (
    boundary_edges,
    boundary_loops,
    check_orientation,
    connected_components,
    euler_characteristic,
    face_adjacency,
    face_adjacency_matrix,
    genus,
    non_manifold_edges,
    non_manifold_vertices,
    one_ring,
    topology_summary,
    unique_edges,
    unreferenced_vertices,
    vertex_faces,
    vertex_neighbors,
)

__all__ = [
    "curvature", "geometry", "sphere_map",
    "Mesh", "boundary_edges", "boundary_loops", "check_orientation", "connected_components",
    "euler_characteristic", "face_adjacency", "face_adjacency_matrix", "genus",
    "non_manifold_edges", "non_manifold_vertices", "one_ring", "topology_summary",
    "unique_edges", "unreferenced_vertices", "vertex_faces", "vertex_neighbors",
]
