"""Canonical triangle-mesh container (RAS+ mm unless ``frame`` says otherwise)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # float64 (N, 3)
    faces: np.ndarray  # int32 (M, 3)
    point_data: dict[str, np.ndarray] = field(default_factory=dict)  # per-vertex arrays
    frame: str | None = None  # e.g. "RAS", "tkRAS"; None = unknown
    units: str | None = None  # e.g. "mm"
    meta: dict = field(default_factory=dict)  # free-form header/provenance

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]
