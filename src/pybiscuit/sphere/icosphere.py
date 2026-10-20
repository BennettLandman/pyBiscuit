"""Icosphere generators that reproduce the icospheres shipped with BISCUIT.

Three subdivision schemes are provided (see ``docs/ports/icosphere.md``):

``"biscuit"`` (default)
    FreeSurfer-style nested icosahedron (pole vertex first, vertex ``k`` of a
    coarse level is vertex ``k`` of every finer level).  Reproduces
    ``sphere_{20480,81920,327680}.vtk`` (levels 5, 6, 7) vertex for vertex and
    face for face.  The template/registration meshes ``reg/*/?h.white.vtk`` use
    the face list of level 7 of this scheme.
``"biscuit_ico4"``
    Same vertices as ``"biscuit"`` up to a 72 degree rotation about z, but with
    the face list kept in parent-child order.  Reproduces ``sphere_5120.vtk``
    (level 4) vertex for vertex and face for face.
``"ugscnn"``
    The UGSCNN / SphericalLabeling ``meshcnn.mesh.icosphere`` generator.
    Reproduces ``icosphere_{5,6}.vtk`` of the parcellation network.

Vertices are float64 on the unit sphere; the shipped files store them
quantised to 4 decimals of the unit sphere (``R = 100`` in the BISCUIT files).
Use :func:`quantize_shipped` to reproduce that quantisation.
"""

from __future__ import annotations

import numpy as np

from pybiscuit.mesh import Mesh

__all__ = [
    "MAX_LEVEL",
    "SCHEMES",
    "SHIPPED_SPHERES",
    "icosphere",
    "icosphere_counts",
    "quantize_shipped",
]

MAX_LEVEL = 7
SCHEMES = ("biscuit", "biscuit_ico4", "ugscnn")

# level -> (scheme, shipped file) for the files that were reproduced.
SHIPPED_SPHERES: dict[int, tuple[str, str]] = {
    4: ("biscuit_ico4", "sphere_5120.vtk"),
    5: ("biscuit", "sphere_20480.vtk"),
    6: ("biscuit", "sphere_81920.vtk"),
    7: ("biscuit", "sphere_327680.vtk"),
}

# Base icosahedron faces (parent-triple convention, see ``_subdivide``).
_BASE_FACES = np.array(
    [
        [0, 4, 3], [0, 5, 4], [0, 1, 5], [0, 2, 1], [0, 3, 2],
        [3, 8, 2], [3, 9, 8], [3, 4, 9], [4, 10, 9], [4, 5, 10],
        [5, 6, 10], [5, 1, 6], [1, 7, 6], [1, 2, 7], [2, 8, 7],
        [8, 9, 11], [9, 10, 11], [10, 6, 11], [6, 7, 11], [7, 8, 11],
    ],
    dtype=np.int64,
)  # fmt: skip


def icosphere_counts(level: int) -> tuple[int, int]:
    """Return ``(n_vertices, n_faces)`` of the level-``level`` icosphere.

    ``n_faces = 20 * 4**level`` and ``n_vertices = 10 * 4**level + 2``
    (level 7: 163,842 vertices, 327,680 faces).
    """
    if level < 0:
        raise ValueError("level must be >= 0")
    return 10 * 4**level + 2, 20 * 4**level


def quantize_shipped(vertices: np.ndarray, radius: float = 100.0) -> np.ndarray:
    """Quantise unit-sphere vertices like the shipped BISCUIT spheres.

    The files hold ``round(x, 4) * radius`` (4 decimals of the unit sphere,
    scaled to ``radius`` mm).  Exact ties can round the other way in the
    originals (about 0.08 % of coordinates at level 7, always by one step).
    """
    return np.round(np.asarray(vertices) * 1e4) / 1e4 * radius


def _base_vertices(offset_deg: float) -> np.ndarray:
    """Pole, upper ring (z=1/sqrt5), lower ring (z=-1/sqrt5), pole."""
    zr = 1.0 / np.sqrt(5.0)
    rr = 2.0 / np.sqrt(5.0)
    up = np.deg2rad(offset_deg + 72.0 * np.arange(5))
    lo = np.deg2rad(offset_deg - 36.0 + 72.0 * np.arange(5))
    v = np.zeros((12, 3))
    v[0] = (0.0, 0.0, 1.0)
    v[1:6] = np.c_[rr * np.cos(up), rr * np.sin(up), np.full(5, zr)]
    v[6:11] = np.c_[rr * np.cos(lo), rr * np.sin(lo), np.full(5, -zr)]
    v[11] = (0.0, 0.0, -1.0)
    v[np.abs(v) < 1e-15] = 0.0
    return v


def _subdivide(verts: np.ndarray, tri: np.ndarray, two_pass: bool) -> tuple[np.ndarray, np.ndarray]:
    """One FreeSurfer-style subdivision step.

    ``tri`` holds parent triples ``(a, b, c)``.  New vertices are numbered in
    order of first use when scanning the parents in list order and, within a
    parent, the edges ``ab, bc, ca``.  Children are ``(a, ca, ab)``,
    ``(ab, bc, b)``, ``(ca, bc, ab)``, ``(ca, c, bc)``: their winding is the
    reverse of the parent triple.  ``two_pass=False`` keeps the four children
    of a parent adjacent; ``two_pass=True`` lists all ``a`` corners first and
    then ``b-corner, centre, c-corner`` for each parent.
    """
    n = len(verts)
    nf = len(tri)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ends = np.stack([np.stack([a, b], 1), np.stack([b, c], 1), np.stack([c, a], 1)], 1)
    ends = ends.reshape(-1, 2)  # row 3*i+e = edge e of parent i
    lo = ends.min(1)
    hi = ends.max(1)
    key = lo * n + hi
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")  # unique ids by first occurrence
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    mid_id = (n + rank[inverse.reshape(-1)]).reshape(nf, 3)
    first_sorted = first[order]
    m = verts[lo[first_sorted]] + verts[hi[first_sorted]]
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    new_verts = np.vstack([verts, m])
    ab, bc, ca = mid_id[:, 0], mid_id[:, 1], mid_id[:, 2]
    ch_a = np.stack([a, ca, ab], 1)
    ch_b = np.stack([ab, bc, b], 1)
    ch_c = np.stack([ca, bc, ab], 1)
    ch_d = np.stack([ca, c, bc], 1)
    if two_pass:
        faces = np.vstack([ch_a, np.stack([ch_b, ch_c, ch_d], 1).reshape(-1, 3)])
    else:
        faces = np.stack([ch_a, ch_b, ch_c, ch_d], 1).reshape(-1, 3)
    return new_verts, faces


def _biscuit(level: int, two_pass: bool) -> tuple[np.ndarray, np.ndarray]:
    verts = _base_vertices(-72.0 if two_pass else 0.0)
    tri = _BASE_FACES
    for _ in range(level):
        verts, faces = _subdivide(verts, tri, two_pass)
        tri = faces[:, [0, 2, 1]]  # next parents: reverse of the children
    # Output winding is the reverse of the parent-triple convention
    return verts, tri[:, [0, 2, 1]]


# ---- UGSCNN / SphericalLabeling scheme ------------------------------------


def _ugscnn_unique_rows(data: np.ndarray, digits: int = 8):
    """Replicates ``unique_rows`` (trimesh) as used by UGSCNN ``subdivide``.

    Rows are rounded to ``digits`` decimals, viewed as ``np.void`` byte strings
    and sorted bytewise (little-endian int64 bytes, lexicographic) by
    ``np.unique``; the *first* occurrence of each distinct row is kept.  The
    byte order is emulated with big-endian reinterpretation + ``lexsort``.
    """
    as_int = np.round(data * 10**digits).astype("<i8")
    sw = np.ascontiguousarray(as_int).view(np.uint8).reshape(-1, 3, 8)
    sw = np.ascontiguousarray(sw).view(">u8").reshape(-1, 3)
    order = np.lexsort((sw[:, 2], sw[:, 1], sw[:, 0]))  # stable
    s = sw[order]
    newgrp = np.ones(len(s), dtype=bool)
    newgrp[1:] = (s[1:] != s[:-1]).any(axis=1)
    gid = np.cumsum(newgrp) - 1
    unique = order[newgrp]
    inverse = np.empty(len(as_int), dtype=np.int64)
    inverse[order] = gid
    return unique, inverse


def _ugscnn(level: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + 5.0**0.5) / 2.0
    v = np.reshape(
        [-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t,
         0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1],
        (-1, 3),
    ) / 1.9021130325903071  # fmt: skip
    f = np.reshape(
        [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
         1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
         3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
         4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1],
        (-1, 3),
    ).astype(np.int64)  # fmt: skip
    for _ in range(level):
        nfaces = len(f)
        pairs = ((0, 1), (1, 2), (2, 0))
        mid = np.vstack([v[f[:, list(g)]].mean(axis=1) for g in pairs])
        mid_idx = np.arange(nfaces * 3).reshape((3, -1)).T
        unique, inverse = _ugscnn_unique_rows(mid)
        mid = mid[unique]
        mid_idx = inverse[mid_idx] + len(v)
        ch = np.column_stack(
            [
                f[:, 0], mid_idx[:, 0], mid_idx[:, 2],
                mid_idx[:, 0], f[:, 1], mid_idx[:, 1],
                mid_idx[:, 2], mid_idx[:, 1], f[:, 2],
                mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2],
            ]
        ).reshape((-1, 3))  # fmt: skip
        nf = np.vstack((f, ch[nfaces:]))
        nf[:nfaces] = ch[:nfaces]
        v = np.vstack((v, mid))
        f = nf
        # normalize() in the original: v += unit * (1 - |v|)
        s = np.sqrt((v**2).sum(axis=1))
        unit = v / s.reshape((-1, 1))
        v = v + unit * (1.0 - s).reshape((-1, 1))
    return v, f


def icosphere(level: int, scheme: str = "biscuit") -> Mesh:
    """Return the level-``level`` icosphere (0 <= level <= 7).

    Vertices are float64 on the unit sphere (exactly renormalised midpoints),
    faces int32, counter-clockwise seen from outside.  ``scheme`` is one of
    :data:`SCHEMES`; see the module docstring.  Counts are given by
    :func:`icosphere_counts`.
    """
    if not isinstance(level, (int, np.integer)) or not 0 <= level <= MAX_LEVEL:
        raise ValueError(f"level must be an integer in 0..{MAX_LEVEL}")
    level = int(level)
    if scheme == "biscuit":
        v, f = _biscuit(level, two_pass=True)
    elif scheme == "biscuit_ico4":
        v, f = _biscuit(level, two_pass=False)
    elif scheme == "ugscnn":
        v, f = _ugscnn(level)
    else:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
    return Mesh(
        v,
        f,
        frame=None,
        units=None,
        meta={"generator": "pybiscuit.sphere.icosphere", "scheme": scheme, "level": level},
    )
