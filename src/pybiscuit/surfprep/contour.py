"""Cutting-plane search for the hemisphere split (spec §4.2 steps 1-2, T39).

Two pieces live here:

* :func:`candidate_offsets` is a port of the MATLAB code that builds the 101
  candidate plane offsets (``hemi_separation.m: getCuttingPlane``).
* :func:`mesh_contour` stands in for the closed ``MeshContour`` binary.  The
  rule that binary uses to choose among the offsets is **not visible in the
  source**; :data:`PLANE_RULES` holds the hypotheses and
  :func:`select_plane` is the *single* place where the choice is made.

HYPOTHESIS H-MC1 (default, ``"shortest"``): ``MeshContour`` returns the plane
whose intersection polyline with the mesh is shortest.  Unverified; it needs a
black-box probe of the image binary (spec §15 item 4).  Nothing else in the
package depends on the rule, only on :func:`select_plane`'s signature.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pybiscuit.mesh.topology import unique_edges

RANGE = (-0.5, 0.5)
STEP = 0.01
SMOOTH_ITERS = 20


def smooth_neighbor_mean(vertices, faces, iterations: int, frozen=None) -> np.ndarray:
    """Jacobi smoothing, vertex := mean of its 1-ring neighbours (no self weight).

    Mirrors ``smooth``/``adjacency`` in ``hemi_separation.m``.  Vertices in
    ``frozen`` (bool mask) and vertices without neighbours keep their position;
    frozen vertices still act as neighbours of the others.
    """
    import scipy.sparse as sp

    v = np.asarray(vertices, dtype=np.float64)
    n = v.shape[0]
    faces = np.asarray(faces)
    e = unique_edges(n, faces).edges
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    w = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    if frozen is not None:
        w = sp.diags((~np.asarray(frozen, bool)).astype(float)) @ w
    deg = np.asarray(w.sum(axis=1)).ravel()
    inv = np.zeros(n)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    cur = v
    for _ in range(iterations):
        cur = (w @ cur) * inv[:, None]
        cur[~nz] = v[~nz]
    return cur


def candidate_offsets(x) -> np.ndarray:
    """The 101 plane offsets ``delta`` of ``getCuttingPlane`` (plane: ``x + delta = 0``).

    ``x`` is the first coordinate of the smoothed reference vertices.  For each
    ``r`` in -0.5:0.01:0.5, start at ``-mean(x) + r`` and move the plane to the
    midpoint between the closest vertex on one side (``j``) and the closest on
    the other (``k``).
    """
    x = np.asarray(x, dtype=np.float64)
    r = RANGE[0] + STEP * np.arange(int(round((RANGE[1] - RANGE[0]) / STEP)) + 1)
    out = np.empty(r.size)
    mx = x.mean()
    for i, ri in enumerate(r):
        d = -mx + ri
        j = int(np.argmin(np.abs(x + d)))
        m = x + d
        if x[j] + d > 0:
            m = -m
        m = np.where(m < 0, np.inf, m)
        k = int(np.argmin(m))
        out[i] = -(x[j] + x[k]) / 2.0
    return out


def _cut_segments(vertices, faces, normal, d):
    """Faces cut by ``normal . p + d = 0`` and the length of each cut segment.

    A face is cut when it has vertices strictly on both sides (signed distance
    ``< 0`` and ``> 0``); a vertex exactly on the plane counts as non-negative.
    """
    s = vertices @ normal + d
    fs = s[faces]
    neg = fs < 0
    cut = neg.any(axis=1) & (~neg).any(axis=1)
    ids = np.flatnonzero(cut)
    if ids.size == 0:
        return ids, np.empty(0)
    tri = faces[ids]
    sv = fs[ids]
    p = vertices[tri]  # (k, 3, 3)
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cross = (sv[:, a] < 0) != (sv[:, b] < 0)
        denom = np.where(cross, sv[:, a] - sv[:, b], 1.0)
        t = sv[:, a] / denom
        q = p[:, a] + t[:, None] * (p[:, b] - p[:, a])
        pts.append((cross, q))
    # exactly two of the three edges cross for a cut triangle
    idx = np.argsort(~np.stack([c for c, _ in pts], axis=1), axis=1, kind="stable")[:, :2]
    qs = np.stack([q for _, q in pts], axis=1)
    q0 = qs[np.arange(ids.size), idx[:, 0]]
    q1 = qs[np.arange(ids.size), idx[:, 1]]
    return ids, np.linalg.norm(q1 - q0, axis=1)


def rule_shortest(lengths: list[float], n_faces: list[int]) -> list[int]:
    """H-MC1: rank planes by total polyline length (stable; ties -> lower index)."""
    return list(np.argsort(np.asarray(lengths), kind="stable"))


def rule_fewest_faces(lengths: list[float], n_faces: list[int]) -> list[int]:
    """H-MC2 (alternative): rank planes by number of cut faces, then length."""
    order = np.lexsort((np.asarray(lengths), np.asarray(n_faces)))
    return list(order)


PLANE_RULES: dict[str, Callable[[list[float], list[int]], list[int]]] = {
    "shortest": rule_shortest,
    "fewest_faces": rule_fewest_faces,
}
DEFAULT_RULE = "shortest"


def select_plane(lengths, n_faces, rule: str = DEFAULT_RULE) -> list[int]:
    """THE isolated MeshContour hypothesis: candidate indices, best first."""
    if rule not in PLANE_RULES:
        raise ValueError(f"unknown plane rule {rule!r}; choose from {sorted(PLANE_RULES)}")
    return [int(i) for i in PLANE_RULES[rule](list(lengths), list(n_faces))]


def mesh_contour(vertices, faces, normal, offsets, rule: str = DEFAULT_RULE):
    """Stand-in for ``MeshContour``: ranked planes and their cut faces.

    Returns ``(ranking, cut_faces)`` where ``ranking`` lists offset indices
    best first and ``cut_faces[i]`` is the ascending 0-based id array of the
    faces cut by plane ``i``.  The binary prints the faces of ``ranking[0]``
    followed by ``ranking[0]`` (spec T39).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    normal = np.asarray(normal, dtype=np.float64)
    cut_faces, lengths, counts = [], [], []
    for d in offsets:
        ids, seg = _cut_segments(vertices, faces, normal, float(d))
        cut_faces.append(ids)
        lengths.append(float(seg.sum()))
        counts.append(int(ids.size))
    return select_plane(lengths, counts, rule), cut_faces
