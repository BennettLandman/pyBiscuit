"""Triangle-mesh topology: edges, adjacency, boundaries, components, Euler/genus.

Everything here is vectorised NumPy/SciPy-sparse and works on meshes of a few
hundred thousand faces in about a second. Every public function accepts either
a :class:`~pybiscuit.mesh.model.Mesh` or ``(vertices, faces)``; ``vertices`` may
also be just the vertex count (an ``int``) since geometry is never used.

Conventions
-----------
* Face ``f = (v0, v1, v2)`` has three directed half-edges
  ``k = 0, 1, 2``: ``(v0, v1)``, ``(v1, v2)``, ``(v2, v0)``. Half-edge ``h``
  belongs to face ``h // 3`` at slot ``h % 3``; face-edge ``k`` is the edge
  *opposite none*, i.e. it starts at corner ``k``.
* Unique (undirected) edges are stored as ``(lo, hi)`` rows sorted
  lexicographically, so edge ids are deterministic.
* "Unreferenced" vertices (used by no face) are ignored by the Euler
  characteristic and by component labelling (label ``-1``).
* A vertex/edge with non-manifold structure is reported, never silently fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

__all__ = [
    "Adjacency",
    "EdgeTable",
    "OneRing",
    "OrientationReport",
    "TopologySummary",
    "boundary_edges",
    "boundary_loops",
    "check_orientation",
    "connected_components",
    "euler_characteristic",
    "face_adjacency",
    "face_adjacency_matrix",
    "genus",
    "non_manifold_edges",
    "non_manifold_vertices",
    "one_ring",
    "topology_summary",
    "unique_edges",
    "unreferenced_vertices",
    "vertex_faces",
    "vertex_neighbors",
]


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #
class Adjacency(NamedTuple):
    """CSR adjacency: row ``i`` is ``indices[indptr[i]:indptr[i + 1]]``."""

    indptr: np.ndarray
    indices: np.ndarray

    def row(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i] : self.indptr[i + 1]]

    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)


def _coerce(mesh, faces=None) -> tuple[int, np.ndarray]:
    """Return ``(n_vertices, faces int64 (M, 3))`` from a Mesh or (vertices, faces)."""
    if faces is None:
        if not hasattr(mesh, "faces"):
            raise TypeError("pass a Mesh, or (vertices, faces)")
        nv, f = int(mesh.n_vertices), mesh.faces
    else:
        nv = int(mesh) if np.isscalar(mesh) else int(np.shape(mesh)[0])
        f = faces
    f = np.asarray(f)
    if f.size == 0:
        return nv, np.zeros((0, 3), dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    f = f.astype(np.int64, copy=False)
    if f.min() < 0 or f.max() >= nv:
        raise ValueError("face index out of range")
    return nv, f


def _csr_from_sorted_rows(rows: np.ndarray, n: int) -> np.ndarray:
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #
@dataclass
class EdgeTable:
    """Unique undirected edges of a triangle mesh.

    Attributes
    ----------
    edges : (E, 2) int64
        ``(lo, hi)`` vertex pairs, lexicographically sorted.
    face_edges : (M, 3) int64
        Edge id of face-edge ``k`` (from corner ``k`` to corner ``k + 1``).
    valence : (E,) int64
        Number of faces using each edge (1 boundary, 2 interior, >2 non-manifold).
    edge_faces : (E, 2) int64
        First two faces using each edge (``-1`` padded). For manifold edges this
        is the complete incident-face list.
    edge_halfedges : (E, 2) int64
        Half-edge ids (``3 * face + slot``) matching ``edge_faces``.
    n_vertices : int
    """

    edges: np.ndarray
    face_edges: np.ndarray
    valence: np.ndarray
    edge_faces: np.ndarray
    edge_halfedges: np.ndarray
    n_vertices: int

    def __len__(self) -> int:
        return self.edges.shape[0]


def unique_edges(mesh, faces=None) -> EdgeTable:
    """Build the :class:`EdgeTable` (unique edges and the edge -> faces map)."""
    nv, f = _coerce(mesh, faces)
    m = f.shape[0]
    a = f.ravel()
    b = np.roll(f, -1, axis=1).ravel()
    key = np.minimum(a, b) * max(nv, 1) + np.maximum(a, b)
    ukey, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    e = ukey.shape[0]
    edges = np.stack([ukey // max(nv, 1), ukey % max(nv, 1)], axis=1)
    order = np.argsort(inv, kind="stable")  # half-edges grouped by edge id
    starts = np.zeros(e, dtype=np.int64)
    if e:
        np.cumsum(counts[:-1], out=starts[1:])
    he = np.full((e, 2), -1, dtype=np.int64)
    he[:, 0] = order[starts]
    two = counts >= 2
    he[two, 1] = order[starts[two] + 1]
    ef = np.where(he >= 0, he // 3, -1)
    return EdgeTable(
        edges=edges,
        face_edges=inv.reshape(m, 3),
        valence=counts.astype(np.int64),
        edge_faces=ef,
        edge_halfedges=he,
        n_vertices=nv,
    )


def non_manifold_edges(mesh, faces=None, *, edges: EdgeTable | None = None) -> np.ndarray:
    """Edge ids used by more than two faces."""
    et = edges if edges is not None else unique_edges(mesh, faces)
    return np.flatnonzero(et.valence > 2)


def boundary_edges(mesh, faces=None, *, edges: EdgeTable | None = None) -> np.ndarray:
    """Edge ids used by exactly one face."""
    et = edges if edges is not None else unique_edges(mesh, faces)
    return np.flatnonzero(et.valence == 1)


# --------------------------------------------------------------------------- #
# vertex adjacency
# --------------------------------------------------------------------------- #
def vertex_faces(mesh, faces=None) -> Adjacency:
    """Vertex -> incident faces (CSR, faces ascending within each vertex)."""
    nv, f = _coerce(mesh, faces)
    flat = f.ravel()
    order = np.argsort(flat, kind="stable")
    return Adjacency(_csr_from_sorted_rows(flat, nv), order // 3)


def vertex_neighbors(mesh, faces=None, *, edges: EdgeTable | None = None) -> Adjacency:
    """Vertex -> adjacent vertices (CSR, sorted ascending; *not* ring-ordered).

    Use :func:`one_ring` for ordered rings.
    """
    et = edges if edges is not None else unique_edges(mesh, faces)
    nv = et.n_vertices
    rows = np.concatenate([et.edges[:, 0], et.edges[:, 1]])
    cols = np.concatenate([et.edges[:, 1], et.edges[:, 0]])
    order = np.argsort(rows * max(nv, 1) + cols, kind="stable")
    return Adjacency(_csr_from_sorted_rows(rows, nv), cols[order])


class OneRing(NamedTuple):
    """Ordered one-ring neighbours (CSR) for manifold vertices.

    ``ring`` follows the face winding (counter-clockwise when viewed from
    outside for outward-oriented meshes). For an interior vertex the ring is
    cyclic with ``degree`` entries and is *not* closed (first != last). For a
    boundary vertex it is an open chain from one boundary neighbour to the
    other with ``degree + 1`` entries. Vertices that are unreferenced,
    non-manifold (several fans, pinch points, non-manifold edges) or sit on
    inconsistently oriented faces have ``valid == False`` and an empty ring.
    """

    indptr: np.ndarray
    indices: np.ndarray
    valid: np.ndarray
    is_boundary: np.ndarray

    def ring(self, v: int) -> np.ndarray:
        return self.indices[self.indptr[v] : self.indptr[v + 1]]


def one_ring(mesh, faces=None) -> OneRing:
    """Ordered one-ring neighbours of every manifold vertex (vectorised).

    Corners ``(v, a, b)`` (face rotated so ``v`` is first) link to the corner
    whose ``a`` equals this corner's ``b``; following that successor chain
    walks the fan around ``v``.
    """
    nv, f = _coerce(mesh, faces)
    m = f.shape[0]
    n_c = 3 * m
    big = max(nv, 1)
    v = f.ravel()
    a = np.roll(f, -1, axis=1).ravel()
    b = np.roll(f, -2, axis=1).ravel()

    empty = OneRing(
        np.zeros(nv + 1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(nv, dtype=bool),
        np.zeros(nv, dtype=bool),
    )
    if m == 0:
        return empty

    key = v * big + a
    order = np.argsort(key, kind="stable")
    sk = key[order]
    bad = np.zeros(nv, dtype=bool)
    dup = sk[1:] == sk[:-1]
    bad[sk[1:][dup] // big] = True  # same directed edge twice at v

    q = v * big + b
    pos = np.minimum(np.searchsorted(sk, q), n_c - 1)
    found = sk[pos] == q
    succ = np.where(found, order[pos], -1)

    deg = np.bincount(v, minlength=nv)
    pred_count = np.bincount(succ[succ >= 0], minlength=n_c)
    start_corners = np.flatnonzero(pred_count == 0)
    n_starts = np.bincount(v[start_corners], minlength=nv)
    bad |= n_starts > 1

    first_corner = np.full(nv, -1, dtype=np.int64)
    first_corner[v[::-1]] = np.arange(n_c - 1, -1, -1)  # smallest corner per vertex
    start = first_corner.copy()
    start[v[start_corners]] = start_corners  # open fans start at the unmatched corner

    cand = np.flatnonzero((deg > 0) & ~bad)
    cur = start[cand]
    st = cur.copy()
    live = cand
    rec_v, rec_t, rec_val = [], [], []
    length = np.zeros(nv, dtype=np.int64)
    ok = np.zeros(nv, dtype=bool)
    is_b = n_starts == 1

    t = 0
    while live.size:
        rec_v.append(live)
        rec_t.append(np.full(live.size, t, dtype=np.int64))
        rec_val.append(a[cur])
        nxt = succ[cur]
        open_end = nxt == -1
        closed_end = (nxt == st) & ~open_end
        if open_end.any():
            lv = live[open_end]
            rec_v.append(lv)
            rec_t.append(np.full(lv.size, t + 1, dtype=np.int64))
            rec_val.append(b[cur[open_end]])
            length[lv] = t + 2
            ok[lv] = (t + 1) == deg[lv]
        if closed_end.any():
            lv = live[closed_end]
            length[lv] = t + 1
            ok[lv] = (t + 1) == deg[lv]
        keep = ~(open_end | closed_end)
        t += 1
        # a fan can never be longer than deg: anything still walking past it is bad
        keep &= deg[live] > t
        live, cur, st = live[keep], nxt[keep], st[keep]

    valid = ok & ~bad & (deg > 0)
    # open fans must have been started from the unmatched corner (n_starts == 1)
    valid &= (is_b & (length == deg + 1)) | (~is_b & (length == deg))
    lens = np.where(valid, length, 0)
    indptr = np.zeros(nv + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    indices = np.zeros(indptr[-1], dtype=np.int64)
    rv = np.concatenate(rec_v)
    rt = np.concatenate(rec_t)
    rval = np.concatenate(rec_val)
    sel = valid[rv]
    indices[indptr[rv[sel]] + rt[sel]] = rval[sel]
    return OneRing(indptr, indices, valid, is_b & valid)


def non_manifold_vertices(mesh, faces=None) -> np.ndarray:
    """Referenced vertices whose face fan is not a single (open or closed) disc.

    Covers pinch vertices, vertices on non-manifold edges and vertices touching
    inconsistently oriented faces (the fan walk follows face winding).
    """
    nv, f = _coerce(mesh, faces)
    ring = one_ring(nv, f)
    referenced = np.zeros(nv, dtype=bool)
    referenced[f.ravel()] = True
    return np.flatnonzero(referenced & ~ring.valid)


def unreferenced_vertices(mesh, faces=None) -> np.ndarray:
    """Vertex ids not used by any face."""
    nv, f = _coerce(mesh, faces)
    used = np.zeros(nv, dtype=bool)
    used[f.ravel()] = True
    return np.flatnonzero(~used)


# --------------------------------------------------------------------------- #
# face adjacency
# --------------------------------------------------------------------------- #
def face_adjacency(mesh, faces=None, *, edges: EdgeTable | None = None) -> np.ndarray:
    """(M, 3) neighbouring face across face-edge ``k`` (``-1`` if none).

    Face-edge ``k`` runs from corner ``k`` to corner ``k + 1``. Boundary edges
    and non-manifold edges (valence > 2) give ``-1``.
    """
    nv, f = _coerce(mesh, faces)
    et = edges if edges is not None else unique_edges(nv, f)
    fe = et.face_edges
    own = np.arange(f.shape[0])[:, None]
    ef = et.edge_faces[fe]  # (M, 3, 2)
    other = np.where(ef[..., 0] == own, ef[..., 1], ef[..., 0])
    other = np.where(et.valence[fe] == 2, other, -1)
    return other


def face_adjacency_matrix(mesh, faces=None, *, edges: EdgeTable | None = None) -> sparse.csr_matrix:
    """Symmetric (M, M) boolean CSR matrix of faces sharing a manifold edge."""
    nv, f = _coerce(mesh, faces)
    adj = face_adjacency(nv, f, edges=edges)
    rows = np.repeat(np.arange(f.shape[0]), 3)
    cols = adj.ravel()
    keep = cols >= 0
    data = np.ones(int(keep.sum()), dtype=bool)
    return sparse.csr_matrix((data, (rows[keep], cols[keep])), shape=(f.shape[0],) * 2)


# --------------------------------------------------------------------------- #
# boundary loops
# --------------------------------------------------------------------------- #
def boundary_loops(mesh, faces=None, *, edges: EdgeTable | None = None) -> list[np.ndarray]:
    """Ordered boundary loops as lists of vertex ids (closed, last != first).

    Loops follow the winding of the boundary half-edges of the first edge in
    each loop. At pinch vertices (several boundary loops touching) the walk
    picks the lowest-numbered unused edge, so the result is deterministic but
    the split between loops there is arbitrary. Open chains (can only occur for
    degenerate input) are returned as-is.
    """
    nv, f = _coerce(mesh, faces)
    et = edges if edges is not None else unique_edges(nv, f)
    be = np.flatnonzero(et.valence == 1)
    if be.size == 0:
        return []
    he = et.edge_halfedges[be, 0]
    face = he // 3
    k = he % 3
    d0 = f[face, k]
    d1 = f[face, (k + 1) % 3]  # directed boundary edges
    nxt: dict[int, list[int]] = {}
    for i in np.argsort(d0 * max(nv, 1) + d1, kind="stable")[::-1]:  # pop() gives lowest first
        nxt.setdefault(int(d0[i]), []).append(int(i))
    und: dict[int, list[int]] = {}
    for i in range(be.size):
        und.setdefault(int(d0[i]), []).append(i)
        und.setdefault(int(d1[i]), []).append(i)
    used = np.zeros(be.size, dtype=bool)
    loops: list[np.ndarray] = []
    for i0 in range(be.size):
        if used[i0]:
            continue
        loop = [int(d0[i0])]
        used[i0] = True
        cur = int(d1[i0])
        while cur != loop[0]:
            loop.append(cur)
            step = -1
            for i in nxt.get(cur, ()):  # prefer consistently directed continuation
                if not used[i]:
                    step = i
                    break
            if step < 0:  # inconsistent winding: fall back to undirected walk
                for i in und.get(cur, ()):
                    if not used[i]:
                        step = i
                        break
            if step < 0:
                break
            used[step] = True
            cur = int(d1[step]) if int(d0[step]) == cur else int(d0[step])
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #
def connected_components(mesh, faces=None) -> tuple[int, np.ndarray, np.ndarray]:
    """Connected components through shared *vertices*.

    Returns ``(n_components, vertex_labels, face_labels)``. Labels are
    contiguous ``0..n-1`` ordered by lowest vertex id; unreferenced vertices
    get label ``-1`` and do not count as components.
    """
    nv, f = _coerce(mesh, faces)
    if f.shape[0] == 0:
        return 0, np.full(nv, -1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    g = sparse.coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(nv, nv))
    _, lab = csgraph.connected_components(g.tocsr(), directed=False)
    used = np.zeros(nv, dtype=bool)
    used[f.ravel()] = True
    out = np.full(nv, -1, dtype=np.int64)
    uniq, inv = np.unique(lab[used], return_inverse=True)
    out[used] = inv.reshape(-1)
    return int(uniq.size), out, out[f[:, 0]]


# --------------------------------------------------------------------------- #
# Euler characteristic / genus
# --------------------------------------------------------------------------- #
def euler_characteristic(mesh, faces=None) -> int:
    """``V - E + F`` over referenced vertices (unreferenced vertices ignored)."""
    nv, f = _coerce(mesh, faces)
    et = unique_edges(nv, f)
    n_ref = nv - unreferenced_vertices(nv, f).size
    return int(n_ref - len(et) + f.shape[0])


@dataclass
class TopologySummary:
    """Aggregate topology facts; see :func:`topology_summary`."""

    n_vertices: int
    n_referenced_vertices: int
    n_edges: int
    n_faces: int
    euler_characteristic: int
    n_components: int
    n_boundary_loops: int
    genus: int  # sum over components
    component_euler: np.ndarray
    component_boundary_loops: np.ndarray
    component_genus: np.ndarray
    n_boundary_edges: int
    n_non_manifold_edges: int
    n_non_manifold_vertices: int
    n_unreferenced_vertices: int
    is_closed: bool
    is_manifold: bool
    is_consistently_oriented: bool

    @property
    def is_closed_manifold(self) -> bool:
        return self.is_closed and self.is_manifold


def topology_summary(mesh, faces=None) -> TopologySummary:
    """Compute chi, components, boundary loops, genus and manifold/orientation flags.

    Per component ``c``: ``chi_c = 2 - 2 g_c - b_c`` (``b_c`` boundary loops), so
    ``g_c = (2 - chi_c - b_c) / 2``. Genus is only meaningful when
    ``is_manifold``; for non-manifold input it is computed by the same formula
    (floor division) and should be disregarded.
    """
    nv, f = _coerce(mesh, faces)
    et = unique_edges(nv, f)
    ncomp, vlab, flab = connected_components(nv, f)
    loops = boundary_loops(nv, f, edges=et)
    used = vlab >= 0
    cv = np.bincount(vlab[used], minlength=ncomp)
    cf = np.bincount(flab, minlength=ncomp)
    ce = np.bincount(vlab[et.edges[:, 0]], minlength=ncomp) if len(et) else np.zeros(ncomp, int)
    cb = np.zeros(ncomp, dtype=np.int64)
    for lp in loops:
        cb[vlab[lp[0]]] += 1
    cchi = cv - ce + cf
    cg = (2 - cchi - cb) // 2
    nm_e = int((et.valence > 2).sum())
    nm_v = non_manifold_vertices(nv, f).size
    orient = check_orientation(nv, f, edges=et)
    return TopologySummary(
        n_vertices=nv,
        n_referenced_vertices=int(used.sum()),
        n_edges=len(et),
        n_faces=f.shape[0],
        euler_characteristic=int(cchi.sum()),
        n_components=ncomp,
        n_boundary_loops=len(loops),
        genus=int(cg.sum()),
        component_euler=cchi,
        component_boundary_loops=cb,
        component_genus=cg,
        n_boundary_edges=int((et.valence == 1).sum()),
        n_non_manifold_edges=nm_e,
        n_non_manifold_vertices=int(nm_v),
        n_unreferenced_vertices=int(nv - used.sum()),
        is_closed=bool((et.valence == 2).all()) if len(et) else False,
        is_manifold=bool(nm_e == 0 and nm_v == 0),
        is_consistently_oriented=orient.is_consistent,
    )


def genus(mesh, faces=None) -> int:
    """Total genus summed over connected components (boundaries accounted for)."""
    return topology_summary(mesh, faces).genus


# --------------------------------------------------------------------------- #
# orientation
# --------------------------------------------------------------------------- #
@dataclass
class OrientationReport:
    """Result of :func:`check_orientation`.

    Attributes
    ----------
    is_consistent : bool
        No directed edge is used by more than one face (the closed-manifold
        criterion: each directed edge once).
    is_orientable : bool
        Faces can be given a globally consistent orientation (False e.g. for a
        Moebius strip). Judged over manifold (valence-2) edges only.
    flipped_faces : (K,) int64
        Faces to reverse to make every component consistent, choosing in each
        face-connected component the *minority* orientation class (ties keep the
        class of the lowest-numbered face).
    inconsistent_edges : (J,) int64
        Edge ids whose two faces traverse the edge in the same direction.
    n_duplicate_directed_edges : int
        Number of distinct directed edges used more than once.
    """

    is_consistent: bool
    is_orientable: bool
    flipped_faces: np.ndarray
    inconsistent_edges: np.ndarray
    n_duplicate_directed_edges: int


def check_orientation(mesh, faces=None, *, edges: EdgeTable | None = None) -> OrientationReport:
    """Check winding consistency and find faces that should be flipped.

    The sign of every manifold edge (+1 opposite directions = consistent, -1 same
    direction) is propagated by BFS over the face graph (one pass, via a virtual
    super-root over all components) to give each face a parity; the minority
    parity of each component is reported as flipped.
    """
    nv, f = _coerce(mesh, faces)
    m = f.shape[0]
    et = edges if edges is not None else unique_edges(nv, f)
    big = max(nv, 1)
    dkey = f.ravel() * big + np.roll(f, -1, axis=1).ravel()
    _, dcounts = np.unique(dkey, return_counts=True)
    ndup = int((dcounts > 1).sum())

    two = np.flatnonzero(et.valence == 2)
    h0, h1 = et.edge_halfedges[two, 0], et.edge_halfedges[two, 1]
    f0, f1 = h0 // 3, h1 // 3
    same = f.ravel()[h0] == f.ravel()[h1]  # same start vertex => same direction
    inconsistent = two[same]
    if m == 0 or two.size == 0:
        return OrientationReport(ndup == 0, True, np.zeros(0, dtype=np.int64), inconsistent, ndup)

    # face graph over manifold edges, edge weight = 1 (consistent) / 2 (flipped)
    w = np.where(same, 2, 1).astype(np.int8)
    ncomp, flab = csgraph.connected_components(
        sparse.coo_matrix((np.ones(two.size, dtype=np.int8), (f0, f1)), shape=(m, m)).tocsr(),
        directed=False,
    )
    roots = np.full(ncomp, -1, dtype=np.int64)
    roots[flab[::-1]] = np.arange(m - 1, -1, -1)  # lowest face per component
    # super-root node m, connected to each component root
    rows = np.concatenate([f0, np.full(ncomp, m)])
    cols = np.concatenate([f1, roots])
    wts = np.concatenate([w, np.ones(ncomp, dtype=np.int8)])
    g = sparse.coo_matrix(
        (np.concatenate([wts, wts]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(m + 1, m + 1),
    ).tocsr()
    order, pred = csgraph.breadth_first_order(g, m, directed=False, return_predecessors=True)
    child = order[1:]
    ew = (np.asarray(g[pred[child], child]).ravel() == 2).astype(np.int8)
    parity = np.zeros(m + 1, dtype=np.int8)
    # parents precede children in BFS order; O(M) scalar pass over plain lists
    par_l = parity.tolist()
    for c, pr, e in zip(child.tolist(), pred[child].tolist(), ew.tolist(), strict=True):
        par_l[c] = par_l[pr] ^ e
    parity = np.asarray(par_l, dtype=np.int8)

    p = parity[:m]
    # orientability: every manifold edge must agree with the tree-derived parity
    orientable = bool(np.all((p[f0] ^ p[f1]) == same.astype(np.int8)))

    ones = np.bincount(flab, weights=p, minlength=ncomp)
    sizes = np.bincount(flab, minlength=ncomp)
    root_par = p[roots]
    # majority class per component; ties keep the root's class
    ones_is_minority = ones * 2 < sizes
    minority = np.where(ones * 2 == sizes, 1 - root_par, np.where(ones_is_minority, 1, 0))
    flipped = np.flatnonzero(p == minority[flab])
    return OrientationReport(ndup == 0, orientable, flipped, inconsistent, ndup)
