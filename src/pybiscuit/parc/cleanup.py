"""Connected-component parcellation cleanup (plan row B7.5).

Port of ``roi_correction.m`` (spec §6.8): iteratively keeps only the largest connected
component of each parcel label on the equal-label mesh subgraph, merges adjacent
removed fragments (marked ``-1``), and reassigns each ``-1`` component either to its
unique neighbouring label or via a local alpha-expansion graph cut (:func:`gcut_from_edges`)
among its candidate neighbouring labels.

Indexing convention:
* By default (``index_base=0``, ``unknown_class=0``), ``label`` contains 0-based class
  indices ``0 .. L - 1`` matching row indices of ``prob`` (shape ``(L, N)``), where
  class ``0`` is the "unknown" / medial-wall class (dropped from ``cand`` when
  ``len(cand) > 1``, matching ``cand = setdiff(cand, 1)`` in 1-based MATLAB
  ``roi_correction.m:75-77``).
* To run with 1-based class indices ``1 .. L`` as in MATLAB ``roi_correction.m``, pass
  ``index_base=1, unknown_class=1``.

 Edge cases documented in spec §6.8 and §14 item 4:
* **Equal-size largest components**: resolved deterministically in favour of the
  component with the **lowest minimum vertex index**, and recorded in
  :attr:`CleanupInfo.size_ties` / :attr:`CleanupInfo.n_size_ties`.
* **Empty candidate set (``len(cand) == 0``)**: occurs when a non-largest fragment is a
  disconnected mesh island with no edges to any kept component.  MATLAB crashes by
  calling ``gcut_`` with ``prob([], ...)``.  Controlled by ``on_empty_cand``:
  ``"restore"`` (default: restores the component's pre-iteration labels so no ``-1``
  labels remain and the loop terminates cleanly), ``"argmax"`` (assigns per-vertex
  ``argmax`` of ``prob``), or ``"raise"`` (raises ``ValueError``).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numba import njit

from .graphcut import faces_to_directed_edges, gcut_from_edges

__all__ = [
    "CleanupInfo",
    "SizeTieEvent",
    "equal_label_components",
    "roi_correction",
]


class SizeTieEvent(NamedTuple):
    """Record of an equal-size tie between largest components of a label."""

    iteration: int
    label: int
    size: int
    kept_min_vertex: int
    tied_min_vertex: int


class CleanupInfo(NamedTuple):
    """Diagnostics returned by :func:`roi_correction` when ``return_info=True``."""

    n_iterations: int
    n_size_ties: int
    size_ties: tuple[SizeTieEvent, ...]
    n_empty_cand: int
    n_local_gcuts: int
    n_single_cand: int


@njit(cache=True)
def _build_adj_csr(n: int, rc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = rc.shape[0]
    deg = np.zeros(n, dtype=np.int32)
    for e in range(m):
        deg[rc[e, 0]] += 1
    head = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        head[i + 1] = head[i] + deg[i]
    cur = head[:-1].copy()
    to = np.empty(m, dtype=np.int32)
    for e in range(m):
        u = rc[e, 0]
        to[cur[u]] = rc[e, 1]
        cur[u] += 1
    return head, to


@njit(cache=True)
def _equal_label_components_numba(
    n: int,
    head: np.ndarray,
    to: np.ndarray,
    parc: np.ndarray,
    bins: np.ndarray,
    queue: np.ndarray,
    comp_min_vertex: np.ndarray,
    comp_size: np.ndarray,
    comp_label: np.ndarray,
) -> int:
    for i in range(n):
        bins[i] = -1
    n_comp = 0
    for start in range(n):
        if bins[start] != -1:
            continue
        cid = n_comp
        n_comp += 1
        lbl = parc[start]
        bins[start] = cid
        comp_min_vertex[cid] = start
        comp_label[cid] = lbl
        q_head = 0
        q_tail = 1
        queue[0] = start
        while q_head < q_tail:
            u = queue[q_head]
            q_head += 1
            for h in range(head[u], head[u + 1]):
                v = to[h]
                if bins[v] == -1 and parc[v] == lbl:
                    bins[v] = cid
                    queue[q_tail] = v
                    q_tail += 1
        comp_size[cid] = q_tail
    return n_comp


def equal_label_components(
    label: np.ndarray, faces: np.ndarray, n_vertices: int | None = None
) -> tuple[int, np.ndarray]:
    """Compute connected components of the equal-label mesh subgraph.

    Components are numbered ``0 .. n_comp - 1`` in ascending order of their minimum
    vertex index.
    """
    parc = np.ascontiguousarray(label, dtype=np.int32)
    n = parc.shape[0] if n_vertices is None else int(n_vertices)
    rc = faces_to_directed_edges(faces)
    head, to = _build_adj_csr(n, rc)
    bins = np.empty(n, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    comp_min_vertex = np.empty(n, dtype=np.int32)
    comp_size = np.empty(n, dtype=np.int32)
    comp_label = np.empty(n, dtype=np.int32)
    n_comp = _equal_label_components_numba(
        n, head, to, parc, bins, queue, comp_min_vertex, comp_size, comp_label
    )
    return int(n_comp), bins


@njit(cache=True)
def _collect_component_neighbors_and_edges(
    idx: np.ndarray,
    bins: np.ndarray,
    cid: int,
    parc: np.ndarray,
    head: np.ndarray,
    to: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all neighbour labels (excluding -1) and local directed edges within ``idx``."""
    k = idx.shape[0]
    total_deg = 0
    for i in range(k):
        u = idx[i]
        total_deg += head[u + 1] - head[u]

    nbr_labels = np.empty(total_deg, dtype=np.int32)
    n_nbr = 0
    local_edges = np.empty((total_deg, 2), dtype=np.int32)
    n_edges = 0

    for i in range(k):
        u = idx[i]
        for h in range(head[u], head[u + 1]):
            v = to[h]
            if bins[v] == cid:
                # Binary search v in sorted idx
                lo = 0
                hi = k - 1
                pos = -1
                while lo <= hi:
                    mid = (lo + hi) >> 1
                    vm = idx[mid]
                    if vm == v:
                        pos = mid
                        break
                    elif vm < v:
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if pos != -1:
                    local_edges[n_edges, 0] = i
                    local_edges[n_edges, 1] = pos
                    n_edges += 1
            else:
                lv = parc[v]
                if lv != -1:
                    nbr_labels[n_nbr] = lv
                    n_nbr += 1

    return nbr_labels[:n_nbr], local_edges[:n_edges]


def roi_correction(
    label: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    prob: np.ndarray,
    *,
    unknown_class: int | None = 0,
    index_base: int = 0,
    on_empty_cand: Literal["restore", "argmax", "raise"] = "restore",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, CleanupInfo]:
    """Port of ``roi_correction(label, v, f, prob)`` (``roi_correction.m:1-100``, spec §6.8).

    Parameters
    ----------
    label:
        1D integer array of length ``N`` with initial per-vertex labels.
    vertices:
        Vertex coordinates ``(N, 3)``.
    faces:
        Triangle faces ``(M, 3)`` (0-based vertex indices).
    prob:
        Class score array ``(L, N)``.
    unknown_class:
        Class id of the "unknown" label (default ``0`` for 0-based indexing, or ``1``
        when ``index_base=1``).  Removed from ``cand`` when ``len(cand) > 1``.  Pass
        ``None`` to disable removal.
    index_base:
        ``0`` (default) when ``label`` values are ``0 .. L - 1``, or ``1`` when ``label``
        values are ``1 .. L`` as in MATLAB.
    on_empty_cand:
        Handling of the ``len(cand) == 0`` edge case (spec §6.8 / §14 item 4):
        ``"restore"`` (default), ``"argmax"``, or ``"raise"``.
    return_info:
        When ``True``, return ``(parc, CleanupInfo)``.
    """
    v = vertices.vertices if hasattr(vertices, "vertices") else vertices
    v_arr = np.asarray(v, dtype=np.float64)
    parc = np.ascontiguousarray(label, dtype=np.int32).ravel().copy()
    p_arr = np.asarray(prob, dtype=np.float64)
    n = parc.shape[0]
    if v_arr.shape != (n, 3):
        raise ValueError(f"vertices shape {v_arr.shape} does not match label length ({n}, 3)")
    if p_arr.ndim != 2 or p_arr.shape[1] != n:
        raise ValueError(f"prob shape {p_arr.shape} incompatible with N={n}")
    if index_base not in (0, 1):
        raise ValueError(f"index_base must be 0 or 1; got {index_base}")
    if index_base == 1 and unknown_class == 0:
        unknown_class = 1

    rc = faces_to_directed_edges(faces)
    head, to = _build_adj_csr(n, rc)

    bins = np.empty(n, dtype=np.int32)
    queue = np.empty(n, dtype=np.int32)
    comp_min_vertex = np.empty(n, dtype=np.int32)
    comp_size = np.empty(n, dtype=np.int32)
    comp_label = np.empty(n, dtype=np.int32)

    bsize_pre = 0
    bsize = int(np.unique(parc).size)

    n_iterations = 0
    size_ties: list[SizeTieEvent] = []
    n_empty_cand = 0
    n_local_gcuts = 0
    n_single_cand = 0

    while bsize_pre != bsize:
        n_iterations += 1
        parc_before = parc.copy()
        bsize_pre = bsize

        n_comp = _equal_label_components_numba(
            n, head, to, parc, bins, queue, comp_min_vertex, comp_size, comp_label
        )
        c_lbl = comp_label[:n_comp]
        c_sz = comp_size[:n_comp]
        c_minv = comp_min_vertex[:n_comp]

        # Sort by (label ascending, size descending, min_vertex ascending)
        order = np.lexsort((c_minv, -c_sz, c_lbl))
        sorted_lbl = c_lbl[order]
        _, first_pos = np.unique(sorted_lbl, return_index=True)
        selected_comps = order[first_pos]

        # Record any equal-size ties for the largest component of a label
        for pos in first_pos:
            if pos + 1 < n_comp and sorted_lbl[pos + 1] == sorted_lbl[pos]:
                c_keep = int(order[pos])
                c_tied = int(order[pos + 1])
                if c_sz[c_keep] == c_sz[c_tied]:
                    size_ties.append(
                        SizeTieEvent(
                            iteration=n_iterations,
                            label=int(c_lbl[c_keep]),
                            size=int(c_sz[c_keep]),
                            kept_min_vertex=int(c_minv[c_keep]),
                            tied_min_vertex=int(c_minv[c_tied]),
                        )
                    )

        if selected_comps.size == n_comp:
            break

        is_kept = np.zeros(n_comp, dtype=np.bool_)
        is_kept[selected_comps] = True
        keep_vertex_mask = is_kept[bins]
        parc[~keep_vertex_mask] = -1
        parc_refine = parc.copy()

        # Recompute connected components so adjacent -1 fragments merge
        n_comp2 = _equal_label_components_numba(
            n, head, to, parc, bins, queue, comp_min_vertex, comp_size, comp_label
        )
        bsize = int(n_comp2)

        # Group vertices by component id using argsort for O(N) slicing
        comp_order = np.argsort(bins, kind="mergesort")
        comp_starts = np.zeros(n_comp2 + 1, dtype=np.int32)
        np.cumsum(comp_size[:n_comp2], out=comp_starts[1:])

        for cid in range(n_comp2):
            if comp_label[cid] != -1:
                continue
            idx = comp_order[comp_starts[cid] : comp_starts[cid + 1]]
            nbr_labels, local_rc = _collect_component_neighbors_and_edges(
                idx, bins, cid, parc, head, to
            )
            cand = np.unique(nbr_labels)
            if cand.size > 1 and unknown_class is not None:
                cand = cand[cand != unknown_class]

            if cand.size == 0:
                n_empty_cand += 1
                if on_empty_cand == "raise":
                    raise ValueError(
                        f"roi_correction: empty candidate set for component {cid} "
                        f"(size {idx.size}, min_vertex {int(idx[0])})"
                    )
                elif on_empty_cand == "argmax":
                    best = np.argmax(p_arr[:, idx], axis=0).astype(np.int32)
                    parc_refine[idx] = best + index_base
                else:
                    parc_refine[idx] = parc_before[idx]
            elif cand.size == 1:
                n_single_cand += 1
                parc_refine[idx] = cand[0]
            else:
                n_local_gcuts += 1
                row_idx = cand - index_base
                new_label = gcut_from_edges(v_arr[idx], local_rc, p_arr[row_idx][:, idx])
                parc_refine[idx] = cand[new_label]

        parc = parc_refine

    if return_info:
        info = CleanupInfo(
            n_iterations=n_iterations,
            n_size_ties=len(size_ties),
            size_ties=tuple(size_ties),
            n_empty_cand=n_empty_cand,
            n_local_gcuts=n_local_gcuts,
            n_single_cand=n_single_cand,
        )
        return parc, info
    return parc
