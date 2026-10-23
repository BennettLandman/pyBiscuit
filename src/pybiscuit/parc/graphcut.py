"""Licence-clean alpha-expansion graph cut for cortical parcellation (plan row B7.4).

Port of ``gcut.m`` and ``gcut_.m`` (spec §6.7) using an independent Numba-compiled
Boykov–Veksler–Zabih (2001) alpha-expansion and Dinic's blocking-flow min-cut solver
(see ``docs/open-questions.md`` Q13; no GCMex / GCoptimization / maxflow code read or
translated).

Mathematical specification (``gcut_.m:1-32``, spec §6.7):

* Input class scores ``prob`` of shape ``(L, N)`` for ``L`` classes and ``N`` vertices.
* Data cost ``D(l, j) = -log(softmax_l(prob[:, j]))`` computed in float64 and cast to
  ``float32``.
* Smoothness cost: Potts metric ``V(l1, l2) = [l1 != l2]`` weighted by Euclidean edge
  length ``w_ij = ||v_i - v_j||`` (``float32``) on the upper triangle ``i < j`` of the
  mesh adjacency matrix (one undirected edge per pair ``{i, j}``).
* Initial labelling: ``label[j] = argmin_l D(l, j)`` (0-based ``0 .. L-1``, first
  minimum on ties).
* Total energy:
  ``E(label) = sum_j D(label[j], j) + sum_{(i, j), i < j} w_ij * [label[i] != label[j]]``.
* Alpha-expansion iterates ``alpha = 0, 1, ..., L - 1`` in fixed order, repeating full
  cycles until a complete cycle over ``0 .. L - 1`` does not strictly decrease the
  energy.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit

__all__ = [
    "GraphCutInfo",
    "alpha_expansion_step",
    "compute_energy",
    "data_cost_from_scores",
    "faces_to_directed_edges",
    "gcut",
    "gcut_from_edges",
    "upper_triangle_edges",
]


class GraphCutInfo(NamedTuple):
    """Diagnostics returned when ``return_info=True``."""

    initial_energy: float
    final_energy: float
    n_cycles: int
    cycle_energies: tuple[float, ...]


def data_cost_from_scores(prob: np.ndarray) -> np.ndarray:
    """Compute float32 data cost ``D(l, j) = -log(softmax_l(prob[:, j]))`` (``gcut_.m:9-12``).

    Follows the MATLAB softmax sequence ``q = exp(p - max(p)); q = q ./ sum(q); D = -log(q)``
    in float64 before casting to float32.  If extreme score differences underflow ``exp``
    in float64 (``p - max(p) < -745``), the exact log-sum-exp identity capped at ``1e30``
    is used so capacities remain finite.
    """
    p64 = np.asarray(prob, dtype=np.float64)
    if p64.ndim != 2 or p64.shape[0] < 1:
        raise ValueError(f"prob must be a 2D array (L, N) with L >= 1; got {p64.shape}")
    p_shift = p64 - np.max(p64, axis=0, keepdims=True)
    q = np.exp(p_shift)
    q_sum = np.sum(q, axis=0, keepdims=True)
    with np.errstate(divide="ignore"):
        d64 = -np.log(q / q_sum)
    if not np.all(np.isfinite(d64)):
        fallback = np.minimum(-p_shift + np.log(q_sum), 1e30)
        d64 = np.where(np.isfinite(d64), d64, fallback)
    return np.ascontiguousarray(d64, dtype=np.float32)


def faces_to_directed_edges(faces: np.ndarray) -> np.ndarray:
    """Extract sorted unique directed edges ``(E, 2)`` from faces (``gcut.m:6-13``)."""
    f = np.asarray(faces, dtype=np.int32)
    if f.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if f.ndim != 2 or f.shape[1] not in (2, 3):
        raise ValueError(f"faces must have shape (M, 3) or (M, 2); got {f.shape}")
    if f.shape[1] == 2:
        rows = np.concatenate([f[:, 0], f[:, 1]])
        cols = np.concatenate([f[:, 1], f[:, 0]])
    else:
        rows = np.concatenate([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]])
        cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]])
    return np.unique(np.column_stack([rows, cols]), axis=0).astype(np.int32, copy=False)


def upper_triangle_edges(
    vertices: np.ndarray, directed_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract upper-triangle edges ``u < v`` and float32 Euclidean lengths ``||v_u - v_v||``."""
    v64 = np.asarray(vertices, dtype=np.float64)
    if v64.ndim != 2 or v64.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3); got {v64.shape}")
    rc = np.asarray(directed_edges, dtype=np.int32)
    if rc.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    if rc.ndim != 2 or rc.shape[1] != 2:
        raise ValueError(f"directed_edges must have shape (E, 2); got {rc.shape}")
    mask = rc[:, 0] < rc[:, 1]
    if not np.any(mask):
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    ut = np.unique(rc[mask], axis=0)
    edges_u = np.ascontiguousarray(ut[:, 0], dtype=np.int32)
    edges_v = np.ascontiguousarray(ut[:, 1], dtype=np.int32)
    diff = v64[edges_u] - v64[edges_v]
    weights = np.ascontiguousarray(np.sqrt(np.sum(diff * diff, axis=1)), dtype=np.float32)
    return edges_u, edges_v, weights


@njit(cache=True)
def _compute_energy_numba(
    label: np.ndarray,
    data_cost: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
) -> float:
    n = label.shape[0]
    m_e = edges_u.shape[0]
    total = 0.0
    for i in range(n):
        total += float(data_cost[label[i], i])
    for e in range(m_e):
        if label[edges_u[e]] != label[edges_v[e]]:
            total += float(weights[e])
    return total


def compute_energy(
    label: np.ndarray,
    data_cost: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute exact Potts + unary energy from float32 terms accumulated in float64."""
    lbl = np.ascontiguousarray(label, dtype=np.int32)
    dc = np.ascontiguousarray(data_cost, dtype=np.float32)
    eu = np.ascontiguousarray(edges_u, dtype=np.int32)
    ev = np.ascontiguousarray(edges_v, dtype=np.int32)
    w = np.ascontiguousarray(weights, dtype=np.float32)
    return float(_compute_energy_numba(lbl, dc, eu, ev, w))


@njit(cache=True)
def _build_csr(
    n: int, edges_u: np.ndarray, edges_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m_e = edges_u.shape[0]
    deg = np.zeros(n, dtype=np.int32)
    for e in range(m_e):
        deg[edges_u[e]] += 1
        deg[edges_v[e]] += 1
    head = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        head[i + 1] = head[i] + deg[i]
    cur = head[:-1].copy()
    to = np.empty(2 * m_e, dtype=np.int32)
    rev = np.empty(2 * m_e, dtype=np.int32)
    h_uv = np.empty(m_e, dtype=np.int32)
    h_vu = np.empty(m_e, dtype=np.int32)
    for e in range(m_e):
        u = edges_u[e]
        v = edges_v[e]
        hu = cur[u]
        cur[u] = hu + 1
        hv = cur[v]
        cur[v] = hv + 1
        to[hu] = v
        to[hv] = u
        rev[hu] = hv
        rev[hv] = hu
        h_uv[e] = hu
        h_vu[e] = hv
    return head, to, rev, h_uv, h_vu


@njit(cache=True)
def _dinic_alpha_move(
    n: int,
    m_e: int,
    alpha: int,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
    head: np.ndarray,
    to: np.ndarray,
    rev: np.ndarray,
    h_uv: np.ndarray,
    h_vu: np.ndarray,
    data_cost: np.ndarray,
    label: np.ndarray,
    cap_s: np.ndarray,
    cap_t: np.ndarray,
    source_cap: np.ndarray,
    sink_cap: np.ndarray,
    cap: np.ndarray,
    level: np.ndarray,
    ptr: np.ndarray,
    queue: np.ndarray,
    stack_node: np.ndarray,
    stack_edge: np.ndarray,
    cand_label: np.ndarray,
) -> int:
    """Construct BVZ 2001 binary s-t graph for ``alpha`` and solve min-cut via Dinic.

    Convention:
    * Source ``s`` (``x_i = 1``, ``i in S``): active vertex ``i`` expands to ``alpha``.
    * Sink ``t`` (``x_i = 0``, ``i in T``): active vertex ``i`` keeps ``label[i]``.

    Under this orientation, ``source_cap[i] = max(0, c(s, i) - c(i, t))`` is positive only
    at vertices where switching to ``alpha`` reduces unary + boundary cost relative to keeping
    ``label[i]``, so BFS/DFS visits only the active candidate region around ``alpha``.
    """
    zero = np.float32(0.0)
    for i in range(n):
        li = label[i]
        if li != alpha:
            cap_s[i] = data_cost[li, i]
            cap_t[i] = data_cost[alpha, i]
        else:
            cap_s[i] = zero
            cap_t[i] = zero

    for e in range(m_e):
        u = edges_u[e]
        v = edges_v[e]
        w = weights[e]
        lu = label[u]
        lv = label[v]
        h1 = h_uv[e]
        h2 = h_vu[e]
        if lu == alpha and lv == alpha:
            cap[h1] = zero
            cap[h2] = zero
        elif lu == alpha and lv != alpha:
            cap_s[v] += w
            cap[h1] = zero
            cap[h2] = zero
        elif lu != alpha and lv == alpha:
            cap_s[u] += w
            cap[h1] = zero
            cap[h2] = zero
        elif lu == lv:
            cap[h1] = w
            cap[h2] = w
        else:
            # Submodular reduction of V(f'_u, f'_v) when lu != lv and neither is alpha:
            # w * [x_u == 0 or x_v == 0] = w * [x_u == 0] + w * [x_u == 1 and x_v == 0]
            cap_s[u] += w
            cap[h1] = w
            cap[h2] = zero

    any_source = False
    for i in range(n):
        if label[i] != alpha:
            diff = cap_s[i] - cap_t[i]
            if diff > zero:
                source_cap[i] = diff
                sink_cap[i] = zero
                any_source = True
            elif diff < zero:
                source_cap[i] = zero
                sink_cap[i] = -diff
            else:
                source_cap[i] = zero
                sink_cap[i] = zero
        else:
            source_cap[i] = zero
            sink_cap[i] = zero

    if not any_source:
        return 0

    q_tail = 0
    while True:
        for idx in range(q_tail):
            level[queue[idx]] = 0
        q_head = 0
        q_tail = 0
        sink_level = 0
        for i in range(n):
            if source_cap[i] > zero:
                level[i] = 1
                queue[q_tail] = i
                q_tail += 1
        while q_head < q_tail:
            u = queue[q_head]
            q_head += 1
            lu = level[u]
            if sink_cap[u] > zero and sink_level == 0:
                sink_level = lu + 1
            if sink_level != 0 and lu >= sink_level - 1:
                continue
            for h in range(head[u], head[u + 1]):
                if cap[h] > zero:
                    v = to[h]
                    if level[v] == 0:
                        level[v] = lu + 1
                        queue[q_tail] = v
                        q_tail += 1
        if sink_level == 0:
            break

        target_u_level = sink_level - 1
        for idx in range(q_tail):
            u = queue[idx]
            ptr[u] = head[u]

        for idx in range(q_tail):
            s_node = queue[idx]
            if level[s_node] != 1 or source_cap[s_node] <= zero:
                continue
            stack_node[0] = s_node
            depth = 0
            while depth >= 0:
                u = stack_node[depth]
                if level[u] == target_u_level and sink_cap[u] > zero:
                    bot = source_cap[s_node]
                    retreat_depth = -1
                    if sink_cap[u] <= bot:
                        bot = sink_cap[u]
                        retreat_depth = depth
                    for d in range(depth):
                        c = cap[stack_edge[d]]
                        if c <= bot:
                            bot = c
                            if retreat_depth == -1 or d < retreat_depth:
                                retreat_depth = d
                    if source_cap[s_node] == bot:
                        retreat_depth = -1
                    source_cap[s_node] -= bot
                    sink_cap[u] -= bot
                    for d in range(depth):
                        h = stack_edge[d]
                        cap[h] -= bot
                        cap[rev[h]] += bot
                    if retreat_depth == -1:
                        break
                    depth = retreat_depth
                    continue

                if level[u] >= target_u_level:
                    level[u] = 0
                    depth -= 1
                    if depth >= 0:
                        ptr[stack_node[depth]] += 1
                    continue

                h = ptr[u]
                end_h = head[u + 1]
                pushed_step = False
                while h < end_h:
                    if cap[h] > zero:
                        v = to[h]
                        if level[v] == level[u] + 1:
                            ptr[u] = h
                            stack_edge[depth] = h
                            depth += 1
                            stack_node[depth] = v
                            pushed_step = True
                            break
                    h += 1
                if not pushed_step:
                    ptr[u] = end_h
                    level[u] = 0
                    depth -= 1
                    if depth >= 0:
                        ptr[stack_node[depth]] += 1

    if q_tail > 0:
        for i in range(n):
            cand_label[i] = label[i]
        for idx in range(q_tail):
            u = queue[idx]
            if label[u] != alpha:
                cand_label[u] = alpha
            level[u] = 0
    return q_tail


@njit(cache=True)
def _alpha_expansion_loop(
    n: int,
    m_e: int,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
    head: np.ndarray,
    to: np.ndarray,
    rev: np.ndarray,
    h_uv: np.ndarray,
    h_vu: np.ndarray,
    data_cost: np.ndarray,
    label: np.ndarray,
) -> tuple[float, int, np.ndarray]:
    num_labels = data_cost.shape[0]
    cap_s = np.zeros(n, dtype=np.float32)
    cap_t = np.zeros(n, dtype=np.float32)
    source_cap = np.zeros(n, dtype=np.float32)
    sink_cap = np.zeros(n, dtype=np.float32)
    cap = np.zeros(2 * m_e, dtype=np.float32)
    level = np.zeros(n, dtype=np.int32)
    ptr = np.zeros(n, dtype=np.int32)
    queue = np.zeros(n, dtype=np.int32)
    stack_node = np.zeros(n + 1, dtype=np.int32)
    stack_edge = np.zeros(n + 1, dtype=np.int32)
    cand_label = np.empty(n, dtype=np.int32)

    cur_energy = _compute_energy_numba(label, data_cost, edges_u, edges_v, weights)
    cycle_energies = np.empty(128, dtype=np.float64)
    n_cycles = 0

    while True:
        cycle_start_energy = cur_energy
        for alpha in range(num_labels):
            q_tail = _dinic_alpha_move(
                n,
                m_e,
                alpha,
                edges_u,
                edges_v,
                weights,
                head,
                to,
                rev,
                h_uv,
                h_vu,
                data_cost,
                label,
                cap_s,
                cap_t,
                source_cap,
                sink_cap,
                cap,
                level,
                ptr,
                queue,
                stack_node,
                stack_edge,
                cand_label,
            )
            if q_tail > 0:
                new_energy = _compute_energy_numba(cand_label, data_cost, edges_u, edges_v, weights)
                if new_energy < cur_energy:
                    cur_energy = new_energy
                    for i in range(n):
                        label[i] = cand_label[i]

        if n_cycles < cycle_energies.shape[0]:
            cycle_energies[n_cycles] = cur_energy
        n_cycles += 1
        if not (cur_energy < cycle_start_energy):
            break

    kept = min(n_cycles, cycle_energies.shape[0])
    return cur_energy, n_cycles, cycle_energies[:kept].copy()


def alpha_expansion_step(
    label: np.ndarray,
    alpha: int,
    data_cost: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run a single alpha-expansion move from ``label`` toward ``alpha`` (0-based).

    Returns ``(new_label, new_energy)`` where ``new_label`` achieves the minimum energy
    over all ``2^(#active)`` binary expansions ``f'_i in {label[i], alpha}``.
    """
    lbl = np.ascontiguousarray(label, dtype=np.int32).copy()
    dc = np.ascontiguousarray(data_cost, dtype=np.float32)
    eu = np.ascontiguousarray(edges_u, dtype=np.int32)
    ev = np.ascontiguousarray(edges_v, dtype=np.int32)
    w = np.ascontiguousarray(weights, dtype=np.float32)
    n = lbl.shape[0]
    m_e = eu.shape[0]
    cur_energy = _compute_energy_numba(lbl, dc, eu, ev, w)
    if m_e == 0:
        for i in range(n):
            if dc[alpha, i] < dc[lbl[i], i]:
                lbl[i] = alpha
        return lbl, float(_compute_energy_numba(lbl, dc, eu, ev, w))

    head, to, rev, h_uv, h_vu = _build_csr(n, eu, ev)
    cap_s = np.zeros(n, dtype=np.float32)
    cap_t = np.zeros(n, dtype=np.float32)
    source_cap = np.zeros(n, dtype=np.float32)
    sink_cap = np.zeros(n, dtype=np.float32)
    cap = np.zeros(2 * m_e, dtype=np.float32)
    level = np.zeros(n, dtype=np.int32)
    ptr = np.zeros(n, dtype=np.int32)
    queue = np.zeros(n, dtype=np.int32)
    stack_node = np.zeros(n + 1, dtype=np.int32)
    stack_edge = np.zeros(n + 1, dtype=np.int32)
    cand_label = np.empty(n, dtype=np.int32)
    q_tail = _dinic_alpha_move(
        n,
        m_e,
        int(alpha),
        eu,
        ev,
        w,
        head,
        to,
        rev,
        h_uv,
        h_vu,
        dc,
        lbl,
        cap_s,
        cap_t,
        source_cap,
        sink_cap,
        cap,
        level,
        ptr,
        queue,
        stack_node,
        stack_edge,
        cand_label,
    )
    if q_tail > 0:
        new_energy = _compute_energy_numba(cand_label, dc, eu, ev, w)
        if new_energy < cur_energy:
            return cand_label, float(new_energy)
    return lbl, float(cur_energy)


def gcut_from_edges(
    vertices: np.ndarray,
    directed_edges: np.ndarray,
    prob: np.ndarray,
    *,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, GraphCutInfo]:
    """Port of ``gcut_(v, rc, prob)`` (``gcut_.m:1-32``, spec §6.7).

    Parameters
    ----------
    vertices:
        Mesh vertex coordinates ``(N, 3)``.
    directed_edges:
        Directed edges ``(E, 2)`` (0-based vertex indices). Only the upper triangle
        ``u < v`` enters the smoothness energy (one undirected edge per pair).
    prob:
        Class score matrix ``(L, N)``.
    return_info:
        When ``True``, also return :class:`GraphCutInfo` with initial/final energies
        and cycle count.

    Returns
    -------
    label:
        1D ``int32`` array of 0-based class labels ``0 .. L - 1`` of length ``N``.
    """
    v = vertices.vertices if hasattr(vertices, "vertices") else vertices
    v_arr = np.asarray(v, dtype=np.float64)
    n = v_arr.shape[0]
    data_cost = data_cost_from_scores(prob)
    if data_cost.shape[1] != n:
        raise ValueError(f"prob vertex count ({data_cost.shape[1]}) does not match vertices ({n})")

    label = np.ascontiguousarray(np.argmin(data_cost, axis=0), dtype=np.int32)
    edges_u, edges_v, weights = upper_triangle_edges(v_arr, directed_edges)
    m_e = edges_u.shape[0]
    init_energy = float(_compute_energy_numba(label, data_cost, edges_u, edges_v, weights))

    if m_e == 0 or data_cost.shape[0] <= 1:
        info = GraphCutInfo(
            initial_energy=init_energy,
            final_energy=init_energy,
            n_cycles=1,
            cycle_energies=(init_energy,),
        )
        return (label, info) if return_info else label

    head, to, rev, h_uv, h_vu = _build_csr(n, edges_u, edges_v)
    final_energy, n_cycles, cyc_e = _alpha_expansion_loop(
        n,
        m_e,
        edges_u,
        edges_v,
        weights,
        head,
        to,
        rev,
        h_uv,
        h_vu,
        data_cost,
        label,
    )
    if return_info:
        info = GraphCutInfo(
            initial_energy=init_energy,
            final_energy=float(final_energy),
            n_cycles=int(n_cycles),
            cycle_energies=tuple(float(x) for x in cyc_e),
        )
        return label, info
    return label


def gcut(
    vertices: np.ndarray,
    faces: np.ndarray,
    prob: np.ndarray,
    *,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, GraphCutInfo]:
    """Port of ``gcut(v, f, prob)`` (``gcut.m:1-16``, spec §6.7).

    Extracts sorted unique directed edges from ``faces`` (0-based) and calls
    :func:`gcut_from_edges`.
    """
    v = vertices.vertices if hasattr(vertices, "vertices") else vertices
    rc = faces_to_directed_edges(faces)
    return gcut_from_edges(v, rc, prob, return_info=return_info)
