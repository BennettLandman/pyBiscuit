# Port note: graph cut and ROI connected-component cleanup (plan rows B7.4, B7.5)

Code: `src/pybiscuit/parc/graphcut.py` (`gcut`, `gcut_from_edges`, `data_cost_from_scores`,
`upper_triangle_edges`, `faces_to_directed_edges`, `compute_energy`, `alpha_expansion_step`)
and `src/pybiscuit/parc/cleanup.py` (`roi_correction`, `equal_label_components`, `CleanupInfo`,
`SizeTieEvent`).
Tests: `tests/test_graphcut.py`.
Report: [graphcut-cleanup-2026-10-22](../reports/graphcut-cleanup-2026-10-22.md).

## Sources ported

From `~/dev/biscuit/BISCUIT-master/SurfPostproc/common/matlab/parc/` (specification in
[orchestrator-spec](../reference/orchestrator-spec.md) §6.7–6.8):

| File | Lines | Ported to | Notes |
|---|---|---|---|
| `gcut.m` | 1–16 | `graphcut.faces_to_directed_edges`, `graphcut.gcut` | Extracts sorted unique directed edges `(rows, cols)` from triangle `(M, 3)` or segment `(M, 2)` faces, then calls `gcut_from_edges`. |
| `gcut_.m` | 1–32 | `graphcut.data_cost_from_scores`, `graphcut.upper_triangle_edges`, `graphcut.gcut_from_edges` | Softmax + `-log` data cost, Potts smoothness weighted by Euclidean edge length on upper-triangle edges, `argmin` initial labelling, $\alpha$-expansion over $0 \dots L-1$. |
| `roi_correction.m` | 1–101 | `cleanup.equal_label_components`, `cleanup.roi_correction` | Connected components of the equal-label mesh subgraph, largest-component selection per label, `-1` fragment merging, 1-ring candidate set (class 0 excluded when `len(cand) > 1`), single-candidate assignment or local `gcut_from_edges`. |

## Licence & clean-room implementation (Q13)

`BISCUIT-master/SurfPostproc/common/matlab/gcut/` bundles Shai Bagon's `GCMex` wrapper around
Olga Veksler's `GCoptimization` and Yuri Boykov / Vladimir Kolmogorov's `maxflow`, which carry a
GPL v2 + research-only licence and a patent notice (`orchestrator-spec.md` §14 item 9,
`open-questions.md` Q13). Per owner decision on Q13 (2026-10-21):

* **No file inside `BISCUIT-master/SurfPostproc/common/matlab/gcut/` was read, inspected, or
  translated.**
* $\alpha$-expansion and float32 $s$-$t$ min-cut / max-flow are implemented cleanly from the
  published Boykov–Veksler–Zabih (2001) paper (*Fast Approximate Energy Minimization via Graph
  Cuts*, IEEE TPAMI 23(11):1222–1239), the Kolmogorov–Zabih (2004) binary submodular normal form
  (*What Energy Functions Can Be Minimized via Graph Cuts?*, IEEE TPAMI 26(2):147–159), and
  Dinic's blocking-flow algorithm compiled with Numba (`@njit(cache=True)`).

## Mathematical and numeric specification

### 1. Data cost, smoothness cost, and total energy (`B7.4`)

Given class score matrix `prob` of shape `(L, N)` and vertex coordinates `vertices` of shape
`(N, 3)`:

1. **Data cost (`gcut_.m:9-12`)**:
   Computed in `float64` in the exact MATLAB operation order and cast to `float32`:
   $$q_{l, j} = \frac{\exp(\text{prob}_{l, j} - \max_{l'} \text{prob}_{l', j})}{\sum_{l'} \exp(\text{prob}_{l', j} - \max_{l''} \text{prob}_{l'', j})}, \qquad D(l, j) = \operatorname{float32}(-\log q_{l, j}).$$
   When extreme synthetic score differences ($<-745$) underflow $\exp$ in `float64`, the exact
   log-sum-exp fallback $\min(-\tilde{p}_{l, j} + \log \sum_{l'} \exp(\tilde{p}_{l', j}), 10^{30})$
   keeps all capacities finite.
2. **Smoothness cost and upper-triangle edge weights (`gcut_.m:4,15-19`, spec §6.7)**:
   Potts metric $V(l_1, l_2) = [l_1 \neq l_2]$. From `directed_edges`, only the strict upper
   triangle $u < v$ is kept (deduplicated lexicographically), so each undirected mesh edge
   $\{u, v\}$ enters once with weight
   $$w_{uv} = \operatorname{float32}\!\left(\sqrt{\sum_{k=0}^{2} (v_{u, k} - v_{v, k})^2}\right).$$
3. **Initial labelling (`gcut_.m:21,25`)**:
   `label[j] = np.argmin(D[:, j])` (`int32`, 0-based $0 \dots L-1$, first minimum on ties).
4. **Total energy**:
   $$E(\text{label}) = \sum_{j=0}^{N-1} D(\text{label}[j], j) + \sum_{(u, v),\, u < v} w_{uv} \cdot [\text{label}[u] \neq \text{label}[v]].$$
   Individual terms are `float32`; the global sum is accumulated in `float64` so single-edge
   improvements on 163,842-vertex meshes are never lost to `float32` catastrophic cancellation.

### 2. Binary $s$-$t$ graph for $\alpha$-expansion and Dinic solver

In each cycle, $\alpha$ iterates over $0, 1, \dots, L - 1$ in fixed ascending order until a full
cycle over all $L$ labels does not strictly decrease $E(\text{label})$.

For a given $\alpha$ and current labelling $f$, vertices with $f_i = \alpha$ are fixed to $\alpha$.
For active vertices ($f_i \neq \alpha$), let $x_i \in \{0, 1\}$ denote the binary decision where
$x_i = 1$ ($i \in S$, source side) expands vertex $i$ to $\alpha$ ($f'_i = \alpha$) and $x_i = 0$
($i \in T$, sink side) keeps $f'_i = f_i$:

* **Unary capacities**: $c(s, i) \mathrel{+}= D(f_i, i)$ (cut when $i \in T$) and
  $c(i, t) \mathrel{+}= D(\alpha, i)$ (cut when $i \in S$).
* **Pairwise capacities for upper-triangle edge $(u, v)$ with weight $w = w_{uv}$**:
  * $f_u = \alpha, f_v = \alpha$: cost $0$.
  * $f_u = \alpha, f_v \neq \alpha$: pays $w$ when $x_v = 0 \implies c(s, v) \mathrel{+}= w$.
  * $f_u \neq \alpha, f_v = \alpha$: pays $w$ when $x_u = 0 \implies c(s, u) \mathrel{+}= w$.
  * $f_u = f_v \neq \alpha$: pays $w$ when $x_u \neq x_v \implies c(u, v) = w, c(v, u) = w$.
  * $f_u \neq f_v$ and neither is $\alpha$: pays $w$ whenever $(x_u, x_v) \neq (1, 1)$.
    Using the identity $[(x_u, x_v) \neq (1, 1)] = [x_u = 0] + [x_u = 1 \wedge x_v = 0]$
    (which equals the minimum over $a_{uv} \in \{0, 1\}$ of the BVZ auxiliary-node construction
    $\min(w, w(1-x_u) + w(1-x_v))$ for all $(x_u, x_v) \in \{0, 1\}^2$), this adds $w$ to
    $c(s, u)$ and sets $c(u, v) = w, c(v, u) = 0$ without adding auxiliary vertices to the graph.
* **2-edge $s \to i \to t$ pre-saturation**:
  Subtracting $\min(c(s, i), c(i, t))$ leaves `source_cap[i] = max(0, c(s, i) - c(i, t))` and
  `sink_cap[i] = max(0, c(i, t) - c(s, i))`. Because `source_cap[i] > 0` holds only at vertices
  where switching to $\alpha$ lowers unary + boundary cost relative to keeping $f_i$, the BFS
  level graph and iterative path-stack DFS only visit the small active candidate region around
  parcel $\alpha$.

### 3. Connected-component cleanup (`roi_correction`, `B7.5`)

`roi_correction(label, vertices, faces, prob)` ports `roi_correction.m:1-100`:

1. **Indexing convention**:
   * Default (`index_base=0, unknown_class=0`): `label` contains 0-based class indices
     $0 \dots L-1$ indexing rows of `prob`, with class `0` as "unknown" (dropped from `cand` when
     `len(cand) > 1`).
   * MATLAB 1-based mode (`index_base=1, unknown_class=1`): `label` contains 1-based class indices
     $1 \dots L$, with class `1` as "unknown".
2. **Equal-size largest component tie-breaker (spec §6.8)**:
   Connected components of the equal-label subgraph are discovered via BFS scanning vertices in
   ascending order $0 \dots N-1$, so component IDs are ordered by their minimum vertex index.
   Components are sorted by `(label ascending, -size, min_vertex_index ascending)` and the first
   component per label is kept. When two or more components of the same label tie for the largest
   size, the component with the **lowest minimum vertex index** wins and a `SizeTieEvent` is
   recorded in `CleanupInfo.size_ties` (`CleanupInfo.n_size_ties`).
3. **Empty-candidate edge case (`len(cand) == 0`, spec §6.8 / §14 item 4)**:
   Occurs only when a non-largest fragment is a disconnected mesh island with no edges to any kept
   component. Controlled by `on_empty_cand`:
   * `"restore"` (default): restores the component's pre-iteration labels so no `-1` labels remain
     and the outer loop terminates cleanly.
   * `"argmax"`: assigns per-vertex `argmax` of `prob`.
   * `"raise"`: raises `ValueError`.
   Every occurrence increments `CleanupInfo.n_empty_cand`.
