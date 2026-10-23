# Graph cut and ROI connected-component cleanup — 2026-10-22

**Plan rows:** `B7.4` (`src/pybiscuit/parc/graphcut.py`), `B7.5` (`src/pybiscuit/parc/cleanup.py`)  
**Port note:** [docs/ports/graphcut.md](../ports/graphcut.md)  
**Tests:** `tests/test_graphcut.py`

## 1. Summary

Implemented a licence-clean port of BISCUIT's cortical parcellation refinement stages:

1. **`pybiscuit.parc.graphcut` (`B7.4`, `gcut.m`, `gcut_.m`, spec §6.7)**:
   * Computes float32 unary data costs $D(l, j) = -\log \operatorname{softmax}_l(\text{prob}_{:, j})$
     and float32 Euclidean edge lengths $w_{uv} = \|v_u - v_v\|_2$ on the strict upper triangle
     ($u < v$) of the mesh adjacency matrix.
   * Initializes `label = argmin_l D(l, j)` (0-based, first minimum on ties) and runs
     Boykov–Veksler–Zabih (2001) $\alpha$-expansion in fixed label order $0 \dots L - 1$ until a
     complete cycle over $0 \dots L - 1$ yields no strict decrease in total energy.
   * Solves each binary $s$-$t$ min-cut via a Numba-compiled (`@njit(cache=True)`) Dinic
     blocking-flow solver with 2-edge $s \to i \to t$ pre-saturation and source-oriented active
     candidate traversal.
   * Strictly adheres to Q13: no C++/MEX files in `BISCUIT-master/SurfPostproc/common/matlab/gcut/`
     (`GCMex` / `GCoptimization` / Boykov–Kolmogorov `maxflow`, GPL v2 + research-only) were read,
     inspected, or translated.

2. **`pybiscuit.parc.cleanup` (`B7.5`, `roi_correction.m`, spec §6.8)**:
   * Finds connected components of the equal-label mesh subgraph via Numba BFS scanning vertices
     in ascending order $0 \dots N - 1$.
   * Keeps the largest component per unique label, breaking equal-size ties deterministically by
     the **lowest minimum vertex index** and recording each tie in `CleanupInfo.size_ties`
     (`SizeTieEvent`).
   * Marks non-largest components as `-1`, recomputes connected components so adjacent `-1`
     fragments merge, and reassigns each `-1` component from its 1-ring neighbour candidate set
     `cand` (dropping `unknown_class=0` when `len(cand) > 1`, or `1` when `index_base=1`):
     * `len(cand) == 1`: direct assignment (`CleanupInfo.n_single_cand`).
     * `len(cand) > 1`: local `gcut_from_edges` on the induced subgraph and `prob[cand, idx]`
       (`CleanupInfo.n_local_gcuts`).
     * `len(cand) == 0` (spec §6.8 / §14 item 4, where MATLAB crashes on disconnected mesh
       islands): handled via `on_empty_cand="restore"` (default), `"argmax"`, or `"raise"`, with
       every event counted in `CleanupInfo.n_empty_cand`.

## 2. Verification & Acceptance Results

### 2.1 Unit and Brute-Force Optimality Tests (`tests/test_graphcut.py`)

All 9 test suites in `tests/test_graphcut.py` pass (`0.68 s`):

* **2-node and 3-node BVZ equivalence (`test_bvz_2node_and_3node_reduction_equivalence`)**:
  Across 200 random 2-node and 3-node subgraphs and all $2^N$ cuts, the submodular reduction
  without auxiliary nodes matches the BVZ auxiliary-node construction
  $\min_{a_{uv} \in \{0,1\}} E(x_u, x_v, a_{uv})$ and the exact Potts energy to $< 10^{-6}$, and
  every single-$\alpha$ move matches the brute-force global minimum.
* **Brute-force single-move optimality on 4–6 vertex graphs
  (`test_single_alpha_expansion_matches_brute_force_on_tiny_graphs`)**:
  Across 240 random connected graphs ($N \in \{4, 5, 6\}$, $L \in \{3, 4\}$, 4 $\alpha$ moves
  each = 960 moves), every single-$\alpha$ expansion move achieves the exact global minimum over
  all $2^N$ binary expansions.
* **Monotone energy & local optimality (`test_full_gcut_energy_monotone_and_locally_optimal_on_tiny_meshes`)**:
  Across 50 random 4-class problems on an octahedron mesh, cycle energies decrease monotonically,
  final energy is never above initial `argmin` energy, and no single-$\alpha$ move from the
  converged labelling can further lower the energy.
* **Cleanup invariants & edge cases (`test_roi_correction_*`)**:
  Verified 1-component-per-label invariant, single-neighbour absorption, multi-neighbour local
  graph cut, merging of adjacent `-1` fragments, exclusion of `unknown_class=0` when
  `len(cand) > 1` (and preservation when `cand == [0]`), 0-based and 1-based indexing modes,
  deterministic lowest-minimum-vertex tie-breaking (`SizeTieEvent`), and all three
  `on_empty_cand` modes (`"restore"`, `"argmax"`, `"raise"`).

### 2.2 End-to-End Parcellation Check vs `biscuit_FC_v2.2.sif` (`open-examples/0050952`)

Replayed full left-hemisphere parcellation ($N = 137{,}067$ vertices, $274{,}130$ faces) on ABIDE
subject `0050952` (`lh`) against the reference container outputs `roi/lh.parc.braincolor.txt` and
`roi/lh.parc.dkt31.txt`:

| Protocol | Classes ($L$) | Folds | Vertex Agreement vs Image | Differing Vertices | Min ROI Dice | Mean ROI Dice | Median ROI Dice | Components / Labels Post-Cleanup | Total `gcut` Time | Total `roi_correction` Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `braincolor` | 50 | 1 | **99.8059 %** | 266 / 137,067 | **0.9893** | 0.9976 | 0.9983 | 50 / 50 | 1.01 s | 0.65 s |
| `dkt31` | 32 | 5 (+ fusion) | **99.8570 %** | 196 / 137,067 | **0.9917** | 0.9982 | 0.9989 | 32 / 32 | 5.16 s (6 calls) | 3.98 s (6 calls) |

Both protocols exceed the E-class acceptance thresholds ($\ge 99.5\%$ vertex agreement and
per-ROI Dice $\ge 0.98$ across every parcel). The $< 0.2\%$ differing vertices lie on 1-vertex-wide
parcel boundaries where upstream feature extraction (`H` curvature / `HSD -d 0` rigid alignment)
and max-flow tie-breaking differ at tie level (Q13).
