"""Tests for alpha-expansion graph cut (B7.4) and ROI connected-component cleanup (B7.5)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from pybiscuit.parc.cleanup import (
    CleanupInfo,
    equal_label_components,
    roi_correction,
)
from pybiscuit.parc.graphcut import (
    alpha_expansion_step,
    compute_energy,
    data_cost_from_scores,
    faces_to_directed_edges,
    gcut,
    gcut_from_edges,
    upper_triangle_edges,
)
from pybiscuit.sphere.icosphere import icosphere


def _brute_force_alpha_expansion(
    label: np.ndarray,
    alpha: int,
    data_cost: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, list[np.ndarray]]:
    """Exhaustively evaluate all 2^N binary expansions f'_i in {label[i], alpha}."""
    n = label.shape[0]
    best_e = float("inf")
    best_labels: list[np.ndarray] = []
    for mask in range(1 << n):
        cand = label.copy()
        for i in range(n):
            if (mask >> i) & 1:
                cand[i] = alpha
        e = compute_energy(cand, data_cost, edges_u, edges_v, weights)
        if e < best_e - 1e-5:
            best_e = e
            best_labels = [cand]
        elif abs(e - best_e) <= 1e-5:
            best_labels.append(cand)
    return best_e, best_labels


def _bvz_auxiliary_node_cut_energy(
    x: np.ndarray,
    label: np.ndarray,
    alpha: int,
    data_cost: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Evaluate exact BVZ 2001 s-t cut cost with auxiliary nodes a_ij minimized over {0, 1}."""
    n = label.shape[0]
    cost = 0.0
    for i in range(n):
        if label[i] == alpha:
            cost += float(data_cost[alpha, i])
        else:
            cost += float(data_cost[alpha, i]) if x[i] == 1 else float(data_cost[label[i], i])
    for e in range(edges_u.shape[0]):
        u, v = int(edges_u[e]), int(edges_v[e])
        w = float(weights[e])
        lu, lv = int(label[u]), int(label[v])
        if lu == alpha and lv == alpha:
            pass
        elif lu == alpha and lv != alpha:
            if x[v] == 0:
                cost += w
        elif lu != alpha and lv == alpha:
            if x[u] == 0:
                cost += w
        elif lu == lv:
            if x[u] != x[v]:
                cost += w
        else:
            # Auxiliary node a_uv in {0 (keep), 1 (alpha)}:
            # If a_uv = 0: cuts (a_uv, t) of cap w -> cost w
            # If a_uv = 1: cuts (u, a_uv) if x[u]==0 and (v, a_uv) if x[v]==0
            cost_a0 = w
            cost_a1 = w * (1 - x[u]) + w * (1 - x[v])
            cost += min(cost_a0, cost_a1)
    return cost


def test_bvz_2node_and_3node_reduction_equivalence():
    """Verify on 2-node and 3-node subgraphs that our s-t reduction equals BVZ and brute force."""
    rng = np.random.default_rng(20261022)
    for n in (2, 3):
        for _ in range(100):
            num_labels = 4
            v = rng.normal(size=(n, 3))
            prob = rng.normal(size=(num_labels, n))
            dc = data_cost_from_scores(prob)
            rc = np.array([[i, j] for i in range(n) for j in range(n) if i != j], dtype=np.int32)
            eu, ev, w = upper_triangle_edges(v, rc)
            lbl = rng.integers(0, num_labels, size=n, dtype=np.int32)
            alpha = int(rng.integers(0, num_labels))

            # Check every binary assignment x in {0, 1}^n
            for mask in range(1 << n):
                x = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int32)
                if any(lbl[i] == alpha and x[i] == 0 for i in range(n)):
                    continue
                cand = np.where(x == 1, alpha, lbl).astype(np.int32)
                e_true = compute_energy(cand, dc, eu, ev, w)
                e_bvz = _bvz_auxiliary_node_cut_energy(x, lbl, alpha, dc, eu, ev, w)
                assert np.isclose(e_true, e_bvz, atol=1e-6)

            # Check single alpha-expansion step matches brute-force minimum
            step_lbl, step_e = alpha_expansion_step(lbl, alpha, dc, eu, ev, w)
            bf_e, _ = _brute_force_alpha_expansion(lbl, alpha, dc, eu, ev, w)
            assert np.isclose(step_e, bf_e, atol=1e-5)
            assert np.isclose(compute_energy(step_lbl, dc, eu, ev, w), step_e, atol=1e-5)


def test_single_alpha_expansion_matches_brute_force_on_tiny_graphs():
    """Every single-alpha expansion move on 4-6 vertex random graphs is a global min-cut."""
    rng = np.random.default_rng(42)
    for n in (4, 5, 6):
        for num_labels in (3, 4):
            for _ in range(40):
                v = rng.uniform(-10.0, 10.0, size=(n, 3))
                # Random connected graph with upper-triangle edges
                edges = [[i, i + 1] for i in range(n - 1)]
                for i in range(n):
                    for j in range(i + 2, n):
                        if rng.random() < 0.5:
                            edges.append([i, j])
                rc = np.array(edges + [[j, i] for i, j in edges], dtype=np.int32)
                eu, ev, w = upper_triangle_edges(v, rc)
                prob = rng.normal(scale=2.0, size=(num_labels, n))
                dc = data_cost_from_scores(prob)
                lbl = rng.integers(0, num_labels, size=n, dtype=np.int32)

                for alpha in range(num_labels):
                    new_lbl, new_e = alpha_expansion_step(lbl, alpha, dc, eu, ev, w)
                    bf_e, _ = _brute_force_alpha_expansion(lbl, alpha, dc, eu, ev, w)
                    assert new_e <= compute_energy(lbl, dc, eu, ev, w) + 1e-6
                    assert np.isclose(new_e, bf_e, atol=1e-5)
                    # Labels only stay lbl[i] or become alpha
                    assert np.all((new_lbl == lbl) | (new_lbl == alpha))


def test_full_gcut_energy_monotone_and_locally_optimal_on_tiny_meshes():
    """Full alpha-expansion decreases energy monotonically and reaches a local optimum."""
    rng = np.random.default_rng(7)
    # Octahedron mesh (6 vertices, 8 faces)
    v_oct = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    f_oct = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    rc = faces_to_directed_edges(f_oct)
    eu, ev, w = upper_triangle_edges(v_oct, rc)

    for _ in range(50):
        prob = rng.normal(scale=2.5, size=(4, 6))
        dc = data_cost_from_scores(prob)
        init_lbl = np.argmin(dc, axis=0).astype(np.int32)
        init_e = compute_energy(init_lbl, dc, eu, ev, w)

        lbl, info = gcut(v_oct, f_oct, prob, return_info=True)
        assert lbl.shape == (6,)
        assert lbl.dtype == np.int32
        assert np.isclose(info.initial_energy, init_e, atol=1e-5)
        assert info.final_energy <= info.initial_energy + 1e-6
        assert info.n_cycles >= 1

        # Cycle energies are monotonically non-increasing
        prev = info.initial_energy
        for ce in info.cycle_energies:
            assert ce <= prev + 1e-6
            prev = ce

        # No single-alpha expansion can strictly decrease the final energy
        for alpha in range(4):
            _, step_e = alpha_expansion_step(lbl, alpha, dc, eu, ev, w)
            assert step_e >= info.final_energy - 1e-5


def test_gcut_determinism_and_edge_variants():
    """Repeated calls and directed/upper-triangle edge variants produce bit-identical results."""
    rng = np.random.default_rng(99)
    mesh = icosphere(2)
    prob = rng.normal(size=(6, mesh.n_vertices))

    lbl1, info1 = gcut(mesh.vertices, mesh.faces, prob, return_info=True)
    lbl2, info2 = gcut(mesh, mesh.faces, prob, return_info=True)
    assert np.array_equal(lbl1, lbl2)
    assert info1 == info2

    # Upper-triangle only vs both directions vs duplicated edges
    rc_both = faces_to_directed_edges(mesh.faces)
    rc_upper = rc_both[rc_both[:, 0] < rc_both[:, 1]]
    rc_dup = np.vstack([rc_both, rc_both])
    assert np.array_equal(lbl1, gcut_from_edges(mesh.vertices, rc_upper, prob))
    assert np.array_equal(lbl1, gcut_from_edges(mesh.vertices, rc_dup, prob))

    # Segment (M, 2) faces in gcut
    assert np.array_equal(lbl1, gcut(mesh.vertices, rc_upper, prob))

    # Single class (L = 1) and empty edges edge cases
    lbl_single = gcut(mesh.vertices, mesh.faces, prob[:1])
    assert np.all(lbl_single == 0)
    lbl_no_edges = gcut_from_edges(mesh.vertices, np.empty((0, 2), dtype=np.int32), prob)
    assert np.array_equal(lbl_no_edges, np.argmin(data_cost_from_scores(prob), axis=0))

    # Extreme score differences (underflow guard in softmax -log)
    prob_extreme = prob * 1000.0
    lbl_ext = gcut(mesh.vertices, mesh.faces, prob_extreme)
    assert lbl_ext.shape == (mesh.n_vertices,)
    assert np.all((lbl_ext >= 0) & (lbl_ext < 6))


def _make_grid_mesh(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular triangulated grid of nx * ny vertices."""
    xs, ys = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
    v = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    faces = []
    for r in range(ny - 1):
        for c in range(nx - 1):
            i0 = r * nx + c
            i1 = i0 + 1
            i2 = i0 + nx
            i3 = i2 + 1
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])
    return v, np.array(faces, dtype=np.int32)


def test_roi_correction_single_neighbor_and_unknown_exclusion():
    """Single-neighbor islands are absorbed; class 0 ('unknown') is dropped when len(cand) > 1."""
    v, f = _make_grid_mesh(6, 6)
    n = v.shape[0]
    prob = np.zeros((4, n), dtype=np.float64)

    # Background is label 2; vertex 7 (interior of left half) is a 1-vertex island of label 1
    # whose main component is on the right half (vertices with x >= 4)
    label = np.full(n, 2, dtype=np.int32)
    right_cols = np.flatnonzero(v[:, 0] >= 4)
    label[right_cols] = 1
    label[7] = 1  # single vertex surrounded solely by label 2

    out, info = roi_correction(label, v, f, prob, return_info=True)
    assert out[7] == 2
    assert info.n_single_cand == 1
    assert info.n_local_gcuts == 0
    n_comp, _ = equal_label_components(out, f, n)
    assert n_comp == len(np.unique(out))

    # Now make vertex 7 border both label 2 and label 0 ("unknown"):
    # Since len(cand) == 2 ({0, 2}) and 0 is unknown_class, 0 is dropped -> cand becomes {2}.
    label2 = label.copy()
    label2[0] = 0
    label2[1] = 0
    label2[6] = 0  # vertex 6 is adjacent to vertex 7
    out2, info2 = roi_correction(label2, v, f, prob, return_info=True)
    assert out2[7] == 2
    assert info2.n_single_cand == 1
    assert info2.n_local_gcuts == 0

    # If an island is surrounded ONLY by class 0 ("unknown"), len(cand) == 1 so class 0 is kept
    label3 = np.zeros(n, dtype=np.int32)
    label3[right_cols] = 1
    label3[7] = 1  # surrounded only by 0
    out3 = roi_correction(label3, v, f, prob)
    assert out3[7] == 0


def test_roi_correction_multi_neighbor_invokes_local_gcut_and_merges_adjacent_fragments():
    """Adjacent non-largest fragments merge into one -1 component and run local gcut."""
    v, f = _make_grid_mesh(8, 6)
    n = v.shape[0]
    # Left region (x <= 2): label 1 (18 vertices)
    # Right region (x >= 3): label 2 (30 vertices)
    # Place a 2-vertex fragment at (x=2, y=2) [v=18] with label 3 and (x=3, y=2) [v=19] with label 1
    # while main component of label 3 is at top-right corner (3 vertices: 45, 46, 47)
    label = np.where(v[:, 0] <= 2, 1, 2).astype(np.int32)
    label[[45, 46, 47]] = 3
    label[18] = 3  # non-largest fragment of label 3 (size 1 < 3)
    label[19] = 1  # disconnected from x<=2 if we set its left neighbors to 2:
    label[[11, 18, 26]] = 3  # size 3 fragment? Wait: make main component of 3 have size 4:
    label[[39, 45, 46, 47]] = 3  # size 4 at top-right
    # Now vertices {11, 18, 26} have label 3 (size 3 < 4 -> marked -1) and vertex 19 has label 1
    # surrounded by {11, 18, 26} on the left and label 2 on the right (size 1 < 15 -> marked -1).
    # When marked -1, {11, 18, 26} and {19} merge into a single 4-vertex -1 component
    # bordering both label 1 and label 2!
    prob = np.zeros((4, n), dtype=np.float64)
    prob[2, [11, 18, 19, 26]] = 5.0  # strong preference for label 2 in local gcut

    out, info = roi_correction(label, v, f, prob, return_info=True)
    assert info.n_local_gcuts >= 1
    assert np.all(out[[11, 18, 19, 26]] == 2)
    n_comp, _ = equal_label_components(out, f, n)
    assert n_comp == len(np.unique(out))


def test_roi_correction_equal_size_tie_breaker_and_1based_mode():
    """Equal-size largest components break ties by lowest min-vertex index (SizeTieEvent)."""
    v, f = _make_grid_mesh(6, 4)
    n = v.shape[0]
    label = np.full(n, 1, dtype=np.int32)
    # Create two disconnected components of label 2, each of size 2:
    # Component A at vertices {0, 1} (min_vertex = 0)
    # Component B at vertices {22, 23} (min_vertex = 22)
    label[[0, 1]] = 2
    label[[22, 23]] = 2
    prob = np.zeros((3, n), dtype=np.float64)

    out, info = roi_correction(label, v, f, prob, return_info=True)
    assert info.n_size_ties == 1
    tie = info.size_ties[0]
    assert tie.label == 2
    assert tie.size == 2
    assert tie.kept_min_vertex == 0
    assert tie.tied_min_vertex == 22
    # Component A ({0, 1}) is kept as label 2; Component B ({22, 23}) is relabelled to 1
    assert np.all(out[[0, 1]] == 2)
    assert np.all(out[[22, 23]] == 1)

    # Verify 1-based mode (labels in 1..3, index_base=1) produces the exact shifted result
    out_1b, info_1b = roi_correction(
        label + 1, v, f, prob, index_base=1, return_info=True
    )
    assert np.array_equal(out_1b, out + 1)
    assert info_1b.n_size_ties == 1


def test_roi_correction_empty_candidate_edge_case():
    """Disconnected mesh island with no remaining neighbors is handled gracefully (spec §6.8)."""
    v1, f1 = _make_grid_mesh(4, 4)
    v2, f2 = _make_grid_mesh(2, 2)
    v2 = v2 + np.array([100.0, 0.0, 0.0])
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    n = len(v)

    # Both islands have label 1; the 4x4 island (size 16) is kept, while the 2x2 island (size 4)
    # becomes -1 globally and has NO 1-ring neighbors outside itself (cand is empty).
    label = np.ones(n, dtype=np.int32)
    prob = np.zeros((3, n), dtype=np.float64)
    prob[2, len(v1) :] = 10.0

    # Default on_empty_cand="restore": restores pre-iteration label (1), no -1 left, no crash
    out_restore, info_restore = roi_correction(label, v, f, prob, return_info=True)
    assert info_restore.n_empty_cand >= 1
    assert np.all(out_restore == 1)

    # on_empty_cand="argmax": assigns argmax of prob (2) to the isolated island
    out_argmax, info_argmax = roi_correction(
        label, v, f, prob, on_empty_cand="argmax", return_info=True
    )
    assert info_argmax.n_empty_cand == 1
    assert np.all(out_argmax[: len(v1)] == 1)
    assert np.all(out_argmax[len(v1) :] == 2)

    # on_empty_cand="raise": raises ValueError
    with pytest.raises(ValueError, match="empty candidate set"):
        roi_correction(label, v, f, prob, on_empty_cand="raise")


def test_performance_and_invariants_on_icosphere():
    """Benchmark gcut + roi_correction on a 2,562-vertex icosphere with 32 classes."""
    rng = np.random.default_rng(1234)
    mesh = icosphere(4)  # 2,562 vertices, 5,120 faces
    v = mesh.vertices * 100.0  # radius 100 mm cortical scale
    f = mesh.faces
    n = v.shape[0]
    num_classes = 32

    # Spatially coherent Voronoi-like class scores + noise so fragments exist pre-cleanup
    centers = v[rng.choice(n, size=num_classes, replace=False)]
    dists = np.linalg.norm(v[None, :, :] - centers[:, None, :], axis=2)
    prob = -0.15 * dists + rng.normal(scale=1.8, size=(num_classes, n))

    # Warmup Numba JIT
    small_mesh = icosphere(1)
    gcut(small_mesh.vertices, small_mesh.faces, rng.normal(size=(4, small_mesh.n_vertices)))
    roi_correction(
        np.zeros(small_mesh.n_vertices, dtype=np.int32),
        small_mesh.vertices,
        small_mesh.faces,
        rng.normal(size=(4, small_mesh.n_vertices)),
    )

    t0 = time.perf_counter()
    pred_gcut, g_info = gcut(v, f, prob, return_info=True)
    t_gcut = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred_clean, c_info = roi_correction(pred_gcut, v, f, prob, return_info=True)
    t_clean = time.perf_counter() - t1

    assert g_info.final_energy < g_info.initial_energy
    assert isinstance(c_info, CleanupInfo)
    # Post-cleanup invariant: every present label has exactly 1 connected component
    n_comp, _ = equal_label_components(pred_clean, f, n)
    assert n_comp == len(np.unique(pred_clean))
    assert t_gcut < 2.0
    assert t_clean < 2.0
