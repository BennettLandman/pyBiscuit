# Port note: MeshLib geodesics (plan row B2.6)

Python: `src/pybiscuit/mesh/geodesic.py` (`GeodesicSolver`, `GeodesicPath`, `sparse_pairwise_distances`,
`dijkstra_distances`). Tests: `tests/test_geodesic.py`. Oracle harness:
`scripts/oracles/harness_geodesic.cpp` (built with `scripts/oracles/build_curve_extraction_nofma.sh`).

## Source (MeshLib `dcef807`) and what is ported

Only the call graph reached from CurveExtraction (`SulcalCurve::{detectNearestPoints,saveGeodesicPath}`):

| Python | C++ |
|---|---|
| `GeodesicSolver.propagate`, `perform_front_propagation*` | `Geodesic/Geodesic.cpp` (wrapper, callbacks) |
| `_march`, `_vertex_distance`, `_sethian`, `_unfold_triangle` | `gw_geodesic/GW_GeodesicMesh.{cpp,inl}` (`PerformFastMarching`, `PerformFastMarchingOneStep`, `ComputeVertexDistance`, `ComputeUpdate_SethianMethod`, `UnfoldTriangle`) |
| `_build_connectivity`, `_rings` | `gw_core/GW_Mesh.cpp::BuildConnectivity`, `GW_Face.inl`, `GW_Vertex.cpp`, `GW_{Vertex,Face}Iterator.cpp` (incl. the open-fan boundary walk) |
| heap functions | `std::make_heap/push_heap/pop_heap` with `GW_GeodesicVertex::CompareVertex` |
| `GeodesicPath` | `Geodesic/GeodesicPath.cpp` |

Not ported (not reached): `GeodesicA`, `GW_GeodesicMeshA`, `GW_Parameterization`, Voronoi/atlas code (these three do not compile upstream; B2.8, Q20), the triangular interpolations (ported separately, [interpolation.md](interpolation.md)),
`ComputeUpdate_MatrixMethod` (`USE_SETHIAN` is defined), the heuristic/`L` callbacks.

## Semantics reproduced

* `GW_Float` is double, vertex positions are float32 widened to double. `GW_VectorStatic::operator/=`
  multiplies by the reciprocal (`x *= 1/f`) - reproduced in every normalisation.
* Weight 1, `bUseUnfolding_ = true`, `GW_INFINITE = 1e9`, `GW_EPSILON = 1e-9`.
* Stop rule: after a vertex is made dead and its neighbours updated, stop if `dist > dmax` or the vertex is an
  end point (`perform_front_propagation(start, end)` therefore leaves a partially marched field; the
  path code reads it). Insertion callback: `nbr_iter <= 1.2 * nverts`.
* Vertex iteration order matters (neighbours are updated in ring order and see each other's new values):
  the ring starts at the vertex's first incident face (lowest index) and follows `GetFaceNeighbor`.
* Heap: `std::make_heap` is called again after every "alive" update (as upstream). Exact ties in
  distance are resolved by the standard library, so `heap="libcxx"` (macOS oracle) and `heap="libstdcxx"`
  (default; the Linux image) are both implemented. On all tested meshes (icosphere, folded, perturbed, cut,
  UV sphere, `lh.white`) the two variants and the oracle agree exactly.
* `GeodesicPath`: distances double, edge vectors float32, Gram determinant evaluated in float32 then widened.
  Closed manifolds only; an edge missing from the edge table (open boundary) clamps the `lower_bound` index
  (upstream reads out of range).
* `sparse_pairwise_distances` = `detectNearestPoints`: symmetric float32 distances among a vertex subset
  (value from the lower-index propagation), stored sparsely; absent entries mean upstream's `FLT_MAX`.

## Validation (provisional: upstream-macos-arm64, `-ffp-contract=off`)

Distances, states, source vertices and `GeodesicPath` points/barycentrics vs the oracle harness
(`D`/`E`/`P` queries) on ico4, folded sphere, perturbed ico4, cut sphere (open), UV sphere and
`reg/mb101/lh.white` (163,842 vertices): **0 differences** (distances bit-identical; paths equal to the
`%.9g` print precision). Speed on `lh.white`: setup 0.16 s, 0.3 ms per `dmax = 4` propagation, 1.1 s for a full march.
A unit-sphere check against great-circle distance and the edge-graph Dijkstra overestimate are in
`tests/test_geodesic.py` (diagnostic).
