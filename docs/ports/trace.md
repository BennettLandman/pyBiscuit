# Port note: TRACE curve extraction (plan rows B8.1-B8.4, with B2.6 geodesics)

Python: `src/pybiscuit/trace/{points,curves,extract}.py`, `src/pybiscuit/mesh/{geodesic,slicer}.py`, CLI
`pybiscuit trace`. Tests: `tests/test_{trace,geodesic,slicer}.py` (goldens in `tests/data/golden/{trace,geodesic,slicer}`).
Oracle: upstream `CurveExtraction` `ded932d` (v1.1.2, the version string the image reports) + MeshLib `dcef807`, built **without
fused multiply-add** by `scripts/oracles/build_curve_extraction_nofma.sh` into
`~/dev/biscuit/oracles/curve_extraction-nofma/` (CurveExtraction + `harness_geodesic` + `harness_slicer`; goldens by
`scripts/oracles/make_golden_trace.py`). The earlier default-flag build (`oracles/curve_extraction`, arm64 FMA) is kept:
on a folded sphere it already gives 2999 vs 3000 gyral points and 185 vs 188 curves, i.e. TRACE's discrete decisions are
FMA-sensitive and the no-FMA build is the only meaningful oracle for the x86-64 image.

## Function map

| Python | C++ (`reference/repos/CurveExtraction/src`) |
|---|---|
| `points.point_fields` | `SulcalPoint::SulcalPoint`/`setCurvature` (SulcalPoint.cpp 31-124), `GyralPoint` same lines |
| `points.seed_mask` | `setSeed` (57-75) |
| `points._local_maxima` and helpers | `localMaxima`, `contour`, `findCandidate`, `maxDeviation`, `longestAxis`, `tanAxis`, `arbitraryAxis` (146-662) |
| `mesh.slicer` | MeshLib `Slicer.cpp`, `AABB.cpp` (tree), `Geom.cpp::{intersection,proj2plane}` |
| `curves.Curves.run/_grouping/_delineation` | `SulcalCurve::{run,grouping,delineation,deleteNearestPoints,detectEndPoints,detectNearestPoints}` |
| `curves._dijkstra`, `_curve_target`, `_curve_dijkstra` | `curve_dijkstra` (888-1077) |
| `curves._extend`, `_extend_target` | `extendCurves_dijkstra` (598-738), `testLoop` |
| `curves._join/_separate/_prune/_delete_*` | `joinCurves(elem,elem,inner)`, `separateBranch`, `pruneCurves`, `deleteCurves`, `deleteJunction` |
| `extract.extract_curves` | `Extraction.cpp::extraction` |
| `extract.write_*` | `saveSulcalCurves`/`saveGyralCurves`, `saveVTK`, `saveGeodesicPath` (cart/bary), `--interm` |

`GyralPoint`/`GyralCurve` are the sulcal code with different constants (tables in the module docstrings of `points.py`
and `curves.py`); one implementation with a `gyral` flag serves both. Not ported because unreachable from `extraction()`:
`curve`, `extendCurves`, `joinCurves(float)`, `grouping` defaults, `concave/convex`, `projAxis`, `saveValleyPoint`.

## Flag table (all ported)

`-i/--inputSurface`, `--inputPoints <prefix>` (reads `<prefix>.spoint/.gpoint`), `--sulcus`, `--gyrus`, `-o/--outputCurve`, `--cart`,
`--bary`, `--noVtk`, `--interm`, `--nojunction`, `--gyralSeed` (0.03), `--sulcalSeed` (-0.05), `--epRad` (4), `--nhDist` (4),
`--lsTh` (2.5), `--prune` (5), `--simplification`, `-s/--smoothIter` (0), `--smoothTensorIter` (3), `--nThreads` (ignored).

**BISCUIT's call (resolved from the MATLAB source, orchestrator spec §8, `curve_extract_pipeline.m:39,47`):**
`CurveExtraction --sulcus|--gyrus -i <h>.gray.vtk|<h>.white.vtk -o <prefix> --nThreads n --epRad r --nhDist r --lsTh d --bary --noVtk`
with `r = trace_r` (both `--epRad` and `--nhDist`) and `d = trace_delta` (`--lsTh`), each through MATLAB `num2str`; every other option stays at
its default (seeds -0.05 / 0.03, `--prune 5`, `-s 0`, `--smoothTensorIter 3`). The configured `trace_r = 4`, `trace_delta = 2.5` equal the
tool defaults, so the port's defaults already reproduce BISCUIT; non-default config values reach the tool. Sulci are extracted on the pial
(`gray`) surface, gyri on the white surface. Driver: `pybiscuit.trace.pipeline` (`curve_argv`, `curve_extract_pipeline`; brief 20), tested by
`tests/test_lgi_chain.py::test_biscuit_trace_argv_is_the_spec_call`. (This replaces the earlier "hypothesis, unverified" paragraph.) The default seeds (-0.05 / 0.03) are tuned for real cortex in mm; the synthetic
fixtures need `--sulcalSeed -0.01 --gyralSeed 0.01`.

## Precision

float32 everywhere MeshLib uses `MathVector`/`Vertex` (positions, normals, curvature fields, point-detection geometry, curve
orientation, Dijkstra weights, the distance matrix); `Geodesic` fast marching and path distances are double (see `geodesic.md`).
No fused multiply-add; association order follows the C++. `exp/acos/atan/sin/cos` use the platform libm (float overloads), so
last-ulp differences vs another libm are possible.

## Quirks reproduced / decisions

* **Pairwise distances** `m_dist` is a dense n x n float matrix upstream (several GB for a cortex); here a sparse symmetric float32
  CSR whose missing entries are `FLT_MAX`. Loops over "all points j with dist < threshold" iterate the stored row in ascending
  order, the same order as the dense loop.
* **Endpoint order** is `std::sort` of `curveElem` by position (x, then y, then z).
* **`GyralCurve::refineCurves` reads an uninitialised `current`**; the clang oracle behaves as a single pass, a compiler producing an
  arbitrary value gives >= 2 passes (the second pass reverses some curves). **Resolved by the image (brief 22):** on the three open
  examples the image's `.gcurve` and `.gcurve.bary` are byte-identical to the port's with `uninit_current="continue"` (>= 2 passes)
  and differ in the direction of some curves with `"exit"`. Default is now `"continue"`; the oracle goldens use `--uninit-current exit`.
* **`std::sort` tie order in the slicer.** Segments are sorted by start-edge id; when a plane passes (to float rounding) through
  a mesh vertex or contains an edge, several segments share an id and the order of equals is implementation-defined (libc++ vs
  libstdc++). A stable sort is used. Observed on the symmetric synthetic fixtures only (planes aligned with an icosphere axis):
  1 of 10,242 point flags (`fold_l8m4`, vertex 871) and 1 of 60 slice planes (group order); curves unaffected. Real cortex has not
  shown it (see the report). Emulating libc++/libstdc++ `std::sort` exactly would remove it.
* `--simplification` together with `--sulcus --gyrus` dereferences a NULL `GyralCurve` upstream (segfault); raised as `RuntimeError`.
  `SimplifyCurves` indexes the distance matrix by list position instead of vertex id (bug kept).
* Open surfaces: the slicer links an unmatched `nextID` to the next larger id (clamped), upstream may read past the array.
* Threads: single-threaded; output does not depend on `--nThreads` upstream either.

## Validation (provisional: upstream-macos-arm64 no-FMA oracle)

Byte-identical `.scurve/.gcurve/.bary/.cart/.vtk/.spoint/.gpoint` on `fold_l5m3` and `fold_l8m4` (radius 50 mm, 10,242 vertices) except
the single tie flag above; see `docs/reports/geodesics-trace-2026-10-20.md` for the template-surface run and runtime.
