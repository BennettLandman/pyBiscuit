# Geodesics + TRACE report (brief 04) - 2026-10-20

Branch `b04-geodesics-trace`. Rows: B2.6, B8.1-B8.4. All evidence is **provisional**: upstream C++ built on macOS arm64 with
`-ffp-contract=off` (`oracles/curve_extraction-nofma`, MeshLib `dcef807`, CurveExtraction `ded932d` = v1.1.2, the string the image
reports). Image goldens (B1.4) override it. Port notes: [geodesic.md](../ports/geodesic.md), [trace.md](../ports/trace.md).

## Headline results

1. **`reg/mb101/lh.white.vtk` (163,842 vertices), default parameters, `--sulcus --gyrus --cart --bary --interm`: all ten output files
   (`.spoint .gpoint .scurve .gcurve .{s,g}curve.vtk .cart .bary`) are byte-identical to the no-FMA oracle** (3324 sulcal / 7207 gyral
   points, 64 / 176 curves, 30 / 215 junctions). Run live in `tests/test_trace.py::test_template_lh_white_live_vs_oracle`
   (needs `PYBISCUIT_SLOW=1`; no template golden is committed).
2. Two synthetic folded-sphere fixtures (`fold_l5m3`, `fold_l8m4`; radius 50 mm, 10,242 vertices) have committed oracle goldens
   (`tests/data/golden/trace/`): curve, `.bary` files byte-identical; point flags identical except **one** of 20,484 (see tie note).
   Acceptance metrics (brief): Jaccard 1.0 (>= 0.99), polyline Hausdorff 0 (<= 1e-3 x radius = 0.05 mm; template 0 vs <= 0.5 mm).
3. Fast marching (`GeodesicSolver`) and `GeodesicPath` are bit-identical to MeshLib (distance, state, source arrays and path points) on six
   meshes incl. `lh.white`; both `std::*_heap` tie behaviours (libc++, libstdc++) are implemented and agree on all tested data.
4. Runtime on `lh.white`: see the table below.

## Runtime on `lh.white` (single thread, M-series Mac)

| Run | Wall time |
|---|---|
| Upstream `CurveExtraction` (no-FMA build, serial `omp.h` stub), `--sulcus --gyrus --cart --bary --interm` | 350 s |
| `pybiscuit trace`, same flags (Numba kernels cached, includes I/O and path files) | 58 s |

Ratio 0.17x of the oracle (target <= 3x). Both runs overlapped with another job on the machine, so treat the figures as indicative.
First call in a fresh environment adds the Numba compile time (cache=True afterwards). Fast marching alone: 0.3 ms per `dmax = 4` propagation.

## Brief-04 acceptance, item by item

| Criterion | Result |
|---|---|
| Geodesic distances vs oracle harness rel <= 1e-5 | 0 difference (bit-identical), `tests/test_geodesic.py::test_golden_vs_meshlib` |
| Unit sphere vs great circle <= 0.5 % mean (diagnostic) | **0.92 % mean** on icosphere(5) (vertices > 0.2 rad from the source); the edge-graph Dijkstra is worse. This is the first-order scheme of the upstream code (identical to the oracle), not a port deviation |
| Curves Jaccard >= 0.99, Hausdorff | 1.0 / 0 on template and fixtures |
| Runtime <= 3x oracle | see table |

## Findings and open questions

* **FMA matters.** On the same folded sphere the default arm64 build and the no-FMA build differ (2999 vs 3000 gyral points, 185 vs
  188 curves). Only the no-FMA build approximates the x86-64 image; the earlier `oracles/curve_extraction` (default flags) should
  not be used as a reference.
* **Uninitialised `current` in `GyralCurve::refineCurves`** (undefined behaviour). The clang oracle behaves as a single extension pass;
  a second pass reverses 57 of 188 curves on a fixture (same point sets). Port default follows the oracle
  (`Curves(uninit_current="exit")`). **Open question for the Linux image run:** which behaviour does the g++-built binary show? If the
  curve direction matters downstream (`.bary` order), compare one subject's `.gcurve` ordering. Added to docs/open-questions.md? - not
  edited (shared file); owner please add: *Q-TRACE-1 gyral refine passes (image compiler)*.
* **`std::sort` ties in the slicer** (documented in trace.md): only on planes aligned with the icosphere axes of the synthetic
  fixtures (1 point flag of 20,484; 1 of 60 planes reorders its contour groups). Not seen on `lh.white`. Exact emulation of libc++ /
  libstdc++ `std::sort` was not implemented.
* **BISCUIT parameter mapping** (`trace_r = 4`, `trace_delta = 2.5`): equal to the defaults of `--epRad/--nhDist` and `--lsTh`;
  hypothesis in trace.md, to be confirmed by the B1.2 command trace.
* `--simplification --sulcus --gyrus` segfaults upstream (NULL `GyralCurve`); raises here.
* Upstream `m_dist` is a dense n x n float matrix (several GB on cortex); the port is sparse with identical results.
* Scope: brief 04 asked to stop if new MeshLib pieces exceeded ~500 lines. `Slicer` + tree (~600 C++ lines) turned out to be needed
  by `SulcalPoint`/`GyralPoint` (not covered by brief 02's ports); it was ported rather than stopping, because the point selection
  is otherwise impossible. Flagging that decision here.

## Requests for shared files (not edited by this branch)

* `docs/agent-briefs/README.md`: mark brief 04 done after merge.
* `docs/open-questions.md`: add Q-TRACE-1 above.
* `docs/ports/oracles.md`: add the `curve_extraction-nofma` build (script `scripts/oracles/build_curve_extraction_nofma.sh`,
  contains `harness_geodesic`, `harness_slicer`).
* `src/pybiscuit/mesh/__init__.py`: export `geodesic`, `slicer`.

## Reproduce

```
scripts/oracles/build_curve_extraction_nofma.sh        # sandbox off: writes ~/dev/biscuit/oracles
python scripts/oracles/make_golden_trace.py             # regenerates tests/data/golden/{trace,geodesic,slicer}
pytest tests/test_geodesic.py tests/test_slicer.py tests/test_trace.py
PYBISCUIT_SLOW=1 pytest tests/test_trace.py -k template
```
