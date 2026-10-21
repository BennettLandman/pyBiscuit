# Port note: Gyrification / local gyrification index (plan row B9.4)

Code: `src/pybiscuit/quant/lgi.py` (driver, tensor field, readers/writers), `quant/_fastmarch.py`
(MeshLib fast marching), `quant/lgi_cli.py` (`pybiscuit lgi`).
Tests: `tests/test_lgi.py`. Oracle: `scripts/oracles/build_gyrification_nofma.sh`
(`oracles/gyrification-nofma/`), goldens `tests/data/golden/gyrification/` made by
`scripts/oracles/make_golden_gyrification.py`. Scale check: `scripts/validation/lgi_scale_check.py`.

Sources: LocalGyrificationIndex `b9fa760` (`src/Gyrification.cpp` all, `wrapper/main.cpp`,
`wrapper/PARSE_ARGS.h`, `script/lgi`), MeshLib `dcef807` (`src/Geodesic/{Geodesic,GeodesicA}.cpp`,
`include/Geodesic/gw/gw_geodesic/GW_GeodesicMesh{,A}.inl`, `gw_core/{GW_Mesh,GW_Face,GW_Vertex,
GW_VertexIterator}`, `src/Mesh.cpp`). The image's Gyrification is v1.2.2, the same as `b9fa760`.

## What is ported

| C++ | Python |
|---|---|
| `open` / `loadScurve[Bary]`, `loadGcurve[Bary]` | `read_curves`, `curve_vertex_ids` (only the union of curve vertices is used downstream) |
| `setupGDist` (`Geodesic`, Sethian update, sources = all curve vertices) | `_fastmarch.propagate_iso` |
| `loadGMap` / `saveGMap` | `read_geodesic_map` / `write_geodesic_map` (`%f %d`) |
| `precomputeArea`, area adjustment | `gyrification` (hull area capped at surface area; `--refHullArea` 77100, else `--refCortexArea`) |
| `setupTensorEqualArea`, `smoothingTensor(10)`, `smoothingScalar(10)` | `lgi.setup_tensor_equal_area` (float32 `Vector` arithmetic as in `Mesh.cpp`) |
| `GeodesicA::setTensor`, `perform_front_propagation` (Qian update, `area_max` stop) | `_fastmarch.set_tensor`, `kernel_range` |
| `kernelArea`, `computeGyrification`, `--intvArea` | `kernel_range` (index-ordered sums) and `gyrification` |
| `saveGI`, `saveKMap1/2` | `write_lgi_maps` (`<out>.{lgi,k1,k2}.map.<int(delta)>.txt`, `%f`) |
| `--inputMap`, `--corr`, `--nThreads` | `input_map=`, `corr=`, `n_threads=` (threads share nothing; results do not depend on it) |

`-r/--radius` is parsed and ignored upstream; it is accepted and ignored here. The `lgi` shell script
(OuterHull -> klaplace -> CurveExtraction -> Gyrification) is not reproduced as a driver: klaplace is
out of scope (licence, Q7) and TRACE is brief 04. Gyrification takes `--outer`, the vertex-wise
hull from klaplace's `-surfaceCorrespondence`; `OuterHull` is B9.2 (`outer-hull.md`).

## Behaviours kept on purpose

* **Closed manifold meshes only.** The GW vertex iterator is reproduced for closed fans; an open
  mesh raises. (BISCUIT feeds closed hemispheres.)
* **Fast marching details**: ring order from the lowest-index face, obtuse-angle unfolding (<= 50
  faces), "usable" neighbours need the *same front* as the dead vertex, `GW_INFINITE = 1e9`,
  `a /= x` is `a *= 1/x` and skipped when `x == 0`, `make_heap` after every improvement of an alive vertex.
* **`std::heap` tie order.** Exact distance ties are broken by heap layout, so the libc++ algorithms
  (oracle, macOS) are default and the libstdc++ ones (Linux image) are selectable with
  `heap="libstdcxx"` / `--heap`. The libstdc++ variant could not be checked against a real Linux
  binary (Q1); it is validated only as a correct heap (sorted pops) and by agreeing with libc++ on
  tie-free inputs. On inputs with exact ties (e.g. the bare icosphere, mirror symmetric curves) the
  nearest-source id of equidistant vertices, and through the tensor field the LGI of nearby
  vertices, depend on the heap flavour; real cortical surfaces have no exact ties.
* **Kernel area bookkeeping**: the cumulative propagated area `A[i]` is recorded in the stop
  callback, which is skipped for the last popped vertex when the heap is empty. Upstream then
  reads a stale value from the previous front of that thread (uninitialised on the first); we keep a
  zero-initialised per-worker array. This only matters when a kernel covers the whole mesh.
* **Tie in `eig3symmetric`** (diagonal tensor with equal diagonal entries): upstream leaves the
  eigenvectors uninitialised; we use the identity, i.e. the z axis, as the principal axis.
* `--corr` requires the same vertex count in surface and outer (upstream reads out of bounds
  otherwise).
* Output text uses `%f` (6 decimals) like the original; the Python API returns full precision.

## libm: why `acosf` is pinned (one switch)

`Mesh::updateNormal` weights face normals by `acos` of float32 dot products, so the C++ tools see the
host's `acosf`. Apple's `acosf` and the fdlibm algorithm differ by 1 ulp on ~8 % of inputs. The
anisotropic kernels are chaotic in that input: perturbing the float32 normals by +-1 ulp (seeded
three-valued noise) changes the LGI of 70-80 of 2,562 vertices by 1e-5..3e-2 (kernel membership at the
cumulative-area cutoff flips when near-tied vertices swap pop order), independent of whether the
perturbation is 1 or 4 ulp. The first CI run on Linux x86-64 showed this signature (18-95 of 2,562
vertices off, max 6e-2) against goldens made with Apple's `acosf`. Perturbing areas, distances or
tensors by 1e-13 relative changes nothing, so double-precision libm differences are harmless.

Design: `libm=` (CLI `--libm`) selects the `acosf` used for the normal weights, defined in
`quant/_libm.py` (`DEFAULT_LIBM`, one line to change):

* `"fdlibm"` (default): the Sun fdlibm single-precision algorithm, reproduced in Numba
  (`_libm.acosf`) and as a C link-time shim for the oracle (`scripts/oracles/patches/acosf-fdlibm/acosf.c`).
  Bit-identical on every platform, so port and goldens agree everywhere (CI on Linux x86-64 passes all
  golden comparisons). `tests/test_lgi.py::test_acosf_matches_fdlibm_table` pins the Python function to
  the C shim (9,006 inputs). No test asserts anything about the host libm.
* `"host"`: whatever `acosf` the host's libm returns.

**This choice is provisional.** The acceptance target is the Linux x86-64 image (Ubuntu 20.04, glibc
2.31 as far as known). Whether its `acosf` is the fdlibm algorithm (glibc's flt-32 `e_acosf.c` is
derived from it) was not verified; the host libm of a current CI runner is not a proof either way
(numpy's own float32 `arccos` is a different SIMD implementation and disagrees with fdlibm on ~1/3 of
inputs). Image goldens (brief 10 / B1.4) decide: if they match `"host"` on a glibc machine and not
`"fdlibm"`, switch `DEFAULT_LIBM` and regenerate the oracle goldens with a matching shim.
`mesh.geometry.vertex_normals` (brief 02) still calls the host libm; `lgi.vertex_normals_f32` is a copy
of that kernel with the selectable `acosf`.

## Validation

Seven oracle cases on a seeded fold fixture (icosphere(4), r = 50 mm, harmonic l = 8, m = 3, a = 0.15,
radial noise to remove exact ties; hull = same vertices on a sphere of 0.8 R so that LGI > 1): default
flags, kernel ratio 1, `--intvArea 100` (4 maps), `--speed 0.5 -m 200`, plain-text curves, `--inputMap`,
`--corr`. Curves are written by the oracle CurveExtraction. Per-vertex LGI, `kArea1/2`, geodesic
distance: rel <= 1e-5 (the files hold 6 decimals); nearest-source ids identical. Analytic diagnostics
(`tests/test_lgi.py`): LGI = 1 for surface = hull; median LGI increases with fold amplitude.
See `docs/reports/lgi-hull-2026-10-20.md` for numbers.

Oracle caveat: `Gyrification` reads VTK through MeshLib's `VtkIO`, which only reads ASCII files
(a binary file makes the run hang), so fixtures are ASCII.
