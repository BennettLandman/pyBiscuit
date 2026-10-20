# Geometry, oracles and remesh report (brief 02) — 2026-10-20

Branch `b2-b4-geometry-remesh`, cut from `b0-b2-b4-foundations` because PR #1 was not yet merged. Full suite: 360 passed (`.venv/bin/pytest -q`), `ruff check .` clean.

**Provisional references:** all E-class evidence below is against *upstream-macos-arm64 oracles*, not the image. Image goldens (B1.4) override them.

**Headline findings**
1. Ports are bit-identical to MeshLib/SphericalRemesh built with `-ffp-contract=off` (matches x86-64 image arithmetic). Default arm64 clang fuses multiply-adds: curvature then differs by median 1.7e-5 (58% of vertices >1e-5), and SphericalRemesh shows 48-8306 closest-face ties per case. `build_meshlib.sh` now defaults to `-ffp-contract=off`; the existing `oracles/meshlib/lib` should be rebuilt.
2. HSD uses MeshLib's `SphericalHarmonics::basis` (double overload; HSD.cpp:365, 826, 1163, 3096), so one port serves B4.2 and B6.2. B6.2 still needs BOBYQA, exponential-map update and HSD I/O.
3. All five oracles build with no numeric source change. Build-only shims: Accelerate LAPACK/BLAS headers for HSD; serial `omp.h` stub for CurveExtraction/Gyrification, which also need `-D_USE_OPENMP` (upstream segfaults on a zero-length per-thread array otherwise). HSD numerics come from Accelerate, not OpenBLAS: expect ulp differences vs image.
4. SphericalRemesh quirks reproduced on purpose: `--ftype Float/INT_16/DOUBLE` write empty files; parcellation call overwrites its own inputs (`data/out.mat.<k>.txt`); `--nneighbor` is nearest for data but per-coordinate median for the surface. 24/84 SH-deformed outputs differ by 1 ulp in the last printed digit (cause untraced).
5. MeshLib quirks ported as-is: `vertex_areas` is 1/3 of incident triangle areas (not Voronoi) with a spurious wrap triangle at boundaries; curvature `umin`/`cmax` naming inverted; uninitialised outputs ported as 0; the tensor is scale-invariant.
6. MeshProperty flag hypotheses (no parity claim): `--tensorsmoothing`~`curvature` tensor iterations, `--surfsmoothing`~`smooth_vertices`, `--datasmoothing`~`smooth_data`; `--prctile/--normbefore/--normafter/--invert/-q` have no MeshLib counterpart.
7. Not verified: master vs image-documented CBT v1.3 / cmorph v1.7 (no image/Linux host).


---

# H1 housekeeping fragment

## read_asc frame
- `pybiscuit.io.freesurfer.read_asc(path, *, frame)`: `frame` is now a required keyword-only
  argument, validated to `"RAS"` or `"tkRAS"` (anything else, including `None`, raises
  `ValueError`; omitting it or passing it positionally raises `TypeError`). MaCRUISE `.asc` are scanner RAS.
- `read_fs_surf` unchanged (binary FreeSurfer surfaces are tkRAS, default kept).
- Updated: tests/test_io_freesurfer.py (added `test_asc_frame_required_and_validated`),
  tests/test_frames.py (both calls `frame="RAS"`; the "converted" comparison still uses the raw vertices).
  No other callers in src/ or scripts/. Historical docs
  (docs/reports/foundations-2026-10-20.md, docs/reports/_fragments/io-freesurfer-frames.md) still describe the
  old default as history; not edited (not owned).

## UGSCNN license
- Source: https://raw.githubusercontent.com/maxjiang93/ugscnn/master/LICENSE; default-branch HEAD
  commit `64e27969c65b4807e41c1e275972dab8bb72c512` (git ls-remote), fetched 2026-10-20.
- Confirmed MIT: "UGSCNN: Spherical CNNs on Unstructured Grids / The MIT License (MIT) /
  Copyright (c) 2018 Chiyu "Max" Jiang, University of California, Berkeley". Verbatim text (1166 bytes)
  written to LICENSES/UGSCNN-MeshConv.txt.
- LICENSES/README.md index row updated (copyright, commit, date).
- SphericalLabeling reference copy: meshcnn/*.py have no copyright/licence header; its LICENSE is
  MIT (c) 2019 Ilwoo Lyu (already in LICENSES/SphericalLabeling.txt).
- Attribution header comment added to src/pybiscuit/sphere/operators.py (before the module docstring).
- B4.1 status untouched.
- Follow-up suggestion (not done, docs/licensing.md not owned): its UGSCNN row can now drop "pending".

---

# O1a fragment: MeshLib + SphericalRemesh oracles

- Commits verified: MeshLib `dcef807`, SphericalRemesh `b677c39`. Toolchain Apple clang 21.0.0, arm64, `-std=c++11 -O2`.
- MeshLib builds in a few seconds, one build-only patch (`<cstddef>` in `Util/Geom.h`). No LAPACK is used by MeshLib or SphericalRemesh, so `-framework Accelerate` is not needed. `READY` marker at `oracles/meshlib/READY`.
- SphericalRemesh needs CLI11 v1.9.1 (CMake downloads it at configure time); fetched once from the GitHub release and cached in `oracles/_dl/` (needs network once; sandbox allowed github.com).
- Smoke command: `SphericalRemesh -s sphere.vtk -i subject.vtk -r ref.vtk -p prop.txt -o out.vtk --outputProperty outprop --bary bary.txt --quiet`. Fixtures: `make_golden.sr_fixture` (icosphere(5) source, icosphere(4)/(6) reference, radial-bump subject, seeded property seed 20261020). 5->6 runs in ~0.65 s.
- Surprises: property output name is derived from the property filename (`outprop.prop.txt`); bary file rows are `id w0 w1 w2` (`-0.000000` appears); text outputs are 6 decimals so goldens compare at ~1e-6. `up5to6` outputs are >5 MB so only `down5to4` and `up3to4` goldens are committed (about 300 KB each); the 5->6 case runs live in the test.
- No `-i` is required for property remeshing; `-c` (HSD coefficients) not exercised in the smoke test.
- FP contraction/FMA: not disabled; compare to x86-64 SIF may differ in last bits.

---

# O1b fragment: HSD / CurveExtraction / Gyrification oracles

All three built with plain `/usr/bin/clang++` (Apple clang 21.0.0, arm64), `-std=c++11 -O2 -DNDEBUG -include cstddef`,
no OpenMP/CUDA, linked against MeshLib dcef807 (O1a's `oracles/meshlib/lib/libMesh.a`; identical output verified
against my own throwaway build of the same sources). Scripts `scripts/oracles/build_{hsd,curve_extraction,gyrification}.sh`.

| tool | commit | version string | status | smoke time |
|---|---|---|---|---|
| HSD | f169595 | v1.3.9 | built | ~5 s (2 subj, icosphere 5, d=8, ico 4, maxIter 3) |
| CurveExtraction | ded932d | v1.1.2 | built | ~2 s (icosphere 5) |
| Gyrification | b9fa760 | v1.2.2 | built | ~1.6 s (icosphere 5) |

Patches (build-only, no numeric source changes; no `.patch` files, header shims instead):
- `patches/hsd-shim/{lapacke.h,cblas.h}` -> Accelerate (`ACCELERATE_NEW_LAPACK`). HSD uses only `dsysv_`, `cblas_dgemm`, `cblas_daxpy`.
  Numerics come from Accelerate rather than OpenBLAS: expect last-bit/ulp differences vs the Linux image.
- `patches/curve-gyr-omp-shim/omp.h`: serial stub (unguarded `#include <omp.h>`).
- Surprise: Curve/Gyrification need `-D_USE_OPENMP` even with the stub; without it nThreads=0 and CurveExtraction segfaults
  (zero-length per-thread array). Upstream bug in non-OpenMP builds.
- Surprise: CLI11 is `file(DOWNLOAD)`ed by CMake; I fetched v1.9.1 (HSD) and v1.9.0 from the GitHub release URL (sha256 in
  `oracles/_deps/cli11-*`: 1.9.1 `52a3bc82...`, 1.9.0 `6f0a1d88...`).
- Version note: master reports HSD v1.3.9 / TRACE v1.1.2 / LGI v1.2.2; the image-documented CBT v1.3 / cmorph v1.7 could
  not be checked here (no image); no visible behavioural discrepancy in --help, but versions are not equal strings.

Smoke commands: see `scripts/oracles/README.md` (## O1b tools). HSD: two icosphere(5) subjects, s2 rotated 8 deg about z with the
property being the same smooth function of the unrotated coordinates; after `-d 8 --icosahedron 4 --maxIter 3` the max
corresponding-vertex mismatch drops 7.0e-2 -> 1.6e-3; deterministic across runs. CurveExtraction defaults (-0.05/0.03 seeds) are tuned for
real cortex in mm: a synthetic bumpy sphere needs `--sulcalSeed -0.01 --gyralSeed 0.01`. Gyrification output name is
`<o>.lgi.map.<adjusted area>.txt` (area baked into the file name).
No goldens committed (smoke only).

---

## B2.4a — MeshLib geometry basics + smoothing

Code `src/pybiscuit/mesh/geometry.py`; tests `tests/test_geometry.py` (24 pass, ~4 s warm);
port note `docs/ports/meshlib-geometry.md` (sections Normals/areas, Smoothing, precision table).
Oracle: `scripts/oracles/harness_geometry.cpp` + `build_harness_geometry.sh` (links
`oracles/meshlib/lib/libMesh.a`), goldens by `make_golden_geometry.py`
(`tests/data/golden/meshlib/geometry_{ico2,ico3,perturb,cut}`, 59-239 KB each).

Measured (port vs MeshLib dcef807, relative per-vertex, criterion 1e-5):
* MeshLib built `-ffp-contract=off` (= x86-64 behaviour): **0 (bit identical)** for neighbour lists,
  face/vertex normals, face/vertex areas, mean and sigma smoothing (1-5 passes), post-smoothing
  normals, float32/float64 data smoothing; on icosphere, uv_sphere, ellipsoid, folded, perturbed,
  cut sphere (open boundary) and `lh.white` (163,842 vertices).
* Default arm64 oracle (clang fuses multiply-add): worst 1.2e-6 (`vertex_areas`, ellipsoid; 7.2e-7
  on `lh.white`); normals/areas/smoothing <= 6e-7; sigma = 0.25 x edge length with 4 passes amplifies
  to 8.7e-5 (2.7e-4 ellipsoid) because of `exp(-d^2/2s^2)`; not a port bug.
* Timings (icosphere 7): neighbour lists 0.09 s, vertex normals 5 ms, 10 smoothing passes 8 ms
  (31 ms with sigma). Ties/discrete decisions: none in this part.

Surprises:
* MeshLib has no face-area or Voronoi vertex-area function. "Vertex area" = `GeodesicA::vertexArea`
  (private; 1/3 of incident triangle areas, wrap-around triangle double counted on boundary vertices);
  "face area" = the inline expression in `SurfaceUtil::sphere`.
* `smoothing` never reads or uses the vertex itself, and `Mesh::openFile` ignores normals in the file.
* MeshLib's VtkIO only reads ASCII legacy VTK, so the harness input is written as ASCII with `%.9g`
  float32 text; numba needs `error_model="numpy"` to mimic C 0/0 -> NaN for stiff sigma.
* Reproducing the oracle bit-for-bit needs `-ffp-contract=off`; the O1a lib (default flags) is FMA-fused.
  Suggest O1a documents this in `oracles.md`.

Exports to add in `pybiscuit/mesh/__init__.py` (not done): `NeighborLists`, `face_areas`,
`face_normals`, `smooth_data`, `smooth_vertices`, `vertex_areas`, `vertex_neighbor_lists`,
`vertex_normals` (suggest exposing as `from . import geometry` rather than flat names).

---

## B2.4b — MeshLib curvature + spherical mapping

Code `src/pybiscuit/mesh/curvature.py`, `src/pybiscuit/mesh/sphere_map.py`; tests
`tests/test_curvature.py`, `tests/test_sphere_map.py` (24 pass incl. 2 reference-marked live-oracle
tests on `reg/mb101/lh.white.vtk`, ~40 s total; 22 pass in ~5 s without reference/binary); port note
`docs/ports/meshlib-geometry.md` (sections Curvature, Spherical mapping, incl. the MeshProperty-flag
hypothesis). Oracle: `scripts/oracles/harness_curvature.cpp` (stand-alone compile command in its header;
uses `#define private public` to reach `tensor/smoothingTensor/principalCurvature`), goldens by
`scripts/oracles/make_golden_curvature.py` -> `tests/data/golden/meshlib/{curv_ico3,curv_ellipsoid,
curv_folded,curv_cut,curv_perturbed,spheremap_ellipsoid,spheremap_folded,spheremap_perturbed}`
(`input.vtk`, `golden.npz`, `command.json`; 0.09-0.39 MB each). The live harness binary sits in
`oracles/meshlib-b24b/harness_curvature` (outside git, built with `-ffp-contract=off` from the MeshLib
sources, not from `oracles/meshlib/lib`).

Measured (port vs MeshLib `dcef807`, `-ffp-contract=off`; acceptance 1e-5 float / 1e-10 double):
* **Bit-identical** (max abs and rel difference 0.0): neighbour lists, face/vertex normals, tensor
  (iter 0 and 3), `cmin/cmax/umin/umax` (double and float overload, directions even without the sign
  allowance), mean/Gaussian, `curvature_quad` (double + float), and `sphere` vertices after every
  checkpoint `max_iter` in {1,2,5,20,100,10000}, type 0 and 1; on the 8 committed fixtures and on
  `lh.white` (163,842 v). Principal directions agree exactly, no sign flips.
* Sphere mapping, `lh.white`: converged with 0 flipped faces after 2,329 (type 0) / 1,882 (type 1)
  passes; inverted faces after 1/2/5/20/100 passes = 75,892/75,415/73,968/68,598/52,567 (type 0),
  75,384/74,488/72,259/64,671/46,011 (type 1) of 327,680, identical to the oracle (iteration-for-iteration
  at the checkpoints). Fixtures converge flip-free to unit norm within 2e-7.
* Ties / undefined cases (see port note): 0 flat vertices, 0 `cmin==cmax`, 0 NaN on `lh.white`; 8 flat
  rim vertices on `cut_sphere(3)`; 16 NaN eigenvector rows on `folded_sphere(0.1,4,2,3)` (iter 0,
  oracle has the same NaNs). Flat-vertex directions are uninitialised upstream, ported as 0, excluded
  from comparisons.
* Timings (warm, M-series): `curvature` on `lh.white` 0.2-0.3 s, `curvature_quad` 0.3 s, `sphere_map`
  2.5 s (type 0) / 1.5 s (type 1); the harness takes ~30 s for the whole dump including 12 sphere runs.

Surprises:
* **FMA sensitivity.** Against MeshLib built with default arm64 flags (`oracles/meshlib/lib`,
  `-ffp-contract=on`) the *same* curvature code differs per vertex by median 1.7e-5, p99 0.9 %, up to
  several times the value for near-zero `cmin`, and 58 % of vertices exceed 1e-5 (tensor up to 3.3e-4
  of its max entry). Cause: float32 face normals differ by an ulp and `acos(n1.n2)` of nearly coplanar
  faces amplifies it. The image binary (x86-64, no FMA) matches the non-fused build, so the 1e-5
  acceptance is only meaningful against a non-fused oracle. Sphere mapping is not sensitive (<= 4.2e-7
  absolute). Suggest `build_meshlib.sh` adds `-ffp-contract=off` (no numeric source change).
* The tensor is scale invariant (angle x length/mean length): it is a relative strength, not a 1/mm
  curvature; `umin` is the eigenvector of `cmax` (naming inverted); `curvature_quad` orders by
  magnitude, `curvature` algebraically. Flat/tie branches leave outputs uninitialised in C++.
* MeshLib's `SurfaceUtil::tensor/smoothingTensor/principalCurvature` are private; no MeshLib source
  was changed (harness macro only).
* Both reference-marked tests need `reference/` and the harness binary; they skip otherwise.

Exports to add in `pybiscuit/mesh/__init__.py` (not done): `Curvature`, `curvature`, `curvature_quad`,
`gaussian_curvature`, `mean_curvature`, `principal_curvature`, `smoothing_tensor`, `tensor`
(from `.curvature`), `SphereMapResult`, `count_inverted_faces`, `sphere_map` (from `.sphere_map`).

---

# B2.5 AABB_Sphere (fragment)

Oracle: upstream MeshLib `dcef807b` built with `clang++ -O2 -std=c++11 -ffp-contract=off` (Apple clang 21,
macOS arm64) via `scripts/oracles/build_harness_aabb.sh`; harness `scripts/oracles/harness_aabb.cpp`
(private overloads reached with `#define private public`, no source change).

## Accuracy (tests/test_aabb_sphere.py, 1e5 seeded unit-sphere queries each)

| case | face index mismatches | max abs bary diff | exact-tie queries (resolved by lowest index, agree with oracle) |
|---|---|---|---|
| icosphere(5) | 0 | 0 (bit identical) | 3 |
| icosphere(6) | 0 | 0 | 0 |
| icosphere(7) | 0 | 0 | 1 |
| deformed(5) fresh / update (normals stale / refreshed) | 0 / 0 / 0 | 0 | 2 |
| deformed(7) fresh / update | 0 / 0 / 0 | 0 | 0 |

Also bit identical: 51,212 structured queries (vertices, edge midpoints, centroids, poles, +-pi cut; ~24,000 exact
ties), eps overloads `bool`/`xcand` at (0,0), (0.01,0.01), (0.02,0.005), `searchTree` candidate lists (20,000 queries,
undeformed and deformed), icosphere schemes biscuit/ugscnn, fixtures icosphere/uv_sphere, and the full tree leaf partition.
Deformed fixture: radial 0.85-0.95 plus small azimuth/elevation shift (see `deformed_sphere` in the test).

## Timing (M-series, 14 numba threads)

ico7 (327,680 faces): tree build 0.75 s; 163,842 queries **0.015-0.02 s** warm (limit 2 s); cold: import + build JIT ~3 s,
first-query JIT ~1.5 s (numba cache enabled, so later processes load from cache).

## Surprises

* `construction()` has a dedicated `n == 2` branch (always two leaves); missing it gave different candidate sets
  (visible only with `searchTree`/leaf dump, not in final answers on ico5 since the extra candidate was never the winner).
* `update()` after `setVertex` uses stale face normals for pole detection (SphericalRemesh/HSD never call `updateNormal`).
* `m_maxPhi` is always 2*pi after `update()` => second (phi+2pi) search happens for every `phi <= 0`.
* Queries inside the surface (radius < mesh radius) are never found and cost all 9 escalation trials.
* float32 triple-product barycentrics: writing vertices as `%.9g` of a float64 and re-reading as float differs by 1 ulp from
  direct rounding, which produced ~3e-6 bary differences; tests round to float32 first.
* Exports to add to `pybiscuit.sphere.__init__`: `SphereAABB`, `barycentric`, `cart2sph`, `closest_face` (from `.aabb`).

---

# B6.2* spherical harmonics (fragment)

* Port: `src/pybiscuit/sphere/spharm.py` (MeshLib dcef807 `SphericalHarmonics::basis`,
  `Series::legendre2`, `cart2sph`); note `docs/ports/spherical-harmonics.md`.
* Finding: HSD (f169595) has no basis of its own; it calls MeshLib's double
  `SphericalHarmonics::basis` (HSD.cpp:365, 826, 1163, 3096). One port serves
  HSD and SphericalRemesh. Dead code in MeshLib: the non-Schmidt `legendre` branch.
* Oracle: `scripts/oracles/harness_sh.cpp`, built by `build_sh.sh` to
  `~/dev/biscuit/oracles/meshlib-sh/harness_sh`; build-only patch = `-include cstddef`.
  (No `oracles/meshlib/READY` existed, so MeshLib sources were compiled directly.)
* Measured vs oracle, degree <= 20, 2,000 random + 16 special + ico3 points:
  float32 max abs 1.4e-16 (99.9991% bitwise identical); float64 max abs 1.3e-13.
  Accept metric used: abs/rel 1e-5 / 1e-7 (f32), 1e-9 / 1e-11 (f64).
* Speed: ico7 vertices (163,842) at degree 20: 0.58 s; synth 0.06 s.
* Surprises: scipy differs from MeshLib by convention only (no Condon-Shortley,
  sin at -m); MeshLib loses ~4e-8 abs within 1e-6 rad of the poles (kept).
  The oracle float overload computes in double, so no float32 precision limit on the oracle.
  Remesh reads coefficients as float32 (%f) and sums delta in float32, HSD in double.
* Tests: `tests/test_spharm.py` (14 passed; 3 golden cases + live-oracle cases that skip
  when the harness is absent).

---

## B4.2 — SphericalRemesh port

Code `src/pybiscuit/sphere/{remesh.py,remesh_cli.py}`, subcommand `pybiscuit spherical-remesh`
(`cli.py`), tests `tests/test_remesh.py` (97 pass incl. 70 oracle cases, ~33 s with both oracles,
~3 s without), port note `docs/ports/spherical-remesh.md`, matrix/golden generator
`scripts/oracles/make_golden_remesh.py` (owned; did not touch `make_golden.py`), new oracle build
`scripts/oracles/build_spherical_remesh_nofma.sh` (`-ffp-contract=off`, unmodified sources; only
`-include cstddef`). 11 new goldens in `tests/data/golden/spherical-remesh/` (<= 813 KB each).

Measured vs upstream SphericalRemesh b677c39 (35 cases + ico7/deg-20 run, two builds):
* **-ffp-contract=off oracle** (= x86-64 arithmetic): all non-SH outputs (property resampling,
  NN, bary, surface, colours, every `--ftype`, parcellation call) are **byte-identical**;
  SH-deformed outputs differ by 1 float32 ulp (<= 1.2e-7) on ~3 % of vertices (SH sum/libm), so
  text differs by <= 1e-6 absolute (one unit in the 6th decimal; 1.2e-5 relative at |x|~0.05),
  faces identical, bary <= 1e-6 (1.1e-5 on ico7). 0 ties anywhere.
* **default arm64 oracle** (FMA): same values within tolerance but 48-8306 face "ties" per case
  where a reference vertex coincides with a sphere vertex (both faces give the same point), and
  `(int)`-truncated outputs (colours, `--ftype int*`) off by one level on those rows.
* Timing: ico7 (163,842 verts), degree-20 deformation, 2 props: port 2.4 s vs oracle 4.2 s.

Surprises:
* `--ftype Float`/`INT_16`/`DOUBLE` pass CLI11 validation but write **empty** `.dat` files (strcmp
  against lower case); reproduced on purpose.
* The parcellation call writes `data/out.mat.<k>.txt` over its own inputs.
* Constructor builds `tan1` from the unprojected sample and relies on dangling `fv()` pointers; the
  observed behaviour is reproduced (bit-identical for pole != 0, deg 0).
* `--color` without `--keepColor` shows up only without `-i`; `--keepColor` indexes the colour map
  by `-s` vertex id although it is read per reference vertex.
* Possible AABB note (not a bug in `aabb.py`): `closest_face` returns exactly the C++ values but
  the ico7 barycentric coefficients are conditioned to ~1e-5 per float32 ulp of a vertex, so a
  1e-6 bary tolerance only holds for bit-identical vertices.
* The O1a goldens `down5to4`/`up3to4` came from the FMA build; they pass the port with tolerance
  (ties handled), not bytes.

Exports to add: `pybiscuit/sphere/__init__.py`: `SphericalRemeshing`, `remesh_resample` (= `remesh.resample`),
`remesh_run` (= `remesh.run`) — prefer `from . import remesh`.
