# Port note: HSD spherical registration (plan rows B6.1, B6.3-B6.6)

Code: `src/pybiscuit/hsd/` (`register.py` class `HSD`, `kernels.py`, `gradient.py`, `guess.py`,
`icosahedron.py`, `geom.py`, `cli.py`); CLI `pybiscuit hsd`. Tests: `tests/test_hsd.py`. Oracle:
`scripts/oracles/build_hsd_nofma.sh` (CLI + `harness_hsd.cpp` internals dump), goldens
`tests/data/golden/hsd/<case>/` from `scripts/oracles/make_golden_hsd.py`, fixtures
`tests/fixtures/hsd_cases.py`. Report: [hsd-2026-10-20](../reports/hsd-2026-10-20.md).

## Pinned sources

| Repo | Commit | Files |
|---|---|---|
| ilwoolyu/HSD | `f169595` (v1.3.9, = the version string of the image binary) | `src/HSD.cpp` (3122 lines), `src/HSD.h`, `src/bobyqa.h` (not ported), `wrapper/main.cpp`, `wrapper/PARSE_ARGS.h`; `src/cuda/*` out of scope (Q10) |
| ilwoolyu/MeshLib | `dcef807` | `Vector`/`Coordinate`/`Statistics` in `Util/Geom.cpp`, `AABB_Sphere`, `SphericalHarmonics::basis`, `SurfaceUtil::smoothing` (ported earlier: `sphere/aabb.py`, `sphere/spharm.py`, `mesh/geometry.py`) |

## Image binary vs upstream, and the CLI

`strings` on `reference/sif_FC_v2.2/.../cmorph/bin/HSD` shows exactly the option set of
`wrapper/PARSE_ARGS.h` at `f169595`: `-s -p -o --outputcoeff -c -d --icosahedron --weight --idprior
--maxIter --icomesh --gres --noguess --resample --writecoeff --fixedSubjects --tmpVar --nThreads`.
**`-l/--landmark`, `--surface`, `--weightMap`, `--locationWeight`, the `*Dir` batch forms and `--filter` are
commented out upstream and absent from the image** (the brief lists them; they cannot be reached).
`main.cpp` therefore always passes `landmark = NULL`, `surf = NULL`, `weightLoc = 0`, `weightMap = 1`:
the landmark term, the location features and the BOBYQA pole search (`initTangentPlane`, only when landmarks
exist) are dead code in the shipped tool. They are **not ported** (hand-back item). `tests::test_cli_flags_match_parse_args_h`
parses the header and compares flag sets. Defaults are the CLI's (`idprior 625`, `degree 15`, `maxIter 20`,
`icosahedron 7`, `gres 3`), not the C++ constructor's. `--nThreads 0` = `OMP_NUM_THREADS`, else all cores
(upstream: 1; results do not depend on the thread count).

## Function map

| HSD.cpp (lines) | Python | Notes |
|---|---|---|
| `icosahedron` 2890-3050 | `icosahedron.icosahedron` | float32 vectors, first-come edge map, per-child orientation fix; bit-identical to the oracle for levels 3-6 (3-5 in committed goldens, 3 and 6 live) |
| `init` 180-744 | `HSD._init` | same order; resampling / pair-wise branches included |
| `initSphericalHarmonics`, `initTangentPlane` (no landmarks) 746-864 | `HSD._init`, `kernels.pole_axis/tangent_plane` | `-c` file: pole `%f`, coefficients `%lf`, degree cropped |
| `initLocalPatch` 866-891 | `register._patch_cache` | neighbour across each directed edge, first insertion wins |
| `initProperties` 893-1065 | `register._init`, `kernels.normalize_property` | median z-score, soft clamp at +-3, twice; float32 `Statistics::*` |
| `initTriangleFlipping`, `initArea` 1067-1140 | `kernels.init_flips`, `face_areas` | |
| `initPairwise` 1176-1239 | `HSD._init_pairwise`, `kernels.pairwise_bary` | `--tmpVar`: property-major, one value per template vertex |
| `updateCoordinate` 1241-1287 | `kernels.update_coordinates` | double SH sum, float32 exponential map and rotations |
| `updateDeformation`, `updateDisplacement` 1289-1519 | `HSD.update_deformation`, `update_displacement` | |
| `updateProperties` 1332-1414 | `HSD.update_properties`, `kernels.patch_search` | cached 2-ring patch search, then `AABB_Sphere::closestFace` |
| `updatePropertyStats` 1416-1475 | `HSD.update_property_stats` | mean of (fixed) subjects, variance (fine levels), 3x ico smoothing, floor at 0.1 mean |
| `trace`, `cost`, `var*`, `testTriangleFlip` 1521-1846 | `HSD.trace/cost`, `kernels.var_*`, `test_triangle_flip` | `m_cost_lm` uninitialised upstream: 0 |
| `minGradientDescent` 1848-1961 | `HSD.min_gradient_descent` | `m_mincost` is `float`; step doubling/halving, `nSuccessIter` |
| `updateGradient*`, `updateHessian`, `updateNewGradient`, `ATB`, `ATDA`, `linear` 1963-2077, 2243-2615 | `HSD.update_gradient`, `gradient.py` | Jacobian rows in Numba (chunks of 16384 samples), `J^T diag J` and sums in NumPy BLAS, `scipy.linalg.lapack.dsysv(lower=1)` |
| `guessInitCoeff` 2617-2845 | `guess.guess_init_coeff` | float loop variables kept (`c1 += step`) |
| `optimization` 2847-2888 | `HSD.optimization` | |
| `saveCoeff`, `saveSphere` 3071-3105 | `HSD.save_coeff`, `save_sphere` | `--resample`: output re-evaluated on the original sphere |
| landmarks (`initLandmarks`, `updateLandmark`, `varLandmarks`, `updateGradientLandmark`, `optimalMeanCost`), location features, `bobyqa.h` | not ported | unreachable from the CLI |

## Algorithm as implemented (what "LM" really is)

* `--icosahedron < 6` (`m_fine_res = 6`): per phase `d = 0..degree` plain gradient descent on
  `E = eta*prop + lambda1*area` (`area` = mean |log(a1/a0)| over faces, from phase 1) with step doubling on
  success and halving on failure; triangle-flip checks shrink the step.
* `--icosahedron >= 6`: from phase 1 the regulariser is the squared angular displacement (`lambda2 = idprior`),
  phases optimise one degree band `[d, d]`, and the direction is `H^-1 g` with
  `H = J^T diag(D) J` (prop + displacement), diagonal damped `H_ii *= 1 + lambda` (`H_ii += lambda H_ii + alpha`,
  `alpha = 0.001/exp(nSuccessIter)` in the final phase) and solved with LAPACK `dsysv` ('L'). `lambda = 0.001 t/(nSuccessIter+1)`.
* Pair-wise mode (2 subjects, 1 fixed): fixed subject is the template (mean = its feature), `--tmpVar` its variance.

## Precision table

| Quantity | C++ type | Port |
|---|---|---|
| sphere vertices, `p`, `p0`, `tan1/2`, pole, `area0/1`, features, properties, samples | `float` | float32, same operation order (`Vector` ops, `Coordinate::rotation/rotPoint/sph2cart/cart2bary`) |
| `exp`/`log`/`acos`/`sin`/`cos` on float | float libm | double libm rounded to float32 (platform libm differs <= 1 ulp) |
| SH basis `Y` | `double` (`basis(double*)`) | float64 |
| coefficients, gradient, Hessian, `delta` sums, mean, variance, energies, `t`, `step` | `double` | float64, sequential sums where C++ is sequential |
| `m_mincost`, `m_eta`, `m_lambda1/2` | `float` | float32 values held in float64 |
| `neg_area_threshold` | `float` | float32 |
| `Statistics::mean/var/median` (properties) | float | float32 sequential |
| Jacobian row, `ATB`, `ATDA` | double, BLAS | rows identical expressions; sums = NumPy (fixed chunking, not BLAS blocking) |

## OpenMP regions (17 `#pragma omp parallel for`) and reduction order

All 17 loops are over independent items (vertices, samples, subjects, properties); none uses an OpenMP
`reduction`. The only cross-item sums are BLAS calls (`ATB` column sums with a ones vector, `ATDA` = `A^T (D A)`),
whose order is the BLAS library's. Port: `prange` over chunks for deformation, patch search and Jacobian rows
(per-item results, thread-count independent); sums via `J.sum(axis=0)` per 16384-row chunk (fixed order) and `J^T (D J)`
per chunk, accumulated in chunk order. Subject loops are sequential (per-subject kernels are parallel inside).
Remaining platform-dependent pieces: libm, BLAS summation (about 1e-13 relative), `dsysv` pivoting.

## How BISCUIT calls HSD (source-derived, brief 11 spec §5.3; implemented in brief 21)

This replaces the single-call hypothesis of brief 05 (spec item H-1: **mismatch**). `surf_reg_macruise.m:55-66` runs
**four chained single-property calls** per hemisphere, template first and fixed (`--fixedSubjects 0`, pair-wise mode),
degree 10, no `--weight`, no `--noguess`, no `--resample`, default `--idprior 625 --maxIter 20 --gres 3`:

| # | `-c C C` | `-s` (template, subject) | `-p` (template mean, subject) | `--icomesh` (level) | `--tmpVar` |
|---|---|---|---|---|---|
| 1 | no | `sphere_327680`, `sphere_20480` | `iH.mean`, `res.iH` | `sphere_5120` (4) | no |
| 2 | yes | `sphere_327680`, `sphere_81920` (rh, MaCRUISE script: swapped) | `sulc.mean`, `res.sulc` | `sphere_20480` (5) | no |
| 3 | yes | `sphere_327680`, `sphere_327680` | `sulc.mean`, `res.sulc` | `sphere_81920` (6) | `sulc.var` |
| 4 | yes | `sphere_327680`, `sphere_327680` | `curv.mean`, `res.H` (= -4H) | `sphere_327680` (7) | `curv.var`; also `-o R R` |

`--outputcoeff C C` in every call. Levels 4-5 run the gradient-descent branch and levels 6-7 the damped Newton branch,
so both run in every registration. Code: `pybiscuit.hsd.chain` (`chain_argv`, `run_chain`, `parc_alignment_argv`);
CLI `pybiscuit register --surf-dir J --template-dir T --sphere-dir S [--hemi lh rh] [--rh-stage2 macruise|lh] [--dry-run]`.
`pybiscuit hsd` stays the plain tool (the original flags).

Consequences found by replaying it (tests: `tests/test_hsd_chain.py`; report: [hsd-chain-2026-10-21](../reports/hsd-chain-2026-10-21.md)):

* **H-2** subject mesh shorter than the property files (calls 1-2: 10,242 / 40,962 vertices against 163,842 lines): only
  the first `nVertex` lines are read; the icosphere vertex order is hierarchical, so they are the right values. Tested
  (cut file == long file) and exercised by the chain goldens (fixtures use the same length ratio).
* **H-3** `--icomesh` takes radius-100 spheres. The guess builds `AABB_Sphere` on the raw mesh and queries it with unit
  vectors, so **every** query misses (642 of 642 in the fixture) and the template mean map used by the guess is 0 everywhere.
  The cost is then the plain sum of squared subject features, and the guess picks an arbitrary rigid rotation
  (degree-0 coefficients 16.3 / -2.2 / 2.2 in the fixture, whatever the true alignment). It runs in call 1 only: calls 2-4 read
  non-zero `-c` coefficients and skip it. Port and oracle agree on the result (golden `guess_icomesh`).
  **Image evidence (brief 22, 2026-10-22): the image does not behave like this.** On the three open examples the
  upstream behaviour sends the native FreeSurfer sphere on a ~135 degree rotation in call 1 and the whole chain ends
  in a poor basin (E1 1.72 vs the image's 0.757, deformed sphere 113 degrees median from the image's); with call 1's
  guess skipped, or with the guess's mean map queried on the unit-radius icosahedron, the whole chain reproduces the
  image's registered sphere to a median of 0.14-0.17 degrees (inside the port's own 1-ulp spread). The default is
  therefore `guess_unit_ico=True` (CLI `--guess-raw-icomesh` restores upstream's raw radius-100 query; the oracle
  comparisons and the golden test use it). Which of the two the image really does (unit query, or no effective guess,
  e.g. a different `AABB_Sphere` build or the UB at `HSD.cpp:316`) is not decidable from the outputs (Q17).
* **New oracle finding (UB, `HSD.cpp:316`)**: `const float *p = V.unit().fv();` in the `--icomesh` branch reads the stack slot of a
  destroyed temporary (`unit()` returns by value). With Apple clang -O2 every icomesh sample is garbage and HSD dies with
  "no closest triangle found"; the original `hsd-nofma` build cannot run any `--icomesh` call. The goldens come from
  `hsd-icomesh-nofma` (`HSD_PATCH_ICOMESH=1 scripts/oracles/build_hsd_nofma.sh`, one-line fix, otherwise identical). The port computes
  the intended value. Whether BISCUIT's gcc build keeps the value is unknown (it must, or production would fail; Q1).
* **Latent upstream abort at ico7.** `updateProperties` exits with "no closest triangle found" when a sample lies within float32 noise of an edge and the float32 `cart2bary` of both adjacent faces is below -1e-5; the oracle's own `AABB_Sphere` does the same on the port's dumped mesh. A port run aborted this way at template scale (call 4); not handled (stopping rules are preserved).
* **H-4** same file as `-c` and `--outputcoeff` for both subjects: the fixed template neither reads nor writes it; tested.
* **H-7** `rh` call 2 order: both orders supported (`--rh-stage2`), both with goldens.
* **H-8** parcellation alignment (`parc.m:61`, `-d 0 --noguess --icosahedron 5`, `-o` = input sphere, native-resolution mesh):
  `parc_alignment_argv`, golden `parc_align`.
* **Chaos.** The per-call result of a call with a poor start (lh call 2 in the fixture, after the arbitrary guess rotation) or a
  Newton stage can depend on one float32 ulp of one input value even inside the oracle (iteration counts 38-45, final energy
  +-1 %). The tests therefore compare the port with the oracle's own 1-ulp spread (stored in the goldens); see the report.

## Known gaps and quirks

* Landmarks / location / BOBYQA not ported (above). `bobyqa.h` header says "Read the accompanying `LICENSE`":
  check the HSD licence text before porting.
* `--icosahedron 0-2` is accepted by the CLI but `m_nSamples` is uninitialised upstream (UB); port uses the real count.
* `--icomesh` with a radius-100 mesh and without `--resample`: `guessInitCoeff` builds `AABB_Sphere` on the raw mesh;
  unit queries lie inside it, so no face is found and the template mean map becomes 0 (reproduced; this **is** BISCUIT's call,
  see the section above and golden `guess_icomesh`).
* `const float *` into destroyed temporaries (`initTangentPlane`) assumed to keep their values (true in both builds tested).
* `m_cost_lm` is uninitialised upstream when there are no landmarks; assumed 0 (constant offset).

## Validation

See the report. Unit level (state A/B, 3 subjects, degree 6): GD path rel <= 9e-9 (energy, gradient); Newton path
energy <= 1e-7, damped direction / Hessian <= 3.5e-6 in the final phase (float32 ulp flips in the deformed vertices
amplified by `1/sqrt(1-x^2)`). Full runs: identical iteration counts on the GD cases (ico4/5, with guess / resample / fixed subject / pair-wise) and on an
ico7 degree-8 Newton-path run (82 iterations, median angular difference 3e-5 deg); the ico6 case differs by one iteration
(64 vs 63, final energy rel 9e-5, median 5e-3 deg, p99 0.027 deg). Platform differences vs the Linux image are untested (owner
decision on the Linux host).
