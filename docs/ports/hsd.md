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

## Hypothesis: how BISCUIT supplies the template (unverified, orchestrator is encrypted)

The image ships per-template files `reg/<tmpl>/{lh,rh}.{curv,sulc,iH}.{mean,var}.txt` (163,842 lines each) and
`lh.white.vtk`/`lh.inflated.vtk`. Consistent call: `HSD -s <template ico sphere> <subject sphere> -p
<tmpl mean x3> <subject curv sulc iH> --fixedSubjects 0 --tmpVar <3 var files concatenated> -d D --icosahedron L --weight w1 w2 w3
-o ... --outputcoeff ... [--resample]` (properties subject-major, template first). This is pair-wise mode; the
three `.var.txt` files must be concatenated property-major into one `--tmpVar` file. To be confirmed by B1.2.

## Known gaps and quirks

* Landmarks / location / BOBYQA not ported (above). `bobyqa.h` header says "Read the accompanying `LICENSE`":
  check the HSD licence text before porting.
* `--icosahedron 0-2` is accepted by the CLI but `m_nSamples` is uninitialised upstream (UB); port uses the real count.
* `--icomesh` with a radius-100 mesh and without `--resample`: `guessInitCoeff` builds `AABB_Sphere` on the raw mesh;
  unit queries lie inside it, so no face is found and the template mean map becomes 0 (reproduced). Verify BISCUIT's call.
* `const float *` into destroyed temporaries (`initTangentPlane`) assumed to keep their values (true in both builds tested).
* `m_cost_lm` is uninitialised upstream when there are no landmarks; assumed 0 (constant offset).

## Validation

See the report. Unit level (state A/B, 3 subjects, degree 6): GD path rel <= 9e-9 (energy, gradient); Newton path
energy <= 1e-7, damped direction / Hessian <= 3.5e-6 in the final phase (float32 ulp flips in the deformed vertices
amplified by `1/sqrt(1-x^2)`). Full runs: identical iteration counts on the GD cases (ico4/5, with guess / resample / fixed subject / pair-wise) and on an
ico7 degree-8 Newton-path run (82 iterations, median angular difference 3e-5 deg); the ico6 case differs by one iteration
(64 vs 63, final energy rel 9e-5, median 5e-3 deg, p99 0.027 deg). Platform differences vs the Linux image are untested (owner
decision on the Linux host).
