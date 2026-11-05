# HSD landmark term, location features and BOBYQA pole search (2026-11-04)

Plan row B6.8. Port note: [ports/hsd.md](../ports/hsd.md) ("Landmarks, location features, pole search").
Licence decision: Q19 (accepted by the owner 2026-11-05: NLopt MIT translation), [clean-room.md](../clean-room.md).

## What was ported

The parts of `HSD.cpp` (ilwoolyu/HSD `f169595`) that upstream disabled in `wrapper/PARSE_ARGS.h` and that the
BISCUIT image binary therefore cannot reach:

| Upstream | Port |
|---|---|
| `initLandmarks` 1142-1174 (vertex indices, `fscanf %d`, unit input sphere, SH basis) | `register._read_landmark_ids`, `HSD._init` |
| `initTangentPlane` with landmarks 831-854 (Fréchet start, `min_bobyqa(2, ..., -PI, PI, 0.1, 1e-6, maxIter)`) | `HSD._landmark_pole`, `kernels.landmark_pole_mean` |
| `optimalMeanCost` 3107-3122 (float32) | `kernels.landmark_pole_cost` |
| `src/bobyqa.h` (**not translated**, see below) | `hsd/bobyqa.py`: translation of NLopt's MIT BOBYQA |
| `updateLandmark` 1317-1330, `varLandmarks` 1694-1749 | `HSD.update_landmark`, `kernels.var_landmarks` |
| `updateGradientLandmark` 2079-2213 (gradient rows + `J^T D J`) | `gradient.accumulate_landmarks` |
| landmark hooks in `init`, `trace`, `updateGradient`, `minGradientDescent` (failure branch) | same places in `register.py` |
| location features (`initProperties` 961-969: surface x, y, z as 3 extra properties), `m_nSurfaceProperties` | `HSD._init`, `nsurf`, `nprop = files + 3` |
| `m_eta = weightMap` (only with landmarks) | `HSD.eta` |
| `main.cpp` `--landmarkDir` (`getListFile` + `getTrimmedList`), `--surfaceDir` (`getListFile`), `if (surf == NULL) weightLoc = 0` | `hsd/cli.py` `_list_dir`, `_trim`, `_subject_names` |

CLI: `pybiscuit hsd` gains `-l/--landmark`, `--landmarkDir`, `--surface`, `--surfaceDir`, `--weightMap`,
`--locationWeight` in a help group titled "upstream-disabled options (NOT in the original HSD binary)"
(`cli.UPSTREAM_DISABLED`). Without them nothing changes: every pre-existing HSD, chain and CLI-flag test and golden
passes unchanged (99 passed, 1 skipped as before; `test_cli_flags_match_parse_args_h` still checks the image flag set,
with the opt-in flags listed separately and checked to be commented out upstream).

Dead code that stays dead: the landmark branch of `guessInitCoeff` (`nLandmark > landmark.size()` is never true), the
`m_feature_weight` values (computed, never used; `--weight` has no effect either), the `subj_id != -1` forms of
`varLandmarks`. Not reproduced: landmark-only runs (`-p` is `required()` upstream, so `nProperties == 0` cannot occur;
it would also leave `m_nSamples` uninitialised); unequal landmark counts and out-of-range indices (UB upstream) raise.

## BOBYQA licence decision

`bobyqa.h` says "Read the accompanying `LICENSE`". The HSD repository's only LICENSE is Lyu's MIT licence, but the
header is Lyu's adaptation of Éric Thiébaut's C BOBYQA (github.com/emmt/Algorithms), whose `LICENSE.md` reads:
Thiébaut's C code is MIT ("Expat"), while Powell's BOBYQA Fortran is listed as "released under the GNU Lesser General
Public License". A derivative of LGPL code cannot go into the Apache-2.0 default install (AGENTS.md), so `bobyqa.h` was
**not read beyond its 10-line `min_bobyqa` wrapper** (call interface: `npt = 2n+1`, `iprint = 0`, `maxfun = max_iter`)
and not translated.

Alternatives checked: Py-BOBYQA (GPL-3, excluded); NLopt `src/algs/bobyqa` (`COPYRIGHT`: MIT, "Copyright (c) 2009
M. J. D. Powell", "Modifications Copyright (c) 2010 Massachusetts Institute of Technology"; its README says Powell
released the Fortran "with no restrictions or charges"). NLopt's `bobyqa.c` is S. G. Johnson's f2c translation of the
same 2009 Powell Fortran. `pybiscuit/hsd/bobyqa.py` translates it line by line, with NLopt's stopping hooks mapped back
to Powell's (`maxfun` at the same places; forced-stop, time, `minf_max`, `ftol` and SGJ's `isinf(suma)` check dropped;
`rhoend` passed directly; no rescaling, which NLopt applies only for unequal initial steps). The MIT notice is in the
module header, `LICENSES/NLopt-BOBYQA.txt` and `NOTICE`. A clean-room implementation from Powell's NA2009/06 report was
not needed.

**How it may differ from upstream numerically.** Both are translations of one Fortran program, so the only expected
differences are arithmetic order and the stopping hooks. Measured: none. `bobyqa.h`, compiled and run as a black box
(`scripts/oracles/harness_bobyqa.cpp`, Apple clang `-O2 -ffp-contract=off`), and the port produce **bit-identical
evaluation sequences and results** on 5 functions × 5 `maxfun` values (quadratic, Rosenbrock, minimum outside the box,
a float32 rippled cost, the pole-cost shape; `maxfun` 3, 5, 8, 20, 200, the last ones ending at `rhoend` after 26-168
evaluations), golden `tests/data/golden/bobyqa`. Residual risks: an FMA build of upstream (x86-64 gcc does not contract
by default; arm64 clang does) and the `RESCUE` path, which none of the test functions triggers.

## Oracle and goldens

Oracle `oracles/hsd-lm-nofma`: `HSD_PATCH_LANDMARKS=1 scripts/oracles/build_hsd_nofma.sh`, i.e. the non-FMA oracle
build with `scripts/oracles/patches/hsd-landmarks/PARSE_ARGS.h.patch` (the six options uncommented; `--landmarkDir`
needed the `/* */` block split; `--filter` and the other `*Dir` forms stay disabled). `harness_hsd.cpp` now builds HSD
exactly as `main.cpp` does (landmark/surface lists, `weightLoc`, `weightMap`), calls `updateGradientLandmark` in the
split gradient, and dumps poles, tangents, deformed landmarks and `m_cost_lm` (still builds and behaves as before when
the lists are empty).

Fixtures (`tests/fixtures/hsd_cases.py:write_landmark_case`): the three warped ico4 subjects of the existing HSD
fixtures, three landmark curves (a latitude arc, a meridian arc, a small circle; 29 vertices, the same indices on every
subject, so correspondence is exact) and surfaces `u * (80 + 8 f(u)) * (1 + 0.05 k)` for the location features.
Goldens `tests/data/golden/hsd/lm_*` (`make_golden_hsd.py`; 1.1 MB). Tests: `tests/test_hsd_landmarks.py` (46 tests,
about 15 s).

## Results (port vs oracle, macOS arm64)

Pole search (`lm_pole`; `--maxIter` 0, 3, 6, 20, 100 = BOBYQA `maxfun`; 3 subjects): pole, tangent basis and
initial deformed landmarks **bit-identical** in all 15 cases.

Unit level (harness `eval` at seeded coefficient states; relative max differences):

| Case | E(A) terms | E(B) terms | landmark cost | raw gradient | direction | Hessian | deformed landmarks |
|---|---|---|---|---|---|---|---|
| lm_gd_d3 (ico4, deg 3) | 2.2e-9 | 5.2e-10 | 0 | 4.3e-9 | 4.3e-9 | - | 0 |
| lm_loc_gd_final (+ location) | 1.8e-9 | 8.2e-10 | 0 | 1.3e-9 | 1.3e-9 | - | 0 |
| lm_newton_band3 (ico6) | 3.3e-8 | 8.4e-10 | 0 | 4.8e-8 | 2.4e-8 | 1.8e-8 | 0 |
| lm_newton_final (ico6, + location) | 5.1e-10 | 1.5e-8 | 0 | 8.1e-9 | 4.4e-6 | 8.7e-6 | 0 |

These are the same levels as the property-only unit tests (energy ≤ 1e-7, Newton direction/Hessian a few 1e-6 from
float32 ulp flips in the deformed vertices).

Full CLI runs (`docs/validation-tolerances.md` HSD rows: median ≤ 0.01°, p99 ≤ 0.1°, energy 0.1 %, coefficients 1e-4):

| Case | iterations port / oracle | final E rel | coeff max diff | angle median (°) | angle p99 (°) |
|---|---|---|---|---|---|
| lm_gd_ico4 (guess, `--weightMap 0.5`) | 15 / 15 | 0 | 0 | 0 | 1.2e-6 |
| lm_loc_ico4 (location only, `--locationWeight 0.7`) | 16 / 16 | 0 | 0 | 0 | 1.2e-6 |
| lm_loc_resample_ico5 (landmarks + location + `--resample`) | 16 / 16 | 0 | 0 | 0 | 1.2e-6 |
| lm_pairwise_ico5 (template fixed) | 16 / 16 | 0 | 0 | 0 | 1.2e-6 |
| lm_dirs_ico4 (`--landmarkDir` with decoys, `--surfaceDir`) | 12 / 12 | 0 | 0 | 0 | 1.2e-6 |
| lm_newton_ico6 (damped Newton, `--maxIter 4`) | 62 / 59 | 4.1e-3 | 2.4e-3 | 0.054 | 0.48 |

On the gradient-descent path the energy logs are identical to the printed 6 digits and the outputs to the written
precision (`%lf` coefficients, `%.9f` vertices; the 1.2e-6° is that print resolution). The damped-Newton run is
chaotic, as the property-only `newton_ico6` already was: both trajectories agree to 6 digits through phase 0 and the
first Newton steps, then drift by about 1e-6 per step. The oracle's own 1-ulp ensemble (6 runs, subject 1's property
file perturbed by one float32 ulp in one value or in about half the values; stored in the golden as `ens_*`) spans
final energy 0.321867-0.32334 (rel spread 4.0e-3), 59-62 iterations, coefficients up to 2.4e-3, angles up to
median 0.054° / p99 0.48°. Its member 0 (one value changed by one ulp) ends at 0.321867 after 62 iterations, where
the port ends (0.321839, 62 iterations). The run passes the Q15 rule `max(policy, 1.5 × spread)` on every metric.

## Not covered / follow-ups

* Image comparison is impossible by construction (the image binary has no such options); the oracle is upstream
  source with only the option lines uncommented.
* BOBYQA `RESCUE` path not exercised; FMA builds of upstream untested.
* Linux x86-64 oracle not run (same status as the rest of HSD, Q1).
