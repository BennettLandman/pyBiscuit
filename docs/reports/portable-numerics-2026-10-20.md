# Portable numerics report (brief 03) — 2026-10-20

Goal: make PR #2 CI green without bit-exact cross-platform assumptions. Target platform is an x86-64 Linux image; goldens come from macOS arm64 oracles.

## Root causes
- **AABB** (`test_golden[aabb_ico3_*]`): the only platform-dependent operation is libm `atan2f` in `cart2sph`. Apple returns pi-1ulp for `atan2f(±0,-x)`; glibc/UCRT return the correctly rounded pi. That flips the pole-face wrap decision (`maxphi-minphi > pi`), changing face boxes, hence 4 closest-face differences (edge-midpoint distance ties, gap <=1e-8) and 22 candidate lists. Reproduced locally by recompiling with a correctly rounded `atan2f` (`test_golden_with_other_libm`): identical failure signature (4 faces; 22 lists).
- **Curvature** (`test_matches_committed_oracle_dumps`): `np.array_equal` on float32 normals computed with different libm/ulp behaviour. Simulated by perturbing normals +-1 ulp: new checks pass, old fail.
- **macOS py3.10**: not a failure; the job was cancelled before any step ran (no runner); versions resolve identically per Python across OSes. Re-check in the next run.

## Policy changes
- `tests/_compare.py`: `assert_close_float32/64`, `assert_same_choice_modulo_ties`, `assert_candidate_sets_close`. `-m bitexact` opt-in marker for bit-exact oracle regression (skipped by default / in CI).
- docs/validation-tolerances.md: "Cross-platform policy" (PROPOSED).
- Code audit: no source change needed; all reductions are explicit scalar Numba loops, no fastmath/FMA contraction in the LLVM IR; remaining platform-dependent sites are libm (`acosf`, `expf`, `atan2f`), documented under `### Determinism` in the port notes.

## Items needing an owner decision
1. AABB eps: `DIST_EPS = 4.8e-7`, `ANGLE_EPS = 1.9e-6` (observed gaps <=1e-8; wrap-flip mismatches explained by ANGLE_EPS).
2. `curvature_quad` outputs checked at 1e-5 rather than 1e-10: input is float32 vertex normals which can differ by an ulp across libm.
3. `test_spharm` float64 golden at 1e-9 (pre-existing, looser than 1e-10): MeshLib's `sqrt(1-cos^2)` amplifies libm ulps near poles.
4. Vertices with normal exactly +x in the quad fit flip curvature sign on 1 ulp of noise (C++ discontinuity ported as is); skipped and counted as ties.
5. Alternative not taken: canonicalise `atan2f(±0,-x)` to Apple's value. Rejected as an Apple libm quirk.
6. No real x86 run was available locally; evidence is simulation plus CI.

---

# b03 fragment: AABB_Sphere portable numerics

## Root cause (per failing CI test, Linux/Windows x86-64)

Goldens were produced on macOS arm64. Reproduced locally by recompiling `aabb.py` with a
correctly rounded `atan2f` (what glibc/UCRT return): exactly the CI signature.

| test | CI | cause |
|---|---|---|
| `aabb_ico3_closest` | 4 faces differ `[3408 3489 3839 3926]` | edge-midpoint / cut queries lie on the surface within float32 noise; two faces tied (distance gap <= 1e-8) and the candidate path differs by an `atan2f` ulp |
| `aabb_ico3_bool_eps`, `aabb_ico3_xcand_eps` | 4 faces differ (golden finds pole face 1198, x86 none) | Apple `atan2f(0, -x)` = pi - 1 ulp, glibc/UCRT = correctly rounded pi; `maxphi - minphi > pi` flips for pole faces, so the face box is `[0, pi]` vs wrap box `[pi, 2pi]` |
| `aabb_ico3_search` | 22 candidate lists differ, first `[8, 56, 79, 93, 170]` | same wrap-box flip changes the union boxes of the leaves/ancestors containing pole faces |

Only `atan2f` is platform dependent; `+ - * / sqrt` are IEEE, no fast-math, no FMA contraction
(0 fma instructions in compiled kernels on arm64 where FMA is baseline), no vector reductions.

## Before / after

| | before | after |
|---|---|---|
| arm64, committed goldens | exact equality | modulo ties; 0 mismatches, 0 ties; `-m bitexact` exact equality passes (6 cases) |
| emulated x86 (`test_golden_with_other_libm`) | fails (4/4/4/22) | passes: closest 4 mismatches = 4 distance ties (gap <= 1e-8); bool 4 and xcand 4 = box-ambiguous (wrap flip); search 22 lists, all under box-ambiguous nodes |
| deformed_update(_refresh) | pass | pass, 0 mismatches in both modes |

Tie counts per case (n = 3927 queries): closest 0 (arm64) / 4 distance ties (other libm); bool_eps 0 / 4 box-ambiguous;
xcand_eps 0 / 4 box-ambiguous; search 0 / 22 lists; deformed_update 0 / 0; deformed_update_refresh 0 / 0.

## Eps justifications

* `DIST_EPS = 4 * 2^-23 = 4.8e-7`: 4 ulp(1); queries are float32 surface points; observed gaps <= 1.0e-8.
* `ANGLE_EPS = 4 * 2^-21 = 1.9e-6`: 4 ulp(2 pi), libm `atan2f` is within 1 ulp across platforms.
* bary abs 1e-6 unchanged.

## Code changes

`_atan2f` isolated (documented single platform-dependent site); public `cart2sph` now uses the
same jitted kernel as the tree (was NumPy float32 `arctan2`, differing from the tree in ~6 %).
No numeric change vs arm64 (goldens bitexact, live oracle tests pass).

---

## B03 tasks 1-3: golden comparisons and determinism audit (non-AABB)

Policy: float32 paths 1e-5 rel, float64 paths 1e-10 rel (via `tests/_compare.py`, no API change,
no helper added); integers exact; bit-exact checks only under `pytest -m bitexact`.

### Per-test before / after

| Test | Before | After |
|---|---|---|
| `test_curvature.py::test_matches_committed_oracle_dumps[*]` | `np.array_equal` on float32 face/vertex normals (the CI failure); `_eq` with `atol=0`; quad outputs at 1e-10; directions `<= 1e-10`/`1e-5` absolute | normals `assert_close_float32`; tensor / cmin / cmax / mean / gauss `assert_close_float64` (1e-10, abs scaled by golden magnitude so near-zero eigenvalues do not fail on relative error); float overloads 1e-5; neighbour lists still exact (int) |
| same, `curvature_quad` + float overload | 1e-10 / 1e-5 with no tie handling | 1e-5 (the quad frame consumes the *float32* vertex normals, which differ by an ulp across `acosf` implementations); frame-degenerate vertices (`abs(n_x) >= 1 - 3e-7`) skipped and counted |
| directions `umin`/`umax` (tensor + quad) | `max component diff <= 1e-10` (resp. 1e-5) | per-vertex bound `10*delta/gap + frame slack`, where `gap` is the relative eigen-gap *recomputed from the golden*; vertices with `gap < 1e3*delta` are counted as ties (`LAST_TIES`) and skipped, untied mismatches fail |
| `test_normals_bitexact_with_arm64_oracle[*]` (new, `@pytest.mark.bitexact`) | -- | the old `np.array_equal` assertions on `face_normals`/`vertex_normals`; 5/5 pass locally with `-m bitexact` |
| `test_sphere_map.py::test_matches_committed_oracle_dumps[*]` | `assert_allclose(rtol=1e-5, atol=1e-7)`; `n_inverted ==` exact | `assert_close_float32(abs_=1e-7)`; `n_inverted` may differ from the count on the golden by at most the number of golden faces with `|value| < 1e-6` (those are decided by float32 rounding) |
| `test_sphere_map.py::test_bitexact_with_arm64_oracle[*]` (new, `bitexact`) | -- | `array_equal` of float32 vertices for all types x ITERS plus exact inverted-face count; 3/3 pass locally |
| `test_spharm.py::test_golden_vs_meshlib_oracle`, `test_live_oracle_degree20` | `assert_allclose` with `F32 = (1e-5, 1e-7)`, `F64 = (1e-9, 1e-11)` | same numbers through `assert_close_float32/64` (no tolerance change) |
| `test_remesh.py::_run_golden` (all `test_golden*`) | tie rows (closest face differs) exempt from value comparison, unverified | unchanged tolerances; additionally, when ties occur and the golden has `deformed.vtk` (`coeff_full`), `_check_tie_geometry` verifies both faces reconstruct the same 3-D point to 1e-5; tie count recorded in stats |
| `test_fixtures.py::test_geometry_hash_constants[*]` | SHA of coordinates rounded to 1e-9 | unchanged; new `test_hash_inputs_far_from_rounding_boundary[*]` proves the hashes are insensitive to libm/SIMD noise (nearest coordinate to a rounding boundary is 3.2e-4 grid steps = 3e-13 abs away, versus <= ~1e-15 platform noise; the test requires >= 1e-5 steps) |

Local results (macOS arm64): `pytest -q` full suite and `-m bitexact`, `ruff check` clean.

### Comparisons reviewed and left exact (justified)

- Integer data: neighbour lists (`nbr`, `nbr_count`, `nbr_indptr`), faces everywhere,
  `test_icosphere` face arrays, `test_operators` sparse patterns/faces.
- Same-code self-consistency, not oracle-derived: `test_spharm` (`float32 == float64.astype(float32)`,
  `from_degree` slice, coefficient crop), `test_icosphere` (nested levels, determinism),
  `test_fixtures` (seeded generators), `test_sphere_map` (input untouched, `type=7 == type=1`),
  `test_curvature` (symmetric tensor, `curvature(m) == curvature(v, f)`), `test_geometry`
  (input untouched, `smooth(iterations=0)`, Mesh vs arrays).
- Round trips through the same code and dtype: `test_io_vtk`, `test_io_gifti`, `test_io_txt`,
  `test_io_freesurfer`, `test_operators::test_npz_roundtrip`, `test_frames` (allclose to nibabel).
- `test_geometry.py` golden/live checks already use per-vertex normwise relative error against
  1e-5 (no abs floor, so stricter than the helper) and passed on x86 CI; left as is.
- `test_remesh.py::compare_file` already tolerance-based (RTOL 1e-5, print-rounding ATOL) with
  tie accounting.

### Code audit (task 3): no code change was necessary

All of `mesh/{geometry,curvature,sphere_map}.py` and `sphere/{spharm,remesh}.py` evaluate
reductions in explicit scalar Numba loops in the C++ order; Numba emits `fmul`/`fadd` without
`contract`/`fast` flags (inspected in the LLVM IR), so no FMA fusion or SIMD reassociation. The one
platform-dependent primitive is libm (`acosf`, `expf`, double `acos`/`sin`/`cos`/`atan2`); for
float32 it is the *system* `acosf`, i.e. glibc on the x86 Linux image, the same function the x86
oracle uses, so replacing it by a portable implementation would move away from the target.
Sites (details in the `### Determinism` subsections of `docs/ports/meshlib-geometry.md`,
`spherical-harmonics.md`, `spherical-remesh.md`):

1. `geometry._vertex_normals` `acosf`; `_normal_pdf` `expf`; `_vertex_areas` double `acos`/`sin`.
2. `curvature._tensor` `acos(beta)`, sign test `cpt > 0` (continuous at 0), `mergesort` on unique keys.
3. `curvature._eig3` `p1 == 0` branch (exactly diagonal tensors persist across platforms) and
   equal-diagonal "unset lambda" ties (mirrors MeshLib).
4. `curvature._principal` selection by `abs(lam)` strict comparisons: eigenvector swap on magnitude ties.
5. `curvature._quad` frame `theta = acos(-n_x)` from a float32 normal: (a) exact `n = +x` gives a
   zero rotation axis and the identity rotation (curvature sign flipped; present in the committed
   ico3/cut/folded goldens at the +x vertex), (b) amplification `ulp / sin(theta)` near the x axis,
   (c) NaN for arguments rounded past 1.
6. `sphere_map` float32 Jacobi iteration and inverted-face stopping test: no libm calls, so
   bit-identical on IEEE platforms; faces within an ulp of zero are the only possible flip.
7. `spharm` Legendre branch thresholds and the pole amplification in `sqrt(1 - x*x)`;
   `synthesize` is a fixed-order scalar accumulation.
8. `remesh` float32 `sin`/`cos` via double libm; nearest-neighbour pick on barycentric
   coefficients; `(int)` truncation (all covered by the existing tie mask).

### What still needed a tie / looser-than-1e-10 argument

- Tensor-path directions: tie skipping and a gap-scaled bound (first-order eigenvector
  perturbation). Values stay at 1e-10.
- `curvature_quad` outputs (float64 code, float32 input): 1e-5 instead of 1e-10, plus the
  frame-degenerate tie set. This is the only float64 output checked at the float32 tolerance.
- `test_spharm` float64 golden: **1e-9** rel (not 1e-10). Pre-existing (not introduced here, and
  it passed on x86 CI); it is looser than `docs/validation-tolerances.md`. Needs an owner
  decision: either accept 1e-9 for the SH basis because of MeshLib's `sqrt(1 - cos^2)` pole
  amplification, or tighten once x86 ulp-level data is available.

### Not verifiable locally

x86-64 behaviour. I simulated x86-like noise instead: random +-1-ulp perturbation of the float32
vertex normals (clipped to [-1, 1]) passes all five curvature goldens with the new checks; the old
exact checks fail on any perturbation, and a +-1-ulp perturbation of *face* normals would exceed
1e-10 on the tensor path (face normals use only exact IEEE operations, so this is not expected).

---

## B03 task 4-5: CI macOS 3.10 job and cross-platform policy

### macOS + Python 3.10 job: not a real failure
- Run 39465318691 (https://github.com/BennettLandman/pyBiscuit/actions/runs/39465318691): job `py3.10 macos-latest`
  (id 118775077449) is `cancelled` with zero steps executed (started 17:21:11Z, ended
  17:36:13Z, no log exists). It never got a runner; it was cancelled at the 15 min run
  boundary/fail-fast, not by a test or install error. `py3.13 macos-latest` in the same run
  passed.
- Newest run 39469312755: macOS jobs were still queued when checked (3m47s), so no result yet.
- Resolved versions per job (pip install logs): py3.13 on all OSes: numpy 2.5.4, scipy 1.18.1,
  numba 0.68.0, llvmlite 0.50.0, nibabel 5.4.2. py3.10 (ubuntu, windows): numpy 2.2.6,
  scipy 1.15.3, numba 0.68.0, nibabel 5.4.2. py3.10 macOS has no install log to compare, but
  there is no evidence of a version-range issue (3.10 resolves the same as other 3.10 jobs).
  Note the numpy 2.2 vs 2.5 split between 3.10 and 3.13: a platform-independent difference.
- No change to pyproject.toml dependencies. `fail-fast: false` is already set in
  .github/workflows/ci.yml (kept). Numba caching not added (minimal change).
- Action: re-check py3.10 macOS in run 39469312755 once it finishes; if it fails there,
  inspect its log (expected to be the same exact-equality failures as the other jobs).

### Cross-platform policy (docs/validation-tolerances.md, PROPOSED)
E means within tolerance with ties explained on any platform; bit-exactness is diagnostic
(`pytest -m bitexact`, oracle platform only); acceptance reference is the x86-64 Linux
image; macOS arm64 oracle builds use -ffp-contract=off but are provisional; argmin/closest-face
mismatches accepted only if recomputed margin <= eps with tie count reported; candidate lists
compared as sets with eps-near-bound exceptions; default float32 rel 1e-5 / float64 rel 1e-10 with
abs scaled by golden magnitude; no tolerance loosened beyond the document; fixed-order
accumulation required where float32 rounding drives a discrete decision.
