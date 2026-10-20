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

### Update after CI run 39472001207: eigenvector direction mismatches on ubuntu and windows

Failures: `curv_ellipsoid` umin_it0 (vertices 110, 134, diff 0.05-0.13) / umin_it3 (windows, 0 and 1,
~1e-4), `curv_folded` umax_it3 (30, 35), all with eigen-gap ~0.4, so the earlier gap-scaled
perturbation bound did not apply: the cause is a genuinely discontinuous branch, not conditioning.

**Cause (found by perturbing the golden tensor by 1e-12 relative and listing vertices whose
direction jumps: 27 in ellipsoid, 10 in cut, 4 in ico3; the CI vertices are among them).**
`echelonEigv` (ported verbatim from MeshLib) builds `B = T - lambda*I`, does a two-row pivot search
on column 0, eliminates only row 1 (row 2 is never used), then back-substitutes dividing by
`A00`/`A11`. For the largest eigenvalue of block-structured tensors at symmetric vertices the whole
pivot column is numerically zero (`|B00|`, `|B10|` ~ 1e-18 against `|T|` ~ 1e-2): the row swap and
the divisor are decided by noise, so the eigenvector jumps (it is not even an eigenvector of T).
Different platform noise lands on different vertices (ubuntu vs windows). Not a code
non-determinism: the same inputs give the same answer; it is the algorithm's branch structure.
No code change (changing it would alter arm64/oracle results).

**Test handling.** `_eigenvector_branch_flag` recomputes, from the golden tensor and its
eigenvalues, whether any of the three solves has (a) pivot column ~0, (b) pivot tie
`|B00| ~ |B10|`, (c) reduced `A11` ~ 0, or (d) `|A22|` within noise of the absolute 1e-8
threshold; margin `tol = 1e3 * 1e-10 = 1e-7` relative to `max|T|`. Flagged vertices are counted
ties for umin/umax (tensor path, double and float overloads). Untied mismatches still fail.
A second finding from the same emulation: `eig3symmetric`'s `acos(r)/3` is ill-conditioned when
`|r|` is within ~1e-8 of 1 (rank-one tensors on ridges: two eigenvalues ~1e-10 of the third);
the small eigenvalues then move by ~1e-10 absolute, above the 1e-10-relative-to-themselves bound.
`_acos_ill_conditioned` (`|r| >= 1 - 1e-6`) flags these; cmin/cmax/mean/gauss (and their float
and public-overload variants) and the directions are skipped there as ties. Tolerances are not
loosened anywhere else.

**Validation** (emulated platform noise, no tolerance change): 200 seeds x 5 goldens, each seed
adding random symmetric relative noise to the tensor (run at 1e-15, 1e-14 and 1e-13) and random
+-1 ulp noise on the float32 vertex normals: 0 failures at every noise level. Tie counts (max over
smoothing iteration 0 / 3, vertices flagged by the branch or acos criterion): cut 77, ellipsoid
104, folded 78, ico3 86, perturbed 0 (of 505 / 1106 / 642 / 642 / 642). These are conservative
(about 9 % of ellipsoid vertices); the criterion is golden-only, so it does not depend on the noise.
