# Report: sulcal depth (brief 17, plan row B9.5) — 2026-10-21

**Result.** `pybiscuit.quant.sulcal_depth` ports the full `sulcal_depth.m` chain (spec §9.2); CLI `pybiscuit sulcal-depth in.vtk out.txt`.
Row B9.5 is **In progress, provisional**: the analytic acceptance passes, the E-class acceptance against the image cannot be run yet (Q1).

## What was built
* New: `quant/sulcal_depth.py` (bwperim, quasi-Euclidean 26-neighbour geodesic distance as a numba Dijkstra, trilinear interp3 with
  IEEE Inf/NaN, the hull/field/seed assembly), `quant/sulcal_depth_cli.py`, `cli.py` dispatch block, `tests/test_sulcal_depth.py` (8 tests).
* Reused from brief 07 without duplication: `_voxelise.voxelise`, and `hull._smooth3_gaussian/_close_ball/_dilate_ball` plus its constants.
  No change to brief 07 files.
* Port note with the semantics reproduced per MATLAB call: [docs/ports/sulcal-depth.md](../ports/sulcal-depth.md).

## Evidence
* Building blocks vs analytic answers: chamfer steps (1, √2, √3, quasi-Euclidean detour 3,1,0 = √2+2), wall/gap detour, unreachable → NaN, seed policy, exact trilinear on a linear field, Inf/NaN weight semantics, 18-connected perimeter count.
* Chain: sphere R = 40: depth 2.4 ± 0.6 mm baseline (hull offset ≈ 2.5 voxels + dilation), radius-independent; grooved sphere: groove bottom − baseline = 2.6 / 6.5 / 10.5 mm for groove depths 4 / 8 / 12 mm, monotone in depth and along the profile.
* `ruff check .` clean; full suite run before the PR (count in the PR text).

## Findings and caveats
1. A convex surface does **not** give 0: the smoothing + threshold places the hull ≈ 2–3 mm outside the pial solid. Spec's "sphere ≈ 0" is true only to that voxel tolerance; this is a property of the algorithm, not a port defect.
2. The thickened field (6 passes) is what keeps the interpolation finite: every vertex neighbourhood lies in the field, so the `Inf`/`NaN` corner cases (spec §9.2) did not occur in any fixture (`np.isfinite` asserted). Real cortex with deep closed pockets or hull-enclosed voids might hit them.
3. Unverified MATLAB behaviours (image run needed, spec §15 item 6): `strel('ball')` decomposition (shared with OuterHull), `bwperim` array-border handling, `bwdistgeodesic` seeds outside the mask (`seeds_outside_mask` knob), `interp3` with Inf neighbours.
4. Requests for others: none (no shared file edited besides the `cli.py` dispatch block and plan.md row B9.5).

## Next
Image run (Q1) → compare `<h>.gray.SD.txt` to `sulcal-depth` output; if the mismatch concentrates at hull-touching vertices, flip `seeds_outside_mask`.
