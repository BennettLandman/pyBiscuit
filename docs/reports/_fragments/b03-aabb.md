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
