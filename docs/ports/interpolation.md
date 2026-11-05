# Port note: gw triangular interpolation (plan row B2.7)

Python: `src/pybiscuit/mesh/interpolation.py` (`InterpolationMesh`, `LinearInterpolation`,
`QuadraticInterpolation`, `CubicInterpolation`, `fit_2nd_order_polynomial_2d`). Tests:
`tests/test_interpolation.py`, fixtures `tests/fixtures/interpolation_cases.py`. Oracle harness:
`scripts/oracles/harness_interpolation.cpp` (built by `scripts/oracles/build_curve_extraction_nofma.sh`),
goldens `tests/data/golden/interpolation/{fold_l3,cut_noisy_l3}` from
`scripts/oracles/make_golden_interpolation.py`.

Not part of BISCUIT: nothing on the BISCUIT call graph reaches these classes. They were ported for
completeness of the MeshLib port (owner request, 2026-11-05). Their only upstream caller is gw's
gradient-descent path `gw_geodesic/GW_GeodesicPath.cpp::AddNewPoint`, which is not ported either
(TRACE uses MeshLib's own `Geodesic/GeodesicPath.cpp`, see [geodesic.md](geodesic.md)).

## Source (MeshLib `dcef807`) and what is ported

All of it is compiled into MeshLib (`CMakeLists.txt`, `SRC_GEOD_FILES`).

| Python | C++ |
|---|---|
| `LinearInterpolation` | `gw_geodesic/GW_TriangularInterpolation_Linear.{h,cpp}` (whole) |
| `QuadraticInterpolation` | `gw_geodesic/GW_TriangularInterpolation_Quadratic.{h,cpp}` (whole) |
| `CubicInterpolation` | `gw_geodesic/GW_TriangularInterpolation_Cubic.{h,cpp}`: constructor, `SetUpTriangularInterpolation`, `ComputeGradient`, `ComputeValue`, `GetType` |
| `InterpolationMesh.set_up`, `.gradient`, `.kind` | `gw_geodesic/GW_GeodesicFace.{h,inl,cpp}` (`SetUpTriangularInterpolation`, `ComputeGradient`, the static interpolation type, default quadratic) |
| `InterpolationMesh.third_vertex` | `gw_core/GW_Face.inl::GetVertex(v1, v2)` / `GetEdgeNumber(v1, v2)` |
| `fit_2nd_order_polynomial_2d` | `gw_maths/GW_Maths.h::Fit2ndOrderPolynomial2D` and `SolveLU` (see "The 6x6 solve") |
| face neighbours | `mesh.geodesic._build_connectivity` (already ported, B2.6) |
| — | `GW_TriangularInterpolation_ABC.h` (interface only) |

Not ported: `GW_TriangularInterpolation_Cubic::{ComputeLocalGradient, SetLocalGradient,
ComputeLocalBasis}` (private, never called by anything; `ComputeLocalGradient` would also assert,
because the object it creates reports the quadratic type). `GW_TriangularInterpolation.{h,inl,cpp}`
are empty files.

## Semantics reproduced

* The interpolation is set up on a face's *stored* vertex order (V0, V1, V2 = face slots 0, 1, 2;
  W_i = the vertex across the edge opposite V_i), but `value`/`gradient` take any order of the
  face's vertices plus a point `x v0 + y v1 + (1-x-y) v2`, as `GW_GeodesicPath` calls them. The
  gradient comes back in that caller frame: `grad = dx e0 + dy e1`, `e0`, `e1` the unit edges to
  `v2`.
* Linear: `value = d0 x + d1 y + d2 (1-x-y)`; gradient from the two directional derivatives along
  the unit edges, inverted with the 2x2 Gram matrix (`det = 1 - dot^2`).
* Quadratic: the three W vertices are unfolded into the face plane through the angles `a..e`
  (`acos` of normalised dot products), a polynomial in `(1, X, Y, XY, X^2, Y^2)` is fitted through
  the six points, and evaluated in the orthonormal frame `u = e0/|e0|`, `v = normalize((u x e1) x u)`,
  origin V2.
* Missing neighbours (open boundary): W0 = (V1+V2)/2, **W1 = (V0+V1)/2** (sic; the matching value
  is (d0+d2)/2), W2 = (V0+V1)/2. With W1 or W2 on the far side of the line, the `acos` argument can
  leave [-1, 1]; C returns NaN and the whole fit becomes NaN. The port does the same (`_acos`); the
  `cut_noisy_l3` golden contains such faces and the NaNs are in the same places.
* Quadratic gradient: `(0, 0)` when `|det| <= 1e-9`. Cubic: no such guard (IEEE `inf`/`nan` on a
  zero determinant).
* Cubic = the quadratic fit (same code upstream), `get_type()` returns `"quadratic"`, as upstream.
  The face dispatcher refuses `cubic` (`ValueError`) where upstream has `GW_ASSERT(GW_False)` and
  then dereferences NULL in release builds.
* `GW_VectorStatic`: `v / f` multiplies by `1/f` (zero vector if `f == 0`), `Normalize` gives
  `(1, 0, 0)` below `1e-9`, dot products and norms accumulate from 0 in component order.
* `GW_Face::GetEdgeNumber(v1, v2)` returns 0 when the edge is absent (so `GetVertex` gives slot 0).

## The 6x6 solve

`Fit2ndOrderPolynomial2D` solves the 6x6 system with `SolveLU`, i.e. `ludcmp` + `lubksb` in
`GW_Maths.h`. Those two routines are Numerical Recipes code (their comments are the book's text),
which is not under an open license. `_lu_solve` is an independent implementation of the same
published algorithm - Crout LU with implicit (row-scaled) partial pivoting, ties to the later row,
a zero pivot replaced by `1e-9`, forward substitution skipping leading zeros, back substitution -
in the same operation order, which is what makes the result bit-identical. Whether that is
acceptable, or whether LAPACK (`numpy.linalg.solve`, max scaled deviation 2.1e-13 on the goldens,
measured 2026-11-05) should replace it, was **Q21** in [open-questions.md](../open-questions.md): on 2026-11-05 the
owner chose to keep `_lu_solve`.

## Precision

Vertex positions are float32 (MeshLib `Mesh`) widened to double; everything else is double, as
`GW_Float`. Transcendentals: `acos`, `cos`, `sin`, `sqrt` from the host libm through Python's
`math`.

**macOS oracle vs. port.** clang on macOS fuses each `sin(a)`/`cos(a)` pair into Apple's
`__sincos_stret`, which differs from separate `sin`/`cos` in the last bit for about 0.45 % of
arguments (1e7 random angles in [0, pi]). Six of the 3,770 golden queries see it (max relative
deviation 6.5e-14). With `__sincos_stret` substituted (macOS-only test), all 3,770 queries are
bit-identical, NaNs included. glibc's `sincos` uses the same algorithms as its `sin`/`cos`, so a
Linux build is expected to match the port's separate calls; not verified on Linux.

## Validation (2026-11-05)

| Golden | Mesh | Queries | Port vs oracle |
|---|---|---|---|
| `fold_l3` | folded sphere a=0.15, Y_6^3, ico3, r=50 (642 v, 1280 f, closed) | 1,760 | bit-identical |
| `cut_noisy_l3` | ico3 cut at z=0.5, sigma=0.02 noise, r=30 (505 v, 960 f, one boundary loop) | 2,010 | 2,004 bit-identical; 6 differ by <= 6.5e-14 (sincos, above); bit-identical with `__sincos_stret` |

Queries: one per face (kind cycling L/Q/C, random vertex order and point) plus the three vertices,
an edge midpoint and the centroid in all three kinds for 32 random faces and every boundary face.
Each row checks value, both gradient components and the six fitted coefficients. Distances are the
port's fast marching (bit-identical to MeshLib, B2.6) written as `%r` and read back by the harness.
Tests: `test_golden_vs_meshlib` (1e-12 relative, NaN pattern equal),
`test_golden_bit_identical_with_apple_sincos` (macOS), plus analytic checks (affine field exact for
linear, quadratic field exact on a flat patch, fit vs `numpy.linalg.solve`, cubic = quadratic,
dispatcher behaviour).
