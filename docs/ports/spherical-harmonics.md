# Port note: spherical-harmonic basis (plan row B6.2*)

Code: `src/pybiscuit/sphere/spharm.py`. Tests: `tests/test_spharm.py`.
Oracle: `scripts/oracles/harness_sh.cpp` (+ `build_sh.sh`, `make_golden_sh.py`).

## Pinned sources

| Repo | Commit | Files / lines |
|---|---|---|
| ilwoolyu/MeshLib | `dcef807b488eb10d81d9608cf9fa7d06b8ed23e8` | `src/Util/SphericalHarmonics.cpp` 16-77 (float), 79-139 (double); `include/Util/SphericalHarmonics.h` 16-21; `src/Util/Geom.cpp` 435-441 (`Coordinate::cart2sph` double), 1452-1534 (`Series::legendre2` double); `include/Util/Geom.h` 13 (`PI`) |
| ilwoolyu/HSD | `f169595` (`f1695956bf833d5f83659db4264ef3cc6faaf950`) | `src/HSD.cpp` (see below) |
| ilwoolyu/SphericalRemesh | `b677c39` | `src/SphericalRemeshing.cpp` 30-230, 237-312; `wrapper/main.cpp`; `wrapper/PARSE_ARGS.h` 36-49 |

## Answer to the B6.2 scope question

**HSD uses the same basis; it has no implementation of its own.** `HSD.cpp:18`
includes `SphericalHarmonics.h` (from MeshLib; `CMakeLists.txt:23`
`find_package(MeshLib REQUIRED)`) and calls the **double** overload
`SphericalHarmonics::basis(m_degree, double vd[3], double *Y)` at
`HSD.cpp:365` (icosphere basis), `:826` (per-subject vertex basis; also used for
"fixed" subjects), `:1163` (landmarks) and `:3096` (resampling). The vertex is
widened float -> double first (`double vd[3] = {v0[0], ...}`). HSD adds no basis
derivatives: the gradient code reuses the same `Y` rows
(`HSD.cpp:1243-1262` `updateCoordinate`, `grad.cu` consume `Y` only). The only
HSD-specific pieces are the coefficient bookkeeping (double precision) and the
pole/tangent-plane exponential map (`initTangentPlane`, `updateCoordinate`).
The "fixed SH bases" of CBT v1.3 therefore means "fixed (non-optimised)
subjects whose basis rows are computed once with this function", not a second
basis.

Consequence for scope: **one port serves B6.2 (HSD) and B4.2 (SphericalRemesh)**.
B6.2 still needs, beyond this module, the HSD cost/optimiser (BOBYQA,
`bobyqa.h`), the tangent-plane/exponential-map update, and the file I/O, which are
not in this row's ownership. The `float` overload (used by SphericalRemesh) is the
`double` code with the result rounded to float32; HSD uses the double one.
Caveat: the `biscuit_FC_v2.2.sif` binary (CBT v1.3, June 2021) was not
executed here (macOS arm64); the MeshLib header says "Update: June 2021", i.e.
matches the pinned source, but binary equivalence is only verified through the
HSD golden runs planned in B6.

## Definition (what is ported)

`basis(degree, p, Y, from)` -> `Y[K]`, `K = (degree+1)^2 - from^2`, degree-major:
column of (l, m) is `l^2 + l + m - from^2`, `m in [-l, l]`.

* `phi = atan2(y, x)` (0 when x = y = 0); elevation `= atan2(z, hypot(x,y))`
  (0 when z == 0); `theta = pi/2 - elevation`; `x = cos(theta)`. The input
  vector need not be unit. Float overload: the float input is cast to double and
  **all arithmetic is double**; each output is rounded to float32. (The
  `cart2sph(const double*)` overload is selected, not the float one.)
* `legendre2(l, x, Pm, schmidt=true)` - the MATLAB `legendre(..., 'sch')`
  recurrence with the rescaling branch for tiny `sin(theta)^l` (`|sn| <= 1.49e-154`,
  only reachable within ~2e-8 rad of the poles at l = 20). `schmidt` is
  `l >= 0`, i.e. always true, so the other branch (`Series::legendre`,
  `factorial`, `sqrt(2)` factor, odd-m sign flip) is **dead code** and not ported.
* `Y(l,0) = sqrt((2l+1)/4pi) * Pm[0]`; for m >= 1 with
  `pre = sqrt((2l+1)/4pi) * Pm[m]`: `Y(l,-m) = pre * sin(m phi)`,
  `Y(l,+m) = pre * cos(m phi)`; `sin/cos(m*phi)` are evaluated directly as
  `sin((m)*phi)` (no angle-addition recurrence).
* Poles: `factor = sqrt(1-x^2) == 0` -> `Pm[0] = x^l`, other `Pm = 0`.

### Convention differences (verified against `scipy.special.sph_harm_y`)

| | MeshLib / HSD / SphericalRemesh | scipy `sph_harm_y(l,m,theta,phi)` |
|---|---|---|
| Type | real | complex |
| Normalisation | orthonormal on the unit sphere (`int Y_i Y_j = delta`, tested) | orthonormal |
| Condon-Shortley phase | **none** (it is cancelled at the end of `legendre2`) | included |
| `Y(l,+m)` (m>0) | `sqrt2 * (-1)^m * Re(scipy)` = `sqrt2 * Re(Y_lm without CS)` | `Re` |
| `Y(l,-m)` (m>0) | `sqrt2 * (-1)^m * Im(scipy)` (sin term, at the **lower** index) | `Im` |
| m = 0 | `Re(scipy)` | |
| Ordering | `l` ascending, within `l`: `m = -l..l` | n/a |
| Low degrees | `[1, y, z, x, xy, yz, (3z^2-1)/2, xz, (x^2-y^2)/2]` x normalisation consts (note +x, not -x) | |

## Public API (stable)

```python
basis(points, degree, from_degree=0) -> ndarray      # (N,3)|(3,) -> (N,K)|(K,)
    # float32 in -> float32 out ("float" overload; SphericalRemesh)
    # otherwise float64 out ("double" overload; HSD)
basis_size(degree, from_degree=0) -> int
sh_index(degree, order, from_degree=0) -> int
legendre_schmidt(n, x) -> ndarray                    # Series::legendre2(..., schmidt=true)
@dataclass ShDeformation(degree:int, pole:ndarray(2), coeff:ndarray(3, (degree+1)**2))
read_sh_coefficients(path, max_degree=-1, dtype=np.float32) -> ShDeformation
write_sh_coefficients(path, deformation) -> None     # HSD saveCoeff format
synthesize(Y, coeff, degree=None) -> ndarray (N,3)   # delta, see below
```

Suggested exports for `pybiscuit/sphere/__init__.py`: `basis`, `basis_size`,
`sh_index`, `ShDeformation`, `read_sh_coefficients`, `write_sh_coefficients`,
`synthesize` (as `sh_basis`/`sh_synthesize` if a prefix is preferred).

## How SphericalRemesh `--deform / -c / --deg / --deg0` consumes SH coefficients

Flags (`wrapper/PARSE_ARGS.h:36-49`): `-c/--coeff <txt>` = HSD coefficient file;
`--deform <vtk>` = output deformed unit sphere; `--deg` (default -1) caps the
degree; `--deg0` (default -1) is the first degree of the sweep.

**File format** (written by `HSD::saveCoeff`, `HSD.cpp:3071-3082`; read by
`SphericalRemeshing.cpp:100-147`): whitespace-separated text:
```
<degree>                       # int
<pole_azimuth> <pole_elev>     # "%f %f", radians
c0_0 c1_0 c2_0                 # (degree+1)^2 lines "%lf %lf %lf", rows in the (l,m) order above
...
```
Column k of a row is the coefficient for component k: `delta[0]` (rotation angle
about the deformed pole axis), `delta[1]`, `delta[2]` (displacement along
`tan1`, `tan2`). Remesh reads with `%f` into **float32**; HSD with `%lf`.
Note HSD's own reader (`HSD.cpp:761-789`) stores rows at stride
`(m_degree+1)^2` of the *current* run's degree, remesh at the (cropped) file
degree.

**Degree handling** (`SphericalRemeshing.cpp:126`, `wrapper/main.cpp:28-33`):
`m_degree = deg>=0 ? min(file_degree, deg) : file_degree`; only the first
`(m_degree+1)^2` rows are read (valid because rows are degree-major). With no
`-c`: degree 0, pole (0,0), coeff 0 (deformation is a no-op, `m_coeff[0..2]=0`).
`main` then does `deg = SR->degree(); if (deg0==-1 || deg0>deg) deg0=deg;` and
loops `for i = deg0..deg: SR->deform(i)`; output file names get the suffix `i`
only when `deg0 < deg`. `deform(d)` uses `d` clipped to `[0, m_degree]`
(negative -> 0). In `reconsCoord` the sum runs over `i < (d+1)^2`; the basis
matrix `m_Y` was built once for the **full** `m_degree` (row stride
`(m_degree+1)^2`) from the *normalised* sphere vertices (float32, the
`--backward` template sphere or the subject sphere), and the coefficient array
has stride `(m_degree+1)^2`. Each vertex' `delta` is accumulated in **float32,
sequentially in i**. Use:

```python
d = read_sh_coefficients(path, max_degree=deg)        # float32, cropped
Y = basis(unit_vertices_float32, d.degree)            # float32 (N,K), K=(d.degree+1)**2
delta = synthesize(Y, d.coeff, degree=cur_deg)        # (N,3) float32 = reconsCoord loop 1
```
The remaining `reconsCoord` steps (pole `sph2cart`, `tan1/tan2` construction
`:149-156`, `axis = unit(axis0 + tan1*d1 + tan2*d2)`, rotation about
`axis0 x axis` by `angle = |(d1,d2)|`, then about `axis` by `d0`, re-normalise)
belong to the remesh port (`Coordinate::rotation/rotPoint/proj2plane/sph2cart`
in Geom.cpp, float32). HSD's counterpart `updateCoordinate` (`HSD.cpp:1238`)
does the same with double `delta`.

## Precision and validation

Oracle harness compiled from MeshLib `Geom.cpp` + `SphericalHarmonics.cpp`
(Apple clang 21, `-O2 -std=c++14`; only build change: `-include cstddef`, because
`Geom.h` uses `NULL` without including a header). No numeric source change.
The oracle is *not* float32-limited: the float overload computes in double, so
the float32 oracle values are the correctly rounded double values.

Metric: elementwise `|a-b| <= atol + rtol*|b|`.
* float32 output: `rtol=1e-5, atol=1e-7` (required <= 1e-5 relative). Measured:
  max abs diff 1.4e-16 over 2,000 seeded random points + 16 special points
  (axes, equator, near-pole 1.5e-8..1e-3 rad) + ico3 vertices, degrees 0, 1, 20 and
  `from=5, 7`; 99.9991% of float32 values bitwise equal (rest differ by at most
  one float32 ulp on values that straddle a rounding boundary).
* float64 output: `rtol=1e-9, atol=1e-11`. Measured max abs 1.3e-13 (max rel
  4.8e-12 on values >1e-3). Differences come from libm `sin/cos/atan2`, and
  from LLVM fp-contraction (FMA) in the C++ build that Numba does not use.
* vs scipy (`sph_harm_y` mapped by the table above): 1e-9 away from the poles
  (`sin(theta) > 1e-3`), <= 4e-8 abs at points within 1e-6 rad of a pole. This
  is **MeshLib's own loss of precision** (`sin(theta)` taken as
  `sqrt(1-cos^2)` of a rounded cos), reproduced exactly (the oracle shows the
  same values), so we deliberately do not "fix" it.

Goldens (committed, no binaries needed): `tests/data/golden/meshlib/sh_deg20_f32`,
`sh_deg12_f64`, `sh_deg9_from5_f32` (`sh.npz` + `command.json`, 72-444 KB each;
328 fixture points, generator `tests/fixtures/sh_points.py`).

Speed: 163,842 points, degree 20 (441 columns): 0.58 s (basis, one thread, incl.
Numba cache load) and 0.06 s for `synthesize`; 289 MB float32 output.
