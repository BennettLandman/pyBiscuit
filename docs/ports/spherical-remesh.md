# Port note: SphericalRemesh (plan row B4.2)

Code: `src/pybiscuit/sphere/remesh.py` (API), `src/pybiscuit/sphere/remesh_cli.py` (CLI
`main(argv)`), subcommand `pybiscuit spherical-remesh` (`src/pybiscuit/cli.py`, passes the
remaining arguments through verbatim). Tests: `tests/test_remesh.py`. Oracle matrix + golden
generator: `scripts/oracles/make_golden_remesh.py`; oracle builds
`scripts/oracles/build_spherical_remesh.sh` (O1a, default arm64 flags) and
`scripts/oracles/build_spherical_remesh_nofma.sh` (new, `-ffp-contract=off`). Goldens:
`tests/data/golden/spherical-remesh/<case>/`.

## Pinned sources

| Repo @ commit | Files / lines |
|---|---|
| ilwoolyu/SphericalRemesh `b677c394e66899c61db85f190f97c4457cd05412` | `src/SphericalRemeshing.cpp` 17-550 (ctor 30-230, `deform` 237-276, `reconsCoord` 278-312, interpolation/median/NN 314-374, `deformSurface` 376-412, `deformData` 414-441, `saveDeformedSurface` 443-475, `saveDeformedProperty` 477-531, `saveDeformedSphere` 533-536, `saveBary` 538-549); `src/SphericalRemeshing.h`; `wrapper/main.cpp` 1-69; `wrapper/PARSE_ARGS.h` 32-50 |
| ilwoolyu/MeshLib `dcef807` | `src/VtkIO.cpp` 187-264 (`VtkIO::save`), `src/Mesh.cpp` 711-736 (`openFile`), 775-791 (`saveFile`); `src/Util/Geom.cpp` 19-190 (`MathVector`), 451-458 (`sph2cart`), 604-607 (`rotPoint`), 624-642 (`rotation`), 820-826 (`proj2plane`), 1783-1798 (`Statistics::median`) |

Dependencies ported elsewhere: `SphereAABB` (`aabb.py`, `AABB_Sphere`) for `closestFace`/`update`,
`spharm.py` (SH basis, coefficient file reader, float32 `synthesize`).

## Flags (wrapper/PARSE_ARGS.h, identical names; `tests/test_remesh.py::test_flags_match_parse_args_h` parses the header)

| Flag | Semantics (as upstream) |
|---|---|
| `-s, --sphere` (required, file) | sphere used for the original parameterisation; carries the input data (`-p`) and the faces that are searched. Vertices are normalised to float32 unit length. |
| `-i, --input` | subject surface (vertex *i* matches sphere vertex *i*); enables remeshed surface output |
| `-c, --coeff` | HSD SH coefficient file (`spharm.read_sh_coefficients`) |
| `-r, --ref` | reference/target sphere; defaults to `-s` (then both copies come from `-s`) |
| `-p, --property` (list) | per-`-s`-vertex text files, `%f` into float32 (fewer values than vertices is fatal, extra values ignored) |
| `--color` | one `r g b` integer triple per **reference** vertex |
| `-o, --output` | remeshed surface VTK |
| `--outputProperty` | prefix: `<prefix>.<name>.txt` (`.dat` with `--ftype`) |
| `--deform` | deformed (normalised) `-s` sphere VTK |
| `--bary` | `<face> <c0> <c1> <c2>` per reference vertex |
| `--ftype` | `int, int16, int32, float, double`; validated case/underscore-insensitively |
| `--nneighbor` | NN data transfer (largest barycentric coefficient) / per-coordinate median of the 3 face vertices for the surface |
| `--deg` (default -1) | cap on SH degree: `m_degree = min(file degree, deg)` |
| `--deg0` (default -1) | first degree of the sweep; `-1` or `> deg` means `deg` |
| `--keepColor` | interpolate the colour map of the `-s` sphere onto the remeshed surface |
| `--quiet` | no progress output |

Defaults, loop and file naming as `main.cpp`: `for i in deg0..deg: deform(i)`, outputs get
the integer suffix `i` appended to the *whole* name (`out.vtk3`, `aug3.curv.txt`,
`bary.txt3`) only when `deg0 < deg`. Without `-c` the degree is 0 (so `--deg/--deg0` do
nothing). CLI exit codes follow CLI11 (105 validation, 106 missing required, 109 unknown
argument); `--help` prints `Spherical Remeshing Tool v1.2.1`.

## Output formats (byte-for-byte as upstream)

* **VTK** (`-o`, `--deform`): `VtkIO::save`, ASCII, no normals, `\n` line endings:
  `# vtk DataFile Version 3.0`, `vtk output`, `ASCII`, `DATASET POLYDATA`,
  `POINTS n float`, vertices `%.9f %.9f %.9f` (float32 value as double, fixed), `POLYGONS m 4m`,
  `3 a b c` lines, blank line. `-o` appends (`saveDeformedSurface`):
  `POINT_DATA n` / `COLOR_SCALARS colors 3` / `%f %f %f` (= channel / 255.0 as double) when colours
  exist, then `POINT_DATA n` (only if no colours were written) / `FIELD ScalarData k` and for
  each property `<name> 1 n float` + `%f` per line + blank line. The FIELD name is the file name
  minus its last 4 characters, after the last `.` (the `_` fallback is **not** used here).
* **Property text** (`--outputProperty`): `%f\n` per vertex. Name = file name minus 4
  characters, after the last `.`; if there is no `.`, after the last `_`; if neither, whole name.
  Directory components are not stripped: `dir.x/plain.txt` -> `x/plain`.
* **`--ftype`**: raw little-endian `fwrite`: `float` float32; `int16` `(short)x`; `int`/`int32`
  `(int)x` (C truncation toward zero); `double`. File extension `.dat`.
* **Bary**: `%d %f %f %f\n`; coefficients are the float32 values of the radial projection,
  printed as double (`-0.000000` occurs).

## Behaviour quirks reproduced (all observed against the oracle)

1. **`--ftype Float|INT_16|DOUBLE`**: CLI11 1.9.1 accepts the spelling (ignore-case /
   ignore-underscore) but passes it on unchanged, and `strcmp` against the lower-case names then
   fails, so *empty* `.dat` files are written. The port reproduces this (`check_ftype` returns
   the string verbatim).
2. **Parcellation call**: `SphericalRemesh -p data/out.mat.1.txt ... -r <native> -s <ico>
   --outputProperty data/out.mat --quiet`: the outputs `data/out.mat.<k>.txt` have the same names
   as the inputs, i.e. they **overwrite the inputs** (read first). Named tests
   `test_parcellation_call_golden`, `test_oracle_matrix[parcellation-*]`. Note `-s` is the
   icosahedral sphere (where the network output lives) and `-r` the native sphere.
3. `--color` without `--keepColor` is only visible when no `-i` is given: `deformSurface` clears
   `m_color` before every remeshing, repopulating it only for `--keepColor` + barycentric. Without
   `-i` the unmodified reference mesh (raw coordinates, *not* normalised) is saved by `-o` with the
   colour map from the file. `--keepColor` indexes the colour map by `-s` vertex id although it is
   read with one row per reference vertex; if the file is shorter than the `-s` mesh the port
   raises (C++: out-of-bounds read).
4. `--nneighbor` is NN for data (strict `>` comparisons; exact ties fall through to the third
   vertex), but **median of three** for surface coordinates, and no colour interpolation.
5. The tangent basis in the constructor is built from `Vector(fmean, m_tan1)` = the *unprojected*
   second sample minus the pole (the `proj2plane` result in `m_tan2` is overwritten), so `tan1` is
   not exactly orthogonal to the pole axis. The C++ keeps `const float*` into destroyed temporaries
   (`...unit().fv()`); with clang -O2 the values survive, and the port reproduces the observed
   result (bit-identical with `-ffp-contract=off` for pole != 0, `tests: coeff_deg0`).
6. The `backward` constructor argument exists (never set by `main`); implemented, untested.
7. `--deform` writes the deformed *and re-normalised* sphere (`m_sphere_subj`), also for
   `deg0..deg == 0` where it is the (twice normalised) input sphere.

## Float32 and precision decisions

* Vertices, coefficients, SH basis (computed in double inside the MeshLib float overload, rounded to
  float32), the `delta` sums (sequential float32), all of `reconsCoord`, `MathVector` operations
  (including `unit()` with `s > 0` test), normalisation, interpolation
  `a*c0 + b*c1 + c*c2` (left to right) and colour `(int)` truncation are float32 exactly as in C++.
  Numba kernel `_recons_all`; `sin`/`cos` are evaluated in double and rounded to float32.
* Input text is parsed as double then rounded (`atof` + cast for VTK, `fscanf %f` for properties);
  the only possible difference to direct decimal->float32 rounding is a double-rounding on exact
  decimal midpoints, which did not occur in any test.
* Closest face / barycentric coefficients: `SphereAABB.closest_face` (float32 throughout).
* The vertices of the `-s` sphere are normalised before the tree is built, the face normals
  used for pole detection are computed from those (upstream computes them from the raw file; the
  direction, which is all that is used, is the same).

## Validation

Oracle: upstream `SphericalRemesh` b677c39 + MeshLib dcef807, Apple clang 21.0.0, `-std=c++11 -O2`,
two builds: **fma** (`oracles/spherical-remesh`, arm64 default fused multiply-add) and **nofma**
(`oracles/spherical-remesh-nofma`, `-ffp-contract=off`, closest to the x86-64 SIF; this one produced
the committed goldens). 35 cases x 2 builds (`pytest tests/test_remesh.py`): property resampling
(ico down/up, ico6, native <-> ico), parcellation call, NN, surface output (+colours, +keepColor),
`-c` with `--deg 0/2`, sweeps (`--deg0/--deg`), deform-only, no `-r`, all `--ftype`.

Comparison: header/face lines identical; numbers `|a-b| <= atol + 1e-5 |b|` with atol 1.1e-6 for
`%f` text, 1e-6 for `%.9f`, 5e-7 for float32 `.dat`; rows with a different closest face (ties) are
exempt and counted.

* **nofma**: 84 files compared, 60 byte-identical; the other 24 are all SH-deformed outputs, where
  `deformed.vtk` differs in the last float32 digit on ~3 % of vertices (<= 1.2e-7, 1 ulp; cause: the
  float32 SH sum / libm `sin`,`cos` of the system vs numba's), with identical faces and
  barycentric coefficients to <= 1e-6 (<= 1.0e-5 on the 163,842-vertex ico7 case). Worst
  non-bary difference: 1.0e-6 absolute (one unit of the 6th decimal; 1.2e-5 relative on values
  around 0.05). Tie count: 0 in every case. Non-SH cases (all `prop_*`, `native_*`, `nneighbor*`,
  `surface*`, `color_*`, `ftype_*`, `parcellation`, `names`): all files byte-identical, including
  every `.dat`.
* **fma** (the build the O1a goldens came from): property/coordinate values agree to the same
  tolerance, but wherever a reference vertex coincides with a sphere vertex (every ico(L-1) vertex
  in ico L, i.e. 486 of 2562 rows in `prop_up`; 8306 of 40962 in `prop_up_ico6`) the fused
  arithmetic returns `1 - 6e-8` instead of `1`, so (a) the closest face of those points can be a
  different one of the vertex's neighbours (counted as ties: 48 / 486 / 8306 / 2031 depending on the
  case; both faces give the same 3-D point, `test_oracle_bary_ties_are_geometric_ties`), (b) `(int)`
  truncation (colours, `--ftype int*`) is off by one level on up to 33 % of those rows. The test
  checks the off-by-one bound (`trunc_slack`) instead of equality for the fma build.
* ico7, native sphere, degree-20 deformation, 2 properties (`test_oracle_ico7_deg20`): port
  2.4 s vs oracle 4.2 s end to end (`-c` with 441 SH terms, 163,842 reference vertices); max
  coefficient difference 1.1e-5 (float32 conditioning of an ico7 face: one ulp of a deformed vertex
  is 1e-5 of its edge), resampled values 1.3e-5 relative.

Goldens committed (all from the nofma build, <= 813 KB each): `prop_down`, `native_to_ico`,
`nneighbor`, `surface`, `color_nokeep_noinput`, `color_keep`, `coeff_full`, `coeff_sweep`,
`ftype_float`, `ftype_int16`, `parcellation` (+ the two O1a goldens `down5to4`, `up3to4`, tested
with tolerance because they were produced by the fma build).

## Deviations

* Only `.vtk` input is read (MeshLib also reads MNI `.obj`; the CLI documents vtk only). Binary VTK
  input is accepted (MeshLib's reader is ASCII-only).
* Failure modes that are undefined behaviour or `exit(1)` upstream (no face found for a point,
  truncated property/colour/coefficient file) raise `RuntimeError`/`ValueError`.
* `(short)`/`(int)` casts of out-of-range floats are C++ UB; numpy's cast is used.
* The `Float`-spelling quirk and the parcellation overwrite are kept deliberately.
