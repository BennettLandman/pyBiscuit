# Port note: AABB_Sphere point location (plan row B2.5)

Python: `src/pybiscuit/sphere/aabb.py` (`SphereAABB`, `barycentric`, `cart2sph`, `closest_face`).
Tests: `tests/test_aabb_sphere.py`. Oracle harness: `scripts/oracles/harness_aabb.cpp`
(build: `scripts/oracles/build_harness_aabb.sh`; goldens: `scripts/oracles/gen_golden_aabb.py`).

## Source (MeshLib commit `dcef807b488eb10d81d9608cf9fa7d06b8ed23e8`)

| Python | C++ |
|---|---|
| `SphereAABB.__init__`, `_face_boxes`, `_build_tree` | `AABB_Sphere(const Mesh*)`, `initTree`, `boundingBox(const Face&)` (AABB_Sphere.cpp 21-31, 275-288, 314-473 `construction`, 475-498 `boundingBox(Face)`) |
| `_node_boxes`, `_refresh`, `update` | `update`, `updateTree`, `boundingBox(node*)` (AABB_Sphere.cpp 229-273 `update`, 500-549 `boundingBox(node)`/`updateTree`) |
| `_find_poles` | pole search in `update()` (AABB_Sphere.cpp 229-273) |
| `_search_one`, `candidates` | `searchTree(phi, theta, node*, ...)` (298-312) and the public `searchTree(const float *p, ...)` (290-296) |
| `closest_face(points)` | public `closestFace(const float*, float*)` (38-60, escalation loop, 9 trials) |
| `closest_face(points, eps1, eps2)` | private `closestFaceBool` (62-163); `poles=False` = private `closestFace(v, coeff, xcand, eps1, eps2)` (165-227) |
| `_eval_face` | candidate loop body shared by both |
| `_bary32/_bary64`, `barycentric` | `Coordinate::cart2bary` float / double (Geom.cpp 469-610) |
| `_dpoint2tri` | `Coordinate::dpoint2tri(t0,t1,t2,p)` (Geom.cpp 900-1137) |
| `_cart2sph`, `cart2sph` | `Coordinate::cart2sph(float*)` (Geom.cpp 443) |
| `_face_normals` | `Face::updateFaceNormal` (Mesh.cpp 414) |

## Semantics (exactly as upstream)

* Everything is **float32** (MeshLib `Vertex` is `float[3]`). Constants: `eps = -1e-5f`,
  `rad = (float)0.0174533`, `PI` double (`2*PI` added in double, stored back to float).
* Query -> `(phi, theta)` = (azimuth, **elevation**). Faces get `(phi, theta)` boxes from
  their vertices; a face spanning more than pi in azimuth is stored as
  `[maxphi, minphi + 2pi]` (wrap box). A second search with `phi + 2pi` is done iff
  `phi + 2pi <= m_maxPhi`. Upstream resets `m_maxPhi` to `2pi` in the last `updateTree(root)` call, so
  for every tree with more than one node this is simply `phi <= 0` (verified from the
  harness dump: `maxPhi = 6.28318548`). A single-leaf tree could raise it; ported.
* Tree: `construction()` ported literally, including: `n == 1` leaf, **`n == 2` always split
  into two leaves**, median of the 2n interval end points as threshold, the
  `score = na*na + nb*nb` (int, 32-bit wrap) vs `(float)na*na + (float)nb*nb` comparison, the
  fallback `dim 0 / threshold 0` when no dimension qualifies, straddlers distributed
  alternately to the smaller side, leaf when a side is empty. The leaf partition is identical to the
  upstream tree (checked with a `leaves` dump of the harness on five meshes, e.g. ico5: 20480 leaves
  identical, ico7: 327,680). Memory layout differs (arrays, iterative build).
* A candidate face is valid iff `r1 >= r0 + eps` (query not behind the plane, `N = unit((b-a) x (c-b))`)
  and the radial projection `v*(r0/r1)` has all barycentric coordinates `>= eps`. Winner = smallest
  `dpoint2tri(a,b,c,v)` (Eberly region tree, float32); equal float distances go to the **lowest
  face index** (`dist == min && index > cand`). The `xcand` overload sorts candidates ascending and
  uses strict `<`, which selects the same face. Bary = `cart2bary` of the radial projection, `err = 0`
  (no snapping), so values can be slightly negative.
* Public overload: trial 0 with enlargement 0 (plus pole faces if both poles found: `theta >= maxTheta`
  adds +z face, `theta <= minTheta` adds -z face), then `rad = 1 degree * 2^k` for k = 0..8 applied to both
  angles, no poles. The `visited` array of upstream only skips faces that already failed, so re-evaluating
  them (not done here: results are order independent) cannot change the answer.
* `update()`: tree topology is **not** rebuilt (as upstream); boxes, `maxPhi`, and poles are recomputed.
  Pole detection uses the stored face normals. SphericalRemesh/HSD move vertices with
  `Vertex::setVertex` and never call `Mesh::updateNormal`, so the normals are those of the
  construction-time mesh. Default `refresh_face_normals=False` reproduces that; `True` equals calling
  `updateNormal()` first (both validated against the harness; the difference only shows in
  which faces are chosen as pole faces, e.g. ico5 `(3, 19750)` vs `(1, -1)` after a deformation).
* No face found: upstream leaves `coeff` untouched (stale) and returns -1; here face `-1`, bary `NaN`.
* **Queries must lie on or outside the mesh surface** (`r1 >= r0 - 1e-5`). A unit query against a
  deformed sphere with radius > 1 locally is "behind" the plane and is not found. This also makes
  such queries slow (all 9 escalation trials, ~100 ms per 1e4 queries on ico7); upstream behaves
  the same way.
* Not ported: `AABB.cpp` (generic 3D tree, not used by AABB_Sphere) and the dead private
  `searchTree(phi_min, ..., trace)` declaration (no definition upstream).

## Precision

`barycentric(..., dtype=np.float32)` and the whole query path reproduce the C++ operation order
(left-to-right dot products, `N.unit()` in place, `1 - c0 - c1`). Built with
`-ffp-contract=off` (no FMA), results are **bit identical** to the harness on every case below
*on the same machine* (same libm; see Determinism for other platforms). Triple-product barycentrics in float32 amplify 1-ulp
input differences to ~1e-6 (observed when vertices were rounded float64 -> text -> float differently),
therefore all tests round vertices to float32 before writing the oracle VTK.

### Determinism

Goal: the discrete decisions (box tests, wrap decision, validity, nearest face, tie-break) must
not depend on the platform. Audit of every site where float32 rounding drives a decision:

| Site (`aabb.py`) | Operations | Platform-independent? |
|---|---|---|
| `_atan2f` / `_cart2sph` (`phi`, `theta` of vertices and queries) | libm `atan2f`, `sqrt`, `x*x+y*y` | **No: `atan2f` only.** Apple libm, glibc and UCRT differ by 1 ulp for ~6 % of arguments, and Apple returns pi - 1 ulp for `atan2f(+-0, -x)` where glibc/UCRT return the correctly rounded pi. Kept as the platform libm (the upstream semantic is float32 `atan2f`; a correctly rounded or hand-written replacement would change the arm64 results). `sqrt` and `+ - * /` are IEEE correctly rounded. |
| `_face_boxes` wrap decision `maxphi - minphi > pi` (double pi) | compare of two `atan2f` results | Inherits the above. A face with a vertex on the +-pi cut (every pole face) flips between box `[0, pi]` (arm64) and wrap box `[pi, 2 pi]` (x86): this is the cause of the `bool`/`xcand` golden failures on CI. |
| `_node_boxes` | float32 `min`/`max` | Yes (selection only, no arithmetic). |
| tree build `_build_tree` (median, `score` int32 wrap vs float compare) | integer / float32 compares of box values | Same inputs as above; identical given identical boxes. |
| `_search_one` / `_candidates_all` box test `ph >= box - eps1 ...` | float32 subtract/add and compare | Yes given identical `phi/theta/box`; ties at a bound are exactly the `atan2f` ulp cases. |
| `_find_poles`, `_face_normals` | float32 cross/sqrt/divide, compares | Yes (IEEE ops, fixed left-to-right order). |
| `_eval_face` validity (`r1 >= r0 + eps`, bary `>= eps`) and `_bary32` | float32 `* + - / sqrt`, fixed order | Yes. |
| `_dpoint2tri` distance and the lowest-index tie-break (`dist == best`) | float32 IEEE ops | Yes: exact ties come from identical inputs. |
| `_closest_all` (`prange`) | one independent query per iteration, no reductions | Yes (no cross-thread accumulation). |

No `np.sum`/`np.dot` on vectors, no `fastmath`, no `parallel` reductions anywhere in the query path
(all dot/cross products are explicit left-to-right scalar expressions). FMA contraction: Numba
sets no `contract`/`fast` flags and emits no `fmuladd`, so LLVM does not fuse `a*b+c` (checked on
arm64, where FMA is baseline: 0 `fmadd/fmla` in the compiled `_closest_one`, `_find_poles`,
`_face_normals`, `_face_boxes`; x86-64 codegen follows the same IR rules). A test asserts that the
module never mentions `fastmath`.

Public `cart2sph()` now calls the same jitted kernel as the tree (it used NumPy's float32
`arctan2`, a different implementation that disagreed with the tree in ~6 % of the values).

Root cause of the CI failures (reproduced locally by recompiling the module with a correctly
rounded `atan2f`, test `test_golden_with_other_libm`; it gives exactly the CI signature: 4 face
differences `[3408 3489 3839 3926]` in `closest`, 4 in `bool`/`xcand`, 22 candidate lists in
`search`):

* `closest` (4 queries: edge midpoints and the (-1, -1e-9, 0) cut query): the query lies on the
  surface to within float32 rounding (distance 1e-10..1.5e-8), two faces are tied to that noise
  and the box/trial path differs by an ulp; the point-triangle distances differ by <= 1e-8.
* `bool`/`xcand` `eps=0.02:0.01` (4 queries near the south pole, golden finds the pole face
  1198, x86 finds none): the pole face has a vertex at phi = pi; Apple's `atan2f` gives
  3.1415925 (box `[0, 3.1415925]`), glibc 3.1415927 > pi so the face is stored as the wrap box.
* `search` (22 lists): same wrap-box difference changes the union box of the leaves (and
  ancestors) containing pole faces; whole leaves enter/leave the candidate list.

Test policy (`tests/test_aabb_sphere.py`; goldens were made on macOS arm64):

* faces: `assert_same_choice_modulo_ties` with the margin recomputed in float64
  (`|dist(chosen) - dist(golden)| <= 4 ulp(1) = 4.8e-7`; observed gaps <= 1e-8), or, for pairs
  not tied in distance, mismatches explained by a box decision within `4 ulp(2 pi) = 1.9e-6` of
  a node bound or of the wrap threshold (`BoxAmbiguity`, recomputed in float64, both wrap
  variants of the sensitive faces);
* candidate lists: sets, differing members must lie under a box-ambiguous node
  (`assert_candidate_sets_close`);
* barycentrics where the face agrees: `<= 1e-6` abs (not loosened);
* synthetic untied mismatches must fail (`test_untied_*`);
* exact equality is kept as `test_golden_bitexact` (`pytest -m bitexact`, arm64 only);
  the live-oracle tests (harness built and run on the same machine, same libm) stay exact.

## Validation

See `docs/reports/_fragments/b25.md` for counts, ties and timings. Goldens:
`tests/data/golden/meshlib/aabb_*` (closest, bool/xcand eps overloads, searchTree candidate lists,
deformed + `update()` with and without normal refresh).
