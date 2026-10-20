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
`-ffp-contract=off` (no FMA), results are **bit identical** to the harness on every case below;
arm64 libm `atan2f` vs numba's `atan2` showed no difference. Triple-product barycentrics in float32 amplify 1-ulp
input differences to ~1e-6 (observed when vertices were rounded float64 -> text -> float differently),
therefore all tests round vertices to float32 before writing the oracle VTK.

## Validation

See `docs/reports/_fragments/b25.md` for counts, ties and timings. Goldens:
`tests/data/golden/meshlib/aabb_*` (closest, bool/xcand eps overloads, searchTree candidate lists,
deformed + `update()` with and without normal refresh).
