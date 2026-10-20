# Port note: icosphere generators (plan row B4.1)

Code: `src/pybiscuit/sphere/icosphere.py`; tests: `tests/test_icosphere.py`.
Public API: `icosphere(level, scheme="biscuit") -> Mesh`, `icosphere_counts(level)`,
`quantize_shipped(vertices, radius=100.0)`, `SCHEMES`, `SHIPPED_SPHERES`, `MAX_LEVEL` (=7).
Counts: `n_faces = 20*4**L`, `n_vertices = 10*4**L + 2` (L=7: 163,842 / 327,680).

## Sources compared

| Shipped file | Level | Verts / faces | Scheme that reproduces it |
|---|---|---|---|
| `sphere/sphere_5120.vtk` (R = 100) | 4 | 2,562 / 5,120 | `biscuit_ico4` |
| `sphere/sphere_20480.vtk` | 5 | 10,242 / 20,480 | `biscuit` |
| `sphere/sphere_81920.vtk` | 6 | 40,962 / 81,920 | `biscuit` |
| `sphere/sphere_327680.vtk` | 7 | 163,842 / 327,680 | `biscuit` |
| `parc/mesh_files/icosphere_5.vtk` (R = 1) | 5 | 10,242 / 20,480 | `ugscnn` |
| `parc/mesh_files/icosphere_6.vtk` | 6 | 40,962 / 81,920 | `ugscnn` |

Faces are identical (same array, same order, same winding) for all six files.

## Scheme "biscuit" (FreeSurfer-style nested ico)

Derived from the files, not from source (none is shipped for these meshes).

* Base icosahedron: vertex 0 = +z pole; 1-5 upper ring (z = +1/sqrt5);
  6-10 lower ring (z = -1/sqrt5); 11 = -z pole. Ring angles for levels 5-7:
  upper `-72 + 72k` deg, lower `-108 + 72k` deg (so vertex 2 = (0.8944, 0, 0.4472),
  vertex 1 = (0.2764, -0.8507, 0.4472)). `sphere_5120` uses the same set
  rotated by +72 deg (vertex 1 on the +x meridian).
* Base parent triples (a, b, c), 20 rows: see `_BASE_FACES` (`[0,4,3],[0,5,4],[0,1,5],...`).
* Subdivision: scan parents in list order; within a parent, edges `ab, bc, ca`;
  an edge gets the next free vertex id the first time it is met (midpoint
  renormalised to the sphere). So level L vertices are a prefix of level L+1.
* Children of parent `(a,b,c)`: `(a,ca,ab)`, `(ab,bc,b)`, `(ca,bc,ab)`, `(ca,c,bc)`;
  their winding is the reverse of the parent triple (outward-facing), so the
  children are flipped to `(x,z,y)` before being used as the next level's parents.
* Face order differs between the two families:
  * `biscuit_ico4` (`sphere_5120`): the four children of parent *i* are rows
    `4i..4i+3` (hierarchical).
  * `biscuit` (levels 5-7): two passes. Rows `0..P-1` = the `a`-corner child of
    every parent in order; rows `P..4P-1` = for each parent
    `(b-corner, centre, c-corner)`. Applied at every level from 1 up (the
    level-k vertex order depends on the level-(k-1) face order, so the rule
    must be applied recursively; verified for 5, 6, 7).

Levels 0-3 have no shipped reference. `icosphere(L)` for L < 4 uses the same
rule as `biscuit` (identical to level 5-7 prefixes); `biscuit_ico4` is only
validated at level 4.

## Scheme "ugscnn"

Port of `reference/repos/SphericalLabeling/meshcnn/mesh.py` (`icosphere`,
`subdivide`, `normalize`, `icosahedron(upward=False)`; MIT). The imported
`mesh_utils.unique_rows` is not in the clone; it is the trimesh function. The
new-vertex order is *not* the first-seen order: midpoints are rounded to 8
decimals, viewed as `np.void` rows and sorted bytewise by `np.unique`
(little-endian int64 bytes, lexicographic), and the first occurrence of each
sorted unique row is kept. We emulate that byte order with a big-endian view +
`np.lexsort` (identical results, no `np.void`). Tie-keying uses 1e-8 rounding,
so the order is robust to float noise but this is trimesh-version behaviour.
UGSCNN spheres have a different base orientation/labelling and order from
BISCUIT's. They are the same polyhedral geometry rotated; the exact vertex
permutation was not computed (not needed by any consumer; B4.3 uses the UGSCNN
mesh only for its own pkl operators).

## Numeric precision / deviations

* Generator vertices are float64 on the unit sphere.
* The BISCUIT files hold coordinates quantised to **4 decimals of the unit
  sphere** (`round(x,4)*100`, then stored as float32 and printed with 9
  decimals). They are therefore *not* on the sphere: radius spans
  99.9923..100.0080. Max |gen*100 - file| = 0.00503 (half a step).
* `quantize_shipped(v)` = `round(v*1e4)/1e4*100` reproduces the files except for
  a few coordinates whose exact value is within ~1e-7 of a rounding tie and which
  the original (unknown float32/other arithmetic) rounded the other way, by
  exactly one step (0.01 mm at R=100): 8 / 10 / 57 / 277 of 3N coordinates for
  levels 4/5/6/7 (0.10 / 0.03 / 0.05 / 0.06 %). Float64, float32,
  rounded-base-constants, round-half-away, iterative-rounding and R=100
  arithmetic variants were tried; none removed them (iterative rounding makes
  it much worse). Everything else matches to the float32 ULP of the stored value
  (<= 6.7e-6 at R=100, i.e. below 1e-7 on the unit sphere).
* So the plan's "max |dxyz| <= 1e-6" cannot be met on the BISCUIT files by any
  exact generator: for those coordinates it is 0.01 (the quantisation step) and
  for the rest it is bounded by the float32 ULP (7.6e-6 at |x| in [64,128)).
  Face lists are exact. On `icosphere_{5,6}.vtk` (6 decimals printed) the max
  diff is 5.0e-7 (<= 1e-6): full pass.

## Template sphere finding

All twelve `reg/{hcp1058,mb101,oasis45}/{lh,rh}.{white,inflated}.vtk` have
163,842 vertices and a face array that is **exactly equal** (same rows, same
order, same winding) to `sphere/sphere_327680.vtk`. So `sphere_327680.vtk` is
the ico7 template; vertex *i* of every `reg/` surface corresponds to vertex *i*
of the sphere. `icosphere(7)` reproduces it. Use it as the single source of
topology for ico7 resampling; do not ship the file.

## Validation

`pytest tests/test_icosphere.py`; measurements in
`docs/reports/_fragments/icosphere.md`.
