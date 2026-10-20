# Port note: MeshLib geometry (plan row B2.4)

Code: `src/pybiscuit/mesh/geometry.py` (B2.4a), `src/pybiscuit/mesh/curvature.py` (B2.4b).
Tests: `tests/test_geometry.py`. Oracle driver: `scripts/oracles/harness_geometry.cpp`
(`build_harness_geometry.sh`, goldens by `make_golden_geometry.py`).
Source: `ilwoolyu/MeshLib` commit `dcef807b488eb10d81d9608cf9fa7d06b8ed23e8` (file/line ranges below).

## Normals/areas

Public API (all accept a `Mesh` or `(vertices, faces)`; vertices are cast to float32 first):

| Function | Returns | MeshLib source |
|---|---|---|
| `vertex_neighbor_lists(faces, n_vertices)` -> `NeighborLists(indptr int64, indices int32)` | CSR ring lists in MeshLib order | `Mesh::connectivity`, `src/Mesh.cpp` L819-L911 |
| `face_normals(vertices, faces)` | float32 (M,3) | `Face::updateFaceNormal`, `Mesh.cpp` L414-L420 |
| `vertex_normals(vertices, faces)` | float32 (N,3) | `Mesh::updateNormal`, `Mesh.cpp` L738-L776 (duplicate of the loop in `setMesh`, L660-L705) |
| `face_areas(vertices, faces)` | float32 (M,) | inline `P.cross(Q).norm()/2` in `SurfaceUtil::sphere`, `SurfaceUtil.cpp` L319 (MeshLib has no face-area function; the `/12` is not applied) |
| `vertex_areas(vertices, faces, neighbors=None)` | float64 (N,) | `GeodesicA::vertexArea`, `src/Geodesic/GeodesicA.cpp` L401-L424 (private there; the only vertex area in MeshLib) |

Semantics worth knowing:

* **Face normal** = `unit(cross(V1-V0, V2-V1))`, float32, zero stays zero. A vertex normal is
  *not* the mean of face normals: each face adds its **unnormalised** cross product
  `(V0-V1) x (V1-V2)` (twice the area vector) times the **interior angle** at that vertex
  (`acos` of the clamped dot of the unit edges), and the sum is normalised.
  Isolated vertices get (0,0,0). `Mesh::openFile` always recomputes normals with this routine,
  so normals stored in a VTK file are ignored.
* **Neighbour lists** (`Vertex::list`) are ordered rings: start at the vertex's first face (in face
  order), follow `next` links; on an open fan the closing vertex is appended (so a boundary vertex
  has `#faces + 1` neighbours); several fans (non-manifold) are concatenated. The order matters
  because float32 sums are order dependent; it is reproduced exactly (verified on closed and
  open meshes, 163,842-vertex `lh.white`).
* **Vertex area** is 1/3 of the incident triangle areas, *not* Voronoi/mixed area. It iterates
  over consecutive ring pairs `(j, j+1 mod n)`, so on a **boundary** vertex it also counts the
  spurious wrap-around triangle (last, first neighbour). A lone triangle gives 2x its area/3 at
  every corner. Ported as is. Edge vectors and norms are float32 (MeshLib `Vector`), `acos/sin`
  and sums are double.
* The `SurfaceUtil::sphere` / curvature routines, `sphere_map` etc. belong to B2.4b.

## Smoothing

All overloads of `SurfaceUtil::smoothing` (`SurfaceUtil.cpp` L18-L36 public, L420-L520 private workers).
MeshLib mutates the mesh/array in place; the Python functions return new arrays.

| C++ | Python |
|---|---|
| `smoothing(Mesh*, int iter)` | `smooth_vertices(vertices, faces, iterations, sigma=None, neighbors=None)` |
| `smoothing(Mesh*, float sigma, int iter)` | `smooth_vertices(..., iterations, sigma=sigma)` |
| `smoothing(const Mesh*, int iter, float* data)` | `smooth_data(faces, data_float32, iterations)` |
| `smoothing(const Mesh*, int iter, double* data)` | `smooth_data(faces, data_float64, iterations)` |

The C++ overloads have no default arguments (`iter` is required); the Python `iterations`
defaults to 1 only for convenience. Iteration semantics:

* Each pass is a Jacobi update: all new values are computed from the previous pass, the vertex
  itself is **not** in the average (`newv = mean(neighbours)`), neighbour lists are those of the
  input topology and never change. After each pass MeshLib calls `updateNormal()`; the Python
  function leaves normals to `vertex_normals(result, faces)` (smoothing itself never reads them).
* `sigma` variant: weight of neighbour `j` is `normal_pdf(dist, 0, sigma)` (`Geom.cpp` L1749), with
  `dist` the float32 edge length and `sigma` cast to float32; result is the weighted mean. The pdf
  is `float(exp_f32(-x*x/(2*sigma*sigma)) / (double(sigma)*sqrt(2*PI)))` (float exp, double
  denominator), reproduced. If all weights underflow (tiny sigma) the result is NaN (0/0) exactly
  as in C; numba is compiled with `error_model="numpy"` so there is no ZeroDivisionError.
  The pdf normalisation constant cancels in the weighted mean but is kept for rounding parity.
* Data smoothing: dtype dispatch float32 -> float32 accumulation (the `float*` overload),
  float64 -> double; other dtypes are converted to float64. A vertex with no neighbours gets
  NaN (0/0). Out-of-range or non-(N,3) input raises.
* There are no boundary special cases: boundary vertices average over their (open-fan) neighbours
  and smoothing shrinks the mesh like any umbrella operator.

## Precision table

| Quantity | MeshLib type | Port | Notes |
|---|---|---|---|
| Vertices (VTK read) | `float` (`atof` then cast) | float32 | harness input written as `%.9g` of float32 values: exact round trip |
| Face / vertex normals | `float`, `acosf`, `sqrtf` | float32 | `acos`/`sqrt` on float32 stay float32 |
| Face areas | `float` | float32 | |
| Vertex areas | float edges, double `acos`/`sin`/sum | float32 edges, float64 rest | not a "double path": the float32 edge norms dominate; compared with the 1e-5 criterion |
| Smoothed vertices | `float` sums, divide by `float(nn)` | float32 | summation order = neighbour-list order |
| `normal_pdf` | float exp, double denominator | same | |
| Data smoothing | `float` / `double` | same | |

## Validation

* Oracle: MeshLib at `dcef807` linked into `scripts/oracles/harness_geometry.cpp`.
* **Floating-point contraction.** arm64 clang (default `-ffp-contract=on`) fuses `a*b+c`; the
  x86-64 image binary (baseline SSE2, no FMA) does not. Against a MeshLib built with
  `-ffp-contract=off` the port is **bit identical** (max error 0 for every output on all fixtures,
  `lh.white` and the committed goldens). Against the default arm64 build (`oracles/meshlib`)
  the largest relative error is 1.2e-6 (`vertex_areas`, ellipsoid; 7.2e-7 on `lh.white`), and
  <= 6e-7 for everything else; a very stiff sigma (0.25 of the mean edge length) amplifies
  to 8.7e-5 (4 iterations), 2.7e-4 on the ellipsoid. That is rounding amplification of
  `exp(-d^2/2s^2)`, not a porting difference. Tests use moderate sigma (~1.5 and ~0.75 mean edge).
* Committed goldens: `tests/data/golden/meshlib/geometry_{ico2,ico3,perturb,cut}/` (0.06-0.24 MB each,
  built with `-ffp-contract=off`, args in `command.json`), checked without binaries.
* Timings (M-series, warm numba): icosphere(7) 327,680 faces: neighbour lists 0.09 s, vertex
  normals 5 ms, 10 mean-smoothing passes 8 ms, 10 sigma passes 31 ms. First call includes JIT
  (`cache=True`, so only once per environment).

## Curvature

Code: `src/pybiscuit/mesh/curvature.py` (B2.4b). Tests: `tests/test_curvature.py`. Oracle driver:
`scripts/oracles/harness_curvature.cpp` (goldens by `scripts/oracles/make_golden_curvature.py`,
`tests/data/golden/meshlib/curv_*`). Source: same MeshLib commit `dcef807`.

| C++ (`SurfaceUtil.cpp` unless noted) | Python | Lines |
|---|---|---|
| `tensor(mesh, Tv)` (private) | `tensor(mesh) -> float64 (N,3,3)` | L522-L624 |
| `smoothingTensor(mesh, Tv, iter)` (private) | `smoothing_tensor(T, mesh, iterations=)` | L626-L694 |
| `principalCurvature(Tv, n, ...)` (private) | `principal_curvature(T) -> Curvature(cmin, cmax, umin, umax)` | L696-L761 |
| `curvature(..., double*, ..., nIterSmoothingTensor)` | `curvature(mesh, n_iter_smoothing_tensor=0)` (dtype float64) | L38-L96 |
| `curvature(..., float*, ...)` | `curvature(..., dtype=np.float32)` | L98-L130 |
| `curvature_quad(double*)` / `(float*)` | `curvature_quad(mesh, dtype=...)` | L132-L282 |
| helpers `LinearAlgebra::eig3symmetric`, `echelonEigv`, `Ab`, `gaussElim`; `Coordinate::rotation`, `rotPoint`, `rotPointInv` | private numba kernels | `Geom.cpp` L1816-L1924, L2049-L2099, L624-L662, L604-L622 |

All functions take a `Mesh` or `(vertices, faces)`; the optional `neighbors=` argument reuses a
`NeighborLists` from `geometry.vertex_neighbor_lists`. `Curvature` fields are (N,), (N,), (N,3), (N,3).
`mean_curvature = (cmin+cmax)/2` and `gaussian_curvature = cmin*cmax` are one-liners (what the harness
dumps as mean/gauss); MeshLib has no shape-index routine.

**Algorithm.** `tensor`: for every edge shared by two faces, `T_e = t t^T * beta * len / mean_len`
with `t` the unit edge tangent (float32), `beta` the signed dihedral angle between the two float32
face normals (`acos` of the clamped dot product; sign `+` when `(n1 x n2) . t > 0`, i.e. convex for
outward normals), `len` the (double-promoted float32) edge length and `mean_len` their mean over all
edges in sorted-edge order. `T_e / nNeighbor(v)` is added to both end vertices (so boundary edges,
with a missing face, contribute nothing and the tensor is **scale invariant**, not a 1/mm curvature).
`smoothingTensor`: `iter` Jacobi passes of the plain mean over the vertex and its ring (weight 1 each).
`principalCurvature`: closed-form 3x3 symmetric eigensolver (`eig3symmetric`, trigonometric method,
eigenvectors from the odd `echelonEigv`); `cmax` is the eigenvalue of largest magnitude, `cmin` the one
of middle magnitude (the third, near zero, belongs to the normal direction), then swapped so
`cmin <= cmax`. Because of that rule `umin` is the eigenvector of `cmax` and `umax` that of `cmin`
(names kept from MeshLib). `curvature_quad` fits `x = a y^2 + b z^2 + c yz + d y + e z + f` by normal
equations + Gauss elimination to the sorted-unique 2-ring (incl. the vertex) in a frame obtained by
rotating the float32 vertex normal onto -x; `Dxx=2a, Dxy=c, Dyy=2b`; eigenvalues in closed form;
`cmin` is the eigenvalue of smaller **magnitude** (unlike `curvature`). Sign: positive for convex
(outward) surfaces in both routines (verified: `icosphere(radius r)` gives `cmin, cmax ~ 1/r` with
quad, and flipping the face order negates).

**Quirks reproduced on purpose** (none is "fixed"):

* *Flat vertices* (all nine |T_jk| < 1e-6): `cmin = cmax = 0`; MeshLib then sets `umin[i][0]` and
  `umax[i][0]` three times each and leaves components 1, 2 uninitialised (stack/heap garbage in the
  float overload). The port returns zeros. The tests and the oracle comparison ignore directions of
  flat vertices. Boundary rim vertices of an open mesh can be flat (8 in `cut_sphere(3)`).
* *Ties*: if two eigenvalue magnitudes are exactly equal no branch of the `cmin`/`cmax` selection
  fires, and exactly tied diagonal tensors leave `lambda` unset in `eig3symmetric` (strict `<`).
  MeshLib then leaves the output slot as it was (uninitialised); the port keeps 0. None occurred in
  any fixture or in `lh.white` (`cmin==cmax` count 0).
* `echelonEigv` eliminates only row 1 (`j < n-1`) and has a `float eps = 1e-8` pivot test; it is ported
  verbatim and gives NaN eigenvectors when all three pivots vanish (16 vertices of
  `folded_sphere(0.1,4,2,3)` at `iter=0`; identical NaNs in the oracle). Real eigenvectors of the
  actual tensors are fine (all bit-identical on `lh.white`).
* `curvature_quad` divides by a zero norm when `Dxy = 0` and `Dxx > Dyy` (NaN directions, as in C++).
* Edge list: the C++ keeps duplicates if a neighbour list repeats a vertex (non-manifold meshes);
  the port assumes unique `(lo, hi)` edges (stable sort by key), exact for manifold meshes and for
  open meshes (the extra closing neighbour of a boundary vertex is an existing neighbour).

**Precision.** Vertices, tangents, face normals, vertex normals (via `geometry.py`), dot/cross
products inside `tensor` and the `cp . tan` sign test are float32; edge length, `beta`, `bLen`, all
tensor accumulation, eigen-solves and the quadratic fit are float64, promoted exactly where the C++
promotes (`(float*float) * double`, `float` dot assigned to `double`). The `float*` overloads compute the
double result and cast `cmin/cmax/umin/umax` to float32 at the end (`SurfaceUtil.cpp` L113-L122). Edge
sums over the sorted edge list and per-vertex accumulation follow the C++ order.

**Validation.** Against MeshLib compiled with `-ffp-contract=off` (what the x86-64 image binary
effectively is) the port is **bit-identical**: max abs/rel difference 0.0 on every output (tensor,
smoothed tensor, `cmin/cmax/umin/umax` double and float, mean/Gaussian, quad double and float) for the 5
committed fixtures, and for `reg/mb101/lh.white.vtk` (163,842 vertices; also 0 flat vertices, 0 ties,
0 NaN). Warm timings on `lh.white`: `curvature` 0.2-0.3 s, `curvature_quad` 0.3 s.
**Floating-point contraction warning:** against the default arm64 build in `oracles/meshlib`
(`-ffp-contract=on`, FMA) the float32 face normals differ by an ulp and the dihedral angle
`acos(n1 . n2)` of nearly coplanar faces amplifies that into tensor differences of up to
3.3e-4 of the maximum tensor entry; per-vertex curvature then differs from this port by a median
1.7e-5 (p99 0.9 %, max several times the value for near-zero `cmin`) and 58 % of vertices exceed 1e-5.
The 1e-5 acceptance is therefore only attainable against a non-fused build; the live oracle used by
the tests is `harness_curvature.cpp` compiled with `-ffp-contract=off` (command in its header).
`curvature_quad` is less sensitive (max rel difference between the two builds: 1e-2 on `cmin`, 9e-6 on `cmax`).

**Which MeshProperty flags these seem to correspond to** (hypothesis from flag names only; MeshProperty
is closed, nothing here claims parity): `--tensorsmoothing N` ~ `nIterSmoothingTensor` of `curvature()`
(`smoothing_tensor`); `--surfsmoothing N` ~ `smooth_vertices(mesh, N)` applied to the surface before
curvature; `--datasmoothing N` ~ `smooth_data(faces, curv, N)` applied to the output scalar; the
curvature outputs (mean, Gaussian, principal values/directions) ~ `mean_curvature`, `gaussian_curvature`,
`Curvature.cmin/cmax/umin/umax`. `--prctile`, `--normbefore`, `--normafter`, `--invert`, `-q` have no
counterpart in `SurfaceUtil.cpp` / `Util/Geom.cpp` (percentile clipping, normalisation and sign flip
would have to come from MeshProperty itself; `Statistics::median` is the only order statistic in
`Geom.cpp`). Which of `curvature`/`curvature_quad`, `float`/`double`, and which outputs BISCUIT actually
uses is B3.1 work (needs the orchestrator calls).

**Not ported / out of scope:** the two `SurfaceUtil::smoothing(Mesh*)` helpers (B2.4a), `normal_pdf`
(B2.4a).

## Spherical mapping

Code: `src/pybiscuit/mesh/sphere_map.py` (B2.4b), `sphere_map(mesh, type=0, max_iter=10000) ->
SphereMapResult(mesh, iterations, n_inverted, converged)`; `count_inverted_faces(vertices, faces)`.
Tests: `tests/test_sphere_map.py`; goldens `tests/data/golden/meshlib/spheremap_*`. Source:
`SurfaceUtil::sphere`, `SurfaceUtil.cpp` L284-L418, plus `Mesh::centering`, `Mesh.cpp` L556-L571.
**Port only**: not adopted as the replacement for FreeSurfer's spherical mapping (B5.7); that is
owner question Q3.

Algorithm: centre the vertices (float32 mean), build per-neighbour-slot float32 weights `w`,
normalise each row to sum 1, project every vertex to the unit sphere, then iterate Jacobi updates
`v_i <- unit( [type==0 ? v_i : 0] + sum_j w_ij v_j )` until no face is inverted or `max_iter` passes
(do-while: at least one pass; `max_iter=0` behaves as 1).
Inversion test per face: `((V1-V0) x (V2-V0)) . (V0+V1+V2) < 0` (float32).
`type == 0` ("area preservation"): `w` accumulates `|P x Q|/2/12` (float32) over the corner of every
face, added to the slot of `id2` in the ring of `id1` and vice versa; `type != 0` ("angle
preservation"): for consecutive ring neighbours `x = cos(angle)` (double, from float32 unit
vectors), `weight = max(x/sqrt(1-x^2), 0.1)`, accumulated into the float weight through a double
sum `float(double(w)+weight)`. Note the weights are fixed once (computed on the centred input), not
updated during iteration. The MeshLib function mutates the mesh and returns nothing; `sphere_map`
returns a new `Mesh` (float32 values stored as float64) plus the pass count, the final flipped-face
count and a `converged` flag (no inverted face).

Validation: the port is **bit-identical** to the `-ffp-contract=off` oracle after every checkpoint
`max_iter in {1,2,5,20,100,10000}` for both types, on the three committed fixtures
(ellipsoid, folded sphere a=0.3, perturbed icosphere) and on `lh.white.vtk` of `mb101`
(163,842 vertices): after 1 / 2 / 5 / 20 / 100 passes the inverted-face counts are 75,892 / 75,415 /
73,968 / 68,598 / 52,567 (type 0) and 75,384 / 74,488 / 72,259 / 64,671 / 46,011 (type 1) of 327,680
faces, identical to the oracle; converged with 0 flipped faces after **2,329** (type 0) and **1,882**
(type 1) passes. Timings (M-series, warm): 2.5 s / 1.5 s for the full runs.
Against the default FMA arm64 `oracles/meshlib` build, vertices differ by <= 4.2e-7 absolute (a few
float32 ulps; 25-60 % of coordinates differ in the last bit).
