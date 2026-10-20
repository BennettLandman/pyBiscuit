# MeshConv operators (`pybiscuit.sphere.operators`) — plan row B4.3

Source: `SphericalLabeling/meshcnn/{mesh.py,mesh_utils.py}` (MIT, commit `37fa7a0`);
the copy shipped in the image (`ctf_plain/SurfPostproc/common/parc/meshcnn/`) differs
from the repo only in `ops.py` syntax (not used here) and has the extra
`mesh_utils.py` that the repo clone lacks. Target files: FC v2.2 image
`ctf_plain/SurfPostproc/common/parc/mesh_files/icosphere_{0..5}.pkl`.

## Levels

`icosphere_0..5.pkl` exist (12 to 10,242 vertices). There is **no** level 6 or 7
pickle in the image (upstream `gen_ico.py` would make 0..7); `icosphere_5.vtk`
and `icosphere_6.vtk` exist as meshes only. `build_operators(level)` works for
any level (level 6: 0.17 s) but only 0..5 can be compared with pickles; levels
5 and 6 vertices/faces are compared with the `.vtk` files.

## Stored fields

Dict with keys `V F nv_prev nv_next G L N NS EW F2V M Seq Intp`
(nf = 20·4^l, nv = 10·4^l + 2):

| key | type / dtype | shape | meaning |
|---|---|---|---|
| `V` | ndarray f8 | (nv,3) | unit-sphere vertices |
| `F` | ndarray **int64** | (nf,3) | faces |
| `nv_prev`, `nv_next` | int | | vertex count of previous/next level |
| `G` | `coo_matrix` f8 | (3nf, nv) | libigl gradient; x, y, z blocks stacked |
| `L` | `coo_matrix` f8 | (nv, nv) | cotangent Laplacian (diag negative) |
| `N` | ndarray f8 | (nf,3) | per-face unit normals |
| `NS`, `EW` | ndarray f8 | (nf,3) | `G @ lat` reshaped (order F); `EW = NS x N` |
| `F2V` | `csc_matrix` f8 | (nv, nf) | area-weighted face-to-vertex average |
| `M` | ndarray f8 | (nv,) | diagonal Voronoi mass |
| `Seq` | tuple of 11 int64 arrays | (nv,) each | vertex permutations for the 11 icosahedron-vertex rotations |
| `Intp` | `csc_matrix` f8 | (nv, nv_prev) | parent interpolation (identity at level 0) |

## How regenerated

Pure NumPy/SciPy (no libigl, no torch). The icosphere is a line-by-line port of
`mesh.py` including the hash-based `unique_rows` (np.unique order decides new
vertex numbering, so it is replicated exactly; `icosphere(upward=False)`). libigl
calls are replaced by closed-form NumPy: `grad` (per-face hat-function gradients
`n x e_opp / 2A`), `cotmatrix`, `per_face_normals`, `doublearea`, and mixed
Voronoi `massmatrix` (Meyer et al.; no obtuse triangles occur on these spheres,
the obtuse branch is untested against a shipped file). `Seq` uses `np.lexsort` of
`np.round(., 9)` exactly as upstream.

Precision: everything float64, as shipped. Index arrays int64.

## Results

All 13 fields at all 6 levels: max abs diff <= 2.5e-14 (limit 1e-6); sparse
fields have the identical stored pattern (including explicit entries); integer
fields (`F`, `Seq`, `nv_*`) are exactly equal; `V`, `N`, `F`, `Seq`, `Intp` are
bit-identical. Per-field numbers: `docs/reports/_fragments/meshconv.md`.
Differences from libigl are float64 round-off only (G/NS/EW grow with level to
~2e-14 because gradients scale with edge length^-1).

No field failed to match; no float32 anywhere in the pickles.

## Security note

The pickles come from a third-party image and are pickles of numpy/scipy objects
saved under Python 2-era module names (protocol 2). Never `pickle.load` them
(or `np.load(allow_pickle=True)`). `safe_load_pickle` uses a `pickle.Unpickler`
subclass whose `find_class` allows exactly the six globals seen with
`pickletools.dis`/`genops` (see the report fragment): `numpy.core.multiarray._reconstruct`,
`numpy.ndarray`, `numpy.dtype`, `scipy.sparse.csc.csc_matrix`,
`scipy.sparse.coo.coo_matrix`, `_codecs.encode` (wrapped so it only accepts
`str -> 'latin1'`). Everything else (os.system, eval, builtins, modern
`numpy._core`/`scipy.sparse._csc` names) raises `SafeUnpicklingError`;
`persistent_load` is refused. Tests craft `os.system`/`eval` payloads inline.

## Shipping

Ship regenerated operators as `.npz` (`save_operators_npz` /
`load_operators_npz`, `allow_pickle=False`) or, better, regenerate on demand
with `build_operators` (<= 0.2 s up to level 6). No `.npz` is committed yet; the
original pickles are never copied into the repo.

## Tests

`tests/test_operators.py`: malicious-pickle rejection and shape/invariant tests
(always run); `@pytest.mark.reference` field comparison, sparsity patterns, and
level-5/6 `.vtk` comparison (needs `reference/`).
