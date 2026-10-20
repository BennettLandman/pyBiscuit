# Foundations report (brief 01) — 2026-10-20

Branch `b0-b2-b4-foundations`. Full suite: 161 passed, 0 skipped (`.venv/bin/pytest -q`); `ruff check .` clean.

**Key findings:** (1) legacy `Surfaces_FreeView/*.asc` is already scanner RAS, not tkRAS; (2) FreeView and MNI inner surfaces are the *same* mesh (250,874 v), related by a rigid transform matching inverse `Surfaces_MNI/aff.txt` (243,822 v is the central surface) — contradicts the brief; meaning deferred to Q4; (3) `sphere_327680.vtk` is the ico7 template sphere (all 12 `reg/*` surfaces share its face array); (4) BISCUIT spheres cannot meet 1e-6 (files are 4-decimal-quantised, float32, radius 100); faces and vertex order match exactly; (5) UGSCNN MeshConv license text is a placeholder, needs upstream text before shipping.


---

### B4.1 icosphere (measured 2026-10-20)

Generator: `pybiscuit.sphere.icosphere.icosphere(level, scheme)`; tests `tests/test_icosphere.py`
(`-m reference` tests skip when `reference/` is absent).

| Shipped file | scheme | faces identical | max abs diff, raw gen*R vs file | after `quantize_shipped` | 1-step tie flips | non-flip max diff |
|---|---|---|---|---|---|---|
| sphere_5120 (L4, R=100) | biscuit_ico4 | yes | 0.00503 | 0.0100 | 8 / 7,686 | 6.4e-6 |
| sphere_20480 (L5) | biscuit | yes | 0.00502 | 0.0100 | 10 / 30,726 | 6.7e-6 |
| sphere_81920 (L6) | biscuit | yes | 0.00502 | 0.0100 | 57 / 122,886 | 6.7e-6 |
| sphere_327680 (L7) | biscuit | yes | 0.00502 | 0.0100 | 277 / 491,526 | 6.7e-6 |
| icosphere_5.vtk (parc, R=1) | ugscnn | yes | 5.0e-7 | n/a | n/a | n/a |
| icosphere_6.vtk (parc, R=1) | ugscnn | yes | 5.0e-7 | n/a | n/a | n/a |

* Order matches vertex for vertex (no permutation needed) for every file; only the
  shipped 4-decimal quantisation makes |dxyz| <= 1e-6 unattainable on BISCUIT files
  (see port note). parc meshes pass the 1e-6 criterion.
* The BISCUIT spheres come in two families: `sphere_5120` (hierarchical face order, base
  ring rotated by +72 deg) and `sphere_{20480,81920,327680}` (two-pass face order). UGSCNN is a third.
* Template: all 12 `reg/*/?h.{white,inflated}.vtk` share `sphere_327680.vtk`'s face array
  exactly -> the template sphere is `sphere_327680.vtk` = `icosphere(7)`.
* Timing: level 7 builds in about 0.05 s (biscuit) / 0.1 s (ugscnn).

---

# B2.1 (FS/GIfTI part) and B2.2 fragment

Files: `src/pybiscuit/io/{freesurfer,gifti,frames}.py`; tests `tests/test_io_freesurfer.py`,
`tests/test_io_gifti.py`, `tests/test_frames.py`. `pytest` on these three files: 14 passed, 0 skipped;
`ruff check` clean on my files.

## Tests
- freesurfer: `test_asc_roundtrip`, `test_asc_bad_count`, `test_surf_roundtrip`, `test_curv_roundtrip`,
  reference `test_read_legacy_asc` x4 (FreeView/MNI x inner/central).
- gifti: `test_gifti_roundtrip` (surface + float and int per-vertex arrays; stored float32/int32).
- frames: `test_vox2ras_tkr_matches_nibabel_mgh`, `test_known_voxel` (1x1x1.5 mm, flip + axis permutation),
  `test_roundtrip_and_header_input`, `test_mesh_helpers`, reference `test_legacy_sample_mask_diagnostic`.

## Diagnostic numbers (T1 184x256x256, 1 mm, oblique affine; SEG_refine has identical affine)
Fraction of vertices inside the SEG_refine non-zero mask dilated by r = 0,1,2,3 voxels (inner surfaces, 250,874 verts):

| vertex set | r=0 | r=1 | r=2 | r=3 |
|---|---|---|---|---|
| FreeView .asc, no conversion (control) | 0.9999 | 1.0000 | 1.0000 | 1.0000 |
| FreeView .asc, tkRAS->scanner via T1 | 0.5986 | 0.6303 | 0.6579 | 0.6843 |
| MNI .asc, no conversion | 0.6078 | 0.6327 | 0.6531 | 0.6715 |

Bounding boxes (mm): FreeView raw x[-61.1,60.9] y[-78.8,77.5] z[-7.4,106.4]; converted
x[-61.2,61.3] y[-96.3,19.2] z[-39.6,115.0]; MNI x[-59.7,61.6] y[-96.1,59.3] z[-40.1,71.2];
T1 scanner extent x[-93,98] y[-133,153] z[-75,137].

**Conclusion: the legacy `Surfaces_FreeView/*.asc` is already in scanner RAS, not tkRAS.** Applying
tkRAS->scanner makes it worse. The `read_asc` default `frame="tkRAS"` (as requested) is therefore wrong for
this file; call `read_asc(p, frame="RAS")`. The reference test only asserts raw >0.95 and raw > converted.
(The conversion math itself is covered by the synthetic-header tests.)

## Surprises (contradict the brief)
1. Vertex counts: in the sample, **inner** is 250,874 / 501,744 in BOTH Surfaces_FreeView and Surfaces_MNI.
   243,822 / 487,640 is the **central** surface (also identical in both dirs); outer is 299,570 / 599,136.
2. Therefore FreeView and MNI are the SAME mesh (identical faces) in two frames, not independent extractions.
   A least-squares affine fit FreeView->MNI has max residual 5e-6 mm, is a rigid transform (rotation ~6 deg,
   translation ~(-3.0,-17.8,-34.6)), and matches the inverse of `Surfaces_MNI/aff.txt`. So the MNI .asc is
   the scanner-RAS surface moved by a rigid (not nonlinear) alignment. Noted only; semantics await Q4.
3. nibabel's MGH header helper is only used in a test; `vox2ras_tkr` is computed directly from shape/zooms
   (LIA convention) and agrees with `MGHHeader.get_vox2ras_tkr`.
4. `tests` is not an importable package, so test files define `MIPAV` locally instead of importing conftest.
5. Outside my files: `ruff check` reports E741 in tests/fixtures/meshes.py and B905 in mesh/topology.py;
   full `pytest` hit `ModuleNotFoundError: conftest` from another agent's test file (not mine).
6. GIfTI stores float32 (vertices/scalars), so round trips are exact only for float32-representable values.

## Exports to add to `src/pybiscuit/io/__init__.py`
```python
from pybiscuit.io.freesurfer import (
    read_asc, write_asc, read_fs_surf, write_fs_surf, read_fs_curv, write_fs_curv,
)
from pybiscuit.io.gifti import read_gifti, write_gifti
from pybiscuit.io.frames import (
    vox2ras_tkr, tkras_to_scanner_matrix, tkras_to_scanner_ras, scanner_ras_to_tkras,
    mesh_tkras_to_scanner, mesh_scanner_to_tkras,
)
```

---

## B2.1 fragment: VTK + per-vertex txt (`pybiscuit.io.vtk`, `pybiscuit.io.txt`)

Tests: `tests/test_io_vtk.py` + `tests/test_io_txt.py`: 29 passed, 0 skipped
(reference tests run here), ruff check/format clean. Run with
`-W error::DeprecationWarning` too.

- VTK round trip (ASCII/binary x float/double, with float32/int32/float64 scalars,
  3-vector, frame/units): `test_roundtrip[...]` (4); `test_no_frame_title_preserved`,
  `test_crlf_and_lookup_table`, `test_non_triangle_raises`, `test_bad_header_raises`,
  `test_large_ascii_speed` (327,680 faces, read asserted < 5 s, ~0.5 s including synthetic write).
- Reference (`@pytest.mark.reference`): `test_reference_reg_vtk[{white,inflated}-{lh,rh}]`
  (3 subjects each: hcp1058, mb101, oasis45; 163,842 v / 327,680 f, ~0.25 s per file),
  `test_reference_sphere_vtk[5120|20480|81920|327680]`,
  `test_reference_icosphere_vtk[5-10242|6-40962]`, `test_reference_pymacruise_surface`
  (241,042 v / 482,500 f, frame RAS, units mm, dtype double; also binary round trip
  keeps frame/units), `test_reference_reg_txt[{mean,var}-{lh,rh}]` (163,842 values).
- Timings: sphere_327680 (163,842 v) 0.3 s; reg white/inflated ~0.25 s each; PyMaCRUISE 0.44 s.

Design: ASCII numeric blocks are located with a regex for the next non-numeric
character and parsed with `np.fromstring(sep=" ")`; binary uses big-endian `frombuffer`.
Reader accepts VTK type names, SCALARS (+LOOKUP_TABLE) / VECTORS / NORMALS in
POINT_DATA (1-comp -> shape (N,), else (N,k), native dtype kept); CELL_DATA/FIELD are
ignored; VERTICES/LINES/TRIANGLE_STRIPS and non-triangle POLYGONS raise ValueError.
`mesh.meta` gets `title`, `vtk_version`, `vtk_ascii`, `points_type`. `write_vtk(mesh, path,
binary=False, dtype=None)`: dtype 'float'|'double' (default = type read, else double);
title written as `PyMaCRUISE coordinate_frame=X units=Y` when frame/units set.

Surprises: reference sphere files are `# vtk DataFile Version 1.0`; sphere_327680.vtk has
CRLF line endings; sphere_N.vtk names are face counts (N faces, N/2+2 vertices);
reg/icosphere files have title "vtk output"; reg files have 5 blank/extra lines between
POINTS and POLYGONS (handled by whitespace skipping). PyMaCRUISE surface is `double`.
Writers use `%.9g` (float) / `%.17g` (double) so round trips are exact.
Per-vertex txt reader: `read_vertex_txt(path, dtype=float64, n_vertices=None)`;
`write_vertex_txt(values, path, fmt=None)`.

`src/pybiscuit/io/__init__.py` exports to add (I did not edit it):
```python
from .txt import read_vertex_txt, write_vertex_txt
from .vtk import read_vtk, write_vtk
```
(plus `"read_vtk", "write_vtk", "read_vertex_txt", "write_vertex_txt"` in `__all__`).
Note: tests import `from conftest import ...` (rootdir-style, no tests/__init__.py).

---

## B4.3 MeshConv operators vs shipped `icosphere_*.pkl`

**Disassembly** (`pickletools.genops`, never loaded): `icosphere_{0..5}.pkl` are
pickle protocol 2 (Python 2 era), 12,414 B (level 0) to 10,984,603 B (level 5).
Opcodes seen: `PROTO EMPTY_DICT MARK SETITEMS GLOBAL NEWOBJ REDUCE BUILD TUPLE*
BIN{PUT,GET} BININT* BINUNICODE NONE NEWTRUE NEWFALSE EMPTY_TUPLE STOP` — no
`STACK_GLOBAL`/`INST`/`OBJ`. GLOBALs, identical in all six files:

- `numpy.core.multiarray _reconstruct`
- `numpy ndarray`
- `numpy dtype`
- `_codecs encode` (used only as `encode(<str>, 'latin1')` for array buffers)
- `scipy.sparse.csc csc_matrix`
- `scipy.sparse.coo coo_matrix`

Loaded with a restricted `Unpickler` allowing exactly these.

**Keys**: `V F nv_prev nv_next G L N NS EW F2V M Seq Intp` (all levels).

**Levels**: 0..5 (no 6/7 pickles; `icosphere_5.vtk`, `icosphere_6.vtk` only).

**Max abs diff, shipped vs `build_operators(level)`** (limit 1e-6; sparse stored
patterns identical, `V F nv_prev nv_next Seq Intp` exactly 0):

| level | G | L | N | NS | EW | F2V | M |
|---|---|---|---|---|---|---|---|
| 0 | 2.2e-16 | 1.8e-15 | 0 | 2.2e-16 | 4.4e-16 | 0 | 2.2e-16 |
| 1 | 8.9e-16 | 8.9e-16 | 0 | 5.6e-16 | 5.6e-16 | 8.3e-17 | 5.6e-17 |
| 2 | 2.2e-15 | 1.3e-15 | 0 | 1.3e-15 | 1.0e-15 | 8.3e-17 | 2.8e-17 |
| 3 | 5.3e-15 | 1.8e-15 | 0 | 3.8e-15 | 3.7e-15 | 8.3e-17 | 6.9e-18 |
| 4 | 1.1e-14 | 1.8e-15 | 0 | 9.5e-15 | 9.5e-15 | 8.3e-17 | 1.7e-18 |
| 5 | 2.5e-14 | 2.2e-15 | 0 | 2.1e-14 | 2.1e-14 | 1.1e-16 | 6.5e-19 |

`.vtk` meshes: level 5 and 6 faces equal; vertices differ by <= 5.0e-7 (the
file prints 6 decimals as `float`).

**Unmatched fields**: none. dtypes: V/G/L/N/NS/EW/F2V/M/Intp float64; F, Seq int64.

---

# Fragment: B0.2 CI, B0.3 asset registry, B0.4 fixtures, licenses

## Files
- `.github/workflows/ci.yml` - ruff check + pytest, {3.10, 3.13} x {ubuntu, macos, windows}, `pip install -e .[dev]`. YAML parsed OK with PyYAML (cannot run Actions locally). `reference` tests auto-skip via tests/conftest.py (no `reference/` in CI).
- `src/pybiscuit/assets.py`, `src/pybiscuit/data/{__init__.py,asset_registry.json}`, `scripts/build_asset_registry.py`; `pyproject.toml` package-data (`pybiscuit.data = ["*.json"]`). API: `load_registry()`, `status_summary()`, `verify(root)`, `cache_dir()` (`$PYBISCUIT_ASSETS` or `~/.cache/pybiscuit/assets`), `sha256_file()`. No download code.
- `tests/test_assets.py` (5 tests; 1 is `@pytest.mark.reference`: file set equals registry, sizes, all SHA-256), `tests/fixtures/{__init__,meshes}.py`, `tests/test_fixtures.py` (25 tests incl. parametrised). Combined 30 tests pass; reference test ran and passed locally.
- `LICENSE` (MIT, "pyBiscuit contributors", 2026), `LICENSES/{BISCUIT,CMorph,HSD,CurveExtraction,LocalGyrificationIndex,SphericalRemesh,SphericalLabeling,MeshLib}.txt` (verbatim copies), `LICENSES/UGSCNN-MeshConv.txt` (placeholder), `LICENSES/README.md` (index + not-included list).

## Registry
76 files, 1,259,450,198 bytes (reg 201M, parc 983M, sphere 18M); JSON is ~11 KB. Paths are relative to `/opt/biscuit` (e.g. `reg/mb101/lh.white.vtk`).

## Fixture hashes (`geometry_hash`: vertices -> `rint(v*1e9)` int64 LE, faces int32 LE, SHA-256)
| generator | sha256 |
|---|---|
| uv_sphere(8,16) | e8add2851dee170432d75169b39bdea355e0181173f726b551aabb930a3fca77 |
| ellipsoid(3,2,1,12,24) | 7eaaa40290a9885f90b126e6b9941ee73e289c058d4eea761e1ec38fb388fea9 |
| icosphere(0) | 47d05533271ff80f1013ab58306c11204e493aa614c07620ddcc889d55811d17 |
| icosphere(3) | 918dff3f719be3f13e9ec5a8a5fedb47c247eeb37998c9216687032098e8330c |
| icosphere(5) | 70b8fa3a73579410680cee8fafbaf2f7190a9ac9b5d8e8749088e9b266ded00b |
| folded_sphere(0.1,4,2,3) | f925ba9f161955586bd1c890ac54c627a380bb7f9d7bb6cff31eeb2f2425ac37 |
| cut_sphere(3,0.5) | 23c3d501031d1af9c3f01e46b937ace71ffa9a2af961ae0f43912b4490d5e47f |
| perturb(icosphere(2),0.01,seed=7) | 0e873076fea7b0084e0d3deda539713061f9c6278a4cc82c92be803a2e02d451 |

Only verified on this machine (macOS arm64, scipy 1.18.1); cross-platform stability relies on the 1e-9 quantisation (a value within ~1e-16 of a rounding boundary could in principle flip; unobserved). Real SH convention is documented in `meshes.py` (scipy `sph_harm_y`, falls back to `sph_harm`).

## Surprises / notes
- `tests/` must NOT be a package: other tests use `from conftest import ...`; mine do the same and import `fixtures` as a top-level package.
- UGSCNN MeshConv has no LICENSE text in `reference/` (SphericalLabeling/meshcnn has no header); placeholder only, verbatim MIT text to be copied from upstream.
- Existing LICENSES/ was an empty dir. The `ruff check --fix .` I ran repo-wide applied 2 auto-fixes (likely only import ordering) - possibly in other agents' files.
- At the time of my last run: `ruff check .` had 1 error in another agent's `src/pybiscuit/mesh/topology.py` (B905 zip strict) and `tests/test_operators.py::test_rejects_codecs_encode_misuse` failed (not mine).

---

# B2.3 topology (fragment)

Module: `src/pybiscuit/mesh/topology.py`. Tests: `tests/test_topology.py` (27 tests).

## Tests
- Synthetic (inline generators: icosphere, UV sphere, torus, cut spheres, flipped faces, two spheres, Moebius strip, bowtie, non-manifold fan): chi=2/genus 0 on spheres; chi=0/genus 1 on torus; 1 boundary loop on a sphere with one cap removed, 2 loops with both caps removed (chi=0); a single flipped face is the only face reported by `check_orientation` (3 inconsistent edges), flipping it back restores consistency; Moebius strip reported non-orientable; 2 components on two disjoint spheres (chi=4); unreferenced vertices ignored in chi/components; 327,680-face performance test.
- `@pytest.mark.reference` (10 cases): `sphere_{5120,20480,81920,327680}.vtk` and the six `reg/{hcp1058,mb101,oasis45}/{lh,rh}.white.vtk` (note: the files live under `.../opt/biscuit/sphere/`, not `biscuit/biscuit/sphere/`). All: chi=2, genus 0, one component, closed manifold, consistently oriented.

## Timings (327,680 faces / 163,842 vertices, this Mac, pure NumPy/SciPy)
unique_edges 0.11 s, one_ring 0.08 s, vertex_faces 0.04 s, vertex_neighbors 0.13 s, face_adjacency 0.11 s, boundary_loops 0.10 s, connected_components 0.02 s, check_orientation 0.18 s, topology_summary (everything) 0.28 s. Numba not needed.

## Findings / surprises
- All three template white surfaces (hcp1058, mb101, oasis45; both hemispheres) are genus 0 closed manifolds; no exceptions. All shipped spheres likewise.
- All reference meshes have consistent winding.

## Design notes
- Orientation: `is_consistent` = no directed edge used twice. `flipped_faces` is the minority winding class per face-connected component (BFS parity over manifold edges, tie keeps lowest-numbered face's class); a flipped face adjacent to it is not blamed.
- `one_ring` is a vectorised fan walk using face winding; vertices with pinches, non-manifold edges, or inconsistent winding get `valid=False` and an empty ring.
- Unreferenced vertices are excluded from chi and get component label -1.
- Genus is per-component `(2 - chi_c - b_c)/2`, summed; meaningful only when `is_manifold`.
- Boundary loops at pinch vertices are split deterministically but arbitrarily.
