# Port note: MaCRUISE hemisphere separation (plan row B5.2)

Python: `src/pybiscuit/surfprep/` (`hemisphere.py`, `contour.py`, `frames.py`, `cli.py`).
Tests: `tests/test_hemisphere.py` (synthetic), `tests/test_hemisphere_legacy.py` (legacy sample, marker `reference`).
CLI: `pybiscuit mesh-contour` (stand-in for the closed `MeshContour`, spec T39), `pybiscuit hemi-split`.
Class: **replacement-tolerant** (spec §4.2): the patch triangulation is not Qhull-identical, so only
vertices with `mask == 1` are comparable vertex-wise. **Status: provisional**, no image run exists (Q1).

## Source (BISCUIT MATLAB, `SurfPostproc/macruise/tessellation/`; read as specification, nothing copied)

| Python | MATLAB |
|---|---|
| `orientation_fix` | `surfNormalCorrection.m` (all 31 lines) |
| `contour.smooth_neighbor_mean`, `candidate_offsets` | `hemi_separation.m` `getCuttingPlane` 76-125, `smooth` 236-244 |
| `contour.mesh_contour`, `select_plane` | **`MeshContour` binary (closed)**, called at `:127`; see hypothesis below |
| bdry flagging in `_split_with_cut` | `getCuttingPlane` 130-137 |
| `_face_index` | `computeIndex` 252-269 |
| `_boundary_loops` | `getBoundary` 142-193 |
| `_hemisphere_components` | `cleanupBoundary` 194-204 |
| `disk_patch` | `circle_patch.m`, `circlePatch` 245-251 |
| `harmonic_fill` | `poisson` 270-287 |
| stitching in `_split_with_cut` | `closeSurface` 324-347 |
| seam smoothing | `:44-52` |
| component split, left/right, `hemi_map` | `connectedComp.m`, `:56-73` |
| `write_outputs` | `write_vtk.m` (6 decimals, `float`), `writeInteger`, `writeInteger3` |

## Hypothesis H-MC1: the MeshContour plane rule (isolated in `contour.select_plane`)

`MeshContour` receives the smoothed reference mesh, the normal `1 0 0` and 101 offsets and prints the faces cut by one plane
plus its index. The rule is not in the source. **H-MC1 (default `shortest`)**: the plane whose intersection polyline with the mesh is
shortest. Alternative **H-MC2 (`fewest_faces`)**: fewest cut faces. Selecting the rule is one function (`select_plane`,
`PLANE_RULES`); `--rule` exposes it on both CLIs. Evidence available now: on the legacy sample all 101 candidates cut 523-560 faces,
so ranking is almost flat and the choice moves the split by at most a few dozen vertices (see the report). To settle it, probe the image
binary with synthetic planes (spec §15 item 4) and add a rule to `PLANE_RULES`.
The CLI prints face ids (one per line) then the chosen index, which is what `hemi_separation.m` parses with `str2num`.

## Deviations from MATLAB

* **D1** island removal keeps the two *largest components* of the kept mesh, not `dmperm` blocks 1 and 2 (spec §14 item 5). Equal on clean meshes.
* **D2** patch boundary points at angles `2*pi*i/m` (m distinct points), loop vertex `i` paired with point `i`, ccw triangles from `scipy.spatial.Delaunay`;
  MATLAB pairs in Qhull hull order with a duplicated end point (spec §14 item 6: the cyclic alignment is unknowable). Interior grid is the same 51x51
  concentric-disk grid (radius `1 - 1/50`), scaled down only when the loop has < 16 vertices so that it stays inside the boundary polygon. The Dirichlet solve is the same unweighted-Laplacian system.
* **D3** if a candidate plane does not give valid loops, the next-ranked plane is tried (up to 5; `--max-plane-tries`); MATLAB exits with "the cutting plane is wrong!". `diagnostics.tried` records each failure.
* **D4** `getBoundary` walks *all* boundary loops (MATLAB: exactly two: first vertex and last unvisited). Loops that belong to cut-off pieces (islands: any component other than the two largest) are dropped with the piece; every other loop is patched. In the legacy sample the plane also clips a 49-vertex knob next to the posterior midline, so MATLAB's exactly-two-loops walk could not close that sample; here the knob is removed (its vertices are absent from `hemi_map`) and the hole it leaves is patched.
* **D5** the 20-iteration smoothed reference is rounded to 6 decimals (MeshContour reads a `write_vtk` file); everything else stays in double (MATLAB double as well).
* Fixed-size patches: every loop gets a 2601-vertex grid, however small the loop (as in MATLAB), so tiny holes carry a dense patch and the count of `mask == 0` vertices is not proportional to the hole area.

## Alignment frame (Q4)

The reference copy matters only through its x axis: the candidate planes are `x = mean(x) + r`, `|r| <= 0.5` mm. Options, all verified on the legacy sample:

| Frame | Function | Result on the legacy sample |
|---|---|---|
| MNI copy (BISCUIT's input) | `reference_from_mni_vertices` | reference |
| `Surfaces_MNI/aff.txt` | `reference_from_affine` (`aff.txt` maps **MNI -> native**; applying its inverse reproduces the MNI `.asc` to 5e-6 mm) | identical to the MNI copy (0 of 250,800 vertices differ) |
| plane fitted to `SEG_refine` L/R labels (default for PyMaCRUISE) | `reference_from_labels`: Fisher/LDA direction between left (37,45,52,56,58,60,62, odd >= 100) and right (36,44,51,55,57,59,61, even >= 100) voxels (ids from spec §4.5), rotated to +x | normal 0.84 deg from the MNI x axis; 99.985 % hemisphere agreement with the MNI-frame split |
| none | `reference=None` (native vertices) | fails or tilts the cut when the head is rotated (the sample is rotated about 6 deg) |

Left/right is decided by the native mean x (smaller = `lh`, as in `connectedComp.m`), so the native frame must have right = +x (RAS).

## Numeric notes

Double precision throughout. Output VTKs use `%f` (6 decimals, `float`), as `write_vtk.m` does, so re-reading them reproduces the rounding the next MATLAB stage sees.
`hemi_map.txt` rows: `<whole-brain index> <hemisphere index> <0 lh | 1 rh>`, 0-based, whole-brain order, patch and discarded vertices omitted.
`<h>.mask.txt`: 1 original, 0 patch. Run time on the 250,874-vertex sample: about 4 s.
