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
