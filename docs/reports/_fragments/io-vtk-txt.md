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
