# Port note: FreeSurfer-route input stage (brief 22)

Code: `src/pybiscuit/surfprep/freesurfer.py`. Tests: `tests/test_surfprep_freesurfer.py` (synthetic + three
open examples). Source: `BISCUIT-master/SurfPostproc/freesurfer/run_postproc_fs_pipeline.m:1-52` (BISCUIT's own
MIT code; the FreeSurfer-licensed `read_surf.m` / `read_curv.m` were not read, NiBabel reads the files). Spec
[§12](../reference/orchestrator-spec.md).

| `run_postproc_fs_pipeline.m` | Port | Notes |
|---|---|---|
| `read_curv` → `fprintf('%f\n')` of `curv`, `sulc`, `inflated.H`, `thickness` | `write_feature_txt` | six decimals; `<h>.<name>.txt` |
| `read_surf` → `write_vtk` of `white`, `pial` (as `gray`), `inflated`, `sphere` | `write_matlab_vtk` | `POINTS n float`, `%f %f %f`, `POLYGONS m 4m`, 0-based faces, no point data, FreeSurfer (tkRAS) frame; byte-identical to the image's `surf/*.vtk` |
| mid surface | `mid_vertices` | `(white + gray)/2` of the **printed** (six-decimal) vertices, because the script re-reads the VTKs it wrote; faces of `white` |
| cortical mask | `read_cortex_label`, `mask_from_label` | `dlmread(label, ' ', 2, 0)(:,1) + 1` → `1`, else `0`, `%d\n`; without the label file the script writes ones (not ported: unreachable in the image runs) |
| `surf_smooth > 0` | not ported | `MeshDeform --smooth`; the image runs show `surf_smooth = 0` on this route |

Frame: the VTKs stay in FreeSurfer's surface frame, as the image writes them (Q14: native RAS layout plus a
`legacy/` view is still an owner decision; this stage is the legacy reproduction).

Evidence: for the three open examples (six hemispheres) `white`, `gray`, `mid`, `inflated`, `sphere` VTKs,
`roi/<h>.mask.txt` and `quant/<h>.thickness.txt` are **byte-identical** to the files the image wrote (test
`test_matches_image_inputs`); `quant/<h>.mid.area.txt` equals `MeshProperty -v` (barycentric vertex area,
`mesh.geometry.vertex_areas`) to print precision (5e-7). [Report](../reports/open-examples-2026-10-22.md).
