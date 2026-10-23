# Port Note: Summary Statistics (`B10.1`) and `legacy/` Output View (`B10.4`)

## 1. Scope and Source Mapping

| Upstream MATLAB file | Lines | `pybiscuit` counterpart | Responsibility |
|---|---|---|---|
| `include/read_bary.m` | 1–32 | `pybiscuit.stats.read_bary` | Parse TRACE `.bary` polyline files into `(v0, v1, w)` arrays |
| `surf_quant_stat_pipeline.m` | 13–21 | `pybiscuit.stats.read_bary(..., unique_rows=True)` | Deduplicate shared junction points via `unique(curve, 'rows')` |
| `surf_quant_stat_pipeline.m` | 24–28 | `pybiscuit.stats.bary_interpolate` | Edge-wise linear interpolation and `>= 0.5` label selection |
| `surf_quant_stat_pipeline.m` | 1–133 | `pybiscuit.stats.compute_roi_and_curve_stats` | ROI-wise, curve-wise, and aggregate (`20000..20012`) mean/std tables |
| `surf_quant_stat_pipeline.m` | 112–133 | `pybiscuit.stats.write_stats_csv` | Format and export `stats.<protocol>.csv` (28 columns, `%.15g` floats) |
| `biscuit.m` | 412–470 | `pybiscuit.stats.remesh_stat_maps`, `run_stats_pipeline` | Resample native maps onto `sphere_327680.vtk` (`T37`) and write `stat/` outputs |
| `biscuit.m` | 538–614 | `pybiscuit.pipeline.legacy` | Populate and validate the 71-file (MaCRUISE) / 69-file (FS) legacy tree |

---

## 2. Algorithm Details (`surf_quant_stat_pipeline.m`)

### 2.1 Measure Order and Stacking (`surf_quant_stat_pipeline.m:1-12, 31-35`)

For each hemisphere $h \in \{\text{lh}, \text{rh}\}$, the 6 cortical measures are stacked as columns of an $N_h \times 6$ `float64` matrix in the fixed order:

1. `MeanCurvature` (`quant/<h>.mid.H.txt`, originally `<h>.mid.vtk.H.txt`)
2. `ShapeIndex` (`quant/<h>.mid.S.txt`, originally `<h>.mid.vtk.S.txt`)
3. `SulcalDepth` (`quant/<h>.gray.SD.txt`, copied to `<h>.mid.SD.txt` in `tmp3`)
4. `Thickness` (`quant/<h>.thickness.txt`, copied to `<h>.mid.CT.txt` in `tmp3`)
5. `SCI` (`quant/<h>.mid.SCI.txt`)
6. `LGI` (`quant/<h>.gray.LGI.txt`, copied to `<h>.mid.LGI.txt` in `tmp3`)

Left and right hemispheres are concatenated vertically (`props = [props_lh; props_rh]`, `parc = [parc_lh; parc_rh]`). Notably, the cortical mask `<h>.mask.txt` is **not** read by `surf_quant_stat_pipeline.m`; medial-wall exclusion happens solely because non-cortical vertices carry parcellation label `0`, and row 1 (`ID == 0`, `"Unknown"`) is skipped (`for i = 2:size(label, 1)`).

### 2.2 Sulcal Curve Barycentric Sampling (`read_bary.m`, `surf_quant_stat_pipeline.m:13-30`)

Only the **pial sulcal curve** (`curve/<h>.gray.scurve.bary`) is used (`surf_quant_stat_pipeline.m` receives `lh.pial.scurve.bary` and `rh.pial.scurve.bary`).

1. **`read_bary.m` parsing**:
   - Line 1: `N` (number of polylines).
   - For each polyline $k = 1 \dots N$: a 1-integer header giving segment count $M_k$, followed by $M_k$ triples `v0 v1 coeff` where `v0, v1` are 0-based vertex indices and `coeff` $\in [0, 1]$ is the weight on `v0`.
2. **Junction deduplication (`unique(curve, 'rows')`)**:
   - Upstream calls `curve_lh = unique(curve_lh, 'rows')` and `curve_rh = unique(curve_rh, 'rows')` on the raw `[v0, v1, coeff]` matrix (`v0 + 1` in MATLAB 1-based indexing, which preserves row order).
   - Because CurveExtraction repeats junction points across incident branches, `unique(..., 'rows')` removes exact duplicate `(v0, v1, coeff)` triples and sorts rows lexicographically by `(v0, v1, coeff)`.
3. **Barycentric interpolation (`bary_parm`)**:
   - Continuous measures are linearly interpolated along each mesh edge:
     $$\text{val}_c = w \cdot \text{props}[v_0] + (1 - w) \cdot \text{props}[v_1]$$
   - Discrete parcellation labels (`parc`) and lobe IDs (`lobe`) are assigned by threshold:
     $$\text{label}_c = \begin{cases} \text{label}[v_0], & w \ge 0.5 \\ \text{label}[v_1], & w < 0.5 \end{cases}$$
   - Note the asymmetry when `w == 0.5`: `bary_parm(parc, curve, true)` tests `curve(:,3) < 0.5` and replaces those entries with `v1`, so `w == 0.5` selects `v0`.

### 2.3 Per-ROI and Aggregate Rows (`surf_quant_stat_pipeline.m:45-109`)

Let `label.csv` have $R$ rows (row 1 is `Unknown,0`, rows $2 \dots R$ are cortical ROIs):

- **Individual ROIs** (output rows $1 \dots R-1$):
  - For each ROI row $i \in \{2 \dots R\}$ with `ID = label.ID(i)`:
    - Vertex subset: `parc == ID` across stacked `lh + rh`.
    - Curve subset: `parc_curve == ID` across stacked `lh + rh`.
    - Mean (`np.mean(..., axis=0)`) and sample standard deviation (`ddof=1`, matching MATLAB `std(X, 0, 1)`).
    - Empty subset ($n = 0$): `nVertices = 0`, mean = `NaN`, stdev = `NaN`.
    - Singleton subset ($n = 1$): `nVertices = 1`, mean = single value, stdev = `NaN`.
- **Aggregate rows**:
  - `20000` (`Whole cortex`): all vertices/curve points whose parcellation ID belongs to `label.ID(2:end)` (`ismember(parc, label.ID(2:end))`).
  - `20001` (`Left hemisphere`), `20002` (`Right hemisphere`): same `ismember` mask restricted to `lh` or `rh`.
  - `20003..20012` (five bilateral lobes: `Left/Right frontal`, `temporal`, `parietal`, `occipital`, `limbic` for lobe codes `1..5`): emitted **only** when `label.csv` has a 3rd column (`lobe`), i.e. `braincolor` (111 output rows: 98 ROIs + 13 aggregates) and omitted for `dkt31` (65 output rows: 62 ROIs + 3 aggregates).

### 2.4 CSV Formatting (`surf_quant_stat_pipeline.m:112-133`)

- Header (28 columns):
  ```
  ID,Label,nVertices,MeanCurvature,ShapeIndex,SulcalDepth,Thickness,SCI,LGI,MeanCurvature_stdev,ShapeIndex_stdev,SulcalDepth_stdev,Thickness_stdev,SCI_stdev,LGI_stdev,nVertices_curve,MeanCurvature_curve,ShapeIndex_curve,SulcalDepth_curve,Thickness_curve,SCI_curve,LGI_curve,MeanCurvature_stdev_curve,ShapeIndex_stdev_curve,SulcalDepth_stdev_curve,Thickness_stdev_curve,SCI_stdev_curve,LGI_stdev_curve
  ```
- Float formatting: MATLAB R2012a `export(stats, 'file', output, 'Delimiter', ',')` formats IEEE-754 double arrays with `%.15g` (`NaN` for NaN). Across all 12,672 float cells in the 6 `open-examples` reference CSVs (`{0050952,0050961,0050971} × {braincolor,dkt31}`), `%.15g` round-trips 100 % of parsed values back to the exact reference string.

---

## 3. Registered-Space Property Maps (`biscuit.m:412-436`)

For each hemisphere $h \in \{\text{lh}, \text{rh}\}$ and each of the 6 measures (`mid.H`, `mid.S`, `gray.SD`, `thickness`, `mid.SCI`, `gray.LGI`), `biscuit.m` calls `SphericalRemesh` (`T37`):

```
SphericalRemesh -s <h>.sphere.reg.vtk -r sphere_327680.vtk -p <native_map.txt> --outputProperty <reg_map.txt>
```

`pybiscuit.stats.remesh_stat_maps` calls `pybiscuit.sphere.remesh.remesh(..., output_precision=6)` and writes the 6 per-hemisphere files in `stat/`:
- `stat/<h>.mid.reg.H.txt`
- `stat/<h>.mid.reg.S.txt`
- `stat/<h>.mid.reg.SCI.txt`
- `stat/<h>.gray.reg.LGI.txt`
- `stat/<h>.gray.reg.SD.txt`
- `stat/<h>.reg.thickness.txt`

Note that `stat/<h>.mid.reg.area.txt` is produced during the registration stage (`T20`, `SphericalRemesh` of `mid.VertArea.txt` onto `< reg_temp >/<h>.white.vtk`), not in `T37`.

---

## 4. Legacy Output View (`B10.4`, `pybiscuit.pipeline.legacy`)

`biscuit.m:538-614` copies scratch intermediates into a 7-subdirectory (or 6-subdirectory on case-insensitive filesystems) layout:
- `surf/` (10 files): `<h>.{white,gray,mid,inflated,sphere}.vtk` (where `<h>.pial.vtk` is renamed to `<h>.gray.vtk`).
- `reg/` (10 files): `<h>.sphere.reg.vtk`, `<h>.sphere.reg.<reg_temp>.coeff`, `<h>.{white,gray,mid}.reg.vtk`.
- `roi/` (2 + 2P files): `<h>.mask.txt` (MaCRUISE route only) and `<h>.parc.<protocol>.txt`.
- `curve/` (8 files): `<h>.{gray,white}.{scurve,gcurve}.{bary,vtk}`.
- `quant/` (14 files): `<h>.gray.{LGI,SD}.txt`, `<h>.mid.{H,S,SCI,area}.txt`, `<h>.thickness.txt` (note the inconsistent `gray`/`mid`/no-surface infix naming).
- `stat/` (14 + P files): `<h>.gray.reg.{LGI,SD}.txt`, `<h>.mid.reg.{H,S,SCI,area}.txt`, `<h>.reg.thickness.txt`, `stats.<protocol>.csv`.
- `pdf/` (3 + P files) and `PDF/` (1 file, `PDF/biscuit.pdf`): on case-insensitive APFS, `pdf/` and `PDF/` share a single directory; `discover_legacy_files` normalizes `biscuit.pdf` to `PDF/biscuit.pdf` and individual page PDFs to `pdf/<name>.pdf`.
