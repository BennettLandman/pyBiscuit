# Brief 18 (Part 1) Report: Summary Statistics (`B10.1`) & `legacy/` Output View (`B10.4`) — 2026-10-22

Branch: `brief-18-stats` (worktree `/Users/landmaba/dev/biscuit/pyBiscuit-18-stats`).
Port note: [`docs/ports/stats.md`](../ports/stats.md).

## 1. Deliverables Summary

1. **`pybiscuit.stats` (`B10.1`, spec §10)**:
   - `read_bary(path, unique_rows=True)` and `read_bary_polylines(path)`: parses CurveExtraction `.bary` polyline files (`read_bary.m`) and applies `unique(curve, 'rows')` (`surf_quant_stat_pipeline.m:14-15`) to deduplicate shared junction points across sulcal branches.
   - `bary_interpolate(values, curve, threshold=False)`: edge-wise linear interpolation for continuous measures and `w >= 0.5 -> v0` / `w < 0.5 -> v1` discrete label assignment (`surf_quant_stat_pipeline.m:24-28`).
   - `compute_roi_and_curve_stats(...)`: computes the 28-column ROI-wise, curve-wise, and aggregate (`20000..20002` for 2-column `label.csv` such as `dkt31`; `20000..20012` when a 3rd `lobe` column is present such as `braincolor`) summary table in measure order `[H, S, SD, thickness, SCI, LGI]` with `ddof=1` sample standard deviation and `NaN` for empty/singleton stds.
   - `write_stats_csv(...)` & `read_stats_csv(...)`: writes and parses the 28-column `stats.<protocol>.csv` using `%.15g` float formatting (matching MATLAB R2012a `dataset.export(..., 'Delimiter', ',')`).
   - `remesh_stat_maps(...)` & `run_stats_pipeline(...)`: resamples the 6 per-hemisphere native measures onto `sphere_327680.vtk` (`T37`) and writes `stat/stats.<protocol>.csv`.
   - Standalone CLI in `src/pybiscuit/stats/cli.py` (`python -m pybiscuit.stats.cli`).
2. **`pybiscuit.pipeline.legacy` (`B10.4`, spec §13.1)**:
   - `expected_legacy_files(protocols=("braincolor", "dkt31"), reg_temp="mb101", route="macruise", include_merged_pdf=True) -> set[str]`: returns the exact 71-file MaCRUISE-route (or 69-file FreeSurfer-route without `PDF/biscuit.pdf`) relative path set, including all `gray`/`mid`/no-infix naming irregularities (`<h>.gray.{LGI,SD}.txt`, `<h>.mid.{H,S,SCI,area}.txt`, `<h>.thickness.txt`, `<h>.reg.thickness.txt`, `PDF/biscuit.pdf`).
   - `legacy_source_candidates(...)`, `populate_legacy_tree(...)`, and `discover_legacy_files(...)` (with macOS case-insensitive `pdf/` vs `PDF/` normalization).
3. **Tests (`tests/test_stats.py`, `tests/test_legacy_tree.py`)**:
   - 19 tests (8 synthetic unit tests + 11 `@pytest.mark.reference` tests against `open-examples/{0050952,0050961,0050971}`).

---

## 2. Validation Against `biscuit_FC_v2.2.sif` Reference Outputs (`open-examples`)

### 2.1 `stats.<protocol>.csv` Verification (`B10.1`)

`compute_roi_and_curve_stats` was run directly on the image's `quant/`, `roi/`, and `curve/<h>.gray.scurve.bary` files for all 3 ABIDE NYU subjects (`0050952`, `0050961`, `0050971`) and both protocols (`braincolor`, `dkt31`), and compared against the reference `stat/stats.<protocol>.csv`:

| Subject | Protocol | Rows | Columns | `ID` & `Label` match | `nVertices` & `nVertices_curve` match | Max Abs Diff | Max Rel Diff | Exact `%.15g` cell match | Individual ROI rows exact `%.15g` match |
|---|---|---|---|---|---|---|---|---|---|
| `0050952` | `braincolor` | 111 | 28 | 111/111 | 111/111 (exact) | `1.279e-13` | `3.137e-14` | `2630 / 2664` (98.7 %) | `2352 / 2352` (**100 %**) |
| `0050952` | `dkt31` | 65 | 28 | 65/65 | 65/65 (exact) | `1.776e-13` | `5.468e-14` | `1529 / 1560` (98.0 %) | `1488 / 1488` (**100 %**) |
| `0050961` | `braincolor` | 111 | 28 | 111/111 | 111/111 (exact) | `1.013e-13` | `4.743e-14` | `2630 / 2664` (98.7 %) | `2352 / 2352` (**100 %**) |
| `0050961` | `dkt31` | 65 | 28 | 65/65 | 65/65 (exact) | `1.119e-13` | `3.115e-14` | `1529 / 1560` (98.0 %) | `1488 / 1488` (**100 %**) |
| `0050971` | `braincolor` | 111 | 28 | 111/111 | 111/111 (exact) | `2.718e-13` | `2.833e-14` | `2631 / 2664` (98.8 %) | `2352 / 2352` (**100 %**) |
| `0050971` | `dkt31` | 65 | 28 | 65/65 | 65/65 (exact) | `2.469e-13` | `2.326e-14` | `1529 / 1560` (98.0 %) | `1488 / 1488` (**100 %**) |

Key observations:
- Every individual ROI row (all 98 cortical ROIs in `braincolor`, including empty ROIs `42, 43` with `0,NaN,...`, and all 62 cortical ROIs in `dkt31`) matches the image's CSV **string-for-string** across all 24 float columns when formatted with `%.15g`.
- The remaining 1.2–2.0 % of cells are all in the large aggregate rows (`20000..20012`, summing $10^4\text{–}3\times 10^5$ vertices) where MATLAB R2012a's multithreaded BLAS reduction order differs from NumPy pairwise summation by 1–7 ulps in the 15th significant digit (`max rel diff <= 5.5e-14`, 8 orders of magnitude below the `1e-6` E-class tolerance).

### 2.2 Registered-Space Property Maps (`stat/<h>.*.reg.*.txt`)

Running `remesh_stat_maps` on `0050952` (`quant/` maps resampled through `reg/<h>.sphere.reg.vtk` onto `sphere_327680.vtk`) reproduces all 12 registered-space files (`{lh,rh}.mid.reg.{H,S,SCI}.txt`, `{lh,rh}.gray.reg.{LGI,SD}.txt`, `{lh,rh}.reg.thickness.txt`) **byte-for-byte** (`SHA-256` identical, `max abs diff = 0.0`).

### 2.3 Legacy Output View (`B10.4`)

For all 3 subjects (`0050952`, `0050961`, `0050971`), comparing `discover_legacy_files(open-examples/<subj>)` against `expected_legacy_files(protocols=("braincolor", "dkt31"), reg_temp="mb101", route="fs", include_merged_pdf=False)` yields:
- Expected count: `69` files (`71` for `route="macruise", include_merged_pdf=True`, which adds `roi/{lh,rh}.mask.txt` and `PDF/biscuit.pdf`).
- Symmetric difference `discovered ^ expected`: `set()` ($\emptyset$) on all 3 subjects.
