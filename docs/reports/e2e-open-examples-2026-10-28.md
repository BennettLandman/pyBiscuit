# End-to-end run vs `biscuit_FC_v2.2.sif` on the open examples — 2026-10-28

**Branch** `coord-main` (after `5538240`, uncommitted work of this report). **Platform** macOS arm64, Python 3.13,
PyTorch 2.15.0, `--thread 8` (0050952) / `--thread 6` (others). **Inputs** the three ABIDE NYU subjects of
`open-examples` (FreeSurfer 7.4.1 `recon-all` directories), FreeSurfer route, template `mb101`, protocols `braincolor`,
`dkt31`. **Reference** the products `biscuit_FC_v2.2.sif --surf_recon freesurfer` wrote into the same directories.

```bash
pybiscuit run open-examples/<s> e2e/<s> --input-type freesurfer --thread 8 --legacy
python scripts/validation/e2e_open_examples.py e2e/<s> open-examples/<s> --json <s>.e2e.json
python scripts/validation/e2e_open_examples.py --summary *.e2e.json
```

Metrics JSON: [e2e-open-examples-2026-10-28/](e2e-open-examples-2026-10-28/). This is the first run of the whole chain:
every stage consumes the port's own upstream results (unlike `open_examples.py`, which feeds each stage the image's
inputs), so differences accumulate as they would for a user.

## What had to change before the chain could run

1. **Stage bodies wrote placeholders.** `pipeline/bodies.py` (`ae4098d`) caught every exception and wrote 3-vertex meshes
   and zero maps, so `pybiscuit run` reported `complete` regardless. Several calls could never have worked (keyword
   names `surfsmoothing=`, `num_bins=` that the functions do not take; FS route `res.curv` written from the wrong map;
   report written into `stages/`). The bodies were rewritten to follow the spec phase by phase in BISCUIT's file names,
   call only the validated ports, and fail loudly ([port note](../ports/pipeline.md)). A failed stage now exits 1.
2. **`parcellate()` had never run.** It called `remesh.resample()` with arguments that function does not have and
   `write_vtk()` with swapped arguments. Fixed to spec §6.3 T29 (resampling the `%f`-written feature files).
3. **`--legacy`** called `populate_legacy_tree` without its `sources` argument and only printed a warning; it now
   assembles `<out>/legacy/` from the stage attempts.
4. The repo venv had no PyTorch (`[parc]` extra), so the parcellation tests had been skipping locally.

## Result

All three runs complete (wall time 22, 13 and 16 min; quant ≈ 60 % of it, TRACE ≈ 20 %). Table and metrics JSON are from
the rerun of 2026-10-30 with the marching-cubes hull (F4) and the libstdc++ heap order in Gyrification
([lgi-intermediates-2026-10-30](lgi-intermediates-2026-10-30.md)); every other column is unchanged from the first run.

| Subject | Hemi | surf/ mask curve/ identical | sphere.reg median / p99 ° | parc agreement bc / dkt | mid H,S,area,thickness identical | SD r / bias mm | LGI r / bias | SCI r |
|---|---|---|---|---|---|---|---|---|
| 0050952 | lh | yes | 0.164 / 0.671 | 0.9794 / 0.982 | yes | 0.9997 / -0.0392 | 0.9992 / 0.0249 | 0.9956 |
| 0050952 | rh | yes | 0.11 / 0.843 | 0.9951 / 0.9996 | yes | 0.99978 / -0.0425 | 0.9957 / 0.0123 | 0.9957 |
| 0050961 | lh | yes | 0.278 / 1.56 | 0.9846 / 0.9976 | yes | 0.99968 / -0.0188 | 0.9995 / -0.0238 | 0.9962 |
| 0050961 | rh | yes | 0.297 / 2.85 | 0.9512 / 0.9857 | yes | 0.99974 / -0.0265 | 0.9985 / -0.00159 | 0.9961 |
| 0050971 | lh | yes | 0.104 / 0.59 | 0.9996 / 0.9995 | yes | 0.99978 / -0.025 | 0.9991 / -0.0338 | 0.9958 |
| 0050971 | rh | yes | 0.198 / 0.727 | 0.9998 / 0.9997 | yes | 0.99978 / -0.0212 | 0.9975 / 0.00322 | 0.9958 |

ROI tables (`stat/stats.<p>.csv`, 111 braincolor / 65 dkt31 rows, ids identical, no NaN mismatches), worst subject:

| Measure | braincolor r / median rel diff | dkt31 r / median rel diff |
|---|---|---|
| Sulcal depth | 0.9995 / 0.8 % | 0.9998 / 0.4 % |
| Thickness | 0.9976 / 0.3 % | 0.9991 / 0.07 % |
| SCI | 0.9952 / 0.9 % | 0.9937 / 0.8 % |
| LGI | 0.9984 / 1.1 % | 0.9988 / 0.9 % |
| Mean curvature | 0.9900 / 8.3 % | 0.9954 / 2.3 % |
| Shape index | 0.9891 / 10.8 % | 0.9965 / 4.9 % |

(Relative differences of H and S are inflated by ROI means near zero; the per-vertex maps are byte-identical, so the ROI
differences come only from the parcellation.)

## Findings

**F1 — exact stages stay exact end to end.** Job-directory surfaces, masks, thickness, TRACE curves and `.bary` files,
mid-surface H, S and vertex areas are byte-identical to the image for all six hemispheres.

**F2 — parcellation is exact given the image's registration.** Re-running `parcellate` on 0050952 with the image's
`reg/<h>.coeff.txt` reproduces `roi/<h>.parc.{braincolor,dkt31}.txt` exactly (agreement 1.0, both hemispheres and
protocols). The 0.04–4.9 % label disagreement above therefore comes entirely from HSD (F3), and parcellation (B7.1–B7.5)
now has real image evidence.

**F3 — registration differs as Q15 predicts.** Native registered spheres are 0.10–0.30° from the image's (median), as in
the per-stage replay; the worst case is 0050961 rh (p99 2.85°, braincolor 95.1 %). Every registered-space map (`stat/*.reg.*`) inherits this (r 0.97–0.999), as do the ROI
tables via the labels.

**F4 — LGI: fixed by marching cubes in OuterHull (2026-10-29).** The first run gave LGI 4.7-7.1 % below the image in every
hemisphere (r 0.984-0.994). Curves and pial were identical and Gyrification is oracle-validated (B9.4), so the hull was the
suspect. `OuterHull` built its iso-surface with marching tetrahedra, whose extra diagonal cuts make a staircase hull
~18 % larger in area than MATLAB's marching cubes (`isosurface`) on the same volume (0050952 lh: 61,474 vs 52,085 mm^2).
With classic marching cubes ([outer-hull.md](../ports/outer-hull.md)) all six hemispheres give r 0.9957-0.9995 and mean
differences of -0.84 to +0.61 % (p99 |diff| 3.5-9.5 % of the 1-99 % range); registered maps r 0.9955-0.9991; ROI LGI
r >= 0.9984, median relative difference <= 1.1 %. What remains is within Q14's chaotic-row latitude (`strel('ball')`
decomposition). The image's `tmp-lgi-*` intermediates (2026-10-30) show klaplace and Gyrification exact on the image's own
inputs and locate the residual in OuterHull's one-voxel noise ([lgi-intermediates-2026-10-30](lgi-intermediates-2026-10-30.md)).

**F5 — sulcal depth** keeps the known B9.5 residual (r ≥ 0.9997, bias −0.02 to −0.04 mm, max ≈ 1 chamfer step).

**F6 — PyTorch version.** With torch 2.15.0, 12 `tests/test_parc_reference.py` dkt31 cases fail their absolute 1e-4
bound on raw network scores (max 4.4e-3 on scores of magnitude ~10², i.e. ~1e-5 relative); the same tests pass under
torch 2.14.1. Labels are unaffected (F2 ran on 2.15.0). Resolved 2026-10-29 (owner): the bound is now 1e-4 relative to the
peak |score| (validation-tolerances.md); all 21 tests pass on 2.15.0.

**F7 — not covered.** The MaCRUISE route (B11.1) still has no image reference; its surfprep (B5.4–B5.7) is draft. The
report stage calls `generate_report_pdfs` without `tiles` (no surface renders) and was not compared with the image's
PDFs. Runtime is dominated by quant (LGI ~5–10 min per hemisphere).
