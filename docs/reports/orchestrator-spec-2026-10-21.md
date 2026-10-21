# Report: orchestrator spec from the BISCUIT MATLAB source (brief 11) — 2026-10-21

Branch `brief-11-orchestrator-spec`. Docs only; no code changed. Main deliverable: [orchestrator-spec](../reference/orchestrator-spec.md)
(stage order, argv of every external call with its `biscuit.conf` mapping, file flows, output tree, in-MATLAB computation).
This report holds the coverage checklist, the reconciliation with the merged and in-flight ports, the answers to Q3/Q4/Q9, and the
surprises. Proposed follow-up briefs: [proposed-12-21](../agent-briefs/proposed-12-21.md). Plan rows and open questions were updated in the same PR.

## 1. Outcome

| Item | Result |
|---|---|
| Spec for the MaCRUISE route (FreeSurfer and CIVET briefly) | Written; every `system()` call in `GenericRecon` is listed (41 call kinds, T1–T41). |
| B5.1 (`run_postproc_macruise_pipeline.m` + `hemi_separation.m` reverse engineering) | **Done (source-derived).** Evidence: spec §4. |
| B1.3 (output tree + orchestrator spec) | **Done, source-derived; image confirmation still needed** (spec §13, §15). |
| Reconciliation with briefs 04–07 | Tables in §3 below: 17 mismatches, gaps or new findings (plus two hypotheses resolved) filed as issues for the owning components; no code touched. |
| Q3, Q4, Q9 | Facts recorded in [open-questions.md](../open-questions.md); Answer column left empty. |
| Plan rows | Rewritten with source citations: B1.2 (reduced to a confirmation), B1.3 and B5.1 (Done, source-derived), B3.1, B3.3, B5.2–B5.7, B7.1, B7.3–B7.5, B9.1, B9.3, B9.5, B9.6, B10.1, B10.4, B12.3. Annotated: B6.5, B8.1. B9.2/B9.4 left untouched (PR #6 owns them); the top Status section is left to the coordinating session (only Key finding 1 got a dated pointer). |
| Proposed briefs | 12–19 added to the index as *Proposed*. |

Limits. The source is dated eight months after the image; the encrypted orchestrator could not be diffed against it, so every statement is
"the source says". The closed binaries (`MeshProperty`, `MeshDeform`, `MeshContour`) were not run (Linux ELF on a Mac host); only
their `strings` flag lists were read. MATLAB toolbox semantics (`strel('ball')`, `interp3`, `bwdistgeodesic`, `smooth3`, `dmperm`, `DelaunayTri`)
are described from documentation knowledge, not from an executable reference, and are marked as such in the spec.

## 2. Top surprises

1. **BISCUIT does not use MaCRUISE's pial surface at all.** From MaCRUISE it takes only the inner surface (twice: native and MNI copy), the raw T1 and `SEG_refine`. The white surface is re-derived and the pial
   surface *and thickness* are produced by FreeSurfer's `mris_make_surfaces` (`-orig_pial white -nowhite`) from the T1 intensity (spec §4.6). A PyMaCRUISE outer-surface pial is therefore a deliberate redesign
   (R-class, Q3), not a fidelity choice.
2. **The MNI surface exists only to place a sagittal cutting plane** (candidate planes within ±0.5 mm of the mean x of the smoothed MNI mesh; a closed binary `MeshContour` then chooses one). The native surface supplies the geometry.
   Hemisphere meshes include a **harmonically filled medial-wall patch** (Delaunay disk per hole), so per-vertex files have more vertices than the cortex; `lh.mask.txt` marks the patch (spec §4.2).
3. **All four HSD calls are single-property, chained by coefficient files, with a rising `--icomesh` level (4→7)**, not one multi-property call; `--tmpVar` is used in the last two only; both HSD optimiser branches (gradient descent for levels < 6, damped Newton for ≥ 6) run in every registration;
   the initial guess is on in all four, with radius-100 `--icomesh` spheres and no `--resample` — the exact condition brief 05 flagged as producing a zero template map in the guess. For `rh`, call 2 has the template and subject spheres swapped relative to `lh` (a source typo also present in the CIVET route).
4. **Output surfaces are in the FreeSurfer surface frame**, not scanner RAS: the `.asc` goes through `MeshDeform -t c_ras` into a FreeSurfer `orig` surface and every later surface is converted back to VTK unchanged. The sign of `-t` is not visible; one image run decides (spec §4.4).
5. **TRACE is run on the pial surface for sulci and on the white surface for gyri**, with `trace_r` feeding both `--epRad` and `--nhDist` and `trace_delta` feeding `--lsTh`; all other TRACE options stay at their defaults (the real-cortex seeds). Only the sulcal curves (pial) are used by the statistics; the gyral curves only feed LGI.
6. **Parcellation features are not the registration features**: `H` is the plain negated mean curvature (not `−4H`), `iH` and `sulc` are re-read from FreeSurfer curv files, each is z-scored and soft-clamped twice, and an extra rigid `HSD -d 0 --noguess --icosahedron 5` against the protocol's `?h.curv.mean.txt` refines the alignment before the network (spec §6). Graph cut and ROI cleanup run on the **white** surface and on every fold, and DKT31 fusion is a probability-weighted vote followed by a second graph cut and cleanup (§6.6).
7. **Sulcal depth is its own voxel pipeline** (1 mm grid, filled pial solid, smooth3 + `strel('ball',15)` closing, `bwdistgeodesic` through the free space, `interp3`), not the `OuterHull` hull of LGI; `SD` and `LGI` are computed on the pial (`gray`) surface and copied to the mid surface by index (spec §9.2).
8. **Dead and fragile code.** The `mri_normalize -aseg` brain refinement never runs; `flag` is undefined on resume; `roi_correction` can call `gcut_` with no candidate labels; the hemisphere split and `roi_correction` rely on `dmperm` block order; the statistics PDF and `*.mid.stat.vtk` are never produced; the cortical mask is not applied in any statistic.
9. **Licensing.** GCMex/GCoptimization/Boykov–Kolmogorov are GPL v2 + "research only" + patent notice (not usable); `VOXELISE.m` is Aitkenhead's; 17 FreeSurfer-licensed MATLAB helpers sit in the tree (see §6).

## 3. Reconciliation with the ports

Legend: **OK** = consistent; **GAP** = port untested or missing for what BISCUIT does; **MISMATCH** = port assumption contradicted by the source; **NEW** = behaviour the port notes did not know.
Each row is an issue for the named component; code was not touched.

### 3.1 TRACE (brief 04, merged, [trace.md](../ports/trace.md))

| # | BISCUIT passes (spec) | Port supports / assumed | Verdict | Issue for the component |
|---|---|---|---|---|
| T-1 | `--epRad r --nhDist r --lsTh d` with `r=trace_r`, `d=trace_delta` (§8) | Port note hypothesis: `trace_r → --nhDist` (probably also `--epRad`), `trace_delta → --lsTh`, "BISCUIT may simply not pass them" | **Resolved** (both passed, same values; defaults 4/4/2.5 equal them) | Update the hypothesis paragraph of `trace.md`; add the BISCUIT call as a named CLI test |
| T-2 | `--bary --noVtk --nThreads n`, no other option (seeds −0.05/0.03, prune 5, smoothing 0/3 at defaults) | All flags present; synthetic fixtures needed non-default seeds | **OK** | Test with the default seeds on a cortex-scale surface (the template `lh.white` is the only one so far) |
| T-3 | Sulcus on `lh.gray.vtk` (FreeSurfer pial), gyrus on `lh.white.vtk` | Validated on `reg/mb101/lh.white.vtk` and folded fixtures only | **GAP** | Need a pial-type surface (or image golden) and a mesh that contains a medial-wall patch |
| T-4 | Hemisphere meshes are closed, include the patch; curves are not masked | Not exercised | **GAP** | Add a case with a flat patch region; confirm curves are allowed on it |
| T-5 | Output prefix `…/lh.gray`, `…/lh.white` → `lh.gray.scurve(.bary)`, `lh.white.gcurve(.bary)` | `.scurve/.gcurve/.bary` byte-identical to the oracle | **OK** | — |
| T-6 | Gyral run on every subject | `GyralCurve::refineCurves` uninitialised `current`; default `exit` matches the clang oracle; image compiler unknown | **GAP, now production-relevant** | Resolve on the first Linux run (spec §15 item 8) |
| T-7 | Inputs are VTK written by `write_vtk` (`%f`, 6 decimals) | Reader accepts ASCII; precision of fixtures differs | **OK** | Round a fixture to 6 decimals and check the curves are unchanged |
| T-8 | `.bary` consumed by `read_bary` (§10) and `Gyrification -s/-g` | Format identical to upstream | **OK** | — |

### 3.2 HSD (brief 05, merged, [hsd.md](../ports/hsd.md))

| # | BISCUIT passes (spec §5.3) | Port supports / assumed | Verdict | Issue for the component |
|---|---|---|---|---|
| H-1 | Four chained calls, one property each, `-c C C --outputcoeff C C`, `--icomesh` 5120→20480→81920→327680 | Hypothesis in `hsd.md`: one call with three properties, `--weight`, concatenated `--tmpVar`, `--icosahedron L` | **MISMATCH** | Replace the hypothesis section; add a regression test that replays the four-call chain (degree 10) on a small case; both optimiser branches are exercised |
| H-2 | `-s <template 327680> <subject sphere of 20480 / 81920 / 327680 vertices>`; property files have 163,842 lines | Fixtures use equal sphere/property lengths | **GAP** | HSD reads the first `nVertex` lines (`HSD.cpp:893-960`); test property files longer than the mesh, template and subject at different resolutions |
| H-3 | `--icomesh sphere_N.vtk` (radius **100**), no `--resample`, guess on, in all four calls | Port reproduces "template mean map 0 in the guess" and asks to verify BISCUIT's call (report item 4) | **NEW, confirmed** | Make the guess behaviour for exactly this configuration a golden against the image; it decides whether the guess is effectively disabled in production |
| H-4 | Same coefficient file as input and output for template and subject (`-c C C --outputcoeff C C`) | Not tested | **GAP** | Fixed subject neither reads nor writes (`HSD.cpp:170-175,761`); add a CLI test with identical in/out paths |
| H-5 | `--tmpVar` = a single-property variance file, only in calls 3–4 (`sulc.var.txt`, `curv.var.txt`) | `--tmpVar` property-major file supported | **OK** | — |
| H-6 | `-d 10`, `--idprior 625`, `--maxIter 20`, `--gres 3`, no `--weight`, no `--nThreads` (env) | Defaults match the wrapper | **OK** | — |
| H-7 | `rh` call 2: `-s S81920 S327680` (reverse of `lh`) | n/a | **NEW** | Treat as an input to the image run (does the image do the same?); until then support both orders |
| H-8 | Parcellation alignment `HSD -s S327680 <native sphere> -o <same> <same> -p <protocol>/curv.mean.txt <H.txt> --fixedSubjects 0 --noguess -d 0 --icosahedron 5`, native-resolution mesh, `-o` = input | Tests use degrees 6/8 | **GAP** | Add `-d 0` (rotation-only) with `--noguess` and an output path equal to the input |
| H-9 | `--icosahedron 5` and `--icomesh` levels 4–7 | Level 4 and 5 GD, 6 and 7 Newton covered; `--icosahedron 0-2` UB only | **OK** | — |
| H-10 | Registered sphere written by call 4 via `-o` on `sphere_327680` | `-o` supported | **OK** | — |

### 3.3 Parcellation (brief 06, merged, [parcellation-net.md](../ports/parcellation-net.md))

| # | BISCUIT does (spec §6) | Port supports / assumed | Verdict | Issue for the component |
|---|---|---|---|---|
| P-1 | Loader reads `*.res.{H,iH,sulc}.txt`, channel order `H, iH, sulc`, first 10,242 values | `predict` takes the three channels as given, order `H, iH, sulc` | **OK** | — |
| P-2 | Network output is raw scores; `gcut_` applies its own softmax (`−log` as data cost) | Raw scores returned (no softmax) | **OK** | — |
| P-3 | Class index ↔ label id from the hemisphere's `Left`/`Right` rows of `label.csv`, class 1 = unknown (0) | `nclass` 50 / 32 consistent; mapping not implemented | **GAP** | Implement in `infer`/fusion from `label.csv` (not from a hard-coded table) |
| P-4 | DKT31: 5 models per hemisphere; per-model gcut + roi_correction; weighted vote with chosen-label probabilities; `log(prob+eps)`; second gcut + roi_correction (§6.6) | `fusion.py` is an interface; "rule pending B7.1" | **Resolved** | Implement `fuse` as specified; braincolor (1 model) skips fusion |
| P-5 | Features: `−H`, recomputed `iH`, `sulc` from curv files, z-score + double soft clamp, `%f` text round trip (§6.2) | Not ported | **GAP** (B7.1) | New brief |
| P-6 | Rigid alignment: `SphericalRemesh --deg 0 --deform` with the HSD coefficients, then `HSD -d 0 --noguess --icosahedron 5` (§6.3); CIVET uses degree 10 | n/a | **GAP** | New brief; depends on H-8 |
| P-7 | Graph cut on the white surface (Potts × edge length, float32, α-expansion label order 0…L−1, until no decrease) and the `roi_correction` loop (§6.7–6.8) | B7.4/B7.5 not started | **GAP** | New brief; max-flow must be licence-clean |
| P-8 | `isphere` = `icosphere_5.vtk` (10,242 vertices), `osphere` = rigidly aligned native sphere; `SphericalRemesh -p … -r osphere -s isphere --outputProperty … --quiet` | `resample_to_native` requires exactly 10,242 vertices | **OK** | — |
| P-9 | Checkpoints: image runs Python 2.7/3.6 with torch 0.4.0 (Singularity recipe); port oracle torch 2.14 | Bit-identical scores shown | **OK** | Record the image's torch version in the oracle note |

### 3.4 LGI, OuterHull, klaplace (brief 07, PR #6 open, [lgi.md](../ports/lgi.md), [outer-hull.md](../ports/outer-hull.md))

| # | BISCUIT does (spec §9) | Port supports / assumed | Verdict | Issue for the component |
|---|---|---|---|---|
| L-1 | `lgi` script → `Gyrification -i pial -o … --outer <hull.corr> -s lh.gray.scurve.bary -g lh.white.gcurve.bary -m 316.000000 --speed 0.200000 --refHullArea 77100.000000 --nThreads N` | All flags and `.bary` readers present; maps `<out>.lgi.map.<int(delta)>.txt` | **OK** | Add a CLI test with the exact argv incl. the `%f`-formatted numbers |
| L-2 | Surfaces are closed hemisphere meshes with the medial patch | Closed-manifold only (raises otherwise) | **OK** | — |
| L-3 | Hull = `OuterHull` of the pial surface, `--outer` from klaplace correspondence; LGI kernel scales with hull area | `OuterHull` provisional, klaplace out of scope (licence, Q7) | **GAP** | Hull area on the pial surface from the image is the single most useful number; plan a licence-clean correspondence |
| L-4 | Sulcal depth uses a *different* hull (1 mm voxel grid, `scale = 1`, 20-voxel padding) built by the same `VOXELISE`/smooth3/`strel('ball',15)` steps | `quant/_voxelise.py`, closing, `fminlbfgs` exist for the 256-grid `OuterHull` | **NEW** | `VOXELISE.m` is byte-identical to the LGI copy; the SD brief can reuse `_voxelise.py` |
| L-5 | Image `OuterHull`/`strel`/`isosurface` behaviours | Three unverifiable MATLAB behaviours listed by brief 07 | **GAP** | Unchanged; the SD pipeline now shares the `strel`/`imclose` uncertainty |
| L-6 | `lgi` copies curves from `<surface dir>/../curve/`; `Gyrification` reads `.bary` only | n/a | **OK** | — |
| L-7 | Heap flavour / `acosf` (libstdc++, glibc) | Both heap flavours implemented; `acosf` pinned to fdlibm, provisional | **GAP** | Needs image goldens (unchanged) |

### 3.5 Other tools BISCUIT calls (merged earlier)

| Tool | BISCUIT call | Port status | Verdict |
|---|---|---|---|
| `SphericalRemesh` | `-s -r -p --outputProperty` (multi-`-p` in parc), `-c … --deform … --deg 10`, `-i -o`, `-s … -c … -r … -o`, `--deg 0`, `--quiet` | All flags in `spherical-remesh.md`; deform-only and `--deg` covered | **OK**; `-p` longer than the sphere is fine, shorter is fatal |
| `MeshProperty` | `-m [-s] [-v] [-q] --surfsmoothing N --tensorsmoothing N [--datasmoothing 25 --prctile 0.1 --normbefore --normafter --invert]` (§3 T17, T19, T25, T26, T31, T38) | B3 not started | **GAP**: the full flag combinations BISCUIT uses are now listed (needed by B3.1) |
| `MeshDeform` | `-t x y z`, `--smooth N -i X -o X` (in place) | B3.3 not started | **GAP**: only translate and smooth are used (no matrix, scale, sphere, noise) |
| `MeshContour` | `<vtk> 1 0 0 d1,…,d101,` → face ids + plane index | not started | **GAP**: selection rule unknown |
| `ShapeComplexityIndex` | `-i mid.vtk -o … -k 3.000000 -s 5` | not started | **GAP** (licence, Q7) |
| `klaplace`, `OuterHull` | inside `lgi` only | see 3.4 | — |
| `mri_*`, `mris_*` | see spec §3 T1–T16 | replacement class | the FreeSurfer 6 steps whose outputs matter are listed in §4 |

## 4. Open questions Q3, Q4, Q9

Summaries; the dated entries are in [open-questions.md](../open-questions.md).

* **Q3 (which FreeSurfer steps run and why).** `mri_convert`/`mri_normalize` produce a conformed, normalised T1 and the conformed `brain.finalsurfs`, `wm`, `filled`, `aseg` volumes synthesised from `SEG_refine` labels (spec §4.5);
  `mris_make_surfaces` twice (white under aseg guidance, then pial + thickness from the white surface by intensity); `mris_inflate` (inflated surface + `sulc`); `mris_sphere`. What each output is used for downstream:
  white (registration, TRACE gyri, parc), pial (TRACE sulci, SD, LGI, mid surface), thickness, `sulc` (registration and parc feature), inflated (iH feature, figures), sphere (HSD input). FreeSurfer's intensity-based
  pial and its thickness are *replaced* data in PyMaCRUISE terms: PyMaCRUISE already has an outer surface and a thickness.
* **Q4 (what the MNI surface is for).** Only the sagittal cutting plane. Any rigid frame in which the mid-sagittal fissure is perpendicular to x and the smoothed mesh's mean x lies inside the fissure works; a PyMaCRUISE-side estimate
  (SEG_refine L/R labels) is sufficient, with the native frame used for the final left/right test (smaller mean x = left).
* **Q9 (legacy output names).** The complete tree is in spec §13.1 (names, formats, what is copied from which scratch directory). Naming irregularities to keep in a `legacy/` view: `gray` for pial; LGI/SD named `gray`, other measures `mid`;
  `stat/<h>.reg.thickness.txt`; `surf/hemi_map.txt`; PDFs `pdf/{reg.<t>,parc.<p>,curve,quant}.pdf` and `PDF/biscuit.pdf` (image wrapper).

## 5. Coverage checklist (all 83 `.m` files)

Status legend: *read* = read in full; *read (relevant parts)*; *skimmed* = grepped/head-read for the facts used; *diff-identical* = byte-identical (checked with `diff`) to an upstream copy already reviewed in brief 07;
*not read: irrelevant* = tooling or unused code; *not read: FreeSurfer licence* = FreeSurfer-licensed helper, skipped under the clean-room rule; *viewed for format facts* = FreeSurfer-derived helper seen once (see §6).
Counts: read 39, read (relevant parts) 2, skimmed 12, diff-identical 3, viewed for format facts 5, not read (irrelevant) 5, not read (FreeSurfer licence) 17 = 83.

Non-`.m` files used: `GenericRecon/biscuit/{biscuit.conf,check_input,run_biscuit}`, `biscuit_mc.conf`, `lgi` (full), `Singularity`, `README`, `VERSION`; `SurfPostproc/common/parc/{test.sh,test.py,loader_individual.py}` (read);
`model.py`, `meshcnn/*`, `cpuModel.py` (brief 11 states they are identical to the image CTF; not re-diffed); `gcut/{GraphCutConstrSparse.cpp,GCoptimization.cpp,graph.h,GraphCut.m,LICENSE.TXT,GC_README.txt}` (relevant parts); the extracted image's `run_biscuit`, `biscuit.conf`, `VERSION`, `parc/*/label.csv`, `sphere/*.vtk` heads, and the `strings` of `MeshProperty`, `MeshDeform`, `MeshContour`.


## 6. Disclosure: FreeSurfer-licensed files

AGENTS.md forbids reading or translating FreeSurfer source. The BISCUIT tree vendors FreeSurfer's MATLAB helpers under `SurfPostproc/macruise/FreeSurfer/` and `SurfPostproc/common/matlab/FreeSurfer/`.
This session **did not read 17 of them** (checklist) and **viewed five** (`fread3`, `fwrite3`, `write_curv`, `write_surf`, `write_surf3`) once, while following the call chain of `write_surf`/`write_curv`/`write_surf3` from the pipeline, before the clean-room concern was
applied to the helper folder. What was taken from them: only the public FreeSurfer file-format facts (big-endian; triangle-surface magic 16777214; curv magic 16777215; counts as int32; coordinates float32; faces int32 0-based)
and the fact that `write_surf3` appends volume-info tags (`valid`, `filename`, `volume`, `voxelsize`, `xras`, `yras`, `zras`, `cras`). Nothing was copied into the repository, and the spec states the format facts in the author's words.
If the owner considers the five views a clean-room breach, the sentences in spec §4.4 and §14 item 10 are the only places that depend on them, and the later porting sessions can be told not to open those folders.

## 7. Proposed next briefs

See [proposed-12-21](../agent-briefs/proposed-12-21.md): 12 MaCRUISE hemisphere separation (B5.2), 13 MeshProperty/MeshDeform black-box port and feature maps (B3, B9.1), 14 FreeSurfer-replacement surface prep (B5.3–B5.7, after Q3),
15 parcellation features + alignment + fusion (B7.1, B7.3), 16 graph cut + ROI cleanup (B7.4–B7.5), 17 sulcal depth (B9.5), 18 statistics, legacy view and pipeline wiring onto brief 08 (B10.1, B10.4), 19 PDF/QA layout (B12.3),
and the two queued follow-ups: 20 LGI on the ported TRACE, 21 HSD four-call replay.

## 8. Not done / follow-ups

* No Linux run: all "must confirm" items are in spec §15.
* `docs/source-map.md` still says the orchestrator is encrypted and lists inferred roles; it was left unchanged because brief 11 asks for docs under `docs/reference/`, the plan and the open questions. A one-line pointer to the spec would be useful (owner or the next coordinating session).
* The trace/hsd/parcellation port notes contain statements this report contradicts (T-1, H-1, P-4); they belong to merged briefs and were not edited here.

## Coverage table

| # | File (under `BISCUIT-master/`) | Status | Note |
|---:|---|---|---|
| 1 | `BISCUIT/FreeSurfer/biscuit.m` | skimmed | diffed against GenericRecon: same minus the MaCRUISE and CIVET branches |
| 2 | `BISCUIT/FreeSurfer/build.m` | not read: irrelevant | `mcc` build script |
| 3 | `BISCUIT/GenericRecon/biscuit.m` | read | entry point; spec §1–§2, §8–§10 |
| 4 | `BISCUIT/GenericRecon/build.m` | read | `mcc` build script; lists compiled folders and the `verLessThan(8.4)` branches |
| 5 | `CurveExtraction/IO/read_curve.m` | read | curve text reader (0-based vertex lines); §8 |
| 6 | `CurveExtraction/Visualization/SurfStatViewData3.m` | skimmed | view layout of the curve figures (axes positions, view angles); §11 |
| 7 | `CurveExtraction/Visualization/run_GeneratePDF_curvExtraction.m` | read | curve page; §11 |
| 8 | `CurveExtraction/Visualization/vecrast_curve.m` | not read: irrelevant | curve variant of the `vecrast` rasteriser |
| 9 | `CurveExtraction/curve_extract_pipeline.m` | read | TRACE argv; §8 |
| 10 | `SurfPostproc/civet/IO/read_obj.m` | read | CIVET `.obj` reader; §12 |
| 11 | `SurfPostproc/civet/run_postproc_civet_pipeline.m` | read | CIVET route; §12 |
| 12 | `SurfPostproc/civet/surf_reg_civet.m` | skimmed | tool calls compared with `surf_reg_macruise.m` (copy, same `rh` swap); §12 |
| 13 | `SurfPostproc/common/matlab/FreeSurfer/fread3.m` | not read: FreeSurfer licence (clean-room) | 3-byte integer reader |
| 14 | `SurfPostproc/common/matlab/FreeSurfer/read_curv.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer curv reader; format is public |
| 15 | `SurfPostproc/common/matlab/FreeSurfer/read_surf.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer surface reader; format is public |
| 16 | `SurfPostproc/common/matlab/Visualization/SurfStatViewData_parc.m` | read | parcellation figure viewer; §11 |
| 17 | `SurfPostproc/common/matlab/Visualization/SurfStatViewData_reg.m` | skimmed | registration figure viewer: six views, `redbluecmap(11)`, grid lines drawn with `patchline` |
| 18 | `SurfPostproc/common/matlab/Visualization/run_GeneratePDF_surf_postproc.m` | read | parcellation page; §11 |
| 19 | `SurfPostproc/common/matlab/Visualization/run_GeneratePDF_surf_reg.m` | read | registration page; §11 |
| 20 | `SurfPostproc/common/matlab/gcut/GraphCut.m` | read (relevant parts) | MATLAB front end of GCMex: `open`, `set`, `expand`; §6.7 |
| 21 | `SurfPostproc/common/matlab/gcut/compile_gc.m` | read | `mex` build lines |
| 22 | `SurfPostproc/common/matlab/gcut/compute_adj.m` | skimmed | scratch script with a hard-coded path; not called |
| 23 | `SurfPostproc/common/matlab/gcut/extended/compact_a_expand_mex.m` | not read: irrelevant | QPBO extension; not called by BISCUIT |
| 24 | `SurfPostproc/common/matlab/gcut/extended/mexall.m` | not read: irrelevant | build script of the QPBO extension |
| 25 | `SurfPostproc/common/matlab/gcut/sparse_adj_matrix.m` | not read: irrelevant | helper for grid graphs; no callers (grep) |
| 26 | `SurfPostproc/common/matlab/parc/extract_features.m` | read | §6.2 |
| 27 | `SurfPostproc/common/matlab/parc/gcut.m` | read | §6.7 |
| 28 | `SurfPostproc/common/matlab/parc/gcut_.m` | read | §6.7 |
| 29 | `SurfPostproc/common/matlab/parc/parc.m` | read | §6 |
| 30 | `SurfPostproc/common/matlab/parc/roi_correction.m` | read | §6.8 |
| 31 | `SurfPostproc/freesurfer/run_postproc_fs_pipeline.m` | read | FreeSurfer route; §12 |
| 32 | `SurfPostproc/freesurfer/surf_reg_fs.m` | read | unreferenced older registration; §12 |
| 33 | `SurfPostproc/freesurfer/surf_reg_fs2.m` | read | registration used by the FreeSurfer route; §12 |
| 34 | `SurfPostproc/freesurfer/surf_reg_fs3.m` | skimmed | diffed against `surf_reg_fs2.m` (path setup differs); unreferenced |
| 35 | `SurfPostproc/macruise/FreeSurfer/MRIfspec.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer MRI I/O helper |
| 36 | `SurfPostproc/macruise/FreeSurfer/MRIread.m` | not read: FreeSurfer licence (clean-room) | used as `MRIread(file,false)` for header, `vol`, `c_ras`; behaviour inferred from the call sites |
| 37 | `SurfPostproc/macruise/FreeSurfer/MRIwrite.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer MRI writer |
| 38 | `SurfPostproc/macruise/FreeSurfer/fread3.m` | viewed for binary-format facts only (FreeSurfer-derived) | 3-byte big-endian integer |
| 39 | `SurfPostproc/macruise/FreeSurfer/fsgettmppath.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 40 | `SurfPostproc/macruise/FreeSurfer/fwrite3.m` | viewed for binary-format facts only (FreeSurfer-derived) | 3-byte big-endian integer |
| 41 | `SurfPostproc/macruise/FreeSurfer/load_analyze.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 42 | `SurfPostproc/macruise/FreeSurfer/load_mgh.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 43 | `SurfPostproc/macruise/FreeSurfer/load_nifti.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 44 | `SurfPostproc/macruise/FreeSurfer/load_nifti_hdr.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 45 | `SurfPostproc/macruise/FreeSurfer/read_asc.m` | read | BISCUIT-authored `.asc` reader; §4.1 |
| 46 | `SurfPostproc/macruise/FreeSurfer/read_curv.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer curv reader |
| 47 | `SurfPostproc/macruise/FreeSurfer/save_mgh.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 48 | `SurfPostproc/macruise/FreeSurfer/strlen.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 49 | `SurfPostproc/macruise/FreeSurfer/vox2rasToQform.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 50 | `SurfPostproc/macruise/FreeSurfer/vox2ras_0to1.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 51 | `SurfPostproc/macruise/FreeSurfer/vox2ras_tkreg.m` | not read: FreeSurfer licence (clean-room) | FreeSurfer helper |
| 52 | `SurfPostproc/macruise/FreeSurfer/write_curv.m` | viewed for binary-format facts only (FreeSurfer-derived) | curv writer (magic 16777215, big-endian); §4.4 |
| 53 | `SurfPostproc/macruise/FreeSurfer/write_surf.m` | viewed for binary-format facts only (FreeSurfer-derived) | triangle surface writer (magic 16777214); §4.4 |
| 54 | `SurfPostproc/macruise/FreeSurfer/write_surf3.m` | viewed for binary-format facts only (FreeSurfer-derived) | BISCUIT variant that appends the volume-info tags (`cras` etc.); §4.4 |
| 55 | `SurfPostproc/macruise/run_postproc_macruise_pipeline.m` | read | §4 |
| 56 | `SurfPostproc/macruise/surf_reg_macruise.m` | read | §5 |
| 57 | `SurfPostproc/macruise/tessellation/circle_patch.m` | read | §4.2 |
| 58 | `SurfPostproc/macruise/tessellation/connectedComp.m` | read | §4.2 |
| 59 | `SurfPostproc/macruise/tessellation/hemi_separation.m` | read | §4.2 |
| 60 | `SurfPostproc/macruise/tessellation/surfNormalCorrection.m` | read | §4.2 |
| 61 | `SurfQuant/Gyrification/gyrification.m` | read | §9.3 |
| 62 | `SurfQuant/SCI/sci.m` | read | §9.4 |
| 63 | `SurfQuant/SulcalDepth/sulcal_depth.m` | read | §9.2 |
| 64 | `SurfQuant/VOXELISE/COMPUTE_mesh_normals.m` | diff-identical to a reviewed copy | identical to `LocalGyrificationIndex/matlab/VOXELISE/` (`diff -q`) |
| 65 | `SurfQuant/VOXELISE/CONVERT_meshformat.m` | diff-identical to a reviewed copy | identical to the LGI repository copy (`diff -q`) |
| 66 | `SurfQuant/VOXELISE/VOXELISE.m` | diff-identical to a reviewed copy | identical to the LGI repository copy (`diff`); Aitkenhead |
| 67 | `SurfQuant/Visualization/SurfStatViewData_quant.m` | skimmed | quant figure viewer: colour binning with `histc`, masked vertices black; §11 |
| 68 | `SurfQuant/Visualization/run_GeneratePDF_quant.m` | read | quant page; §11 |
| 69 | `SurfQuant/surf_quant_pipeline.m` | read | §9 |
| 70 | `SurfQuantStat/IO/read_bary.m` | read | §10 |
| 71 | `SurfQuantStat/Visualization/render_3D_surfaces_onecolor.m` | read | statistics figure renderer (dead path) |
| 72 | `SurfQuantStat/Visualization/run_GeneratePDF_property.m` | read (relevant parts) | head and tail; statistics PDF is dead code; §11 |
| 73 | `SurfQuantStat/Visualization/saveTightFigure.m` | skimmed | figure-to-PDF helper |
| 74 | `SurfQuantStat/surf_quant_stat_pipeline.m` | read | §10 |
| 75 | `common/alternatives/allwords.m` | skimmed | string splitter used for `parc_protocol` on MATLAB < 8.2 (`biscuit.m:69`) |
| 76 | `common/alternatives/paruly.m` | skimmed | parula look-alike used when MATLAB < 8.4 |
| 77 | `common/alternatives/patchline.m` | skimmed | MathWorks line-patch helper used by the registration figure |
| 78 | `common/matlab/IO/read_vtk.m` | read | §4.1 |
| 79 | `common/matlab/IO/write_property.m` | read | §9.5 |
| 80 | `common/matlab/IO/write_vtk.m` | read | §4.1 |
| 81 | `common/matlab/Visualization/genLaTeX_header.m` | read | §11 |
| 82 | `common/matlab/Visualization/vecrast.m` | skimmed | header only: 300 dpi raster inside a vector PDF |
| 83 | `common/matlab/read_config/read_config.m` | read | §1.2 |
