# BISCUIT orchestrator specification (from the MATLAB source)

**Derived 2026-10-21** by brief 11 from the unencrypted BISCUIT source Ilwoo Lyu supplied
(`~/dev/biscuit/BISCUIT-master/`, outside git, files dated 2022-02-10). This document is the
de-facto spec for the encrypted `biscuit.m` family. It replaces the trace-based plan of B1.3;
an image run (Q1) still has to **confirm** it, because the source is dated eight months after
the image build (`biscuit_FC_v2.2.sif`, 2021-06-25) and the image remains the acceptance
reference ([AGENTS.md](../../AGENTS.md)).

Companion documents: the coverage checklist, the per-port mismatch tables and the surprises are in
[orchestrator-spec-2026-10-21](../reports/orchestrator-spec-2026-10-21.md); proposed next briefs in
[proposed briefs](../agent-briefs/proposed-12-19.md).

## 0. How to read this

* **Variant used.** `BISCUIT/GenericRecon/` (it holds the MaCRUISE and CIVET branches). `BISCUIT/FreeSurfer/biscuit.m`
  is the FreeSurfer-only entry point (§12). Paths below are relative to `BISCUIT-master/` and cited as `file:line`.
* **Image vs source (checked 2026-10-21).** `GenericRecon/biscuit/check_input` is byte-identical to the image's.
  `GenericRecon/biscuit/run_biscuit` differs in exactly one place: the image appends the PDF merge (§11). The
  image's `biscuit.conf` differs only in the four `mc_*` paths (flat names: `target_image_GMimg_innerSurf.asc`,
  `target_image_GMimg_innerSurf_mni.asc`, `T1.nii.gz`, `SEG_refine.nii.gz`; source: `Surfaces_FreeView/…`,
  `Surfaces_MNI/…`, `Raw/T1.nii.gz`, `SegRefine/SEG_refine.nii.gz`) and `reg_temp = mb101` is the default
  in both (`biscuit_mc.conf` in the source says `oasis45`, the MASI production value). Parcellation `test.sh`,
  `model.py`, `meshcnn/ops.py` are identical to the image CTF (brief 11 context). The compiled orchestrator in the
  image could not be diffed against the source; the HSD/parc/TRACE calls below are **source**, not image, facts.
* **Licence.** BISCUIT is MIT, © 2019 Ilwoo Lyu. Excerpts here are limited to single lines of argv
  construction with their location. Third-party material inside the tree is *not* MIT and must not be
  copied: `SurfPostproc/macruise/FreeSurfer/*` and `SurfPostproc/common/matlab/FreeSurfer/*` carry
  FreeSurfer licence headers (clean-room rule, AGENTS.md); `SurfPostproc/common/matlab/gcut/` is Bagon's GCMex
  with Veksler's GCoptimization and Boykov–Kolmogorov max-flow (GPL v2 text plus "research only" terms and a
  patent notice, `gcut/LICENSE.TXT`, `GC_README.txt`); `VOXELISE.m` is Aitkenhead's; `vecrast.m`, `patchline.m`,
  `paruly.m` and `allwords.m` are File Exchange code. See §14.
* **Notation.** `$ROOT` = `$BISCUIT_ROOT` (`/opt/biscuit` in the image); `$OUT` = output directory;
  `$TMP` = `$OUT/.tmp`; `<h>` = `lh` or `rh`; "native" = the MaCRUISE whole-brain frame (scanner RAS);
  `sphere_N` = `$ROOT/sphere/sphere_N.vtk`.

## 1. Invocation and configuration

### 1.1 Wrapper (`BISCUIT/GenericRecon/biscuit/run_biscuit`)

`run_biscuit [options] <input_dir> <output_dir>`; both directories must exist (`run_biscuit:160-161`).
Options overwrite `biscuit.conf` values by `sed` substitution on the conf line (first textual match of the old value on
that line, so `--trace_delta 3` on `trace_delta = 2.5` becomes `3`): `--proj --subj --sess --thread --surf_recon --reg
--parc --trace_r --trace_delta --curv_smooth --lgi_kernel --lgi_aniso --lgi_ref --sci_kernel --sci_bins`.
It writes the effective conf to `$TMP/biscuit.conf` after deleting the lines of the *other* surface-recon routes
(prefix filters `fs_`, `mc_`, `cv_` and their `% freesurfer|macruise|civet` comment headers, `run_biscuit:185-193`), runs `check_input`, `cd $TMP`, then
`xvfb-run -a --server-args="-screen 0 1024x768x24 -ac +extension GLX" biscuit $IN $OUT $CONF` (`run_biscuit:182-199`), and removes `$TMP`.
`check_input` verifies, for the template `reg/<reg_temp>/`: the twelve `{lh,rh}.{iH,sulc,curv}.{mean,var}.txt`; for each protocol
`parc/<p>/`: `label.csv lh.pth rh.pth lh.curv.mean.txt rh.curv.mean.txt` (warnings if `lut.csv` or `url.txt` are missing).
Environment (Singularity `%environment`): `BISCUIT_ROOT=/opt/biscuit`, `PATH` includes `/opt/biscuit:/opt/biscuit/bin`,
`FREESURFER_HOME=/opt/biscuit`, `SUBJECTS_DIR=''`, `OMP_NUM_THREADS=1`.

### 1.2 Configuration file and defaults

`common/matlab/read_config/read_config.m`: `name = value` lines; `%` starts a comment line; a value that `str2double`
accepts is numeric, anything else a string; keys repeated become cell arrays; blank value starts a matrix block
(unused). Values are read by `biscuit.m`; **each numeric parameter has a hard-coded fallback** used when the key is absent,
empty or not numeric (`biscuit.m:328-336`, `372-396`), so a conf without the keys behaves like the defaults.

| conf key (default) | consumed at | effect / exact argv fragment |
|---|---|---|
| `project`, `subject`, `session` (empty) | `biscuit.m:48-50` | strings in the PDF headers only |
| `omp_num_threads` (1) | `biscuit.m:31-35` | `setenv('OMP_NUM_THREADS', n)`; `CurveExtraction --nThreads n` (§8); parcellation batch width (§6.5); inherited by `HSD`, `Gyrification`, `klaplace` through the environment |
| `surf_recon` (`freesurfer` in the shipped conf; `macruise` for this spec) | `biscuit.m:138,192,222` | selects the branch; case-insensitive |
| `mc_surf`, `mc_surf_mni`, `mc_rawT1`, `mc_rawSeg` | `biscuit.m:198-201` | relative to `<input_dir>`; `.asc` or `.asc.gz` (§4.1) |
| `surf_smooth` (0) | `biscuit.m:133-137` → `run_postproc_macruise_pipeline.m:248-254` | `MeshDeform --smooth N -i <h>.white.vtk -o <h>.white.vtk` and the same for `<h>.gray.vtk`, in place, before inflation and the mid surface |
| `reg_temp` (`mb101`) | `biscuit.m:67`, `surf_reg_macruise.m:16` | template directory `$ROOT/reg/<reg_temp>/`; PDF name `reg.<reg_temp>.pdf` |
| `parc_protocol` (`braincolor,dkt31`) | `biscuit.m:68-72` | spaces removed, split on `,`; one parcellation, one PDF and one `stats.<p>.csv` per entry |
| `trace_r` (4) | `biscuit.m:328-336` | `eprad = nhdist = trace_r` → `--epRad <r> --nhDist <r>` (§8) |
| `trace_delta` (2.5) | `biscuit.m:328-336` | `lsth` → `--lsTh <delta>` |
| `curv_smoothness` (2) | `biscuit.m:372-379` → `surf_quant_pipeline.m:15,20` | `MeshProperty … -s -m --surfsmoothing %d --tensorsmoothing %d` (same number twice, printed with `%d`) |
| `lgi_kernel_size` (316) | `biscuit.m:382` → `gyrification.m:8` | `lgi … --kernel %f` (`316.000000`) → `Gyrification -m` |
| `lgi_anisotropy` (0.2) | `biscuit.m:385` | `lgi … --anisotropy %f` → `Gyrification --speed` |
| `lgi_ref_area` (77100) | `biscuit.m:388` | `lgi … --ref %f` → `Gyrification --refHullArea` |
| `sci_kernel_size` (3) | `biscuit.m:391` → `sci.m:7` | `ShapeComplexityIndex … -k %f` |
| `sci_bins` (5) | `biscuit.m:394` → `sci.m:7` | `ShapeComplexityIndex … -s %d` |

Not parameters: every other numeric constant in the tables below (smoothing iterations 3/5/25, percentile 0.1, HSD degree 10,
TRACE seeds, graph-cut weights) is hard-coded.

## 2. Stage order (MaCRUISE route)

`biscuit.m` runs four phases; each starts by copying previously produced outputs back into a scratch dir
(`$TMP/tmp1…tmp4`, `biscuit.m:96-130,312-326,353-367,413-451`), and every step is guarded by an `exist()` test, so a
re-run into the same `$OUT` skips finished work (the guards are a primitive resume; pyBiscuit's stage signatures replace them).

| # | Stage | Source | Detail |
|---|---|---|---|
| 1.1 | Input conversion `.asc` → VTK (native and MNI copy) | `run_postproc_macruise_pipeline.m:26-53` | §4.1 |
| 1.2 | Hemisphere separation → `<h>.target_image_GMimg_innerSurf.vtk`, `tab.txt`, `<h>.mask.txt` | `…:56`, `tessellation/hemi_separation.m` | §4.2 |
| 1.3 | Volume preparation (`mri_convert`, `mri_normalize`) | `…:67-84` | §4.3 |
| 1.4 | FreeSurfer `orig` surfaces from the corrected inner surface | `…:86-108` | §4.4 |
| 1.5 | `brain.finalsurfs`, `wm`, `filled`, `aseg` volumes | `…:110-208` | §4.5 |
| 1.6 | `mris_make_surfaces` white, then pial + thickness; VTK export; optional `MeshDeform --smooth` | `…:210-254` | §4.6 |
| 1.7 | smoothwm, `mris_inflate` (inflated + `sulc`), `mris_sphere` | `…:256-281` | §4.7 |
| 1.8 | mid surface `(white+gray)/2` | `…:283-297` | §4.8 |
| 1.9 | Surface registration: curvature/iH/sulc features, four HSD calls, re-tessellation | `surf_reg_macruise.m` | §5 |
| 1.10 | Vertex-area maps (`MeshProperty -v`) | `…:302-314` | §5.5 |
| 1.11 | Parcellation per protocol (+ `parc.<p>.pdf`) | `…:316-366`, `common/matlab/parc/parc.m` | §6 |
| 2 | TRACE: sulcal curves on `gray`, gyral curves on `white`; `curve.pdf` | `biscuit.m:310-347`, `curve_extract_pipeline.m` | §8 |
| 3 | Quantification: H, S, sulcal depth, LGI, SCI; `quant.pdf` | `biscuit.m:351-409`, `surf_quant_pipeline.m` | §9 |
| 4 | Statistics (`stats.<p>.csv`) and registered-space property maps | `biscuit.m:411-470`, `surf_quant_stat_pipeline.m` | §10 |
| 5 | PDF merge (image wrapper only) | image `run_biscuit` (last lines) | §11 |

A port should treat "Phase" boundaries as copy points only: the only cross-phase file dependencies are listed in §9.0.

## 3. External tool call catalogue (MaCRUISE route)

Every `system()` call, in execution order. `$J` = the phase scratch dir (`$TMP/tmp1` in phase 1; paths are then
`$J/<h>.…`). `sphere_N` is `$ROOT/sphere/sphere_N.vtk` (found with `which`, i.e. `addpath(genpath($ROOT))`,
`biscuit.m:40`). `$T` = `$ROOT/reg/<reg_temp>/<h>` (template prefix). A trailing `(×h)` = once per hemisphere.
Numbers in argv are formatted exactly as shown (`num2str` for `trace_*` and the `MeshDeform -t` translation, `%f`/`%d`
for LGI/SCI/curvature, see §1.2).

| ID | Tool | argv (as built) | Call site | Reads → writes |
|---|---|---|---|---|
| T1 | `mri_convert` | `-odt uchar <mc_rawT1> $J/mri/T1.mgz` | `run_postproc_macruise_pipeline.m:69` | T1 → `mri/T1.mgz` |
| T2 | `mri_convert` | `$J/mri/T1.mgz $J/mri/T1.mgz [ --cw256][ --conform] -nc` (flags only if a dim > 256 / voxel ≠ 1 mm) | `:80` | in place |
| T3 | `mri_normalize` | `-g 1 -mprage $J/mri/T1.mgz $J/mri/T1.mgz` | `:82` | in place |
| T4 | `mri_convert` | `<mc_rawSeg> $J/mri/aseg.tmp.mgz <flag as T2, may be empty> -rt nearest` | `:84` | SEG_refine → `mri/aseg.tmp.mgz` (deleted at `:89`) |
| T5 | `MeshDeform` (×h) | `-i $J/<h>.target_image_GMimg_innerSurf.vtk -o $J/<h>.target_image_GMimg_innerSurf.corrected.vtk -t <c_r> <c_a> <c_s>` | `:97,104` | hemisphere inner surface → translated copy |
| T6 | `mri_convert` | `--conform -nc -odt uchar $J/mri/brain.finalsurfs.mgz $J/mri/brain.finalsurfs.mgz` | `:115` | in place |
| T7 | `mri_convert` | `--conform -nc -odt uchar $J/mri/wm.mgz $J/mri/wm.mgz` | `:127` | in place |
| T8 | `mri_convert` | `--conform -nc -rt nearest -odt uchar $J/mri/filled.mgz $J/mri/filled.mgz` | `:139` | in place |
| T9 | `mri_convert` | `--conform -nc -rt nearest -odt uchar $J/mri/aseg.mgz $J/mri/aseg.mgz` | `:195` | in place |
| T10 | `mri_normalize` | `-mprage -aseg $J/mri/aseg.mgz $J/mri/brain.finalsurfs.mgz $J/mri/brain.finalsurfs.mgz` | `:202` | **dead code in a fresh run** (guard `~exist(brain.finalsurfs.mgz)`, but T6's output already exists) |
| T11 | `mris_make_surfaces` (×h) | `-aseg aseg -noaparc -whiteonly -mgz -T1 brain.finalsurfs $J <h>` | `:213,216` | `surf/<h>.orig`, `mri/*` → `surf/<h>.white` |
| T12 | `mris_make_surfaces` (×h) | `-noaparc -orig_white white -orig_pial white -aseg aseg -nowhite -mgz -T1 brain.finalsurfs $J <h>` | `:220,223` | → `surf/<h>.pial`, `surf/<h>.thickness` |
| T13 | `MeshDeform` (×4, if `surf_smooth>0`) | `--smooth <surf_smooth> -i $J/<h>.white.vtk -o $J/<h>.white.vtk` and `…gray.vtk` | `:250-253` | in place |
| T14 | `MeshDeform` (×h) | `--smooth 3 -i $J/<h>.white.vtk -o $J/<h>.smoothwm.vtk` | `:258,265` (also `:322,343`, `surf_reg_macruise.m:37,91`, guarded) | smoothwm |
| T15 | `mris_inflate` (×h) | `$J/<h>.smoothwm.vtk $J/<h>.inflated.vtk` | `:260,267` | → `<h>.inflated.vtk`, `<h>.sulc` (same dir) |
| T16 | `mris_sphere` (×h) | `$J/<h>.inflated.vtk $J/<h>.sphere.vtk` (after writing `<h>.smoothwm` as FS binary) | `:275,279` | → `<h>.sphere.vtk` |
| T17 | `MeshProperty` (×h) | `-i $J/<h>.white.vtk -o $J/<h> -m  --surfsmoothing 5 --tensorsmoothing 5` | `surf_reg_macruise.m:24,78` | → `<h>.H.txt` (then rewritten as −4·H, §5.1) |
| T18 | `SphericalRemesh` (×h) | `-s $J/<h>.sphere.vtk -r sphere_327680 -p $J/<h>.H.txt --outputProperty $J/<h>.res` | `:30,84` | → `<h>.res.H.txt` |
| T19 | `MeshProperty` (×h) | `-m -i $J/<h>.inflated.vtk -q -o $J/<h>.inflated --normafter --normbefore --prctile 0.1 --datasmoothing 25 --invert` | `:43,97` (also `run_postproc…:330,351`) | → `<h>.inflated.H.txt` (renamed `<h>.iH.txt`) |
| T20 | `SphericalRemesh` (×h, ×2) | `-s $J/<h>.sphere.vtk -r sphere_327680 -p $J/<h>.iH.txt --outputProperty $J/<h>.res` and the same with `<h>.sulc.txt` | `:49-52,103-106` | → `<h>.res.iH.txt`, `<h>.res.sulc.txt` |
| T21 | `HSD` ×4 (×h) | §5.3 | `:55-66,109-120` | → `<h>.coeff.txt`, `<h>.res.sphere.reg.vtk` |
| T22 | `SphericalRemesh` (×h) | `-s $J/<h>.sphere.vtk -c $J/<h>.coeff.txt --deform $J/<h>.sphere.reg.vtk --deg 10` | `:127,150` | → `<h>.sphere.reg.vtk` |
| T23 | `SphericalRemesh` (×h, ×3) | `-i $J/<h>.<s>.vtk -s $J/<h>.sphere.vtk -c $J/<h>.coeff.txt -r sphere_327680 -o $J/<h>.<s>.reg.vtk --deg 10`, `<s>` ∈ {gray, white, mid} | `:131-139,154-162` | → re-tessellated surfaces |
| T24 | `SphericalRemesh` (×h; PDF only) | `-s $J/<h>.res.sphere.reg.vtk -r sphere_327680 -p $J/<h>.res.H.txt --outputProperty $J/<h>.reg --deg 10` (plus a `--deform` fallback for the `.res.sphere.reg.vtk` file) | `:176-190` | → `<h>.reg.H.txt` |
| T25 | `MeshProperty` (×h, ×2) | `-v -i $J/<h>.mid.vtk -o $J/<h>.mid` and `-v -i $J/<h>.mid.reg.vtk -o $J/<h>.mid.reg` | `run_postproc…:304-313` | → `<h>.mid.VertArea.txt`, `<h>.mid.reg.VertArea.txt` |
| T26 | `MeshProperty` (×h, parc features) | `-i $J/<h>.white.vtk -o <tmp>/<id> -m --surfsmoothing 5 --tensorsmoothing 5` | `extract_features.m:4` | → `<id>.H.txt` |
| T27 | `SphericalRemesh` (parc, ×h) | `-r mesh_files/icosphere_5.vtk -s $J/<h>.sphere.vtk -c $J/<h>.coeff.txt --deg 0 --deform <tmp>/<id>.sphere.vtk` | `parc.m:56-59` | → rigidly aligned native sphere |
| T28 | `HSD` (parc, ×h) | `-s sphere_327680 <tmp>/<id>.sphere.vtk -o <tmp>/<id>.sphere.vtk <tmp>/<id>.sphere.vtk -p $ROOT/parc/<p>/<h>.curv.mean.txt <tmp>/<id>.H.txt --fixedSubjects 0 --noguess -d 0 --icosahedron 5` | `parc.m:61` | refines rigid alignment |
| T29 | `SphericalRemesh` (parc, ×h) | `-r mesh_files/icosphere_5.vtk -s <tmp>/<id>.sphere.vtk -p <tmp>/<id>.iH.txt <tmp>/<id>.sulc.txt <tmp>/<id>.H.txt --outputProperty <tmp>/<id>.res` | `parc.m:64-67` | → `<id>.res.{iH,sulc,H}.txt` |
| T30 | `test.sh` (parc, per model) | `<ckpt> 5 <tmp> <fold> <nclass> <tmp>/<id>.sphere.vtk mesh_files/icosphere_5.vtk` (→ `python test.py … --feat 64 --in_ch 3 --min_level 0 --max_level 5 --no-cuda`; inside: `SphericalRemesh -p <class files> -r <osphere> -s <isphere> --outputProperty <tmp>/<fold> --quiet`) | `parc.m:76-85`, `test.sh`, `test.py:35-40` | → `<tmp>/<fold>.<class>.txt` |
| T31 | `MeshProperty` (×h, quant) | `-i <mid.vtk> -o <tmp3>/<h>.mid -s -m --surfsmoothing <curv_smoothness> --tensorsmoothing <curv_smoothness>` | `surf_quant_pipeline.m:15,20` | → `<h>.mid.S.txt`, `<h>.mid.H.txt` |
| T32 | `lgi` (×h) | `-i $OUT/surf/<h>.gray.vtk --white $OUT/surf/<h>.white.vtk --kernel <lgi_kernel_size %f> --anisotropy <lgi_anisotropy %f> --ref <lgi_ref_area %f>` (cwd `$TMP`) | `gyrification.m:8` | → `<h>.gray.lgi.map.<k>.txt`, renamed `<h>.gray.LGI.txt` (`:13-15`) |
| T33 | `OuterHull` (inside `lgi`, ×h) | `<gray.vtk> $TMP/tmp-lgi-<h>.gray/<h>.gray.hull.vtk` | `lgi` script, line 68 | hull |
| T34 | `klaplace` (inside `lgi`) | `-thread <OMP_NUM_THREADS> -dims 128 <gray.vtk> <hull.vtk> -surfaceCorrespondence <tmpdir>/<h>.gray.corr`, then `-conv <tmpdir>/<h>.gray.corr_warpedMesh.vtp <tmpdir>/<h>.gray.hull.corr.vtk` | `lgi:83-84` | vertex-wise outer surface |
| T35 | `Gyrification` (inside `lgi`) | `-i <gray.vtk> -o <tmpdir>/<h>.gray --outer <hull.corr.vtk> -s <tmpdir>/<h>.gray.scurve.bary -g <tmpdir>/<h>.white.gcurve.bary -m <kernel> --speed <aniso> --refHullArea <ref> --nThreads <OMP_NUM_THREADS>` (the `.bary` files are copied from `$OUT/curve/` by `lgi:111-112`) | `lgi:134` | → `….lgi.map.<k>.txt` |
| T36 | `ShapeComplexityIndex` (×h) | `-i <mid.vtk> -o <tmp3>/<h>.mid.SCI.txt -k <sci_kernel_size %f> -s <sci_bins %d>` | `sci.m:7` | → `<h>.mid.SCI.txt` |
| T37 | `CurveExtraction` (×h, sulcus on gray, gyrus on white) | `--sulcus\|--gyrus -i <surf>/<h>.gray.vtk\|<h>.white.vtk -o $J2/<h>.gray\|<h>.white --nThreads <omp_num_threads> --epRad <trace_r> --nhDist <trace_r> --lsTh <trace_delta> --bary --noVtk` | `curve_extract_pipeline.m:39,47` | → `<o>.scurve`/`.gcurve`, `<o>.scurve.bary`/`.gcurve.bary` |
| T38 | `MeshProperty` (curve PDF colouring, ×h, ×2 surfaces) | `-i <gray\|white>.vtk -o $J2/<h>.mid -m --surfsmoothing 5 --tensorsmoothing 5` (output `<h>.mid.H.txt` deleted at `curve_extract_pipeline.m:61`) | `:32-33` | PDF only |
| T39 | `MeshContour` (hemisphere split) | `<tmp.vtk> 1.000000 0.000000 0.000000 <d1>,<d2>,…,<d101>,` (101 plane offsets `%f`, trailing comma) | `hemi_separation.m:127` | stdout → face ids + chosen plane index |
| T40 | `SphericalRemesh` (stats, ×h, ×6) | `-s $OUT/reg/<h>.sphere.reg.vtk -r sphere_327680 -p <tmp4>/<h>.mid.<M>.txt --outputProperty $J4/<h>.mid.reg`, `<M>` ∈ {H, S, SD, CT, LGI, SCI} | `surf_quant_stat_pipeline.m:461-463` | → `<h>.mid.reg.<M>.txt` |
| T41 | pdflatex / gs (PDF) | §11 | `run_GeneratePDF_*.m` | PDFs |

`MeshProperty`, `MeshDeform` and `MeshContour` are closed binaries; the flag set printed by the image binary
is `-i -o -m -s -v -q --writevtk --tensorsmoothing --surfsmoothing --datasmoothing --prctile --normbefore --normafter --invert`
(MeshProperty; `-m` mean curvature `.H.txt`, `-s` shape index `.S.txt`, `-v` vertex area `.VertArea.txt`; other letters give `.K`/`.C`
maps and are not used by BISCUIT), and `-i -o --matrix -t/--translate --translateCentered --sphere --scale --noise --smooth -c/--center --invertFace`
(MeshDeform), from `strings` on the extracted binaries 2026-10-21. `MeshContour` has no usage text beyond ` vtk x y z d`.

## 4. Phase 1a — from MaCRUISE outputs to FreeSurfer-style surfaces

Entry: `run_postproc_macruise_pipeline(tmp1, rawT1, rawSeg, surf, surf_mni, $ROOT, reg_temp, protocol, surf_smooth, …)`
(`biscuit.m:221`). The whole function is wrapped in "output exists?" guards (`:7-17`, `:60-66`).

### 4.1 Inputs and conversion (`:26-53`, `read_asc.m`, `write_vtk.m`)

* Files: `mc_surf` (native inner surface, scanner RAS, FreeSurfer ASCII `.asc`), `mc_surf_mni` (the same mesh moved by a rigid
  transform), `mc_rawT1` (NIfTI), `mc_rawSeg` (`SEG_refine.nii.gz`, BrainCOLOR labels). `.gz` inputs are gunzipped into `$J`.
* `read_asc`: drops lines starting `#`; line 1 `nV nF`; then `nV` lines `x y z 0` and `nF` lines `i j k 0` with **0-based** indices.
  Faces are not re-based; `write_vtk` writes them as given. No frame conversion happens (the legacy `.asc` is already scanner RAS, brief 01).
* `write_vtk`: legacy ASCII POLYDATA, `POINTS n float`, coordinates `%f %f %f` (**6 decimals**), `POLYGONS nF 4nF`, `3 i j k` (0-based).
  Every intermediate VTK is therefore rounded to 1e-6 mm; `read_vtk` re-reads with `sscanf`. Per-vertex `.txt` files are `%f\n` (or `%d\n`).
* Results: `$J/target_image_GMimg_innerSurf.vtk` (native) and `…_innerSurf_mni.vtk` (MNI copy).

### 4.2 Hemisphere separation (`hemi_separation.m`, `connectedComp.m`, `circle_patch.m`, `surfNormalCorrection.m`)

`hemi_separation(native.vtk, lh.vtk, rh.vtk, tab.txt, lh.mask.txt, rh.mask.txt, mni.vtk)` (`:1-74`). The native surface supplies the geometry;
**the MNI surface is used only to place the cutting plane** (`getCuttingPlane(refsurfname)`, `:9`), which assumes the MNI mesh has the
same vertex order and faces. Steps:

1. **Orientation fix** (`surfNormalCorrection`, `:1-31`): take the vertex with minimum x; mean unit normal of its faces; `n = [sign(n_x) 0 0]`;
   project all vertices on `n` relative to that vertex; if more vertices have positive projection than not, reverse every face (`f(:,3:-1:1)`).
   Applied to the native mesh (`:3`) and, separately, to the MNI mesh (`:78`).
2. **Cutting plane** (`getCuttingPlane`, `:76-140`) on the MNI mesh: Laplacian smooth (vertex := mean of 1-ring neighbours, `W*v ./ deg`,
   20 iterations, no self weight; edges deduplicated with `unique(…,'rows','first')`) → write `<root>/_tmp_<name>.vtk` →
   101 candidate planes `x = const`: for `r = −0.5:0.01:0.5`, `delta_i = −mean(x) + r`; `j` = vertex with minimum `|x + delta_i|`;
   `M = x + delta_i` (sign flipped if `x_j + delta_i > 0`), entries `< 0` set to `inf`, `k = argmin M`; the plane is set to the midpoint:
   `delta_i = −(x_j + x_k)/2`. So all candidates lie within 0.5 mm of the smoothed mesh's mean x. **`MeshContour` (T39)** receives the
   smoothed mesh, the normal `1 0 0` and the 101 offsets, and prints the 0-based ids of the faces cut by the plane it selects followed by the
   0-based index of the selected offset (`:127-133`: `vid = str2num(…)`, `vid(end)` = offset index, the rest are face ids). The selection rule
   is not visible in the source (closed binary; hypothesis: the plane with the shortest cut; unverified, needs a black-box probe). Boundary vertices
   `bdry` := all vertices of the cut faces; any non-flagged vertex whose neighbours are *all* flagged is added (`:136`).
   (A commented-out PCA block at `:86-105` shows the author also considered estimating the orientation from the mesh.)
3. **Index** (`computeIndex`, `:252-269`): `invalid_f` = faces whose three vertices all lie in `bdry`; for every vertex the list of its remaining faces
   (`fid{v,1}`) and the number of removed incident faces (`fid{v,2}`).
4. **Boundary loops** (`getBoundary`, `:142-193`): boundary vertices = vertices with ≥ 1 kept and ≥ 1 removed face. Walk loop 1 from the first
   boundary vertex, at each step taking, among the kept faces of the current vertex *in stored order*, the next vertex in face winding that is an unvisited
   boundary vertex; stop when the start is reached again; walk loop 2 from the last unvisited boundary vertex; reverse both. Failure to close prints
   "the cutting plane is wrong!" and `exit(1)`. **Exactly two loops are assumed.**
5. **Island cleanup** (`cleanupBoundary`, `:194-204`): connected components of the mesh with boundary vertices disconnected, numbered by
   `dmperm(W'+speye)` block order; every vertex in a block with number > 2 (excluding the boundary vertices) joins `bdry` and the index is rebuilt.
   **This silently assumes that blocks 1 and 2 are the two hemispheres**, which depends on MATLAB's block ordering, not on component size. The Python
   rule must be "keep the two largest components"; both give the same result on a clean mesh (to be verified on the legacy sample).
6. **Patch creation** (`circle_patch.m`, `circlePatch`, `:245-251`): per loop with `m` vertices a disk mesh is built by `DelaunayTri` on a 51×51
   concentric-disk-mapped grid (radius `1 − 1/50`) plus `m` boundary points on the unit circle (`bphi = (0:m−1)·2π/(m−1)`; the first and last
   points coincide to rounding, so the hull `k = DT.convexHull; k = k(1:end−1)` has `m` entries). The hull indices `k(i)` are paired with loop vertices
   `tr(i)` in hull order — **the cyclic alignment between the hull start and the loop start is whatever Qhull returns**.
7. **Harmonic fill** (`poisson`, `:270-287`): unweighted graph Laplacian `A = D − W` of the patch triangulation; rows of boundary vertices replaced by identity;
   right-hand side zero except `b(k,:) = v(tr,:)`; solve `x = A\b` (sparse direct) for the three coordinates. (`poisson_` at `:289` is unused.)
8. **Stitch** (`closeSurface`, `:324-347`): vertex list `[v; x]`; patch boundary vertices are replaced by the loop vertices (`pf0(pf==k(i)) = tr(i)`);
   vertices left with no kept face (the removed bridge band) and the duplicated patch-boundary vertices are deleted and indices compacted;
   `vorgnew` = 1-based original vertex id for original vertices, `0` for patch vertices. New faces are appended after the original faces.
9. **Smoothing around the seam** (`:44-52`): region = loop vertices ∪ patch vertices, grown by one ring (`find(W(list,:)==1)`); every vertex outside
   the region is frozen; 5 Jacobi iterations of neighbour-mean smoothing (`smooth`, `:236-244`: `vtmp = (W*vtmp) .* (1./deg)`, no self weight).
10. **Components → hemispheres** (`connectedComp.m`): blocks of `dmperm(W'+speye)`; `m1`, `m2` = mean position of blocks 1 and 2;
    if `m1(1) > m2(1)` the labels are swapped, so **the hemisphere with smaller x (native frame) is `lh`**; blocks > 2 are discarded
    (`vorg(:,1) = 0`). Vertices keep their relative order within each hemisphere (original vertices first, patch vertices last);
    faces are those with all three vertices in the hemisphere, renumbered; `lh.target_image_GMimg_innerSurf.vtk`/`rh.…` are written.
11. **Outputs.** `<h>.mask.txt`: `1` for original cortex vertices, `0` for patch (medial-wall) vertices (`:62-69`, `writeInteger(…, ~mask)`).
    `tab.txt` (copied to `$OUT/surf/hemi_map.txt`): three integers per original vertex that ended in a hemisphere, **all 0-based**:
    `<index in the whole-brain native surface> <index in the hemisphere mesh> <0 lh | 1 rh>`, in whole-brain order (`:72-73`).
    If `$OUT/surf/hemi_map.txt` exists on a re-run it is copied back as `tab.txt` before phase 1, which, with `lh.mask.txt` etc. present, skips the
    separation (`run_postproc…:7-11`, `biscuit.m:194-196`).

Consequences for a port: the hemisphere meshes include the filled medial wall, so vertex counts exceed the cortex vertex counts by the patch size;
the patch triangulation (`DelaunayTri`/Qhull) cannot be reproduced bit-for-bit. Vertex-wise parity is meaningful only where `mask == 1`.
`MeshContour` and the left/right test are the only two places where the native frame matters (x increases toward the right).

### 4.3 Volume preparation (`:67-89`)

T1 (T1.mgz via `mri_convert -odt uchar`); `MRIread(…,false)` reads the header only (FreeSurfer-licensed helper; not read). `flag = ''`; if any
dimension > 256 add ` --cw256`; if any voxel size ≠ 1 add ` --conform`; if `flag` is non-empty append ` -nc` and convert in place. Then `mri_normalize -g 1 -mprage`.
The SEG is converted with the **same** `flag` and `-rt nearest` (T4). Quirk: `flag` is defined only inside the `~exist(T1.mgz)` branch, so a re-run
with an existing `T1.mgz` hits an undefined variable at T4.
Header read afterwards: `T1.c_r, c_a, c_s` (centre RAS of the conformed volume) and the volumes `T1.vol`, `seg.vol`.

### 4.4 `orig` surfaces (`:91-108`)

`xform = [−1 0 0 c_r; 0 0 −1 c_a; 0 1 0 c_s]` (the conformed LIA orientation with centre `c_ras`, `:93`). Per hemisphere: T5 translates the hemisphere inner
surface by `(c_r, c_a, c_s)` formatted with `num2str` (≈ 4–5 significant digits: the translation is quantised to ~1e-4 mm); the result is written as a
FreeSurfer binary triangle surface (`write_surf`: magic 16777214, a creation-line text header, big-endian int32 counts, float32 coordinates, int32 0-based faces)
and copied by `write_surf3` with an appended volume-info tag block (`valid = 1`, `filename = Seg.nii.gz`, `volume = 256 256 256`, `voxelsize = 1 1 1`, `xras yras zras`
from `xform` rows, `cras` from its last column). **The sign convention of `MeshDeform -t` is not visible** (adds vs subtracts); FreeSurfer expects tkRAS
(= scanner RAS − `c_ras`) vertex coordinates, so `-t` is expected to move the surface by `−c_ras`. Whatever it does, **every later FreeSurfer-derived surface,
and therefore `surf/<h>.white.vtk`, `gray`, `mid`, `inflated`, `sphere` and all measures, are in this FreeSurfer surface frame, not in the native scanner RAS**:
`surf_convert_fs_vtk` (`:369-372`) writes the coordinates read from the FreeSurfer files unchanged. (To be confirmed by comparing an image output with the input `.asc`.)

### 4.5 Auxiliary volumes (`:110-208`)

All are derived from `T1.mgz` (normalised T1) and `seg` (SEG_refine) at the conformed grid and then `mri_convert --conform -nc [-rt nearest] -odt uchar`.
Label numbers are SEG_refine/BrainCOLOR ids as used in the code.

* `brain.finalsurfs.mgz` (`:111-118`): `T1` with voxels where `seg == 0` set to 0 (T6).
* `wm.mgz` (`:121-130`): start from `T1`; set to 0 wherever `seg ∉ {44, 45, 35}`; set to **250** where `seg ∈ {36,37,51,52,55,56,57,58,59,60,61,62}` (T7).
* `filled.mgz` (`:133-142`): zeros; **255** where `seg ∈ {37,45,52,56,58,60,62}`; **127** where `seg ∈ {36,44,51,55,57,59,61}` (T8). (255 = left, 127 = right in FreeSurfer's `filled` convention.)
* `aseg.mgz` (`:145-198`): copy of `seg` with these replacements, each tested on the **original** `seg` (so no chaining):
  `45→2, 44→41, 58→12, 56→13, 57→51, 55→52, 50→5, 49→44, 52→4, 51→43, 60→10, 59→49, 62→28, 61→60, 48→17, 47→53, 36→50, 37→11, 31→54, 32→18,
  39→8, 38→47, 41→7, 40→46, 23→58, 30→26, 35→16, 11→15, 71→47, 72→47, 73→47`; every label ≥ 100 with odd value → **3** (left cortex), even → **42** (right cortex) (T9).
* The "refine brain" block (`:200-208`, T10) never runs in a fresh run (see T10). `brain.finalsurfs` is therefore the masked, normalised T1 only.

### 4.6 White, pial, thickness (`:210-254`)

T11 refines the white surface from `surf/<h>.orig` under aseg guidance (`-whiteonly`); T12 then produces `surf/<h>.pial` and `surf/<h>.thickness` from the intensity
image **with `-orig_pial white`**, i.e. the pial is *deformed from the white surface by FreeSurfer's intensity model*; MaCRUISE's own outer surface is never used.
`<h>.thickness.txt` = `read_curv(surf/<h>.thickness)` printed `%f\n` (`:226-235`). `<h>.white.vtk` and `<h>.gray.vtk` = `read_surf` → `write_vtk` of `surf/<h>.white` and `surf/<h>.pial`
(`:236-247`). Optional T13 in-place smoothing. Both meshes keep the vertex order of `surf/<h>.orig` (hence of the hemisphere mesh from §4.2): white–pial correspondence is by index.
The outputs `lh.gray.vtk` etc. use the word *gray* for the pial surface everywhere downstream.

### 4.7 Inflation and sphere (`:256-281`)

T14 smoothwm (`MeshDeform --smooth 3`, a 3-iteration smoothing of the white surface; the commented alternative was `mris_smooth -n 3 -nw`), T15 `mris_inflate`
writes `<h>.inflated.vtk` and the sulcal-depth map `<h>.sulc` (FreeSurfer binary curv, read later with `read_curv`; it is *the* `sulc` feature), T16 `mris_sphere`
after `<h>.smoothwm.vtk` has been written as FreeSurfer binary `<h>.smoothwm` (needed by FreeSurfer as the metric surface). The guards at `:256-281` are written so
that a missing `rh.inflated.vtk` or `rh.sphere.vtk` re-runs only the right hemisphere (the `lh` guard of `:270` tests the same file twice, harmless).
The FreeSurfer tools are called with `.vtk` file names; the FC image's builds read and write VTK.

### 4.8 Mid surface (`:283-297`)

`v_mid = (v_white + v_gray)/2`, faces of white; `<h>.mid.vtk`. (An alternative `mris_expand -thickness` is commented out.)

## 5. Surface registration (`surf_reg_macruise.m`, `$J` = `tmp1`)

Called as `surf_reg_macruise(J, …, $ROOT, reg_temp, J/lh, J/rh, 'white')` (`run_postproc…:300`). The two hemispheres are processed
identically (the call lists are mirror images, with one asymmetry flagged in §5.3). Registration runs on the **white** surface features
and on `sphere_327680` (163,842 vertices, the icosphere with the same vertex order as the templates).

### 5.1 Features

| Feature | Computation | File |
|---|---|---|
| `H` (white-surface mean curvature) | T17 → `<h>.H.txt`; then the file is **rewritten** as `−4·H` (`%f\n`) (`:26-29`) | `<h>.H.txt` |
| `iH` (inflated mean curvature) | T19 on `<h>.inflated.vtk` (normalised: `--normbefore --normafter --prctile 0.1`, data-smoothed 25×, sign inverted, quadratic fit `-q`) → renamed `<h>.iH.txt` (`:43-44`) | `<h>.iH.txt` |
| `sulc` | `read_curv(<h>.sulc)` (FreeSurfer curv written by `mris_inflate`) printed `%f\n` (`:45-47`) | `<h>.sulc.txt` |

Each is resampled onto `sphere_327680` by `SphericalRemesh -s <h>.sphere.vtk -r sphere_327680 -p <file> --outputProperty <h>.res`
(T18/T20): output name = property file name minus `.txt` after its last `.` → `<h>.res.H.txt`, `<h>.res.iH.txt`, `<h>.res.sulc.txt`.
The `H` branch runs when `<h>.sphere.reg.vtk` **or** `pdf/reg.<temp>.pdf` is missing; the others only when the sphere is missing (`:23-34`).

### 5.2 Templates and spheres

`$T = $ROOT/reg/<reg_temp>/<h>`: `$T.{iH,sulc,curv}.{mean,var}.txt` (163,842 lines each, the ico7 vertex order). Template `curv` pairs with the
feature `H` (the file named `curv` holds the mean of the `−4H` feature, §5.1). Spheres: `sphere_5120, 20480, 81920, 327680` have 2,562 / 10,242 / 40,962 /
163,842 vertices (icosahedral levels 4–7, hierarchical vertex order).

### 5.3 The four HSD calls (T21)

Written for `lh` (`surf_reg_macruise.m:55-66`); `C` = `$J/<h>.coeff.txt`, `S5120 … S327680` = `sphere_N`, `T` = template prefix. `rh` is
identical except where noted. `-s` lists template first, subject second; `--fixedSubjects 0` fixes the template (pair-wise mode).

| # | argv (after `HSD`) | Subject sphere | `--icomesh` | Opt. level |
|---|---|---|---|---|
| 1 | `-s S327680 S20480 --outputcoeff C C -p T.iH.mean.txt J/lh.res.iH.txt --fixedSubjects 0 -d 10 --icomesh S5120` | 20480 (10,242 v) | 5120 (2,562) | 4 |
| 2 | `-c C C -s S327680 S81920 --outputcoeff C C -p T.sulc.mean.txt J/lh.res.sulc.txt --fixedSubjects 0 -d 10 --icomesh S20480` | 81920 (40,962 v) | 20480 (10,242) | 5 |
| 3 | `-c C C -s S327680 S327680 --outputcoeff C C -p T.sulc.mean.txt J/lh.res.sulc.txt --fixedSubjects 0 -d 10 --icomesh S81920 --tmpVar T.sulc.var.txt` | 327680 | 81920 (40,962) | 6 |
| 4 | `-c C C -s S327680 S327680 -o J/lh.res.sphere.reg.vtk J/lh.res.sphere.reg.vtk --outputcoeff C C -p T.curv.mean.txt J/lh.res.H.txt --fixedSubjects 0 -d 10 --icomesh S327680 --tmpVar T.curv.var.txt` | 327680 | 327680 (163,842) | 7 |

* **One property per call.** `nProperties/nSubj = 1`; `--weight` is never given (weight 1). Properties go in the order iH → sulc → sulc → H(curv);
  `--tmpVar` (variance prior) only for the two finest stages, with the **single-property** `.var.txt` of that feature.
* **Chained by coefficient files.** Call 1 starts without `-c`; calls 2–4 read the previous result from `C`, and write `C` again. HSD reads `-c` only for non-fixed
  subjects and writes `--outputcoeff` and `-o` only for non-fixed subjects (`HSD.cpp:170-175,761`), so using one file name for both subjects is safe.
  Degree is capped at 10 in every call; the coefficient file carries pole (`%f`) and `%lf` coefficients.
* **Defaults in force:** `--idprior 625`, `--maxIter 20`, `--gres 3`, `--nThreads 0` (= `OMP_NUM_THREADS`), initial guess **on** (no `--noguess`), no `--resample`,
  `--icosahedron` default 7 overridden by `--icomesh`.
* **Mesh resolution vs property length.** The subject mesh has fewer vertices than the 163,842-line property files in calls 1–2 (and the template mesh equals
  the property length); HSD reads the **first `nVertex` lines** of each property file (`HSD.cpp:893-960`, `initProperties(…, nLines, …)`).
* **`--icomesh` selects the optimisation icosahedron** by vertex count (`HSD.cpp:299-309`): 2,562 → level 4, 10,242 → 5, 40,962 → 6, 163,842 → 7. Levels < 6 use the
  gradient-descent branch, levels ≥ 6 the damped Newton branch (hsd.md: `m_fine_res = 6`), so **both branches run in every registration**.
* **rh asymmetry.** For `rh`, call 2 is `-s S81920 S327680` (`:112`: template sphere = 81920, subject sphere = 327680), the reverse of `lh` (`:58`), and of
  `surf_reg_fs2.m`. Whether this is intended is unknown (flag, not fixed).
* Output of call 4: `<h>.res.sphere.reg.vtk`, the registered `sphere_327680` (used only for the PDF/energy), and the final `C`.

### 5.4 Re-tessellation

* T22: `SphericalRemesh -s <h>.sphere.vtk -c C --deform <h>.sphere.reg.vtk --deg 10` — the subject's native sphere deformed by the coefficients (`reg/<h>.sphere.reg.vtk`).
* T23: for `s ∈ {gray, white, mid}`: `SphericalRemesh -i <h>.<s>.vtk -s <h>.sphere.vtk -c C -r sphere_327680 -o <h>.<s>.reg.vtk --deg 10`: each surface resampled onto
  the 163,842-vertex template tessellation (`reg/<h>.<s>.reg.vtk`). The `mid.reg` surface feeds `stat/<h>.mid.reg.area.txt` (T25).
* Registered-space property maps (§10) use `SphericalRemesh -s <h>.sphere.reg.vtk -r sphere_327680 -p … --outputProperty …reg` (T40), **without** `--deg`.

### 5.5 Energies and areas

* After registration (PDF only, `:172-209`): `T24` produces `<h>.reg.H.txt` by remeshing `res.H` over the registered uniform sphere (`--deg 10`).
  Energies printed in the PDF: `E0 = ‖z(T.curv.mean) − z(<h>.res.H)‖² / 163842` and `E1` with `<h>.reg.H.txt`, `z` = MATLAB `zscore` (mean 0, std with N−1). Format `%.3f`.
* T25: vertex-area maps of the mid surface and of the registered mid surface (`MeshProperty -v`).

## 6. Parcellation (`parc.m`, `extract_features.m`, `gcut.m`, `gcut_.m`, `roi_correction.m`)

`parc(J, 5, <h>, $ROOT, <protocol>, 0)` for each hemisphere and protocol (`run_postproc…:333,354`); result written `%d\n` to `<h>.parc.<protocol>.txt`
(`:334-336`) and loaded back. The last argument (**degree**) is 0 for MaCRUISE and FreeSurfer, **10 for CIVET** (`run_postproc_civet…:120,137`).
Inputs: `<h>.white.vtk` (vertices + faces), `<h>.sphere.vtk`, `<h>.coeff.txt`, `<h>.inflated.H` (FreeSurfer curv, from `lh.inflated.H.txt` via `write_curv` with the
face count, `run_postproc…:325-332`, re-created by T19 if missing), `<h>.sulc`, and from `$ROOT/parc/<p>/`: `label.csv`, `<h>*.pth`, `<h>.curv.mean.txt`.

### 6.1 Labels and classes (`parc.m:2-31`)

`label.csv` has a header line (skipped); column 1 = name, column 2 = integer id, optional column 3 = lobe code (1 frontal, 2 temporal, 3 parietal, 4 occipital, 5 insula;
only `braincolor` has it). Rows whose name starts with `Left` (case-insensitive, first 4 characters) form `label_lh`, rows starting `Right` form `label_rh`;
`label = [0, ids of that hemisphere in file order]`. **Class index `c` (1-based) ↔ `label(c)`; class 1 is "unknown" = 0**; `nclass = length(label)`
(braincolor 50, dkt31 32 — equal to the output channel counts of the shipped networks). The final vector is `label(pred)'`.

### 6.2 Features (`extract_features.m`, 38 lines)

1. `iH = read_curv(<h>.inflated.H)`; `sulc = read_curv(<h>.sulc)`; `H = −(T26 output)` where T26 = `MeshProperty -i <h>.white.vtk -o <tmp> -m --surfsmoothing 5 --tensorsmoothing 5`.
   (`H` here is the plain mean curvature negated, **not** the −4·H of §5.1.)
2. For each of `iH`, `sulc`, `H` (call it `x`), in this order of operations (`:8-28`):
   `x = zscore(x)`; soft clamp; `x = x / std(x)`; soft clamp again, where
   soft clamp: `x < −3 → −3 − (1 − exp(3 + x))`, `x > 3 → 3 + (1 − exp(3 − x))` (maps (−∞,−3) to (−4,−3) and (3,∞) to (3,4), continuous at ±3).
   `zscore` and `std` use the N−1 denominator.
3. Written as `%f\n` (6 decimals, i.e. rounded) to `<tmp>.H.txt`, `<tmp>.iH.txt`, `<tmp>.sulc.txt` in a fresh `tempname` directory.
   Read from `curv` files, so `iH` and `sulc` carry the float32 rounding of FreeSurfer's binary curv format.

### 6.3 Alignment and resampling (`parc.m:47-67`)

1. T27: rotate/deform the native sphere with the registration coefficients **truncated to degree `degree`** (0: the rigid part only; CIVET: 10):
   `SphericalRemesh -r icosphere_5.vtk -s <h>.sphere.vtk -c <h>.coeff.txt --deg <degree> --deform <tmp>.sphere.vtk`.
2. T28 (refines the rigid alignment against the protocol template, `d = 0`, icosahedron level 5, no initial guess):
   `HSD -s sphere_327680 <tmp>.sphere.vtk -o <tmp>.sphere.vtk <tmp>.sphere.vtk -p $ROOT/parc/<p>/<h>.curv.mean.txt <tmp>.H.txt --fixedSubjects 0 --noguess -d 0 --icosahedron 5`.
   The subject sphere has the native vertex count; the template file has 163,842 lines; `-o` overwrites the input.
3. T29: features sampled onto the level-5 icosphere using the aligned sphere:
   `SphericalRemesh -r icosphere_5.vtk -s <tmp>.sphere.vtk -p <tmp>.iH.txt <tmp>.sulc.txt <tmp>.H.txt --outputProperty <tmp>.res` → `<tmp>.res.{iH,sulc,H}.txt`.
   `icosphere_5.vtk` is `$ROOT/SurfPostproc/common/parc/mesh_files/icosphere_5.vtk` (10,242 vertices).

### 6.4 Network (`parc.m:70-87`, `test.sh`, `test.py`, `loader_individual.py`)

For each checkpoint `<h>*.pth` of the protocol (MATLAB `dir` order; `lh.pth, lh_2.pth …`): `test.sh <ckpt> 5 <tmp> <fold> <nclass> <aligned native sphere> <icosphere_5.vtk>`
runs `python test.py --ckpt … --export_file <fold> --test-batch-size 1 --data_folder <tmp> --max_level 5 --min_level 0 --feat 64 --in_ch 3 --out_ch <nclass> --isphere
<icosphere_5.vtk> --osphere <aligned native sphere> --no-cuda`. The loader globs `<tmp>/*.res.*.txt`, picks the files named `H`, `iH`, `sulc` **in that channel
order**, takes the first 10,242 values of each. The network output (raw last-layer scores, 50 or 32 channels, no softmax) is written `%f` per class to
`<tmp>/<fold>.<class>.txt` and remeshed to the native sphere: `SphericalRemesh -p <class files> -r <osphere> -s <isphere> --outputProperty <tmp>/<fold> --quiet`
(the remeshed files overwrite the class files). `parc.m` loads them into a `nclass × nVertex` matrix per fold (`:89-95`).

### 6.5 Parallelism

`thread = str2num(getenv('OMP_NUM_THREADS'))`; the folds are launched as background shell jobs, `wait` after every `thread` jobs and after the last
(`:72-85`). No effect on results.

### 6.6 Per-model refinement (`:89-108`) and label fusion (`:110-123`)

For each fold `m` (scores `P_m`, shape `nclass × nV`):

1. `pred = gcut(v, f, P_m)` with `v, f` = **white** surface of the hemisphere (§6.7), then `pred = roi_correction(pred', v, f, P_m)`; store `pred_cand(m,:) = pred`.
2. `Q = exp(P_m − max_c P_m)` normalised over classes (softmax per vertex); `pred_prob(m, j) = Q(pred(j), j)` (the probability of the chosen label under that model).
3. If there is more than one model (`fold > 1` after the loop, i.e. `nfold > 1`; dkt31 has 5, braincolor 1): for each class `i`: `prob(i,:) = Σ_m pred_prob(m,:) · [pred_cand(m,:) == i]`;
   `prob = prob ./ Σ_i prob`; `prob = log(prob + eps)` (`eps` = 2.2204e−16); `pred = gcut(v,f,prob)`; `pred = roi_correction(pred', v, f, prob)`.
   With a single model the result of step 1 is the final prediction.
4. `pred = label(pred)'`.

Note the divide: a vertex with zero total vote in step 3 yields `0/0 = NaN` (each model always votes, so this cannot happen when `nfold ≥ 1`).

### 6.7 Graph cut (`gcut.m`, `gcut_.m`; Bagon's GCMex)

`gcut(v, f, prob)` (`gcut.m:1-16`): `F = f+1`; `rc` = sorted unique directed edges (both directions of every face edge, `unique(…,'rows','first')`);
`gcut_(v, rc, prob)` (`gcut_.m:1-32`), with `L = size(prob,1)`:

* Smoothness cost matrix `V = ones(L) − eye(L)` (Potts).
* Data cost `D(l, j) = −log softmax_l(prob(:, j))` (`exp(prob − max)`, normalised, `−log`).
* Neighbour weights `w_ij = ‖v_i − v_j‖` (Euclidean edge length on the white surface, mm). The sparse matrix holds each directed edge; `GraphCutConstrSparse.cpp:92-97`
  adds a neighbour pair only for the **upper triangle**, so each undirected edge enters once with weight `w_ij`.
* Initial labelling: `argmin_l D(l, j)` (`[~,output]=min(prob)`, first minimum on ties, 1-based → `output−1` for the library).
* Energy: `Σ_j D(label_j, j) + Σ_{(i,j)} w_ij · V(label_i, label_j)`; minimised by **α-expansion** (`GraphCut('expand', gch)` → `GCoptimization::expansion()`):
  cycles `for each label α in order 0 … L−1` (`m_random_label_order = 0`) until a full cycle does not decrease the energy (`start_expansion`,
  `GCoptimization.cpp:574-594`). Capacities and energies are **float32** (`Graph::captype = float`, `graph.h:73-76`; `single(Dc(:))`, `GraphCut.m:390`).
  Max-flow is Boykov–Kolmogorov. The result is a local minimum; ties between equal-cost cuts follow the BK implementation.
* Output labels `L'+1`.

### 6.8 Connected-component cleanup (`roi_correction.m`)

`roi_correction(label, v, f, prob)` (`:1-100`) works on the vertex adjacency of `f` (`A`), repeated until the "number of blocks" stops changing:

```
parc = label; bsize_pre = 0; bsize = #unique(parc)
while bsize_pre ~= bsize
    parc_refine = parc; bsize_pre = bsize
    bins   = components of the subgraph of edges whose endpoints have equal label         (ccomp = dmperm blocks, :96-99)
    binsize= tabulate(bins) counts;  id(j) = label of component j
    [sorted, comp] = sortrows([id binsize], [1, -2]);  keep = first row per label (unique 'first')   % largest component of each label
    small = components that are not `keep`;  if none: break
    parc(ismember(bins, small)) = -1                                   % mark every non-largest fragment
    bins   = components of the equal-label subgraph again (the -1 fragments now merge among themselves)
    bsize  = #unique(bins)
    for each component j whose label is -1:
        cand = unique labels of the neighbours of its vertices, minus {-1}
        if numel(cand) > 1: cand = cand \ {1}                          % drop class 1 ("unknown") when other candidates exist
        if numel(cand) == 1: parc_refine(bins==j) = cand
        else: new = gcut_(v(bins==j,:), local edges, prob(cand, bins==j));  parc_refine(bins==j) = cand(new)   % local graph cut among candidates
    parc = parc_refine
end
```

Details a port must keep: neighbours are the 1-ring of the face list; "largest component per label" ties are resolved by the **lowest component number**
(`sortrows` is stable, `unique(...,'first')`), and the component numbering is `dmperm`'s block order — equal-sized components of one label are therefore a
tie that the Python rule must define (propose: lowest minimum vertex index) and flag when it fires; if `cand` has **zero** entries (a fragment surrounded only by `-1`
or by its own label) the local `gcut_` call receives an empty label set and MATLAB errors (the source has no guard).
The `prob` argument is the score matrix of the fold (or `log(prob+eps)` in the fusion call); `gcut_` applies its own softmax again.

### 6.9 Outputs and masks

`<h>.parc.<p>.txt`: one integer label id per vertex of the hemisphere mesh (0 = unknown). The cortical mask is **not** derived from the parcellation on the MaCRUISE route:
`<h>.mask.txt` comes from the hemisphere separation (§4.2). (On the FreeSurfer route without `?h.cortex.label`, and for CIVET, the mask is `parc > 0`,
`run_postproc_fs_pipeline.m:113-119`; with the label file it is the label's vertex set.) Medial-wall patch vertices are classified like any vertex; the statistics
(§10) do not apply the mask.

The parcellation PDF `parc.<p>.pdf` uses the labels with `lut.csv` colours (§11).

## 7. MATLAB behaviours that the Python equivalents must match

| Construct | Where | Semantics to reproduce | Python hazard |
|---|---|---|---|
| `%f` text output | all writers | 6 decimals, round half away from zero in the C library; values re-read by the next tool are the rounded ones | `np.savetxt('%f')` is the same; keep the rounding between stages when comparing to the image |
| `textread`, `sscanf('%f')` | `read_vtk`, `read_asc` | whitespace-separated, trailing `0` column of `.asc` ignored | — |
| `unique(A,'rows','first')` | adjacency building | lexicographic row order | `np.unique(axis=0)` sorts the same way |
| `sparse(i,j,v)` with repeats | `connectedComp`, stats | repeated indices are **summed** (all call sites deduplicate first) | `coo_matrix` also sums |
| `dmperm(W'+speye)` | components in `hemi_separation`, `connectedComp`, `roi_correction`, `cleanupBoundary` | block (≈ connected-component) numbering follows the block-triangular permutation, not size or first index | `scipy.sparse.csgraph.connected_components` numbers by first vertex; semantics-sensitive uses are marked in §4.2 and §6.8 |
| `sortrows(X,[1,-2])` | `roi_correction:48` | stable: by column 1 ascending, column 2 descending | `np.lexsort` is stable too; state the key order |
| `tabulate` | `roi_correction` | counts of each integer 1..K | — |
| `setdiff`, `ismember`, `unique` (no 'stable') | everywhere | sorted outputs | `np.setdiff1d`/`np.unique` sort |
| `find(triu(A))` + `ind2sub` | `roi_correction:32,56` | column-major order of the upper-triangle non-zeros | only the *set* of edges matters here |
| `zscore`, `std`, `mean`, `std` of stacked data | features, stats, PDFs | N−1 denominator; `mean([])` = NaN | `np.std(ddof=1)` |
| `prctile` | not used in MATLAB code (the `--prctile` is MeshProperty's) | — | — |
| `min`/`max` ties | `min(prob)`, `min(abs(x))` | first occurrence | `np.argmin` is the same |
| `num2str(x)` | `MeshDeform -t`, `--nThreads`, `--epRad`, `--lsTh` | integers print as integers; non-integers with 4 digits after the point (`num2str(2.5)` = `2.5`, `num2str(-1.2345678)` = `-1.2346`) | use `%.4g`-like carefully; for `trace_*` values with more than 4 decimals the tools receive a rounded number |
| `sprintf('%d', non-integer)` | `--surfsmoothing %d` | prints `2.500000e+00` style | validate `curv_smoothness` as integer |
| `strcat` | `flag` building (`:73-78`) | drops trailing white space of its arguments (leading kept) | build the flag list explicitly |
| `readtable('label.csv','HeaderLines',1)` | `parc.m`, stats, PDFs | column types inferred, first column text | `csv` module; keep column 3 optional |
| `which(file,'-all')`, `dir('<p>/lh*.pth')` | asset lookup | search path = `genpath($ROOT)`; `dir` returns names sorted | resolve explicitly from the protocol folder |
| `verLessThan('matlab','8.2')` | many | the MCR in the image is R2012a (v7.17): **the `< 8.2` branches are the ones that run** (`dataset`/`allwords`, `paruly` instead of `parula` for `< 8.4`) | follow the old branch when it differs |
| `exit(1)` on any error | `biscuit.m:473` | the process dies; partial outputs remain | raise |
| `interp3`, `bwdistgeodesic`, `imfill`, `smooth3`, `strel`, `imclose`, `imdilate`, `bwperim`, `padarray`, `DelaunayTri`, `histc` | sulcal depth, hemisphere patch, quant PDF | see §9.2 and §4.2 | the Image Processing Toolbox calls need black-box validation (no Octave reference: brief 07) |

## 8. Phase 2 — TRACE (`curve_extract_pipeline.m`, `biscuit.m:310-347`)

Inputs are the **output** surfaces `$OUT/surf/<h>.gray.vtk` (pial; sulci) and `$OUT/surf/<h>.white.vtk` (gyri), i.e. the closed hemisphere meshes of §4.6 *including the medial-wall patch*
(`biscuit.m:338,340`). Scratch `$TMP/tmp2` receives any existing `$OUT/curve/*`. Per call, the order is `lh` then `rh`; the order of calls is sulcus (gray) then gyrus (white).

Exact argv (T37; `n = omp_num_threads`, `r = trace_r`, `d = trace_delta`, both through `num2str`):

```
CurveExtraction --sulcus -i $OUT/surf/lh.gray.vtk  -o $TMP/tmp2/lh.gray  --nThreads n --epRad r --nhDist r --lsTh d --bary --noVtk
CurveExtraction --gyrus  -i $OUT/surf/lh.white.vtk -o $TMP/tmp2/lh.white --nThreads n --epRad r --nhDist r --lsTh d --bary --noVtk     (and rh)
```

(`curve_extract_pipeline.m:39,47`; a previous variant without `--noVtk` is commented out at `:38,46`.) A call is skipped when both `<o>.scurve`/`.gcurve` and `<o>.…curve.bary` exist (`:37,45`).
Everything else is at the tool default: seeds `--sulcalSeed −0.05`, `--gyralSeed 0.03`, `--prune 5`, `-s 0`, `--smoothTensorIter 3`, no `--nojunction`, no `--simplification`, no `--cart`, no `--interm`.
Products: `lh.gray.scurve`, `lh.gray.scurve.bary`, `lh.white.gcurve`, `lh.white.gcurve.bary` (and `rh.*`), copied to `$OUT/curve/` (`*.bary`, `*.?curve`, `biscuit.m:346`).
Only the **sulcal** `.bary` is read again by BISCUIT (stats, §10); both `.bary` files are read by `lgi` (§9.3). The figures (§11) read `.scurve`/`.gcurve` through `read_curve.m` (vertex-index
lines, 0-based, `str2num(line)+1`) and `.bary` through `read_bary.m` (§10). T38 (`MeshProperty … -m …` on the same surfaces) only colours the inflated-surface figure and its output is deleted (`:61`).

## 9. Phase 3 — quantification (`surf_quant_pipeline.m`, `biscuit.m:351-409`)

### 9.0 Files and cross-phase dependencies

| Phase | Needs (from `$OUT`) |
|---|---|
| 3 | `surf/<h>.{mid,gray,white}.vtk`; `curve/<h>.gray.scurve.bary`, `curve/<h>.white.gcurve.bary` (the `lgi` script reads them as `<surf dir>/../curve/<basename>.{s,g}curve.bary` (`lgi:111-112`), so **surfaces must sit in a sibling directory of `curve/`**); `quant/<h>.thickness.txt` (copied to `<h>.mid.CT.txt` in the scratch dir, `biscuit.m:369-370`); `roi/<h>.mask.txt` |
| 4 | `surf/<h>.mid.vtk`; `roi/<h>.mask.txt`, `roi/<h>.parc.<p>.txt`; `curve/<h>.gray.scurve.bary`; `reg/<h>.sphere.reg.vtk`; the phase-3 maps (renamed, `biscuit.m:426-452`) |

The surface kept in `$OUT/surf/` and used from phase 2 on is the same set that phase 1 produced (`biscuit.m:280-283`).

### 9.1 Mean curvature and shape index (T31)

On the **mid** surface: `MeshProperty -i <mid.vtk> -o <tmp3>/<h>.mid -s -m --surfsmoothing N --tensorsmoothing N` with `N = curv_smoothness` (default 2) → `<h>.mid.S.txt` (shape index),
`<h>.mid.H.txt` (mean curvature) (`surf_quant_pipeline.m:13-22`). Skipped if both exist. `MeshProperty` is the closed binary of plan row B3.

### 9.2 Sulcal depth (`SulcalDepth/sulcal_depth.m`; `VOXELISE/*`)

On the **gray (pial)** surface of each hemisphere; output `<h>.gray.SD.txt`, copied to `<h>.mid.SD.txt`; the depth is applied to the mid surface by index.
`sulcal_depth(surf, prefix)` with `pad = [20 20 20]`, `scale = 1`:

1. `outer_hull` (`:41-73`): `v := v − min(v)` per axis; `v := v·scale`; `vox = ceil(max(v))` (voxels per axis); `volume = VOXELISE(vox_x, vox_y, vox_z, struct('vertices',v,'faces',f+1), 'xyz')`
   (Aitkenhead; identical to LGI's `VOXELISE.m`, diff-checked 2026-10-21; brief 07's `quant/_voxelise.py` ports it); `padarray(volume, pad)` with zeros; `volume = imfill(volume, 26, 'holes')`
   (holes = background not 26-connected to the border) → `volume0`, the filled pial solid.
2. Hull: `double(volume)·255`; `smooth3(volume,'gaussian', 2·ceil(2·2)+1 = 9, 2)` (9×9×9 kernel, σ = 2); `imclose(volume, strel('ball',15,15))`; `imdilate(·, strel('ball',1,1))`; `hull = (volume > 25)·255`.
   `strel('ball',R,H)` is a **non-flat** structuring element (grey-scale offsets up to `H` inside radius `R`; default `N = 8` line-decomposition approximation) applied to the 3-D array plane by plane.
   Same code as `OuterHull.m` (brief 07 found its three MATLAB behaviours unverifiable without MATLAB/Octave). The voxel size here is **1 mm** (scale 1), unlike `OuterHull` which uses a 256-voxel longest side.
3. `distance(volume0, hull)` (`:75-100`): `field = ~volume0 & hull`; six times: `field(bwperim(~field,18) | bwperim(volume0,18)) = true` (the 18-connected shell of the complement
   of `field` and the 18-connected perimeter of the tissue are added); `seed = bwperim(hull, 18)`; `D = bwdistgeodesic(field, find(seed), 'quasi-euclidean')` — geodesic chamfer distance
   (steps 1, √2, √3) from the hull shell through `field`; voxels outside `field` are `NaN`.
   (A commented-out fast-marching alternative sits below it.)
4. `D2 = min(Inf, D)` (`min` ignores `NaN`, so unreachable voxels become `Inf`); build coordinate grids `X(:,i,:) = i−1−pad(2)`, `Y(i,:,:) = i−1−pad(1)`, `Z(:,:,i) = i−1−pad(3)`;
   `sd = interp3(X, Y, Z, D2, v(:,2), v(:,1), v(:,3))` — linear interpolation at the (shifted) vertex positions; `sd = sd/scale`; `sd(sd == Inf) = 0`. Printed min and max; written `%f\n`.
   Voxel `k` of the VOXELISE grid is taken to sit at coordinate `k−1` (the grid is `ceil(max(v))` cells over `[0, max(v)]`, so its true cell size is `max(v)/vox ≤ 1`: the approximation is part of the algorithm).
   Linear interpolation touching an `Inf` neighbour gives `Inf` (→ 0) or `NaN` (MATLAB's `0·Inf`), a behaviour that needs an image run to pin down.
The CBT v1.1 notes (README) record "SD: switched gray surface for depth measure (`?h.mid.SD.txt` is deprecated)" — the mid copy is kept for compatibility.

### 9.3 Local gyrification index (`Gyrification/gyrification.m`; `BISCUIT/GenericRecon/lgi`)

T32 on the pial surface of each hemisphere; the process working directory is `$TMP` (the wrapper `cd`s there), where the `lgi` script keeps its scratch directory `tmp-lgi-<name>` and leaves its result. The tcsh `lgi` script (v1.3.3m) then: (1) T33 `OuterHull <pial> <tmpdir>/<name>.hull.vtk`;
(2) T34 `klaplace -thread $OMP_NUM_THREADS -dims 128 <pial> <hull> -surfaceCorrespondence <tmpdir>/<name>.corr`, then `klaplace -conv <name>.corr_warpedMesh.vtp <name>.hull.corr.vtk`;
(3) copies `<surfdir>/../curve/<name>.scurve.bary` and `<whitedir>/../curve/<whitename>.gcurve.bary` into `<tmpdir>` (it *checks that `CurveExtraction` exists but does not run it*);
(4) T35 `Gyrification -i <pial> -o <tmpdir>/<name> --outer <name>.hull.corr.vtk -s <name>.scurve.bary -g <whitename>.gcurve.bary -m <kernel> --speed <anisotropy> --refHullArea <ref> --nThreads <OMP_NUM_THREADS>`;
(5) moves `<name>.lgi.map.*.txt` to the working directory. `gyrification.m` moves it to `<tmp3>/<h>.gray.LGI.txt` (`:13-15`) and `surf_quant_pipeline.m:45` copies it to `<h>.mid.LGI.txt`.
The kernel is passed to the script as `%f` (`316.000000`), so the map file is named from the tool's own formatting of that number (`lgi.map.316.txt` in the oracle, port note `lgi.md`).
Skipped when `<h>.gray.LGI.txt` exists. The output is per vertex of the hemisphere mesh, on the pial surface.

### 9.4 Shape complexity index (T36)

`ShapeComplexityIndex -i <mid.vtk> -o <tmp3>/<h>.mid.SCI.txt -k <sci_kernel_size, %f> -s <sci_bins, %d>` on the mid surface.

### 9.5 Thickness, areas, assembled files

Thickness is FreeSurfer's (`<h>.thickness.txt`, §4.6); copied to `<h>.mid.CT.txt`. Vertex areas (`<h>.mid.VertArea.txt`) come from phase 1 (`quant/<h>.mid.area.txt`).
`write_property` (`:78-79`) writes `<tmp3>/<h>.mid.vtk` with the fields `H, S, Depth, Gyrification, SCI, Thickness, Mask` (legacy `FIELD ScalarData 7`); this file is not copied to `$OUT`.
The figures are described in §11. `biscuit.m:403-405` copies `?h.*.H.txt`, `?h.*.S.txt`, `?h.gray.SD.txt`, `?h.gray.LGI.txt`, `?h.*.SCI.txt` into `$OUT/quant/`.

## 10. Phase 4 — statistics (`surf_quant_stat_pipeline.m`, `biscuit.m:411-470`)

Run once per protocol (`biscuit.m:453-455`) in `$TMP/tmp4`, which first receives the previous `stat/*` and the quant maps renamed as `?h.mid.{CT,LGI,SD}.txt`
(CT from `thickness`, LGI/SD from the `gray` files, `:444-452`). Arguments: mid surfaces, masks, the **sulcal** `.bary` files `curve/<h>.gray.scurve.bary`, the parcellations `roi/<h>.parc.<p>.txt`,
`reg/<h>.sphere.reg.vtk`, `output_pdf = false`.

1. **Region list.** `label.csv` rows plus a prepended row `Unknown, 0` (`:12-27`); `parc_h(~ismember(parc_h, region)) = 0`.
2. **Vertex-wise ROI statistics** (`stats`, `:391-455`): measures in the column order **[H, S, Depth, Thickness, SCI, LGI]**. For region `r`: stack the rows of `lh` with `parc == r` and the rows
   of `rh` with `parc == r`; `avg = mean`, `stdev = std` (N−1), `nv = count`. No cortical-mask test. An empty ROI gives `NaN`.
3. **Curve-wise statistics.** `read_bary` (`read_bary.m`): line 1 = number of curves; per curve a line with the point count, then rows `v1 v2 w` (`%d %d %f`; the reader skips blank lines).
   All rows of all curves are stacked, `unique(…,'rows')` (sorted; **removes duplicate junction points**, CBT v1.2d), `v1, v2 += 1`. Value at a curve point: `p(v1)·w + p(v2)·(1−w)`;
   its parcel: `parc(v1)` if `w ≥ 0.5` else `parc(v2)` (`th = 0.5`, `:65-69`). Per ROI: mean, std, count over the curve points of both hemispheres.
4. **Aggregates.** `2` = hemispheres: ids `20001` (lh) / `20002` (rh) = vertices with a non-zero label; `5` lobes (only when `label.csv` has a third column, i.e. braincolor):
   `parc_5 = (lobe−1)·2 + [rh] + 20003`, ids `20003…20012` = {L,R} × {frontal, temporal, parietal, occipital, insula}; "whole cortex" `20000` = mean/std over all vertices (and curve points) of both
   hemispheres whose label is in the real label list (`Unknown` removed before this step, `:121-123,142-168`). Names: `Whole cortex`, `Left hemisphere`, `Right hemisphere`, `Left|Right frontal|temporal|parietal|occipital lobe`, `Left|Right insula`.
5. **Table** `stats.<protocol>.csv` (`:177-262`): rows = `label.csv` rows (ids and names as in the file, the synthetic `Unknown`/0 row dropped at the end), then 20000–20002, then 20003–20012. Columns:
   `ID, Label, nVertices, MeanCurvature, MeanCurvature_stdev, ShapeIndex, ShapeIndex_stdev, SulcalDepth, SulcalDepth_stdev, Thickness, Thickness_stdev, SCI, SCI_stdev, LGI, LGI_stdev`,
   then the same twelve measure columns with a `_curve` suffix and `nVertices_curve` (names: `nVertices_curve, MeanCurvature_curve, MeanCurvature_stdev_curve, …, LGI_stdev_curve`). `nVertices` repeats `nv(:,1)`.
   Written by `writetable` (R2013b+) or, in the R2012a path that runs in the image, by `export(label,'file',…,'Delimiter',',')` (Statistics Toolbox `dataset`); the numeric format of that export is not
   visible in the source.
6. **Registered-space maps** (T40): `SphericalRemesh -s reg/<h>.sphere.reg.vtk -r sphere_327680 -p <tmp4>/<h>.mid.<M>.txt --outputProperty <tmp4>/<h>.mid.reg` for `M ∈ {H, S, SD, CT, LGI, SCI}`, each only if
   `<h>.mid.reg.<M>.txt` is missing. Copies out (`biscuit.m:460-468`): `stat/*.csv`; `stat/<h>.mid.reg.{H,S,SCI}.txt`; `stat/<h>.gray.reg.LGI.txt` (from `mid.reg.LGI`); `stat/<h>.gray.reg.SD.txt`; `stat/<h>.reg.thickness.txt`
   (from `mid.reg.CT`). `stat/<h>.mid.reg.area.txt` was copied in phase 1.
7. The stats figure and `<h>.mid.stat.vtk` are produced only when `output_pdf` is true; BISCUIT passes `false` (`biscuit.m:454`), so neither is made and nothing is merged into the PDF.

## 11. PDF figures and assembly

Every page is a LaTeX document (`common/latex/fig.txt`) compiled with `pdflatex` and re-saved with Ghostscript `-dPDFSETTINGS=/printer` (`run_GeneratePDF_*`): paper width 8.5 in, margin 0.1 in, `showframe` (draws the
geometry frame), header table (title; `Project/Subject/Session/Version/date`; references and software links with Font Awesome icons; author contact), the figure table, and the line
"BISCUIT: Brain Shape Computing Toolbox -- An Automated Pipeline for Cortical Morphometry". `Version` = contents of `$ROOT/VERSION` (the image: `2.2 (CBT_v1.3)`; cmorph `1.7cpu`).
Surface renders are MATLAB `trisurf` figures (OpenGL renderer under `xvfb`, `camlight`, `lighting phong`, `material dull`, `shading flat`) saved by `vecrast` (a 300 dpi raster inside a vector PDF).
Six or eight sub-views per tile (`SurfStatViewData_*`): lateral/medial of each hemisphere (azimuth ±90), top (`view(0,90)`), bottom (`view(0,-90)`), anterior/posterior (`view(0,0)`, `view(180,0)`).

| PDF (file in `$OUT/pdf`) | Title / `#pheight` | Content |
|---|---|---|
| `reg.<temp>.pdf` | "HSD: Cortical Surface Registration", 10.5 in | 3 × 2 tiles of 2.6 in: columns *Spherical Mapping* (native sphere, `H` with the mask zeroed), *Spherical Deformation* (registered sphere, same data), *Template (<temp>)` (`sphere_327680`, template `curv.mean`, colour range halved); rows lh, rh; caption `Left|Right Hemisphere (Energy: E0 → E1)` (`%.3f`); colour map `redbluecmap(11)`, range ±0.3 (±0.15 for the template) |
| `parc.<p>.pdf` | "Cortical Surface Parcellation", 8.7 in | 2 × 2 tiles of 4.0 in: white, gray, mid, inflated (lh shifted to `x ≤ −1`, rh to `x ≥ 1`); colours from `lut.csv` of the protocol (`id,R,G,B` ÷ 255; HSV ramp if missing; unlabeled black); links from `url.txt` |
| `curve.pdf` | "TRACE: Curve Extraction", 9.2 in | rows *Sulcal Curves* and *Gyral Curves*; tiles: the surface the curves were extracted on (gray for sulcal, white for gyral; files are named `<type>_mid.pdf`) and the inflated surface (curvature colours ±0.12) |
| `quant.pdf` | "Cortical Surface Quantification", 11.7 in | 3 × 2 tiles: mean curvature (parula/`paruly`, ±0.05), shape index (±1), thickness (jet, 1–5), sulcal depth (jet, 0–30), LGI (jet, 1–11), SCI (jet, 0–1); masked vertices (`Mask == 0`) are `−Inf` → black |

`curve_extract_pipeline` also writes the single-type pages `sulcus.pdf` and `gyrus.pdf` (5.7 in) into `tmp2/pdf`; the combined `curve.pdf` appears after the gyral call (when all four figure PDFs exist) and only it is copied to `$OUT/pdf` (`biscuit.m:347`).
The statistics PDF (`run_GeneratePDF_property.m`, 6 measures × 4 views × entire/curve ROIs, ranges H ±0.1, S ±0.7, SD 0–30, thickness 1–5, LGI 1–5, SCI 0.2–0.8) is dead code on this path.

**PDF merge (image wrapper, not in the source).** After `biscuit` succeeds: `cd $OUT/pdf && mkdir $OUT/PDF && gs -dNOPAUSE -sDEVICE=pdfwrite -dPrinted=false -sOUTPUTFILE=$OUT/PDF/biscuit.pdf -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dQUIET -dBATCH reg.*.pdf parc.* curve.pdf quant.pdf`
(image `run_biscuit:199-202`; page order by shell glob: `reg.<temp>.pdf`, `parc.braincolor.pdf`, `parc.dkt31.pdf`, `curve.pdf`, `quant.pdf`). `mkdir` fails when `PDF/` already exists, which skips the merge on a re-run.
The two example PDFs in `reference/vuiis_pdfs/` carry pages with MediaBox 612 × 626.4 pt (8.7 in) and 612 × 662.4 pt (9.2 in), consistent with `#pheight` above.

## 12. Other routes (brief)

**FreeSurfer** (`SurfPostproc/freesurfer/run_postproc_fs_pipeline.m`, registration `surf_reg_fs2.m`; `surf_reg_fs.m` and `surf_reg_fs3.m` are unreferenced older variants). Inputs `surf/{lh,rh}.{white,sphere,inflated,pial}` and
`surf/{lh,rh}.{curv,sulc,inflated.H,thickness}` (FreeSurfer curv files converted to `<name>.txt`, `%f\n`), optional `label/?h.cortex.label` (mask = vertices listed in the label; without it the mask is `parc > 0` after parcellation).
Surfaces are read with `read_surf` and written as VTK (`pial` → `gray`). No hemisphere splitting, no volume work, no `mris_*`. Registration is the same four HSD calls, but with `lh.res.H.txt` (the resampled `inflated.H`) as stage-1 input,
stage 4 on the FreeSurfer `curv` map (`lh.res.curv.txt`) instead of −4·H, and the correct `rh` argument order. Parcellation degree 0.
**CIVET** (`civet/run_postproc_civet_pipeline.m`, `surf_reg_civet.m`): `.obj` surfaces converted with `read_obj`; thickness from the supplied text files; `MeshDeform --smooth 3` + `mris_inflate` for the inflated surface;
the sphere is a bundled `sphere/civet_<h>_sphere_<N>.vtk`; mask is `parc > 0`; parcellation degree **10**; the registration code is a copy of `surf_reg_macruise.m` (including the `rh` stage-2 argument swap).
The FreeSurfer entry point `BISCUIT/FreeSurfer/biscuit.m` equals the GenericRecon one minus the MaCRUISE and CIVET branches.

## 13. Output tree

### 13.1 Final tree (`$OUT`, MaCRUISE route, protocols `braincolor,dkt31`, template `<t>`)

Derived from `biscuit.m:260-307,343-347,400-407,457-468` and the wrapper. Format: `V` = legacy ASCII VTK POLYDATA, 6-decimal; `T` = one value per line (`%f` or `%d`); `B` = `.bary`/curve text.

```
surf/   lh.white.vtk  rh.white.vtk  lh.gray.vtk  rh.gray.vtk  lh.mid.vtk  rh.mid.vtk
        lh.inflated.vtk  rh.inflated.vtk  lh.sphere.vtk  rh.sphere.vtk        (V, closed hemisphere meshes incl. medial patch; FreeSurfer surface frame, §4.4)
        hemi_map.txt                                                          (T3: whole-brain idx, hemisphere idx, 0/1 hemisphere; MaCRUISE only)
reg/    lh.white.reg.vtk lh.gray.reg.vtk lh.mid.reg.vtk lh.sphere.reg.vtk (+ rh)   (V; template tessellation, 163,842 vertices; sphere.reg = native-tessellation sphere deformed)
        lh.coeff.txt  rh.coeff.txt                                            (HSD coefficients, degree 10: pole line, then x/y/z per SH basis index)
roi/    lh.mask.txt rh.mask.txt                                               (T, 1 = cortex, 0 = medial patch)
        lh.parc.braincolor.txt lh.parc.dkt31.txt (+ rh)                       (T, integer label ids)
curve/  lh.gray.scurve  lh.gray.scurve.bary  lh.white.gcurve  lh.white.gcurve.bary  (+ rh)   (B)
quant/  lh.thickness.txt rh.thickness.txt   lh.mid.area.txt rh.mid.area.txt
        lh.mid.H.txt lh.mid.S.txt lh.gray.SD.txt lh.gray.LGI.txt lh.mid.SCI.txt (+ rh)
stat/   lh.mid.reg.area.txt rh.mid.reg.area.txt
        lh.mid.reg.H.txt lh.mid.reg.S.txt lh.mid.reg.SCI.txt  lh.gray.reg.LGI.txt lh.gray.reg.SD.txt lh.reg.thickness.txt (+ rh)
        stats.braincolor.csv  stats.dkt31.csv
pdf/    reg.<t>.pdf  parc.braincolor.pdf  parc.dkt31.pdf  curve.pdf  quant.pdf
PDF/    biscuit.pdf                                                           (image wrapper only)
```

`$OUT/.tmp` is deleted by the wrapper; the source `biscuit.m` itself leaves `tmp1…tmp4` (the `rm -rf` lines are commented out), which holds all intermediates. On the FreeSurfer and CIVET routes `surf/hemi_map.txt` is absent. Naming caveat: LGI and SD are named `gray` while all other measures are named `mid`, and `stat/` renames `mid.reg.CT` to `reg.thickness`.

### 13.2 Intermediates worth keeping in a port (`$TMP/tmp1`, MaCRUISE)

`<h>.target_image_GMimg_innerSurf.vtk` (hemisphere inner surface, with patch), `tab.txt`, `mri/{T1,brain.finalsurfs,wm,filled,aseg}.mgz`, `surf/<h>.{orig,white,pial,thickness}`, `<h>.smoothwm.vtk`,
`<h>.sulc`, `<h>.inflated.H.txt`/`<h>.inflated.H`, `<h>.{H,iH,sulc}.txt`, `<h>.res.{H,iH,sulc}.txt`, `<h>.res.sphere.reg.vtk`, `<h>.reg.H.txt`, parcellation scratch (`tempname` directory, deleted at the end of `parc`).

## 14. Quirks, source bugs, licence and drift

1. `rh` HSD stage 2 argument order is swapped relative to `lh` (macruise `:112`, civet `:112`), correct in `surf_reg_fs2.m`.
2. `brain.finalsurfs` "refine" block is dead in a fresh run (§4.5, T10); the mask/normalisation therefore never uses `aseg`.
3. `flag` undefined on a re-run with an existing `T1.mgz` (§4.3).
4. `roi_correction` can call `gcut_` with an empty candidate set (§6.8).
5. `dmperm` block order decides which components are hemispheres/islands in `hemi_separation` (§4.2); the Python rule must be semantic, not positional.
6. `getBoundary` and the patch triangulation assume exactly two boundary loops and a hull that starts at the loop start; `MeshContour` picks the plane by an unknown rule.
7. Outputs are in the FreeSurfer surface frame (§4.4), not scanner RAS; `trace`, LGI and all measures inherit it. Resolving the `MeshDeform -t` sign needs one image run.
8. The R2012a runtime (`verLessThan('matlab','8.2')` true; `DelaunayTri`, `dataset`, `allwords`, `paruly`) selects the older code branches; behaviour of newer MATLAB differs (e.g. `writetable`, `parula`).
9. `GCMex`/`GCoptimization`/Boykov–Kolmogorov are GPL v2 with a research-only clause and a patent notice: do not vendor or translate them; a licence-clean max-flow with the same float32 energy and label order is required
   (port class: exact, tie-sensitive).
10. FreeSurfer-licensed MATLAB helpers (`macruise/FreeSurfer`, `common/matlab/FreeSurfer`): not reproduced. The binary formats they use (triangle surface magic 16777214, curv magic 16777215, big-endian) are the public FreeSurfer formats.
11. **Drift (source 2022-02 vs image 2021-06).** No difference found in `check_input`, `run_biscuit` (except the PDF merge), `biscuit.conf` keys and the parcellation Python. The `VERSION` file in the source is `CBT_v1.3`, as in the image. The orchestrator
    itself (encrypted) cannot be diffed; the first Linux run must compare at least: the four HSD argv strings, the TRACE argv, the MeshProperty argv, and the output file list of §13.

## 15. What the first Linux run must confirm

Source facts that only the image can settle (each is cheap to check from an instrumented run or a golden):
(1) the four HSD argv strings and the `rh` stage-2 order; (2) the frame of `surf/*.vtk` against the input `.asc` (the `MeshDeform -t` sign, §4.4);
(3) the number of patch vertices and the vertex counts of `lh.target_image_GMimg_innerSurf.vtk`; (4) `MeshContour`'s plane choice (probe with synthetic planes);
(5) `MeshProperty` semantics of `--normbefore --normafter --prctile 0.1 --datasmoothing 25 --invert -q` (probe with an analytic surface);
(6) sulcal-depth values for the `strel('ball')` / `interp3` / `bwdistgeodesic` behaviours (§9.2) and hull area; (7) the numeric format of `stats.<p>.csv`;
(8) the `GyralCurve` uninitialised-`current` branch under the image's compiler (open item of brief 04); (9) the heap flavour in `Gyrification` and the `acosf` of the image's glibc (open items of brief 07).
