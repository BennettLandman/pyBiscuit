# I/O contract

## Inputs

### Primary: a PyMaCRUISE BASE (or PLUS) run directory

pyBiscuit reads the run's `run.json` (`final_outputs`) to locate:

| Product | PyMaCRUISE file (current naming) | Use |
|---|---|---|
| Inner (WM/GM) surface | `inner_surface.vtk` (RAS mm, whole brain, header `PyMaCRUISE coordinate_frame=RAS units=mm`) | white surface |
| Central surface | `central_surface.vtk` | optional reference for mid surface |
| Outer (GM/CSF) surface | `outer_surface.vtk` | pial target |
| Level sets | inner/central/outer level sets `.nii.gz`, `thickness_levelset.nii.gz` | pial deformation, thickness |
| Refined labels | `SEG_refine.nii.gz` (BrainCOLOR labels) | hemisphere assignment, medial wall |
| T1 | working/reference T1 (`.nii.gz`, RAS) | intensity sampling if needed |
| Thickness summary | `inner_surface_thickness.vtk`, `thickness_summary.json` | comparison only |

Rules:
- Verify `status` and stage completion in `run.json`; refuse incomplete runs
  unless `--allow-incomplete`.
- Record input paths, SHA-256, PyMaCRUISE version and run signature in
  pyBiscuit's manifest.
- Surface and volume frames must agree (RAS mm). Reject on mismatch.
- PyMaCRUISE does **not** produce MNI-space surfaces (template absent; see its
  `docs/io-contract.md`). BISCUIT's `mc_surf_mni` input is therefore not
  available on this route; see the hemisphere step in plan.md and
  [open-questions.md](open-questions.md).

### Legacy: MaCRUISE v3 output (for reference parity)

Exactly what `biscuit_FC_v2.2.sif --surf_recon macruise` consumes:

| `biscuit.conf` key | File | MaCRUISE v3 source |
|---|---|---|
| `mc_surf` | `target_image_GMimg_innerSurf.asc` | `Output/Surfaces_FreeView/` |
| `mc_surf_mni` | `target_image_GMimg_innerSurf_mni.asc` | `Output/Surfaces_MNI/target_image_GMimg_innerSurf.asc` (renamed) |
| `mc_rawT1` | `T1.nii.gz` | `Raw/T1.nii.gz` (sample: `output/Raw/T1.nii.gz`) |
| `mc_rawSeg` | `SEG_refine.nii.gz` | `Output/SegRefine/SEG_refine.nii.gz` |

`~/dev/mipav/sample-macruise/output/` (and `100249/outputs/`) contain these
files and are the first full-pipeline reference cases. FreeView `.asc` is
FreeSurfer ASCII surface (`#!ascii`, `nV nF`, `x y z 0`, `i j k 0`) in tkRAS;
the loader must convert to scanner RAS using the T1 header and record it.

### Later routes

`surf_recon = freesurfer` (FS6 `surf/` + `label/`) and `civet` exist in the
original. FreeSurfer-route *input reading* needs only NiBabel (no FreeSurfer
install) and is the cleanest way to validate the exact-class components in
isolation; see plan phase B1.

## Configuration

`biscuit.conf` keys and defaults (image defaults; MASI production override
`reg_temp = oasis45`):

```
omp_num_threads = 1        surf_smooth = 0           reg_temp = mb101
parc_protocol = braincolor,dkt31
trace_r = 4                trace_delta = 2.5         curv_smoothness = 2
lgi_kernel_size = 316      lgi_anisotropy = 0.2      lgi_ref_area = 77100
sci_kernel_size = 3        sci_bins = 5
```

pyBiscuit accepts the same keys from a TOML/INI file and CLI flags named as in
`run_biscuit` (`--reg`, `--parc`, `--trace_r`, …). It must also read an
original `biscuit.conf` unchanged.

## Outputs

The original's output tree is not yet documented (the wrapper deletes `$TMP`;
only `PDF/biscuit.pdf` and stage outputs remain). Plan row B1.3 records the exact
tree from a reference run; pyBiscuit then mirrors names in a `legacy/` view.

Native pyBiscuit layout (BIDS-derivatives-friendly):

```
<out>/
├── run.json                      manifest: inputs+hashes, config, versions, assets+hashes, stages, frames
├── stages/<stage>/attempt-NN/    checkpointed stage products + provenance.json
├── surf/{lh,rh}.{white,pial,mid,inflated,sphere,sphere.reg}.surf.gii
├── metric/{lh,rh}.{curv,sulc,iH,thickness,shape_index,sulcal_depth,lgi,sci}.func.gii
├── label/{lh,rh}.{braincolor,dkt31}.label.gii
├── curves/{lh,rh}.{scurve,gcurve}{,.bary}  (+ .vtk polylines)
├── stats/{roi,curve,cortex}_<measure>.tsv
├── legacy/                       original BISCUIT names/formats (VTK + .txt) for drop-in use
└── qa/biscuit.pdf, qa/index.html, qa/figures/*.png
```

All coordinates RAS+ mm. Per-vertex `.txt` (one value per line) is kept in
`legacy/` because HSD/TRACE/parcellation tooling expects it.
