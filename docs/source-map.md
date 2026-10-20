# Source map

Inventory of everything `biscuit_FC_v2.2.sif` runs on the MaCRUISE route,
where its source lives, and where it lands in pyBiscuit. Line counts are
upstream `master` at clone time (2026-10-20); the image binaries are CBT v1.3 /
cmorph v1.7 (June 2021) and may differ.

**Port class.** *Exact*: match the reference binary on identical inputs.
*Replacement*: no portable source (FreeSurfer, closed binaries) or owner-chosen
redesign; judged on downstream agreement. *Asset*: data, not code.

## Orchestration (MATLAB, encrypted)

`/opt/biscuit/biscuit` is an MCR v7.17 (MATLAB R2012a) executable. Its CTF
archive holds 84 encrypted `.m` files and 61 plaintext files; see
[reference/ctf-manifest.tsv](reference/ctf-manifest.tsv).

| CTF module | Role (inferred from names + PDF + wrapper) | pyBiscuit target |
|---|---|---|
| `biscuit/biscuit.m` | entry; reads conf; dispatches route; stages; PDF | `pybiscuit.pipeline` |
| `common/matlab/read_config` | parses `biscuit.conf` (`key = value`, `%` comments) | `pybiscuit.config` |
| `common/matlab/IO/{read,write}_vtk.m`, `write_property.m` | VTK legacy + per-vertex `.txt` | `pybiscuit.io` |
| `SurfPostproc/macruise/run_postproc_macruise_pipeline.m` | MaCRUISE route surface prep | `pybiscuit.surfprep.macruise` |
| `SurfPostproc/macruise/tessellation/hemi_separation.m`, `connectedComp.m`, `circle_patch.m`, `surfNormalCorrection.m` | split whole-brain inner surface into closed lh/rh meshes | `pybiscuit.surfprep.hemisphere` |
| `SurfPostproc/macruise/FreeSurfer/*` | FS I/O helpers (MRIread, write_surf, read_asc, curv) | `pybiscuit.io.freesurfer` (via NiBabel) |
| `SurfPostproc/macruise/surf_reg_macruise.m` | builds features, calls HSD | `pybiscuit.register` |
| `SurfPostproc/freesurfer/*`, `civet/*` | FS / CIVET routes | later phase (see open questions) |
| `SurfPostproc/common/matlab/parc/{parc,extract_features,gcut,gcut_,roi_correction}.m` | parcellation driver, features, graph-cut smoothing, ROI cleanup | `pybiscuit.parc` |
| `SurfPostproc/common/matlab/gcut/*` | GCMex (Boykov–Kolmogorov alpha-expansion) | `pybiscuit.parc.graphcut` |
| `CurveExtraction/curve_extract_pipeline.m`, `IO/read_curve.m` | runs TRACE for sulci/gyri | `pybiscuit.trace` |
| `SurfQuant/surf_quant_pipeline.m` | runs all measures | `pybiscuit.quant` |
| `SurfQuant/SulcalDepth/sulcal_depth.m` + `VOXELISE/*` | sulcal depth via voxelised hull | `pybiscuit.quant.sulcal_depth` |
| `SurfQuant/Gyrification/gyrification.m` | wraps `lgi` script | `pybiscuit.quant.lgi` |
| `SurfQuant/SCI/sci.m` | wraps ShapeComplexityIndex | `pybiscuit.quant.sci` |
| `SurfQuantStat/surf_quant_stat_pipeline.m`, `IO/read_bary.m` | ROI-wise, curve-wise, whole-cortex stats | `pybiscuit.stats` |
| `*/Visualization/*`, `common/latex/*`, `vecrast*.m` | MATLAB figures + LaTeX → PDF pages | `pybiscuit.viz` (redesign) |

## Plaintext Python in the CTF (parcellation)

| File | Notes |
|---|---|
| `SurfPostproc/common/parc/{model,test,loader_individual,cpuModel}.py`, `meshcnn/*.py` | UGSCNN-style Spherical U-Net (MeshConv), torch 1.4. `test.sh` args: `--feat 64 --in_ch 3 --min_level 0 --no-cuda`. Output per-class probabilities resampled to native sphere with `SphericalRemesh`. |
| `mesh_files/icosphere_{0..5}.pkl`, `icosphere_{5,6}.vtk` | Mesh operators (Laplacian, gradient) for MeshConv. Regenerate & verify against these. |
| Upstream twin | `ilwoolyu/SphericalLabeling` (MIT) — training code + same model. |

## Compiled tools

| Binary | Source | Size | Class | pyBiscuit module | Main Python building blocks |
|---|---|---|---|---|---|
| `cmorph/bin/HSD` | ilwoolyu/HSD `src/HSD.cpp` 3.1k + `bobyqa.h` 2.9k (+CUDA 1.3k, optional) | large | exact | `pybiscuit.hsd` | NumPy/Numba; SciPy `lstsq`/`cho_solve` for LM step; port `bobyqa.h` (MIT-like, Powell/Thiébaut) |
| `cmorph/bin/CurveExtraction` | ilwoolyu/CurveExtraction ~4.9k | large | exact | `pybiscuit.trace` | SciPy `csgraph.dijkstra`; exact geodesic port from MeshLib `GeodesicA` |
| `cmorph/bin/Gyrification` | ilwoolyu/LocalGyrificationIndex `src/Gyrification.cpp` 1k | medium | exact | `pybiscuit.quant.lgi` | anisotropic fast marching (Numba) |
| `cmorph/bin/OuterHull` | LGI `matlab/OuterHull.m` + VOXELISE + ICP_finite (MATLAB, source available) | medium | exact | `pybiscuit.quant.hull` | SciPy ndimage morphology; marching cubes (scikit-image) |
| `cmorph/bin/klaplace` | ilwoolyu/klaplace ~2k (VTK5) | medium | exact | `pybiscuit.quant.klaplace` | Laplace on voxel grid, RK45 streamlines (SciPy) |
| `cmorph/bin/lgi` | tcsh script | small | exact | `pybiscuit.quant.lgi` | — |
| `cmorph/bin/SphericalRemesh` | ilwoolyu/SphericalRemesh 0.6k | small | exact | `pybiscuit.sphere.remesh` | AABB on sphere + barycentric interpolation |
| `biscuit/bin/ShapeComplexityIndex` | NIRALUser/ShapeComplexityIndex ~1.7k + transportSimplex (EMD) | medium | exact | `pybiscuit.quant.sci` | EMD via SciPy `linprog` or direct port |
| `biscuit/bin/MeshProperty` | **closed** (Lyu; flags: `--tensorsmoothing --surfsmoothing --datasmoothing --prctile --normbefore --normafter --invert -q`) | — | exact (black-box) | `pybiscuit.mesh.property` | MeshLib `Util/SurfaceUtil.cpp` curvature as starting point |
| `biscuit/bin/MeshDeform` | **closed** (`--matrix --translate --scale --sphere --smooth --noise --invertFace`) | — | exact (black-box) | `pybiscuit.mesh.deform` | trivial affine/smoothing |
| `biscuit/bin/MeshContour` | **closed**, no usage strings | — | black-box TBD | `pybiscuit.mesh.contour` | — |
| MeshLib (`libMesh.so`) | ilwoolyu/MeshLib (MIT) incl. Peyré geodesic toolbox | large | exact where used | `pybiscuit.mesh` | — |
| `biscuit/bin/mri_convert`, `mri_normalize` | FreeSurfer 6 | — | replacement | `pybiscuit.surfprep.volume` | NiBabel; simple intensity normalisation |
| `biscuit/bin/mris_make_surfaces` | FreeSurfer 6 | — | replacement | `pybiscuit.surfprep.pial` | deform white → PyMaCRUISE outer level set (proposed) |
| `biscuit/bin/mris_inflate` | FreeSurfer 6 | — | replacement | `pybiscuit.surfprep.inflate` | metric-preserving smoothing; `sulc` accumulation |
| `biscuit/bin/mris_sphere` | FreeSurfer 6 (+ `lib/bem/ic*.tri`) | — | replacement | `pybiscuit.surfprep.sphere` | inflate → project → distortion-minimising relaxation |

## Assets

| Asset | Size | Content | Notes |
|---|---|---|---|
| `biscuit/reg/{mb101,oasis45,hcp1058}/` | 201 MB | `?h.{curv,sulc,iH}.{mean,var}.txt` (163,842 vertices = ico7), `?h.white.vtk`, `?h.inflated.vtk` | HSD templates. MASI production uses `--reg oasis45`; image default `mb101`. |
| `biscuit/parc/braincolor/` | 170 MB | `?h.pth` (one model each), `label.csv`, `lut.csv`, `?h.curv.mean.txt`, `url.txt` | BrainCOLOR protocol |
| `biscuit/parc/dkt31/` | 813 MB | `?h.pth`, `?h_{2..5}.pth` (5-model ensemble), labels, LUT | DKT31 protocol |
| `biscuit/sphere/sphere_{5120,20480,81920,327680}.vtk` | 18 MB | icospheres (faces = 20·4^k) | resampling targets |
| `biscuit/lib/bem/ic{1..7}.tri`, `*.dat` | 17 MB | FreeSurfer icosahedra for `mris_sphere` | replaced; do not ship |

## Papers

- HSD: Lyu et al., *Med Image Anal* 57:72–88, 2019.
- TRACE: Lyu et al., *IEEE TMI* 37(7):1653–1663, 2018.
- LGI: Lyu et al., *Med Image Anal* 48:244–258, 2018; MICCAI 2017.
- Sulcal depth: Lyu et al., *Proc SPIE* 10574, 2018.
- SCI: Kim et al., *NeuroImage* 135:163–176, 2016.
- Parcellation: Parvathaneni et al., MICCAI 2019; Lyu et al. spherical data augmentation (SphericalLabeling).
- klaplace: Lee et al., *Proc SPIE* 9784, 2016.
