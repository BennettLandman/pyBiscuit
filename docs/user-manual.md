# pyBiscuit User Manual & CLI Reference

**pyBiscuit** is a pure Python / C-extension-free cortical surface morphometry pipeline. It replaces the legacy Singularity container (`biscuit_FC_v2.2.sif`) and its encrypted MATLAB R2012a, FreeSurfer 6 binaries, and C++ MEX dependencies with a cross-platform, multi-threaded implementation that runs natively on macOS (Apple Silicon / Intel) and Linux.

---

## 1. Installation

### Requirements
- **Python**: 3.10 – 3.13
- **Core dependencies**: `numpy`, `scipy`, `nibabel`, `numba`
- **Optional extras**:
  - `[parc]`: `torch>=2`, `safetensors` (for spherical U-Net inference)
  - `[viz]`: `pyvista`, `matplotlib` (for 3D off-screen rendering)
  - `[dev]`: `pytest`, `ruff`, `build`

### Install from Source
```bash
# Clone the repository
git clone https://github.com/BennettLandman/pyBiscuit.git
cd pyBiscuit

# Install core pipeline
pip install -e .

# Or install with deep learning parcellation and visualization extras:
pip install -e ".[parc,viz]"
```

### Model & Template Assets
pyBiscuit uses reference registration templates (`mb101`, `oasis45`, `hcp1058`), icosphere meshes (`sphere_5120` to `sphere_327680`), and trained parcellation checkpoints (`braincolor`, `dkt31`).

Fetch or verify assets:
```bash
# Verify status of default cache (~/.cache/pybiscuit/assets)
pybiscuit info

# Populate assets from a local extracted reference directory
pybiscuit fetch-assets --from-dir /path/to/extracted/opt/biscuit

# Or verify existing cache integrity
pybiscuit fetch-assets --verify-only
```
You can override the asset location at runtime by setting the `PYBISCUIT_ASSETS` environment variable.

---

## 2. Quickstart with Provided Sample Data

Sample datasets are hosted in the **[`pyBiscuit-examples`](https://github.com/BennettLandman/pyBiscuit-examples)** repository:
```bash
git clone https://github.com/BennettLandman/pyBiscuit-examples.git open-examples
```

A comprehensive, step-by-step tutorial based on real sample subjects (`open-examples/0050952`, `0050961`, `0050971`) is available in:
**[docs/sample-data-walkthrough.md](sample-data-walkthrough.md)**

Quick commands:
```bash
# 1. Dry-run pipeline inspection on sample subject
pybiscuit run open-examples/0050952 /tmp/run_demo --dry-run

# 2. View 3D cortical surfaces, thickness, LGI, and parcellations in WebGL:
open open-examples/0050952/qa/index_embedded.html

# 3. Generate multi-subject cohort mosaic and parity report:
pybiscuit cohort open-examples/0050952 open-examples/0050961 open-examples/0050971 -o /tmp/cohort_demo
open /tmp/cohort_demo/cohort_report.html
```

---

## 3. Pipeline Execution (`pybiscuit run`)

The primary entry point is `pybiscuit run`, which automatically detects input types and coordinates all execution stages.

### Command Syntax
```bash
pybiscuit run <INPUT_DIR> <OUTPUT_DIR> [OPTIONS]
```

### Supported Input Adapters
pyBiscuit natively recognizes three input directory structures:
1. **PyMaCRUISE run directory** (recommended, modern):
   - Contains `run.json`, inner/outer surface meshes, and `SEG_refine` conformed segmentation.
2. **MaCRUISE v3 directory** (legacy):
   - Contains `Surfaces_FreeView/` with `target_image_GMimg_innerSurf.asc`, `aff.txt`, and conformed MRI.
3. **FreeSurfer subject directory**:
   - Contains `surf/` (`lh.white`, `lh.pial`, `lh.sphere`, `rh.white`, `rh.pial`, `rh.sphere`).

You can explicitly force an adapter using `--input-type pymacruise|macruise_v3|freesurfer`.

### Common Options
- `--threads N`: Number of parallel CPU threads (default: hardware concurrency or config).
- `--resume`: Skip completed stages whose input signatures and assets match.
- `--until STAGE`: Run up to and including a specified stage (`surfprep`, `features`, `register`, `parc`, `trace`, `quant`, `stats`, `report`).
- `--dry-run`: Plan stages, verify signatures and inputs, but do not execute.
- `--template NAME`: Registration template (default: `mb101`).
- `--protocols PROTO1 PROTO2`: Parcellation protocols (default: `braincolor dkt31`).
- `--legacy`: In addition to modern GIfTI/TSV outputs, write the exact `legacy/` output tree matching `biscuit_FC_v2.2.sif`.

### Example Run
```bash
pybiscuit run /data/subjects/sub-01 /data/derivatives/sub-01 \
    --thread 8 \
    --parc "braincolor,dkt31" \
    --legacy
```

### Legacy SIF & `run_biscuit` Drop-in Compatibility

For backwards-compatibility with existing SLURM batch scripts, cluster pipelines, and Singularity wrapper workflows built for `biscuit_FC_v2.2.sif`:

1. **Direct `run_biscuit` CLI command**:
   pyBiscuit registers `run_biscuit` and `biscuit` as top-level console script aliases:
   ```bash
   run_biscuit --surf_recon freesurfer --reg mb101 /INPUTS /OUTPUTS
   ```
2. **Direct `pybiscuit` invocation without `run`**:
   `pybiscuit` automatically detects legacy flags and routes directly to the pipeline runner:
   ```bash
   pybiscuit --surf_recon freesurfer --reg mb101 /INPUTS /OUTPUTS
   ```
3. **Apptainer / Singularity SIF drop-in execution**:
   The `pybiscuit.sif` container image runscript executes `run_biscuit` syntax directly:
   ```bash
   singularity run --cleanenv \
       -B /path/to/inputs:/INPUTS \
       -B /path/to/outputs:/OUTPUTS \
       pybiscuit.sif \
       --surf_recon freesurfer \
       --reg mb101 \
       --thread 4 \
       /INPUTS /OUTPUTS
   ```
   All original `run_biscuit` flags are accepted verbatim: `--reg`, `--surf_recon`, `--parc`, `--thread`, `--proj`, `--subj`, `--sess`, `--trace_r`, `--trace_delta`, `--curv_smooth`, `--lgi_kernel`, `--sci_kernel`.

---

## 3. Pipeline Stages

| Stage | Name | Description | Key Products |
|---|---|---|---|
| 1 | `surfprep` | Hemisphere separation (MaCRUISE) or FreeSurfer ingest, white/pial refinement, mid-surface $(white + pial)/2$, inflation, spherical conformal mapping | `lh.white.surf.gii`, `lh.pial.surf.gii`, `lh.mid.surf.gii`, `lh.inflated.surf.gii`, `lh.sphere.surf.gii` |
| 2 | `features` | Multi-scale mean curvature ($H$), shape index ($S$), integrated curvature ($iH$), sulcal depth ($sulc$) | `lh.curv.func.gii`, `lh.sulc.func.gii` |
| 3 | `register` | Four-call Harmonic Spherical Deformation (HSD) multi-resolution registration to template | `lh.sphere.reg.surf.gii`, `lh.coeff.txt` |
| 4 | `parc` | Spherical U-Net inference on icosahedron, fold fusion, graph-cut energy minimization, ROI island cleanup | `lh.braincolor.label.gii`, `lh.dkt31.label.gii` |
| 5 | `trace` | Sulcal fundic and gyral crown point extraction, Dijkstra graph path tracing, pruning | `lh.scurve.bary`, `lh.gcurve.bary`, curves VTK |
| 6 | `quant` | Outer cerebral hull, klaplace Dirichlet PDE streamlines, Local Gyrification Index (LGI), Shape Complexity Index (SCI), sulcal depth (SD) | `lh.gray.LGI.txt`, `lh.mid.SCI.txt`, `lh.gray.SD.txt`, `lh.thickness.txt` |
| 7 | `stats` | Vertex-area-weighted ROI and curve summary statistics; resampling to template tessellation | `stats.braincolor.csv`, `stats.dkt31.csv`, `stat/*.reg.*` |
| 8 | `report` | Pure-Python QA PDF report generation and offline NiiVue HTML viewer | `PDF/biscuit.pdf`, `qa/index.html` |

---

## 4. Standalone CLI Tools

Each core algorithm can also be invoked as a standalone CLI utility:

### HSD Registration (`pybiscuit hsd` / `pybiscuit register`)
```bash
# Full four-call BISCUIT chain
pybiscuit register -s input_sphere.vtk -t template_dir -o out_dir -h lh

# Low-level HSD binary equivalent
pybiscuit hsd input_sphere.vtk target.vtk output_prefix --deg 8 --nThreads 4
```

### Sulcal / Gyral Curve Extraction (`pybiscuit trace`)
```bash
pybiscuit trace -i lh.white.vtk -o lh.white --gyrus --nhDist 20 --lsTh 0.5
pybiscuit trace -i lh.pial.vtk -o lh.gray --sulcus --nhDist 20 --lsTh 0.5
```

### Quantification Tools
```bash
# Local Gyrification Index (LGI)
pybiscuit lgi -i lh.pial.vtk -o lh.lgi --outer lh.hull.corr.vtk -s lh.scurve.bary -g lh.gcurve.bary

# Shape Complexity Index (SCI)
pybiscuit sci -i lh.mid.vtk -o lh.mid.SCI.txt -k 30.0 -s 100

# klaplace PDE Streamlines Surface Correspondence
pybiscuit klaplace lh.pial.vtk lh.hull.vtk -surfaceCorrespondence lh.corr -thread 8

# Sulcal Depth
pybiscuit sulcal-depth lh.pial.vtk -o lh.gray.SD.txt

# Spherical Remeshing
pybiscuit spherical-remesh -p input_sphere.vtk -s target_mesh.vtk -r reg_coeff.txt -o out_mesh.vtk
```

### Visualization & QA Reports
```bash
# Generate QA PDF report
pybiscuit report -o /data/derivatives/sub-01 --subject sub-01

# Generate Cohort Mosaic & Parity Report
pybiscuit cohort /data/runs/sub-01 /data/runs/sub-02 /data/runs/sub-03 \
    -o /data/reports/cohort \
    --protocols braincolor dkt31
```

---

## 5. Output Data Structure

### Modern Layout (BIDS Derivatives)
```
<out_dir>/
├── surf/                       # Cortical surface meshes (GIfTI)
│   ├── lh.white.surf.gii
│   ├── lh.pial.surf.gii
│   ├── lh.mid.surf.gii
│   ├── lh.inflated.surf.gii
│   └── lh.sphere.reg.surf.gii
├── quant/                      # Morphometric scalar maps (GIfTI & text)
│   ├── lh.thickness.func.gii
│   ├── lh.mid.H.txt
│   ├── lh.mid.S.txt
│   ├── lh.gray.SD.txt
│   ├── lh.gray.LGI.txt
│   └── lh.mid.SCI.txt
├── roi/                        # Parcellation label maps (GIfTI)
│   ├── lh.braincolor.label.gii
│   └── lh.dkt31.label.gii
├── curve/                      # TRACE sulcal and gyral curves
│   ├── lh.gray.scurve
│   ├── lh.white.gcurve
│   └── lh.gray.scurve.bary
├── stats/                      # Regional summary tables
│   ├── stats.braincolor.csv
│   └── stats.dkt31.csv
├── qa/                         # Interactive offline NiiVue WebGL viewer
│   └── index.html
└── PDF/                        # QA visual report
    └── biscuit.pdf
```

### Legacy Layout (`legacy/` directory)
When `--legacy` is specified, `pybiscuit` populates a `legacy/` sub-folder that mirrors the exact directory layout and naming conventions of `biscuit_FC_v2.2.sif` (including `stat/`, `roi/`, `quant/`, and `PDF/biscuit.pdf`).

---

## 6. Troubleshooting & FAQ

### 1. `PermissionError` when accessing reference datasets in sandboxed environments
In strict sandboxed environments without filesystem elevation, external paths outside the workspace require explicit access or can be skipped during normal non-reference testing. Standard testing runs cleanly via `pytest -m "not reference"`.

### 2. Can pyBiscuit run headlessly on compute clusters?
**Yes.** All core algorithms (registration, parcellations, curves, shape measures, stats) and report generators (PDF and NiiVue offline HTML) run headlessly with zero display server dependencies (`Xvfb`, `DISPLAY`, or OpenGL contexts are NOT required for pipeline completion or QA PDF generation).

### 3. Out of memory during Parcellation or Registration?
For memory-constrained systems (e.g. < 8 GB RAM), use `--threads 2` or `--threads 4`. The U-Net parcellation inference and graph cut operate on one hemisphere at a time.

---

## 7. Scientific References & PubMed Citations

- **Hierarchical Spherical Deformation (HSD)**:
  Lyu, I., Kang, H., Woodward, N. D., Styner, M. A., & Landman, B. A. (2019). *Hierarchical spherical deformation for cortical surface registration*. **Medical Image Analysis**, 57, 72–88.
  [PubMed: 31280090](https://pubmed.ncbi.nlm.nih.gov/31280090/) | [DOI: 10.1016/j.media.2019.06.013](https://doi.org/10.1016/j.media.2019.06.013)

- **TRACE Sulcal & Gyral Curve Extraction**:
  Lyu, I., Kim, S. H., Woodward, N. D., Styner, M. A., & Landman, B. A. (2018). *TRACE: A Method for Robust and Consistent Sulcal and Gyral Curve Extraction on Cortical Surfaces*. **IEEE Transactions on Medical Imaging**, 37(11), 2488–2498.
  [PubMed: 29969416](https://pubmed.ncbi.nlm.nih.gov/29969416/) | [DOI: 10.1109/TMI.2018.2849886](https://doi.org/10.1109/TMI.2018.2849886)

- **Spherical Deep Parcellation**:
  Lyu, I., Kang, H., Woodward, N. D., & Landman, B. A. (2019). *Cortical surface parcellation using spherical convolutional neural networks*. **Medical Image Analysis**.
  [PubMed: 31803864](https://pubmed.ncbi.nlm.nih.gov/31803864/)

- **Local Gyrification Index (LGI) & Shape Complexity (SCI)**:
  Lyu, I., Kim, S. H., Girault, J. B., Mostapha, M., Park, S. C., Nir, T. M., Jahanshad, N., Thompson, P. M., Piven, J., Hazlett, H. C., & Styner, M. A. (2018). *A Cortical Shape-Adaptive Approach to Local Gyrification Index*. **Medical Image Analysis**, 48, 244–258.
  [PubMed: 29887663](https://pubmed.ncbi.nlm.nih.gov/29887663/) | [DOI: 10.1016/j.media.2018.06.009](https://doi.org/10.1016/j.media.2018.06.009)

- **Sulcal Depth**:
  Lyu, I., Kang, H., Woodward, N. D., & Landman, B. A. (2018). *Sulcal Depth-based Cortical Shape Analysis in Normal Healthy Control and Schizophrenia Groups*. **Proceedings of SPIE**, 10574, 105740U.
  [PubMed: 29725164](https://pubmed.ncbi.nlm.nih.gov/29725164/) | [DOI: 10.1117/12.2293375](https://doi.org/10.1117/12.2293375)

- **Consistent Cortical Reconstruction (MaCRUISE)**:
  Huo, Y., Plassard, C. A., Carass, A., Resnick, S. M., Pham, D. L., Prince, J. L., & Landman, B. A. (2016). *Consistent cortical reconstruction and multi-atlas brain segmentation*. **NeuroImage**, 138, 197–210.
  [PubMed: 27184203](https://pubmed.ncbi.nlm.nih.gov/27184203/) | [DOI: 10.1016/j.neuroimage.2016.05.030](https://doi.org/10.1016/j.neuroimage.2016.05.030)

- **3D Whole-Brain Multi-Atlas Segmentation (SLANT / BrainCOLOR)**:
  Huo, Y., Xu, Z., Xiong, Y., Aboud, K., Parvathaneni, P., Bao, S., Bermudez, C., Resnick, S. M., Cutting, L. E., & Landman, B. A. (2019). *3D whole brain segmentation using spatially localized atlas network tiles (SLANT)*. **NeuroImage**, 194, 105–119.
  [PubMed: 30849524](https://pubmed.ncbi.nlm.nih.gov/30849524/) | [DOI: 10.1016/j.neuroimage.2019.03.010](https://doi.org/10.1016/j.neuroimage.2019.03.010)
