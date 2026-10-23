# pyBiscuit Sub-components Reference Manual & CLI Examples

This document provides comprehensive documentation for each of the core sub-components that make up the **pyBiscuit** pipeline. Each section includes:
1. **Background & Purpose**: What binary, script, or algorithm it ports.
2. **Algorithm & Math**: How it is implemented clean-room without GPL or MATLAB dependencies.
3. **CLI Usage & Flags**: Complete command line syntax, flags, and options.
4. **Python API Usage**: Module paths, function signatures, and return values.
5. **Concrete Example Invocations**: Executable examples using the provided sample dataset (`open-examples/0050952`).

---

## Table of Contents
1. [Surface Preparation (`pybiscuit.surfprep`)](#1-surface-preparation-pybiscuitsurfprep)
2. [Curvature & Shape Properties (`pybiscuit.mesh.property`)](#2-curvature--shape-properties-pybiscuitmeshproperty)
3. [Mesh Deformation & Transformation (`pybiscuit.mesh.deform`)](#3-mesh-deformation--transformation-pybiscuitmeshdeform)
4. [Spherical Remeshing (`pybiscuit.sphere.remesh`)](#4-spherical-remeshing-pybiscuitsphereremesh)
5. [Harmonic Spherical Deformation Registration (`pybiscuit.hsd`)](#5-harmonic-spherical-deformation-registration-pybiscuithsd)
6. [Deep Spherical Parcellation & Graph Cut (`pybiscuit.parc`)](#6-deep-spherical-parcellation--graph-cut-pybiscuitparc)
7. [TRACE Sulcal & Gyral Curve Extraction (`pybiscuit.trace`)](#7-trace-sulcal--gyral-curve-extraction-pybiscuittrace)
8. [Cortical Morphometry Quantification (`pybiscuit.quant`)](#8-cortical-morphometry-quantification-pybiscuitquant)
   - [A. Outer Cerebral Hull (`pybiscuit outer-hull`)](#a-outer-cerebral-hull-pybiscuit-outer-hull)
   - [B. Laplace PDE Streamlines (`pybiscuit klaplace`)](#b-laplace-pde-streamlines-pybiscuit-klaplace)
   - [C. Local Gyrification Index (`pybiscuit lgi` & `lgi-pipeline`)](#c-local-gyrification-index-pybiscuit-lgi--lgi-pipeline)
   - [D. Shape Complexity Index (`pybiscuit sci`)](#d-shape-complexity-index-pybiscuit-sci)
   - [E. Geodesic Sulcal Depth (`pybiscuit sulcal-depth`)](#e-geodesic-sulcal-depth-pybiscuit-sulcal-depth)
9. [Summary Statistics & Registered Maps (`pybiscuit.stats`)](#9-summary-statistics--registered-maps-pybiscuitstats)
10. [Visualization, QA Reports & Cohort Parity (`pybiscuit.viz`)](#10-visualization-qa-reports--cohort-parity-pybiscuitviz)

---

## 1. Surface Preparation (`pybiscuit.surfprep`)

### Background & Purpose
Replaces `run_postproc_macruise_pipeline.m`, `hemi_separation.m`, and FreeSurfer `mris_*` binary dependencies. It processes conformed volumes and raw continuous cortical surfaces into clean, genus-0, topologically correct left and right hemisphere meshes.

### Sub-commands & Algorithms
- **`hemi-split`**: Cuts a closed bilateral brain mesh along the interhemispheric fissure, closes the medial cut boundary using 2D Delaunay disk patching with harmonic interior coordinate filling, and smooths the seam.
- **`mesh-contour`**: Evaluates 101 candidate planes oriented perpendicular to the sagittal axis ($x$) and selects the optimal fissure boundary according to the minimal cut-contour length rule.
- **Volume Synthesis**: Synthesizes FreeSurfer-compatible `aseg.mgz`, `wm.mgz`, `filled.mgz`, and `brain.finalsurfs.mgz` from BrainCOLOR segmentations without ID chaining.
- **Surface Refinement & Inflation**:
  - `refine_white`: Laplacian vertex smoothing of white surface under segmentation guidance.
  - `deform_pial`: Level-set ray marching along white surface vertex normals to locate the outer cortical boundary.
  - `compute_mid_surface`: Exact coordinate average $(v_{\text{white}} + v_{\text{pial}}) / 2$ preserving white surface triangulation.
  - `inflate_surface`: Produces `smoothwm` (3-pass smoothing), `inflated` (outward metric expansion), and signed sulcal depth (`sulc`).
  - `map_to_sphere`: Distortion-minimizing spherical conformal relaxation onto a 100 mm radius sphere.

### CLI Examples
```bash
# Hemisphere separation on a bilateral surface
pybiscuit hemi-split bilateral_mesh.vtk -o /tmp/split_output --plane-x 0.0

# Synthesize FreeSurfer auxiliary volumes from conformed segmentation
python -c '
from pathlib import Path
from pybiscuit.surfprep.volume import prepare_auxiliary_volumes
prepare_auxiliary_volumes(t1="T1.nii.gz", seg="SEG_refine.nii.gz", out_dir="/tmp/fs_vols")
'
```

---

## 2. Curvature & Shape Properties (`pybiscuit.mesh.property`)

### Background & Purpose
Pure Python port of `MeshProperty` (Ilwoo Lyu). Estimates the local second fundamental form and curvature tensor across triangular surface meshes.

### Computes
- **Mean Curvature ($H$)**: Average of principal curvatures $\frac{k_1 + k_2}{2}$.
- **Gaussian Curvature ($K$)**: Product $k_1 \cdot k_2$.
- **Principal Curvatures ($k_1, k_2$)**: Maximum and minimum normal curvatures.
- **Shape Index ($S$)**: Local shape morphology index in $[-1, +1]$:
  $$S = \frac{2}{\pi} \arctan \frac{k_2 + k_1}{k_2 - k_1}, \quad k_1 \ge k_2$$
- **Curvedness ($C$)**: Measure of curvature intensity $\sqrt{\frac{k_1^2 + k_2^2}{2}}$.

### CLI Syntax & Flags
```bash
pybiscuit mesh-property <input.vtk> [options]
```

| Flag | Description |
|---|---|
| `-m`, `--mean` | Compute mean curvature ($H$) and output `<prefix>.H.txt` |
| `-s`, `--shape-index` | Compute Shape Index ($S$) and output `<prefix>.S.txt` |
| `-k`, `--gaussian` | Compute Gaussian curvature ($K$) and output `<prefix>.K.txt` |
| `-v`, `--area` | Compute barycentric vertex area and output `<prefix>.VertArea.txt` |
| `--surfsmoothing N` | Vertex smoothing iterations before curvature estimation (default: 0) |
| `--tensorsmoothing N` | Curvature tensor averaging iterations over 1-ring neighbors |
| `--datasmoothing N` | Scalar smoothing iterations of final output values |
| `--prctile P` | Percentile clamping limit (e.g. `0.1` for 0.1% to 99.9%) |
| `--normbefore` / `--normafter` | Z-score normalize scalar values before or after smoothing |
| `--invert` | Invert scalar signs (multiply by $-1$) |
| `-q`, `--quiet` | Suppress console logging |

### Example on Sample Data
```bash
# Compute smoothed mean curvature and shape index on sample mid-surface:
pybiscuit mesh-property \
  open-examples/0050952/surf/lh.mid.vtk \
  -m -s -v \
  --surfsmoothing 2 --tensorsmoothing 2
```

---

## 3. Mesh Deformation & Transformation (`pybiscuit.mesh.deform`)

### Background & Purpose
Port of `MeshDeform` (Ilwoo Lyu). Applies geometric transformations, coordinate offsets, and smoothing to triangular meshes.

### CLI Syntax & Flags
```bash
pybiscuit mesh-deform -i <input.vtk> -o <output.vtk> [options]
```

| Flag | Description |
|---|---|
| `-t x y z` | Translate vertex coordinates (subtracts `(x, y, z)`) |
| `-m <matrix.txt>` | Apply $4 \times 4$ affine transformation matrix |
| `--smooth N` | Apply $N$ iterations of uniform Laplacian mesh smoothing |
| `--scale S` | Uniformly scale vertex coordinates by factor $S$ |

### Example on Sample Data
```bash
# Smooth the white surface by 3 iterations (FreeSurfer smoothwm equivalent):
pybiscuit mesh-deform \
  -i open-examples/0050952/surf/lh.white.vtk \
  -o /tmp/lh.smoothwm.vtk \
  --smooth 3
```

---

## 4. Spherical Remeshing (`pybiscuit.sphere.remesh`)

### Background & Purpose
Port of `SphericalRemesh` (Ilwoo Lyu). Resamples 3D surface coordinates and scalar properties from a subject's spherical mesh onto a standard target sphere (such as `sphere_327680.vtk` or icosahedron) guided by spherical harmonic deformation coefficients.

### CLI Syntax & Flags
```bash
pybiscuit spherical-remesh -p <input_sphere.vtk> -s <target_mesh.vtk> -r <coeff.txt> -o <out_mesh.vtk> [options]
```

| Flag | Description |
|---|---|
| `-p` | Source spherical parametrization of subject mesh |
| `-s` | Subject 3D surface to resample (`white`, `gray`, `mid`) |
| `-r` | Spherical harmonic registration coefficient text file |
| `--ref` | Target uniform sphere (default: `sphere_327680.vtk`) |
| `--deg N` | Spherical harmonics degree expansion (default: 10) |
| `-d`, `--outputProperty` | Resample per-vertex scalar file instead of surface geometry |

### Example on Sample Data
```bash
# Resample subject mid-surface onto template sphere using image registration coefficients:
pybiscuit spherical-remesh \
  -p open-examples/0050952/surf/lh.sphere.vtk \
  -s open-examples/0050952/surf/lh.mid.vtk \
  -r open-examples/0050952/reg/lh.coeff.txt \
  -o /tmp/lh.mid.reg.vtk \
  --ref reference/sif_FC_v2.2/root/opt/biscuit/sphere/sphere_327680.vtk \
  --deg 10
```

---

## 5. Harmonic Spherical Deformation Registration (`pybiscuit.hsd`)

### Background & Purpose
Port of `HSD` (Ilwoo Lyu). Performs non-rigid spherical cortical surface registration by expanding deformation fields in a real spherical harmonics basis, optimizing energy balances between feature similarity (mean curvature and sulcal depth) and landmark/distortion regularization.

### Two Levels of Execution
1. **`pybiscuit register`** (High-level): Executes the complete 4-stage multi-resolution HSD chain used by BISCUIT (handling initial rigid guess, multiresolution degree 2/4/8 schedules, and order-dependent refinement).
2. **`pybiscuit hsd`** (Low-level): Direct invocation of the single-stage HSD optimizer.

### CLI Syntax (`pybiscuit register`)
```bash
pybiscuit register -s <sphere.vtk> -t <template_dir> -o <out_dir> -h <lh|rh> [options]
```

| Option | Description |
|---|---|
| `-s` | Native input sphere mesh (`<h>.sphere.vtk`) |
| `-t` | Directory containing registration template (e.g. `reg/mb101`) |
| `-o` | Output directory for `<h>.coeff.txt` and `<h>.sphere.reg.vtk` |
| `-h` | Hemisphere (`lh` or `rh`) |
| `--route` | Registration route: `macruise` or `fs` (FreeSurfer) |
| `--nThreads N` | Parallel CPU threads for Numba/OpenMP kernels |

### Example on Sample Data
```bash
pybiscuit register \
  -s open-examples/0050952/surf/lh.sphere.vtk \
  -t reference/sif_FC_v2.2/root/opt/biscuit/reg/mb101 \
  -o /tmp/hsd_demo \
  -h lh \
  --route fs \
  --nThreads 4
```

---

## 6. Deep Spherical Parcellation & Graph Cut (`pybiscuit.parc`)

### Background & Purpose
Replaces `parc.m`, `gcut.m`, and `roi_correction.m`. Takes multi-scale cortical features ($H, iH, sulc$), projects them onto an icosahedron (level 5), runs a spherical convolutional neural network (Spherical U-Net) to predict regional class logits, and refines label boundaries using an $\alpha$-expansion graph-cut Potts model with connected-component topological correction.

### Pipeline Stages
1. **Feature Extraction**: $Z$-score normalization and soft-clamping of curvature features.
2. **U-Net Inference**: Evaluates trained PyTorch weights across 5 cross-validation folds.
3. **Probability Fusion**: Softmax probability averaging on native spherical coordinates.
4. **Graph Cut**: Licence-clean Boykov-Veksler-Zabih (BVZ 2001) $\alpha$-expansion using Numba-compiled Dinic max-flow.
5. **ROI Island Cleanup**: Eliminates disjoint spurious fragments and enforces contiguous cortical regions.

### CLI Syntax
```bash
python -m pybiscuit.parc.cli \
  --sphere <sphere.reg.vtk> \
  --white <white.vtk> \
  --features <curv.txt> <sulc.txt> <iH.txt> \
  --protocol <braincolor|dkt31> \
  --hemi <lh|rh> \
  -o <out_dir>
```

### Python API Example
```python
from pathlib import Path
import numpy as np
from pybiscuit.parc.pipeline import parcellate

labels = parcellate(
    sphere_reg=Path("open-examples/0050952/reg/lh.sphere.reg.vtk"),
    white_mesh=Path("open-examples/0050952/surf/lh.white.vtk"),
    features=[
        np.loadtxt("open-examples/0050952/quant/lh.mid.H.txt"),
        np.loadtxt("open-examples/0050952/surf/lh.sulc"),
        np.loadtxt("open-examples/0050952/surf/lh.inflated.H"),
    ],
    protocol="dkt31",
    hemi="lh",
)
print("Parcellated vertices:", len(labels))
```

---

## 7. TRACE Sulcal & Gyral Curve Extraction (`pybiscuit.trace`)

### Background & Purpose
Port of `CurveExtraction` (TRACE; Ilwoo Lyu). Extracts major sulcal fundic tracks (on pial surfaces) and gyral crown ridges (on white surfaces) by identifying geodesic height extrema and connecting them via Dijkstra graph shortest paths.

### CLI Syntax & Flags
```bash
pybiscuit trace -i <surface.vtk> -o <output_prefix> [options]
```

| Flag | Description |
|---|---|
| `-i` | Input surface mesh (typically pial for sulci, white for gyri) |
| `-o` | Output file prefix |
| `--sulcus` | Extract sulcal fundic curves |
| `--gyrus` | Extract gyral crown curves |
| `--nhDist D` | Neighborhood distance radius in mm (default: 20.0) |
| `--lsTh T` | Level set threshold for point filtering (default: 0.5) |
| `--epRad R` | End point search radius in mm |
| `--bary` | Emit barycentric coordinate representations (`.bary`) |

### Example on Sample Data
```bash
# 1. Sulcal fundi on pial surface
pybiscuit trace \
  -i open-examples/0050952/surf/lh.gray.vtk \
  -o /tmp/lh.gray \
  --sulcus --nhDist 20 --lsTh 0.5

# 2. Gyral crowns on white surface
pybiscuit trace \
  -i open-examples/0050952/surf/lh.white.vtk \
  -o /tmp/lh.white \
  --gyrus --nhDist 20 --lsTh 0.5
```
*Outputs:* `<prefix>.scurve`, `<prefix>.scurve.bary`, `<prefix>.gcurve`, `<prefix>.gcurve.bary`.

---

## 8. Cortical Morphometry Quantification (`pybiscuit.quant`)

### A. Outer Cerebral Hull (`pybiscuit outer-hull`)
*Port of `OuterHull`*. Generates a smooth, enveloping outer cerebral hull mesh around the pial surface by voxelizing the mesh, applying morphological 3D closing and dilation, extracting an isosurface, and registering it via iterative closest points (ICP).
```bash
pybiscuit outer-hull \
  open-examples/0050952/surf/lh.gray.vtk \
  -o /tmp/lh.hull.vtk
```

### B. Laplace PDE Streamlines (`pybiscuit klaplace`)
*Port of `klaplace`*. Solves the 3D Laplace Dirichlet boundary value problem ($\nabla^2 \phi = 0, \left.\phi\right|_{\text{pial}} = 0, \left.\phi\right|_{\text{hull}} = 1$) on a 3D grid and traces Runge-Kutta 4th-order (RK4) gradient streamlines from every pial vertex to the outer hull.
```bash
pybiscuit klaplace \
  open-examples/0050952/surf/lh.gray.vtk \
  /tmp/lh.hull.vtk \
  -surfaceCorrespondence /tmp/lh.corr \
  -thread 4
```
*Output:* `/tmp/lh.corr_warpedMesh.vtp` (mesh with pial topology projected smoothly onto the hull).

### C. Local Gyrification Index (`pybiscuit lgi` & `lgi-pipeline`)
*Port of `Gyrification` and `lgi` script*. Computes the ratio between the local pial surface area and the corresponding outer hull area within overlapping geodesic circular discs.
```bash
# End-to-end chaining of hull, correspondence, curves, and gyrification:
pybiscuit lgi-pipeline \
  open-examples/0050952/surf/lh.gray.vtk \
  open-examples/0050952/curve/lh.gray.scurve.bary \
  open-examples/0050952/curve/lh.white.gcurve.bary \
  /tmp/lgi_output \
  --hemi lh \
  --nThreads 4
```
*Output:* `/tmp/lgi_output/lh.gray.LGI.txt`.

### D. Shape Complexity Index (`pybiscuit sci`)
*Port of `ShapeComplexityIndex`*. Quantifies multi-scale structural convolution on the mid-cortical surface using Numba-accelerated fast marching geodesic patches and 1D Earth Mover's Distance (EMD) transportation simplex cost.
```bash
pybiscuit sci \
  -i open-examples/0050952/surf/lh.mid.vtk \
  -o /tmp/lh.mid.SCI.txt \
  -k 3.0 \
  -s 5
```

### E. Geodesic Sulcal Depth (`pybiscuit sulcal-depth`)
*Port of `sulcal_depth.m`*. Voxelizes the pial surface, computes a quasi-Euclidean distance transform from the enclosing morphological hull boundary, and interpolates depths back onto the surface vertices.
```bash
pybiscuit sulcal-depth \
  open-examples/0050952/surf/lh.gray.vtk \
  -o /tmp/lh.gray.SD.txt
```

---

## 9. Summary Statistics & Registered Maps (`pybiscuit.stats`)

### Background & Purpose
Replaces `surf_quant_stat_pipeline.m`. Aggregates all six cortical measures ($H, S, \text{Depth}, \text{Thickness}, \text{SCI}, \text{LGI}$) within each parcellation region, weighting vertex values by their barycentric Voronoi area.

### Features
- Computes mean and standard deviation for cortical regions and curve vertices.
- Aggregates whole cortex, hemisphere-level, and lobar metrics.
- Resamples native subject scalar maps through registered spheres onto the standard `sphere_327680` template.
- Writes identical CSV tables (`stats.dkt31.csv`, `stats.braincolor.csv`) and template maps (`stat/*.reg.*`).

### CLI Syntax
```bash
python -m pybiscuit.stats.cli <subject_or_run_dir> [options]
```

### Example on Sample Data
```bash
python -m pybiscuit.stats.cli \
  open-examples/0050952 \
  --protocols braincolor dkt31 \
  --stat-dir /tmp/stats_demo \
  --quiet
```
*Outputs:* `/tmp/stats_demo/stats.braincolor.csv`, `/tmp/stats_demo/stats.dkt31.csv`, and template-space maps `stat/*.reg.*`.

---

## 10. Visualization, QA Reports & Cohort Parity (`pybiscuit.viz`)

### A. QA Multi-Page PDF Generator (`pybiscuit report`)
Creates a merged `PDF/biscuit.pdf` with exact page heights:
- Registration page (`reg.<template>.pdf`, 8.5 × 10.5 in).
- Parcellation pages (`parc.<protocol>.pdf`, 8.5 × 8.7 in).
- Curve extraction page (`curve.pdf`, 8.5 × 9.2 in).
- Morphometric quantification page (`quant.pdf`, 8.5 × 11.7 in).
```bash
pybiscuit report \
  -o /tmp/report_demo \
  --subject 0050952 \
  --protocols braincolor dkt31
```

### B. Cohort Mosaic & Parity Analysis (`pybiscuit cohort`)
Aggregates an entire cohort of runs, evaluates cross-subject intra-class correlation ($\text{ICC}$), and generates interactive HTML and JSON reports:
```bash
pybiscuit cohort \
  open-examples/0050952 \
  open-examples/0050961 \
  open-examples/0050971 \
  -o /tmp/cohort_analysis \
  --protocols braincolor dkt31

# Open the HTML report:
open /tmp/cohort_analysis/cohort_report.html
```

### C. Offline Interactive WebGL Viewer (`qa/index.html`)
Builds a browser-based 3D cortical viewer utilizing bundled NiiVue with zero external web server requirements:
```python
from pathlib import Path
from pybiscuit.viz.html import collect_inputs, write_viewer

# Collect all gifti surface and metric files
gii_files = list(Path("open-examples/0050952/qa").glob("*.gii"))
vi = collect_inputs(gii_files, title="Subject 0050952 QA")

# Write self-contained base64 viewer
write_viewer("/tmp/viewer_demo.html", vi, embed=True)
```
Open `/tmp/viewer_demo.html` in any browser to rotate, zoom, and inspect meshes, overlays, and labels in real-time.
