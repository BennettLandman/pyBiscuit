# Sample Data Walkthrough & Visualization Guide

This guide provides concrete, step-by-step instructions to run pyBiscuit, reproduce morphometry outputs, and interact with visualizations using the provided sample datasets located in `open-examples/`.

---

## 1. Provided Sample Datasets

The repository includes three real reference subjects from the ABIDE I (NYU) cohort, processed with FreeSurfer 7.4.1 and validated against real `biscuit_FC_v2.2.sif` outputs:

```
open-examples/
├── 0050952/   # Subject 1 (Full FreeSurfer inputs + reference BISCUIT outputs)
├── 0050961/   # Subject 2
└── 0050971/   # Subject 3
```

Each subject directory contains:
- `surf/`: FreeSurfer native surfaces (`lh.white`, `lh.pial`, `lh.sphere`, `lh.inflated`) and metric files (`lh.curv`, `lh.sulc`, `lh.thickness`).
- `reg/`: Reference HSD spherical registration coefficients and registered spheres (`lh.sphere.reg.vtk`).
- `curve/`: Sulcal and gyral TRACE curves (`lh.gray.scurve.bary`, `lh.white.gcurve.bary`).
- `quant/`: Reference morphometric measures (`lh.gray.LGI.txt`, `lh.mid.SCI.txt`, `lh.gray.SD.txt`, `lh.thickness.txt`, `lh.mid.H.txt`, `lh.mid.S.txt`).
- `roi/`: DKT31 and BrainCOLOR parcellations (`lh.parc.dkt31.txt`, `lh.parc.braincolor.txt`).
- `stat/`: Template-space registered maps and summary statistics (`stats.dkt31.csv`, `stats.braincolor.csv`).
- `pdf/`: Reference QA report (`biscuit.pdf`).

---

## 2. Step-by-Step Walkthrough

Ensure your virtual environment is active:
```bash
source /Users/landmaba/dev/biscuit/pyBiscuit/.venv/bin/activate
export PYTHONPATH=src
```

### Step 1: Inspect Sample Inputs with `--dry-run`
Verify that pyBiscuit detects the FreeSurfer input layout, finds all required surfaces and metrics, and plans the execution stages:

```bash
pybiscuit run /Users/landmaba/dev/biscuit/open-examples/0050952 /tmp/run_0050952 --dry-run
```

---

### Step 2: Run Individual Morphometric Analysis on Sample Meshes

You can execute pyBiscuit's standalone tools directly on sample surfaces:

#### A. Sulcal & Gyral Curve Extraction (TRACE)
Extract sulcal fundic curves from the pial surface and gyral crown curves from the white surface:
```bash
# Sulcal curves on pial mesh (lh.gray.vtk)
pybiscuit trace \
  -i /Users/landmaba/dev/biscuit/open-examples/0050952/surf/lh.gray.vtk \
  -o /tmp/lh.gray --sulcus --nhDist 20 --lsTh 0.5

# Gyral curves on white mesh (lh.white.vtk)
pybiscuit trace \
  -i /Users/landmaba/dev/biscuit/open-examples/0050952/surf/lh.white.vtk \
  -o /tmp/lh.white --gyrus --nhDist 20 --lsTh 0.5
```
*Outputs:* `/tmp/lh.gray.scurve`, `/tmp/lh.gray.scurve.bary`, `/tmp/lh.white.gcurve`, `/tmp/lh.white.gcurve.bary`.

#### B. Shape Complexity Index (SCI)
Calculate multi-scale Earth Mover's Distance shape complexity on the mid-cortical surface:
```bash
pybiscuit sci \
  -i /Users/landmaba/dev/biscuit/open-examples/0050952/surf/lh.mid.vtk \
  -o /tmp/lh.mid.SCI.txt \
  -k 3.0 -s 5
```

#### C. Sulcal Depth (SD)
Compute geodesic sulcal depth through the 3D morphological hull:
```bash
pybiscuit sulcal-depth \
  /Users/landmaba/dev/biscuit/open-examples/0050952/surf/lh.gray.vtk \
  -o /tmp/lh.gray.SD.txt
```

---

### Step 3: Interactive Visualizations

#### A. 3D WebGL Interactive Viewer (`qa/index.html`)
pyBiscuit provides a browser-based, zero-dependency 3D cortical viewer powered by bundled NiiVue:

```bash
# Open the self-contained offline viewer for subject 0050952:
open /Users/landmaba/dev/biscuit/open-examples/0050952/qa/index_embedded.html
```

**Viewer Controls:**
- **Rotate**: Left-click + drag.
- **Zoom**: Right-click + drag or scroll wheel.
- **Pan**: Middle-click + drag.
- **Camera Views**: Click any face on the 3D orientation cube to snap to Lateral, Medial, Dorsal, Ventral, Anterior, or Posterior views.
- **Layer Switching**:
  - *Surface dropdown*: `white`, `pial`, `mid`, `inflated`.
  - *Measure dropdown*: `thickness` (1–5 mm), `sulcal_depth` (0–30 mm), `lgi` (2–10), `sci` (0–1).
  - *Labels dropdown*: `dkt31` (Mindboggle-101) or `braincolor`.
  - *Opacity slider*: Adjust transparent blend between cortical surface and overlay.

#### B. QA PDF Report (`PDF/biscuit.pdf`)
Generate the multi-page QA PDF report matching the exact page geometries (Registration, Parcellation, Curves, and Quantification):
```bash
pybiscuit report \
  -o /tmp/report_demo \
  --subject 0050952 \
  --protocols braincolor dkt31

# Open the generated report:
open /tmp/report_demo/PDF/biscuit.pdf
```

---

### Step 4: Multi-Subject Cohort Mosaic & Parity Report

Aggregate statistics across all three sample subjects, evaluate inter-subject consistency (ICC), and check parity vs reference goldens:

```bash
pybiscuit cohort \
  /Users/landmaba/dev/biscuit/open-examples/0050952 \
  /Users/landmaba/dev/biscuit/open-examples/0050961 \
  /Users/landmaba/dev/biscuit/open-examples/0050971 \
  -o /tmp/cohort_report_demo \
  --protocols braincolor dkt31

# Open the interactive HTML cohort report:
open /tmp/cohort_report_demo/cohort_report.html
```

The report provides:
1. **Subject Inventory**: Validation of complete outputs for all subjects.
2. **Regional Morphometric Distributions**: Means, standard deviations, and ranges across 65 DKT31 and 111 BrainCOLOR ROIs.
3. **Outlier Flags**: Automated detection of anomalous ROI measurements based on robust $z$-scores.
4. **JSON Sidecar**: Machine-readable data exported to `/tmp/cohort_report_demo/cohort_summary.json`.

---

### Step 5: Verify Model & Template Assets

Verify that all 52 model checkpoints and templates are valid and match committed SHA-256 digests:
```bash
pybiscuit fetch-assets --verify-only
```
Or check the one-line summary:
```bash
pybiscuit info
```
