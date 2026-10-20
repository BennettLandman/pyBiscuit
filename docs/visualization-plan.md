# Visualization plan

Visualization is a first-class deliverable (unlike PyMaCRUISE, which excludes
rendering). It lives in the optional `pybiscuit[viz]` extra so the core
install stays light and headless. Everything here is MIT/BSD-licensed.

## 1. Standard exports (core, no extra needed)

Downstream tools are the cheapest viewer. Every run writes:

- **GIfTI** via NiBabel: `*.surf.gii` (white, pial, mid, inflated, sphere,
  sphere.reg), `*.func.gii` (all measures), `*.label.gii` (with the protocol
  LUT embedded as the label table). Opens in Connectome Workbench, FreeView,
  NiiVue, Surf Ice, MRIcroGL — no FreeSurfer license needed.
- **VTK legacy** (`legacy/`) with measures as point data, as the original did.
- **TSV** stats with a JSON sidecar describing columns (BIDS-style).

## 2. Static QA report — `pybiscuit report`

Reproduce the original five-page `biscuit.pdf` content, rendered
off-screen with PyVista (VTK; OSMesa/EGL wheels on headless Linux, native on
macOS) and laid out with matplotlib (`PdfPages`), replacing MATLAB + LaTeX +
Xvfb + Ghostscript.

| Page | Original content | pyBiscuit figure |
|---|---|---|
| 1 HSD registration | per hemisphere: spherical mapping → spherical deformation → template; energy before → after | sphere coloured by `curv`, before/after HSD, template mean; energy trace plot |
| 2–3 Parcellation (one per protocol) | labels on white, gray, middle, inflated surfaces | 4 surfaces × lateral/medial × lh/rh, LUT colours, legend |
| 4 TRACE | sulcal and gyral curves on original and inflated surfaces | curves as tubes over curvature-shaded surfaces |
| 5 Quantification | mean curvature (±0.05), shape index (±1), thickness (1–5 mm), sulcal depth (0–30 mm), LGI (2–10), SCI (0–1) | same measures, same fixed colour limits, lateral/medial views |
| 6 (new) Stats | — | ROI table summary, outlier flags vs cohort norms if supplied |
| 7 (new) Provenance | — | inputs, hashes, versions, config, run time, warnings |

Fixed camera presets (lateral, medial, dorsal, ventral, anterior, posterior)
make figures comparable across subjects. Colour maps: perceptually uniform
(`viridis`/`cividis`) for magnitudes, diverging (`RdBu_r`) for signed curvature
and SI, protocol LUTs for labels. Also emit each panel as PNG for DAX/XNAT QA.

## 3. Interactive viewing

- **`pybiscuit view <run>`** — desktop PyVista window: hemisphere/surface
  selector (white/pial/mid/inflated/sphere), measure dropdown with colour
  limits, label overlay with hover read-out (ROI name, value), curve toggle.
- **Self-contained HTML (`qa/index.html`)** — embeds NiiVue (BSD-2, WebGL)
  loading the run's GIfTI files: works offline in any browser, shareable on
  XNAT/web servers, no Python needed to view. Includes the T1 + SEG_refine as
  volume context with surface overlay.
- **Jupyter** — `pybiscuit.viz.show(run, measure="lgi")` using PyVista's
  trame backend.

## 4. Cohort and parity views

- **Cohort mosaic** (`pybiscuit report --cohort runs/*`): thumbnails per
  subject + per-measure ROI distributions to spot failures quickly.
- **Parity view** for validation: reference vs pyBiscuit side-by-side and
  vertex-wise difference maps, Bland–Altman plots for ROI measures, Dice
  per ROI as a bar chart. Used in `docs/reports/` evidence.
- **Template-space group maps**: measures resampled to the template sphere
  (via `sphere.reg`) and displayed on the template inflated surface — the
  basis for downstream group statistics.

## 5. Implementation notes

- Rendering code never runs in the core pipeline path; `report` and `view`
  read finished outputs only. The pipeline must succeed with no display.
- Snapshot tests: render fixed synthetic meshes at low resolution and compare
  images with a perceptual tolerance; skip when no GL context is available.
- Decide later whether the PDF should match the original's exact page
  geometry (useful for DAX processors that parse it) — see open questions.
