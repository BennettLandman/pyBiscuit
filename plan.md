# pyBiscuit porting plan

## Status

**2026-10-20 — briefs 01–03 merged to `main` (`e6c223a`).** Done vs upstream oracles: B0, B2.1–B2.5, B4.2, B4.3, SH basis (B6.2); B4.1 partial (shipped spheres are 4-decimal quantised). Cross-platform policy PROPOSED in validation-tolerances.md; brief 03 lists 6 owner decisions in [its report](docs/reports/portable-numerics-2026-10-20.md). CI runs one Linux job per PR; the full OS matrix is a manual dispatch. Image tool versions equal upstream master. Next: briefs 04–07 in parallel ([index](docs/agent-briefs/README.md)); B1 needs Q1; B5/B7.1/B7.4–5/B9.5/B10 wait for the MATLAB source.

## Objective

A `pip install pybiscuit` package that, given a PyMaCRUISE run, reproduces
BISCUIT 2.2 (`biscuit_FC_v2.2.sif`, `--surf_recon macruise`) products —
registration, parcellation, TRACE curves, six shape measures, statistics —
within [validation tolerances](docs/validation-tolerances.md), with no MATLAB,
FreeSurfer, Singularity, or compiled build, and with first-class
visualization ([plan](docs/visualization-plan.md)).

Completion = every B-row below **Done** with evidence, the full-pipeline
parity report (B11) accepted by the owner, and a tagged release installable
from a clean environment on Linux x86-64, macOS arm64, and Windows.

## Key findings that shape the plan

1. **Orchestrator is opaque.** `/opt/biscuit/biscuit` is MATLAB R2012a (MCR
   v7.17) with 84 encrypted `.m` files. Structure is known from the manifest
   ([docs/reference/ctf-manifest.tsv](docs/reference/ctf-manifest.tsv));
   behavior must come from instrumented container runs (B1). Asking Ilwoo Lyu
   for the MATLAB source (Q7) would remove the largest uncertainty.
2. **Most compute is open source (MIT).** HSD, TRACE, Gyrification,
   SphericalRemesh, MeshLib, and the parcellation net (already Python) are
   available. OuterHull is MATLAB with source. klaplace and SCI have source
   but no license file.
3. **Three closed binaries**: MeshProperty, MeshDeform, MeshContour —
   black-box re-implementation against the image.
4. **FreeSurfer 6 binaries do the surface generation** on the MaCRUISE route
   (`mri_convert`, `mri_normalize`, `mris_make_surfaces`, `mris_inflate`,
   `mris_sphere`) and need a FreeSurfer license. These are clean-room
   *replacements* (R-class), the main scientific risk.
5. **PyMaCRUISE gaps vs BISCUIT inputs**: no MNI-space surface (BISCUIT uses
   `innerSurf_mni.asc`), no FreeView `.asc` export; whole-brain surfaces in RAS
   VTK. pyBiscuit handles hemisphere split itself (Q4).
6. **No BISCUIT reference outputs exist locally.** The legacy MaCRUISE v3
   outputs in `~/dev/mipav/sample-macruise/output` and `100249/outputs` are
   valid BISCUIT inputs, so references can be produced on Linux (Q1).

## Architecture

```
src/pybiscuit/
  config.py            biscuit.conf reader, defaults, CLI overrides
  io/                  vtk.py, freesurfer.py (surf/curv/asc), gifti.py, txt.py, pymacruise.py
  mesh/                Mesh model, topology, normals, smoothing, curvature/property (MeshProperty),
                       aabb_sphere, barycentric, geodesic (exact + Dijkstra), icosphere, deform
  surfprep/            hemisphere.py, pial.py, inflate.py, sphere.py, features.py (sulc/curv/iH)
  sphere/remesh.py     SphericalRemesh port (resampling on spheres)
  hsd/                 basis.py (SH), energy.py, optimize.py (LM + BOBYQA), register.py
  parc/                unet.py (MeshConv), infer.py, fusion.py, graphcut.py, cleanup.py
  trace/               sulcal_point.py, sulcal_curve.py, gyral_*.py, extract.py
  quant/               curvature.py, sulcal_depth.py, hull.py, klaplace.py, lgi.py, sci.py
  stats/               roi.py, curve.py, cortex.py
  pipeline.py          stage graph, checkpoints, resume, run.json
  assets.py            registry + SHA-256 + download/cache (pooch-style)
  viz/                 render.py, report.py, view.py, html.py   ([viz] extra)
  cli.py
```

Performance: Numba for vertex/face loops (HSD energy & Jacobian, fast
marching, Dijkstra helpers, AABB queries); `prange` replaces OpenMP. Torch only
for parcellation inference.

## Phases and dependency graph

```
B0 scaffold ─┬─ B1 reference harvest (Linux) ─────────────────────────────┐
             └─ B2 mesh core + I/O ─┬─ B3 MeshProperty                    │
                                    ├─ B4 SphericalRemesh + icospheres    │
                                    ├─ B5 surface prep (R-class) ─┐       │
                                    ├─ B6 HSD ───────────────────┤        │
                                    ├─ B8 TRACE ─────────────────┤        │
                                    └─ B9 quant (hull, klaplace,  │       │
                                       LGI, SD, SCI) ─────────────┤       │
                       B4 + B6 ──── B7 parcellation ─────────────┤        │
                                           B10 stats + pipeline ─┴─ B11 parity ─ B13 release
                                           B12 visualization (after B2; finishes after B10)
```

Parallelizable after B2: B3, B4, B5, B6, B8, B9 are independent workstreams
(separate files, separate goldens). B7 needs B4 (+ B6 for registered spheres).

## Next steps (2026-10-20)

```
brief 03 portable numerics ──► merge PR #1, PR #2 ──┬─ brief 04 geodesics + TRACE   (B2.6, B8)
                                                    ├─ brief 05 HSD                 (B6)
                                                    ├─ brief 06 parcellation net    (B7.2–B7.3)
                                                    └─ brief 07 OuterHull + LGI     (B9.2, B9.4)
owner track (in parallel): Q1 image runs → B1 goldens → re-validate all E-rows against the image
waiting on the MATLAB source: B5, B7.1, B7.4–B7.5, B9.5, B10 → then brief 08 (orchestrator spec + MaCRUISE surface prep)
later: B12 visualization (GIfTI exports + renderer can start once B10 output layout is fixed); B9.3 klaplace and B9.6 SCI after licences (Q7)
```

Owner actions that unblock the most:
1. **Merge PR #1 now**; merge PR #2 after brief 03 turns CI green.
2. **Q1:** pick ACCRE or a GitHub Actions x86-64 job (see open-questions). Image goldens turn every "provisional (oracle)" row into an accepted one, and they unblock B3 (MeshProperty).
3. **Ask Lyu (Q7):** licences for klaplace and ShapeComplexityIndex (NIRAL), sources of MeshProperty/MeshContour/MeshDeform, and permission to redistribute the models and templates.
4. **Decide** B4.1 tolerance (accept one quantisation step), the cross-platform policy (brief 03), and Q12 tolerances.

When the MATLAB source arrives, the next brief is the orchestrator spec (B1.3 from source, not traces), then B5 surface prep, B7.1/B7.4/B7.5, B9.5, B10.

## Step-by-step acceptance table

Status: **Todo / In progress / Blocked (reason) / Done (evidence)**. Class
per [validation-tolerances](docs/validation-tolerances.md).

### B0 — Scaffold

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B0.1 | `pyproject.toml` (setuptools, src layout, py3.10–3.13), extras `[parc]`, `[viz]`, `[dev]`, `[macruise]`; MIT LICENSE + `LICENSES/` upstream notices | `pip install -e .[dev]` and `pybiscuit --version` on macOS arm64 + Linux | Done (pyproject, `pybiscuit --version`/`info`; tests/test_cli.py; macOS arm64 only — Linux unchecked) |
| B0.2 | CI (GitHub Actions): lint (ruff), pytest matrix (3.10, 3.13 × ubuntu, macos, windows) | green on empty test suite | Done (CI green on PR #1: 6/6 jobs, run 39456903772) |
| B0.3 | `scripts/extract_reference.py` maintained; `assets.py` registry with SHA-256 of every template/model file | registry matches extracted `reference/` hashes | Done (76 files/1.26 GB in `data/asset_registry.json`; tests/test_assets.py incl. reference match) |
| B0.4 | Synthetic mesh fixture generators (sphere, ellipsoid, analytic "folded" sphere r(θ,φ)=1+a·Y_lm, icospheres 0–7) with seeds | fixtures deterministic across platforms (hash test) | Done (tests/fixtures/meshes.py; tests/test_fixtures.py hash tests; arm64 only) |

### B1 — Reference harvesting (Linux x86-64; needs Q1)

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B1.1 | `scripts/reference/run_full.sh`: run the image on legacy sample (`sample-macruise/output`) with intermediates kept (see [reference-artifacts](docs/reference-artifacts.md)); `--reg oasis45` and default `mb101` | complete output tree + `.tmp` archived with hashes | Todo |
| B1.2 | Command trace: `execve` log (strace or PATH shims) of every sub-tool call with argv, cwd, inputs/outputs, timing | `docs/reports/orchestrator-trace-*.md` listing the full call graph | Todo |
| B1.3 | Write `docs/reference/biscuit-output-tree.md` (every output file: name, format, meaning) and `docs/reference/orchestrator-spec.md` (stage order, parameters, file flow — the de-facto spec for biscuit.m) | owner review | Todo |
| B1.4 | Per-tool goldens on synthetic fixtures + tier-1 intermediates for HSD, CurveExtraction, Gyrification, klaplace, OuterHull, MeshProperty (all flag combos used in B1.2), ShapeComplexityIndex, SphericalRemesh, MeshDeform, parc `test.sh` | `tests/data/golden/<tool>/` with `command.json` | In progress (O1 oracles built, provisional: MeshLib, SphericalRemesh, HSD, CurveExtraction, Gyrification, upstream-macos-arm64; goldens in tests/data/golden/{meshlib,spherical-remesh}; image goldens still needed — docs/ports/oracles.md) |
| B1.5 | FS-input route run on one FreeSurfer 6 subject (if Q5 yes) | output tree archived | Todo |
| B1.6 | Cohort reference set (Q2): ≥10 subjects with MaCRUISE v3 inputs + BISCUIT outputs; ≥2 scan–rescan pairs if available | manifest of paths + hashes (data stays on ACCRE) | Todo |

### B2 — Mesh core and I/O

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B2.1 | Readers/writers: VTK legacy ASCII+binary (POINTS/POLYGONS/POINT_DATA, float/double), FreeSurfer ASCII `.asc`, FS binary surf/curv (NiBabel), GIfTI, per-vertex `.txt`, PyMaCRUISE VTK header frame tag | round-trip tests; reads every VTK in `reference/` and template dirs | Done (tests/test_io_*.py; reference files load with correct counts; see report) |
| B2.2 | Coordinate frames: tkRAS↔scanner RAS from T1 header for `.asc`; manifest records | legacy `.asc` overlays T1 correctly (unit test with synthetic header) | Done (tests/test_frames.py synthetic; diagnostic: legacy .asc is already scanner RAS, converted fraction in mask 0.60 vs raw 1.00) |
| B2.3 | Topology: edges, one-ring, face adjacency, boundary loops, connected components, Euler/genus, orientation consistency | tests on fixtures | Done (tests/test_topology.py 27 pass; all spheres + template white surfaces chi=2 genus 0) |
| B2.4 | Geometry: face/vertex normals (MeshLib weighting), areas, Voronoi areas, Laplacian smoothing variants as in MeshLib `Util/SurfaceUtil.cpp` | match MeshLib (compile-free: compare against values derived from goldens) | Done, oracle-provisional (geometry.py, curvature.py, sphere_map.py; tests/test_geometry.py, test_curvature.py, test_sphere_map.py; bit-identical vs non-FMA MeshLib incl. reg/mb101/lh.white; [report](docs/reports/geometry-remesh-2026-10-20.md)) |
| B2.5 | Spherical AABB / point location + barycentric interpolation (MeshLib `AABB_Sphere`) | E-class vs SphericalRemesh goldens (via B4) | Done, oracle-provisional (sphere/aabb.py; tests/test_aabb_sphere.py 29 pass; 0 face mismatches, bary diff 0 on 1e5 queries x 5 cases; ico7 163,842 queries 0.02 s warm) |
| B2.6 | Geodesics: Dijkstra on edge graph (SciPy) + exact geodesic port of MeshLib `GeodesicA`/Peyré (Numba) | distances on analytic sphere within 0.1 %; E-class vs TRACE `.bary` later | Done, oracle-provisional (mesh/geodesic.py, mesh/slicer.py; tests/test_geodesic.py, test_slicer.py; fast marching + GeodesicPath bit-identical vs no-FMA MeshLib incl. reg/mb101/lh.white; unit-sphere mean error 0.9 % vs 0.5 % diagnostic target - intrinsic to the upstream scheme; [report](docs/reports/geodesics-trace-2026-10-20.md), [note](docs/ports/geodesic.md)) |

### B3 — MeshProperty (closed binary, black-box)

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B3.1 | Map every MeshProperty invocation from B1.2 (flags → outputs) | table in `docs/ports/meshproperty.md` | Todo |
| B3.2 | Implement curvature tensor estimate, mean/Gaussian/principal curvatures, shape index, curvedness; `--surfsmoothing`, `--tensorsmoothing`, `--datasmoothing`, `--prctile`, `--normbefore/after`, `--invert`, `-q` | E-class vs goldens for each flag combo used | Todo |
| B3.3 | MeshDeform (affine, translate, scale, sphere projection, smoothing, invertFace) and MeshContour (behavior from B1.2) | E-class | Todo |

### B4 — SphericalRemesh and icospheres

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B4.1 | Icosphere generator reproducing `sphere_{5120..327680}.vtk`, template ico7 (163,842 v), `icosphere_{0..6}` vertex order | byte-identical vertex order or documented permutation | Partial (faces + vertex order identical for ico4–7; vertices match only to one 4-decimal quantisation step, 1e-6 unreachable for float32 radius-100 files; UGSCNN icosphere_5/6 ≤5e-7; tests/test_icosphere.py 33 pass) |
| B4.2 | Port SphericalRemesh (`-p -r -s --outputProperty`, deformation option) | E-class vs goldens | Done, oracle-provisional (sphere/remesh.py, `pybiscuit spherical-remesh`; tests/test_remesh.py 97 pass; byte-identical vs non-FMA oracle except 24/84 SH-deformed files 1-ulp last digit) |
| B4.3 | Regenerate MeshConv operator `.pkl` files from icospheres; confirm equality with shipped pkl | max abs diff ≤ 1e-6 | Done (tests/test_operators.py 27 pass; all 13 fields, levels 0–5, max diff 2.5e-14; safe unpickler) |

### B5 — Surface preparation (MaCRUISE route; R-class; Q3, Q4)

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B5.1 | Reverse-engineer `run_postproc_macruise_pipeline.m` + `hemi_separation.m` from B1.2/B1.3: what is computed from innerSurf, MNI surf, T1, SEG_refine, and in which order | `docs/ports/macruise-surfprep.md` | Todo |
| B5.2 | Hemisphere separation + medial-wall closure (`circle_patch`, `connectedComp`, `surfNormalCorrection` equivalents) using SEG_refine L/R labels (Q4) | each hemisphere closed, genus 0, consistent orientation; vertex-assignment agreement with reference ≥ 99 % on legacy sample | Todo |
| B5.3 | Volume prep replacing `mri_convert`/`mri_normalize` (only what downstream needs) | documented; downstream checks in B5.4 | Todo |
| B5.4 | Pial surface with white–pial vertex correspondence: deform white along PyMaCRUISE outer level set (default) vs clean-room `mris_make_surfaces`-like model (if Q3 requires) | R-class geometry metrics vs reference pial | Todo |
| B5.5 | Mid surface; thickness (vertex-correspondence and symmetric closest-point variants; pick by reference agreement) | R-class thickness metrics | Todo |
| B5.6 | Inflation (metric-preserving smoothing) with FreeSurfer-equivalent `sulc` accumulation; `iH` = mean curvature of inflated surface; `curv` per MeshProperty settings | R-class: sulc/curv/iH r ≥ 0.95 vs reference after registration | Todo |
| B5.7 | Spherical mapping (inflate → project → distortion-minimising relaxation; no flipped triangles) | zero flipped faces; areal distortion report; HSD energy after registration within 5 % of reference | Todo |

### B6 — HSD spherical registration (E-class)

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B6.1 | Port note: map `HSD.cpp` functions, image binary version vs upstream `master` (`f169595`) | `docs/ports/hsd.md` | Done (docs/ports/hsd.md; image HSD CLI == upstream wrapper at `f169595`, landmark/location flags absent from both) |
| B6.2 | Real spherical harmonics basis as implemented (CBT v1.3 "fixed SH bases") | E vs analytic + golden coefficients | Done, oracle-provisional (SH basis sphere/spharm.py as used by HSD; HSD integration validated in tests/test_hsd.py) |
| B6.3 | Feature resampling on icosahedron (`--icosahedron`, `--icomesh`, `--resample`), AABB cache | E | Done, oracle-provisional (hsd/icosahedron.py, kernels.py; construction state bit-identical vs non-FMA oracle; tests/test_hsd.py; [report](docs/reports/hsd-2026-10-20.md)) |
| B6.4 | Energy (feature similarity vs template mean/var + rigidity/distortion regularization), Jacobian/Hessian 2nd-order approx | per-iteration energy matches golden to 1e-6 relative on fixtures | Done for GD path (energy/gradient <= 9e-9); Newton path energy <= 1e-7, Hessian/direction <= 3.5e-6 (float32 ulp noise), so the 1e-8 unit target is not met there; oracle-provisional ([report](docs/reports/hsd-2026-10-20.md)) |
| B6.5 | Optimizer: Levenberg–Marquardt step + BOBYQA (port of `bobyqa.h`), initial guess ("improved initial guess" in v1.3), multi-resolution schedule used by BISCUIT (from B1.2) | E on fixtures; final energy within 0.1 % on legacy sample | Done for the shipped CLI (property-only), oracle-provisional: full runs match (final energy <= 9e-5 rel, median angle <= 0.005 deg); BOBYQA + landmark/location paths not ported (unreachable, see port note) |
| B6.6 | Numba-parallel performance; optional GPU backend deferred (Q10) | lh+rh within 2× of reference single-thread wall time | Done: prange kernels, fixed-order chunk reductions; ico7 degree 8: port 37 s (1 thread) vs oracle 81 s |

### B7 — Parcellation (E-class)

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B7.1 | Feature extraction (`extract_features.m`): which 3 channels, normalisation with `parc/<p>/?h.curv.mean.txt`, resampling to icosphere | match `.tmp` feature files from B1.1 | Todo |
| B7.2 | Spherical U-Net in modern PyTorch; load the original `.pth` (torch 1.4 format) and convert to a safetensors asset | per-class output E vs `test.sh` goldens | Done, oracle-provisional (parc/unet.py, checkpoint.py, infer.py; all 12 `.pth` converted via audited `weights_only=True` load; scores bit-identical (max abs diff 0.0, argmax 100 %) to the image's original `test.py` code under torch 2.14.1, 24 cases; shipped-pickle vs regenerated operators 0.0; CPU 0.60 s/hemisphere (2.15 s 1 thread); tests/test_parc_{core,net,reference}.py; [port note](docs/ports/parcellation-net.md), [report](docs/reports/parcellation-net-2026-10-20.md)) |
| B7.3 | Probability resampling to native sphere (B4) + DKT31 5-model fusion | E | Partial (resampling done: `infer.resample_to_native` vs original text round trip + non-FMA SphericalRemesh, max abs diff 5e-7, argmax 100 %; outputs are raw scores, `test.py` has no softmax; DKT31 fusion is not in `test.py`: `parc/fusion.py` is an interface, rule pending B7.1) |
| B7.4 | Graph-cut label smoothing (`gcut.m`; alpha-expansion, Potts or as reconstructed) with a license-clean max-flow | labels E vs reference post-gcut | Todo |
| B7.5 | `roi_correction.m` (connected-component cleanup) | final labels E | Todo |

### B8 — TRACE curve extraction (E-class)

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B8.1 | Port note + parameter map (`trace_r`, `trace_delta`, smoothing, `--bary`, junctions) from B1.2 | `docs/ports/trace.md` | Done ([docs/ports/trace.md](docs/ports/trace.md): function map, flag table, trace_r/trace_delta hypothesis, precision notes) |
| B8.2 | SulcalPoint / GyralPoint (fundic/crown point selection, line simplification) | point sets E | Done, oracle-provisional (trace/points.py; spoint/gpoint identical on lh.white, folded fixtures differ by 1 tie flag in 20,484; tests/test_trace.py) |
| B8.3 | SulcalCurve / GyralCurve (graph construction, Dijkstra tracing, pruning) | curves E (Jaccard, Hausdorff) | Done, oracle-provisional (trace/curves.py; curve files byte-identical vs no-FMA CurveExtraction on lh.white and 2 folded fixtures; open question: GyralCurve uninitialised `current`) |
| B8.4 | Writers `.scurve/.gcurve/.bary` + VTK | byte-compatible text format | Done, oracle-provisional (trace/extract.py, `pybiscuit trace`; .scurve/.gcurve/.bary/.cart/.vtk byte-identical vs oracle) |

### B9 — Quantification

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B9.1 | Mean curvature + shape index as BISCUIT calls MeshProperty (`curv_smoothness`) | E | Todo |
| B9.2 | OuterHull port (VOXELISE, morphology, ICP_finite, isosurface) | E (hull geometry p99 ≤ 0.1 mm) | Todo |
| B9.3 | klaplace port (Laplace field on grid `-dims 128`, RK45 streamlines, surface correspondence) — license per Q7 | E | Todo |
| B9.4 | Gyrification: anisotropic H-J wavefront kernel (`lgi_kernel_size`, `lgi_anisotropy`, `lgi_ref_area`), hull-area ratio | E | Todo |
| B9.5 | Sulcal depth (`sulcal_depth.m`: geodesic path through volume to hull; gray surface per CBT v1.1 notes) | E vs reference | Todo |
| B9.6 | SCI (`sci_kernel_size`, `sci_bins`; transportation-simplex EMD) | E | Todo |

### B10 — Statistics and pipeline

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B10.1 | ROI-wise, curve-wise (`.bary`-based), whole-cortex stats (`surf_quant_stat_pipeline.m`) incl. CBT v1.2c/d fixes | tables E vs reference | Todo |
| B10.2 | Stage graph, checkpoints, `--resume` with signature verification, `run.json` | resume tests | Todo |
| B10.3 | Input adapters: PyMaCRUISE run dir (primary), legacy MaCRUISE v3 dir, original `biscuit.conf` | integration tests on synthetic run | Todo |
| B10.4 | `legacy/` output view mirroring B1.3 names | file-tree diff vs reference = ∅ (names) | Todo |

### B11 — End-to-end parity

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B11.1 | Legacy-input parity: same MaCRUISE v3 inputs → pyBiscuit vs image | all E rows pass on intermediates; R rows pass | Todo |
| B11.2 | Cohort parity (B1.6): ICC/Dice tables vs scan–rescan variability | `docs/reports/parity-cohort-*.md`, owner acceptance | Todo |
| B11.3 | PyMaCRUISE-input runs (BASE) on the same subjects; compare to B11.1 outputs | report differences attributable to PyMaCRUISE vs pyBiscuit | Todo |

### B12 — Visualization ([plan](docs/visualization-plan.md))

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B12.1 | GIfTI/TSV exports (core) | loads in NiiVue + Workbench (manual check recorded) | In progress (writers + LUT label tables + TSV/JSON done; NiBabel round trip and `lut.csv` exact-match tests pass; NiiVue manual check recorded, Workbench not checked; [report](docs/reports/viz-foundation-2026-10-21.md)) |
| B12.2 | Off-screen renderer, camera presets, colour conventions | snapshot tests | In progress (renderer, presets, fixed limits, panel composer, CLI done; snapshot tests pass on macOS arm64; Linux `viz` CI job result pending; [report](docs/reports/viz-foundation-2026-10-21.md)) |
| B12.3 | `pybiscuit report` PDF (pages 1–7) + PNG panels | owner visual review vs original PDF | Todo |
| B12.4 | `qa/index.html` with embedded NiiVue | opens offline in Chrome/Safari/Firefox | In progress (NiiVue 0.69.0 bundled, hash recorded; generator + tests; rendered in the in-app browser over HTTP, `file://` in Chrome/Safari/Firefox not yet checked; [report](docs/reports/viz-foundation-2026-10-21.md)) |
| B12.5 | `pybiscuit view` desktop + Jupyter | manual checklist | In progress (desktop `view` + `render` CLIs done by brief 09; Jupyter and manual checklist outstanding) |
| B12.6 | Cohort mosaic + parity views (used by B11 reports) | used in B11.2 report | Todo |

### B13 — Release

| ID | Step | Acceptance | Status |
|---|---|---|---|
| B13.1 | Asset hosting + `fetch-assets` (Q6, Q8) | clean-machine install + fetch + synthetic run | Todo |
| B13.2 | User manual (install, run, outputs, troubleshooting), CLI reference | beta tester completes run unaided | Todo |
| B13.3 | Version 0.1.0 tag; wheel; container recipe (optional) | install from tag on 3 OSes | Todo |

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| No Linux reference host (Q1) | no acceptance evidence at all | prioritise; meanwhile build against upstream source + synthetic tests |
| Encrypted orchestrator hides parameters | silent divergences | B1.2 command trace; ask author (Q7) |
| FS replacements shift sulc/curv distributions vs templates built with FreeSurfer | worse HSD/parcellation | R-class acceptance; calibrate feature normalisation on cohort; consider re-deriving template stats (owner decision) |
| Upstream source ≠ image binaries | E-class failures | goldens from image binaries are authoritative; diff versions in port notes |
| HSD runtime in Python | usability | Numba parallel, icosahedral resampling, profile early (B6.6) |
| License gaps (klaplace, SCI, GCO, models, HCP template) | cannot redistribute | Q7 outreach; independent re-implementations from papers |

## Progress recording

Update the Status column with evidence links; update the Status section at the
top with date and next step; add reports under `docs/reports/`.
