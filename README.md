# pyBiscuit

Python-native port of **BISCUIT** (Brain Shape Computing Toolbox; Ilwoo Lyu) and
the CMorph tools it wraps, packaged as one `pip`-installable library and CLI.
pyBiscuit takes the cortical surfaces and labels produced by
[PyMaCRUISE](https://github.com/BennettLandman/pymacruise) and produces
spherical registration, cortical parcellation, sulcal/gyral curves (TRACE), and
per-vertex, per-ROI, and per-curve shape measures (curvature, shape index,
thickness, sulcal depth, local gyrification index, shape complexity index),
plus QA figures and an interactive viewer.

**Status (2026-10-20): planning.** No algorithm code exists yet. Start with
[AGENTS.md](AGENTS.md), then [plan.md](plan.md).

## Why a port

The production tool is a Singularity image (`biscuit_FC_v2.2.sif`, CBT v1.3,
2021) containing an encrypted MATLAB R2012a (MCR v7.17) orchestrator, x86-64
C++ binaries, FreeSurfer 6 binaries that require a FreeSurfer license,
PyTorch 1.4 models, TeX Live, and Xvfb. It cannot run on macOS/ARM, cannot be
inspected at the orchestration level, and is pinned to 2020-era dependencies.

## Target pipeline (MaCRUISE route)

```
PyMaCRUISE run (inner/outer surfaces, level sets, SEG_refine, T1)
  └─ pyBiscuit
      1. hemisphere split + closure          (replaces hemi_separation.m)
      2. white / pial / mid / inflated / sphere + sulc, curv, iH
                                             (replaces FreeSurfer mris_* tools)
      3. HSD spherical registration          (port of ilwoolyu/HSD)
      4. spherical U-Net parcellation        (port of BISCUIT parc + graph cut)
      5. TRACE sulcal/gyral curves           (port of ilwoolyu/CurveExtraction)
      6. measures: curvature, SI, thickness, sulcal depth, LGI, SCI
                                             (MeshProperty, LGI, klaplace, SCI)
      7. ROI-, curve-, and cortex-wise statistics
      8. QA report + viewer
```

## Repository map

| Path | Purpose |
|---|---|
| [AGENTS.md](AGENTS.md) | Onboarding and working rules for agents and contributors |
| [plan.md](plan.md) | Phased, step-by-step port and validation plan with acceptance table |
| [docs/source-map.md](docs/source-map.md) | Every original component → source availability → Python target |
| [docs/io-contract.md](docs/io-contract.md) | Inputs from PyMaCRUISE / legacy MaCRUISE, outputs, manifest |
| [docs/reference-artifacts.md](docs/reference-artifacts.md) | SIF images, extraction, how to harvest reference outputs |
| [docs/validation-tolerances.md](docs/validation-tolerances.md) | Proposed acceptance classes and metrics |
| [docs/visualization-plan.md](docs/visualization-plan.md) | Reports, viewers, export formats |
| [docs/licensing.md](docs/licensing.md) | Licenses of code, models, and templates |
| [docs/open-questions.md](docs/open-questions.md) | Owner decisions pending |

## Attribution

BISCUIT, HSD, TRACE/CurveExtraction, LocalGyrificationIndex, SphericalRemesh,
SphericalLabeling, and MeshLib are © Ilwoo Lyu, MIT License. See
[docs/licensing.md](docs/licensing.md) for all third-party components.
