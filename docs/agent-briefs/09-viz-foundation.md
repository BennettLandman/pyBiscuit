# Agent brief 09 — Visualization foundation (B12.1, B12.2, B12.4)

**Written 2026-10-20.** Independent of briefs 04–08. Read
[AGENTS.md](../../AGENTS.md), [visualization-plan.md](../visualization-plan.md),
and the [parallel rules](README.md#parallel-sessions).

## Context

Visualization reads finished products only and must never be needed by the
pipeline. Real pipeline outputs don't exist yet, so build against products
that do: template surfaces and features
(`reference/sif_FC_v2.2/root/opt/biscuit/reg/*/?h.{white,inflated}.vtk`,
`?h.{curv,sulc,iH}.mean.txt`), shipped spheres, parcellation LUTs
(`parc/*/label.csv`, `lut.csv`), PyMaCRUISE surfaces, and fixtures.

## Scope

| Row | Deliverable |
|---|---|
| B12.1 | Core (no extra): `pybiscuit.io.gifti` extensions to write `*.surf.gii` with anatomical metadata (structure, `AnatomicalStructurePrimary` CortexLeft/CortexRight, geometric type), `*.func.gii` for measures, and `*.label.gii` with the LUT embedded as the label table (from `label.csv` + `lut.csv`; HSV fallback when `lut.csv` is absent, as `check_input` warns). TSV + JSON sidecar writer for stats. |
| B12.2 | `[viz]` extra (`pyvista`, `matplotlib`): off-screen renderer (`pybiscuit.viz.render`) with camera presets (lateral, medial, dorsal, ventral, anterior, posterior; hemisphere-aware), colormap conventions from visualization-plan.md (fixed limits per measure, matching the original PDF), label rendering with the LUT, and curve overlay as tubes (input: polylines, e.g. the `.bary`-derived curves later). Panel composer: N surfaces × M views → PNG. |
| B12.4 | `qa/index.html`: a self-contained offline viewer embedding **NiiVue** (BSD-2), pinned version, bundled as package data in `[viz]` (no CDN at view time), loading GIfTI surfaces + overlays + labels and optionally a NIfTI volume. Record the NiiVue version, licence, and SHA-256. |
| — | `pybiscuit view <files…>` minimal desktop PyVista window (surface + scalar/label selector), and `pybiscuit render <files…> --out dir` CLI. |

## Acceptance

- GIfTI outputs load back with NiBabel with correct metadata, and open in
  NiiVue (automated: headless-browser check if feasible; otherwise a
  documented manual check with a screenshot in the report). The label table
  matches `lut.csv` exactly.
- Snapshot tests at low resolution on fixtures with a perceptual tolerance;
  skipped when no OpenGL/OSMesa context is available. CI: install `[viz]` in one
  job, using OSMesa or `xvfb` on Linux, and record which works.
- Example gallery in the report: template white/inflated with `curv`/`sulc`
  means, a sphere, and a PyMaCRUISE inner surface, rendered by the new code.
- Core install and the non-viz CI job are unchanged (no viz imports outside
  `pybiscuit.viz`).

## Out of scope

The full PDF report (B12.3), which needs real pipeline outputs, and cohort/parity
views (B12.6).
