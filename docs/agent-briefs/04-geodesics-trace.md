# Agent brief 04 — Geodesics + TRACE curve extraction (B2.6, B8)

**Written 2026-10-20.** Runs in parallel with briefs 05–07 after brief 03
merges. Read [AGENTS.md](../../AGENTS.md), [plan.md](../../plan.md),
[parallel rules](README.md#parallel-sessions), and the brief 02 report.

## Context

- The image's CurveExtraction reports **TRACE v1.1.2**, the same string as
  upstream `ded932d`. Port from that commit.
- The oracle exists at `~/dev/biscuit/oracles/curve_extraction/` (built
  without `-ffp-contract=off`; rebuild it with that flag first and record
  both builds).
- BISCUIT's exact TRACE arguments (`trace_r = 4`, `trace_delta = 2.5`) map to
  CLI flags in a way the encrypted orchestrator hides. Port **all** flags;
  identify the likely mapping (e.g. `--epRad`, `--nhDist`, `--lsTh`) from the
  source and the TMI paper, and list it as a hypothesis in the port note.

## Scope

| Row | Deliverable |
|---|---|
| B2.6 | `pybiscuit.mesh.geodesic`: Dijkstra on the edge graph (SciPy) **and** a port of MeshLib `Geodesic.cpp` / `GeodesicPath.cpp` (the classes CurveExtraction uses). Port `GeodesicA` and the Peyré `gw` code only as far as CurveExtraction's call graph reaches. |
| B8.1 | `docs/ports/trace.md`: function map of `SulcalPoint/SulcalCurve/GyralPoint/GyralCurve/Extraction.cpp`, the flag table, and precision notes |
| B8.2 | Sulcal and gyral point selection, including line simplification |
| B8.3 | Curve tracing: graph, Dijkstra extension, pruning, junction handling |
| B8.4 | Writers for `.scurve`, `.gcurve`, `.bary`, `--cart`, and the VTK output; `pybiscuit trace` subcommand with the original flag names (`-i -o --sulcus --gyrus --bary --cart --noVtk --nojunction --smoothIter --smoothTensorIter …`) |

## Acceptance (E-class vs oracle; provisional until image goldens)

- Fixtures: `folded_sphere` variants (several `l, m, a`), plus one template
  surface `reg/mb101/lh.white.vtk`. Run the template case live (no large golden).
- Geodesic distances vs the oracle harness: rel ≤ 1e-5. On a unit sphere
  vs the analytic great-circle distance: ≤ 0.5 % mean (diagnostic).
- Curves: point-set Jaccard ≥ 0.99; `.bary` polyline Hausdorff ≤ 0.5 mm
  (template) or ≤ 1e-3 × radius (fixtures); explain any tie-driven
  differences, using brief 03's helpers.
- Runtime on `lh.white` (163,842 vertices): report it; target ≤ 3× the oracle.

## Stop conditions

Standard (AGENTS.md), plus: if TRACE needs MeshLib pieces outside brief 02's
ports that are large (>500 lines), stop and report a scope estimate first.
