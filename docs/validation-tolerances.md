# Validation and tolerances (PROPOSED — owner approval pending)

Numbers below are starting proposals for review. Rows in plan.md cite the
class and metric they are accepted against. Changing a threshold is an owner
decision recorded here with date and rationale.

## Acceptance classes

**E — exact port, identical inputs.** Feed the same files to the reference
binary (in the image) and to pyBiscuit.

| Product | Metric | Proposed threshold |
|---|---|---|
| Per-vertex scalars (curvature, SI, LGI, SCI, depth) | max abs diff / robust range; Pearson r | ≤ 1e-4 relative (float64 vs float32 noise), r ≥ 0.9999 |
| Deformed sphere (HSD) | per-vertex angular distance | median ≤ 0.01°, p99 ≤ 0.1°; final energy within 0.1 % |
| HSD SH coefficients | max abs diff | ≤ 1e-4 |
| Curves (TRACE) | vertex-set Jaccard on `.scurve`/`.gcurve`; Hausdorff on `.bary` polylines | Jaccard ≥ 0.99, Hausdorff ≤ 0.5 mm |
| Parcellation network output (pre graph cut) | per-class probability max abs diff; argmax agreement | ≤ 1e-4; ≥ 99.9 % vertices |
| Final labels (post graph cut + cleanup) | vertex agreement; per-ROI Dice | ≥ 99.5 %; every ROI Dice ≥ 0.98 |
| Hull / klaplace correspondence | vertex distance | p99 ≤ 0.1 mm |
| Stats tables | relative diff per cell | ≤ 1e-4 |

Deviations caused by tie-breaking or iteration order must be explained in the
component's port note; they can pass with owner sign-off.

**R — replacement, downstream agreement.** The FreeSurfer stage replacements
(pial, inflation, sphere, sulc) cannot be exact. Judge them by what BISCUIT
reports downstream, on the same MaCRUISE inputs:

| Product | Metric | Proposed threshold |
|---|---|---|
| White / pial geometry | symmetric surface distance to reference | mean ≤ 0.2 mm, p95 ≤ 0.5 mm |
| `sulc`, `curv`, `iH` (after HSD to template) | vertex-wise r in template space | ≥ 0.95 |
| Thickness | vertex-wise r; ROI mean diff | r ≥ 0.9; |ROI diff| ≤ 0.1 mm |
| Parcellation | per-ROI Dice vs reference | median ≥ 0.90, min ≥ 0.75 |
| ROI-wise measures (SD, LGI, SCI, curvature) | ICC(3,1) across cohort | ≥ 0.90 per ROI-measure; report all |
| Curve-wise measures | ICC across cohort | ≥ 0.85 |

Benchmark R against test–retest reproducibility of the original pipeline
where scan–rescan data exist: an R-class difference no larger than the
original's own retest variability is acceptable.

**D — diagnostic only.** Anatomical plausibility checks (self-intersection
count, genus, area distortion of spheres, label connectivity). Reported, not
gating, unless an owner promotes them.

## Evidence

Each validated row records: command, input hashes, image hash (for reference
runs), pyBiscuit commit, metrics JSON path, and pass/fail per metric.
