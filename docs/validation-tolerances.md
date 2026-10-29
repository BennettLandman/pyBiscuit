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
| Parcellation network output (pre graph cut) | max abs diff relative to peak \|score\|; argmax agreement | ≤ 1e-4 relative; ≥ 99.9 % vertices |
| Final labels (post graph cut + cleanup) | vertex agreement; per-ROI Dice | ≥ 99.5 %; every ROI Dice ≥ 0.98 |
| Hull / klaplace correspondence | vertex distance | p99 ≤ 0.1 mm |
| Stats tables | relative diff per cell | ≤ 1e-4 |

Deviations caused by tie-breaking or iteration order must be explained in the
component's port note; they can pass with owner sign-off.

- **Chaotic stages (adopted 2026-10-22, Q14 / Q15):**
  - **LGI (Q14):** When the reference wavefront cutoff is sensitive to 1-ulp coordinate/geodesic differences at cortex scale, LGI is evaluated on the distributional criterion: vertex Pearson $r \ge 0.9999$, p99 relative diff $\le 10^{-3}$, and per-ROI mean relative diff $\le 10^{-3}$.
  - **Oracle-spread rule (Q15, HSD & LGI):** Where a 1-ulp input perturbation moves the reference or port trajectory beyond the static E-class bound, a call or chain passes if it sits within $\max(\text{policy bound},\; 1.5 \times \text{measured 1-ulp spread})$, with the perturbation ensemble recorded alongside the metrics. Stable calls remain held to the static E-class bounds above.

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

## Cross-platform policy (PROPOSED — owner approval pending)

Goldens and oracle dumps are produced on one platform and checked on several.
NumPy SIMD reductions, summation order, FMA contraction and libm differ across
platforms, so bit-identity is not an acceptance criterion.

- **Meaning of E.** E-class means "within tolerance, with ties explained", on
  any platform. It never means bit-identical.
- **Bit-exactness is diagnostic only.** Bit-for-bit oracle comparison lives
  behind `pytest -m bitexact`, run on the oracle platform for regression
  hunting. It is skipped by default and in CI, and never gates acceptance.
- **Reference platform.** The platform for the acceptance goal is the x86-64
  Linux image. Oracle builds made on macOS arm64 use `-ffp-contract=off` to
  approximate x86 arithmetic but remain provisional until reproduced in the
  image.
- **Tie handling.** A mismatch of an argmin / closest-face style decision is
  accepted only when the recomputed margin between the golden choice and the
  chosen one is ≤ eps. The test recomputes the margin and reports the tie
  count. An untied mismatch fails.
- **Candidate sets.** Candidate lists (e.g. AABB search) are compared as sets.
  A difference is accepted only for boxes whose bounds are within eps of the
  query; any other difference fails.
- **Default float tolerances.** float32 products: rel 1e-5. float64 products:
  rel 1e-10. The absolute tolerance is scaled by the golden's magnitude
  (`abs_ = rel * max|golden|`, with a floor documented per test). Per-product
  thresholds in the E, R and D tables above take precedence where stricter or
  looser by explicit entry. No tolerance may be loosened beyond this document
  to make a platform pass; a check that cannot pass is reported with its
  measured difference and decided by the owner.
- **Determinism requirements for code.** Where float32 rounding drives a
  discrete decision (AABB pruning, nearest face, tie-prone argmin), accumulate
  in a defined fixed order (explicit loops in Numba, or fixed-order sums), not
  `np.sum` over vectors NumPy may SIMD-reduce. Each such site is documented in
  the component's port note.

## Evidence

Each validated row records: command, input hashes, image hash (for reference
runs), pyBiscuit commit, metrics JSON path, and pass/fail per metric.
