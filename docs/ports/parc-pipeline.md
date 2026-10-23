# Port Note: Parcellation Pipeline (`B7.1`, `B7.3`)

**Author:** pyBiscuit porting team  
**Date:** 2026-10-22  
**Sources:**
- `BISCUIT-master/SurfPostproc/common/matlab/parc/parc.m`
- `BISCUIT-master/SurfPostproc/common/matlab/parc/extract_features.m`
- `docs/reference/orchestrator-spec.md` (§6.1–§6.6)

## 1. Mathematical and Algorithmic Specifications

### 1.1 Protocol Labels (`spec §6.1`)
- Protocol `label.csv` contains a header and rows `Name, ID [, Lobe]`.
- For `hemi == 'lh'`, rows whose name begins with `Left` (case-insensitive) are selected.
- For `hemi == 'rh'`, rows whose name begins with `Right` (case-insensitive) are selected.
- A synthetic class `0` (`Unknown`) is prepended, yielding `label = [0, ids of that hemisphere]`.
- Class count is 50 for BrainColor and 32 for DKT31, matching U-Net channel outputs.

### 1.2 Feature Extraction (`spec §6.2`, `extract_features.m`)
- Three scalar features are extracted on the white matter surface:
  1. `iH`: from FreeSurfer inflated mean curvature (`<hemi>.inflated.H`).
  2. `sulc`: from FreeSurfer sulcal depth (`<hemi>.sulc`).
  3. `H`: negated mean curvature on the white surface (`-mean_curvature(white, surfsmoothing=5, tensorsmoothing=5)`).
- Normalisation pipeline for each feature `x`:
  1. `x = (x - mean(x)) / std(x, ddof=1)`
  2. Soft clamp: `x < -3 -> -3 - (1 - exp(3 + x))`, `x > 3 -> 3 + (1 - exp(3 - x))`
  3. `x = x / std(x, ddof=1)`
  4. Soft clamp again
  5. Text format rounding to 6 decimals (`%f`).

### 1.3 Spherical Alignment & Resampling (`spec §6.3`)
- Native sphere is rigidly aligned using registration coefficients truncated to degree 0 (or 10 for CIVET) via `SphericalRemesh --deg <degree> --deform <out>`.
- Alignment is refined against template `parc/<p>/<h>.curv.mean.txt` using `HSD -d 0 --noguess --icosahedron 5` with `-o <in>`.
- Features are resampled onto `icosphere_5.vtk` (10,242 vertices) via barycentric interpolation.
- Network channel ordering is `[H, iH, sulc]`.

### 1.4 Refinement & Multi-Model Fusion (`spec §6.6`)
- For each fold/checkpoint:
  1. Raw per-class scores are remeshed back onto the aligned native sphere.
  2. Initial labelling via `gcut(v, f, scores)` and `roi_correction(pred, v, f, scores)`.
  3. Per-vertex softmax probability `pred_prob(fold, j) = softmax(scores)[pred(j), j]`.
- If single model (`braincolor`): final labelling is `label[pred]`.
- If multi-model (`dkt31` 5 folds):
  1. Accumulate votes: `prob[c, :] = sum_m (pred_prob[m, :] * (pred[m, :] == c))`.
  2. Normalise over classes: `prob = prob / sum(prob, axis=0)`.
  3. Log transform: `prob = log(prob + eps)`.
  4. Final graph cut and cleanup: `gcut(v, f, prob)` and `roi_correction`.
  5. Map to protocol IDs: `label[pred]`.
