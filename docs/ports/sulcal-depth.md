# Port note: sulcal depth (plan row B9.5) — provisional, no oracle

Code: `src/pybiscuit/quant/sulcal_depth.py`, `sulcal_depth_cli.py` (`pybiscuit sulcal-depth in.vtk out.txt`).
Tests: `tests/test_sulcal_depth.py`. Source: `BISCUIT-master/SurfQuant/SulcalDepth/sulcal_depth.m`
(2022-02; not in the repo), spec [§9.2](../reference/orchestrator-spec.md). Reused, not duplicated:
`quant/_voxelise.voxelise` (Aitkenhead `VOXELISE`, identical to LGI's) and `quant/hull.{_smooth3_gaussian,
_close_ball,_dilate_ball}` (brief 07). Only the pieces specific to this script are new.

## Chain and what each MATLAB call is taken to mean

| `sulcal_depth.m` | Port | Semantics reproduced / assumed |
|---|---|---|
| `v - min(v)`, `*scale`(1), `vox = ceil(max(v))` | same | cell count per axis from the shifted extent; true cell size is `max(v)/vox <= 1`, voxel `k` taken at coordinate `k-1` (kept, spec §9.2) |
| `VOXELISE(vox, fv, 'xyz')` | `_voxelise.voxelise` | brief 07 port, unchanged |
| `padarray(volume, [20 20 20])` | `np.pad` zeros | exact |
| `imfill(volume, 26, 'holes')` | `binary_fill_holes`, 3x3x3 structure | background 26-connected to the array border; border voxels of the padded volume are background |
| `smooth3(.., 'gaussian', 9, 2)` | `hull._smooth3_gaussian` | separable 9-tap Gaussian, sigma 2, zero-padded edge ("replicate" is irrelevant: 20 voxels of zeros) |
| `imclose(.., strel('ball',15,15))`, `imdilate(.., strel('ball',1,1))` | `hull._close_ball/_dilate_ball` | **nonflat ellipsoid whose neighbourhood excludes the rim (`x^2+y^2 < r^2`; image evidence, brief 22, see below), per z-slice** (MATLAB's N=8 line decomposition approximates it; slice-wise assumption). Same unverified behaviour as OuterHull |
| `volume > 25` | same | |
| `bwperim(X, 18)` | `bwperim`: `X & ~binary_erosion(X, 18-structure, border_value=1)` | perimeter = set voxel with an unset 18-neighbour; array border is *not* a perimeter (toolbox erodes with ones padding — assumed). Matters only at the array faces, ≥ 20 voxels from the hull |
| `field`, six thickenings | `_field` | each pass: `field |= bwperim(~field,18) | bwperim(volume0,18)` — the tissue perimeter term is constant, as in the script |
| `bwdistgeodesic(field, find(seed), 'quasi-euclidean')` | `bwdistgeodesic`: Dijkstra (numba, binary heap) on the 26-neighbourhood restricted to the mask, steps 1, √2, √3 | shortest quasi-Euclidean path = the toolbox's converged chamfer propagation. Voxels outside the mask / unreachable: NaN. **Seeds outside the mask** (hull-shell voxels inside the tissue): dropped by default; `seeds_outside_mask='keep'` uses them as sources. MATLAB's choice is unverified |
| `D2 = min(Inf, D)` | `NaN -> Inf` | `min` ignores NaN |
| `interp3(X,Y,Z,D2, v(:,2), v(:,1), v(:,3))` | `interp3_linear(D2, v + pad)` | with `X` varying along dim 2 and `Y` along dim 1 the call is plain trilinear interpolation in (dim1, dim2, dim3) at `v`; coordinate of index `i` is `i - 1 - pad` (1-based), i.e. `i0 - pad` 0-based. IEEE weighted sum: positive weight × Inf = Inf (→ 0), **zero weight × Inf = NaN** (kept as NaN, not zeroed, as the script's `sd == inf` test would leave it). Whether R2012a's `interp3` linear path forms products this way is unverified |
| `sd/scale`, `sd==Inf → 0`, `%f\n` | same | six decimals |

CLI also prints min/max like the script (`min` ignores NaN).

## Numeric precision
All float64. MATLAB `smooth3`/`imclose` operate on doubles here (`double(volume)`), as does the port; `bwdistgeodesic`
returns double. Dijkstra sums step lengths in a different order than the toolbox's sweep; differences are ~1e-15 relative.

## Behaviour observed (analytic fixtures, `tests/test_sulcal_depth.py`)
* Sphere R = 40 mm: depth 1.0–3.4, mean 2.4 mm, σ < 0.6. This is the **baseline offset** of the smoothed hull
  (σ = 2 Gaussian, threshold 25/255 puts the hull ≈ 2.5 voxels outside the solid, +1 dilation); the depth of a convex
  surface is not 0 but this constant, independent of radius (30 vs 40 mm differ < 0.5 mm). Spec expectation "≈ 0" holds
  within the voxel tolerance (< 4 mm); a downstream consumer should not subtract anything — the script does not.
* Sphere with a meridian groove (Gaussian profile, w = 2.5 mm, ico-5, R = 40): groove-bottom depth − baseline = 2.6, 6.5, 10.5 mm
  for groove depths 4, 8, 12 mm (tolerance 2.5 mm asserted for the deeper two; narrow groove vs 1.5 mm mesh edge under-samples
  the bottom). Depth falls monotonically away from the groove and returns to baseline beyond 12 mm.
* Runtime: ico-5 sphere (R = 40) 5.8 s including numba compile of the cached kernels.

## Open items (need an image run, spec §15 item 6)
`strel('ball')` decomposition; `bwperim` border; `bwdistgeodesic` seed handling and exact accumulation; `interp3` with
Inf neighbours; hull area. Acceptance E (p99 ≤ 1e-4 relative) is therefore **not** claimed.

## Image evidence (brief 22, 2026-10-22)

First comparison against real output (`quant/<h>.gray.SD.txt` of three FreeSurfer-route subjects). With the closed disc
as `strel('ball')` neighbourhood the port was **0.71 mm too deep on average** (r = 0.9989): the hull dilation
`strel('ball',1,1)` grew the hull by a voxel. Excluding the rim makes `ball(1,1)` the single centre pixel (the dilation only
adds its height) and brings the bias to -0.04 mm (r = 0.9997, 54 % of the vertices equal to print precision, the rest
differ by at most one chamfer step, 1.41 mm). Threshold sweeps (22-26) and other neighbourhood radii do not do better. The
remaining differences are voxel-scale hull details (MATLAB's N = 8 line decomposition of the ball and the heights it
produces are not reproduced), so the stage stays **below the E-class bound** (r >= 0.9999, max abs / range <= 1e-4)
and is reported as such. The fix was inferred from the image, not from MATLAB source; it applies to `OuterHull` too.
