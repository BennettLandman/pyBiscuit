# Port note: OuterHull (plan row B9.2) — provisional, no oracle

Code: `src/pybiscuit/quant/hull.py`, `quant/_voxelise.py`, `quant/hull_cli.py` (`pybiscuit outer-hull in.vtk out.vtk`).
Tests: `tests/test_hull.py`. Source: LocalGyrificationIndex `b9fa760`, `matlab/OuterHull.m` (105 lines),
`matlab/VOXELISE/{VOXELISE,COMPUTE_mesh_normals,CONVERT_meshformat}.m`, `matlab/ICP_finite/{ICP_finite,fminlbfgs}.m`,
`matlab/{read,write}_vtk.m`. The image's `OuterHull` is the MATLAB-compiled `OuterHull.m` run on MCR 7.17
(R2012a), so `verLessThan('matlab','8.5')` is true and the `ICP_finite` branch (not `pcregrigid`) is the one ported.

## Why provisional

There is no oracle. MATLAB is not available; GNU Octave could not be installed (the `conda.anaconda.org`
channel is unreachable from the work machine, `brew` bottles need the same kind of network). Several steps
call MATLAB library code whose source is not here. They were re-derived from the documented behaviour:

| Step in `OuterHull.m` | Port | Deviation / risk |
|---|---|---|
| `VOXELISE(vox..., 'xyz')` | `_voxelise.voxelise`: ray casting along x, y, z, majority vote, vertex-on-ray branch, 1e-12 rounding, even/odd rule, neighbour correction | faithful line by line; only the first of x/y/z gets the "extend grid" treatment (the `elseif` chain), kept |
| `padarray(...,[20 20 20])`, `imfill(volume, 26, 'holes')` | `np.pad`, `binary_fill_holes` with a 26-neighbour structure | equal by definition |
| `smooth3(volume,'gaussian',9,2)` | separable 3-D Gaussian (size 9, sigma 2) | float summation order only |
| `imclose(volume, strel('ball',15,15))`, `imdilate(..., strel('ball',1,1))` | **exact** nonflat ellipsoid SE (height `H*sqrt(1-r^2/R^2)`) applied **per z-slice** | MATLAB's default `N = 8` approximates the ball by 8 line-shaped SEs; the decomposition is not reproducible without the toolbox source. Also assumes MATLAB pads the 2-D neighbourhood with singleton dimensions (slice-wise). Both unverified |
| `isosurface(volume, 1)` | classic marching cubes (Lorensen & Cline, Bourke's public-domain table, `quant/_mc_tables.py`), one vertex per cut voxel edge, faces oriented outward | vertex and face order may differ from MATLAB's; the triangulation is the same per cube (no ambiguity resolution in either). Until 2026-10-29 this was marching tetrahedra, whose extra diagonal cuts made the hull of a cortical surface ~18 % larger in area and LGI 5-7 % low (see below) |
| axis swap, translate/scale to the mesh box, `[0.5 0 -0.5]`, `*1.02` | same arithmetic | the `*1.02` scales about the origin (as in the script), ICP then removes the resulting offset |
| largest connected component via `dmperm` | largest component (`scipy.sparse.csgraph`) | `dmperm` returns the *first* block, not necessarily the largest; identical when the iso-surface is one piece (the usual case) |
| `ICP_finite` (rigid, `fminlbfgs`, `knnsearch`) | `icp_finite`, `fminlbfgs`: BFGS (6 variables), forward differences with `gstep = initialStepLength/1e6`, Wolfe bracketing + sectioning, cubic interpolation, same stopping rules (`TolX = extent/1000`, `TolFun` 1e-6, `MaxFunEvals` 599); `cKDTree` for nearest neighbours | line-by-line port; nearest-neighbour ties and summation order can differ in the last bits |
| `write_vtk` (`%f`, 6 decimals) | `write_vtk` float `%.9g` | more digits |

`grid=256` (voxels along the longest side) and the other constants are the script's values; `grid` can be
lowered for experiments (the CLI exposes `--grid`).

## Evidence available (D class)

`tests/test_hull.py`: voxelisation of a sphere / ellipsoid vs the analytic volume; iso-surface of a voxel
ball is closed, outward, radius within 0.6 voxel; the closing fills a narrow slot per slice; `fminlbfgs`
solves a quadratic; ICP recovers a 2 degree / 0.8 mm rigid offset to < 0.5 mm; on an ellipsoid
(60 x 45 x 35 mm) and a folded sphere the hull is a closed genus-0 surface that encloses the input with a
margin of 1 to 2.5 mm (the `1.02` factor plus the closing). At grid 256 an ellipsoid takes ~6 s.

## To close the row

Run `OuterHull` from the image (Q1) on the fixtures and on a cortical surface and compare hull vertices
by symmetric surface distance (p99 <= 0.1 mm, validation-tolerances.md), or obtain Octave/MATLAB. Open
questions the oracle would settle: `strel('ball')` decomposition, slice-wise application, `dmperm` block order.

## Image evidence (brief 22)

`hull._ball` (shared with sulcal depth) now excludes the rim of the structuring-element disc (`x^2 + y^2 < r^2`, centre kept):
`strel('ball',1,1)` is then the centre pixel. Inferred from the image's `quant/<h>.gray.SD.txt` (bias +0.71 -> -0.04 mm), see
[sulcal-depth.md](sulcal-depth.md); the same `strel` calls are in `OuterHull.m`, so the LGI hull follows. Not verified against MATLAB.

## Image evidence: marching cubes (2026-10-29)

The end-to-end run ([e2e report](../reports/e2e-open-examples-2026-10-28.md), F4) found LGI 4.7-7.1 % below the image in all
six hemispheres. On 0050952 lh the marching-tetrahedra hull had area 61,474 mm^2, the marching-cubes hull of the same volume
52,085 mm^2 (-15 %); with it LGI vs the image went from r 0.9856 / mean diff -0.225 to r 0.9992 / +0.025 (+0.6 %).
`isosurface` reproduces scikit-image's `marching_cubes(method="lorensen")` vertex set and area exactly on an ellipsoid
(checked once, scikit-image is not a dependency); `tests/test_hull.py::test_isosurface_is_marching_cubes` pins the
one-voxel case. Sulcal depth does not use the iso-surface and is unchanged.

Direct check against the image's own hull (`tests/data/golden-image/outer-hull/bumpy_ico5`): the port now gives 277,674
vertices vs 277,680 and area 38,993 vs 39,120 mm^2 (-0.3 %), p99 nearest-vertex distance 0.0001 mm in both directions. The
B9.2 acceptance test had measured only image -> port distance; the tetrahedra vertex set contains every marching-cubes vertex,
so it passed. `test_outer_hull_vs_image_golden` now also checks port -> image distance, vertex count and area.

## Cortex-scale image evidence (2026-10-30)

The image's `tmp-lgi-*/<h>.gray.hull.vtk` for six open-examples hemispheres ([report](../reports/lgi-intermediates-2026-10-30.md)):
vertex count within 0.2 %, area -0.3 to +0.03 %, median distance 0.003-0.03 mm, but ~6 % of vertices one voxel (0.58-0.66 mm)
off in both directions, in small scattered patches. Including the closing ball's rim raises that to 7.3 %. Not resolved: the
remaining suspects are MATLAB internals (`strel('ball')` decomposition, `smooth3`/VOXELISE rounding at the threshold). This is
what remains in end-to-end LGI (klaplace and Gyrification match the image on its own intermediates).
