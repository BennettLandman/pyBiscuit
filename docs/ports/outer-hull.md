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
| `isosurface(volume, 1)` | marching tetrahedra (Kuhn 6-tet decomposition), vertices shared, faces oriented outward | different triangulation and vertex order from MATLAB's marching cubes; the surface differs only through the piecewise-linear interpolation. Hull has ~2x the triangles |
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
