# LGI chain vs the image's intermediates — 2026-10-30

**Branch** `coord-main`. **Reference** `tmp-lgi-{lh,rh}.gray/` from `biscuit_FC_v2.2.sif` on the three open-examples
subjects (FreeSurfer route), sent by Ilwoo Lyu on 2026-10-30: `<h>.gray.hull.vtk` (OuterHull), `<h>.gray.hull.corr.vtk`
(klaplace correspondence), `<h>.gray.scurve.bary`, `<h>.white.gcurve.bary`. The final products sent with them
(`*_output_biscuit.zip`) are byte-identical to `open-examples/` (64 files per subject, PDFs not compared), so the image's
LGI maps are the ones of the earlier runs. Data stays in `~/dev/biscuit/image-runs/` (not in git).

```bash
python scripts/validation/lgi_intermediates_vs_image.py image-runs/x/tmp-lgi/<s> open-examples/<s> e2e-heap/<s> --json <s>.lgi.json
python scripts/validation/lgi_intermediates_vs_image.py --summary *.lgi.json
```

Metrics JSON: [lgi-intermediates-2026-10-30/](lgi-intermediates-2026-10-30/). `RUN_DIR` is a full `pybiscuit run` (FS route,
marching-cubes hull, libstdc++ heap order); the script also feeds the image's own intermediates to the port to isolate each
stage.

## Result

| Subject | Hemi | hull area ours/image | hull p50 / p99 mm | corr p50 / p99 / max mm | LGI chain r / rel | Gyrification on image corr r / rel / max abs | klaplace on image hull p50 / p99 / max mm |
|---|---|---|---|---|---|---|---|
| 0050952 | lh | 0.9983 | 0.003 / 0.64 | 0.087 / 1.02 / 36.9 | 0.9992 / +0.61 % | 0.999998 / -0.001 % / 0.32 | 0.001 / 0.00 / 5.0 |
| 0050952 | rh | 1.0003 | 0.014 / 0.64 | 0.099 / 5.87 / 41.0 | 0.9957 / +0.30 % | 0.999991 / +0.004 % / 0.34 | 0.000 / 0.00 / 4.7 |
| 0050961 | lh | 0.9974 | 0.003 / 0.58 | 0.052 / 0.58 / 26.1 | 0.9995 / -0.71 % | 0.999993 / +0.007 % / 0.27 | 0.000 / 0.00 / 2.6 |
| 0050961 | rh | 0.9975 | 0.026 / 0.59 | 0.113 / 2.92 / 30.7 | 0.9985 / -0.04 % | 0.999995 / -0.009 % / 0.47 | 0.000 / 0.00 / 9.0 |
| 0050971 | lh | 0.9983 | 0.003 / 0.64 | 0.065 / 0.90 / 34.3 | 0.9991 / -0.84 % | 0.999981 / -0.027 % / 0.49 | 0.001 / 0.00 / 3.4 |
| 0050971 | rh | 0.9976 | 0.007 / 0.64 | 0.083 / 1.19 / 24.1 | 0.9975 / +0.08 % | 0.999990 / -0.000 % / 0.22 | 0.000 / 0.00 / 7.5 |

Hull distances are image vertex to nearest port vertex (the reverse direction gives the same p99). The `.bary` curve files
are byte-identical to the image's in all six hemispheres (and to `curve/` of the earlier image run).

## Findings

**L1 — Gyrification matches the image once the heap order is the image's.** On the image's correspondence mesh, the port with
the libstdc++ `std::heap` tie order gives r 0.99998–0.999998 and mean differences ≤ 0.03 % in all six hemispheres; on 0050952
lh 93.9 % of vertices equal the image's `%f` output exactly, against 13.4 % with the libc++ order (the macOS oracle build)
and 26.8 % with Apple's `acosf` instead of the pinned fdlibm one. The image therefore breaks fast-marching ties like
libstdc++ and its `acosf` behaves like fdlibm (glibc), which settles the two provisional choices in
[lgi.md](../ports/lgi.md). The LGI path (`gyrification`, `lgi_pipeline`, `pybiscuit lgi`) now defaults to `libstdcxx`, as
SCI and the geodesic solver already did. The remaining p99 |diff| 0.02–0.08 is the 1-ulp chaos at the kernel-area cutoff
documented in Q14; it meets Q14's r ≥ 0.9999.

**L2 — klaplace is exact.** On the image's hull, the port's correspondence equals the image's with median 0.0005 mm and p99
0.004 mm; a few vertices per hemisphere land 2.6–9 mm away (streamlines ending on a neighbouring hull facet). This is the
first cortex-scale check of B9.3 (before: one synthetic golden).

**L3 — OuterHull is right to within one voxel.** Marching cubes (2026-10-29) brought the hull to the image's vertex count
(±0.2 %) and area (−0.3 to +0.03 %), median distance 0.003–0.03 mm. About 6 % of the image's vertices are one voxel
(0.58–0.66 mm, the 256-voxel grid) from the port's, in both directions (45 % outside, 55 % inside), without orientation
bias, in 2,609 small patches (median 2 vertices; 0050952 lh). Including the rim of the closing ball makes it worse (7.3 %).
The remaining candidates are MATLAB library internals with no source here (`strel('ball')`'s 8-line decomposition, `smooth3`
/ VOXELISE rounding near the threshold); see [outer-hull.md](../ports/outer-hull.md).

**L4 — the hull's voxel noise is what is left in end-to-end LGI.** Each one-voxel step moves the klaplace streamline ends of
nearby pial vertices (correspondence p99 0.6–5.9 mm, outliers to 41 mm), and Gyrification divides by the corresponding
outer area, so the chain gives r 0.9957–0.9995 and mean differences −0.84 to +0.61 %, p99 |diff| 0.33–1.33 LGI units. With
Gyrification and klaplace exact, closing that gap needs MATLAB's morphology internals (the compiled `OuterHull` does not
write its intermediate volume), not more porting of the C++ tools. Under Q14/Q15 this is the chaotic-row latitude: ROI LGI
agrees at r ≥ 0.9984, median relative difference ≤ 1.1 % ([e2e report](e2e-open-examples-2026-10-28.md)).

## Addendum 2026-11-02 — the image's `lgi` logs

Ilwoo Lyu sent the stdout of the six `lgi` runs that produced these intermediates (`<s>_<h>.lgi.log`, kept in
`~/dev/biscuit/image-runs/x/logs/`, not in git). They cover the LGI stage only (OuterHull, klaplace, Gyrification), not
the whole pipeline, so they are not the command traces B1.1 asks for. He also confirmed that the image's MATLAB was
compiled with R2012a and that `OuterHull.m` in LocalGyrificationIndex is the source, which is the file already ported.

```bash
python scripts/validation/lgi_logs_vs_image.py image-runs/x/logs image-runs/x/tmp-lgi/<s> open-examples/<s> e2e-heap/<s> \
    --json docs/reports/lgi-intermediates-2026-10-30/<s>.logs.json
```

| Subject | Hemi | flags = port defaults | log: adjusted ratio / regions corrected | port on image corr | port chain | klaplace grid spacing: log / image hull / port hull |
|---|---|---|---|---|---|---|
| 0050952 | lh | yes | 0.693616 / 18028 | 0.693616 / 18028 | 0.693439 / 18086 (-0.03 %) | 1.34005 / 1.34005 / 1.34005 |
| 0050952 | rh | yes | 0.688707 / 17934 | 0.688707 / 17934 | 0.689834 / 17962 (+0.16 %) | 1.31113 / 1.31113 / 1.31113 |
| 0050961 | lh | yes | 0.552818 / 16681 | 0.552818 / 16681 | 0.554635 / 16740 (+0.33 %) | 1.21056 / 1.21056 / 1.21056 |
| 0050961 | rh | yes | 0.544725 / 16358 | 0.544725 / 16358 | 0.545647 / 16382 (+0.17 %) | 1.2175 / 1.2175 / 1.21749 |
| 0050971 | lh | yes | 0.646141 / 17477 | 0.646141 / 17477 | 0.648302 / 17704 (+0.33 %) | 1.35163 / 1.35163 / 1.35163 |
| 0050971 | rh | yes | 0.658473 / 17688 | 0.658473 / 17688 | 0.658807 / 17862 (+0.05 %) | 1.3316 / 1.3316 / 1.3316 |

**L5 — the logs agree with L1–L4.** The logged flags (`klaplace -thread 1 -dims 128`, `Gyrification -m 316 --speed 0.2
--refHullArea 77100`) are the port's defaults; `# of seed points` equals the pial vertex count. On the image's correspondence
mesh, the port's kernel-area step reproduces the logged `Adjusted ratio` to all six printed digits and the
`outer hull regions corrected` count exactly. With the port's own hull, the ratio is −0.03 to +0.33 % off and the
corrected-region count 0.1–1.3 % higher: OuterHull's one-voxel noise again (L3). klaplace's grid spacing (longest side
of the hull's bounding box / 128) matches on both hulls (one 1e-5 difference, 0050961 rh). Nothing in the logs points at
a new cause; the OuterHull gap still needs MATLAB's `strel('ball')`/`imclose`/`smooth3` internals (outer-hull.md).
Metrics JSON: `lgi-intermediates-2026-10-30/<s>.logs.json`.

## Not received

The bundle's other jobs (MaCRUISE route runs with intermediates, FreeSurfer-route reruns with command traces) were not part
of this delivery; B1.1, B1.2, B5.4–B5.7 and B11.1 still need them.
